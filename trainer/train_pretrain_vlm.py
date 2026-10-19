import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import itertools
import glob
import time
import warnings
import datasets
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import LocalMultiSourceVLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn

warnings.filterwarnings('ignore')


def _global_sum(value, device):
    result = torch.tensor(value, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return int(result.item())


def _save_stage_checkpoint(step, tokens_seen):
    """Keep immutable milestone weights without allowing them to fill the disk."""
    moe_suffix = '_moe' if vlm_config.use_moe else ''
    latest = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
    milestone = (
        f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}'
        f'_step{step:07d}_tok{tokens_seen:012d}.pth'
    )
    tmp = milestone + '.tmp'
    try:
        os.link(latest, tmp)
    except OSError:
        import shutil
        shutil.copy2(latest, tmp)
    os.replace(tmp, milestone)
    pattern = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}_step*_tok*.pth'
    # Step is zero-padded in the filename, so lexical order is chronological
    # even when several files share the same coarse filesystem mtime.
    milestones = sorted(glob.glob(pattern))
    while len(milestones) > args.keep_stage_checkpoints:
        old = milestones.pop(0)
        os.remove(old)
        Logger(f'Removed old milestone checkpoint: {old}')
    Logger(f'Saved milestone checkpoint: {milestone}')


def train_epoch(epoch, loader, iters, start_step=0, tokens_seen=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    loss_sum = 0.0
    aux_loss_sum = 0.0
    log_batches = 0
    local_tokens_since_sync = 0
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        if step > iters:
            break
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        local_tokens_since_sync += int(labels.ne(-100).sum().item())
        pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()} if isinstance(pixel_values, dict) else pixel_values.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=pixel_values)
            loss = res.loss + res.aux_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at micro_step={step}: {loss.item()}')
            loss_sum += loss.detach().item()
            aux_loss_sum += res.aux_loss.detach().item() if res.aux_loss is not None else 0.0
            log_batches += 1
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            tokens_seen += _global_sum(local_tokens_since_sync, args.device)
            local_tokens_since_sync = 0
            spend_time = time.time() - start_time
            current_loss = loss_sum / max(log_batches, 1)
            current_aux_loss = aux_loss_sum / max(log_batches, 1)
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            optimizer_updates = step // args.accumulation_steps
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}] micro_step:[{step}/{iters}], optimizer_updates:{optimizer_updates}, tokens_seen:{tokens_seen}, avg_loss:{current_loss:.4f}, avg_logits_loss:{current_logits_loss:.4f}, avg_aux_loss:{current_aux_loss:.4f}, lr:{current_lr:.8f}, eta:{eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min, "tokens_seen": tokens_seen})
            loss_sum = 0.0
            aux_loss_sum = 0.0
            log_batches = 0

        should_save = step % args.save_interval == 0 or step == iters
        # Every DDP rank must participate in the token-count reduction, even
        # though only rank 0 writes files.
        if should_save and local_tokens_since_sync:
            tokens_seen += _global_sum(local_tokens_since_sync, args.device)
            local_tokens_since_sync = 0
        if should_save and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vlm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}  # 半精度保存并移到CPU
            ckp_tmp = ckp + '.tmp'
            torch.save(clean_state_dict, ckp_tmp)
            os.replace(ckp_tmp, ckp)
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scaler=scaler, tokens_seen=tokens_seen)
            _save_stage_checkpoint(step, tokens_seen)
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, pixel_values, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return tokens_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10000, help="模型保存间隔")
    parser.add_argument("--keep_stage_checkpoints", type=int, default=3, help="最多保留多少个不可覆盖的阶段权重")
    parser.add_argument('--hidden_size', default=1024, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=450, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_config", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'pretrain_repair_packed.json'), help="本地多源数据配置 JSON；默认按有效 token 比例混合的文本预训练配置")
    parser.add_argument("--steps_per_epoch", type=int, default=50000, help="每轮micro-batch数；实际optimizer更新数为该值/accumulation_steps")
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_weight_path', default=None, type=str, help="直接指定初始化权重路径（优先于from_weight）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，1=冻结+解冻首尾层，2=完全冻结仅训练proj）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer 路径（worker 懒加载用）")
    parser.add_argument("--vision_model_path", type=str, default="../model/siglip2-base-p32-256-ve", help="视觉模型路径（worker 懒加载用）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 启动时打印设备诊断信息，便于排查 CPU/GPU 误用
    Logger(f'torch version: {torch.__version__}, cuda available: {torch.cuda.is_available()}, '
           f'device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}, '
           f'args.device: {args.device}')
    if 'cuda' not in args.device:
        Logger('⚠️ WARNING: 训练设备不是 CUDA！当前会在 CPU 上运行，速度会慢几十倍。'
               '若驱动正常但 cuda available=False，请检查 nvidia-uvm 模块是否加载、'
               '或重装 torch 为 cu130 build: pip install torch --index-url https://download.pytorch.org/whl/cu130')
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, from_weight_path=args.from_weight_path, device=args.device, freeze_llm=args.freeze_llm)
    train_ds = LocalMultiSourceVLMDataset(args.data_config, tokenizer, preprocess=preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_seq_len, stage='pretrain', tokenizer_path=args.tokenizer_path, vision_model_path=args.vision_model_path)
    Logger(f'Pretrain configured source mix (unit={train_ds.mixing_unit}, '
           f'shuffle_sources={train_ds.shuffle_sources}):')
    for name, weight, ratio, filters in train_ds.source_summary():
        Logger(f'  {name}: weight={weight}, ratio={ratio:.1f}%, filters={filters or "none"}')
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step, tokens_seen = 0, 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        tokens_seen = int(ckp_data.get('tokens_seen', 0))
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        setup_seed(42 + epoch)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, collate_fn=vlm_collate_fn,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            tokens_seen = train_epoch(epoch, itertools.islice(loader, skip, None), args.steps_per_epoch, start_step, tokens_seen, wandb)
        else:
            tokens_seen = train_epoch(epoch, loader, args.steps_per_epoch, 0, tokens_seen, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
