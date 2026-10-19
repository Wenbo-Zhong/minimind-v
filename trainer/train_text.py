#!/usr/bin/env python3
"""Pure language pretraining/SFT with audited tokens and exact stream checkpoints.

Single GPU, main-process data loading only. No imports of VLM/SigLIP components.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer
from dataset.text_data import TextMixture
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload, path):
    temp = str(path) + '.tmp'
    torch.save(payload, temp)
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pretrain', 'sft'], required=True)
    parser.add_argument('--data_config')
    parser.add_argument('--model_config', default=str(ROOT / 'configs/lm_085b.json'))
    parser.add_argument('--tokenizer_path', default=str(ROOT / 'model'))
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resume', help='This trainer checkpoint, including data cursor and RNG')
    parser.add_argument('--init_weight', help='Explicit weights/checkpoint for a new stage')
    parser.add_argument('--allow_random_sft', action='store_true', help='Small pipeline tests only')
    parser.add_argument('--max_steps', type=int, default=1000, help='Total optimizer updates; fixed across resume')
    parser.add_argument('--stop_after', type=int, help='Stop/save early without changing the LR horizon')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--keep_checkpoints', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This stateful baseline supports a single process/GPU; DDP is not implemented')
    if args.resume and args.init_weight:
        parser.error('--resume and --init_weight are mutually exclusive')
    if min(args.max_steps, args.batch_size, args.accumulation_steps, args.save_interval, args.log_interval, args.threads, args.keep_checkpoints) < 1:
        parser.error('step, batch, interval and thread counts must be positive')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.max_steps:
        parser.error('--stop_after must be within the total LR horizon')
    if args.stage == 'sft' and not (args.resume or args.init_weight or args.allow_random_sft or args.dry_run):
        parser.error('SFT needs an explicit --init_weight or --resume')
    args.data_config = args.data_config or str(ROOT / 'configs' / ('pretrain_text_085b.json' if args.stage == 'pretrain' else 'sft_text_calculator.json'))
    args.learning_rate = args.learning_rate if args.learning_rate is not None else (3e-4 if args.stage == 'pretrain' else 2e-5)
    if args.learning_rate <= 0 or args.warmup_steps < 0 or args.grad_clip <= 0:
        parser.error('invalid learning rate, warmup or gradient clip')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    config = MiniMindConfig(**json.loads(Path(args.model_config).read_text()))
    if config.use_moe:
        parser.error('Initial text baseline supports dense models only')
    if config.vocab_size < len(tokenizer) or config.eos_token_id != tokenizer.convert_tokens_to_ids('<|im_end|>'):
        parser.error('Model vocabulary/EOS is incompatible with tokenizer')
    if args.max_seq_len > config.max_position_embeddings:
        parser.error('Sequence length exceeds model context')
    data = TextMixture(args.data_config, tokenizer, args.max_seq_len, args.stage)
    tokenizer_hashes = {str(p.relative_to(args.tokenizer_path)): file_hash(p)
                        for p in sorted(Path(args.tokenizer_path).glob('*'))
                        if p.is_file() and (p.suffix in ('.json', '.jinja', '.txt', '.model'))}
    metadata = {'version': 1, 'stage': args.stage, 'model_config': config.to_dict(),
                'tokenizer_sha256': tokenizer_hashes, 'data_config': data.config,
                'data_manifest': [stream.manifest for stream in data.streams],
                'exclusion_keys_sha256': hashlib.sha256(json.dumps(sorted(data.exclusions)).encode()).hexdigest(),
                'training': {k: getattr(args, k) for k in ('max_steps', 'batch_size', 'accumulation_steps',
                            'max_seq_len', 'learning_rate', 'warmup_steps', 'weight_decay', 'grad_clip', 'seed', 'dtype', 'device')},
                'code_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in
                                (Path(__file__), ROOT / 'dataset/text_data.py', ROOT / 'model/model_minimind.py', ROOT / 'eval_calculator.py')}}
    # Count parameters without allocating/initializing a 0.85B model on a dry run.
    with torch.device('meta'):
        prototype = MiniMindForCausalLM(config)
    count = sum(p.numel() for p in prototype.parameters())
    del prototype
    print(f'Pure text {args.stage}: params={count:,}, random_init={not (args.resume or args.init_weight)}, vision=disabled', flush=True)
    print(f'Data: main process, shuffled files/row groups + buffer={data.config.get("shuffle_buffer", 256)}, resumable cursor', flush=True)
    total_weight = sum(s['weight'] for s in data.sources)
    for source in data.sources:
        print(f'  {source["name"]}: target_loss_tokens={source["weight"] / total_weight:.1%}', flush=True)
    if args.dry_run:
        print(json.dumps({'metadata': metadata, 'parameters': count}, ensure_ascii=False, default=str), flush=True)
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Output is nonempty; resume explicitly or use a new directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    model = MiniMindForCausalLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    step = 0
    origin = {'kind': 'random', 'seed': args.seed}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['metadata'] != metadata:
            raise ValueError('Resume fingerprint/config mismatch; refusing to silently alter this run')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data.load_state_dict(checkpoint['data_state'])
        step = checkpoint['step']
        origin = checkpoint.get('origin', {'kind': 'unknown'})
        restore_rng(checkpoint['rng'])
    elif args.init_weight:
        weights = torch.load(args.init_weight, map_location='cpu', weights_only=False)
        if 'metadata' in weights and weights['metadata']['model_config'] != metadata['model_config']:
            raise ValueError('Initialization model configuration mismatch')
        if 'metadata' in weights and weights['metadata']['tokenizer_sha256'] != tokenizer_hashes:
            raise ValueError('Initialization tokenizer fingerprint mismatch')
        model.load_state_dict(weights.get('model', weights), strict=True)
        origin = {'kind': 'checkpoint', 'path': str(Path(args.init_weight).resolve()),
                  'size': Path(args.init_weight).stat().st_size,
                  'mtime_ns': Path(args.init_weight).stat().st_mtime_ns}
        del weights
    (output / 'run.json').write_text(json.dumps(metadata | {'origin': origin}, ensure_ascii=False, indent=2, default=str))
    snapshot = output / 'tokenizer'
    snapshot.mkdir(exist_ok=True)
    for name in tokenizer_hashes:
        if not (snapshot / name).exists():
            shutil.copy2(Path(args.tokenizer_path) / name, snapshot / name)
    model.train()
    start_time, initial_tokens = time.monotonic(), sum(data.loss_tokens)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    finish = args.stop_after or args.max_steps
    warmup = min(args.warmup_steps, args.max_steps // 10)
    while step < finish:
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        # Token-weight the microbatches, including unequal assistant lengths in SFT.
        batches = [data.batch(args.batch_size) for _ in range(args.accumulation_steps)]
        counts = [int(labels[:, 1:].ne(-100).sum()) for _, labels in batches]
        effective = sum(counts)
        next_step = step + 1
        if warmup and next_step <= warmup:
            lr = args.learning_rate * next_step / warmup
        else:
            progress = (next_step - warmup - 1) / max(1, args.max_steps - warmup - 1)
            lr = args.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        mean_loss = 0.0
        for (ids, labels), tokens in zip(batches, counts):
            context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                       if args.device.startswith('cuda') and args.dtype == 'bfloat16' else nullcontext())
            with context:
                result = model(ids.to(args.device), labels=labels.to(args.device))
                loss = result.loss * (tokens / effective)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss before optimizer update {next_step}')
            loss.backward()
            mean_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step = next_step
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        if step % args.log_interval == 0 or step == finish:
            log = {'step': step, 'loss': mean_loss, 'lr': lr, 'grad_norm': float(norm),
                   'loss_tokens': sum(data.loss_tokens), 'source_counters': data.summary(),
                   'update_seconds': time.monotonic() - began,
                   'tokens_per_second': (sum(data.loss_tokens) - initial_tokens) / (time.monotonic() - start_time),
                   'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith('cuda') else 0}
            print(json.dumps(log, ensure_ascii=False), flush=True)
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(log) + '\n')
        if step % args.save_interval == 0 or step == finish:
            # Only save after an optimizer update: no pending gradients/prefetch.
            atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': step, 'metadata': metadata, 'origin': origin, 'rng': rng_state(),
                         'data_state': data.state_dict()}, output / 'last.pt')
            milestone = output / f'step_{step:08d}.pt'
            if not milestone.exists():
                os.link(output / 'last.pt', milestone)
            for old in sorted(output.glob('step_*.pt'))[:-args.keep_checkpoints]:
                old.unlink()
            print(f'Saved resumable checkpoint: {output / "last.pt"}', flush=True)
    print(f'Finished at optimizer step {step}; this is not a full pretraining baseline.', flush=True)


if __name__ == '__main__':
    main()
