# 0.85B 纯文本训练方案（2026-10-20）

当前由用户在终端手动运行预训练，输出目录为 `out/text085b_pretrain_manual_v1/`，进度见该目录 `metrics.jsonl`。到达 100M token pilot 终点后停止，再手动执行验证；不会自动继续到 1B 或启动 SFT。旧后台任务已停止，launcher 和历史补丁已归入本地忽略的 `archive/legacy_*/`。

## 推荐入口与阶段

统一入口：`trainer/train_text.py --stage pretrain|sft`。这是独立纯语言路径，直接构造 `MiniMindForCausalLM`，不导入或加载视觉编码器。默认模型配置 `configs/lm_085b.json`：hidden=1280、32 层、16 个 attention heads、8 个 KV heads、绑定 embedding，当前词表下精确为 **847,304,960** 个参数。

新预训练默认为随机初始化。SFT 必须显式提供 `--init_weight`；`--resume` 只恢复同一个实验，不能悄悄更换 tokenizer、数据文件、训练参数或代码。历史 `train_pretrain_vlm.py` / `train_sft_vlm.py` 保留原有用途，不会自动转成这次的新模型。今后的纯文本实验使用上述新入口，不再沿用它们的旧权重默认值。

## 语料与比例

预训练默认配置 `pretrain_text_085b.json`，有效 next-token loss 目标：SkyPile 45%、FineWeb-Edu 35%、Infinity code 10%、Infinity math 10%。四类均等长 packing，每 100 个 pack 是完整比例周期。后两类仍是指令数据，而非独立原始代码/数学书；此次未假称增加了原始预训练语料。计算器指令留在 SFT，先保留网页语料的语言覆盖。

SFT 默认配置 `sft_text_calculator.json`：

| 来源 | 有效 assistant loss token 目标 |
| --- | ---: |
| Calc-ape 计算器数学（中英文各一份） | 20% |
| 已整理中文通用 | 25% |
| Tulu 短指令／对话（以英文为主，并非语言检测配额） | 20% |
| 已整理代码 | 15% |
| GSM8K 训练集无工具数学 | 15% |
| 基础算术 | 5% |

按来源累计有效监督 token / 权重的欠额选下一条样本，而非按行数混合，避免长数学解答挤掉通用对话。短窗口会偏离目标，最终以每个来源实际 `loss_tokens` 为准。20% 是起始实验比例，既给工具调用足够监督，也保留 80% 其他能力；它不是已证明最优的比例。

### 计算器数据转换

```bash
/home/zwb/miniconda3/envs/minimind-v/bin/python dataset/build_calculator_sft.py
```

只读取 `/home/zwb/ai/data/sft/open_math/Calc-ape210k/data/train-*.parquet`，不重复读取 `original-splits`；输出到新目录 `dataset/prepared/calculator_v1`，已有目录会拒绝覆盖。默认转换已执行：

- 扫描 195,179 条原题，保留训练 130,934 题／261,868 条双语对话，内部验证 1,240 题／2,480 条双语对话。
- 训练数据共有 10,563,650 个有效 assistant 监督 tokens；验证为 100,142。
- 排除 Calc-ape 原验证／测试题，以及本地 GSM8K test、CMATH dev/test 的规范化同题匹配；按题去重，再按原中文问题 hash 划出约 1% 内部验证，同题两个语言版本保持在同一侧。
- `<gadget>/<output>/<result>` 转成与评测完全一致的 `<calculator>` → user `Calculator result: ...` → assistant 最终答案多轮对话。
- 用实际评测计算器逐步复算并检查输出、最终结果；最多 4 次调用。丢弃解析失败、无调用、超过预算、复算不一致等样本，不静默教模型编造工具返回值。
- 计算器增加了范围受限的整数幂 `**`（指数 -8 到 8），支持原数据中的简单平方；模型仍只能提交算术表达式，不能执行任意代码。
- prompt、user、工具返回值均 mask；仅 assistant 的调用、最终答案及 ChatML 结束标记参与 SFT loss。超出序列长度整条拒收，不保留残缺回答。

详细统计在 `dataset/prepared/calculator_v1/report.json`，数据、tokenizer 与协议 hash 在 `manifest.json`。这是算术一致性检查，并不证明自动翻译或原题的语义解释全部正确。

## 随机性、覆盖与断点

历史读取器仅打乱来源顺序；JSONL 多 worker 仍各自扫描整文件，Parquet 分片逻辑随 row-group 数变化。新入口采用主进程状态流，打乱文件／row group，并做每来源 256 条文档 buffer shuffle；语料耗尽后重新打乱。不会每次短训都从原始第一行开始。

当前仅支持单进程／单 GPU，没有 DataLoader workers 或预取，检测到多进程启动会明确拒绝，防止未实现的 worker/rank 分片静默重复数据。72GB 单卡实测已能运行目标配置，先保证恢复准确，再根据瓶颈考虑并行读取。

checkpoint 保存：完整模型配置、tokenizer SHA256（同时复制 tokenizer 快照）、数据配置、文件大小和纳秒 mtime 清单、排除题目集合指纹、代码 SHA256、来源 loss token/样本/拒收/读取轮次计数、packing 剩余 tokens、来源调度、文件位置、shuffle buffer、随机数状态、FP32 模型和 AdamW 状态。只在完成 optimizer update 后原子保存，避免丢失半个梯度累积周期。默认保留最近 2 个阶段 checkpoint；`last.pt` 与对应阶段文件通过硬链接共享磁盘内容。

`loss_tokens` 严格按 `labels[:, 1:] != -100` 统计；packing 边界每块损失一个预测位置。SFT 梯度累积也按有效 token 加权，避免短回答和长回答按条数获得相同权重。

CPU 测试覆盖 JSONL/Parquet 游标、shuffle buffer、packing、来源比例、工具 loss mask、同题过滤、训练/验证分离，以及带 dropout 的连续训练 vs 中断续训：模型参数与最终 RNG 逐项一致。GPU 可复现性仍受 CUDA kernel 等因素影响，不能据 CPU 测试宣称 GPU 位级一致。

注意：来源循环计数包含 buffer 预读；实际训练量看有效 loss tokens。文件清单不等同于所有百 GB 原始文件的内容 hash。尚未完成全量网页质量过滤、全库去重或语义近重复污染检查；当前做的是留出集规范化同题过滤与确定性分割，不能声称没有任何近重复泄漏。

## 已执行的短跑验证

以下是历史管线验证记录，**不是训练完成的语言模型**。相应权重已清理，日志和验证结果仍留在本地忽略的 `out/` 下。

- `out/text085b_pipeline_smoke_v2`：从零预训练 20 次 optimizer updates，batch=4、seq=1024、accum=4，合计 **327,360** 有效预测 tokens。最后 loss=8.5211；运行期间四类来源都实际参与训练，末尾近似 45/35/10/10。实测约 **16,626 有效 tokens/s**，PyTorch 峰值 allocated 显存 **31.79 GiB**。该吞吐来自很短的单卡测试，未摊入模型初始化和最终 checkpoint 保存，不能当长期保证。
- 同目录 `validation_loss.json`：每来源 4 个 1024-token 留出 pack；中文 9.1819、英文 8.5994、代码 7.9859、数学 6.8375。样本很小，只验证评测通路与 loss 有限。
- `out/text085b_sft_pipeline_smoke_v1`：显式从上述短跑检查点启动 SFT，8 次 updates；6 个来源均进入训练，计算器来源 34 条对话、1,343 个有效监督 tokens，占本次总 7,222 tokens 的 **18.6%**，短窗口接近 20% 目标。工具来源未被读取器静默丢弃。最后 loss=8.8860，峰值 allocated 26.33 GiB；短跑 loss 波动不能用来判定能力提升。
- 同目录 `validation_loss.json` 检查了 6 类 SFT 留出 loss。`generation_smoke.json` 验证纯文本 checkpoint 可以进入应用题／计算器评测入口，未加载 SigLIP；这两题 16-token 输出不代表能力评测。
- 完整 CPU 回归测试 23 项通过。

## 当前 pilot 参数与后续阶段

先做约 100M token 的预训练 pilot，并保留一个 1B token 的总 LR horizon，避免到 pilot 末尾就提前衰减到底。代码、数据和 tokenizer 在同一次续训期间必须保持一致；有意改变实验配置应另开输出目录、显式初始化，不能假装无缝续训。

在仓库根目录执行：

```bash
PYTHON=/home/zwb/miniconda3/envs/minimind-v/bin/python
# 只打印配置和指纹，不创建或训练完整模型。
"$PYTHON" trainer/train_text.py --stage pretrain --dry_run --output_dir out/text085b_pretrain_manual_v1

# 约 100M pilot：实际是 100,008,480 有效预测 tokens。
# 首次启动命令；当前手动任务已在运行，不要重复启动。
# 约 1.7 小时的理想估算不含 I/O 波动、验证和保存。
"$PYTHON" trainer/train_text.py --stage pretrain \
  --max_steps 61096 --stop_after 6110 --batch_size 4 --accumulation_steps 4 \
  --max_seq_len 1024 --learning_rate 0.0003 --warmup_steps 100 \
  --save_interval 1000 --log_interval 20 --output_dir out/text085b_pretrain_manual_v1

# 固定中英文／代码／数学分项验证，逐渐增加 batches_per_source 提高可信度。
"$PYTHON" eval_text_loss.py --checkpoint out/text085b_pretrain_manual_v1/last.pt \
  --data_config configs/pretrain_text_085b_validation.json --batches_per_source 128 \
  --output out/text085b_pretrain_manual_v1/validation_loss.json
```

预期启动日志：`params=847,304,960, random_init=True, vision=disabled`，随后显示 main process、buffer=256 及 45/35/10/10 目标。日志应持续出现四来源实际计数、有限 loss/梯度和显存数据。

pilot 后先核对留出 loss、各来源重复轮次和吞吐，再决定是否恢复至 1B token（总计 1,000,019,328），保持相同训练参数和 `--max_steps 61096`，通过 `--resume .../last.pt` 恢复。不要将原始语料总量当作已训练 token 计数。1B 也只是下一阶段实验预算，并不等于模型已经具备通用能力。

文本预训练取得稳定分项基线后，再运行 SFT，例如：

```bash
"$PYTHON" trainer/train_text.py --stage sft \
  --init_weight out/text085b_pretrain_manual_v1/last.pt \
  --max_steps 2000 --batch_size 4 --accumulation_steps 4 --max_seq_len 1024 \
  --learning_rate 0.00002 --output_dir out/text085b_sft_v1
"$PYTHON" eval_text_loss.py --checkpoint out/text085b_sft_v1/last.pt \
  --data_config configs/sft_text_calculator_validation.json --batches_per_source 128 \
  --output out/text085b_sft_v1/validation_loss.json
"$PYTHON" eval_generalization.py --weight out/text085b_sft_v1/last.pt \
  --max_samples 100 --max_new_tokens 512 --calculator \
  --output out/text085b_sft_v1/generalization_calculator.json
```

SFT 每步的有效监督 token 数随回答长度变化，训练量以 counters 为准。随后在相同题目、预算、评分规则下比较开／关工具模式，记录调用率和正确率。视觉扩展继续后置。

## 本地归档与重建边界

旧 SFT 补丁配置、针对固定小测的修补脚本、旧 VLM 文本评测入口和非关键报告已移至 `archive/legacy_*/`，不加入 Git。关键历史结果见 [基线索引](../reports/baselines/README.md)。不要再将旧补丁混入当前训练。

当前 SFT 数据的重建链保留：`gen_arithmetic.py`、`gen_arithmetic_curriculum.py`、`gen_identity.py` → `dataset/build_curated_sft.py` → `dataset/build_balanced_sft.py`；数学与工具来源分别由 `dataset/build_gsm8k_sft.py`、`dataset/build_calculator_sft.py` 生成。训练数据本身仍在原处，当前训练及断点指纹未修改。
