# 预训练混合配置

当前纯文本实验使用 `trainer/train_text.py --stage pretrain`，默认读取 `pretrain_text_085b.json`。完整训练参数、数据限制与断点恢复说明见 [训练方案](text_training_plan.md)。

| 来源 | 有效预测 token 目标 |
| --- | ---: |
| SkyPile 中文网页 | 45% |
| FineWeb-Edu 英文教育文本 | 35% |
| Infinity-Instruct 7M_domains/code | 10% |
| Infinity-Instruct 7M_domains/math | 10% |

四种来源等长 packing，每块 `max_seq_len - 1` 个有效预测位置；每 100 个 pack 为完整配比周期，尾部不完整周期有少量偏差。文件／row group 及来源内文档 buffer 均打乱；当前主进程单卡读取，checkpoint 保存游标、buffer、packing 余量、来源调度和随机数状态。

代码／数学来源是指令对话，完整序列均参与因果预训练 loss，不是独立原始代码或数学书语料。语料存量不是已训练量，实际消耗以各来源 `loss_tokens` 为准。当前尚未完成全库去重、网页质量筛选和语义近重复污染检查。

`pretrain_text_085b_validation.json` 使用相同来源的确定性留出分割，分项验证入口是 `eval_text_loss.py`。

## 保留的 VLM 配置

`trainer/train_pretrain_vlm.py` 仍默认使用 `pretrain_repair_packed.json`，比例同为 45/35/10/10，但使用旧 VLM 读取器，仅打乱来源顺序，不能等同于新纯文本数据管线。`local_datasets.json` 和 `local_datasets_stage2.json` 保留给历史多模态流程；它们不参与当前纯文本实验。

VLM 入口有自己的模型尺寸和旧权重初始化默认值。新 0.85B 从零训练统一使用 `trainer/train_text.py`，不要仅替换数据配置便假定模型结构和初始化也已切换。
