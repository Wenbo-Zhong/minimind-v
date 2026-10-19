# 历史评测基线

这些文件保留原始内容，仅从仓库根目录搬到本目录。它们记录旧模型和工具接入实验，不代表当前从零训练的 0.85B 模型能力。旧 checkpoint 已按用户要求删除。

| 文件 | 保留目的 |
| --- | --- |
| [gsm8k_balanced_v2_vs_qwen3_06b_strict_audit.json](gsm8k_balanced_v2_vs_qwen3_06b_strict_audit.json) | 100 题严格评分：旧 MiniMind 与 Qwen 对比 |
| [sft_micro_repair_generalization_v2_full_strict.json](sft_micro_repair_generalization_v2_full_strict.json) | 旧 micro-repair 的全量严格基线：GSM8K 0/1319、CMATH 4/1098 |
| [qwen3_06b_192_100.json](qwen3_06b_192_100.json) | 历史 192-token 非严格 Qwen 基线：12/100、60/100；不可与严格结果混用 |
| [sft_gsm8k_balanced_v2_1024.eval.json](sft_gsm8k_balanced_v2_1024.eval.json) | 旧模型 10 题固定能力小测 9/10；不是留出集泛化成绩 |
| [qwen3_06b_calculator_smoke10.json](qwen3_06b_calculator_smoke10.json) | 512-token 计算器接入验证：GSM8K 2/10、CMATH 10/10 |
| [qwen3_06b_no_calculator_smoke10.json](qwen3_06b_no_calculator_smoke10.json) | 同题同预算无工具对照：2/10、7/10 |
| [minimind_calculator_smoke10.json](minimind_calculator_smoke10.json) | 旧 MiniMind 开放工具：1/10、0/10，实际没有调用 |
| [minimind_no_calculator_smoke10.json](minimind_no_calculator_smoke10.json) | 同题同预算无工具对照：0/10、0/10 |

JSON 内 `weight`、`rescored_from` 等字段保持实验发生时的原始文件名，不代表当前文件系统上仍存在对应权重。非关键旧结果和原始副本保存在本地忽略目录 `archive/legacy_*/`，归档清单记录原路径和 SHA256；它们没有被当成同一次实验或相同评分结果覆盖。

新的运行输出放到 `out/<实验名>/`，不要覆盖本目录的历史记录。语料存量统计保留在仓库根目录 `pretrain_corpus_audit.{md,json}`。
