# 应用题计算器评测

`eval_generalization.py`（MiniMind）和 `compare_generalization_lmstudio.py`（LM Studio Qwen）共用 `eval_calculator.py`。默认不启用工具，使用 `--calculator` 启用。两个模型得到相同的中英文工具说明、算术实现和调用上限，工具不会读取题目标准答案。

模型输出 `<calculator>算式</calculator>` 后，程序执行算式，将真实结果作为下一条 user 消息返回，再由模型作答。使用普通对话角色便于兼容两个模型；这是共享文本协议，不是 Qwen 原生 function calling 能力评测。不强制调用，不替模型列式。如果一次输出包含调用及提前生成的答案，保留调用前缀、丢弃其后文本，执行后重新请求模型回答。原始输出和丢弃文本均记录在结果中。

支持十进制数字、括号及 `+ - * / // %` 与整数幂 `**`（指数 -8 到 8）。仅解释受限 AST，不执行 Python 代码；表达式长度、节点数、数值大小均有限制。十进制运算精度为 40 位。错误也会返回模型并消耗一次调用机会。

`--max-new-tokens`（API）/`--max_new_tokens`（MiniMind）是**每题所有轮次合计**的模型生成 token 上限，包括被丢弃的输出。各模型使用各自 tokenizer，因此相同 token 数并不代表相同字符数。工具结果不计入生成预算。默认最多 4 次调用，可用 `--max-tool-calls` 修改。API 必须返回 `usage.completion_tokens`；Qwen 请求沿用 `reasoning_effort=none`。工具请求、工具返回值不会直接作为最终答案评分；最终模型回答沿用原有严格数字评分器。

小样本验证命令（仓库根目录执行；MiniMind 命令须等待新的 SFT checkpoint 生成）：

```bash
PYTHON=/home/zwb/miniconda3/envs/minimind-v/bin/python
mkdir -p reports/evaluations
"$PYTHON" compare_generalization_lmstudio.py --model qwen3-0.6b --calculator --max-samples 10 --max-new-tokens 512 --output reports/evaluations/qwen3_06b_calculator_smoke10.json
"$PYTHON" eval_generalization.py --weight out/text085b_sft_v1/last.pt --calculator --max_samples 10 --max_new_tokens 512 --output reports/evaluations/minimind_calculator_smoke10.json
```

移除 `--calculator` 即运行无工具基线，务必另设输出文件。正式比较时同时固定题目、checkpoint、提示词、工具配置、生成预算和评分规则；不要直接把这里的 512 token 小样本结果与历史 192 token 结果比较。每条 `dialogue` 保存模型输出、工具表达式/结果、生成 tokens、调用次数和停止原因。前 10 题只用于验证接入，不能代表全量泛化能力。计算器可用但模型未调用时，成绩仍应按实际输出记录，不额外人工辅助。

验证：`python -m unittest discover -s tests -p test_eval_calculator.py -v`。

## 2026-10-20 小样本实测

每个数据集取前 10 题，每题合计 512 生成 tokens，启用工具时最多 4 次调用。当时的 MiniMind 权重为 `out/sft_gsm8k_balanced_v2_1024.pth`，现已清理；以下历史结果保存在 `reports/baselines/`，不是新 0.85B 模型的成绩。

| 结果文件 | GSM8K | CMATH | 调用工具的题数 |
| --- | --- | --- | --- |
| `qwen3_06b_no_calculator_smoke10.json` | 2/10 | 7/10 | 0/20 |
| `qwen3_06b_calculator_smoke10.json` | 2/10 | 10/10 | 20/20 |
| `minimind_no_calculator_smoke10.json` | 0/10 | 0/10 | 0/20 |
| `minimind_calculator_smoke10.json` | 1/10 | 0/10 | 0/20 |

这是接入验证，不能外推全量准确率。工具提示也会改变模型输出，分数差异不能全部归因于算术计算。MiniMind 本次没有调用工具，后续需要训练工具协议；Qwen 的 GSM8K 错题仍存在列式错误。
