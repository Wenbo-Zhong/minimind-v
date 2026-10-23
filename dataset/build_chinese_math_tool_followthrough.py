#!/usr/bin/env python3
"""Convert verified Chinese math train/validation rows to calculator dialogues."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.build_chinese_math_sft import (  # noqa: E402
    EQUATION_RE,
    FINAL_NUMBER_RE,
    close_numbers,
    evaluate_expression,
)
from dataset.text_data import encode_sft  # noqa: E402
from eval_calculator import CALCULATOR_PROMPT, SYSTEM_PROMPT, calculate  # noqa: E402

DEFAULT_SOURCE = ROOT / "dataset/prepared/chinese_math_verified_v1"
DEFAULT_OUTPUT = ROOT / "dataset/prepared/chinese_math_tool_followthrough_v1"
FINAL_PROMPT = (
    "请解答下面的数学应用题，简要计算，并将最终数值答案单独放在最后一行，"
    "格式为“最终答案：数字”。\n"
)
PERCENT_RE = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%")


def tool_expression(expression):
    expression = expression.replace(",", "").replace("×", "*").replace("÷", "/")
    return PERCENT_RE.sub(r"(\1 / 100)", expression)


def to_dialogue(row):
    turns = row["conversations"]
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        raise ValueError("unexpected_verified_row_format")
    prompt = turns[0]["content"]
    question = prompt.split("\n", 1)[-1].strip()
    reasoning_and_answer = turns[-1]["content"]
    equations = list(EQUATION_RE.finditer(reasoning_and_answer))
    if not equations:
        raise ValueError("no_checked_equations")
    final_numbers = FINAL_NUMBER_RE.findall(reasoning_and_answer.rsplit("最终答案：", 1)[-1])
    if not final_numbers:
        raise ValueError("missing_final_answer")
    gold = final_numbers[-1].replace(",", "")

    dialogue = [
        {"role": "system", "content": SYSTEM_PROMPT + CALCULATOR_PROMPT},
        {"role": "user", "content": FINAL_PROMPT + question},
    ]
    verified_results = []
    for match in equations:
        source_expression = match.group("expression")
        source_result = match.group("result").replace(",", "")
        # Recheck the source annotation, then express percentages in syntax the
        # bounded calculator supports (where % otherwise means modulo).
        if not close_numbers(evaluate_expression(source_expression), Decimal(source_result)):
            raise ValueError("source_equation_mismatch")
        expression = tool_expression(source_expression)
        result = calculate(expression)
        if not close_numbers(Decimal(result), Decimal(source_result)):
            raise ValueError("calculator_expression_mismatch")
        verified_results.append(result)
        dialogue.extend([
            {"role": "assistant", "content": f"<calculator>{expression}</calculator>"},
            {"role": "user", "content": "Calculator result: " + result},
        ])
    if not close_numbers(Decimal(verified_results[-1]), Decimal(gold)):
        raise ValueError("final_equation_mismatch")
    dialogue.append({"role": "assistant", "content": "最终答案：" + gold})
    return dialogue, len(equations), gold, question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "model")
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; use another directory: {args.output_dir}")
    for split in ("train", "validation"):
        if not (args.source_dir / f"{split}.jsonl").is_file():
            parser.error(f"missing verified source split: {args.source_dir / (split + '.jsonl')}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    args.output_dir.mkdir(parents=True)
    counts, categories, token_counts = Counter(), Counter(), Counter()
    seen_by_split = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        output_path = args.output_dir / f"{split}.jsonl"
        with (args.source_dir / f"{split}.jsonl").open(encoding="utf-8") as source, \
                output_path.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                dialogue, calls, gold, question = to_dialogue(row)
                encoded = encode_sft(tokenizer, dialogue, args.max_length)
                if encoded is None:
                    counts[f"rejected_{split}_overlong_or_invalid"] += 1
                    continue
                if question in seen_by_split[split]:
                    counts[f"rejected_{split}_duplicate"] += 1
                    continue
                seen_by_split[split].add(question)
                fingerprint = hashlib.sha256((split + "\0" + question).encode()).hexdigest()
                target.write(json.dumps({
                    "id": f"zh-tool-followthrough-{split}-" + fingerprint[:16],
                    "source": "chinese_math_tool_followthrough_verified",
                    "category": row.get("category"),
                    "difficulty": row.get("difficulty"),
                    "verified_tool_calls": calls,
                    "conversations": dialogue,
                }, ensure_ascii=False) + "\n")
                labels = encoded[1]
                eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
                final_eos = max(index for index, token_id in enumerate(encoded[0]) if token_id == eos)
                if labels[final_eos] != eos:
                    raise ValueError("final_assistant_eos_not_supervised")
                counts[split] += 1
                counts[f"{split}_tool_calls"] += calls
                categories[f"{split}_{row.get('category', 'unknown')}"] += 1
                token_counts[split] += sum(value != -100 for value in labels[1:])

    report = {
        "source_dir": str(args.source_dir),
        "source_report": str(args.source_dir / "report.json"),
        "output_dir": str(args.output_dir),
        "source_license": "MIT (shangshang/math-reasoning-zh; verified source report)",
        "construction": [
            "reuses only the already verified train and validation splits",
            "re-evaluates every extracted source equation and generated calculator expression",
            "requires the final calculator result to match the verified final answer",
            "each assistant tool call is followed by its actual calculator result; the final answer is a separate assistant turn",
            "checks the final assistant <|im_end|> is supervised by encode_sft",
            "inherits the source exclusion of GSM8K test and CMATH dev/test questions",
        ],
        "max_length": args.max_length,
        "counts": counts,
        "categories": categories,
        "effective_assistant_tokens": token_counts,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
