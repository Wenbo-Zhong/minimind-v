#!/usr/bin/env python3
"""Validate and convert Chinese arithmetic word problems into SFT examples."""
import argparse
import ast
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.text_data import encode_sft, load_exclusions, question_key

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
EQUATION_RE = re.compile(
    rf"(?P<expression>{NUMBER}%?(?:\s*[+\-×*/÷]\s*{NUMBER}%?)+)\s*=\s*(?P<result>{NUMBER})"
)
FINAL_PROMPT = (
    "请解答下面的数学应用题，简要写出计算过程，并将最终数值答案单独放在最后一行，"
    "格式为“最终答案：数字”。\n"
)
FINAL_NUMBER_RE = re.compile(NUMBER)


def evaluate_expression(expression):
    """Safely evaluate the arithmetic expressions used in this data source."""
    expression = expression.replace(",", "").replace("×", "*").replace("÷", "/")
    expression = re.sub(r"((?:\d+(?:\.\d+)?))%", r"(\1/100)", expression)
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(ast.get_source_segment(expression, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported arithmetic syntax")

    with localcontext() as context:
        context.prec = 40
        result = visit(tree)
    if not result.is_finite():
        raise ValueError("non-finite arithmetic result")
    return result


def close_numbers(left, right):
    left, right = Decimal(left), Decimal(right)
    return abs(left - right) <= max(Decimal("1e-8"), abs(right) * Decimal("1e-8"))


def validate_row(row):
    question = str(row.get("question", "")).strip()
    reasoning = str(row.get("chain_of_thought", "")).strip()
    final_answer = str(row.get("final_answer", "")).strip()
    if not question or not reasoning or not final_answer:
        raise ValueError("missing_question_or_answer")

    matches = list(EQUATION_RE.finditer(reasoning))
    if not matches:
        raise ValueError("no_verifiable_equations")
    results = []
    for match in matches:
        actual = evaluate_expression(match.group("expression"))
        expected = Decimal(match.group("result").replace(",", ""))
        if not close_numbers(actual, expected):
            raise ValueError("arithmetic_mismatch")
        results.append(expected)

    answer_numbers = FINAL_NUMBER_RE.findall(final_answer)
    if not answer_numbers:
        raise ValueError("missing_numeric_final_answer")
    gold = answer_numbers[-1].replace(",", "")
    if not close_numbers(results[-1], gold):
        raise ValueError("final_answer_mismatch")
    return question, reasoning, gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/zwb/ai/data/sft/open_math/math-reasoning-zh/data/math_reasoning_zh.jsonl"),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "dataset/prepared/chinese_math_verified_v1"
    )
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "model")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--exclude_questions",
        nargs="*",
        default=[
            "/home/zwb/ai/data/eval/gsm8k/main/test-00000-of-00001.parquet",
            "/home/zwb/ai/data/eval/cmath/cmath_dev.jsonl",
            "/home/zwb/ai/data/eval/cmath/cmath_test.jsonl",
        ],
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input dataset not found: {args.input}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; use another directory: {args.output_dir}")

    exclusions = load_exclusions(args.exclude_questions)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    args.output_dir.mkdir(parents=True)
    handles = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation")
    }
    counts, rejected = Counter(), Counter()
    seen = set()
    token_counts = Counter()
    try:
        for line in args.input.open(encoding="utf-8"):
            if not line.strip():
                continue
            counts["scanned"] += 1
            try:
                row = json.loads(line)
                question, reasoning, gold = validate_row(row)
                key = question_key(question)
                if key in exclusions:
                    raise ValueError("evaluation_question_overlap")
                if key in seen:
                    raise ValueError("duplicate_question")
                turns = [
                    {"role": "user", "content": FINAL_PROMPT + question},
                    {
                        "role": "assistant",
                        "content": reasoning + "\n最终答案：" + gold,
                    },
                ]
                encoded = encode_sft(tokenizer, turns, args.max_length)
                if encoded is None:
                    raise ValueError("overlong_or_invalid_chat")
            except (ValueError, ArithmeticError, SyntaxError, TypeError, KeyError) as exc:
                rejected[str(exc)[:100]] += 1
                continue

            seen.add(key)
            fingerprint = hashlib.sha256(key.encode()).hexdigest()
            split = "validation" if int(fingerprint[:16], 16) % 10 == 0 else "train"
            handles[split].write(
                json.dumps(
                    {
                        "id": "chinese-math-" + fingerprint[:16],
                        "source": "chinese_math_equations_verified",
                        "category": row.get("category"),
                        "difficulty": row.get("difficulty"),
                        "conversations": turns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[split] += 1
            counts[f"{split}_{row.get('category', 'unknown')}"] += 1
            token_counts[split] += sum(label != -100 for label in encoded[1][1:])
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "source_license": "MIT (shangshang/math-reasoning-zh, local README metadata)",
        "checks": [
            "every extracted arithmetic equality is re-evaluated with bounded Decimal arithmetic",
            "final numeric answer must match the final checked equation",
            "normalized duplicate and listed evaluation questions are excluded",
            "deterministic 90/10 train/validation split by normalized question hash",
        ],
        "max_length": args.max_length,
        "counts": counts,
        "rejections": rejected,
        "effective_assistant_tokens": token_counts,
        "exclude_questions": args.exclude_questions,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
