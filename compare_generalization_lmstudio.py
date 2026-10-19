#!/usr/bin/env python3
"""Run the MiniMind GSM8K/CMATH generalization prompts against LM Studio."""
import argparse
from eval_calculator import add_calculator_args, calculator_settings, run_dialogue
import json
from pathlib import Path

import requests

from eval_generalization import (
    extract_prediction,
    gsm_gold,
    normalize_number,
    read_gsm_rows,
)


def load_cmath(path, limit):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--gsm8k-path",
        default="/home/zwb/ai/data/eval/gsm8k/main/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--cmath-path", default="/home/zwb/ai/data/eval/cmath/cmath_test.jsonl"
    )
    parser.add_argument("--output", default="qwen3_06b_generalization_100.json")
    add_calculator_args(parser)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_tool_calls < 0:
        parser.error("invalid token budget or tool call limit")
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")

    base = args.base_url.rstrip("/")
    model_ids = requests.get(base + "/models", timeout=10).json().get("data", [])
    if args.model not in {item.get("id") for item in model_ids}:
        parser.error(f"Model {args.model!r} is not listed by LM Studio: {model_ids}")

    gsm_rows = read_gsm_rows(args.gsm8k_path, args.max_samples)
    cmath_rows = load_cmath(args.cmath_path, args.max_samples)
    results = {
        "gsm8k": {"passed": 0, "total": 0, "results": []},
        "cmath": {"passed": 0, "total": 0, "results": []},
    }
    output = Path(args.output)

    def save(complete=False):
        output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "calculator": calculator_settings(args),
                    "complete": complete,
                    "max_samples_per_dataset": args.max_samples,
                    "max_new_tokens": args.max_new_tokens,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def generate_turn(messages, remaining_tokens):
        response = requests.post(
            base + "/chat/completions",
            json={
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": remaining_tokens,
                "reasoning_effort": "none",
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        if message.get("tool_calls"):
            raise ValueError("Server returned native tool_calls; this evaluation requires raw calculator XML")
        answer = (message.get("content") or "").strip()
        count = payload.get("usage", {}).get("completion_tokens")
        if count is None:
            raise ValueError("Server must report completion_tokens to enforce the shared token budget")
        return answer, count

    def evaluate(dataset, row, question, gold, user_prompt):
        answer, dialogue = run_dialogue(generate_turn, user_prompt, args.max_new_tokens,
                                        args.calculator, args.max_tool_calls)
        pred = extract_prediction(answer)
        passed = pred is not None and normalize_number(gold) == pred
        bucket = results[dataset]
        bucket["total"] += 1
        bucket["passed"] += int(passed)
        bucket["results"].append(
            {
                "question": question,
                "gold": normalize_number(gold),
                "pred": pred,
                "passed": passed,
                "answer": answer,
                "dialogue": dialogue,
            }
        )
        print(
            f"[{dataset} {bucket['total']}] {'PASS' if passed else 'FAIL'} "
            f"gold={normalize_number(gold)} pred={pred} answer={answer[:180]!r}",
            flush=True,
        )
        if bucket["total"] % 5 == 0:
            save()

    print(f"LM Studio model: {args.model}", flush=True)
    print(f"GSM8K: evaluating {len(gsm_rows)} examples", flush=True)
    for row in gsm_rows:
        prompt = (
            "Solve this math word problem. Show only enough work to calculate it. "
            "Put the final numeric answer on the last line as `Final answer: <number>`.\n"
            + row["question"]
        )
        evaluate("gsm8k", row, row["question"], gsm_gold(row["answer"]), prompt)

    print(f"CMATH: evaluating {len(cmath_rows)} examples", flush=True)
    for row in cmath_rows:
        prompt = (
            "请解答下面的数学题。请简要计算，并将最终数值答案单独放在最后一行，格式为“最终答案：数字”。\n"
            + row["question"]
        )
        evaluate("cmath", row, row["question"], row["golden"], prompt)

    save(complete=True)
    for name, summary in results.items():
        print(f"{name}: {summary['passed']}/{summary['total']}", flush=True)
    print("Result:", output, flush=True)


if __name__ == "__main__":
    main()
