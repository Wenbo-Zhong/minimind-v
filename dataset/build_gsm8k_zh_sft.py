#!/usr/bin/env python3
"""Build answer-verified Chinese GSM8K training augmentation for SFT."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.text_data import encode_sft, load_exclusions, question_key

FINAL_MARKER = re.compile(r"(?:答案\s*(?:是|为)\s*[:：]?|最终答案\s*[:：]?|####)\s*", re.I)
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
OFFICIAL_FINAL = re.compile(r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
PROMPT = (
    "请解答下面的数学应用题，简要写出计算过程，并将最终数值答案单独放在最后一行，"
    "格式为“最终答案：数字”。\n"
)


def canonical_number(value):
    return Decimal(str(value).replace(",", ""))


def official_gold(answer):
    matches = OFFICIAL_FINAL.findall(str(answer))
    if not matches:
        raise ValueError("missing_official_gsm8k_gold")
    return canonical_number(matches[-1])


def parse_chinese_final(answer):
    """Return the last explicitly marked final numeric answer, with 万/亿 support."""
    matches = list(FINAL_MARKER.finditer(str(answer)))
    if not matches:
        raise ValueError("missing_explicit_chinese_final_answer")
    tail = str(answer)[matches[-1].end():].lstrip()
    # Do not mistake a clock time (e.g. 4:30) for a numeric final answer.
    match = NUMBER.match(tail)
    if not match:
        raise ValueError("unparseable_chinese_final_answer")
    end = match.end()
    if end < len(tail) and tail[end] == ":":
        raise ValueError("clock_or_colon_final_answer")
    value = canonical_number(match.group())
    suffix = tail[end:end + 1]
    if suffix == "万":
        value *= Decimal(10000)
    elif suffix == "亿":
        value *= Decimal(100000000)
    return value


def validate_row(row, official):
    if row.get("split") != "train":
        raise ValueError("not_training_split")
    question = str(row.get("question_zh", "")).strip()
    english = str(row.get("question", "")).strip()
    answer = str(row.get("answer_zh", "")).strip()
    if not question or not english or not answer:
        raise ValueError("missing_question_or_answer")
    english_key = question_key(english)
    if english_key not in official:
        raise ValueError("question_missing_from_official_gsm8k_train")
    gold = official[english_key]
    if canonical_number(row.get("answer_only", "")) != gold:
        raise ValueError("answer_only_mismatches_official_gsm8k_gold")
    if parse_chinese_final(answer) != gold:
        raise ValueError("chinese_final_mismatches_official_gsm8k_gold")
    return question, english_key, format(gold, "f")


def official_training_map(path):
    result = {}
    table = pq.read_table(path, columns=["question", "answer"])
    for row in table.to_pylist():
        key = question_key(row["question"])
        if key in result and result[key] != official_gold(row["answer"]):
            raise ValueError("conflicting_official_gsm8k_rows")
        result[key] = official_gold(row["answer"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/home/zwb/ai/data/sft/open_math/GSM8K_zh/GSM8K_zh.json"))
    parser.add_argument("--official_train", type=Path, default=Path("/home/zwb/ai/data/eval/gsm8k/main/train-00000-of-00001.parquet"))
    parser.add_argument("--output_dir", type=Path, default=ROOT / "dataset/prepared/gsm8k_zh_verified_v1")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "model")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--exclude_questions", nargs="*", default=[
        "/home/zwb/ai/data/eval/gsm8k/main/test-00000-of-00001.parquet",
        "/home/zwb/ai/data/eval/cmath/cmath_dev.jsonl",
        "/home/zwb/ai/data/eval/cmath/cmath_test.jsonl",
    ])
    args = parser.parse_args()
    for path in (args.input, args.official_train):
        if not path.is_file():
            parser.error(f"required dataset not found: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists; use another directory: {args.output_dir}")

    official = official_training_map(args.official_train)
    exclusions = load_exclusions(args.exclude_questions)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    args.output_dir.mkdir(parents=True)
    handles = {split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
               for split in ("train", "validation")}
    counts, rejected, seen = Counter(), Counter(), set()
    try:
        with args.input.open(encoding="utf-8") as source:
            for row in json.load(source):
                counts["scanned"] += 1
                try:
                    question, english_key, gold = validate_row(row, official)
                    key = question_key(question)
                    if key in exclusions or english_key in exclusions:
                        raise ValueError("evaluation_question_overlap")
                    if key in seen or english_key in seen:
                        raise ValueError("duplicate_question")
                    turns = [
                        {"role": "user", "content": PROMPT + question},
                        {"role": "assistant", "content": row["answer_zh"].strip() + "\n最终答案：" + gold},
                    ]
                    encoded = encode_sft(tokenizer, turns, args.max_length)
                    if encoded is None:
                        raise ValueError("overlong_or_invalid_chat")
                except (ValueError, ArithmeticError, TypeError, KeyError) as exc:
                    rejected[str(exc)[:120]] += 1
                    continue
                seen.update((key, english_key))
                fingerprint = hashlib.sha256(english_key.encode()).hexdigest()
                split = "validation" if int(fingerprint[:16], 16) % 10 == 0 else "train"
                handles[split].write(json.dumps({
                    "id": "gsm8k-zh-" + fingerprint[:16],
                    "source": "gsm8k_zh_answer_verified",
                    "conversations": turns,
                }, ensure_ascii=False) + "\n")
                counts[split] += 1
                counts[split + "_assistant_tokens"] += sum(label != -100 for label in encoded[1][1:])
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "input": str(args.input), "official_train": str(args.official_train),
        "output_dir": str(args.output_dir), "source_license": "MIT (meta-math/GSM8K_zh dataset card)",
        "source_note": "Chinese translations of GSM8K training problems; not novel problem coverage.",
        "checks": [
            "train split only; official GSM8K train question and answer_only cross-checked",
            "last explicit Chinese final answer must equal official GSM8K gold (including 万/亿)",
            "GSM8K test and CMATH dev/test question overlap excluded",
            "normalized duplicates excluded; deterministic 90/10 split by original question hash",
        ],
        "max_length": args.max_length, "counts": counts, "rejections": rejected,
        "exclude_questions": args.exclude_questions,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
