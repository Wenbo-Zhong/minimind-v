#!/usr/bin/env python3
"""Turn verified GSM8K train annotations into calculator-dialogue SFT data."""
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
from dataset.build_gsm8k_sft import GOLD_RE, validate_worked_steps
from dataset.text_data import encode_sft, question_key
from eval_calculator import calculate, SYSTEM_PROMPT, CALCULATOR_PROMPT

ANNOTATION = re.compile(r"<<(.*?)>>", re.S)
PROMPT = (
    'Solve this math word problem. Show a concise calculation, then put the final numeric answer '
    'on the last line as `Final answer: <number>`.\n'
)


def close_numbers(left, right):
    left, right = Decimal(left), Decimal(right)
    return abs(left - right) <= max(Decimal('1e-8'), abs(right) * Decimal('1e-8'))


def convert(question, answer, max_calls):
    matches = list(ANNOTATION.finditer(answer))
    if not matches or len(matches) > max_calls:
        raise ValueError('call_count_outside_budget')
    if not validate_worked_steps(answer):
        raise ValueError('invalid_annotated_arithmetic')
    golds = GOLD_RE.findall(answer)
    if not golds:
        raise ValueError('missing_final_answer')
    gold = golds[-1].replace(',', '')

    turns = [
        {'role': 'system', 'content': SYSTEM_PROMPT + CALCULATOR_PROMPT},
        {'role': 'user', 'content': PROMPT + question.strip()},
    ]
    cursor = 0
    assistant_turn = ''
    for match in matches:
        expression, expected = match.group(1).rsplit('=', 1)
        expression = expression.strip()
        actual = calculate(expression)
        if not close_numbers(actual, calculate(expected.strip())):
            raise ValueError('calculator_annotation_mismatch')
        turns.extend([
            # Keep tool requests identical to the protocol instruction: the
            # assistant emits only the tool call. GSM8K's prose around the
            # annotated equations is omitted from intermediate turns because
            # rationale plus a call conflicts with "output ONLY" and is not
            # representable by the evaluator's tool-turn protocol.
            {'role': 'assistant', 'content': f'<calculator>{expression}</calculator>'},
            {'role': 'user', 'content': 'Calculator result: ' + actual},
        ])
        cursor = match.end()

    tail = GOLD_RE.sub('', answer[cursor:]).strip()
    tail = re.sub(r'\s+', ' ', tail)
    final = tail.strip()
    final = (final + '\n' if final else '') + 'Final answer: ' + gold
    turns.append({'role': 'assistant', 'content': final})
    return turns, len(matches), gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('/home/zwb/ai/data/eval/gsm8k/main/train-00000-of-00001.parquet'))
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'dataset/prepared/gsm8k_calculator_v5')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'model')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_calls', type=int, default=4)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'GSM8K train file not found: {args.input}')
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists; use another directory: {args.output_dir}')

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    args.output_dir.mkdir(parents=True)
    files = {split: (args.output_dir / f'{split}.jsonl').open('w', encoding='utf-8')
             for split in ('train', 'validation')}
    counts, reasons = Counter(), Counter()
    token_counts = Counter()
    try:
        rows = pq.read_table(args.input, columns=['question', 'answer']).to_pylist()
        for row in rows:
            counts['scanned'] += 1
            question, answer = str(row['question']).strip(), str(row['answer'])
            try:
                turns, calls, gold = convert(question, answer, args.max_calls)
                encoded = encode_sft(tokenizer, turns, args.max_length)
                if encoded is None:
                    raise ValueError('overlong_or_invalid_chat')
            except (ValueError, ArithmeticError, SyntaxError, TypeError, ZeroDivisionError) as exc:
                reasons[str(exc)[:100]] += 1
                continue

            # Match TextMixture's document split key so the tool and no-tool
            # versions of one GSM8K question always land in the same split.
            key = hashlib.sha256(question_key(PROMPT + question).encode()).hexdigest()
            split = 'validation' if int(key[:16], 16) % 100 == 0 else 'train'
            files[split].write(json.dumps({
                'id': 'gsm8k-' + key[:16],
                'source': 'gsm8k_calculator_verified',
                'tool_calls': calls,
                'conversations': turns,
            }, ensure_ascii=False) + '\n')
            counts[split] += 1
            counts[f'{split}_tool_calls'] += calls
            token_counts[split] += sum(label != -100 for label in encoded[1][1:])
            counts[f'{split}_steps_{calls}'] += 1
    finally:
        for handle in files.values():
            handle.close()

    report = {
        'input': str(args.input), 'output_dir': str(args.output_dir),
        'max_length': args.max_length, 'max_calls': args.max_calls,
        'counts': counts, 'rejections': reasons,
        'effective_assistant_tokens': token_counts,
        'split': 'SHA256(question_key(prompt + question)) modulo 100; train 99%, validation 1%',
        'source_license': 'MIT (OpenAI GSM8K)',
        'validation_exclusion': 'GSM8K test is a separate official source split and was not read.',
    }
    (args.output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == '__main__':
    main()
