#!/usr/bin/env python3
"""Convert Calc-ape TRAIN only to verified calculator dialogs; never trust outputs."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from dataset.text_data import encode_sft, load_exclusions, question_key
from eval_calculator import calculate, SYSTEM_PROMPT, CALCULATOR_PROMPT

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zwb/ai/data')


def close_numbers(a, b):
    a, b = Decimal(a), Decimal(b)
    return abs(a - b) <= max(Decimal('1e-8'), abs(b) * Decimal('1e-8'))


def convert(row, max_calls=4):
    root = ET.fromstring('<chain>' + row['chain'] + '</chain>')
    children = list(root)
    if root.text and root.text.strip():
        raise ValueError('unparsed chain text')
    calls, last = [], None
    index = 0
    while index < len(children) and children[index].tag == 'gadget':
        gadget = children[index]
        if gadget.get('id') != 'calculator' or index + 1 >= len(children) or children[index + 1].tag != 'output':
            raise ValueError('invalid gadget/output structure')
        expression = (gadget.text or '').strip()
        actual = calculate(expression)
        expected = (children[index + 1].text or '').split('=')[0].strip()
        if not close_numbers(actual, calculate(expected)):
            raise ValueError('tool output mismatch')
        calls.append((expression, actual))
        last = actual
        index += 2
    if not calls or len(calls) > max_calls:
        raise ValueError('call count outside budget')
    if index != len(children) - 1 or children[index].tag != 'result':
        raise ValueError('invalid final result structure')
    if any(node.tail and node.tail.strip() for node in children):
        raise ValueError('unparsed chain tail')
    gold = calculate(row['result'])
    if not close_numbers(last, gold) or not close_numbers(last, calculate(children[index].text)):
        raise ValueError('final result mismatch')
    return calls, last


def make_turns(question, calls, answer, language):
    if language == 'zh':
        prompt = '请解答下面的数学题。请简要计算，并将最终数值答案单独放在最后一行，格式为“最终答案：数字”。\n'
        final = '最终答案：' + answer
    else:
        prompt = ('Solve this math word problem. Show only enough work to calculate it. '
                  'Put the final numeric answer on the last line as `Final answer: <number>`.\n')
        final = 'Final answer: ' + answer
    turns = [{'role': 'system', 'content': SYSTEM_PROMPT + CALCULATOR_PROMPT},
             {'role': 'user', 'content': prompt + question}]
    for expression, result in calls:
        turns.extend([{'role': 'assistant', 'content': f'<calculator>{expression}</calculator>'},
                      {'role': 'user', 'content': 'Calculator result: ' + result}])
    turns.append({'role': 'assistant', 'content': final})
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(DATA / 'sft/open_math/Calc-ape210k/data'))
    parser.add_argument('--output_dir', default=str(ROOT / 'dataset/prepared/calculator_v1'))
    parser.add_argument('--tokenizer', default=str(ROOT / 'model'))
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_calls', type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'Output already exists; use a new directory: {output}')
    files = sorted(Path(args.source).glob('train-*.parquet'))
    if not files:
        raise FileNotFoundError('No train shards in ' + args.source)
    excluded = sorted(str(p) for p in Path(args.source).glob('validation-*.parquet'))
    excluded += sorted(str(p) for p in Path(args.source).glob('test-*.parquet'))
    excluded += [str(DATA / 'eval/gsm8k/main/test-00000-of-00001.parquet'),
                 str(DATA / 'eval/cmath/cmath_dev.jsonl'), str(DATA / 'eval/cmath/cmath_test.jsonl')]
    blocked = load_exclusions(excluded)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    output.mkdir(parents=True)
    counts, reasons, tokens = Counter(), Counter(), Counter()
    seen = set()
    handles = {s: (output / (s + '.jsonl')).open('w') for s in ('train', 'validation')}
    try:
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    counts['scanned_questions'] += 1
                    questions = [str(row.get('question_chinese') or '').strip(), str(row.get('question') or '').strip()]
                    keys = {question_key(q) for q in questions if q}
                    if not keys or keys & blocked or keys & seen:
                        reasons['heldout_overlap_or_duplicate'] += 1
                        continue
                    seen.update(keys)
                    try:
                        calls, answer = convert(row, args.max_calls)
                    except (ValueError, ArithmeticError, SyntaxError, ET.ParseError, TypeError) as exc:
                        reasons[str(exc)[:100]] += 1
                        continue
                    # Split by original question, keeping both languages together.
                    ident = hashlib.sha256(questions[0].encode()).hexdigest()
                    split = 'validation' if int(ident[:16], 16) % 100 == 0 else 'train'
                    kept = False
                    for question, language in zip(questions, ('zh', 'en')):
                        if not question:
                            continue
                        turns = make_turns(question, calls, answer, language)
                        encoded = encode_sft(tokenizer, turns, args.max_length)
                        if encoded is None:
                            reasons['overlong_or_invalid_chat'] += 1
                            continue
                        entry = {'id': row['id'], 'source': 'calc_ape_verified', 'language': language,
                                 'conversations': turns, 'tool_calls': len(calls)}
                        handles[split].write(json.dumps(entry, ensure_ascii=False) + '\n')
                        counts[split + '_dialogs'] += 1
                        tokens[split] += sum(x != -100 for x in encoded[1][1:])
                        kept = True
                    if kept:
                        counts[split + '_questions'] += 1
    finally:
        for handle in handles.values():
            handle.close()
    report = {'source_files': files, 'excluded_files': excluded, 'counts': counts,
              'rejections': reasons, 'effective_assistant_tokens': tokens,
              'max_length': args.max_length, 'max_calls': args.max_calls,
              'protocol_sha256': hashlib.sha256(CALCULATOR_PROMPT.encode()).hexdigest()}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
