#!/usr/bin/env python3
"""Convert the local GSM8K train split into checked MiniMind SFT JSONL."""
import argparse
import ast
import json
import math
import operator
import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path('/home/zwb/ai/data/eval/gsm8k/main/train-00000-of-00001.parquet')
DEFAULT_OUTPUT = ROOT / 'dataset/curated_sft/balanced/gsm8k_train_verified.jsonl'
GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
TAG_RE = re.compile(r"<<(.*?)>>\s*([-+]?\d+(?:\.\d+)?)?")


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    raise ValueError(f'unsupported expression node: {ast.dump(node)}')


def number(value):
    return float(str(value).replace(',', '').strip())


def validate_worked_steps(answer):
    for equation in re.findall(r'<<(.*?)>>', answer):
        if '=' not in equation:
            return False
        expression, result = equation.rsplit('=', 1)
        try:
            actual = safe_eval(ast.parse(expression.strip(), mode='eval'))
            expected = number(result)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
            return False
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), expected, rel_tol=1e-3, abs_tol=1e-2
        ):
            return False
    return True


def clean_answer(answer, gold):
    body = GOLD_RE.sub('', answer).strip()

    def replace_tag(match):
        equation, repeated_result = match.groups()
        if repeated_result:
            return repeated_result
        if '=' in equation:
            return equation.rsplit('=', 1)[1].strip()
        return equation

    body = TAG_RE.sub(replace_tag, body)
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    return f'{body}\nFinal answer: {gold}' if body else f'Final answer: {gold}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'GSM8K training parquet not found: {args.input}')

    table = pq.read_table(args.input, columns=['question', 'answer'])
    rows = table.to_pylist()
    accepted = []
    rejected_missing_gold = 0
    rejected_bad_work = 0
    total_annotations = 0

    for row in rows:
        answer = str(row['answer'])
        gold_matches = GOLD_RE.findall(answer)
        if not gold_matches:
            rejected_missing_gold += 1
            continue
        gold = gold_matches[-1].replace(',', '')
        annotations = re.findall(r'<<(.*?)>>', answer)
        total_annotations += len(annotations)
        if not validate_worked_steps(answer):
            rejected_bad_work += 1
            continue
        assistant = clean_answer(answer, gold)
        accepted.append({
            'conversations': [
                {
                    'role': 'user',
                    'content': (
                        'Solve this math word problem. Show a concise calculation, '
                        'then put the final numeric answer on the last line as '
                        '`Final answer: <number>`.\n' + str(row['question']).strip()
                    ),
                },
                {'role': 'assistant', 'content': assistant},
            ],
            'source': 'gsm8k_train_verified',
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as target:
        for row in accepted:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')

    report = {
        'input': str(args.input),
        'output': str(args.output),
        'input_rows': len(rows),
        'accepted_rows': len(accepted),
        'rejected_missing_gold': rejected_missing_gold,
        'rejected_invalid_worked_solution': rejected_bad_work,
        'checked_arithmetic_annotations': total_annotations,
        'validation': 'all accepted answers have a GSM8K final answer and all <<expression=result>> annotations pass safe arithmetic evaluation',
    }
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
