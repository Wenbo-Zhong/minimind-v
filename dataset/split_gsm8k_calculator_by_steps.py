#!/usr/bin/env python3
"""Split verified GSM8K calculator dialogues into short and long tool chains."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path,
                        default=Path('dataset/prepared/gsm8k_calculator_v3'))
    parser.add_argument('--output_dir', type=Path,
                        default=Path('dataset/prepared/gsm8k_calculator_steps_v1'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists: {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    counts = Counter()
    steps = Counter()
    outputs = {
        split: {
            group: (args.output_dir / f'{split}_{group}.jsonl').open('w', encoding='utf-8')
            for group in ('short_1_2', 'long_3_4')
        }
        for split in ('train', 'validation')
    }
    try:
        for split in ('train', 'validation'):
            source = args.input_dir / f'{split}.jsonl'
            if not source.is_file():
                raise FileNotFoundError(source)
            seen = set()
            for line in source.open(encoding='utf-8'):
                if not line.strip():
                    continue
                row = json.loads(line)
                calls = int(row['tool_calls'])
                if calls < 1 or calls > 4:
                    raise ValueError(f"Unexpected call count for {row.get('id')}: {calls}")
                user_text = '\n'.join(t['content'] for t in row['conversations']
                                       if t['role'] == 'user')
                if user_text in seen:
                    raise ValueError(f'Duplicate question in {split}: {row.get("id")}')
                seen.add(user_text)
                group = 'short_1_2' if calls <= 2 else 'long_3_4'
                outputs[split][group].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[f'{split}_{group}'] += 1
                steps[f'{split}_{group}_calls'] += calls
                steps[f'{split}_{group}_n{calls}'] += 1
    finally:
        for groups in outputs.values():
            for handle in groups.values():
                handle.close()

    report = {
        'input_dir': str(args.input_dir),
        'output_dir': str(args.output_dir),
        'partition': 'short_1_2: 1-2 verified calculator calls; long_3_4: 3-4 calls',
        'counts': counts,
        'step_distribution': steps,
        'leakage_note': 'Preserves the source train/validation partition; source excludes official GSM8K test.',
    }
    (args.output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == '__main__':
    main()
