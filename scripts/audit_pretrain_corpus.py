"""Read-only corpus audit using the local tokenizer; write only the JSON report.

Parquet row counts/native token counts are exact. Local-tokenizer totals are
estimates: stratified row sampling for parquet, uniform byte-position sampling
for JSONL. JSONL uses the containing line (not the next line), so averaging
tokens / physical line bytes accounts for length-biased sampling correctly.
The intervals quantify sampling error, not deduplication or corpus quality.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def mean_se(values):
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0


def containing_line(handle, offset):
    start = max(0, offset - 65536)
    while True:
        handle.seek(start)
        prefix = handle.read(offset - start)
        newline = prefix.rfind(b'\n')
        if newline >= 0 or start == 0:
            handle.seek(start + newline + 1 if newline >= 0 else 0)
            return handle.tell(), handle.readline()
        start = max(0, start - 65536)


def audit_jsonl(files, tokenizer, rng, samples):
    details, estimate, variance, row_estimate = [], 0, 0, 0
    hashes, unique_offsets, empty, invalid = set(), set(), 0, 0
    repeated_text_draws = 0
    for path in files:
        size = path.stat().st_size
        ratios, row_ratios, truncation_ratios = [], [], []
        with path.open('rb') as handle:
            for _ in range(samples):
                offset, line = containing_line(handle, rng.randrange(size))
                # Every sampled byte must contribute; malformed/empty rows have
                # zero usable tokens, rather than silently excluding them.
                row_ratios.append(1 / len(line))
                try:
                    row = json.loads(line)
                    text = row.get('text', row.get('content', ''))
                except (ValueError, UnicodeError):
                    invalid += 1
                    text = ''
                length = len(tokenizer.encode(str(text), add_special_tokens=False).ids) if text else 0
                empty += not bool(text)
                ratios.append(length / len(line))
                truncation_ratios.append(min(length, 449) / len(line))
                location = (str(path), offset)
                if location not in unique_offsets and text:
                    digest = hashlib.sha256(str(text).encode()).hexdigest()
                    repeated_text_draws += digest in hashes
                    hashes.add(digest)
                    unique_offsets.add(location)
        average, se = mean_se(ratios)
        value = size * average
        estimate += value
        variance += (size * se) ** 2
        row_estimate += size * statistics.mean(row_ratios)
        details.append({'file': str(path), 'bytes': size, 'draws': samples,
                        'estimated_text_tokens': value, 'standard_error': size * se,
                        'estimated_rows': size * statistics.mean(row_ratios),
                        'estimated_tokens_after_449_token_document_truncation': size * statistics.mean(truncation_ratios)})
        print(f'JSONL {path.name}: estimated {value / 1e9:.3f}B tokens', flush=True)
    return {'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
            'estimated_rows': row_estimate, 'estimated_local_text_tokens': estimate,
            'sampling_95pct_interval': [estimate - 1.96 * math.sqrt(variance), estimate + 1.96 * math.sqrt(variance)],
            'unique_sampled_records': len(unique_offsets), 'sampled_exact_text_duplicates': repeated_text_draws,
            'empty_draws': empty, 'invalid_draws': invalid, 'details': details}


def audit_parquet(files, tokenizer, rng, groups, rows):
    total_rows, native_tokens, estimate, variance, details = 0, 0, 0, 0, []
    for path in files:
        pf = pq.ParquetFile(path)
        file_rows = pf.metadata.num_rows
        # Read only the small integer column for the exact source-token sum.
        source_tokens = sum(sum(batch.column(0).to_pylist()) for batch in pf.iter_batches(columns=['token_count']))
        chosen = sorted(rng.sample(range(pf.num_row_groups), min(groups, pf.num_row_groups)))
        lengths = []
        for group in chosen:
            texts = pf.read_row_group(group, columns=['text']).column(0).to_pylist()
            selected = rng.sample(texts, min(rows, len(texts)))
            lengths.extend(len(item.ids) for item in tokenizer.encode_batch(selected, add_special_tokens=False))
        average, se = mean_se(lengths)
        # Groups are close to equal sized in this corpus. Use variability between
        # group means for a conservative cluster-aware sampling standard error.
        group_means = [statistics.mean(lengths[i:i + rows]) for i in range(0, len(lengths), rows)]
        if len(group_means) > 1:
            se = max(se, mean_se(group_means)[1])
        value = file_rows * average
        estimate += value
        variance += (file_rows * se) ** 2
        total_rows += file_rows
        native_tokens += source_tokens
        details.append({'file': str(path), 'rows': file_rows, 'row_groups': pf.num_row_groups,
                        'native_token_count': source_tokens, 'sampled_rows': len(lengths),
                        'estimated_local_text_tokens': value, 'standard_error': file_rows * se})
        print(f'PARQUET {path.name}: {file_rows:,} rows, {source_tokens:,} source tokens', flush=True)
    return {'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
            'exact_rows': total_rows, 'exact_native_token_count': native_tokens,
            'estimated_local_text_tokens': estimate,
            'sampling_95pct_interval': [estimate - 1.96 * math.sqrt(variance), estimate + 1.96 * math.sqrt(variance)],
            'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/zwb/ai/data/text'))
    parser.add_argument('--tokenizer', type=Path, default=Path('model/tokenizer.json'))
    parser.add_argument('--jsonl-samples-per-file', type=int, default=32)
    parser.add_argument('--parquet-groups-per-file', type=int, default=4)
    parser.add_argument('--parquet-rows-per-group', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('pretrain_corpus_audit.json'))
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    rng = random.Random(args.seed)
    report = {'seed': args.seed, 'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              'tokenizer_vocab_size': tokenizer.get_vocab_size(),
              'limitations': ['Estimated local-tokenizer totals, not a full tokenization.',
                              'No full-corpus deduplication; sampled duplicates cannot establish a corpus-wide duplicate rate.',
                              'Historical training tokens cannot be reconstructed without run settings or counters.'],
              'fineweb_edu': audit_parquet(sorted((args.data_root / 'fineweb-edu-10BT/sample/10BT').glob('*.parquet')), tokenizer, rng, args.parquet_groups_per_file, args.parquet_rows_per_group),
              'skypile': audit_jsonl(sorted((args.data_root / 'skypile-subset/data').glob('*.jsonl')), tokenizer, rng, args.jsonl_samples_per_file)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
