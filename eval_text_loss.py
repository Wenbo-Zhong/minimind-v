#!/usr/bin/env python3
"""Evaluate the new pure-text checkpoint on per-source validation loss."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from dataset.text_data import TextMixture
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_text import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_config', required=True)
    parser.add_argument('--tokenizer_path', default=str(Path(__file__).resolve().parent / 'model'))
    parser.add_argument('--batches_per_source', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.batches_per_source < 1:
        parser.error('batches_per_source must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    metadata = checkpoint['metadata']
    for name, expected in metadata['tokenizer_sha256'].items():
        if file_hash(Path(args.tokenizer_path) / name) != expected:
            raise ValueError('Tokenizer fingerprint mismatch: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(**metadata['model_config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.eval().to(args.device)
    data = TextMixture(args.data_config, tokenizer, args.max_seq_len, metadata['stage'])
    # Refuse training subsets in this validation-only entry point.
    for source in data.sources:
        if source.get('split') != 'validation' and '/validation.jsonl' not in source['files']:
            raise ValueError('Expected an explicit validation source: ' + source['name'])
    results = {}
    with torch.inference_mode():
        for index, source in enumerate(data.sources):
            tokens, weighted_loss = 0, 0.0
            for _ in range(args.batches_per_source):
                if data.stage == 'sft':
                    ids, labels = data._row(index)
                else:
                    while len(data.pending[index]) < args.max_seq_len:
                        data.pending[index].extend(data._row(index))
                    ids = data.pending[index][:args.max_seq_len]
                    del data.pending[index][:args.max_seq_len]
                    labels = ids
                ids, labels = torch.tensor([ids], device=args.device), torch.tensor([labels], device=args.device)
                count = int(labels[:, 1:].ne(-100).sum())
                context = torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
                with context:
                    loss = model(ids, labels=labels).loss
                tokens += count
                weighted_loss += float(loss) * count
            results[source['name']] = {'loss': weighted_loss / tokens, 'loss_tokens': tokens}
            print(source['name'], results[source['name']], flush=True)
    payload = {'checkpoint': args.checkpoint, 'data_config': data.config, 'max_seq_len': args.max_seq_len,
               'batches_per_source': args.batches_per_source, 'results': results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
