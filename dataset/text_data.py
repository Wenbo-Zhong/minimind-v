"""Stateful single-process text streams: shuffled files/row groups and document buffers.

The initial trainer intentionally supports one GPU and no DataLoader workers, so
checkpointed cursors include every consumed document and no prefetched work.
"""
import copy
import glob
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pyarrow.parquet as pq
import torch


def question_key(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', str(text)).lower())


def conversations(row):
    turns = row.get('conversations', row.get('messages', []))
    if isinstance(turns, str):
        turns = json.loads(turns)
    return [{'role': {'human': 'user', 'gpt': 'assistant'}.get(t.get('role', t.get('from')), t.get('role', t.get('from'))),
             'content': str(t.get('content', t.get('value', '')))} for t in (turns or [])]


def load_exclusions(paths):
    keys = set()
    for pattern in paths:
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        for path in files:
            if path.endswith('.parquet'):
                reader = (r for b in pq.ParquetFile(path).iter_batches(batch_size=1024) for r in b.to_pylist())
            else:
                reader = (json.loads(line) for line in Path(path).open() if line.strip())
            for row in reader:
                for field in ('question', 'question_chinese'):
                    if row.get(field):
                        keys.add(question_key(row[field]))
    return keys


def encode_sft(tokenizer, turns, max_length):
    """Reject whole overlong chats; supervise only assistant body and ChatML EOS."""
    if (not turns or turns[-1]['role'] != 'assistant'
            or any(t['role'] not in ('system', 'user', 'assistant') for t in turns)
            or any('<|im_start|>' in t['content'] or '<|im_end|>' in t['content'] for t in turns)):
        return None
    ids = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=False)
    if len(ids) > max_length:
        return None
    header = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    labels = [-100] * len(ids)
    index = 0
    while index < len(ids):
        if ids[index:index + len(header)] == header:
            start = index + len(header)
            end = next((j for j in range(start, len(ids)) if ids[j] == eos), None)
            if end is None:
                return None
            labels[start:end + 1] = ids[start:end + 1]
            index = end + 1
        else:
            index += 1
    if not any(x != -100 for x in labels[1:]):
        return None
    return ids, labels


class DocumentStream:
    def __init__(self, source, seed, buffer_size):
        self.source = source
        patterns = source['files']
        patterns = [patterns] if isinstance(patterns, str) else patterns
        files = sorted({p for pattern in patterns for p in glob.glob(pattern)})
        if not files:
            raise FileNotFoundError(source['files'])
        self.manifest = [(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files]
        self.units = []
        for path in files:
            if path.endswith('.parquet'):
                self.units.extend((path, group) for group in range(pq.ParquetFile(path).num_row_groups))
            elif path.endswith('.jsonl'):
                self.units.append((path, None))
            else:
                raise ValueError('Text stream requires parquet or JSONL: ' + path)
        self.rng = random.Random(seed)
        self.buffer_size = buffer_size
        self.order = list(range(len(self.units)))
        self.rng.shuffle(self.order)
        self.unit_index = 0
        self.offset = 0  # byte offset for JSONL, row offset for Parquet
        self.cycle = 0
        self.buffer = []
        self._reader = None
        self.rows_read = 0

    def close(self):
        if self._reader is not None and hasattr(self._reader, "close"):
            self._reader.close()
        self._reader = None

    def __del__(self):
        if hasattr(self, "_reader"):
            self.close()

    def _open(self):
        path, group = self.units[self.order[self.unit_index]]
        if group is None:
            handle = open(path, 'rb')
            handle.seek(self.offset)
            self._reader = handle
        else:
            # Bounded Arrow batches, including for Infinity's 100k-row groups.
            batches = pq.ParquetFile(path).iter_batches(batch_size=1024, row_groups=[group])
            def rows():
                skip = self.offset
                for batch in batches:
                    if skip >= batch.num_rows:
                        skip -= batch.num_rows
                        continue
                    for row in batch.slice(skip).to_pylist():
                        yield row
                    skip = 0
            self._reader = rows()

    def _raw(self):
        while True:
            if self._reader is None:
                self._open()
            _, group = self.units[self.order[self.unit_index]]
            if group is None:
                line = self._reader.readline()
                if line:
                    self.offset = self._reader.tell()
                    if not line.strip():
                        continue
                    row = json.loads(line)
                else:
                    row = None
            else:
                row = next(self._reader, None)
                if row is not None:
                    self.offset += 1
            if row is not None:
                self.rows_read += 1
                return row
            if hasattr(self._reader, 'close'):
                self._reader.close()
            self._reader = None
            self.offset = 0
            self.unit_index += 1
            if self.unit_index == len(self.order):
                if self.rows_read == 0:
                    raise ValueError("empty data source: " + self.source["name"])
                self.cycle += 1
                self.unit_index = 0
                self.rng.shuffle(self.order)

    def next(self):
        if not self.buffer:
            self.buffer = [self._raw() for _ in range(self.buffer_size)]
        index = self.rng.randrange(len(self.buffer))
        row = self.buffer[index]
        self.buffer[index] = self._raw()
        return row

    def state_dict(self):
        return {key: copy.deepcopy(getattr(self, key)) for key in
                ('order', 'unit_index', 'offset', 'cycle', 'buffer', 'rows_read')} | {'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if self._reader is not None and hasattr(self._reader, 'close'):
            self._reader.close()
        self._reader = None
        for key in ('order', 'unit_index', 'offset', 'cycle', 'buffer', 'rows_read'):
            setattr(self, key, copy.deepcopy(state[key]))
        self.rng.setstate(state['rng'])


class TextMixture:
    def __init__(self, config_path, tokenizer, max_length, stage):
        self.config = json.loads(Path(config_path).read_text())
        self.sources = self.config['sources']
        self.stage, self.tokenizer, self.max_length = stage, tokenizer, max_length
        if max_length < 2 or stage not in ('pretrain', 'sft'):
            raise ValueError('invalid sequence length or stage')
        names = [s['name'] for s in self.sources]
        if len(set(names)) != len(names) or not names:
            raise ValueError('source names must be unique and nonempty')
        if any(type(s['weight']) is not int or s['weight'] <= 0 for s in self.sources):
            raise ValueError('source weights must be positive integers')
        self.rng = random.Random(self.config.get('seed', 42))
        size = self.config.get('shuffle_buffer', 256)
        if size < 1:
            raise ValueError('shuffle_buffer must be positive')
        self.streams = [DocumentStream(s, self.config.get('seed', 42) + i * 997, size)
                        for i, s in enumerate(self.sources)]
        self.exclusions = load_exclusions(self.config.get('exclude_questions', []))
        self.pending = [[] for _ in self.sources]
        self.loss_tokens = [0 for _ in self.sources]
        self.examples = [0 for _ in self.sources]
        self.rejected = [0 for _ in self.sources]
        self.schedule = []

    def _row(self, index):
        source = self.sources[index]
        for _ in range(100000):
            row = self.streams[index].next()
            if any(row.get(k) != v for k, v in source.get('filters', {}).items()):
                self.rejected[index] += 1
                continue
            turns = conversations(row)
            questions = [t['content'] for t in turns if t['role'] == 'user']
            questions += [row[k] for k in ('question', 'question_chinese', 'text', 'content') if isinstance(row.get(k), str)]
            # Also strip instruction lines prepended by the local GSM8K builder.
            candidates = {question_key(q) for q in questions}
            for question in questions:
                lines = question.splitlines()
                candidates.update(question_key('\n'.join(lines[i:])) for i in range(1, min(16, len(lines))))
            if candidates & self.exclusions:
                self.rejected[index] += 1
                continue
            # A document-level split remains identical after file order changes.
            identity = (row.get('text', row.get('content', ''))
                        or question_key(next((t['content'] for t in turns if t['role'] == 'user'), ''))
                        or json.dumps(turns, ensure_ascii=False))
            split = source.get('split')
            if split not in (None, 'train', 'validation'):
                raise ValueError('invalid source split: ' + str(split))
            if split:
                is_validation = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % 100 == 0
                if is_validation != (split == 'validation'):
                    self.rejected[index] += 1
                    continue
            if self.stage == 'sft':
                encoded = encode_sft(self.tokenizer, turns, self.max_length)
                if encoded is not None:
                    return encoded
            else:
                text = row.get('text', row.get('content', ''))
                if not text and turns:
                    text = self.tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
                if text and len(text) <= 200000:
                    return self.tokenizer(text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
            self.rejected[index] += 1
        raise ValueError(f'Source {source["name"]} produced no usable sample in 100000 attempts')

    def next(self):
        if self.stage == 'pretrain':
            if not self.schedule:
                self.schedule = [i for i, s in enumerate(self.sources) for _ in range(s['weight'])]
                self.rng.shuffle(self.schedule)
            index = self.schedule.pop()
            while len(self.pending[index]) < self.max_length:
                self.pending[index].extend(self._row(index))
            ids = self.pending[index][:self.max_length]
            del self.pending[index][:self.max_length]
            labels = ids.copy()
        else:
            # Deficit scheduling targets assistant loss tokens, rather than rows.
            indices = list(range(len(self.sources)))
            self.rng.shuffle(indices)
            index = min(indices, key=lambda i: self.loss_tokens[i] / self.sources[i]['weight'])
            ids, labels = self._row(index)
        effective = sum(x != -100 for x in labels[1:])
        self.loss_tokens[index] += effective
        self.examples[index] += 1
        return ids, labels, index

    def batch(self, batch_size):
        rows = [self.next() for _ in range(batch_size)]
        length = max(len(r[0]) for r in rows)
        ids = [r[0] + [self.tokenizer.pad_token_id] * (length - len(r[0])) for r in rows]
        labels = [r[1] + [-100] * (length - len(r[1])) for r in rows]
        # Right padding cannot affect any preceding causal target.
        return torch.tensor(ids), torch.tensor(labels)

    def state_dict(self):
        return {'streams': [s.state_dict() for s in self.streams], 'rng': self.rng.getstate(),
                **{k: copy.deepcopy(getattr(self, k)) for k in
                   ('pending', 'loss_tokens', 'examples', 'rejected', 'schedule')}}

    def load_state_dict(self, state):
        if len(state['streams']) != len(self.streams):
            raise ValueError('source count changed')
        for stream, saved in zip(self.streams, state['streams']):
            stream.load_state_dict(saved)
        self.rng.setstate(state['rng'])
        for key in ('pending', 'loss_tokens', 'examples', 'rejected', 'schedule'):
            setattr(self, key, copy.deepcopy(state[key]))

    def summary(self):
        return {source['name']: {'loss_tokens': self.loss_tokens[i], 'examples': self.examples[i],
                                'rejected': self.rejected[i], 'read_cycles': self.streams[i].cycle}
                for i, source in enumerate(self.sources)}
