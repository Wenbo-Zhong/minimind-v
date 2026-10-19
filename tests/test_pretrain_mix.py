"""CPU checks for token-weighted mixing and packed instruction sources."""
import collections
import itertools
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from dataset.lm_dataset import LocalMultiSourceVLMDataset
from model.model_vlm import MiniMindVLM


class TinyTokenizer:
    eos_token_id = 256
    pad_token_id = 257

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(text.encode('utf-8')))

    def apply_chat_template(self, conversations, **kwargs):
        return json.dumps(conversations, ensure_ascii=False)


class PretrainMixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'rows.jsonl'
        self.source.write_text(json.dumps({'text': 'a' * 100}) + '\n')
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'black').save(buffer, format='JPEG')
        self.black_image = buffer.getvalue()

    def dataset(self, weights=(45, 35, 10, 10), **overrides):
        config = {'mixing_unit': 'tokens', 'shuffle_sources': True, 'seed': 42,
                  'sources': [{'name': str(i), 'stages': ['pretrain'], 'format': 'json',
                               'files': str(self.source), 'packing': True, 'weight': weight}
                              for i, weight in enumerate(weights)]}
        config.update(overrides)
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        return LocalMultiSourceVLMDataset(path, TinyTokenizer(), max_length=16, stage='pretrain')

    def tagged_samples(self, source, *args):
        yield [{'role': 'raw_ids', 'content': [int(source['name'])] * 16}], self.black_image

    def test_effective_token_shares_match_each_complete_cycle(self):
        dataset = self.dataset()
        with patch.object(dataset, '_samples', side_effect=self.tagged_samples), \
                patch.object(MiniMindVLM, 'image2tensor', return_value=torch.zeros(1)):
            batches = list(itertools.islice(iter(dataset), 200))
        for start in (0, 100):
            totals = collections.Counter()
            for ids, labels, _ in batches[start:start + 100]:
                # Loss predicts labels[1:], not labels[0].
                totals[int(ids[0])] += int(labels[1:].ne(-100).sum())
            self.assertEqual(totals, {0: 45 * 15, 1: 35 * 15, 2: 10 * 15, 3: 10 * 15})
        self.assertGreater(len(set(int(item[0][0]) for item in batches[:10])), 1)

    def test_conversations_are_packed_and_prompt_tokens_are_supervised(self):
        turns = [{'from': 'human', 'value': 'Write code'}, {'from': 'gpt', 'value': 'print(1)'}]
        self.source.write_text(json.dumps({'conversations': turns}) + '\n')
        dataset = self.dataset(weights=(1,))
        samples = list(dataset._samples(dataset.sources[0]))
        expected = list(json.dumps([{'role': 'user', 'content': 'Write code'},
                                    {'role': 'assistant', 'content': 'print(1)'}]).encode())
        expected.append(256)
        actual = [token for sample, _ in samples for token in sample[0]['content']]
        self.assertEqual(actual, expected[:len(expected) // 16 * 16])
        with patch.object(MiniMindVLM, 'image2tensor', return_value=torch.zeros(1)):
            ids, labels, _ = dataset._encode(*samples[0])
        self.assertTrue(torch.equal(ids, labels))

    def test_missing_source_fails_before_iteration(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()

    def test_empty_source_fails_instead_of_renormalizing(self):
        self.source.write_text('')
        with self.assertRaisesRegex(ValueError, '没有可用 packed'):
            next(iter(self.dataset()))

    def test_token_mix_requires_packing(self):
        sources = [{'name': 'invalid', 'files': str(self.source), 'packing': False}]
        with self.assertRaisesRegex(ValueError, '开启 packing'):
            self.dataset(sources=sources)

    def test_legacy_sample_mode_keeps_weighted_order(self):
        dataset = self.dataset(weights=(2, 1), mixing_unit='samples', shuffle_sources=False)
        with patch.object(dataset, '_samples', side_effect=self.tagged_samples), \
                patch.object(MiniMindVLM, 'image2tensor', return_value=torch.zeros(1)):
            batches = list(itertools.islice(iter(dataset), 6))
        self.assertEqual([int(batch[0][0]) for batch in batches], [0, 0, 1, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
