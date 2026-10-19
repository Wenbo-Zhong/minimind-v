"""Coverage, loss-mask and exact resume tests for the text-only baseline."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from dataset.build_calculator_sft import convert, make_turns
from dataset.text_data import DocumentStream, TextMixture, encode_sft
from eval_calculator import calculate

ROOT = Path(__file__).resolve().parents[1]


class TextDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, sources, **kwargs):
        path = self.root / ('config' + str(len(list(self.root.glob('config*')))) + '.json')
        path.write_text(json.dumps({'sources': sources, 'shuffle_buffer': 3, 'seed': 42, **kwargs}))
        return path

    def test_parquet_units_cover_each_row_once(self):
        for i in range(2):
            pq.write_table(pa.table({'text': [f'{i}/{j}' for j in range(11)]}), self.root / f'{i}.parquet', row_group_size=4)
        stream = DocumentStream({'name': 'test', 'files': str(self.root / '*.parquet')}, 8, 2)
        rows = [stream._raw()['text'] for _ in range(22)]
        self.assertEqual(len(set(rows)), 22)
        self.assertEqual(set(rows), {f'{i}/{j}' for i in range(2) for j in range(11)})

    def test_state_restores_buffers_packing_and_schedule(self):
        for i in range(2):
            file = self.root / f'{i}.jsonl'
            file.write_text(''.join(json.dumps({'text': f'document {j} from {i}. ' * 5}) + '\n' for j in range(15)))
        path = self.config([{'name': str(i), 'weight': i + 1, 'files': str(self.root / f'{i}.jsonl')} for i in range(2)])
        a = TextMixture(path, self.tokenizer, 16, 'pretrain')
        for _ in range(7): a.next()
        state = a.state_dict()
        expected = [a.next() for _ in range(25)]
        b = TextMixture(path, self.tokenizer, 16, 'pretrain')
        b.load_state_dict(state)
        self.assertEqual([b.next() for _ in range(25)], expected)
        self.assertEqual(a.summary(), b.summary())

    def test_packed_loss_token_ratios(self):
        file = self.root / 'rows.jsonl'
        file.write_text(json.dumps({'text': 'abcdefgh ' * 100}) + '\n')
        path = self.config([{'name': str(i), 'weight': w, 'files': str(file)} for i, w in enumerate((45, 35, 10, 10))])
        data = TextMixture(path, self.tokenizer, 16, 'pretrain')
        for _ in range(100): data.next()
        self.assertEqual(data.loss_tokens, [w * 15 for w in (45, 35, 10, 10)])

    def test_tool_result_and_user_tokens_are_masked(self):
        calls = [('91 - 36', '55')]
        turns = make_turns('91减36', calls, '55', 'zh')
        ids, labels = encode_sft(self.tokenizer, turns, 1024)
        supervised = self.tokenizer.decode([x for x in labels if x != -100])
        self.assertIn('<calculator>91 - 36</calculator>', supervised)
        self.assertIn('最终答案：55', supervised)
        self.assertNotIn('Calculator result', supervised)
        self.assertNotIn('91减36', supervised)
        self.assertEqual(supervised.count('<|im_end|>'), 2)
        self.assertIsNone(encode_sft(self.tokenizer, turns, 16))

    def test_sft_deficit_scheduler_targets_loss_tokens(self):
        sources = []
        for i, (weight, repeat) in enumerate([(2, 2), (1, 12)]):
            file = self.root / f'{i}.jsonl'
            file.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'test'},
                {'role': 'assistant', 'content': 'answer ' * repeat}]}) + '\n')
            sources.append({'name': str(i), 'weight': weight, 'files': str(file)})
        data = TextMixture(self.config(sources), self.tokenizer, 256, 'sft')
        for _ in range(100): data.next()
        self.assertAlmostEqual(data.loss_tokens[0] / sum(data.loss_tokens), 2 / 3, delta=0.04)
        self.assertGreater(data.examples[0], data.examples[1])

    def test_calc_conversion_verifies_each_step(self):
        row = {'chain': '<gadget id="calculator">5 ** 2</gadget><output>25</output><result>25</result>', 'result': '25'}
        calls, answer = convert(row)
        self.assertEqual(calls, [('5 ** 2', '25')])
        self.assertEqual(answer, '25')
        row['chain'] = row['chain'].replace('<output>25', '<output>26')
        with self.assertRaisesRegex(ValueError, 'output mismatch'): convert(row)
        self.assertEqual(calculate('2 ** -3'), '0.125')

    def test_empty_file_fails(self):
        file = self.root / 'empty.jsonl'
        file.touch()
        stream = DocumentStream({'name': 'empty', 'files': str(file)}, 42, 2)
        with self.assertRaisesRegex(ValueError, 'empty data'): stream.next()

    def test_exclusion_detects_instruction_prefix(self):
        heldout = self.root / 'heldout.jsonl'
        heldout.write_text(json.dumps({'question': 'How much is 20 plus 7?'}) + '\n')
        file = self.root / 'rows.jsonl'
        file.write_text(''.join(json.dumps({'messages': [
            {'role': 'user', 'content': q}, {'role': 'assistant', 'content': '27'}]}) + '\n'
            for q in ['Solve this problem.\nHow much is 20 plus 7?', 'A different question']))
        data = TextMixture(self.config([{'name': 'x', 'weight': 1, 'files': str(file)}],
                                      exclude_questions=[str(heldout)]), self.tokenizer, 128, 'sft')
        for _ in range(10):
            ids, _, _ = data.next()
            self.assertNotIn('20 plus 7', self.tokenizer.decode(ids))
        self.assertGreater(data.rejected[0], 0)

    def test_train_validation_split_is_disjoint(self):
        file = self.root / 'rows.jsonl'
        file.write_text(''.join(json.dumps({'text': 'document ' + str(j)}) + '\n' for j in range(300)))
        train = TextMixture(self.config([{'name': 'x', 'files': str(file), 'weight': 1, 'split': 'train'}]), self.tokenizer, 16, 'pretrain')
        val = TextMixture(self.config([{'name': 'x', 'files': str(file), 'weight': 1, 'split': 'validation'}]), self.tokenizer, 16, 'pretrain')
        train_rows = {tuple(train._row(0)) for _ in range(600)}
        val_rows = {tuple(val._row(0)) for _ in range(20)}
        self.assertFalse(train_rows & val_rows)

    def test_cpu_training_resume_matches_uninterrupted(self):
        file = self.root / 'rows.jsonl'
        file.write_text(''.join(json.dumps({'text': 'sample number ' + str(i)}) + '\n' for i in range(12)))
        data = self.config([{'name': 'x', 'weight': 1, 'files': str(file)}])
        model = self.root / 'model.json'
        model.write_text(json.dumps({'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                                     'num_key_value_heads': 1, 'dropout': 0.1}))
        full, resumed = self.root / 'full', self.root / 'resumed'
        base = [sys.executable, str(ROOT / 'trainer/train_text.py'), '--stage', 'pretrain',
                '--data_config', str(data), '--model_config', str(model), '--device', 'cpu',
                '--dtype', 'float32', '--threads', '2', '--max_steps', '4', '--max_seq_len', '12',
                '--accumulation_steps', '2', '--log_interval', '1']
        def run(extra):
            result = subprocess.run(base + extra, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run(['--output_dir', str(full)])
        run(['--output_dir', str(resumed), '--stop_after', '2'])
        run(['--output_dir', str(resumed), '--resume', str(resumed / 'last.pt')])
        a = torch.load(full / 'last.pt', weights_only=False, map_location='cpu')
        b = torch.load(resumed / 'last.pt', weights_only=False, map_location='cpu')
        self.assertEqual(a['step'], b['step'])
        self.assertEqual(a['data_state'], b['data_state'])
        for name in a['model']:
            self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
        self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))


if __name__ == '__main__':
    unittest.main()
