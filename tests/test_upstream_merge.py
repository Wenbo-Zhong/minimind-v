"""Regression coverage for upstream fixes combined with local trainer changes."""
from contextlib import nullcontext
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from model.model_minimind import MiniMindConfig, MOEFeedForward


class UpstreamMergeTests(unittest.TestCase):
    def test_normalized_top1_router_receives_task_gradient(self):
        torch.manual_seed(7)
        config = MiniMindConfig(hidden_size=8, num_hidden_layers=1, use_moe=True,
                               num_experts=4, num_experts_per_tok=1,
                               moe_intermediate_size=16, router_aux_loss_coef=0.0,
                               norm_topk_prob=True)
        module = MOEFeedForward(config)
        inputs = torch.randn(2, 3, 8, requires_grad=True)
        module(inputs).square().sum().backward()
        self.assertIsNotNone(module.gate.weight.grad)
        self.assertGreater(module.gate.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(module.gate.weight.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0)

    def test_partial_final_update_is_in_saved_checkpoint(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, ids, **kwargs):
                return SimpleNamespace(loss=self.weight * ids.float().mean(),
                                       aux_loss=self.weight.new_zeros(()))
        class Scaler:
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        for name in ['trainer.train_pretrain_vlm', 'trainer.train_sft_vlm']:
            with self.subTest(trainer=name), tempfile.TemporaryDirectory() as directory:
                trainer = importlib.import_module(name)
                model = Model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                args = SimpleNamespace(device='cpu', accumulation_steps=4, epochs=1,
                                       learning_rate=0.1, grad_clip=10, log_interval=3,
                                       save_interval=99, save_dir=directory, save_weight='test')
                saved = []
                def checkpoint(*args, **kwargs):
                    saved.append(float(kwargs['model'].weight.detach()))
                values = {'args': args, 'model': model, 'optimizer': optimizer,
                          'scaler': Scaler(), 'autocast_ctx': nullcontext(),
                          'vlm_config': SimpleNamespace(use_moe=False, hidden_size=8),
                          'Logger': lambda *_: None, 'get_lr': lambda *_: 0.1,
                          'vlm_checkpoint': checkpoint, '_save_stage_checkpoint': lambda *_: None}
                batches = [(torch.ones(1, 2, dtype=torch.long),
                            torch.ones(1, 2, dtype=torch.long), torch.zeros(1)) for _ in range(3)]
                with patch.multiple(trainer, create=True, **values), \
                     patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                    trainer.train_epoch(0, batches, iters=3)
                self.assertEqual(step.call_count, 1)
                self.assertEqual(saved, [float(model.weight.detach())])
                self.assertLess(saved[0], 1.0)
                weights = torch.load(Path(directory) / 'test_8.pth', weights_only=True)
                self.assertAlmostEqual(float(weights['weight']), saved[0], places=3)


if __name__ == '__main__':
    unittest.main()
