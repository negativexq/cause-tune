from __future__ import annotations

import unittest

import torch

from fineforge.training import (
    count_model_input_tokens,
    count_supervised_tokens_after_shift,
)


class TrainingTokenTests(unittest.TestCase):
    def test_counter_counts_shifted_targets_not_optimizer_batches(self) -> None:
        labels = torch.tensor(
            [
                [-100, -100, 10, 11, 12],
                [20, 21, -100, -100, -100],
            ]
        )
        attention_mask = torch.ones_like(labels)
        # The first row has three supervised targets after shifting. The
        # second row loses its first label because it is at sequence position 0.
        self.assertEqual(count_supervised_tokens_after_shift(labels), 4)
        self.assertEqual(count_model_input_tokens(attention_mask), 10)
