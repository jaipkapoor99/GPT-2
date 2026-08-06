"""Full-validation metric tests."""

import torch
import torch.nn.functional as F

from scripts.validate import sequence_cross_entropy


def test_sequence_cross_entropy_matches_per_sequence_reference():
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))

    losses = sequence_cross_entropy(logits, targets)
    expected = torch.stack(
        [
            F.cross_entropy(logits[index], targets[index])
            for index in range(len(targets))
        ]
    )

    torch.testing.assert_close(losses, expected)
