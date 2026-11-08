import unittest

import torch
import torch.nn.functional as F

from vencoder.whisper.model import LayerNorm


class WhisperPrecisionTests(unittest.TestCase):
    def test_layer_norm_with_half_precision_weights(self):
        layer = LayerNorm(4).half()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
        actual = layer(x)
        expected = F.layer_norm(
            x.float(), (4,), layer.weight.float(), layer.bias.float(), layer.eps,
        ).half()
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
