import unittest

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.model import CompactCNN
from src.preprocess import preprocess_sketch


class CoreSmokeTests(unittest.TestCase):
    def test_model_output_shape(self):
        model = CompactCNN(num_classes=10)
        output = model(torch.zeros(2, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_preprocess_shape_range_and_blank(self):
        image = Image.new("RGBA", (160, 160), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.line((35, 80, 125, 80), fill=(0, 0, 0, 255), width=7)
        arr = preprocess_sketch(image)
        self.assertEqual(arr.shape, (1, 28, 28))
        self.assertTrue(np.isfinite(arr).all())
        self.assertGreaterEqual(float(arr.min()), 0.0)
        self.assertLessEqual(float(arr.max()), 1.0)
        self.assertLess(float(arr.min()), 1.0)

    def test_blank_canvas_returns_white_bitmap(self):
        image = Image.new("RGB", (120, 120), "white")
        arr = preprocess_sketch(image)
        self.assertEqual(arr.shape, (1, 28, 28))
        self.assertAlmostEqual(float(arr.mean()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
