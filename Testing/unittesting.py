import unittest
from processing.basic_ops import (
    add,
    subtract,
    multiply,
)
from processing.signal_digitize.sampling import sampling
from processing.signal_digitize.quantization import quantization


class TestSignalOps(unittest.TestCase):

    def setUp(self):
        # Common test signals
        self.signal1 = ([0, 1, 2, 3], [1, 2, 3, 4])
        self.signal2 = ([0, 1, 2, 3], [10, 20, 30, 40])

    def test_add(self):
        indices, values = add(self.signal1, self.signal2)
        self.assertEqual(values, [11, 22, 33, 44])
        self.assertEqual(indices, [0, 1, 2, 3])

    def test_subtract(self):
        indices, values = subtract(self.signal2, self.signal1)
        self.assertEqual(values, [9, 18, 27, 36])

    def test_multiply(self):
        indices, values = multiply(self.signal1, self.signal2)
        self.assertEqual(values, [10, 40, 90, 160])

    def test_sample_signal(self):
        indices, values = sampling(self.signal1, rate=3)
        self.assertEqual(len(values), 2)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices, [0, 3])


    def test_quantize_signal(self):
        indices, values = quantization(self.signal1, levels=3)
        self.assertEqual(len(values), len(self.signal1[1]))
        # Check that values are correctly quantized
        self.assertTrue(all(v in [1, 2.5, 4] for v in values))



if __name__ == "__main__":
    unittest.main()
