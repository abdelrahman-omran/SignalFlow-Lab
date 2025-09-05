import unittest
from processing.basic_ops import (
    add,
    subtract,
    multiply,
)

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


if __name__ == "__main__":
    unittest.main()
