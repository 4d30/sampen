import unittest
import numpy as np
from sampen import sampen_gpu

class TestSampleEntropy(unittest.TestCase):

    def test_entropy_returns_float_or_nan(self):
        data = np.random.rand(1000)
        result = sampen_gpu(data, m=2, r=0.2)
        self.assertTrue(isinstance(result, float) or np.isnan(result))

    def test_entropy_nan_when_no_matches(self):
        data = np.zeros(100)  # identical points — likely no m+1 matches
        result = sampen_gpu(data, m=10, r=1e-12)
        print(f"Test result: {result}")
        # Check if result is NaN or inf
        self.assertTrue(np.isnan(result) or np.isinf(result))

    def test_entropy_inf_when_m_matches_but_not_m_plus_1(self):
        data = np.linspace(0, 1, 100)
        result = sampen_gpu(data, m=2, r=1e-12)
        self.assertTrue(np.isinf(result) or np.isnan(result))

if __name__ == '__main__':
    unittest.main()
