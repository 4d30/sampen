# GPU-Accelerated Sample Entropy in Python

This repository provides a fast, GPU-accelerated implementation of **Sample Entropy (SampEn)** for time series analysis using Python, Numba, and CUDA.

Sample Entropy is a widely-used statistical measure that quantifies the complexity or regularity of a time series. This implementation leverages GPU parallelism to efficiently compute SampEn for large datasets. It has been benchmarked as 100x faster than CPU-based implementations.

## Features

-  Efficient GPU-based kernel using [Numba CUDA](https://numba.pydata.org/numba-doc/latest/cuda/index.html)
-  Chebyshev distance for sequence similarity
-  Dynamic chunk processing to handle large time series
-  Graceful handling of edge cases (zero matches, infinite entropy)

## Requirements

- Python 3.8+
- NumPy
- Numba
- CUDA-enabled GPU and NVIDIA drivers

Install dependencies via:

```bash
git clone https://github.com/4d30/sampen.git
cd sampen
pip install -e .
```




## Usage
```python
from sampen import sampen
import numpy as np

data = np.random.rand(10000)  # your time series data
m = 2                         # template length
r = 0.2                       # similarity threshold

entropy = sampen_gpu(data, m, r)
print("Sample Entropy:", entropy)
```

