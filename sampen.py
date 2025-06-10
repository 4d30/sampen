#!/usr/bin/env python

import sys
import operator as op

import statistics as sts
from array import array

import time
import numpy as np
from numba import cuda


@cuda.jit
def sampen_kern(data, m, r, counts_m_arr, counts_m_plus_1_arr):
    """
    Sample Entropy kernel

    Parameters
    ----------
    data : numpy.ndarray
        The input data
    m : int
        The length of the template
    r : int
        The radius
    counts_m : numpy.ndarray
        The number of similar sequences of length m
    counts_m_plus_1 : numpy.ndarray
        The number of similar sequences of length m+1
    """
    i = cuda.grid(1)

    N = len(data)

    if i < N - m:

        counts_m_arr[i] = 0
        counts_m_plus_1_arr[i] = 0
        count_m = 0
        count_m_plus_1 = 0

        # create template sequences
        template_m = data[i:i + m]
        template_m_plus_1 = data[i:i + m + 1]

        # loop through other points to count similar sequences
        for j in range(i + 1, N - m):
            # chebyshev distances
            dist_m = 0
            for k in range(m):
                dist_m = max(dist_m, abs(data[j + k] - template_m[k]))

            if dist_m < r:
                count_m += 1  # similar sequence found for length m

                # calculate distance for sequences of length m+1
                dist_m_plus_1 = 0
                for k in range(m + 1):
                    dist_m_plus_1 = max(dist_m_plus_1,
                                        abs(data[j + k]
                                            - template_m_plus_1[k]))
                if dist_m_plus_1 < r:
                    count_m_plus_1 += 1  # similar found for length m+1

        counts_m_arr[i] = count_m
        counts_m_plus_1_arr[i] = count_m_plus_1


def sampen_gpu(data, m, r):
    """
    Wrapper function to calculate sample entropy on the GPU.
    Parameters
    ----------
    data : array
        The data to calculate sample entropy on
    m : int
        The length of the template
    r : float
        The radius of the chebyshev distance
    """
    N = len(data)
    if N > np.iinfo(np.uint64).max:
        raise ValueError("N is too large for uint64")

    cuda.select_device(0)
    cuda.to_device(data)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    count_m = 0
    count_m_plus_1 = 0

    chunk_size = 100000
    num_chunks = (N - m) // chunk_size + 1
    data_chunks = np.array_split(data, num_chunks)
    count_m_chunk = cuda.device_array(chunk_size, dtype=np.uint64)
    count_m_plus_1_chunk = cuda.device_array(chunk_size, dtype=np.uint64)
    for i, data_chunk in enumerate(data_chunks):
        sampen_kern[blocks_per_grid, threads_per_block](data_chunk, m, r,
                                                        count_m_chunk,
                                                        count_m_plus_1_chunk)
        count_m_chunk_host = count_m_chunk.copy_to_host()
        count_m_chunk_host = count_m_chunk_host[:len(data_chunk) - m]
        count_m_plus_1_chunk_host = count_m_plus_1_chunk.copy_to_host()
        count_m_plus_1_chunk_host = count_m_plus_1_chunk_host[:len(data_chunk)
                                                              - m]
        count_m += np.sum(count_m_chunk_host)
        count_m_plus_1 += np.sum(count_m_plus_1_chunk_host)

    # final calc
    if count_m_plus_1 == 0 and count_m != 0:
        sampen = np.inf
    elif count_m == 0:
        sampen = np.nan
    else:
        sampen = -np.log(count_m_plus_1 / count_m)
    return sampen


if __name__ == "__main__":
    def main():
        m = 2         # embedding dimension
        r = 0.2       # tolerance threshold
        N = 10000     # length of the fake time series test data

        np.random.seed(0)
        data = np.random.rand(N).astype(np.float32)

        t_gpu = time.time()
        en_gpu = sampen_gpu(data, m, r)
        t_gpu = time.time() - t_gpu
        print("GPU-based Sample Entropy:", en_gpu)

    main()
