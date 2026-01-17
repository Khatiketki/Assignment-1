# matrix_mul_complete.py
# Single-thread + multi-process "multi-threaded style" + tests + benchmark (speedup)
# Run:
#   python matrix_mul_complete.py test
#   python matrix_mul_complete.py bench
#   python matrix_mul_complete.py bench 2048 2048 2048 2

import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count


def matmul_single(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Single-threaded (pure Python loops). Correct but slow for large matrices."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast baseline using optimized BLAS (single call)."""
    return A @ B


def _worker_rows(args):
    A, B, row_start, row_end = args
    return A[row_start:row_end, :] @ B


def matmul_parallel(A: np.ndarray, B: np.ndarray, workers: int) -> np.ndarray:
    """
    Parallel matrix multiplication by splitting output rows across processes.
    Safe: each process computes disjoint row blocks, then we stack results.
    """
    M = A.shape[0]
    workers = max(1, min(workers, cpu_count(), M))

    base = M // workers
    rem = M % workers

    tasks = []
    r = 0
    for w in range(workers):
        take = base + (1 if w < rem else 0)
        tasks.append((A, B, r, r + take))
        r += take

    with Pool(processes=workers) as pool:
        blocks = pool.map(_worker_rows, tasks)

    return np.vstack(blocks)


def run_tests():
    cases = [
        (1, 1, 1),
        (1, 1, 5),
        (2, 1, 3),
        (2, 2, 2),
        (3, 4, 2),
        (4, 3, 5),
        (7, 8, 9),
        (16, 16, 16),
        (31, 17, 29),
    ]
    rng = np.random.default_rng(123)

    for (M, K, N) in cases:
        A = rng.normal(size=(M, K))
        B = rng.normal(size=(K, N))

        C_ref = matmul_numpy(A, B)
        C_single = matmul_single(A, B)
        if not np.allclose(C_ref, C_single, rtol=1e-9, atol=1e-9):
            raise AssertionError(f"Mismatch in single-thread: M={M} K={K} N={N}")

        for w in [1, 2, 4, 8]:
            C_par = matmul_parallel(A, B, w)
            if not np.allclose(C_ref, C_par, rtol=1e-9, atol=1e-9):
                raise AssertionError(f"Mismatch in parallel: M={M} K={K} N={N} workers={w}")

    print("All tests passed ✅")


def bench(M=1024, K=1024, N=1024, iters=3):
    rng = np.random.default_rng(42)
    A = rng.normal(size=(M, K))
    B = rng.normal(size=(K, N))

    # Baseline: NumPy (already highly optimized)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = matmul_numpy(A, B)
    t_numpy = (time.perf_counter() - t0) / iters
    print(f"Benchmark: A={M}x{K}, B={K}x{N}, iters={iters}")
    print(f"NumPy        time={t_numpy:.6f}s  speedup=1.00x (baseline)")

    # Parallel (processes). Often slower than NumPy BLAS on a single machine due to overhead,
    # but included to match the assignment-style "thread counts" experiment.
    for w in [1, 4, 16, 32, 64, 128]:
        w_eff = max(1, min(w, cpu_count(), M))
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = matmul_parallel(A, B, w_eff)
        t_par = (time.perf_counter() - t0) / iters
        print(f"Parallel w={w:3d} (eff {w_eff:3d}) time={t_par:.6f}s  speedup={t_numpy/t_par:.2f}x vs NumPy")


if __name__ == "__main__":
    # Usage:
    #   python matrix_mul_complete.py test
    #   python matrix_mul_complete.py bench
    #   python matrix_mul_complete.py bench M K N iters
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        run_tests()
    else:
        if len(sys.argv) >= 6 and sys.argv[1] == "bench":
            M = int(sys.argv[2]); K = int(sys.argv[3]); N = int(sys.argv[4]); iters = int(sys.argv[5])
            bench(M, K, N, iters)
        else:
            bench()
