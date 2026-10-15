import numpy as np
import time
from numba import njit, prange

def make_problem(n):
    rng = np.random.default_rng(42)
    L = np.tril(rng.standard_normal((n, n)))
    np.fill_diagonal(L, np.abs(np.diag(L)) + n)
    b = rng.standard_normal(n)
    return L.astype(np.float64), b.astype(np.float64)

def bench(func, L, b, runs=3):
    func(L, b.copy())  # warmup
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        x = func(L, b.copy())
        times.append(time.perf_counter() - t0)
    return min(times), x

# Step 0: Naive Python
def trsv_0_python(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x

# Step 1: Numba JIT
@njit
def trsv_1_numba(L, b):
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x

# Step 2: Loop reorder (column-oriented)
@njit
def trsv_2_reorder(L, b):
    n = L.shape[0]
    x = b.copy()
    for j in range(n):
        x[j] /= L[j, j]
        for i in range(j + 1, n):
            x[i] -= L[i, j] * x[j]
    return x

# Step 3: Fastmath
@njit(fastmath=True)
def trsv_3_fastmath(L, b):
    n = L.shape[0]
    x = b.copy()
    for j in range(n):
        x[j] /= L[j, j]
        for i in range(j + 1, n):
            x[i] -= L[i, j] * x[j]
    return x

# Step 4: Loop tiling (cache blocking)
@njit(fastmath=True)
def trsv_4_tiled(L, b):
    n = L.shape[0]
    x = b.copy()
    TILE = 32
    for j in range(n):
        x[j] /= L[j, j]
        i = j + 1
        while i < n:
            i_end = min(i + TILE, n)
            for ii in range(i, i_end):
                x[ii] -= L[ii, j] * x[j]
            i = i_end
    return x

# Step 5: NumPy (BLAS reference)
def trsv_5_numpy(L, b):
    return np.linalg.solve(L, b)

if __name__ == "__main__":
    n = 512
    L, b = make_problem(n)
    x_ref = np.linalg.solve(L, b.copy())
    print(f"\nDense Triangular Solver  n={n}\n")
    print(f"{'Implementation':<28} {'Time(s)':<12} {'Speedup':<10} {'Correct'}")
    print("-" * 62)
    t_base = None
    steps = [
        ("0_python_naive",   trsv_0_python),
        ("1_numba_jit",      trsv_1_numba),
        ("2_loop_reorder",   trsv_2_reorder),
        ("3_fastmath",       trsv_3_fastmath),
        ("4_tiled",          trsv_4_tiled),
        ("5_numpy_blas",     trsv_5_numpy),
    ]
    for name, fn in steps:
        t, x = bench(fn, L, b)
        if t_base is None:
            t_base = t
        diff = float(np.max(np.abs(x - x_ref)))
        ok = "PASS" if diff < 1e-6 else f"FAIL diff={diff:.2e}"
        print(f"{name:<28} {t:<12.6f} {t_base/t:<9.2f}x  {ok}")