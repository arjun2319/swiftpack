\# Dense Triangular Solver — Optimization Sequence



I have applied the same 6-step optimization sequence from matmul.md to a dense lower-triangular solver (TRSV): solve Lx = b where L is lower triangular, using forward substitution.



\## Problem to be solved



Given a lower triangular matrix L and vector b, find x such that Lx = b.



Forward substitution (row-oriented):

&#x20; x\[i] = (b\[i] - sum\_{j<i} L\[i,j] \* x\[j]) / L\[i,i]



Key difference from GEMM: x\[i] depends on x\[0]...x\[i-1], so rows can't be computed independently. This data dependency makes TRSV memory-bound rather than compute-bound.



\## Matrix Generation



A random lower triangular matrix from N(0,1) is ill-conditioned. I enforced diagonal dominance:

&#x20; np.fill\_diagonal(L, np.abs(np.diag(L)) + n)



This brings the condition number from \~10^17 down to \~1.09, ensuring numerical stability.



\## Results (n = 512)



| Step | Implementation   | Time (s)  | Speedup  | Correct |

|-----:|:----------------|----------:|---------:|:-------:|

|    0 | python\_naive    | 0.048614  |   1.00x  | PASS    |

|    1 | numba\_jit       | 0.000101  | 483.72x  | PASS    |

|    2 | loop\_reorder    | 0.000267  | 181.94x  | PASS    |

|    3 | fastmath        | 0.000349  | 139.17x  | PASS    |

|    4 | tiled\_2d        | 0.000146  | 333.89x  | PASS    |

|    5 | numpy\_blas      | 0.022049  |   2.20x  | PASS    |



\## Step-by-Step procedure



Step 0 — Naive Python: Pure Python nested loops. Baseline: 48.6 ms.



Step 1 — Numba JIT (484x): @njit compiles the identical loop structure to native machine code. Same algorithm, \~484x faster. Python interpreter overhead is the dominant cost.



Step 2 — Loop Reorder (182x): Switches to column-oriented traversal. Column j of L is accessed sequentially for better spatial locality.



Step 3 — Fastmath (139x): @njit(fastmath=True) allows reassociation and approximate FP optimizations. Small effect on this access pattern.



Step 4 — 2D Cache-Blocked Tiling (334x): Processes a 32x32 block of L at a time. The diagonal tile is solved first as a small triangular system within the block, then all rows below are updated together. This keeps both the L submatrix block and the relevant x values in cache during the update, reducing memory traffic compared to column-by-column traversal.



Step 5 — NumPy BLAS (2.2x): np.linalg.solve calls LAPACK dtrtrs internally. Slower than Numba at n=512 — function call overhead dominates at this size.



\## Observations



1\. Peak speedup: 484x from Numba JIT alone

2\. 2D tiling gives \~2x improvement over simple loop reorder due to cache blocking

3\. TRSV is memory-bound: the sequential data dependency prevents the parallelism that gives GEMM large gains

4\. NumPy/BLAS is not the ceiling for TRSV at small n — Numba beats it



\## Files



1\. benchmarks/trsv\_numba\_bench.py — benchmark implementation

2\. doc/trsv.md — this document

