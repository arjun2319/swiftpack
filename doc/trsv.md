\# Dense Triangular Solver — Optimization Sequence



I have applied the same 6-step optimization sequence from matmul.md to a dense lower-triangular solver (TRSV): solve Lx = b where L is lower triangular, using forward substitution.



\## Problem to be solved



Given a lower triangular matrix L and vector b, find x such that Lx = b.



Forward substitution (row-oriented):

&#x20; x\[i] = (b\[i] - sum\_{j<i} L\[i,j] \* x\[j]) / L\[i,i]



Key difference from GEMM: x\[i] depends on x\[0]...x\[i-1], so rows can't be computed independently. This data dependency makes TRSV memory-bound rather than compute-bound.



\## Matrix Generation



A random lower triangular matrix from N(0,1) is ill-conditioned. We will enforce diagonal dominance:

&#x20; np.fill\_diagonal(L, np.abs(np.diag(L)) + n)



This brings the condition number from \~10^17 down to \~1.09, while ensuring numerical stability.Below are the results 



\## Results (n = 512)



| Step | Implementation | Time (s)  | Speedup | Correct |

|------|----------------|-----------|---------|---------|

| 0    | python\_naive   | 0.046925  | 1.00x   | PASS    |

| 1    | numba\_jit      | 0.000086  | 546x    | PASS    |

| 2    | loop\_reorder   | 0.000259  | 181x    | PASS    |

| 3    | fastmath       | 0.000236  | 199x    | PASS    |

| 4    | tiled          | 0.000255  | 184x    | PASS    |

| 5    | numpy\_blas     | 0.016157  | 2.90x   | PASS    |



\## Step-by-Step procedure 



Step 0 — Naive Python: Pure Python nested loops. Baseline: 46.9 ms



Step 1 — Numba JIT (546x): @njit compiles the identical loop structure to native machine code. Same algorithm, 546x faster. Python interpreter overhead is the dominant cost.



Step 2 — Loop Reorder (181x): Switches to column-oriented traversal. Column j of L is accessed sequentially for better spatial locality.



Step 3 — Fastmath (199x): @njit(fastmath=True) allows reassociation and approximate FP optimizations.



Step 4 — Loop Tiling (184x): Tiles the inner update in blocks of 32 for cache reuse. Gains grow at larger n.



Step 5 — NumPy BLAS (2.90x): np.linalg.solve calls LAPACK dtrtrs. Slower than Numba at n=512 due to function call overhead.



\## Observations



1\. Peak speedup: 546x from Numba JIT 

2\. TRSV is memory-bound: the sequential data dependency prevents the parallelism that gives GEMM large gains

3\. NumPy/BLAS is not the ceiling for TRSV at small n — Numba beats it



\## Files



1\. benchmarks/trsv\_numba\_bench.py — benchmark implementation

2\. doc/trsv.md — this document

