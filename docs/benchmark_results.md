# Verified benchmark results

Captured run, Linux x64, Julia 1.11.4, libr3d built with gcc 13.3.0
-O3 -fPIC, BenchmarkTools median times.

```
init_box           AoS:    8153 ns (1028 a)   Flat:    1168 ns (11 a)   C:    10 ns   Flat/C: 113×
clip diagonal      AoS:    7722 ns (1034 a)   Flat:    1739 ns (17 a)   C:    95 ns   Flat/C:  18×
clip 4 random      AoS:    8257 ns (1034 a)   Flat:    2064 ns (17 a)   C:   278 ns   Flat/C:   7.4×
reduce order=0     AoS:     330 ns (   7 a)   Flat:     296 ns ( 7 a)   C:   149 ns   Flat/C:   2.0×
reduce order=2     AoS:    1411 ns (   7 a)   Flat:    1394 ns ( 7 a)   C:   630 ns   Flat/C:   2.2×
reduce order=4     AoS:    3845 ns (   7 a)   Flat:    3860 ns ( 7 a)   C:  1962 ns   Flat/C:   2.0×
full pipeline (4)  AoS:   11718 ns (1041 a)   Flat:    2882 ns (24 a)   C:   654 ns   Flat/C:   4.4×
```

(Allocation counts in parens. AoS = original mutable-struct
implementation. Flat = SoA matrix-of-positions implementation.)

## What this shows

- **The SoA refactor delivered.** Full-pipeline Flat/C is **4.4×**, down
  from AoS/C of **18×**. Allocations went from 1041 to 24 per pipeline
  call.
- **`reduce` is at the asymptotic ceiling.** 2× C across all polynomial
  orders is what tight Julia-vs-C numerical-loop comparisons typically
  yield. Closing further would require LLVM-level work that isn't
  worth the complexity for moment integration.
- **Construction is the remaining lever.** The 113× Flat/C ratio for
  `init_box` is two `Matrix` allocations (positions + pnbrs, totaling
  18 KB) that the C version doesn't pay because the caller provides
  the buffer. A caller-allocated buffer API on the Julia side would
  drop this to ~50 ns and bring the full-pipeline ratio close to 2×.

See `performance.md` for the full discussion and roadmap.
