# Performance

All numbers from a single Apple Silicon laptop with `gcc -O3 -fPIC` for
`libr3d` and Julia 1.12. Reproducible via `R3DBenchmarks.run_all()`.

## Headline: optimized buffered path

| Operation (3D) | R3D.Flat (FlatBuffer reused) | C r3d  | Ratio |
|---|---:|---:|---:|
| `init_box!` (cube)              | **21.9 ns** (0 allocs) | 11.6 ns | 1.89× |
| init+clip 1 random plane         | **114 ns**           | 96 ns   | 1.19× |
| init+clip 4 random planes        | **301 ns**           | 284 ns  | 1.06× |
| init+clip 8 random planes        | **527 ns**           | 522 ns  | **1.01×** |
| Full pipeline (4 planes, ord=2)  | **793 ns**           | 610 ns  | 1.30× |
| `reduce` ord = 0                 | 262 ns               | 135 ns  | 1.95× |
| `reduce` ord = 2                 | 880 ns               | 631 ns  | 1.40× |
| `reduce` ord = 4                 | 2.72 μs              | 2.06 μs | 1.32× |
| voxelize 16³ grid, ord = 0       | 1.52 ms              | 1.15 ms | 1.33× |

| Operation (2D) | R3D.Flat | C r2d | Ratio |
|---|---:|---:|---:|
| init+clip 4 random + reduce ord=2 | **157 ns** (0 allocs) | 141 ns  | 1.11× |
| init+clip 8 random + reduce ord=2 | **155 ns**            | 141 ns  | 1.10× |
| voxelize 8² grid, ord = 0         | 5.58 μs               | 5.73 μs | **0.97×** |
| voxelize 32² grid, ord = 0        | 90.2 μs               | 90.8 μs | **0.99×** |
| voxelize 64² grid, ord = 0        | 363 μs                | 369 μs  | **0.98×** |

## Pattern

Use [`R3D.Flat.FlatBuffer`](@ref) (alias for `FlatPolytope`) and reuse
it across iterations:

```julia
buf = R3D.Flat.FlatBuffer{3,Float64}(64)
for poly in polys
    R3D.Flat.init_box!(buf, poly.lo, poly.hi)
    R3D.Flat.clip!(buf, poly.planes)
    R3D.Flat.moments!(out, buf, order)
end
```

This eliminates all per-call allocation. The original "AoS" reference
path is still in the package but intentionally unoptimized for
cross-validation; it shows up at 5–10× C and should not be used in
production.

## Where the remaining gap is

`reduce` order 0 at 1.95× C is the per-call function-dispatch overhead
unique to Julia. As order grows the per-call overhead amortizes
against more work and the ratio drops to 1.30×. Voxelization inherits
this ratio almost exactly because it is dominated by per-leaf
`moments!` calls.

Closing the remaining ~30% would require manual SIMD or function-call
elimination at the LLVM level — explicitly out of scope. Where it
matters (single small clips in a tight loop), the
[`StaticFlatPolytope`](@ref) variant trims another 20% on `init_box!`
by keeping the polytope on the function frame via `MMatrix`.
