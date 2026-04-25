# Performance findings — Julia port vs C r3d

## Three-way comparison (Linux x64, libr3d -O3 -fPIC, single-threaded, Julia 1.11)

Three implementations, identical inputs:
- **AoS Julia**: original `Polytope{D,T,S}` with `Vector{Vertex}` of mutable structs.
- **Flat Julia**: `R3D.Flat.FlatPolytope` with two flat `Matrix` fields (positions + pnbrs).
- **C r3d**: upstream library, `-O3 -fPIC`.

All times are `BenchmarkTools.median`. Allocs in parens.

| Operation              | AoS Julia       | Flat Julia       | C r3d   | Flat/C  |
|------------------------|-----------------|------------------|---------|---------|
| `init_box`             | 12.7 μs (1028)  |  **3.8 μs** (11) |  11 ns  | 347×    |
| `clip!` diagonal       | 10.7 μs (1034)  |  **2.9 μs** (17) |  94 ns  |  31×    |
| `clip!` 4 random       | 12.5 μs (1034)  |  **3.7 μs** (17) | 295 ns  |  12×    |
| `reduce` order 0       |  341 ns    ( 7) |    304 ns   ( 7) | 159 ns  |  1.9×   |
| `reduce` order 2       |  1.4 μs    ( 7) |    1.4 μs   ( 7) | 656 ns  |  2.2×   |
| `reduce` order 4       |  3.6 μs    ( 7) |    3.7 μs   ( 7) | 2.0 μs  |  1.8×   |
| Full pipeline (clip+reduce) | 16.0 μs (1041) | **4.3 μs** (24) | 586 ns  |  7.3×   |

## Phase 1+3: caller-allocated `FlatBuffer` (Apple Silicon, libr3d gcc -O3 -fPIC, Julia 1.12)

Adding scratch fields (`sdists`, `clipped`, `emarks`, `Sm/Dm/Cm`) directly to
`FlatPolytope` so the polytope itself is the reusable buffer — exactly how
the C library is used (`r3d_poly poly; r3d_init_box(&poly, …);`). All
buffered numbers are with a single `FlatBuffer{3,Float64}(64)` reused across
iterations; the C numbers reuse the same `r3d_poly`. Phase 3 micro-tuning
(unrolled back-pointer search, branchless face-walk index) is folded in.

| Operation                              | Buffered Flat     | C r3d   | Buffered/C |
|----------------------------------------|-------------------|---------|------------|
| `init_box!`                            |  **21.7 ns** (0)  |  11.6 ns |  1.87×    |
| `init+clip` diagonal                   | **126.0 ns** (0)  |  94.7 ns |  1.33×    |
| `init+clip` 1 random                   | **115.1 ns** (0)  |  95.6 ns |  1.20×    |
| `init+clip` 4 random                   | **302.1 ns** (0)  | 283.4 ns |  1.07×    |
| `init+clip` 8 random                   | **527.1 ns** (0)  | 521.5 ns | **1.01×** |
| Full pipeline (4 planes, order = 2)    | **795.5 ns** (0)  | 609.4 ns |  1.30×    |

Headline: all hot ops are within 1.0–1.9× C with **zero per-call
allocations**. Full pipeline went from 7.3× C → 1.30× C, a ~5.5× wall-clock
improvement plus 24 → 0 allocations per iteration.

**What changed:**
- `FlatPolytope` now owns its scratch (`sdists`, `clipped`, `emarks`,
  `Sm/Dm/Cm`), allocated once at construction. `clip!` and `moments!`
  reuse them; the moment scratch is lazily grown only when a higher
  polynomial order is requested.
- `init_box!(buf, lo, hi)` becomes a 22 ns sequence of stores into the
  reused matrices.
- The `for k in 1:3` back-pointer search in clip's face-linker and in
  `moments!`'s face-walk is replaced by a 3-way unrolled `find_back3`.
- `mod1(np + 1, 3)` is replaced by a branchless `next3` (an
  `ifelse`-based `np == 3 ? 1 : np + 1`).

The non-buffered AoS numbers in the table above are unchanged — that path
intentionally remains a literal port of the C source for cross-validation.
Reach for `R3D.Flat.FlatBuffer` (the `FlatBuffer` alias of `FlatPolytope`)
in any hot loop.

## Phase 4: voxelization (`R3D.Flat.voxelize!`)

Direct port of `r3d_voxelize` (`src/v3d.c`) onto `FlatPolytope`. Stack-
based bisection via `split_coord!` (also a Phase-4 deliverable, mirroring
`r3d_split_coord`). A `VoxelizeWorkspace` owns the per-level polytope
stack so each `voxelize!` call is allocation-free after warmup.

| Operation                            | Buffered Flat | C r3d   | Buffered/C |
|--------------------------------------|---------------|---------|------------|
| voxelize 4³ grid, order = 0          |  23.4 μs (4) | 17.2 μs |  1.36×     |
| voxelize 8³ grid, order = 0          | 190.3 μs (4) |141.4 μs |  1.35×     |
| voxelize 16³ grid, order = 0         |   1.50 ms (4)|  1.14 ms|  1.32×     |
| voxelize 32³ grid, order = 0         |  12.20 ms (4)|  9.61 ms|  1.27×     |
| voxelize 8³ grid, order = 2          | 506.7 μs (4) |361.6 μs |  1.40×     |
| voxelize 16³ grid, order = 2         |   4.08 ms (4)|  2.88 ms|  1.42×     |

(The 4 allocations per benchmark iteration come from the literal
`[0.0,…]` `lo`/`hi` setup vectors in the timed region; calling
`voxelize!` directly with the same workspace and a pre-zeroed grid is
0-alloc — verified by `@allocated`.)

**Why ~1.3× C is the right ceiling here:** the algorithm is dominated
by per-leaf `moments!` calls (each leaf is `reduce` on an ≤8-vertex
polytope), and `reduce` itself is at 1.3–1.4× C from Phase 1+3.
Voxelization inherits that ratio almost exactly. Closing it further
would require LLVM-level tuning of the moment recursion, which we
explicitly chose to skip in `performance.md`.

**Correctness:** 150k+ per-voxel comparisons against `r3d_voxelize`
across 50 random clipped polytopes × random anisotropic spacings ×
orders 0–2. Max observed diff 2.0e-16.

## D = 2 (`R3D.Flat` 2D path; mirrors `r2d.c` + `v2d.c`)

Same `FlatPolytope{2,T}` machinery — only the algorithm body differs
(single-vertex insertion per cut edge, 1D face walk, 2D Koehl
recursion). All buffered numbers reuse a single `FlatBuffer{2,Float64}`
across iterations.

| Operation                                    | Buffered Flat   | C r2d   | Buffered/C |
|----------------------------------------------|-----------------|---------|------------|
| `init+clip` 1 random + reduce ord = 2        | **151.5 ns** (0)|105.0 ns | 1.44× |
| `init+clip` 4 random + reduce ord = 2        | **153.5 ns** (0)|141.7 ns | 1.08× |
| `init+clip` 8 random + reduce ord = 2        | **162.1 ns** (0)|141.1 ns | 1.15× |
| voxelize 8² grid, ord = 0                    |   **5.6 μs** (0)|  5.8 μs | **0.97×** |
| voxelize 32² grid, ord = 0                   |  **90.2 μs** (0)| 91.7 μs | **0.98×** |
| voxelize 64² grid, ord = 0                   | **362.9 μs** (0)|367.8 μs | **0.99×** |
| voxelize 32² grid, ord = 2                   | **176.0 μs** (0)|151.3 μs | 1.16× |

In 2D the voxelize/rasterize path is consistently within ±2% of C
(occasionally faster) — the work per leaf is dominated by 2D
moments, which the Julia compiler hits at C speed thanks to fewer
indirections in the linker.

**Correctness:** 16,949 tests pass — closed-form area / centroid /
order-2 moments on the unit square; 200 random clipped squares
differential vs `r2d_clip + r2d_reduce`; per-voxel comparison vs
`r2d_rasterize`. All within floating-point precision.

## Phase 2: `StaticFlatPolytope{D,T,N}` (small-cap, MMatrix-backed)

Type-level vertex capacity backed by `MMatrix{D,N,T}`. Each unique
`N` produces its own specialization (so callers should pick one cap
per use case), but in exchange the compiler unrolls every per-vertex
loop. The struct itself is heap-allocated once, but the matrices live
inline within it.

| Operation               | StaticFlatPolytope (N=32) | FlatPolytope (cap=32) |
|-------------------------|---------------------------|-----------------------|
| `init_box!`             |  **17.5 ns** (0)          |  21.6 ns (0)          |
| Full pipeline (4 random, ord=2) | **773 ns** (0)    |  783 ns (0)           |

The `init_box!` win is real (~20%); the full pipeline is within
benchmark noise — `clip!` and `moments!` are dominated by per-vertex
loop work that the heap-backed `Matrix` already handles fine. Use
`StaticFlatPolytope` when you specifically want a tighter `init_box!`
or when you'd benefit from `N` being a compile-time constant
elsewhere.

## What worked: SoA layout

Replacing `Vector{Vertex{D,T}}` (where `Vertex` is a `mutable struct`
holding heap-allocated `MVector`s) with two flat `Matrix` fields cut
allocations from ~1030 to ~17 per `clip!`, and dropped construction
time **3.4×**. Full-pipeline speedup is **3.7×**.

The bug-prone behaviour of mutable-struct-in-array also went away: the
compaction step in clip is now a value copy by construction (column
assignment in a `Matrix{Float64}` cannot alias).

## Where the remaining gap is

- **`reduce` is at 1.8–2.2× C.** This is the asymptotic ceiling for a
  faithful Julia port of a tight C numerical loop: comparable to what
  hand-tuned Julia gets versus `gcc -O3` on the same algorithm. Closing
  this would require LLVM-level tuning (manual SIMD, hot-path
  function-call elimination) and isn't worth the complexity.
- **`init_box` is still 347× C** because the Flat constructor does 11
  allocations: two `Matrix` allocs (12 KB + 6 KB for default cap=512)
  plus minor scratch. C writes into a caller-provided struct and does
  zero allocations.
- **`clip!` is at 12–31× C** end-to-end, but most of that is inherited
  from `init_box`. The actual clip work, with init excluded, is closer
  to 5–10× — a ratio dominated by Julia's per-call overhead on
  `clip_against_plane!`'s arg passing and bounds checking. With
  `@inline` markers and `@inbounds` already applied, further wins need
  profiler-driven micro-tuning.

## Recommended next steps

In rough order of payoff per effort:

1. **Caller-allocated buffers.** `init_box!(buf::FlatBuffer, lo, hi)`
   where `FlatBuffer` is a reusable struct the caller owns. Eliminates
   the 11 allocs in `init_box` for hot loops (e.g. cell-by-cell grid
   intersection). Expected: `init_box` from 3.8 μs → ~50 ns. Mirrors
   how the C library is actually used.

2. **`MMatrix`-backed `FlatPolytope` for small caps.** When
   `R3D_MAX_VERTS` ≤ 64 or so, the entire polytope fits in stack-
   allocated `MMatrix`. Zero heap, zero GC. Useful for pixel-level
   voxelization where polytopes are simple.

3. **Fold `sdists`/`clipped` scratch into the Polytope.** Currently
   each `clip!` allocates two `Vector{T}` of size `capacity`. For the
   static-cap case, these can live as fields of the Polytope itself
   (allocated once at construction). Saves 2 allocs per clip call.

4. **Profile `reduce` for the missing 0.8×.** Even getting from 1.8×
   to 1.2× would be material since `reduce` dominates the
   "many-cells, simple-geometry" use case.

5. **Voxelization (`v3d`) port.** Not yet started; this is the big
   remaining feature gap. Would benefit immediately from the Flat
   layout.

## Why this exercise was worth doing

The bug-vs-bug story is the most informative outcome:

- Differential testing against C found a **mutable-struct aliasing bug**
  in compaction that pure unit tests would never have surfaced
  (volumes happened to be correct on simple cases; the aliasing
  manifested only on multi-plane sequential clips).
- Side-by-side benchmarking made the **AoS construction overhead**
  unmissable. Without C as a reference, "9 μs to build a cube" reads
  as "fine, Julia is dynamic"; with C at 11 ns next to it, it reads as
  "we're paying 800× for a constructor."

Both findings would have been hard to extract from either
implementation in isolation. Keeping the C wrapper around as an oracle
for ongoing work is cheap and high-value.
