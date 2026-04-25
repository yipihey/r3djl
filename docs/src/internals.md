# Internals

A pointer for contributors. The hot path lives in
[`R3D.jl/src/flat.jl`](https://github.com/yipihey/r3djl/blob/main/R3D.jl/src/flat.jl).

## Storage

```julia
mutable struct FlatPolytope{D,T}
    positions::Matrix{T}     # D × capacity
    pnbrs::Matrix{Int32}     # D × capacity (1-based; 0 = "unset")
    nverts::Int
    capacity::Int
    # scratch — sized to capacity, reused across calls:
    sdists::Vector{T}
    clipped::Vector{Int32}
    emarks::Matrix{Bool}
    Sm::Array{T,3}            # moments scratch (lazily grown by order)
    Dm::Array{T,3}
    Cm::Array{T,3}
    moment_order::Int
end
```

Two key design choices:

1. **The polytope IS the buffer.** The C library's pattern is
   `r3d_poly poly; r3d_init_box(&poly, …);` — caller owns storage,
   reuses across calls. We mirror that: `FlatPolytope` owns its scratch,
   so `clip!` / `moments!` are zero-alloc once the polytope exists.
2. **SoA layout** (`Matrix{T}` for positions, not `Vector{Vertex{T}}`)
   matches the C struct's memory layout for prefetch friendliness, and
   the column-copy in compaction has no aliasing risk.

`StaticFlatPolytope{D,T,N,DN}` swaps the heap-backed matrices for
`MMatrix{D,N,T,DN}` so `N` is at the type level. Each unique `N`
specializes; the win is mainly in `init_box!` (~20% faster) because
the per-vertex loop unrolls.

## Algorithm map

| Function | Upstream C | Julia line range |
|----------|-----------|------------------|
| `init_box!` (D=3) | `r3d_init_box` (`r3d.c:670`) | `flat.jl:130–158` |
| `init_box!` (D=2) | `r2d_init_box` (`r2d.c:???`) | `flat.jl` 2D block |
| `clip_plane!` (D=3) | `r3d_clip` (`r3d.c:45`) | `flat.jl:175–270` |
| `clip_plane!` (D=2) | `r2d_clip` (`r2d.c:45`) | `flat.jl` 2D block |
| `moments!` (D=3) | `r3d_reduce` (`r3d.c:253`) | `flat.jl:300–410` |
| `moments!` (D=2) | `r2d_reduce` (`r2d.c:229`) | `flat.jl` 2D block |
| `split_coord!` | `r3d_split_coord` (`v3d.c:126`) | `flat.jl:430–560` |
| `voxelize!` (D=3) | `r3d_voxelize` (`v3d.c:59`) | `flat.jl:580–680` |
| `voxelize!` (D=2) | `r2d_rasterize` (`v2d.c:59`) | `flat.jl` 2D block |
| `split!` | `r3d_split` (`r3d.c:141`) | `flat.jl` "additional ops" block |
| `is_good` | `r3d_is_good` (`r3d.c:477`) | same |
| `shift_moments!` | `r3d_shift_moments` (`r3d.c:404`) | same |

## Key kernel primitives

- `find_back3(pnbrs, vnext, vcur)` — 3-way unrolled search for the
  back-pointer slot. Replaces a `for k in 1:3` linear scan.
- `next3(np)` — branchless `np == 3 ? 1 : np + 1`. Replaces
  `mod1(np + 1, 3)` so the inner loop has no division.
- `_copy_polytope!` / `_copy_polytope_2d!` — column-copy used by
  `split_coord!` and `voxelize!` for the trivial pass-through case.

## Aliasing in `split_coord!` and `split!`

`voxelize!` reuses the popped stack slot as `out0` when calling
`split_coord!(in, out0, out1, …)`, so `in === out0`. The compaction
loop must cache `total_verts = in.nverts` **before** zeroing
`out0.nverts`, otherwise the loop bound becomes 0. The same gotcha
applies to `split!` because the same compaction shape is used.

## Tests as documentation

The differential tests in `R3D.jl/test/runtests.jl` (especially the
"Differential vs C" testsets) are the most reliable guide to API
contracts. When porting a new C function:

1. Add the C wrapper to `R3D_C.jl/src/R3D_C.jl`.
2. Implement the Julia version in `flat.jl`.
3. Add a randomized differential testset that drives both with the
   same input.
4. Verify the new test fails before the implementation lands.
