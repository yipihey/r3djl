# 3D voxelization

`R3D.Flat.voxelize!` ports `r3d_voxelize` from upstream `v3d.c`. It
splits a polyhedron recursively along the longest grid axis until each
leaf occupies one voxel, then accumulates per-voxel moments into a
caller-allocated grid.

## Quick example

```julia
using R3D

# A clipped tetrahedron — the polytope you want to voxelize.
poly = R3D.Flat.tet((0.0, 0.0, 0.0),
                    (1.5, 0.0, 0.0),
                    (0.0, 1.5, 0.0),
                    (0.0, 0.0, 1.5))
plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0]/√3),
                              -0.7/√3)
R3D.Flat.clip!(poly, [plane])

# One-shot convenience: returns (grid, ibox_lo, ibox_hi).
d = (0.1, 0.1, 0.1)
grid, lo, hi = R3D.Flat.voxelize(poly, d, 1)   # order=1 → vol + ∫x + ∫y + ∫z
total_volume = sum(@view grid[1, :, :, :])
```

`grid` has shape `(nmom, ni, nj, nk)` where `nmom = num_moments(3, order)`.
The first axis (moments) is contiguous in memory, matching the access
pattern in the per-leaf accumulator.

## Hot-loop pattern (zero allocations)

For tight inner loops — e.g. iterating over many polytopes against the
same grid — allocate the workspace and the destination grid once:

```julia
ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
ni = nj = nk = 32
d = (1/ni, 1/nj, 1/nk)
grid = zeros(Float64, R3D.num_moments(3, 1), ni, nj, nk)

for cell in cells
    fill!(grid, 0.0)
    R3D.Flat.voxelize!(grid, cell.poly, (0,0,0), (ni,nj,nk), d, 1;
                        workspace = ws)   # 0 allocations
    # …consume grid…
end
```

The workspace owns a stack of `FlatPolytope`s sized to the recursion
depth (`ceil(log2 ni) + ceil(log2 nj) + ceil(log2 nk) + 2`). Pre-allocated
to depth 64 by default; grown lazily for huge grids.

## Performance

| Grid             | Order | R3D.Flat       | C r3d  | Ratio |
|------------------|------:|---------------:|-------:|------:|
| 4³               |     0 | 23.4 μs        | 17.2 μs|  1.36× |
| 8³               |     0 | 191 μs         | 140 μs |  1.37× |
| 16³              |     0 | 1.50 ms        | 1.13 ms|  1.33× |
| 32³              |     0 | 12.4 ms        | (var.) | ~1.30× |
| 8³               |     2 | 511 μs         | 372 μs |  1.37× |
| 16³              |     2 | 3.94 ms        | 2.79 ms|  1.41× |

The ~1.3× ceiling is set by the per-leaf `moments!` cost (which itself
runs at 1.3–1.4× C). Closing it further would require LLVM-level
tuning of the moment recursion. See [Performance](performance.md) for
the full table.

## Correctness

Cross-checked against `r3d_voxelize` over 50 random clipped polytopes ×
random anisotropic spacings × orders 0–2 — **150,000+ per-voxel
comparisons**, max diff 2e-16. See `R3D.Flat voxelize` testset in
`test/runtests.jl`.
