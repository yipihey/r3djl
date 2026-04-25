# R3D.jl

A Julia port of [Devon Powell's r3d](https://github.com/devonmpowell/r3d):
fast, robust polyhedral and polygon clipping, analytic moment integration,
and conservative voxelization (rasterization).

## Features

- **D = 2 and D = 3** with the same API: `init_box!`, `init_simplex!`,
  `init_poly!`, `clip!`, `split!`, `moments!`, `voxelize!`,
  `shift_moments!`, `is_good`, plus the affine ops `translate!` /
  `scale!` / `rotate!` / `shear!` / `affine!`.
- **Performance within 1.0–1.4× of upstream C `libr3d`** across every hot
  path, with **zero per-call allocations** when using a reused
  `FlatBuffer`. See [Performance](performance.md) for verified numbers.
- **`StaticFlatPolytope{D,T,N}`** — `MMatrix`-backed small-cap variant
  for tight inner loops where the polytope stays below ~32 vertices.
- **AD-friendly** — `ForwardDiff` works through `clip!` + `moments!`
  end-to-end (allocate the polytope as `FlatPolytope{3,T}` matching
  the input element type).
- **Cross-validated against the C library** at floating-point
  precision: 177,000+ tests pass, including 150,000+ per-voxel
  comparisons against `r3d_voxelize`.

## Install

Once registered:

```julia
using Pkg
Pkg.add("R3D")
```

For now (pre-registration), develop the package locally:

```julia
Pkg.develop(url = "https://github.com/yipihey/r3djl.git", subdir = "R3D.jl")
```

The C wrapper `R3D_C` (used for differential testing and benchmarks) is
optional; it requires the upstream `libr3d` shared library — set
`ENV["R3D_LIB"]` before `using R3D_C`. A
[BinaryBuilder](https://binarybuilder.org) recipe is checked in at
[`r3d_jll/build_tarballs.jl`](https://github.com/yipihey/r3djl/blob/main/r3d_jll/build_tarballs.jl)
and will publish `r3d_jll` to JuliaHub once merged into Yggdrasil.

## Quick start (3D)

```julia
using R3D

# Allocate the buffer once, reuse it
buf = R3D.Flat.FlatBuffer{3,Float64}(64)
out = zeros(Float64, R3D.num_moments(3, 1))   # volume + centroid

R3D.Flat.init_box!(buf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0]/√3),
                              -1/√3)
R3D.Flat.clip!(buf, [plane])
R3D.Flat.moments!(out, buf, 1)

vol = out[1]
centroid = (out[2], out[3], out[4]) ./ vol
```

## Quick start (2D)

```julia
using R3D

sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}([1.0, 1.0]/√2), -0.5/√2)
R3D.Flat.clip!(sq, [plane])

m = R3D.Flat.moments(sq, 1)              # area + first moments
area = m[1]
centroid = (m[2], m[3]) ./ area
```

## Where to look next

- [2D rasterization](2d.md) — voxelize a clipped 2D polygon.
- [3D voxelization](voxelize.md) — grid voxelization with reused workspace.
- [Performance](performance.md) — verified benchmarks vs C.
- [Internals](internals.md) — implementation walkthrough.
- [API reference](api.md) — every exported symbol.

## Reference

Powell, D. & Abel, T. (2015). *An exact general remeshing scheme applied
to physically conservative voxelization*. J. Comput. Phys. 297: 340-356.
[arXiv:1412.4941](https://arxiv.org/abs/1412.4941)
