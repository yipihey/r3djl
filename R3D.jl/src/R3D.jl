"""
    R3D

Pure-Julia port of [r3d](https://github.com/devonmpowell/r3d): fast and
robust polytope clipping, analytic moment integration, and conservative
voxelization.

# Quick start

```julia
using R3D

# Build a unit cube
cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

# Clip with a plane: keep the half-space x + y + z ≥ 1
plane = R3D.Plane{3,Float64}([1.0, 1.0, 1.0] / sqrt(3), -1/sqrt(3))
R3D.clip!(cube, [plane])

# Compute volume and centroid
m = R3D.moments(cube, 1)
volume = m[1]
centroid = (m[2], m[3], m[4]) ./ volume
```

# Type parameters

`Polytope{D,T,S}` is parametric in:
- `D` — dimension (2, 3, 4, ...)
- `T` — coordinate type (`Float64`, `Float32`, AD duals, ...)
- `S` — storage strategy: `StaticStorage{N}` (fixed cap) or
  `DynamicStorage` (growable)

# Reference

Powell, D. & Abel, T. (2015). *An exact general remeshing scheme applied
to physically conservative voxelization*. J. Comput. Phys. 297: 340-356.
arXiv:1412.4941
"""
module R3D

using StaticArrays

include("types.jl")
include("clip.jl")
include("init.jl")
include("moments.jl")
include("flat.jl")
include("intexact.jl")

# Public API
export Polytope, Vertex, Vec, Plane
export StaticStorage, DynamicStorage
export clip!, moments, moments!, num_moments
export init_box!, init_tet!, init_simplex!
export box, tet
export poly_center

end # module R3D
