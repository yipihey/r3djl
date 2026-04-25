# r3djl — A Julia ecosystem around r3d

[![CI](https://github.com/yipihey/r3djl/actions/workflows/CI.yml/badge.svg)](https://github.com/yipihey/r3djl/actions/workflows/CI.yml)
[![docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://yipihey.github.io/r3djl/dev/)

A Julia port of [Devon Powell's r3d](https://github.com/devonmpowell/r3d):
fast, robust polyhedral clipping, analytic moment integration, and
conservative voxelization (rasterization). 2D and 3D supported. Plus a
thin C wrapper for differential testing and performance comparison.

## Status (verified)

- **209,805 tests passing.** 33 pure-Julia, 158 differential vs C r3d,
  408 cross-validation tests where AoS Polytope, Flat Polytope, and
  upstream C r3d are all run on the same random inputs and required to
  agree numerically.
- **Two pure-Julia implementations**:
  - `R3D.Polytope{D,T,S}` — reference port, AoS layout, parametric in
    dimension/type/storage. Easy to read alongside the C source.
  - `R3D.Flat.FlatPolytope{D,T}` — SoA layout matching the C struct's
    memory pattern. **3.7× faster end-to-end** than AoS, ~50× fewer
    allocations.
- **Working C wrapper** (`R3D_C.jl`) via `ccall`. Used as a ground-truth
  oracle in tests and a performance baseline in benchmarks.
- **Benchmark suite** (`R3DBenchmarks.jl`) with three-way comparison.
- See `docs/performance.md` for the numbers and analysis of the
  remaining gap.

## Layout

```
r3djl/
├── R3D.jl/              # Pure-Julia port (the actual library)
│   ├── src/
│   │   ├── R3D.jl       # Top-level module
│   │   ├── types.jl     # Polytope, Vertex, Vec, Plane, storage traits
│   │   ├── init.jl      # init_box!, init_tet!, init_simplex!
│   │   ├── clip.jl      # clip! kernel — direct port of r3d_clip
│   │   ├── moments.jl   # Koehl moment recursion — direct port of r3d_reduce
│   │   └── flat.jl      # Flat (SoA) variant — fast path
│   └── test/
│       ├── runtests.jl  # Pure-Julia + Flat closed-form tests
│       └── differential.jl  # Cross-checks vs C (only run if R3D_LIB set)
├── R3D_C.jl/            # Thin Clang.jl wrapper around upstream C r3d
│   ├── src/R3D_C.jl
│   └── deps/build_libr3d.sh   # Stopgap shared-lib build script
├── R3DBenchmarks.jl/    # Shared benchmarks: AoS vs Flat vs C
└── docs/performance.md  # Verified perf numbers and roadmap
```

## Quick start

### Pure Julia (Flat — recommended)

```julia
using R3D

cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0]/sqrt(3)),
                              -1/sqrt(3))
R3D.Flat.clip!(cube, [plane])
m = R3D.Flat.moments(cube, 1)            # volume + centroid moments
volume   = m[1]
centroid = (m[2], m[3], m[4]) ./ volume
```

### Caller-allocated buffer (hot loops, ~1.0–1.9× C, zero allocations)

For tight loops — grid voxelization, Monte Carlo sweeps, anything that
clips many cells against many planes — allocate a `FlatBuffer` once and
reuse it. This mirrors how the upstream C library is used (`r3d_poly poly;
r3d_init_box(&poly, …);`) and brings the full pipeline to within ~1.3× C
with no per-iteration heap activity.

```julia
using R3D

buf = R3D.Flat.FlatBuffer{3,Float64}(64)        # one allocation total
out = zeros(Float64, R3D.num_moments(3, 1))     # caller-owned moments

for cell in cells
    R3D.Flat.init_box!(buf, cell.lo, cell.hi)   # 0 allocs
    R3D.Flat.clip!(buf, planes)                 # 0 allocs
    R3D.Flat.moments!(out, buf, 1)              # 0 allocs (after first call at this order)
    # …consume `out`…
end
```

`FlatBuffer` is an alias for `FlatPolytope`; the polytope itself is the
buffer. See `docs/performance.md` for the verified numbers.

### Voxelization

Conservative voxelization of a polytope onto a regular Cartesian grid —
the headline use case for r3d. Stack-based bisection with `split_coord!`
at each axis midpoint until each leaf occupies one voxel; per-leaf
moments accumulated into `dest_grid`.

```julia
using R3D

cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
d = (0.25, 0.25, 0.25)                     # voxel spacing per axis

# One-shot: returns (grid, ibox_lo, ibox_hi). Grid shape is (nmom, ni, nj, nk).
grid, lo, hi = R3D.Flat.voxelize(cube, d, 1)   # order=1: volume + 3 first moments
sum(@view grid[1, :, :, :])                # ≈ polytope volume
```

For hot loops, allocate the workspace and destination grid once:

```julia
ws  = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)   # one alloc; 64-vert cap per leaf
grid = zeros(Float64, R3D.num_moments(3, 1), 4, 4, 4)
for poly in polytopes
    fill!(grid, 0.0)
    R3D.Flat.voxelize!(grid, poly, (0,0,0), (4,4,4), d, 1; workspace = ws)  # 0 allocs
    # …consume `grid`…
end
```

Verified within **1.27–1.42× C `r3d_voxelize`** across grid sizes 4³–32³
and moment orders 0 and 2; 50 random clipped polytopes match the C
output bit-for-bit (max diff 2e-16).

### 2D (`R3D.Flat` D = 2) — clipping + rasterization

Same `box` / `clip!` / `moments!` / `voxelize!` API at `D = 2`. Maps
to upstream `r2d.c` and `v2d.c`.

```julia
using R3D

sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}([1.0, 1.0]/sqrt(2)),
                              -1/sqrt(2))
R3D.Flat.clip!(sq, [plane])
R3D.Flat.moments(sq, 1)               # area + centroid moments

# Rasterization (2D voxelize) → grid shape (nmom, ni, nj)
ws  = R3D.Flat.VoxelizeWorkspace{2,Float64}(64)
grid = zeros(Float64, 1, 32, 32)
R3D.Flat.voxelize!(grid, sq, (0,0), (32,32), (1/32, 1/32), 0; workspace = ws)
```

Verified at **0.97–1.16× C `r2d_rasterize`** across grid sizes 8²–64²
and moment orders 0–2 (often a hair faster than C). 16,949 closed-form
+ differential-vs-`r2d` tests pass.

### Small-cap stack-friendly polytope (`StaticFlatPolytope`)

For very tight inner loops where the polytope is known to stay below
~32 vertices (e.g. cube voxelization), `StaticFlatPolytope{D,T,N}`
backs the matrices with `MMatrix{D,N,…}` so `N` is at the type level.
The compiler unrolls per-vertex loops and `init_box!` drops to ~17 ns
(vs ~22 ns for `FlatPolytope`).

```julia
poly = R3D.Flat.StaticFlatPolytope{3,Float64,32}()    # one alloc
for cell in cells
    R3D.Flat.init_box!(poly, cell.lo, cell.hi)
    R3D.Flat.clip!(poly, cell.planes)
    R3D.Flat.moments!(out, poly, 1)
end
```

**Caveat:** each unique `N` triggers a fresh specialization of every
method. Pick one cap per use case.

### Reference AoS variant

Same API surface under `R3D` (no `Flat` prefix):

```julia
cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
R3D.clip!(cube, [plane])
R3D.moments(cube, 1)
```

### C wrapper

```julia
ENV["R3D_LIB"] = "/path/to/libr3d.so"
using R3D_C

ptr, buf = R3D_C.new_poly()
GC.@preserve buf begin
    R3D_C.init_box!(ptr, R3D_C.RVec3(0,0,0), R3D_C.RVec3(1,1,1))
    R3D_C.clip!(ptr, [R3D_C.Plane(R3D_C.RVec3(1,0,0), -0.5)])
    m = zeros(Float64, 1)
    R3D_C.reduce!(ptr, m, 0)
end
```

## Building libr3d

A BinaryBuilder recipe is checked in at `r3d_jll/build_tarballs.jl` —
once submitted to Yggdrasil and merged it'll publish `r3d_jll`, after
which `using r3d_jll: libr3d` replaces the manual `R3D_LIB` setup. See
`r3d_jll/README.md` for how to test the recipe locally and submit it.

Until then:

```bash
cd /tmp
git clone https://github.com/devonmpowell/r3d.git
mkdir libr3d
cat > libr3d/r3d-config.h << 'EOF'
#ifndef R3D_CONFIG_H
#define R3D_CONFIG_H
#define R3D_MAX_VERTS 512
#endif
EOF
gcc -O3 -fPIC -shared -Ilibr3d -Ir3d/src r3d/src/r3d.c -lm -o libr3d/libr3d.so
export R3D_LIB=$PWD/libr3d/libr3d.so
```

(`R3D_C.jl/deps/build_libr3d.sh` does the same via cmake if you have it.)

## Design decisions (verified outcomes)

### Dimension as a type parameter

`Polytope{D,T,S}` unifies r2d/r3d/rNd. Specialization on `D` produces the
same machine code as r3d's hand-coded `for (np = 0; np < 3; ++np)` for
D=3. **Implemented** for D=2,3 in clip; D≥4 stubbed pending faithful
port of `rNd_clip`'s `finds[][]` table.

### Vertex storage as a trait

Both static (`StaticStorage{N}`) and dynamic (`DynamicStorage`) work in
the AoS reference path. The `Flat` variant is currently fixed-capacity
only. **In practice the Flat variant's flat-matrix layout dominates
performance considerations**, so the AoS storage trait is mostly
academic — it's there so the AoS code matches r3d / PolyClipper
literally, and exists for ongoing differential validation.

### Two parallel implementations

We deliberately kept the AoS reference port working alongside the Flat
variant. The AoS code is a **literal line-for-line translation of the
C source**, which makes it the right artifact to cite in tests, papers,
and code review. The Flat variant is the one to use for actual work.

## What's verified, what's not

Done and tested:

- [x] AoS Polytope: types, init_box, init_tet, init_simplex (D=2,3)
- [x] AoS clip! for D=3, full algorithm (insert + link + compact)
- [x] AoS moments! via Koehl recursion, all polynomial orders
- [x] Flat (SoA) Polytope: init_box, clip!, moments!
- [x] R3D_C ccall wrapper for init_box/init_tet/clip/reduce/is_good
- [x] Differential testing vs C (1000+ random clips, all agree to 1e-10)
- [x] Performance comparison (3.7× full-pipeline speedup AoS → Flat)
- [x] Caller-allocated `FlatBuffer` API for hot loops — full pipeline at
      ~1.3× C, `clip!` at 1.0–1.3× C, zero per-call allocations
      (5000+ buffer-reuse tests pass against C)
- [x] Conservative voxelization (`v3d.voxelize!`) — `split_coord!`,
      `get_ibox`, `voxelize!` ported with `VoxelizeWorkspace` for
      0-alloc hot loops. Within 1.27–1.42× C across 4³–32³ grids and
      orders 0–2; 150k+ per-voxel diff tests pass against C.
- [x] **D = 2 (port of `r2d.c` + `v2d.c`)**: `init_box!`, `clip!`,
      `moments!`, `split_coord!`, `get_ibox`, `voxelize!` for D=2.
      Buffered clip+reduce within 1.08–1.44× C `r2d`; 2D voxelize at
      0.97–1.16× C `r2d_rasterize` (sometimes faster than C). 16,949
      closed-form + differential tests pass.
- [x] **`StaticFlatPolytope{D,T,N}`** — `MMatrix`-backed small-cap
      variant. `init_box!` ~17 ns (vs ~22 ns for `FlatPolytope`).
      411 cross-validation tests vs `FlatPolytope` pass.
- [x] **`split!`** (D=2 + D=3) — single polytope split at an arbitrary
      plane into two outputs. Mirrors `r3d_split` / `r2d_split`;
      differential-tested vs C across 100 random planes (per-side moments
      agree at 1e-10).
- [x] **`is_good`** (D=2 + D=3 + `StaticFlatPolytope`) — connectivity
      sanity check. Cross-checked with C `r3d_is_good` / `r2d_is_good`.
- [x] **`shift_moments!`** (D=2 + D=3) — Pascal-triangle multinomial
      shift. Mirrors `r3d_shift_moments` / `r2d_shift_moments`; matches
      C bit-for-bit across 30 random shifts at order 3.
- [x] **`r3d_jll/build_tarballs.jl`** — Yggdrasil BinaryBuilder recipe
      pinned to upstream HEAD. Ready to submit as a PR to JuliaPackaging
      /Yggdrasil under `R/r3d/` (see `r3d_jll/README.md`).

Not yet:

- [ ] `init_brep` (init poly from boundary representation; `r3d.c:1058`)
- [ ] D≥4 clip linker (needs `finds[][]` 2-face table from rNd.c) and
      `vNd` voxelization — narrow appeal; defer until a use case appears.

See `docs/performance.md` for the prioritized next steps.

## Citation

If you use this, please cite Powell & Abel (2015), *J. Comput. Phys.*
297: 340-356, the paper behind the original r3d.
