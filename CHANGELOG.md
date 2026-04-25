# Changelog

All notable changes to R3D.jl, R3D_C.jl, and R3DBenchmarks.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (HierarchicalGrids overlap-layer support)

- `init_simplex!(buf, vertices)` collection wrappers for D = 2 and D = 3.
- D = 3 alias `init_simplex!` forwarding to `init_tet!` so the same
  API name covers both dimensions.
- Phase 2 overlap-layer helpers, all 0-alloc:
  - `aabb(poly)` returning `((lo…), (hi…))` `NTuple`s for D = 2, D = 3,
    and `StaticFlatPolytope` D = 3.
  - `box_planes(lo, hi)` and `box_planes!(out, lo, hi)` for D = 2 (4
    planes) and D = 3 (6 planes), in `(+axis_k, -axis_k)` pair order.
  - `is_empty(poly)` predicate.
  - `volume(poly)` returning the 0-th moment scalar without allocating.
  - `copy!(dst, src)` for D = 2 and D = 3.
- `init_box!(::FlatPolytope{D,T}, lo, hi)` and
  `init_simplex!(::FlatPolytope{D,T}, vertices)` for `D ≥ 4` —
  vertices and `pnbrs` only; the `finds[D][D]` 2-face table for
  `clip!` is not yet populated. `clip!` and `moments!` for `D ≥ 4`
  raise informative errors pointing at `docs/phase3_status.md`.
- `bench_overlap_2d` synthetic benchmark: 1024 random triangles ×
  `32²` Eulerian grid → ~530 ns per overlap pair.
- `docs/overlap_example.md` and `examples/overlap_triangle_box.jl` —
  worked end-to-end overlap with closed-form sanity check.
- `docs/phase3_status.md` — status note for the D ≥ 4 (rNd) port,
  documenting the `finds[][]` 2-face table requirement, dimension
  scaffolding, and validation infrastructure plan.

### Notes

- An init signature surprise relative to the overlap-layer prompt:
  the D = 3 simplex constructor was already named `init_tet!`. Both
  names now work (`init_simplex!` is an alias plus the
  collection-style wrapper).

## [0.1.0] — 2026-04-25

Initial public release. Highlights:

- `Base.show` for `FlatPolytope` and `StaticFlatPolytope`.
- `translate!`, `scale!`, `rotate!`, `shear!`, `affine!` for D=2 and D=3
  (mirror upstream `r3d_translate` / `r3d_rotate` / `r3d_scale` /
  `r3d_shear` / `r3d_affine` and the 2D analogs).
- `init_tet!`, `init_simplex!` (D=2 triangle), `init_poly!` for D=2 and
  D=3 (general convex polytope from vertex + face list).
- `voxelize_batch!` — multi-threaded batched voxelization helper
  (`Threads.@threads` over a vector of polytopes, one workspace per
  thread).
- ForwardDiff AD compatibility verified end-to-end through
  `clip!` + `moments!` (gradient matches finite-difference reference).
- Documenter.jl docs site auto-deployed to gh-pages.
- `examples/` directory with `voxelize_clipped_tet.jl`,
  `ad_shape_optimization.jl`, `rasterize_2d.jl`, `monte_carlo_volume.jl`.
- GitHub Actions CI matrix (Julia LTS / 1 / nightly × Linux / macOS),
  CompatHelper, TagBot.
- `Aqua.jl` quality tests (no piracy, no method ambiguities, all deps
  used, no unbound type params, no undefined exports).
- Yggdrasil PR opened for `r3d_jll`:
  <https://github.com/JuliaPackaging/Yggdrasil/pull/13595>.
- `R3D_C.__init__` resolves the library in this order:
  `ENV["R3D_LIB"]` (override) → `r3d_jll` (when available) → warn.
- Real UUIDs and `description` fields populated in all `Project.toml`s.

The session that produced this release added, in order:

- **Phase 1 (caller-allocated buffers)** — `FlatPolytope` owns its
  scratch (`sdists`, `clipped`, `emarks`, moment `Sm/Dm/Cm`); the
  polytope itself is the reusable buffer. `init_box!` 22 ns / 0 allocs;
  full pipeline 795 ns at **1.30× C** (down from 7.3× C).
- **Phase 3 (clip kernel micro-tuning)** — 3-way unrolled `find_back3`,
  branchless `next3`. Clip 8 random planes at **1.01× C**.
- **Phase 4 (3D voxelization)** — `split_coord!`, `get_ibox`,
  `voxelize!`, `VoxelizeWorkspace`. Within 1.27–1.42× C across grids
  4³–32³ and orders 0–2. Heavy differential testing (150k+ per-voxel
  comparisons vs `r3d_voxelize`).
- **Phase 6 (D=2 + 2D rasterization)** — full port of `r2d.c` and
  `v2d.c`. 2D voxelize at **0.97–0.99× C** (Julia at parity).
- **Phase 2 (`StaticFlatPolytope`)** — `MMatrix`-backed small-cap variant.
  `init_box!` 17.5 ns (down from 21.6 ns).
- **Additional ops** — `split!`, `is_good`, `shift_moments!` for D=2
  and D=3.
- **`r3d_jll`** BinaryBuilder recipe pinned to upstream HEAD; PR
  pending against JuliaPackaging/Yggdrasil.

176,146 tests passing.

[Unreleased]: https://github.com/yipihey/r3djl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yipihey/r3djl/releases/tag/v0.1.0
