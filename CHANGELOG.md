# Changelog

All notable changes to R3D.jl, R3D_C.jl, and R3DBenchmarks.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Base.show` for `FlatPolytope` and `StaticFlatPolytope` (pretty-printing
  in the REPL).
- `translate!`, `scale!`, `rotate!`, `shear!`, `affine!` for D=2 and D=3
  (mirror upstream `r3d_translate` / `r3d_rotate` / `r3d_scale` /
  `r3d_shear` / `r3d_affine` and the 2D analogs).
- `init_simplex!` for D=2 (triangle constructor).
- `init_poly!` for D=2 (closed polygon from CCW vertex list) and D=3
  (vertex + face list); mirrors `r2d_init_poly` / `r3d_init_poly`.
- `init_brep!` (D=3): boundary-representation initializer mirroring
  `r3d_init_brep`.
- AD compatibility (`ForwardDiff`) verified end-to-end.
- Multi-threaded batched voxelization helper.
- Documenter.jl docs site published to gh-pages.
- `examples/` directory with `voxelize_clipped_tet.jl`,
  `ad_shape_optimization.jl`, `rasterize_2d.jl`, `monte_carlo_volume.jl`.
- GitHub Actions CI matrix (Julia LTS / 1 / nightly × Linux / macOS).
- `Aqua.jl` quality testset.

### Changed
- All three `Project.toml`s now have real `uuid`s; descriptions populated.
- `R3D_C.jl` prefers `using r3d_jll: libr3d` when available; falls back
  to `ENV["R3D_LIB"]`.

## [0.1.0] — TBD

Initial public release. The session that produced this release added,
in order:

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
