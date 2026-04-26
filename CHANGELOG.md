# Changelog

All notable changes to R3D.jl, R3D_C.jl, and R3DBenchmarks.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### `voxelize_fold!` — basis-agnostic leaf-callback hook

- New `R3D.Flat.voxelize_fold!(callback, state, poly, ibox_lo, ibox_hi,
  d, order; workspace)` for D = 2 and D = 3. Walks the same `r3d_voxelize`
  / `r2d_rasterize` recursion as `voxelize!`, but at each non-empty
  leaf cell calls `state = callback(state, i, j[, k], m)` where `m` is
  a workspace-owned moment-vector view. Lets consumers fuse downstream
  contractions (basis projection, SpMV, weighted accumulation) into
  the leaf step without R3D committing to any basis convention.
- `voxelize!` is now a thin wrapper that calls `voxelize_fold!` with a
  closure that writes `m` into the corresponding column of `dest_grid`.
  The 151k+ existing voxelize differential-vs-C tests pass unchanged
  (refactor is bit-for-bit invariant).
- Callback contract documented: `m` is overwritten on the next leaf
  (consume in callback or copy out); leaf visitation is stack-LIFO,
  not lexicographic; empty leaves are skipped.
- Hot-loop is 0-alloc when the callback is a hoisted closure or named
  function. With a do-block constructed inside `@allocated`, the
  measurement charges JIT compilation cost — a Julia gotcha worth
  documenting; in real consumer code the do-block compiles once.
- New `R3DBenchmarks.bench_voxelize_fold_3d` micro-bench:
  measured speedup is modest (1.0–1.1× across grids 8³–32³ and orders
  0–3) because per-leaf moments computation dominates wall-time. The
  fold's wins are elsewhere — memory (no `nmom × N` intermediate),
  cleaner downstream API, and multi-row SpMV cache reuse.

### Phase 3 (D ≥ 4) — clip + 0th-moment ported

- `FlatPolytope{D,T}` carries a lazily-allocated `finds::Array{Int32,3}`
  and `nfaces::Int` for `D ≥ 4`'s 2-face connectivity table.
  `init_box!` and `init_simplex!` for `D ≥ 4` populate it mirroring
  upstream `rNd_init_box` / `rNd_init_simplex`.
- `clip!(::FlatPolytope{D,T}, planes)` for `D ≥ 4` — full port of
  `rNd_clip` (`src/rNd.c:26–171`) including the 2-face boundary-walk
  linker.
- `moments(poly, 0)` / `moments!(out, poly, 0)` for `D ≥ 4` — port of
  `rNd_reduce`'s LTD recursion, with one safety improvement:
  degenerate Gram-Schmidt steps are skipped instead of producing NaN.
- `R3D_C` exposes per-dimension wrappers: `Poly4{N}`/`Poly5{N}`/`Poly6{N}`
  struct mirrors and `init_box4!`/`clip4!`/`reduce4!`/`new_poly4`
  (and analogous for D=5, D=6). Each loads its own
  `libr3d_{4,5,6}d.dylib` (built with `-DRND_DIM=N` against a
  conditionalized `rNd.h`); load is gated on `ENV["R3D_LIB_4D"]` etc.
- 140 new tests: closed-form D-simplex / D-box / slab volumes at
  D = 4,5,6 and 100 random-clip differential comparisons against C
  `rNd_clip + rNd_reduce` at D = 4 (max relative diff 3.2e-15, well
  under the 1e-10 acceptance bar).

Higher-order moments (P ≥ 1) for `D ≥ 4` remain stubbed (informative
error). See `docs/phase3_status.md` for the path forward (Lasserre or
D-generic Koehl).

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
