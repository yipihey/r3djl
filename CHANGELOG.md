# Changelog

All notable changes to R3D.jl, R3D_C.jl, and R3DBenchmarks.jl are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Hot loop heap-free at D ≥ 4: `voxelize_fold!` allocations 286 KB → 0

Steady-state per-call heap budget for `voxelize_fold!` at D ≥ 4 is
now **0 bytes**, down from ~286 KB on the existing
unit-D=4-box-over-4×4×4×4-grid benchmark (and ~5.8 MB on the first
call due to per-test-closure recompilation that masked the steady-
state cost). Three fixes:

- **Closure-boxed `spax` / `split_index`** in the bisection loop:
  the `ntuple(k -> k == spax ? ... : ..., Val(D))` calls captured
  mutable locals from the enclosing scope, forcing Julia to box
  them. Factored each into a small `@inline` helper
  (`_argmax_extent`, `_axis_unit`, `_replace_at`,
  `_shifted_index`) that takes its dependencies as parameters, so
  the closures don't escape the helper's frame. Fix accounts for
  ~245 KB of the 286 KB.
- **`MVector{D,Bool}` `processed` set in `_reduce_helper_nd`'s
  recursion**: each recursive call's mutation of
  `processed[i] = true; ...; = false` for backtracking forced the
  MVector onto the heap. Replaced with a `UInt32` bitfield
  (immutable by-value scalar; backtracking is implicit because the
  caller's value is unchanged). Removes the recursion's per-call
  allocation entirely.
- **`MMatrix{D,D,T}` LTD scratch in `_reduce_nd_zeroth!`**:
  Julia's escape analysis can't keep a stack-resident MMatrix
  alive across the recursive `_reduce_helper_nd` call boundary.
  Moved the scratch onto a new `ltd_scratch::Matrix{T}` field on
  `FlatPolytope` (lazy-allocated in the constructor for D ≥ 4,
  empty otherwise). Helper signature relaxed to take a
  `Matrix{T}` instead of `MMatrix{D,D,T}`. Last 144 bytes go to 0.

The existing test of the steady-state alloc budget on a
4×4×4×4 voxelize is now `@test a == 0`. The alloc test was wrapped
in a function so the closure type stays consistent across warmup
and measurement (each toplevel `do ... end` is a distinct
anonymous-function type that triggers fresh compilation inside
`@allocated`, swamping the actual hot-loop cost — that's how the
~5.8 MB number arose).

### Bug fix: D ≥ 4 sequential `clip!` corrupts simplex polytopes

Sequential `clip!` calls on a D ≥ 4 simplex (or any polytope whose
vertices happen to land exactly on a later cut plane) produced wrong
volumes — symmetry-related quadrants disagreed and the four-quadrant
decomposition of a unit D = 4 simplex summed to ~94 % of 1/24
instead of exactly 1/24. The same workflow on a D ≥ 4 box was
correct, so single `clip!` calls and box-only sequential clips were
unaffected.

The root cause is in `_clip_plane_nd!` Step 3: when a kept vertex
`vcur` lies exactly on the cut plane (`sdists[vcur] == 0`), the
weighted-average cut-position formula
`(vnext * sd_vcur - vcur * sd_vnext) / (sd_vcur - sd_vnext)`
collapses to `vcur`. New vertices coincident with `vcur` break the
simple-polytope invariant that the LTD moments recursion in
`_reduce_helper_nd` depends on (each vertex must have D linearly
independent outgoing edges). The upstream C `rNd_reduce` has the
same issue and produces NaN on the same inputs.

The fix nudges `sd_vcur` up by a tiny tolerance
(`eps(T) * 256 * span`) for the cut-position computation only,
shifting the new vertex ε-close-but-distinct from `vcur`. The
combinatorial topology is unchanged; the volume error is O(eps(T))
per cut, well below floating-point precision in any realistic
moments computation. Random clip planes never trigger the nudge
(measure-zero `sd_vcur == 0` event), so the existing 100-trial
differential tests against the C library remain bit-exact.

This also fixes the corresponding `voxelize_fold!` / `voxelize!`
path: pre-fix, voxelizing a unit D = 4 simplex over a 2^4 grid
returned a strictly increasing 0, 1/384, 2/384, 3/384, 4/384
volumes for the five non-empty cells, summing to 0.026 ≠ 1/24.
Post-fix the four corner cells all equal 1/384 and the origin cell
holds the bulk, summing to exactly 1/24.

Regression test: `D ≥ 4 sequential simplex clips preserve volume +
symmetry` (11 assertions covering the four-quadrant decomposition
of a unit D = 4 simplex plus the corresponding 2^4 voxelize_fold!
cells).

### Facet ((D−1)-face) tracking foundation for D ≥ 4

Universal building block for the deferred higher-order moments work.
Standard analytic moment schemes (Lasserre's recursive face
decomposition, simplex-fan triangulation, divergence-theorem
recursion) all need facet enumeration, which the existing data
structure didn't expose: `pnbrs` gives 1-faces, `finds` gives
2-faces, but nothing tracked codim-1.

- New fields on `FlatPolytope{D,T}`: `facets::Matrix{Int32}` (lazy
  `D × capacity` for D ≥ 4, empty placeholder for D ≤ 3) and
  `nfacets::Int`. `facets[k, v]` = ID of the facet OPPOSITE edge slot
  k of vertex v. Sentinel 0 = unset.
- `init_box!(::FlatPolytope{D,T})` for D ≥ 4: `2D` facets indexed
  `(2k-1, 2k)` for the (lo, hi) sides of axis k. Each vertex's
  facets[k, v] determined by its bit k.
- `init_simplex!(::FlatPolytope{D,T}, vertices)` for D ≥ 4: D+1
  facets where facet `u` is opposite vertex u. For vertex v,
  facets[k, v] = pnbrs[k, v] (the facet opposite the edge to slot
  k's neighbour is the one opposite that neighbour).
- `_clip_plane_nd!`: each clip creates ONE new facet (the cut
  hyperplane). Step 3 sets `facets[1, new_v] = new_facet_id` for
  every newly-inserted vertex (slot 1 points back to the kept-side
  vcur, so the facet opposite slot 1 IS the cut). Other slots
  inherit from vcur using the same k_new → k_orig mapping as the
  finds row-0 fill. Step 5 compaction copies facets columns
  alongside positions/pnbrs/finds. `nfacets` bumped by 1 per cut.
- `walk_facets(callback, poly)` enumerates each facet ID exactly
  once. `walk_facet_vertices(callback, poly, fid)` calls callback
  for each vertex incident to a given facet (zero-alloc).
- `_copy_polytope_nd!` copies the facets table too (used by D ≥ 4
  voxelize fold/split).

Tests (~30 new): box/simplex closed-form facet counts for D = 4, 5,
6; per-facet vertex-count invariants; single-clip facet propagation
(D=4 box clipped by `x[1] ≥ 0.5` → 9 facets, the new cut facet
contains exactly the 8 newly-inserted vertices); 3 sequential
orthogonal clips bump `nfacets` by 3.

The Lasserre moment recursion on top of facets (closing
higher-order moments at D = 4) is the natural next session's work.
D = 5 / D = 6 need additional intermediate codim-face tracking
layers (same structural pattern).

### `voxelize_fold!` / `voxelize!` extended to D ≥ 4 (order = 0)

- D-generic `get_ibox`, `voxelize_fold!`, and `voxelize!` for D = 4, 5, 6.
  Same callback contract as the lower-dimension versions:
  `state = callback(state, idx::NTuple{D,Int}, m::AbstractVector{T})`
  per non-empty leaf cell.
- Per-split implementation uses a "two-clips" shortcut (copy `cur`
  to `out1`, then in-place clip `cur` for `out0` with the negative-half
  plane and clip `out1` with the positive-half plane) instead of
  porting `split_coord!` to D ≥ 4. ~2× the per-split work of a true
  in-place split, but reuses the already-debugged D ≥ 4 `clip!`
  kernel. Trivial to swap in a real `split_coord!` later.
- Order limitation: only `order = 0` (volume) at D ≥ 4 today —
  higher-order moments need the deferred Lasserre / D-generic Koehl
  port. `order ≥ 1` raises a clear `AssertionError`.
- Closed-form tests at D = 4, 5, 6: unit D-box voxelized over an
  N^D grid sums to 1.0 and every cell equals 1/N^D.

### Bug fix: D ≥ 4 `clip!` linker corrupted `finds[][]` on sequential clips

- The inside-loop branch in the 2-face boundary walker was
  incrementing `nfaces` per traversal step rather than per walk. Each
  walk should commit ONE new face ID and assign it to all the
  patches along the boundary; the upstream C version increments
  `*nfaces` only after the do-while loop. The Julia port now caches
  the new face ID before the walk and increments only at the end.
- Symptom: a single clip on a fresh polytope produced correct results
  (the existing 100-trial diff-vs-C testset stayed green), but a
  second clip on the resulting polytope raised
  `_find_face_in_finds_row: face not in row of vertex`. Voxelization
  exposed this because it does many sequential clips on copies of
  the same polytope.
- New regression test: three orthogonal half-space clips of a unit
  D-box at D = 4, 5, 6 give the expected volumes (0.5, 0.25, 0.125).

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
