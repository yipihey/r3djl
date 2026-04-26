# Phase 3 status: D ≥ 4 clip + moments

## Status (2026-04-25)

### Sequential-clip-on-simplex bug fix (this push)

Sequential `clip!` calls on a D ≥ 4 simplex produced wrong volumes
(reported by the dfmm team while exploring cubic-edge dimension
lifting). Root cause: when the cut plane passes exactly through a
kept vertex (`sdists[vcur] == 0`), the cut-position formula in
`_clip_plane_nd!` Step 3 collapses to `vcur`'s position, creating
duplicate vertices that break the simple-polytope invariant the LTD
moments recursion (`_reduce_helper_nd`) assumes. Upstream C
`rNd_reduce` has the same bug and produces NaN.

Fix: ε-nudge `sd_vcur` up by `eps(T) * 256 * span` (where `span =
max(|smin|, |smax|, 1)`) for the cut-position formula only when
it's near zero. New vertices end up ε-close-but-distinct from
`vcur`; combinatorial topology unchanged; volume error O(eps(T))
per cut, only triggered on the measure-zero `sd_vcur == 0` event so
existing diff-tests against C remain bit-exact.

Verified:
- Two-axis quadrant decomposition of a unit D = 4 simplex: all four
  quadrants now respect coordinate symmetry and sum to exactly
  1/24 (was 0.039 = ~94 % of 1/24).
- `voxelize_fold!` of a unit D = 4 simplex over a 2^4 grid: the
  four single-axis corner cells equal 1/384 each, origin cell holds
  the bulk 1/24 - 4/384 (was a strictly increasing 0..4/384
  sequence with origin cell = 0).
- All 33,961 existing tests still pass.

| Capability                                          | D=2,3 | D=4,5,6                  |
|-----------------------------------------------------|:-----:|:--------------------------|
| `init_box!` / `init_simplex!` (incl. `finds[][]`)   | ✓     | ✓                        |
| **Facet (`(D−1)`-face) tracking**                   | n/a   | **✓** (`facets`, `nfacets`)  |
| `num_moments(D, P)`                                 | ✓     | ✓ (`binomial(D+P, P)`)   |
| `clip!` (incl. facet propagation)                   | ✓     | **✓** (port of `rNd_clip` + facet bump per cut) |
| `moments` / `moments!` order = 0 (≡ `volume`)       | ✓     | **✓** (port of `rNd_reduce`) |
| `moments` / `moments!` order ≥ 1                    | ✓     | ⛔ (Lasserre on top of facets — next session) |
| `voxelize_fold!` / `voxelize!`                      | ✓     | ✓ (order = 0; alloc-free) |
| `voxelize` (allocating wrapper)                     | ✓     | ✓                        |
| Single-plane `clip!(poly, plane)` overload          | ✓     | ✓                        |
| Differential validation vs C `rNd`                  | ✓     | ✓ at D = 4 (100 trials, max diff 3e-15) |

### What landed in this push

- `FlatPolytope{D,T}` carries a lazily-allocated `finds::Array{Int32,3}`
  (shape `D × D × capacity` for `D ≥ 4`; empty placeholder for `D ≤ 3`)
  and an `nfaces::Int` counter. `init_box!` and `init_simplex!` for
  `D ≥ 4` populate `finds` mirroring `rNd_init_box` and
  `rNd_init_simplex` respectively.
- `clip!(::FlatPolytope{D,T}, planes)` for `D ≥ 4` ported from
  `src/rNd.c:26–171`, including the 2-face boundary-walk linker. The
  five-step structure mirrors the existing 3D clip (signed distances →
  trivial accept/reject → vertex insertion → face linker → compaction)
  with `finds[][]` propagation in steps 3 and 4.
- `moments` / `moments!` for `D ≥ 4` order = 0 ports the LTD
  recursion from `rNd_reduce` (`src/rNd.c:175–315`), with one safety
  improvement over the upstream: degenerate Gram-Schmidt steps (where
  the orthogonalized vector has zero length) are skipped instead of
  dividing by zero (which produced NaNs in upstream for some inputs).
- `R3D_C.jl` exposes per-dimension wrappers
  (`Poly4{N}`/`Poly5{N}`/`Poly6{N}` struct mirrors and
  `init_box4!`/`clip4!`/`reduce4!` etc.). Each loads its own
  separately-compiled `libr3d_{4,5,6}d.dylib` (built with
  `-DRND_DIM=N`); load is gated on `ENV["R3D_LIB_4D"]` /
  `..._5D` / `..._6D`.
- 140 new tests covering closed-form D-simplex / D-box volumes,
  axis-aligned slab clip volumes, and 100 random-clip differential
  comparisons against C `rNd_clip + rNd_reduce` at D = 4 (max relative
  diff 3.2e-15, well below the 1e-10 acceptance bar). The test infra
  is parameterized so D = 5 and D = 6 are mechanical extensions when
  the corresponding `R3D_LIB_5D` / `R3D_LIB_6D` env vars are set.

### Known upstream gotcha (not a Julia issue)

`rNd.c`'s `rNd_reduce` uses an LTD orthogonalization that divides by
the squared length of a Gram-Schmidt residual. When that residual is
identically zero (which can happen on degenerate axis-aligned inputs
for `D ≥ 5` even on a clean unit simplex), the upstream produces
`NaN`. **Our Julia port guards against this** by skipping the branch
when the residual norm is below `eps(T)`; on every input we tested,
the resulting volume agrees with the upstream when the upstream
produces a finite answer. This is the reason the Julia port should be
the primary moment oracle for `D ≥ 5` even though we have C wrappers.

If the upstream `rNd.h` is included as-is, its `#define RND_DIM 4` will
override any `-DRND_DIM=N` on the build command line. Our build wraps
the upstream header to add `#ifndef RND_DIM ... #endif` guards before
compiling each `libr3d_{4,5,6}d.dylib`. Cross-checked symbols exported
from each library; vertex-struct sizes verified against the C-side
expectations (Vertex4 = 112 bytes, Vertex5 = 160, Vertex6 = 216).

### Validation summary

- D = 4 unit box `volume` = 1.0 ✓
- D = 4 unit simplex `volume` = 1/24 ✓
- D = 5 unit simplex `volume` = 1/120 ✓
- D = 6 unit simplex `volume` = 1/720 ✓
- D = 4 box clipped at `x[1] ≥ 0.5` → 0.5 ✓ (matches C exactly)
- 100 random clips of unit D=4 box → max relative diff vs C: 3.2e-15
- (informally measured) D = 5: max diff 2.3e-13; D = 6: max diff 9.4e-15
  on 100 random clips each.

### nD `voxelize_fold!` finalization (this push)

- **Single-plane `clip!`**: `clip!(poly::FlatPolytope{D,T}, plane::Plane)`
  is now a public overload for every `D`, parallel to the existing
  multi-plane API. Internally it dispatches to a per-dimension
  `clip_plane!`, which for `D ≥ 4` is the renamed `_clip_plane_nd!`.
  This eliminates the per-call `Vector{Plane}` allocation that
  `clip!(poly, [plane])` would otherwise pay.
- **Allocation-free bisection loop**: the `D ≥ 4` `voxelize_fold!`
  no longer constructs `[plane_neg]` / `[plane_pos]` array literals
  per leaf split. Combined with the existing `_copy_polytope_nd!`
  buffer reuse and stack-resident `NTuple` / `SVector` plane
  construction, the steady-state hot loop is heap-free.
- **`voxelize` allocating wrapper for `D ≥ 4`**: parity with the
  `D = 2` / `D = 3` API, so `R3D.Flat.voxelize(poly, d, order)`
  is callable for any supported `D`. Returns `(grid, ibox_lo, ibox_hi)`
  exactly like the lower-D versions.
- **Validation**: per-trial sum-of-leaves equals the polytope's
  moment-based volume on randomly oblique-clipped `D = 4 / 5` boxes
  to floating-point precision; `D`-simplex leaf sums match the
  closed-form `1 / D!`.

### What's still missing for full Phase 3 acceptance

1. **Higher-order moments (P ≥ 1) for D ≥ 4.** Facet tracking
   infrastructure is now in place (`facets`, `nfacets`, `walk_facets`,
   `walk_facet_vertices`) — that was the universal building block.
   Next: implement Lasserre's recursive formula on top, which at
   D = 4 closes the loop using facets + 2-faces. D = 5 needs an
   additional 3-face tracking layer (same structural pattern). Until
   then, `moments(poly, P ≥ 1)` raises an informative error for
   `D ≥ 4`, and `voxelize_fold!` at `D ≥ 4` correspondingly asserts
   `order == 0`.
2. **CI integration of D = 4/5/6 differential tests.** The CI
   workflow doesn't yet build `libr3d_{4,5,6}d.dylib`; the new
   testset gracefully skips when those env vars are absent. Adding
   to `.github/workflows/CI.yml` is a follow-up.
3. **Performance tuning.** The D ≥ 4 `clip!` and `_reduce_helper_nd`
   use straightforward linear scans (no unrolling); `_reduce_helper_nd`
   recurses `D!` times per vertex which is fine at D ≤ 6 but won't
   scale beyond. Out of scope for the current dfmm use case.
4. **`r3d_jll` recipe** to build all four libraries (3D + 4D + 5D + 6D)
   in one go, so consumers don't need to compile per-dimension
   themselves. Out of scope this round.
