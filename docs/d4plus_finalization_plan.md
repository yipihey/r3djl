# D ≥ 4 Finalization Plan

## Progress log

| commit | what landed |
|---|---|
| `ad0e230` (2026-04-26) | Pre-plan baseline: D ≥ 4 core hot path working — `init_box!`, `init_simplex!`, `clip!`, `clip_plane!`, `moments(., 0)`, `voxelize_fold!`, `voxelize!`, `voxelize`, `get_ibox`, `aabb`, `volume`, `is_empty`, `walk_facets`, `walk_facet_vertices`, plus differential validation against C `rNd_clip` / `rNd_reduce`. dfmm cubic-edge lifting unblocked at order = 0. |
| `e55931f` | This document. |
| `33daa27` | **Phase B.2 / B.3 / B.4 / B.5** — D ≥ 4 API parity for `box`, `simplex`, `aabb`, `box_planes`, `box_planes!`, `copy!`. |
| `b4975bf` | **Phase E.1** — CI builds `libr3d_4d/5d/6d.dylib`; D ≥ 4 differential testset now actually exercises C upstream (was silently skipping). |
| `755060d` | **Phase A foundation** — `facet_normals` + `facet_distances` fields on `FlatPolytope` populated by `init_box!` / `init_simplex!` / `_clip_plane_nd!` / `_copy_polytope_nd!`. |
| `bd3e0d2` | **Phase A** — Lasserre higher-order moments at D = 4 (P ≥ 1), validated to closed-form on simplex + box for P ∈ {1, 2, 3}. |
| `a955ce4` | Symmetric companion to the c4496ab clip!-on-boundary-vertex fix, surfaced via voxelize_fold! at non-power-of-2 grid sizes. |
| `(this push)` | **Phase B.1** — `split_coord!` D ≥ 4 public API (wrapper over `clip_plane!`). voxelize_fold! D ≥ 4 refactored to use it. |

This document closes the remaining gaps so D ≥ 4 reaches feature
parity with the D = 2 / D = 3 surface area, in priority order.

## Phase A — Higher-order moments (P ≥ 1) at D ≥ 4

The headline blocker. `moments(poly, P ≥ 1)` currently raises an
informative error for D ≥ 4. dfmm's full pipeline (cubic-edge
remap with order ≥ 1 polynomials) needs this.

**Status (post `755060d`)**: foundation in place — every facet
carries an outward-unit normal `n_F` and signed distance `d_F`
through `init_box!` / `init_simplex!` / `clip!` / `copy!`. The
formula's outer sum can be evaluated; what remains is the inner
`∫_{F} x^α dA` term.

**Remaining work (next session)**:

1. **Multinomial expansion helper.** Compute coefficient of `y^β`
   in `(c + B y)^α` for given α (4D multi-index, |α| ≤ P), β (3D
   multi-index, |β| ≤ |α|), c ∈ ℝ⁴, B ∈ ℝ^{4×3}. ~80 LOC. Pre-
   computes the `K_4 × K_3` mixing matrix once per facet, where
   `K_D = num_moments(D, P)`.

2. **(D−1)-dim facet moments.** Two viable paths:
   - **(a) Reconstruct a CCW-oriented 3D polytope** from each 4D
     facet — extract the in-facet vertices, build a 3D `pnbrs`
     table from the 3-of-4 in-facet slots at each vertex with
     CCW orientation matching the facet's outward normal, project
     positions via `B^T (x − c)`, then call existing `moments!`
     `D = 3`. The CCW orientation is the hard part: `pnbrs[k, u]`
     for the 3D facet must visit the in-facet edges in the order
     induced by the facet's tangent-space orientation, which the
     existing 4D `finds[][]` 2-face data implicitly encodes but
     needs a careful walk to recover.
   - **(b) Direct recursive Lasserre.** At codim 1, compute a
     facet's moments by recursing into its (codim-1-of-facet =
     codim-2-of-polytope = our `finds`) faces. At codim 2, recurse
     into 1-dim edges (vertex pairs); 1-D moments are closed
     form. ~300 LOC but uses only existing connectivity machinery
     and avoids the orientation problem.
   - Recommendation: do (b) — uses what we have, generalizes
     cleanly to D = 5 (we'd recurse one more level using a
     3-face tracking layer), and dodges the "rebuild a 3D
     polytope from a 4D facet with correct CCW pnbrs" problem
     which is non-trivial in its own right.

3. **`moments!` dispatch update.** Replace the `error("order ≥ 1
   not yet implemented")` path in `moments!(::FlatPolytope{D,T})`
   with a call to a new `_reduce_nd_higher!` for `D ≥ 4`, `P ≥ 1`.
   Keep the order-0 fast path (`_reduce_nd_zeroth!`) untouched.

4. **`voxelize_fold!` D ≥ 4** lifts its `@assert order == 0` once
   `_reduce_nd_higher!` is in place.

5. **Validation.** Upstream C `rNd_reduce` only computes
   moments[0] — the higher-moments code in `src/rNd.c` is
   `#if 1 ... #else (no-op) #endif` blocked off — so we have NO
   C oracle for D ≥ 4 P ≥ 1. Instead validate against:
   - **Closed-form unit D-simplex moments**: `∫_{Δ_D} x^α dV =
     α! / (D + |α|)!` where `α! = α_1! α_2! ... α_D!`.
   - **Closed-form unit D-box moments**: `∫_{[0,1]^D} x^α dV =
     ∏_j 1 / (α_j + 1)`.
   - **Symmetry under coordinate swap**: any moment with α
     invariant under a coordinate permutation must equal the
     same moment with the permuted α (e.g. ∫ x_1 dV =
     ∫ x_2 dV on a coordinate-symmetric polytope).
   - **voxelize-fold consistency**: sum-over-voxels of per-cell
     moments equals whole-polytope moment within fp precision.

### Approach: Lasserre's recursive face-decomposition formula

For a D-polytope `P` with facets `Fᵢ` and outward normals `nᵢ`,
moment of monomial `x^α` (multi-index, |α| = k ≥ 1) reduces to
boundary integrals:

```
∫_P x^α dV  =  (1 / (D + |α|)) Σᵢ (nᵢ · cᵢ) ∫_{Fᵢ} x^α dA
```

where `cᵢ` is any point on facet `Fᵢ` (e.g. its first vertex). Each
facet integral is itself a (D−1)-dim moment integral, recursing down
to a closed-form 1-D integral or a 2-D triangle/polygon integral
(handled by the existing 2D `moments!`).

The recursion uses three building blocks:
- **Facet enumeration** (already shipped: `walk_facets`,
  `walk_facet_vertices`).
- **Codim-k face tracking** for k ∈ {2, …, D−2}. We have codim-2
  (`finds`, `nfaces`) and codim-1 (`facets`, `nfacets`). For D = 4
  Lasserre closes the loop using just facets + 2-faces (= edges
  here, since D − 2 = 2). For D = 5 we also need 3-face tracking.
  For D = 6, additionally 4-face tracking. Same structural pattern
  as `facets` (lazy `Array{Int32, k+1}` field on `FlatPolytope`,
  populated in `init_box!`/`init_simplex!`, propagated in
  `_clip_plane_nd!`).
- **Per-facet coordinate frame**: project from D-dim to a
  (D−1)-dim subspace orthogonal to `nᵢ`, using a small
  Gram–Schmidt step (the LTD recursion already does this; reuse).

### Per-D scope

| D | Faces needed | Approx LOC | Notes |
|---|---|---|---|
| 4 | facets + 2-faces (already there) | ~400 | Closes immediately. |
| 5 | + 3-faces | ~700 | Add 3-face tracking layer first. |
| 6 | + 3-faces + 4-faces | ~1000 | Two more layers + recursion. |

dfmm needs D = 5 first. Plan: ship D = 4 to validate the recursion
(closed-form simplex moments matching analytic), then D = 5
(closed-form simplex moments + diff vs C `rNd_reduce` order > 0).

### Deliverables

- `_reduce_nd_higher!` and `_reduce_helper_lasserre` in
  `R3D.jl/src/flat.jl`, replacing the current "order ≥ 1 not
  implemented" error path.
- 3-face tracking layer (struct field + init + clip propagation +
  walker), 4-face layer for D = 6.
- `voxelize_fold!` D ≥ 4 lifts the `@assert order == 0` once
  Lasserre lands.
- New diff-test sweep: 100 random clips at D ∈ {4, 5} with
  `order ∈ {1, 2, 3}`; max-relative-diff vs C `rNd_reduce` < 1e-10.
- Closed-form simplex moments at D = 4, 5 (∫ xᵃ over the unit
  simplex has a known formula in terms of multinomial / binomial
  coefficients).

### Estimated effort

D = 4: 1–2 sessions. D = 5: another 1–2 sessions on top.
D = 6: incremental, can lag.

## Phase B — Missing infrastructure ops

Small, mostly-mechanical extensions. Land before Lasserre so
dfmm's call-site code doesn't need per-D forks.

### B.1 `split_coord!(poly, out0, out1, split_pos, axis)` for D ≥ 4

**Status: shipped as a wrapper over `clip_plane!`.** Public API
parity with the lower-D versions (same signature, same aliasing
contract), and `voxelize_fold!` D ≥ 4 now delegates to it.

A true single-pass split (sharing the boundary walker between the
two halves) would shave 10–20 % off per-split CPU at D ≥ 4 by
avoiding redundant facet-traversal work, but adds ~250 LOC of
careful 2-face linker code. The wrapper matches the existing
voxelize_fold! cost (no copy is added when `in === out0`) and the
underlying `clip_plane!` is already heap-free and tuned. Replace
with a true single-pass implementation if a future benchmark
shows the linker work is bottlenecking.

### B.2 `box_planes` / `box_planes!` for D ≥ 4

Returns the `2D` axis-aligned planes bounding a D-box, used by
`overlap_layer` consumers (HierarchicalGrids). Trivial:
~30 LOC, no algorithmic content. Mirror the 3D signature.

### B.3 `copy!` public overload for D ≥ 4

`R3D.Flat.copy!(dst::FlatPolytope{D,T}, src)` for any D. Internal
`_copy_polytope_nd!` already exists; just expose it. ~10 LOC.

### B.4 `aabb(::StaticFlatPolytope{D,T,N,DN})` for D ≥ 4

Currently only D = 3. Same loop as the existing dynamic-storage
version. ~10 LOC each.

### B.5 Construction conveniences

- `box(lo::NTuple{D,T}, hi::NTuple{D,T})` — generic constructor +
  `init_box!` for any D. Currently only D = 3 has this convenience.
  D = 2 has the same constraint. Make it D-generic. ~20 LOC.
- `simplex(verts)` — analogue of `tet()` for D ≥ 4 simplices. ~15 LOC.
- `init_tet!` / `tet()` — strictly a D = 3 concept; promote to
  `init_simplex!` aliases. No-op for D ≥ 4 (already there).
- `init_poly!(poly, verts, faces)` for D = 4, 5, 6 — non-trivial
  because face → vertex → 2-face connectivity must be consistent
  with how `clip!` and `moments!` walk. Defer until a concrete
  use case demands it (dfmm uses `init_simplex!`/`init_box!`
  exclusively).

### Estimated effort

All of B except B.5's `init_poly!`: one session, ~250 LOC + tests.

## Phase C — Affine operations parity

3D has `rotate!`, `affine!` (general matrix); D ≥ 4 has `translate!`,
`scale!`, `shear!` only.

### C.1 `affine!(poly, A::AbstractMatrix)` for D ≥ 4

Apply a `D × D` linear map (or `D × (D+1)` augmented for
translation). Trivial: loop over `poly.positions`, multiply by `A`.
~20 LOC. Subsumes `rotate!` (caller passes an SO(D) matrix).

### C.2 `rotate!(poly, ::SMatrix{D,D,T})` for D ≥ 4

Thin wrapper over `affine!` enforcing `det = +1` (orthonormal). The
2D scalar-θ and 3D axis-angle signatures are dimension-specific;
for D ≥ 4 only the matrix form makes sense. ~15 LOC.

### C.3 D = 2 / D = 3 specializations of `translate!`/`scale!`/`shear!`

Currently only the D-generic version exists. Performance parity
with the existing 2D/3D code is the only motivation; not blocking
correctness. Defer unless benchmark shows a gap.

### Estimated effort

C.1 + C.2: half a session, ~50 LOC + tests.

## Phase D — `StaticFlatPolytope{D,T,N,DN}` for D ≥ 4

Currently 3D-only. The static variant uses fixed-size `SVector`
storage (compile-time capacity), enabling stack-resident polytopes
with zero-alloc clip + moments loops. Useful for tight inner loops
that don't reuse a workspace.

### Approach

Generalize the existing `StaticFlatPolytope{3,T,N,DN}` to any D ≥ 4
by parameterizing the field shapes. Need:
- `init_box!` for `StaticFlatPolytope{D,T,N,DN}`.
- `clip!` / `clip_plane!` (translate `_clip_plane_nd!` to operate
  on `MVector`/`MArray` storage rather than `Matrix`/`Array`).
- `moments!` (port `_reduce_nd_zeroth!` + Lasserre once it's there).
- `aabb`, `volume`, `is_empty`.

`finds`, `facets`, and the codim-k face fields all need
fixed-capacity static counterparts (`MArray{Tuple{D,D,N},Int32}`
etc.).

### Estimated effort

Mechanical but bulky: ~600 LOC across the existing static-storage
code patterns. One full session per D once the dynamic-storage
counterpart is in place. Lower priority — dfmm doesn't currently
use `StaticFlatPolytope` even at D = 3.

## Phase E — CI + distribution

### E.1 CI builds `libr3d_{4,5,6}d.dylib`

`.github/workflows/CI.yml` currently builds only `libr3d.dylib`
(D = 2 + D = 3). The D = 4/5/6 differential testset gracefully
skips when `R3D_LIB_4D` / `_5D` / `_6D` are unset — meaning we
have no nightly diff-vs-C signal at D ≥ 4 in CI. Add per-D
build steps (mirroring the local `/tmp/libr3d/` layout used during
development) and export the env vars to the test runner. ~50 lines
of YAML. Required before Phase A's diff-test sweep is meaningful.

### E.2 `r3d_jll` recipe

Yggdrasil build script that produces all four libraries
(`libr3d`, `libr3d_4d`, `libr3d_5d`, `libr3d_6d`) so consumers
don't need to compile per-dimension themselves. Out of scope for
the immediate dfmm work; opens the door to publishing R3D.jl as
a registered package without per-user C-toolchain requirements.

### Estimated effort

E.1: half a session. E.2: one session, plus Yggdrasil PR review
turnaround.

## Phase F — Performance tuning (lower priority)

`_clip_plane_nd!` and `_reduce_helper_nd` use straightforward
linear scans. `_reduce_helper_nd` recurses `D!` times per vertex —
fine at D ≤ 6 but exponential. Out of scope until there's a
concrete D ≥ 7 use case or a proven benchmark gap at D = 5/6.

Possible avenues if and when needed:
- `@generated`-function unrolling for small D.
- Iterative form with explicit stack instead of recursion.
- Replace `ltd_scratch` with a `MMatrix` that's truly stack-pinned
  (requires inlining the recursion).

## Recommended sequencing

1. **Phase B.1–B.4** (small infrastructure) — one session, no deps.
2. **Phase E.1** (CI builds D = 4/5/6) — needed to gate Phase A.
3. **Phase A at D = 4** (Lasserre proof-of-concept) — one session.
4. **Phase A at D = 5** (3-face tracking + Lasserre) — one or two
   sessions; unblocks dfmm cubic-edge lifting fully.
5. **Phase B.5 `init_poly!` D ≥ 4** — only when a use case appears.
6. **Phase C** (affine matrix ops) — opportunistic.
7. **Phase D** (static variant D ≥ 4) — only if benchmarked need.
8. **Phase E.2** (`r3d_jll`) — when publishing the package.
9. **Phase F** (perf tuning) — only on benchmark evidence.

After Phases 1–4 the only D = 3 capabilities not in D ≥ 4 are
`StaticFlatPolytope` and the convenience aliases (`tet`,
`init_tet!`), which are dimension-specific by design.

## Out of scope here

- D > 6 support. The C `rNd` upstream caps out at 6; our diff-test
  oracle disappears beyond that. The Julia code is dimension-generic
  in shape, but `_reduce_helper_nd`'s `D!`-deep recursion makes
  D ≥ 7 impractical until Phase F lands.
- Allocation-free StaticFlatPolytope at D ≥ 4 (Phase D) before
  there's a benchmarked workload that needs it.
- Removing the `clip!`-only fallback for `Vector{Plane}` callers;
  the single-plane overload is now the fast path but the
  multi-plane API stays as the documented external surface.
