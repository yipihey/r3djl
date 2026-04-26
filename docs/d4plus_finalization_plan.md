# D ≥ 4 Finalization Plan

Status as of `ad0e230` (2026-04-26): D ≥ 4 has the core hot path
working — `init_box!`, `init_simplex!`, `clip!`, `clip_plane!`,
`moments(., 0)`, `voxelize_fold!`, `voxelize!`, `voxelize`,
`get_ibox`, `aabb`, `volume`, `is_empty`, `walk_facets`,
`walk_facet_vertices`, plus differential validation against C
`rNd_clip` / `rNd_reduce`. The dfmm cubic-edge dimension lifting is
unblocked at order = 0.

This document closes the remaining gaps so D ≥ 4 reaches feature
parity with the D = 2 / D = 3 surface area, in priority order.

## Phase A — Higher-order moments (P ≥ 1) at D ≥ 4

The headline blocker. `moments(poly, P ≥ 1)` currently raises an
informative error for D ≥ 4. dfmm's full pipeline (cubic-edge
remap with order ≥ 1 polynomials) needs this.

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

3D version splits a polytope by an axis-aligned plane into two
output polytopes in a single pass (avoids the
`copy_polytope` + two `clip!` shortcut the D ≥ 4 `voxelize_fold!`
currently uses). Same boundary-walker structure as
`_clip_plane_nd!`, but produces both half-spaces in one traversal.
~150 LOC. Drops the `_copy_polytope_nd!` call in
`voxelize_fold!`'s hot loop, halving its memory footprint.

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
