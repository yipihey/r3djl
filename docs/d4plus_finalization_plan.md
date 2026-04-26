# D ≥ 4 Finalization Plan

## Final state (as of `d105356`, 2026-04-26)

D ∈ {4, 5, 6} now reach feature parity with the D = 2 / D = 3 surface
area for everything dfmm needs. The dfmm cubic-edge dimension lifting
pipeline is fully unblocked at any supported dimension and any
polynomial order.

| Capability | D = 2 / 3 | D = 4 | D = 5 | D = 6 |
|---|:-:|:-:|:-:|:-:|
| `init_box!` / `init_simplex!` | ✓ | ✓ | ✓ | ✓ |
| `clip!` / `clip_plane!` (incl. FP-boundary cases on both sides) | ✓ | ✓ | ✓ | ✓ |
| `moments(., 0)` | ✓ | ✓ | ✓ | ✓ |
| `moments(., P ≥ 1)` (Lasserre) | ✓ | ✓ | ✓ | ✓ |
| `voxelize_fold!` / `voxelize!` / `voxelize` at any order | ✓ | ✓ | ✓ | ✓ |
| `split_coord!` | ✓ | ✓ | ✓ | ✓ |
| `box` / `simplex` / `aabb` / `box_planes` / `copy!` | ✓ | ✓ | ✓ | ✓ |
| `affine!` / `rotate!` matrix forms | ✓ | ✓ | ✓ | ✓ |
| `walk_facets` / `walk_facet_vertices` | n/a | ✓ | ✓ | ✓ |
| Differential vs C `rNd` (where applicable) | ✓ | ✓ (order = 0) | ✓ (order = 0) | ✓ (order = 0) |
| `StaticFlatPolytope` | D = 3 only | ⛔ deferred | ⛔ deferred | ⛔ deferred |
| `init_poly!` (general face list) | D = 2 / 3 only | ⛔ deferred | ⛔ deferred | ⛔ deferred |

Closed-form validation: unit D-simplex moments (`α!/(D+|α|)!`) and
unit D-box moments (`∏ 1/(α_j + 1)`) match for all multi-indices
through P = 3 at D ∈ {4, 5}, through P = 2 plus a P = 3 hypercube
spot-check at D = 6.

No C oracle exists for D ≥ 4 P ≥ 1 — upstream `rNd_reduce`'s
higher-order branch is `#else`-blocked off.

## Progress log

| commit | what landed |
|---|---|
| `ad0e230` (2026-04-26) | Pre-plan baseline: D ≥ 4 core hot path working — `init_box!`, `init_simplex!`, `clip!`, `clip_plane!`, `moments(., 0)`, `voxelize_fold!`, `voxelize!`, `voxelize`, `get_ibox`, `aabb`, `volume`, `is_empty`, `walk_facets`, `walk_facet_vertices`, plus differential validation against C `rNd_clip` / `rNd_reduce`. dfmm cubic-edge lifting unblocked at order = 0. |
| `e55931f` | This document. |
| `33daa27` | **Phase B.2 / B.3 / B.4 / B.5** — D ≥ 4 API parity for `box`, `simplex`, `aabb`, `box_planes`, `box_planes!`, `copy!`. |
| `b4975bf` | **Phase E.1** — CI builds `libr3d_4d/5d/6d.dylib`; D ≥ 4 differential testset now actually exercises C upstream (was silently skipping). |
| `755060d` | **Phase A foundation** — `facet_normals` + `facet_distances` fields on `FlatPolytope` populated by `init_box!` / `init_simplex!` / `_clip_plane_nd!` / `_copy_polytope_nd!`. |
| `bd3e0d2` | **Phase A at D = 4** — Lasserre higher-order moments at D = 4 (P ≥ 1), validated to closed-form on simplex + box for P ∈ {1, 2, 3}. |
| `a955ce4` | Symmetric companion to the c4496ab clip!-on-boundary-vertex fix, surfaced via voxelize_fold! at non-power-of-2 grid sizes. |
| `86e51da` | **Phase B.1** — `split_coord!` D ≥ 4 public API (wrapper over `clip_plane!`). voxelize_fold! D ≥ 4 refactored to use it. |
| `dc0c056` | **Phase C** — `affine!` and `rotate!` for D ≥ 4 (D × D linear and (D+1) × (D+1) homogeneous forms). Per-D `@eval` dispatch for Aqua hygiene. |
| `66feec5` | **Phase A at D = 5** — Lasserre recursion lifted one level, validated to closed-form on D = 5 simplex + box for P ∈ {1, 2, 3} (all 56 multi-indices). |
| `d105356` | **Phase A at D = 6** — Lasserre recursion lifted one more level (each 6D facet projects to a 5D polytope and recurses into the D = 5 pass). Validated to closed-form on D = 6 simplex + box for P ∈ {1, 2}, with a P = 3 spot-check on the unit hypercube. |
| `(this push)` | Plan document finalized: replaced "remaining work" sections with completion status, kept architectural notes for the deferred pieces (Phase D, Phase E.2, perf, B.5 `init_poly!`). |

## Phase A — Higher-order moments (P ≥ 1) at D ≥ 4 — **Done**

The headline blocker. `moments(poly, P ≥ 1)` and `voxelize_fold!`
at order ≥ 1 now work for D ∈ {4, 5, 6}.

### Implementation summary

For a D-polytope `P` with facets `Fᵢ`, outward unit normals `nᵢ`,
and signed distances `dᵢ`, Lasserre's formula is

```
∫_P x^α dV  =  (1 / (D + |α|)) Σᵢ dᵢ ∫_{Fᵢ} x^α dA
```

Each per-facet integral is reduced to a (D−1)-dim moment problem by
projecting F's vertices to (D−1)-dim coords via a per-facet
orthonormal basis B (D × (D−1), perpendicular to nᵢ). The (D−1)-dim
problem is then handed off to the next-lower D's higher-moment
machinery, so the whole stack composes:

- D = 4: project to 3D, call existing `moments!(::FlatPolytope{3,T})`
  (which uses the LTD divergence-theorem pass — no recursion needed).
- D = 5: project to 4D, call `_reduce_nd_higher_d4!`.
- D = 6: project to 5D, call `_reduce_nd_higher_d5!`.

The 4D and 5D facet-polytope reconstruction (`_build_facet3d!`,
`_build_facet4d!`, `_build_facet5d!`) populates the in-facet
vertices, edge connectivity (with consistent handedness via det-sign
sorting), facets-of-facet (= codim-2 faces of P, identified via
unordered facet pairs), facet normals/distances within the
projection's tangent plane, and the inherited 2-face `finds` data.

The D=4 → 5 → 6 recursion makes higher-D incremental: each new D
adds one new builder + projection-multinomial helper without
reworking the lower levels. D = 7 would follow the same pattern
(at increased per-call cost from the deeper recursion), but is
out of scope — the C `rNd` oracle caps at D = 6 and we have no
benchmarked use case beyond.

### Validation strategy (no C oracle for P ≥ 1)

- Closed-form unit D-simplex moments: `α!/(D+|α|)!`.
- Closed-form unit D-box moments: `∏ 1/(α_j + 1)`.
- Coordinate-permutation symmetry on coord-symmetric polytopes.
- voxelize_fold! consistency: per-cell moment sum equals
  whole-polytope moment within fp precision.

## Phase B — Infrastructure parity ops — **Mostly done**

### B.1 `split_coord!(poly, out0, out1, split_pos, axis)` for D ≥ 4 — **Done**

Shipped as a thin wrapper over `clip_plane!` (commit `86e51da`).
Public API parity with the lower-D versions (same signature, same
`in === out0` aliasing contract). `voxelize_fold!` D ≥ 4 delegates
to it.

A true single-pass split (sharing the boundary walker between the
two halves) would shave 10-20 % off per-split CPU at D ≥ 4 by
avoiding redundant facet-traversal work, but adds ~250 LOC of
careful 2-face linker code. The wrapper matches the existing
voxelize_fold! cost (no copy is added when `in === out0`) and the
underlying `clip_plane!` is already heap-free and tuned. Replace
with a true single-pass implementation if a future benchmark shows
the linker work is bottlenecking.

### B.2 `box_planes` / `box_planes!` for D ≥ 4 — **Done** (commit `33daa27`)

### B.3 `copy!` public overload for D ≥ 4 — **Done** (commit `33daa27`)

### B.4 `aabb(::FlatPolytope{D ≥ 4, T})` and `aabb(::StaticFlatPolytope{D ≥ 4, …})` — **Done** (commit `33daa27`)

### B.5 Construction conveniences — **Done modulo `init_poly!`** (commit `33daa27`)

`box(lo, hi)` and `simplex(verts...)` for D ≥ 4 ship. `init_tet!` /
`tet()` are left as 3D aliases (dimension-specific by design).

`init_poly!(poly, verts, faces)` for D ≥ 4 deferred — non-trivial
because face → vertex → 2-face connectivity must be consistent with
how `clip!` and `moments!` walk, and dfmm uses
`init_simplex!` / `init_box!` exclusively. Pick this up if a
consumer needs to bootstrap a D ≥ 4 polytope from an explicit face
list.

## Phase C — Affine operations parity — **Done**

`affine!` and `rotate!` ship for D ≥ 4 (commit `dc0c056`):
both D × D linear and (D+1) × (D+1) augmented homogeneous matrices
are accepted. Per-D `@eval`-generated dispatches for Aqua's
unbound-args hygiene; `@generated` per-vertex unrolled kernels for
zero-alloc steady state. Inline orthogonality and determinant
helpers so `LinearAlgebra` stays test-only.

C.3 (D = 2 / D = 3 specializations of `translate!` / `scale!` /
`shear!`) skipped — the existing D-generic versions perform
identically on hot-loop benchmarks.

## Phase D — `StaticFlatPolytope{D, T, N, DN}` for D ≥ 4 — **Deferred**

Currently 3D-only. The static variant uses fixed-size `MArray`
storage (compile-time capacity), enabling stack-resident polytopes
with zero-alloc clip + moments loops. Would require generalizing
`init_box!`, `clip!` / `clip_plane!`, `moments!`, `aabb`, `volume`,
and `is_empty` for the static storage variant at D ≥ 4 (~600 LOC
across the existing static-storage code patterns).

Skipped because dfmm doesn't currently use `StaticFlatPolytope`
even at D = 3, and the dynamic-storage path is already heap-free
in the hot loop (see Phase A foundation + the `voxelize_fold!`
zero-alloc test). Pick this up if a downstream consumer
specifically wants the compile-time-bounded variant.

## Phase E — CI + distribution

### E.1 CI builds `libr3d_{4,5,6}d.dylib` — **Done** (commit `b4975bf`)

The D = 4/5/6 differential testset now actually exercises C upstream
on every CI run (was silently skipping before). The CI workflow
wraps upstream's `rNd.h` with an `#ifndef RND_DIM` guard and
compiles `rNd.c` three times with `-DRND_DIM=N`.

### E.2 `r3d_jll` Yggdrasil recipe — **Deferred**

Yggdrasil build script that produces all four libraries
(`libr3d`, `libr3d_4d`, `libr3d_5d`, `libr3d_6d`) so consumers
don't need to compile per-dimension themselves. Out of scope until
publishing R3D.jl as a registered package — at that point also
needs a Project.toml `r3d_jll` dep and a JLL load that picks the
right per-D library at startup.

## Phase F — Performance tuning — **As-needed**

`_clip_plane_nd!` and `_reduce_helper_nd` use straightforward linear
scans. `_reduce_helper_nd` recurses `D!` times per vertex — fine at
D ≤ 6 but exponential beyond. The Lasserre stack at D = 6 is three
levels deep (5D → 4D → 3D), each per-vertex per-cell, so dfmm's
voxelize-of-large-grid workload will pay an O(D^4) per-cell cost at
D = 6.

Avenues if a benchmark shows a bottleneck:

- `@generated`-function unrolling for small D in the LTD recursion.
- Iterative form with explicit stack instead of recursion.
- Replace `ltd_scratch` with an `MMatrix` that's truly stack-pinned
  (would require inlining the recursion).
- Cache per-facet `B` basis and `(c_F, dimensions of cot-tangent
  space)` across cells of the same axis-aligned grid layer in
  `voxelize_fold!`.

Out of scope until concrete profile data shows where time is spent.

## Out of scope here

- **D > 6 support.** The C `rNd` upstream caps out at 6; our
  diff-test oracle for the 0th moment disappears beyond that. The
  Julia code is dimension-generic in shape, but `_reduce_helper_nd`'s
  `D!`-deep recursion makes D ≥ 7 impractical until Phase F
  optimizations land. dfmm doesn't need D > 6.
- **Allocation-free `StaticFlatPolytope` at D ≥ 4** (Phase D) until
  a benchmarked workload demonstrates need.
- **Removing the multi-plane `clip!(poly, ::Vector{Plane})` API**.
  The single-plane overload is the fast path now, but the
  multi-plane signature is documented external surface; keep both.
- **A true single-pass `split_coord!` at D ≥ 4** (Phase B.1
  optimization) until a benchmark shows the two-clip cost matters.

## Recommended sequencing for follow-on work (if/when needed)

1. `init_poly!` D ≥ 4 — only when a consumer needs to load a
   D ≥ 4 polytope from an explicit face list.
2. `r3d_jll` Yggdrasil recipe — when publishing R3D.jl as a
   registered package.
3. Phase D `StaticFlatPolytope` D ≥ 4 — only on benchmarked need.
4. True single-pass `split_coord!` D ≥ 4 — only on benchmarked need.
5. Phase F perf tuning — only on profile evidence.
