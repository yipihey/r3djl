# Optimization roadmap

This document plans the next phase of work on `R3D.jl`, ordered by
**expected payoff per hour of work**. Each section states a hypothesis,
the proposed change, the predicted speedup, and the validation criterion
(both numerical correctness and the benchmark we expect to move).

The North Star: get **full pipeline to within 2× of C**, matching what
`reduce` already achieves. Current state (verified): full pipeline is
4.4× C; `reduce` is 2.0× C. The factor-of-2.2 gap is concentrated
entirely in `init_box` and per-call overhead in `clip!`.

## Status

- [x] **Phase 1 — caller-allocated buffers (DONE).** `FlatPolytope`
      now owns its scratch (`sdists`, `clipped`, `emarks`, moment
      `Sm/Dm/Cm`); the polytope IS the reusable buffer.
      `init_box!(buf, lo, hi)`: 22 ns, 0 allocs (target was < 100 ns).
      Full pipeline buffered: 795 ns at **1.30× C** (target was
      < 1.5 μs / ~2.3× C).
- [x] **Phase 3 — clip kernel micro-tuning (DONE).** 3-way unrolled
      `find_back3` and branchless `next3` dropped clip times another
      5–17%. `clip!`(8 random) now at **1.01× C**.
- [x] **Phase 4 — voxelization (DONE).** `split_coord!`, `get_ibox`,
      `voxelize!` ported from `src/v3d.c`. `VoxelizeWorkspace` keeps
      hot loops 0-alloc. Within **1.27–1.42× C** across grids 4³–32³
      and orders 0–2; 150k+ per-voxel diff tests against C all pass.
- [x] **Phase 6 (D = 2 half) + 2D voxelization (DONE).** Full port of
      `r2d.c` and `v2d.c` to `FlatPolytope{2,T}`. Buffered clip+reduce
      at 1.08–1.44× C; 2D voxelize at 0.97–1.16× C (often faster).
      16,949 differential tests pass.
- [x] **Phase 2 — `StaticFlatPolytope{D,T,N}` (DONE).** MMatrix-backed
      small-cap variant. `init_box!` 17.5 ns (down from 21.6 ns; ~20%
      faster). Full pipeline within noise of `FlatPolytope` (the
      remaining wall time lives in clip/moments loops, not array
      access). 411 cross-validation tests vs `FlatPolytope` pass.
- [x] **Phase 5 — `r3d_jll` BinaryBuilder recipe (DONE).** Recipe at
      `r3d_jll/build_tarballs.jl` pinned to upstream HEAD, ready to
      submit to Yggdrasil under `R/r3d/`.
- [x] **Additional polytope ops (DONE).** `split!` (D=2+D=3),
      `is_good` (D=2+D=3+StaticFlatPolytope), `shift_moments!`
      (D=2+D=3) ported from r3d.c / r2d.c. 2,721 differential tests
      vs C all pass.
- [ ] Phase 6 (D ≥ 4 half) — `finds[][]` 2-face table from `rNd.c`.
      Deferred: narrow user appeal, and validation requires a separate
      libr3d build with `RND_DIM=4` not currently available.

See `docs/performance.md` for the post-Phase-1+3 measurements. The
sections below are the original plan, kept for historical context.

## Phase 1: Caller-allocated buffers (highest leverage)

**Hypothesis.** The 113× Flat/C ratio for `init_box` is dominated by two
`Matrix` allocations (positions: 3×512×8B = 12 KB; pnbrs: 3×512×4B =
6 KB). Removing those moves us to roughly C-ish overhead.

**Change.** Add a `FlatBuffer{D,T,N}` type that owns the storage and is
reused across calls:

```julia
struct FlatBuffer{D,T,N}
    positions::MMatrix{D,N,T}    # stack-allocated for small N
    pnbrs::MMatrix{D,N,Int32}
end

# OR for larger N:
mutable struct FlatBuffer{D,T}
    positions::Matrix{T}
    pnbrs::Matrix{Int32}
    capacity::Int
end

# API: caller-allocates once, reuses across calls
buf = FlatBuffer{3,Float64}(512)
for cell in cells
    poly = init_box!(buf, cell.lo, cell.hi)   # reuses buf, returns view
    clip!(poly, planes)
    moments!(out, poly, 1)
end
```

This is exactly how the C library is used in practice — callers pass in
a `r3d_poly *poly` they've already allocated. Mirroring that pattern
eliminates the per-call alloc cost.

**Predicted impact.**
- `init_box` from 1168 ns → ~50–100 ns (remaining cost is the eight
  vertex-position writes and the pnbrs table copy)
- `clip!` per call drops by ~150 ns (no per-call sdists/clipped allocs;
  these become buffer fields too)
- Full pipeline from 2882 ns → ~1500 ns (~2.3× C)

**Validation.**
- Add `buf` reuse to all benchmark scenarios; allocations should drop to
  near zero on repeat calls.
- Differential test: thousand random clips with a single reused buffer
  must agree with C (catches any state-leakage between calls).
- Checked: `init_box!(buf, lo, hi)` returns a `FlatPolytopeView` rather
  than a fresh allocation; `clip!`, `moments!` accept the view.

**Effort.** ~4 hours: ~150 LOC of new types + buffer-aware methods, plus
threading the changes through the bench suite. The clip kernel itself
needs no changes — it already takes a `FlatPolytope`-like argument.

## Phase 2: Stack-allocated polytopes for small caps

**Hypothesis.** When `R3D_MAX_VERTS` is small enough (≤ 32), the entire
polytope fits in `MMatrix` and lives on the stack. Useful for grid
voxelization where each cell starts as a simple box.

**Change.** Parameterize `FlatPolytope` over `N` (capacity) at the type
level when `S = StaticStorage{N}`:

```julia
struct StaticFlatPolytope{D,T,N}
    positions::MMatrix{D,N,T,Tnp}
    pnbrs::MMatrix{D,N,Int32,Tnp2}
    nverts::Int
end
```

For `D=3, N=32, T=Float64`, the struct is 768 + 384 + 8 = 1160 bytes —
fits comfortably in L1.

**Predicted impact.**
- `init_box` for `N=32`: ~10 ns (matches C, since stack alloc is free).
- `clip!` for cube + 1 plane on `N=32` cap: ~100 ns total — within 1.1×
  of C.
- Tradeoff: each method now specializes on `N`, so compile time grows
  if many `N` values are used. In practice users pick one cap.

**Validation.**
- Numerical: all existing differential tests pass with `StaticFlatPolytope{3,Float64,32}`.
- Allocation: `@allocated` for `init_box!(::StaticFlatPolytope)` returns 0.
- Compile time: `@time @eval init_box!(...)` for a fresh `N` should stay
  under 0.5 s.

**Effort.** ~3 hours, mostly type plumbing. Algorithm code is
copy-paste from `flat.jl` with type annotations adjusted.

## Phase 3: Profile-driven `clip!` micro-tuning

**Hypothesis.** With buffers eliminated, `clip!` per-call cost is
dominated by:
1. The `for k in 1:3` linear search for the back-pointer in
   `link_new_vertices!` — a known unrolling target.
2. `mod1(np + 1, 3)` — should compile well but worth checking.
3. The `numunclipped != v` branch in compaction — unpredictable, may
   hurt branch prediction.

**Change.** Pull out a profile, find the actual hotspot, fix what shows
up. Speculation:

```julia
# Replace linear search with a 3-way unroll
@inline function find_back(p::FlatPolytope, vnext::Int, vcur::Int)
    @inbounds begin
        p.pnbrs[1, vnext] == vcur && return 1
        p.pnbrs[2, vnext] == vcur && return 2
        return 3   # must be slot 3 by elimination
    end
end

# Replace mod1(np+1, 3) with a branchless next-index lookup:
@inline next3(np) = np == 3 ? 1 : np + 1
```

**Predicted impact.** 10–30% on `clip!` — small but compounds with
Phase 1. Real numbers needed.

**Validation.**
- `@code_native` of `clip_plane!` should show no `mod` or `div`
  instructions in the inner loop.
- Differential tests still pass.

**Effort.** ~2 hours, profile-driven. May find nothing; in that case
we've confirmed `reduce`'s 2× ceiling is also `clip`'s ceiling.

## Phase 4: Voxelization (`v3d`)

**Status.** Not yet ported. Upstream is `src/v3d.{c,h}`, ~260 LOC.
`v3d_voxelize` takes a polytope and a grid spec, returns a per-voxel
moment field.

**Approach.** Follow the same pattern as `clip!`/`moments!`:

1. Direct port of `v3d_voxelize` to `R3D.Flat.voxelize!` operating on
   `FlatPolytope` + a pre-allocated 3D grid array.
2. Differential tests against `R3D_C.voxelize!` (which I haven't wrapped
   yet — add it in `R3D_C.jl`).
3. Benchmark scenarios: small polytope on small grid; large polytope on
   large grid; PolyClipper-style worst case.

This is the biggest **feature** gap. Without voxelization the port is
useful for quadrature-on-cells but not for mass-conserving remeshing,
which is r3d's headline use case.

**Effort.** ~1 day. Algorithm is well-defined; the work is plumbing
plus differential test setup.

## Phase 5: BinaryBuilder recipe for `r3d_jll`

Once the Julia port is feature-complete, the C wrapper becomes a
testing tool only — but for that role, ergonomics matter. A JLL
removes the "build libr3d locally first" step.

**Effort.** ~3 hours, mostly fighting Yggdrasil's CI. Not blocking
anything else.

## Phase 6: D=2 and D≥4 clip linkers

The current Julia port has `link_new_vertices!` for D=3 only. D=2 has
a placeholder `error()` and D≥4 needs the `finds[][]` 2-face table from
`rNd_clip`.

**D=2 effort.** ~1 hour. The 2D linker is ~30 lines in `r2d.c` and the
algorithm is simpler than 3D (each vertex has 2 neighbours, no face
walk needed).

**D≥4 effort.** ~1 day. The `finds[][]` table propagation is the only
nontrivial part; the rest follows the same pattern as D=3.

## Out-of-scope (for now)

- **Shift-poly for high-order moments.** Not used in any current test;
  defer until a use case appears.
- **Multi-threading.** Single-threaded perf is the prerequisite; SIMD
  within a single clip is constrained by the inherent sequential
  dependency between planes.
- **AD compatibility.** The Flat layout's `Matrix{T}` is friendly to
  ForwardDiff (positions can be `Dual`s); should work today but not
  validated.

## Sequencing

I'd attack these as:

```
Phase 1 (buffers)  ─────────────────┐
                                    │ → benchmark sanity check
Phase 2 (small-N)  ─┐               │
                    │ ← Phase 1 done│
                    ↓               │
Phase 3 (profile)  ─┴───────────────┤ → "we hit the wall" or new fix
                                    │
Phase 4 (v3d)  ────────── ──────────┤ → headline feature
                                    │
Phase 5 (jll), Phase 6 (Dim) ───────┘ → polish
```

Phases 1, 2, 3 all touch the clip kernel and should land together to
avoid merge friction. Phase 4 is independent.

## Decision points

Three numbers I want to see before/after Phase 1, all on the standard
machine:

1. `init_box` allocations: target **0** (down from 11)
2. `init_box` time: target **< 100 ns** (down from 1168)
3. Full pipeline (4 random planes, order=2): target **< 1.5 μs** (down
   from 2882 ns; ~2.3× C)

If we hit those, Phase 1 is a clean win and we can move on. If they're
not met, profile first to find what we missed before adding more code.
