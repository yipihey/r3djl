# Phase 3 status: D ≥ 4 clip + moments

The HierarchicalGrids overlap-layer prompt asks `R3D.jl` to handle
polytopes in `D = 4, 5, 6` so that cubic-edge dimension lifting can
keep the overlap moments exact. This page tracks what's landed, what's
explicitly stubbed, and the path to a complete port.

## Status (2026-04-25)

| Capability                                 | D=2,3 | D≥4               |
|--------------------------------------------|:-----:|:------------------|
| `init_box!` / `init_simplex!`              | ✓     | ✓ (vertices+pnbrs only — no `finds`) |
| `num_moments(D, P)`                        | ✓     | ✓ (`binomial(D+P, P)` for any `D`) |
| `clip!`                                    | ✓     | ⛔ (informative stub) |
| `moments` / `moments!`                     | ✓     | ⛔ (informative stub) |
| Differential validation vs C `rNd`         | ✓     | ⛔ (needs separate libr3d build per D) |

`init_box!` and `init_simplex!` for `D ≥ 4` are wired up and tested:
they construct the right vertices and the bit-hack / cyclic neighbour
pnbrs, but they don't yet populate the `finds[D][D]` 2-face table that
the C `rNd_clip` linker walks. Until that table exists, `clip!` and
`moments!` for `D ≥ 4` raise an informative error pointing here.

## What's actually involved

### 1. Carry the `finds[D][D]` 2-face table

`rNd_vertex` in `src/rNd.h` carries an additional `finds[RND_DIM][RND_DIM]`
array per vertex — the index of the 2-face containing the half-edge to
neighbour `a` and the half-edge to neighbour `b`. In `D = 3` 2-faces
coincide with edges, which is why the existing `find_back3` shortcut
works; for `D ≥ 4` you genuinely need the table.

This means `FlatPolytope` needs a new field along the lines of
`finds::Array{Int32, 3}` (sized `D × D × capacity`), populated by
`init_box!` and `init_simplex!` for `D ≥ 4`, propagated through
`clip!`, and copied / resized by the existing buffer machinery.

The C `rNd_init_box` (lines 548–575) and `rNd_init_simplex` (lines
505–546) are short and mechanical — they enumerate `(np, np1)` axis
pairs and assign `nfaces` IDs.

### 2. Port `rNd_clip`

`src/rNd.c:26–171`. The structure mirrors `r3d_clip`: signed
distances → trivial accept/reject → vertex insertion on cut edges →
**face linking via the `finds[][]` table** → compaction.

The face-linking step (lines 102–155) is the genuinely D-generic piece
and the one missing in the current `R3D.Flat` clip kernels. It walks
around the boundary of each new 2-face and patches up `pnbrs` and
`finds` entries.

About 150 lines of careful porting.

### 3. Generalize `moments!` to `D ≥ 4`

This is harder than it looks. The existing 3D Koehl recursion in
`flat.jl:moments!` walks each 2-face of the polytope and triangulates
it from a base vertex; the trinomial Pascal pyramid then handles the
per-triangle moment integration.

For `D ≥ 4` the analogous decomposition is recursive: a `D`-polytope
is split into `D`-simplices anchored at one vertex via a recursive
descent through `(D-1)`-, `(D-2)`-, …, 2-faces. The `rNd_reduce`
function in `src/rNd.c:252–315` does this for the **0-th moment only**
(it's not generalized to higher orders in upstream either — the C
function literally only writes `moments[0]`).

So this work splits into two pieces:

- **0-th moment (volume)**: port `rNd_reduce` and validate
  differentially vs C.
- **Higher orders**: implement Lasserre's recursive decomposition
  (Lasserre, J.B. *Integration on a convex polytope*, Proc. AMS 1998)
  or a direct generalization of Koehl. Closed-form check on unit
  simplex / unit box.

### 4. Differential testing infrastructure

`libr3d` is currently built once with `RND_DIM = 3` baked in.
`rNd_clip` and friends use a different compile-time `RND_DIM`. To
diff-test `D = 4`, `5`, `6` we need three separate shared libraries:

```bash
gcc -O3 -fPIC -shared -DRND_DIM=4 rNd.c -lm -o libr3d_4d.dylib
gcc -O3 -fPIC -shared -DRND_DIM=5 rNd.c -lm -o libr3d_5d.dylib
gcc -O3 -fPIC -shared -DRND_DIM=6 rNd.c -lm -o libr3d_6d.dylib
```

`R3D_C.jl` then needs parallel `Poly4{N}`, `Poly5{N}`, `Poly6{N}`
struct mirrors and `clip4!` / `reduce4!` / etc. wrappers per
dimension, each pointing at its own `dlopen`'d library.

Once that's in place the differential test pattern from
`test/runtests.jl` extends mechanically: 100 random clips of a
unit box per dimension, agreement at `1e-10` against the C output.

## Suggested sequencing for the implementing agent

1. **Carry `finds`** — add the field, allocate it lazily for `D ≥ 4`,
   populate in `init_box!` / `init_simplex!`. Add a `finds_view(poly, v)`
   accessor for legibility.
2. **Build `libr3d_4d.dylib`** and wire `R3D_C.jl` to expose
   `R3D_C.clip4!` / `R3D_C.reduce4!`. Validate the constructor
   produces the same vertex set + connectivity as `rNd_init_box(D=4)`.
3. **Port `rNd_clip` for `D = 4`**. Get differential clip tests green
   (pre-clip + post-clip 0-th moment match C `rNd_reduce`).
4. **Generalize**: extend the same code to `D = 5` and `D = 6`. The
   algorithm is dimension-generic in principle; the only change is the
   loop bounds.
5. **Higher-order moments**: port `rNd_reduce` for `moments[0]`, then
   add Lasserre's decomposition for higher orders with closed-form
   simplex/box tests.

The total budget is the prompt's 2–4 weeks — most of it in steps 3 and
5, with the `finds[][]` propagation in step 3 being the trickiest
single piece.

## What works today (no Phase 3 needed)

- The 2D dfmm path uses only `D = 2`. The overlap loop in
  `docs/overlap_example.md` is fully supported, runs zero-alloc, and
  matches floating-point precision against `r2d_clip` /
  `r2d_reduce` / `r2d_rasterize`.
- The 3D dfmm path uses only `D = 3`. Same caveats: every operation is
  tested at floating-point precision against `r3d`.
- `D ≥ 4` `init_box!` / `init_simplex!` are usable for *building* a
  lifted polytope; they just can't be clipped or integrated yet.
