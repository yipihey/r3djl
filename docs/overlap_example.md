# Worked example: triangle ∩ box overlap with moments

This page is a runnable, end-to-end reference for the operation that
`HierarchicalGrids.jl`'s overlap layer (and any other consumer that
intersects two polytopes and integrates polynomial moments) needs from
`R3D.jl`. It also pins down the conventions the user-visible API uses
so callers don't have to reverse-engineer them from the source.

The full script lives at `examples/overlap_triangle_box.jl`; below
we annotate each step.

## What we're computing

Given:
- a Lagrangian triangle `T` with vertices `v₁`, `v₂`, `v₃` (CCW),
- an axis-aligned Eulerian box `B = [x_lo, x_hi] × [y_lo, y_hi]`,

compute the polynomial moments
`∫_{T ∩ B} x^a y^b dV` for all `(a, b)` with `a + b ≤ 3`
(10 numbers, in the graded-lex order documented below).

The closed-form check we use: for the corner case where the box clips
exactly half a triangle along a single axis-aligned line, the
intersection area equals an analytically known fraction of the
triangle's area.

## Step 1: build both polytopes

```julia
using R3D

# Lagrangian triangle (CCW for positive area)
verts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
tri = R3D.Flat.FlatPolytope{2,Float64}(64)
R3D.Flat.init_simplex!(tri, verts)

# Eulerian box
box_lo = (0.25, 0.0)
box_hi = (1.0,  0.5)
box = R3D.Flat.box(box_lo, box_hi)
```

After `init_simplex!`, `tri.nverts == 3`. After `box(...)`, the returned
polytope has `nverts == 4`. Both are reusable buffers with all scratch
storage owned by the polytope; subsequent `clip!`/`moments!` calls are
zero-allocation.

## Step 2: turn the box into clipping planes

`R3D.Plane{D,T}(n, d)` represents the half-space `n·x + d ≥ 0` (the
"positive" side is the one we **keep**; clip discards the negative
side). For a 2D axis-aligned box `[x_lo, x_hi] × [y_lo, y_hi]` the
inside-the-box half-spaces are:

| face        | normal    | offset `d` |
|-------------|-----------|-----------:|
| `x ≥ x_lo`  | `(+1, 0)` | `-x_lo`    |
| `x ≤ x_hi`  | `(-1, 0)` | `+x_hi`    |
| `y ≥ y_lo`  | `(0, +1)` | `-y_lo`    |
| `y ≤ y_hi`  | `(0, -1)` | `+y_hi`    |

Phase 2 ships `R3D.Flat.box_planes(lo, hi)` as a one-liner; you can also
spell it out:

```julia
planes = [
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 1.0,  0.0), -box_lo[1]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(-1.0,  0.0),  box_hi[1]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 0.0,  1.0), -box_lo[2]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 0.0, -1.0),  box_hi[2]),
]
```

## Step 3: clip the triangle (in place)

`clip!` mutates the polytope to its intersection with the supplied
half-spaces. It returns `true` on success, `false` only on capacity
overflow (which the default 64-vertex buffer here will never hit for a
4-plane clip of a triangle).

```julia
ok = R3D.Flat.clip!(tri, planes)
@assert ok
# `tri` now holds the polygon T ∩ B.
# If the intersection is empty, tri.nverts == 0.
```

The polytope is consumed in place. If the consumer needs the original
triangle later, copy it first (`R3D.Flat.copy!(other, tri)` exists in
Phase 2). The recommended hot-loop pattern is to keep one persistent
`work` buffer and re-init it from the source vertices at each
iteration:

```julia
work = R3D.Flat.FlatPolytope{2,Float64}(64)
for (lag_simplex, eul_leaf) in candidate_pairs
    R3D.Flat.init_simplex!(work, lag_simplex.verts)   # 0 alloc
    R3D.Flat.box_planes!(planes_buf, eul_leaf.lo, eul_leaf.hi)  # 0 alloc
    R3D.Flat.clip!(work, planes_buf)                  # 0 alloc
    R3D.Flat.is_empty(work) && continue
    R3D.Flat.moments!(moments_buf, work, 3)           # 0 alloc
    # …consume moments_buf…
end
```

## Step 4: integrate moments to order 3

```julia
m = R3D.Flat.moments(tri, 3)   # length-10 vector
```

For D=2, P=3 the order is the standard graded-lex monomial ordering
with the first index descending within each total order:

| index `m[k]` | monomial      | exponents `(a, b)` |
|-------------:|---------------|--------------------|
|         `1`  | `1`           | `(0, 0)`           |
|         `2`  | `x`           | `(1, 0)`           |
|         `3`  | `y`           | `(0, 1)`           |
|         `4`  | `x²`          | `(2, 0)`           |
|         `5`  | `xy`          | `(1, 1)`           |
|         `6`  | `y²`          | `(0, 2)`           |
|         `7`  | `x³`          | `(3, 0)`           |
|         `8`  | `x²y`         | `(2, 1)`           |
|         `9`  | `xy²`         | `(1, 2)`           |
|        `10`  | `y³`          | `(0, 3)`           |

`R3D.num_moments(2, 3) == 10` and the same `(a + b ≤ 3, a descending)`
pattern generalizes to higher orders. Use the in-place form
`R3D.Flat.moments!(out, tri, 3)` for zero-allocation reuse.

For D=3 (P=3 → 20 moments) the ordering is similarly graded-lex with
`i` descending, then `j` descending, `k = corder - i - j`:
`1, x, y, z, x², xy, xz, y², yz, z², x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³`.

`R3D.num_moments(D, P) = binomial(D + P, P)` for any `D ≥ 1` and `P ≥ 0`.

## Step 5: closed-form check

The right-triangle `T` with vertices `(0,0), (1,0), (0,1)` has area
`0.5`. The box `[0.25, 1] × [0, 0.5]` cuts `T` along `x = 0.25` and
`y = 0.5`. Walking the intersection boundary CCW gives a quadrilateral:

- `(0.25, 0)` — left edge of B meets the bottom of T,
- `(1, 0)` — bottom-right corner of T also lies on B's bottom edge,
- `(0.5, 0.5)` — top of B intersects T's hypotenuse `x + y = 1`,
- `(0.25, 0.5)` — top-left of the inside of B.

By the shoelace formula the area is
`½·|0·(0 − 0.5) + 1·(0.5 − 0) + 0.5·(0.5 − 0) + 0.25·(0 − 0.5)| = 0.25`.

```julia
@assert isapprox(m[1], 0.25; atol=1e-12)
```

## Performance budget

For consumers that intersect O(10⁵) Lagrangian–Eulerian pairs per
timestep, the per-pair budget is a few microseconds with **zero per-pair
heap allocations** when:

- The work polytope is allocated once outside the loop.
- The plane buffer is pre-allocated to the right length.
- The moments buffer is pre-allocated to `R3D.num_moments(D, order)`.
- Only the in-place forms `init_simplex!`, `box_planes!`, `clip!`,
  `moments!` are used inside the loop.

Verified: 209,805 tests pass including allocation-checked hot-loop
tests.
