# IntExact: pure-integer polytope clipping with exact-rational moments

`R3D.IntExact` is a parallel implementation of the `R3D.Flat` polytope
clipping API in pure-integer arithmetic. Vertices are stored as
shared-denominator integer pairs (`positions_num[k, v] /
positions_den[v]`), and the clip kernel propagates that representation
forward without ever constructing a `Rational` on the hot path.
`volume_exact`, `area_exact`, and `moments_exact` produce exact
`Rational{R}` results: no floating-point rounding anywhere in the
pipeline.

## When to reach for IntExact (and when not to)

Pick IntExact when:

- Your mesh has integer coordinates by construction. Lattice meshes,
  AMR cells indexed at integer multiples of a base spacing, and
  Hilbert-curve indexed cubes all qualify.
- Bit-exact reproducibility matters. Conservative field transfer and
  mesh remap need `sum_cells m_cell == m_whole` to hold *exactly*, not
  approximately, or mass leaks across the mesh boundary at the
  rounding-error scale.
- Your polytopes don't get clipped so deeply that even `BigInt` is
  impractical for your throughput target.

Pick `R3D.Flat` when:

- Coordinates are inherently real (smoothed-particle-hydro positions,
  Lagrangian advection, anything where the input came from a
  floating-point integration).
- You need maximum throughput on D ∈ {2, 3} polynomial moments.
  IntExact's `D in {2, 3}` Koehl recursion is exact-rational
  but slower than Flat's float kernel. (Polynomial moments at
  `D ∈ {4, 5, 6}` are now available via simplex-decomposition,
  see below — and unlike Flat at `D ≥ 4`, they're exact.)
- Maximum throughput beats reproducibility. The integer kernel runs
  GCD reductions and multi-word multiplies that float skips.

## Picking the storage type `T`

The IntExact module docstring summarizes the recommendation:

| input coord                         | clips                  | recommended `T` |
|-------------------------------------|------------------------|-----------------|
| `Int16`, axis-aligned cuts          | many (voxelize-style)  | `Int64`         |
| `Int16`, oblique integer planes     | <= ~3                  | `Int128`        |
| any                                 | unbounded depth        | `BigInt`        |

A few caveats:

- The per-clip GCD reduction is a heuristic, not a hard bound. It
  empirically keeps the bit-width tight for the workloads we care
  about, but adversarial input can defeat it.
- For axis-aligned cuts on integer-vertex polytopes the denominators
  stay 1 forever (the lazy-GCD invariant). The IntExact test suite
  pins this as a regression test, so any change that breaks it would
  fail CI.
- For oblique integer-coefficient cuts, denominators grow but
  empirically stay small. See the bench table at the bottom of this
  doc for what "small" means in practice.
- For arbitrary chains of cuts (e.g. iterated remap onto rotated
  meshes), `BigInt` is the only safe fallback.

## API tour

The public surface is small. Every snippet below is runnable as-is
(`julia --project=<env> -e '<snippet>'`); the expected output sits in a
trailing comment.

### Constructors

```julia
using R3D

# 3D unit box at integer corners.
b3 = R3D.IntExact.box((0, 0, 0), (10, 10, 10))
println(R3D.IntExact.volume_exact(b3))   # 1000//1

# 2D rectangle.
b2 = R3D.IntExact.box((0, 0), (4, 4))
println(R3D.IntExact.area_exact(b2))     # 16//1

# 2D triangle (3 vertices) and 3D tetrahedron (4 vertices).
tri = R3D.IntExact.simplex((0, 0), (3, 0), (0, 4))
println(R3D.IntExact.area_exact(tri))    # 6//1
```

For more control, allocate the buffer yourself and call
`init_box!` / `init_tet!` / `init_simplex!` with explicit `T`:

```julia
using R3D

buf = R3D.IntExact.IntFlatPolytope{3, Int128}(64)
R3D.IntExact.init_box!(buf, [0, 0, 0], [10, 10, 10])
println(R3D.IntExact.volume_exact(buf))   # 1000//1
```

### Clip

`clip!` accepts a single `Plane{D, T}` or any iterable of them. The
plane convention matches `R3D.Flat`: the half-space `{x : n.x + d >=
0}` is retained.

```julia
using R3D

b = R3D.IntExact.box((0, 0, 0), (2, 2, 2))
plane = R3D.Plane{3, Int}(R3D.Vec{3, Int}(1, 1, 1), -3)
R3D.IntExact.clip!(b, plane)
v = R3D.IntExact.volume_exact(b)
println(v, " (", Float64(v), ")")        # 4//1 (4.0)
```

Cross-check against `R3D.Flat` to validate (this is the same
recipe the test suite uses):

```julia
using R3D

intpoly = R3D.IntExact.box((0, 0, 0), (10, 10, 10))
R3D.IntExact.clip!(intpoly,
    R3D.Plane{3, Int}(R3D.Vec{3, Int}(1, 2, 3), -30))
v_int = R3D.IntExact.volume_exact(intpoly)

flatpoly = R3D.Flat.box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
R3D.Flat.clip!(flatpoly,
    [R3D.Plane{3, Float64}(R3D.Vec{3, Float64}(1.0, 2.0, 3.0), -30.0)])
v_flt = R3D.Flat.moments(flatpoly, 0)[1]

println(v_int, "  ", v_flt)               # 500//1  500.0
```

### Areas, volumes, and moments

`area_exact` (D=2) and `volume_exact` (D=3) both accept an optional
accumulator type `R`. Pass `R = BigInt` when the per-edge or
per-tetrahedron numerator products risk overflowing the storage `T`:

```julia
using R3D

b = R3D.IntExact.box((0, 0, 0), (10, 10, 10))
R3D.IntExact.clip!(b,
    R3D.Plane{3, Int}(R3D.Vec{3, Int}(1, 2, 3), -30))
println(R3D.IntExact.volume_exact(b))            # 500//1, Rational{Int}
println(R3D.IntExact.volume_exact(b, BigInt))    # 500//1, Rational{BigInt}
```

`moments_exact(poly, P)` returns every monomial moment `int x^a y^b
z^c dV` with `a + b + c <= P` as a `Vector{Rational{R}}`. The order
matches `R3D.Flat.moments(poly, P)` exactly (lex by total degree).

```julia
using R3D

b = R3D.IntExact.box((0, 0, 0), (2, 2, 2))
R3D.IntExact.clip!(b, R3D.Plane{3, Int}(R3D.Vec{3, Int}(1, 1, 1), -3))
m = R3D.IntExact.moments_exact(b, 1)             # P = 1: [vol, x, y, z]
centroid = (m[2] // m[1], m[3] // m[1], m[4] // m[1])
println(centroid)                                # (61//48, 61//48, 61//48)
```

Use the in-place form `moments_exact!(out, poly, P)` to control the
output element type (and therefore the accumulator type) explicitly,
e.g. `out::Vector{Rational{BigInt}}` for unbounded-precision
accumulation.

### Affine ops

`translate!`, `scale!`, `affine!`, and `rotate!` mutate the polytope
in-place. Translation and integer scale stay heap-free; rational
scale and `affine!` with a denominator do per-vertex GCD reduction.

```julia
using R3D

b = R3D.IntExact.box((0, 0, 0), (1, 1, 1))
R3D.IntExact.translate!(b, (5, 5, 5))
R3D.IntExact.scale!(b, 2)
println(R3D.IntExact.volume_exact(b))            # 8//1

# rotate! accepts only integer-orthogonal matrices: signed permutations
# and reflections. General SO(3) needs irrational entries (use
# R3D.Flat.rotate! for those).
A = [0 -1 0; 1 0 0; 0 0 1]                       # 90 deg about z
R3D.IntExact.rotate!(b, A)
println(R3D.IntExact.volume_exact(b))            # 8//1
```

### Voxelization

`voxelize_fold!` runs the same `r3d`-style bisection recursion as
`R3D.Flat.voxelize_fold!`, but with exact-rational cell moments. The
callback receives one cell at a time:

```julia
using R3D

T = Int64
ws = R3D.IntExact.IntVoxelizeWorkspace{3, T}(64)
poly = R3D.IntExact.IntFlatPolytope{3, T}(64)
R3D.IntExact.init_box!(poly, [0, 0, 0], [4, 4, 4])

total = R3D.IntExact.voxelize_fold!(0 // 1, poly,
    (0, 0, 0), (4, 4, 4),
    (T(1), T(1), T(1)), 0;
    workspace = ws) do acc, idx, m
    acc + m[1]
end
println(total)                                   # 64//1
```

Allocate the workspace once and reuse it across many calls — the
inner bisection loop is heap-free apart from the per-call exact-
rational moment scratch.

## End-to-end remap-style example

This is the use case IntExact was built for: source mesh against
target mesh, exact intersection volume per pair. We take a 4x4x4
integer-vertex grid, translate every source cell by `(1, 0, 0)`, and
sum the exact intersection volumes against every target cell. The
overlap region is `[1, 4] x [0, 4] x [0, 4]`, total volume `3 x 4 x 4
= 48` exactly:

```julia
using R3D

function remap_overlap()
    T = Int64
    total   = zero(Rational{T})
    n_pairs = 0
    for sx in 0:3, sy in 0:3, sz in 0:3
        for tx in 0:3, ty in 0:3, tz in 0:3
            src_lo = (T(sx + 1), T(sy),     T(sz))
            src_hi = (T(sx + 2), T(sy + 1), T(sz + 1))
            tgt_lo = (T(tx),     T(ty),     T(tz))
            tgt_hi = (T(tx + 1), T(ty + 1), T(tz + 1))
            if src_hi[1] <= tgt_lo[1] || src_lo[1] >= tgt_hi[1] ||
               src_hi[2] <= tgt_lo[2] || src_lo[2] >= tgt_hi[2] ||
               src_hi[3] <= tgt_lo[3] || src_lo[3] >= tgt_hi[3]
                continue
            end
            poly = R3D.IntExact.box(src_lo, src_hi)
            planes = [
                R3D.Plane{3, T}(R3D.Vec{3, T}( 1, 0, 0), -tgt_lo[1]),
                R3D.Plane{3, T}(R3D.Vec{3, T}(-1, 0, 0),  tgt_hi[1]),
                R3D.Plane{3, T}(R3D.Vec{3, T}( 0, 1, 0), -tgt_lo[2]),
                R3D.Plane{3, T}(R3D.Vec{3, T}( 0,-1, 0),  tgt_hi[2]),
                R3D.Plane{3, T}(R3D.Vec{3, T}( 0, 0, 1), -tgt_lo[3]),
                R3D.Plane{3, T}(R3D.Vec{3, T}( 0, 0,-1),  tgt_hi[3]),
            ]
            R3D.IntExact.clip!(poly, planes)
            if poly.nverts > 0
                total += R3D.IntExact.volume_exact(poly)
                n_pairs += 1
            end
        end
    end
    return total, n_pairs
end

total, n_pairs = remap_overlap()
println("non-empty pairs = ", n_pairs)            # 48
println("total overlap   = ", total)              # 48//1
```

The intersection volumes per pair are integers here only because the
cuts are axis-aligned. Replace any plane with an oblique normal and
the per-pair `total` would carry exact rational fractions; the global
sum would remain integer (mass conservation).

## Limitations called out explicitly

- Polynomial moments at `D ∈ {2, 3}` use the divergence-theorem Koehl
  recursion (a direct exact-rational port of `R3D.Flat.moments!`).
  Polynomial moments at `D ∈ {4, 5, 6}` use simplex decomposition:
  fan-triangulate into D-simplices via the same recursive walker that
  `volume_exact` uses, and on each leaf evaluate the closed-form
  `∫_S x^α dV = |det(A)| · ∫_{Δ_D} (v_0 + A·t)^α dt` with the
  multinomial expansion of `(v_0 + A·t)^α` integrated termwise via
  the Dirichlet identity `∫_{Δ_D} t^β dt = β!/(D + |β|)!`. Sqrt-free,
  exact-rational throughout — and unlike Flat's Lasserre at `D ≥ 4`
  (which orthonormalizes via `sqrt` and accumulates significant FP
  error on clipped polytopes), the IntExact moments are bit-exact by
  construction.
- Volume and polynomial moments at `D ∈ {4, 5, 6}` use a generic
  recursive enumeration of the facet-intersection lattice. Cost
  scales as `O(M^D)` where `M = poly.nfacets`, plus per-simplex
  `O((P+1)^D)` for moment evaluation at order P. Fine at unit-test
  scale, slow for production hot loops at high `D` and `P` (a future
  optimization could memoize the polynomial-in-`t` representations
  shared across α values).
- `D = 4` `clip!` does not ε-nudge boundary vertices (the float
  `R3D.Flat.clip_plane!` does, with `eps(T) * 256`). When two
  sequential axis-aligned cuts intersect exactly at a vertex of the
  polytope (`sd_num[vcur] == 0`), the integer cut formula produces
  two distinct vertex instances at the same geometric position (one
  per cut facet meeting there). `volume_exact` handles this via a
  pre-pass that builds a canonical-vertex map and unions in-facet
  adjacencies — so axis-aligned voxelization-grid workloads that hit
  exact corner coincidences give the correct volume (the original
  bug-report case `v_np == v_pn` now holds bit-exactly). Other
  IntExact operations (`moments_exact`, `voxelize_fold!`) at `D >= 4`
  would need analogous canonical-pass treatment when added — the
  D = 3 path doesn't expose this since its polynomial-moments
  recursion never enumerates 2-faces directly.
- `rotate!` accepts only integer-orthogonal matrices. The assertion
  `A * A' == I` triggers if you pass anything else; use
  `R3D.Flat.rotate!` for general `SO(D)` rotations (their entries
  contain irrationals and aren't representable in IntExact).
- The per-clip GCD reduction is heuristic. It dramatically slows the
  bit-width growth in practice, but there is no closed-form bound on
  numerator/denominator size across arbitrary cut chains. When in
  doubt about overflow, run with `T = BigInt` (or `R = BigInt`
  accumulator on the moment / volume calls).

## What does the overflow look like in practice?

Output of `R3D.jl/bench/intexact_overflow.jl`, which clips a
`[0, 10]^3` cube against eight oblique integer-coefficient planes and
records the maximum vertex bit-width per storage type:

```
| T       | depth_reached | max_num_bits | max_den_bits | status                 |
|---------|---------------|--------------|--------------|------------------------|
| Int32   |             8 |            8 |            6 | ok                     |
| Int64   |             8 |            8 |            6 | ok                     |
| Int128  |             6 |            7 |            4 | ok                     |
| BigInt  |             8 |            8 |            6 | ok                     |
```

The numbers are deliberately undramatic: at `N = 8` clips with
small-integer coefficients, even `Int32` survives without overflow.
The point is the methodology, not the conclusion. Re-run the harness
with your consumer's actual clip-coefficient distribution and depth
to get a relevant safety margin. (`Int128`'s lower `depth_reached`
here reflects one seed where the polytope was wholly clipped away
before reaching `N`; it's not a `T` fault — see the harness's
`:polytope_emptied` status comment.)
