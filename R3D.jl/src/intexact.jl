"""
    R3D.IntExact

Pure-integer polytope clipping and exact volume computation, parallel to
[`R3D.Flat`](@ref) but with rational arithmetic carried as
shared-denominator integer pairs (no `Rational` allocations on the hot
path). Prototype: `D = 3` only.

Each vertex `v` represents the spatial position
`positions_num[:, v] / positions_den[v]` — a single denominator per
vertex, shared across its `D` coordinate numerators. The clip kernel
propagates this representation forward exactly without ever
constructing a `Rational`. `volume_exact` produces a single
`Rational{T}` at the end via fan-triangulation.

# Type parameters

- `D :: Int` — currently only `D = 3` is supported.
- `T <: Signed` — the integer type for both numerators and denominators.

# Recommended `T`

| input coord | clips | recommended `T` |
|-------------|-------|-----------------|
| `Int16`, axis-aligned cuts | many (voxelize-style) | `Int64` |
| `Int16`, oblique integer planes | ≤ ~3 | `Int128` |
| any | unbounded depth | `BigInt` |

For Int16 inputs through ≥ 4 oblique clips, denominators and 3-fold
numerator products in the `volume_exact` triangulation step can exceed
`Int128` — call `volume_exact(poly, BigInt)` (or use `T = BigInt`
throughout) when in doubt. A periodic GCD reduction inside `clip!`
significantly slows the bit-width growth in practice but is not a
hard guarantee.

# Indexing

1-based `pnbrs` with `0` as the "unset" sentinel — same as
`R3D.Flat.FlatPolytope`.
"""
module IntExact

using StaticArrays: SVector, MVector
using ..R3D: Plane, Vec, num_moments

# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

"""
    IntFlatPolytope{D,T}

Pure-integer polytope buffer. Layout mirrors `R3D.Flat.FlatPolytope`
but with shared-denominator integer position pairs:

```
positions_num :: Matrix{T}    # D × capacity
positions_den :: Vector{T}    # capacity (kept positive)
pnbrs         :: Matrix{Int32}
```

After every `clip!`, the invariant `positions_den[v] > 0 ∀ v ≤ nverts`
holds. The clip kernel does an opportunistic GCD reduction on every
new vertex to keep numerators / denominators tight.
"""
mutable struct IntFlatPolytope{D,T<:Signed}
    positions_num::Matrix{T}    # D × capacity
    positions_den::Vector{T}    # capacity
    pnbrs::Matrix{Int32}        # D × capacity (1-based; 0 = unset)
    nverts::Int
    capacity::Int

    # Per-clip scratch (signed-distance numerators; den is positions_den[v]).
    sd_num::Vector{T}           # capacity
    clipped::Vector{Int32}      # capacity

    # 2-face connectivity for D ≥ 4 (rNd port). Empty for D ≤ 3.
    # `finds[a, b, v]` = `finds[b, a, v]` = the index of the 2-face
    # containing v's edges to neighbours a and b. Sentinel `0` = unset.
    # Used by the D ≥ 4 clip linker (Step 4) and by `volume_exact`'s
    # fan-triangulation to enumerate 2-faces incident to each facet.
    finds::Array{Int32, 3}
    nfaces::Int

    # (D−1)-face ("facet") IDs for D ≥ 4. Empty for D ≤ 3. Same
    # semantics as `R3D.Flat.FlatPolytope.facets`: `facets[k, v]` is
    # the facet OPPOSITE edge-slot k of vertex v. Used by
    # `volume_exact` D ≥ 4 to enumerate facets and their incident
    # vertex sets for the fan triangulation.
    facets::Matrix{Int32}
    nfacets::Int
end

function IntFlatPolytope{D,T}(capacity::Integer = 64) where {D,T<:Signed}
    @assert D >= 2 "IntFlatPolytope: D ≥ 2 required"
    capacity = Int(capacity)
    finds  = D >= 4 ? zeros(Int32, D, D, capacity) :
                      Array{Int32, 3}(undef, 0, 0, 0)
    facets = D >= 4 ? zeros(Int32, D, capacity) :
                      Matrix{Int32}(undef, 0, 0)
    IntFlatPolytope{D,T}(zeros(T, D, capacity),
                         ones(T, capacity),
                         zeros(Int32, D, capacity),
                         0, capacity,
                         Vector{T}(undef, capacity),
                         Vector{Int32}(undef, capacity),
                         finds, 0,
                         facets, 0)
end

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    init_box!(poly::IntFlatPolytope{3,T}, lo, hi) -> poly

Initialize as the axis-aligned integer box with corners `lo` and `hi`
(both `AbstractVector` of integer coordinates). Same vertex labelling
and `pnbrs` table as `R3D.Flat.init_box!`. Every vertex starts with
`positions_den = 1`.
"""
function init_box!(poly::IntFlatPolytope{3,T},
                   lo::AbstractVector, hi::AbstractVector) where {T}
    @assert poly.capacity >= 8
    poly.nverts = 8

    @inbounds begin
        poly.positions_num[1,1] = T(lo[1]); poly.positions_num[2,1] = T(lo[2]); poly.positions_num[3,1] = T(lo[3])
        poly.positions_num[1,2] = T(hi[1]); poly.positions_num[2,2] = T(lo[2]); poly.positions_num[3,2] = T(lo[3])
        poly.positions_num[1,3] = T(hi[1]); poly.positions_num[2,3] = T(hi[2]); poly.positions_num[3,3] = T(lo[3])
        poly.positions_num[1,4] = T(lo[1]); poly.positions_num[2,4] = T(hi[2]); poly.positions_num[3,4] = T(lo[3])
        poly.positions_num[1,5] = T(lo[1]); poly.positions_num[2,5] = T(lo[2]); poly.positions_num[3,5] = T(hi[3])
        poly.positions_num[1,6] = T(hi[1]); poly.positions_num[2,6] = T(lo[2]); poly.positions_num[3,6] = T(hi[3])
        poly.positions_num[1,7] = T(hi[1]); poly.positions_num[2,7] = T(hi[2]); poly.positions_num[3,7] = T(hi[3])
        poly.positions_num[1,8] = T(lo[1]); poly.positions_num[2,8] = T(hi[2]); poly.positions_num[3,8] = T(hi[3])

        for v in 1:8
            poly.positions_den[v] = one(T)
        end

        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(5); poly.pnbrs[3,1] = Int32(4)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(6); poly.pnbrs[3,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(4); poly.pnbrs[2,3] = Int32(7); poly.pnbrs[3,3] = Int32(2)
        poly.pnbrs[1,4] = Int32(1); poly.pnbrs[2,4] = Int32(8); poly.pnbrs[3,4] = Int32(3)
        poly.pnbrs[1,5] = Int32(8); poly.pnbrs[2,5] = Int32(1); poly.pnbrs[3,5] = Int32(6)
        poly.pnbrs[1,6] = Int32(5); poly.pnbrs[2,6] = Int32(2); poly.pnbrs[3,6] = Int32(7)
        poly.pnbrs[1,7] = Int32(6); poly.pnbrs[2,7] = Int32(3); poly.pnbrs[3,7] = Int32(8)
        poly.pnbrs[1,8] = Int32(7); poly.pnbrs[2,8] = Int32(4); poly.pnbrs[3,8] = Int32(5)
    end
    return poly
end

"""
    init_tet!(poly::IntFlatPolytope{3,T}, verts) -> poly

Initialize as the tetrahedron with the four given integer-coordinate
vertices. Same `pnbrs` table as `R3D.Flat.init_tet!`.
"""
function init_tet!(poly::IntFlatPolytope{3,T},
                   verts::NTuple{4,<:AbstractVector}) where {T}
    @assert poly.capacity >= 4
    poly.nverts = 4

    pnbrs = ((2, 4, 3), (3, 4, 1), (1, 4, 2), (2, 3, 1))

    @inbounds for i in 1:4
        for k in 1:3
            poly.positions_num[k, i] = T(verts[i][k])
        end
        poly.positions_den[i] = one(T)
        for k in 1:3
            poly.pnbrs[k, i] = Int32(pnbrs[i][k])
        end
    end
    return poly
end

# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

"""
    clip!(poly::IntFlatPolytope{3,T}, planes) -> Bool

Clip in-place against an iterable of integer-coefficient `Plane{3,T}`s.
Returns `true` on success, `false` on capacity overflow. Same five-step
structure as `R3D.Flat.clip!` (signed-distance numerators → trivial
accept/reject → vertex insertion via integer linear combination →
3-D face-walk linker → compaction), with shared-denominator
arithmetic and per-vertex GCD reduction at the end.

The plane convention is identical: `clip!` retains the half-space
`{x : n · x + d ≥ 0}`. Coordinates of `n` and `d` must be integers
of type `T`.
"""
function clip!(poly::IntFlatPolytope{3,T},
               planes::AbstractVector{Plane{3,T}}) where {T}
    poly.nverts <= 0 && return false
    @inbounds for plane in planes
        ok = clip_plane!(poly, plane)
        ok || return false
        poly.nverts == 0 && return true
    end
    return true
end

"""
    clip!(poly::IntFlatPolytope{3,T}, plane::Plane{3,T}) -> Bool

Single-plane convenience overload.
"""
function clip!(poly::IntFlatPolytope{3,T}, plane::Plane{3,T}) where {T}
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

function clip_plane!(poly::IntFlatPolytope{3,T}, plane::Plane{3,T}) where {T}
    onv = poly.nverts
    sd_num = poly.sd_num
    clipped = poly.clipped
    n1 = plane.n[1]; n2 = plane.n[2]; n3 = plane.n[3]
    δ = plane.d

    # Step 1: signed-distance numerators (denominator = positions_den[v]).
    # Since positions_den[v] > 0 by invariant, sign(sd) = sign(sd_num).
    any_inside  = false
    any_outside = false
    @inbounds for v in 1:onv
        s = n1 * poly.positions_num[1, v] +
            n2 * poly.positions_num[2, v] +
            n3 * poly.positions_num[3, v] +
            δ  * poly.positions_den[v]
        sd_num[v] = s
        if s >= zero(T)
            any_inside = true
            clipped[v] = Int32(0)
        else
            any_outside = true
            clipped[v] = Int32(1)
        end
    end

    # Step 2: trivial accept/reject.
    if !any_outside
        return true
    end
    if !any_inside
        poly.nverts = 0
        return true
    end

    # Step 3: insert new vertices on cut edges.
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:3
            vnext = Int(poly.pnbrs[np, vcur])
            (vnext == 0 || clipped[vnext] == 0) && continue

            poly.nverts >= poly.capacity && return false
            new_idx = (poly.nverts += 1)

            Sa = sd_num[vcur]      # ≥ 0
            Sb = sd_num[vnext]     # < 0
            da = poly.positions_den[vcur]
            db = poly.positions_den[vnext]

            # The clip formula
            #   newpos = (-Sb * da * pa_q + Sa * db * pb_q) / (Sa * db - Sb * da)
            # where pa_q = pa_num / da is the rational position. Substituting:
            #   newpos = (-Sb * pa_num + Sa * pb_num) / (Sa * db - Sb * da)
            # so the new numerators come out as a single integer linear
            # combination of pa_num and pb_num — no extra `da, db` factors.
            new_den = Sa * db - Sb * da
            # Sa ≥ 0 and Sb < 0 with da, db > 0 ⇒ new_den > 0 strictly.
            new_num1 = Sa * poly.positions_num[1, vnext] - Sb * poly.positions_num[1, vcur]
            new_num2 = Sa * poly.positions_num[2, vnext] - Sb * poly.positions_num[2, vcur]
            new_num3 = Sa * poly.positions_num[3, vnext] - Sb * poly.positions_num[3, vcur]

            # Lazy GCD reduction: divide num/den by their common factor.
            # Cheap (~tens of cycles for Int64) and keeps subsequent clips
            # in tighter integer ranges.
            g = gcd(gcd(gcd(new_den, new_num1), new_num2), new_num3)
            if g > one(T)
                new_den  = div(new_den,  g)
                new_num1 = div(new_num1, g)
                new_num2 = div(new_num2, g)
                new_num3 = div(new_num3, g)
            end

            poly.positions_num[1, new_idx] = new_num1
            poly.positions_num[2, new_idx] = new_num2
            poly.positions_num[3, new_idx] = new_num3
            poly.positions_den[new_idx]    = new_den

            poly.pnbrs[1, new_idx] = Int32(vcur)
            poly.pnbrs[2, new_idx] = Int32(0)
            poly.pnbrs[3, new_idx] = Int32(0)
            poly.pnbrs[np, vcur]   = Int32(new_idx)

            clipped[new_idx] = Int32(0)
        end
    end

    # Step 4: link new vertices around faces (3-D face-walk; same
    # algorithm as `R3D.Flat.clip_plane!`).
    @inbounds for vstart in (onv + 1):poly.nverts
        vcur = vstart
        vnext = Int(poly.pnbrs[1, vcur])
        np = 0
        while true
            if poly.pnbrs[1, vnext] == Int32(vcur)
                np = 1
            elseif poly.pnbrs[2, vnext] == Int32(vcur)
                np = 2
            else
                np = 3
            end
            vcur = vnext
            pnext = (np == 3) ? 1 : (np + 1)
            vnext = Int(poly.pnbrs[pnext, vcur])
            vcur <= onv || break
        end
        poly.pnbrs[3, vstart] = Int32(vcur)
        poly.pnbrs[2, vcur]   = Int32(vstart)
    end

    # Step 5: compact, dropping clipped vertices.
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                for i in 1:3
                    poly.positions_num[i, numunclipped] = poly.positions_num[i, v]
                    poly.pnbrs[i, numunclipped]         = poly.pnbrs[i, v]
                end
                poly.positions_den[numunclipped] = poly.positions_den[v]
            end
            clipped[v] = Int32(numunclipped)
        else
            clipped[v] = Int32(0)
        end
    end
    poly.nverts = numunclipped
    @inbounds for v in 1:poly.nverts, np in 1:3
        old = Int(poly.pnbrs[np, v])
        poly.pnbrs[np, v] = old == 0 ? Int32(0) : clipped[old]
    end
    return true
end

# ---------------------------------------------------------------------------
# Exact volume via fan-triangulation
# ---------------------------------------------------------------------------

"""
    volume_exact(poly::IntFlatPolytope{3,T}, ::Type{R} = T) -> Rational{R}

Return the polytope's volume as an exact `Rational{R}`. The default
accumulator type `R = T` keeps the calculation in the same integer
type the polytope is stored in; pass `R = BigInt` (or another wider
type) to defend against overflow at the per-tetrahedron 3-fold
numerator product.

Algorithm: walk the boundary face graph (`emarks` to avoid revisiting
edges, same scheme as `R3D.Flat.moments!`), fan-triangulate each face,
and sum signed `det(M) / 6` per tetrahedron from the polytope's
internal vertex 1 — fully exact, no floating point at any step.

The signed sum is robust to the polytope's vertex orientation; for
a positively-oriented r3d polytope the result is non-negative.
"""
function volume_exact(poly::IntFlatPolytope{3,T},
                      ::Type{R} = T) where {T,R<:Signed}
    nv = poly.nverts
    nv == 0 && return zero(Rational{R})

    emarks = falses(nv, 3)
    sum_rat = zero(Rational{R})

    @inbounds for vstart in 1:nv, pstart in 1:3
        emarks[vstart, pstart] && continue

        # Initialize face walk from this edge.
        pnext = pstart
        vcur = vstart
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])
        v0n1 = R(poly.positions_num[1, vcur])
        v0n2 = R(poly.positions_num[2, vcur])
        v0n3 = R(poly.positions_num[3, vcur])
        v0d  = R(poly.positions_den[vcur])

        # Step to the second edge of this face.
        np = 0
        for k in 1:3
            if poly.pnbrs[k, vnext] == Int32(vcur)
                np = k
                break
            end
        end
        vcur = vnext
        pnext = (np == 3) ? 1 : (np + 1)
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])

        # Fan: each (v0, v1, v2) triangle contributes 6V_signed = det.
        while vnext != vstart
            v2n1 = R(poly.positions_num[1, vcur]);  v2d = R(poly.positions_den[vcur])
            v2n2 = R(poly.positions_num[2, vcur])
            v2n3 = R(poly.positions_num[3, vcur])
            v1n1 = R(poly.positions_num[1, vnext]); v1d = R(poly.positions_den[vnext])
            v1n2 = R(poly.positions_num[2, vnext])
            v1n3 = R(poly.positions_num[3, vnext])

            # 6V_num = det([v0; v1; v2]_num) over common denom v0d * v1d * v2d.
            sixv_num = -v2n1 * v1n2 * v0n3 + v1n1 * v2n2 * v0n3 +
                        v2n1 * v0n2 * v1n3 - v0n1 * v2n2 * v1n3 -
                        v1n1 * v0n2 * v2n3 + v0n1 * v1n2 * v2n3
            sixv_den = v0d * v1d * v2d

            sum_rat += sixv_num // sixv_den

            np = 0
            for k in 1:3
                if poly.pnbrs[k, vnext] == Int32(vcur)
                    np = k
                    break
                end
            end
            vcur = vnext
            pnext = (np == 3) ? 1 : (np + 1)
            emarks[vcur, pnext] = true
            vnext = Int(poly.pnbrs[pnext, vcur])
        end
    end

    return sum_rat // 6
end

# ===========================================================================
# D = 2 — same shared-denominator-per-vertex representation, 2D variant.
# `pnbrs[k, v]` for k ∈ {1, 2}; pnbrs == 0 is the unset sentinel.
# Vertex labelling matches `R3D.Flat.init_box!` D = 2 (CCW from
# lower-left). The 2D linker is much simpler than the 3D face-walk:
# one new vertex per cut edge, then a pair-up step that walks the
# pnbrs[1] cycle to find each new vertex's partner across the cut.
# Area is signed-shoelace on the rational vertex coords.
# ===========================================================================

"""
    init_box!(poly::IntFlatPolytope{2,T}, lo, hi) -> poly

Initialize as the axis-aligned integer rectangle with corners `lo`
and `hi`. Vertex labelling and `pnbrs` table mirror
`R3D.Flat.init_box!` D = 2 (CCW from lower-left). All denominators
start at 1.
"""
function init_box!(poly::IntFlatPolytope{2,T},
                   lo::AbstractVector, hi::AbstractVector) where {T}
    @assert poly.capacity >= 4
    poly.nverts = 4

    @inbounds begin
        poly.positions_num[1,1] = T(lo[1]); poly.positions_num[2,1] = T(lo[2])
        poly.positions_num[1,2] = T(hi[1]); poly.positions_num[2,2] = T(lo[2])
        poly.positions_num[1,3] = T(hi[1]); poly.positions_num[2,3] = T(hi[2])
        poly.positions_num[1,4] = T(lo[1]); poly.positions_num[2,4] = T(hi[2])

        for v in 1:4
            poly.positions_den[v] = one(T)
        end

        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(4)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(4); poly.pnbrs[2,3] = Int32(2)
        poly.pnbrs[1,4] = Int32(1); poly.pnbrs[2,4] = Int32(3)
    end
    return poly
end

"""
    init_simplex!(poly::IntFlatPolytope{2,T}, v1, v2, v3) -> poly

Initialize as the triangle with the three given integer-coordinate
vertices, listed in CCW order around the boundary so the resulting
shoelace area is positive. `pnbrs[1, v]` = next vertex CCW;
`pnbrs[2, v]` = previous (mirroring `R3D.Flat.init_simplex!` D = 2).
"""
function init_simplex!(poly::IntFlatPolytope{2,T},
                       v1::AbstractVector, v2::AbstractVector,
                       v3::AbstractVector) where {T}
    @assert poly.capacity >= 3
    poly.nverts = 3
    @inbounds begin
        poly.positions_num[1,1] = T(v1[1]); poly.positions_num[2,1] = T(v1[2])
        poly.positions_num[1,2] = T(v2[1]); poly.positions_num[2,2] = T(v2[2])
        poly.positions_num[1,3] = T(v3[1]); poly.positions_num[2,3] = T(v3[2])
        for v in 1:3
            poly.positions_den[v] = one(T)
        end
        # CCW: pnbrs[1] = next, pnbrs[2] = previous.
        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(3)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(1); poly.pnbrs[2,3] = Int32(2)
    end
    return poly
end

"""
    clip!(poly::IntFlatPolytope{2,T}, planes) -> Bool
    clip!(poly::IntFlatPolytope{2,T}, plane::Plane{2,T}) -> Bool

Clip in-place against integer-coefficient `Plane{2,T}`(s). Returns
`true` on success, `false` on capacity overflow. Same five-step
structure as `R3D.Flat.clip_plane!` D = 2 (signed-distance numerators
→ trivial accept/reject → vertex insertion via integer linear
combination → 2-D pair-up linker → compaction).

The plane convention is identical to D = 3: retain
`{x : n · x + d ≥ 0}`.
"""
function clip!(poly::IntFlatPolytope{2,T},
               planes::AbstractVector{Plane{2,T}}) where {T}
    poly.nverts <= 0 && return false
    @inbounds for plane in planes
        ok = clip_plane!(poly, plane)
        ok || return false
        poly.nverts == 0 && return true
    end
    return true
end

function clip!(poly::IntFlatPolytope{2,T}, plane::Plane{2,T}) where {T}
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

function clip_plane!(poly::IntFlatPolytope{2,T}, plane::Plane{2,T}) where {T}
    onv = poly.nverts
    sd_num = poly.sd_num
    clipped = poly.clipped
    n1 = plane.n[1]; n2 = plane.n[2]
    δ = plane.d

    # Step 1: signed-distance numerators (denominator = positions_den[v]
    # which is positive by invariant, so sign(sd) = sign(sd_num)).
    any_inside  = false
    any_outside = false
    @inbounds for v in 1:onv
        s = n1 * poly.positions_num[1, v] +
            n2 * poly.positions_num[2, v] +
            δ  * poly.positions_den[v]
        sd_num[v] = s
        if s >= zero(T)
            any_inside = true
            clipped[v] = Int32(0)
        else
            any_outside = true
            clipped[v] = Int32(1)
        end
    end

    if !any_outside
        return true
    end
    if !any_inside
        poly.nverts = 0
        return true
    end

    # Step 2: insert ONE new vertex per cut edge (D = 2 has no
    # multi-face-per-vertex case — each kept vertex meets at most
    # two clipped neighbours, at most one per slot). Mirrors r2d.c
    # lines 80–93. The slot opposite `np` (= `3 - np` in 1-based)
    # back-links to vcur; `pnbrs[np, new_idx]` stays unset (sentinel
    # 0) for Step 3 to fill.
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:2
            vnext = Int(poly.pnbrs[np, vcur])
            (vnext == 0 || clipped[vnext] == 0) && continue

            poly.nverts >= poly.capacity && return false
            new_idx = (poly.nverts += 1)

            Sa = sd_num[vcur]      # ≥ 0
            Sb = sd_num[vnext]     # < 0
            da = poly.positions_den[vcur]
            db = poly.positions_den[vnext]

            # Same integer linear-combination cut formula as D = 3:
            # new_pos = (Sa·pb − Sb·pa) / (Sa·db − Sb·da), where the
            # da, db factors cancel out of the numerator algebraically.
            new_den  = Sa * db - Sb * da   # > 0 since Sa ≥ 0, Sb < 0, da, db > 0
            new_num1 = Sa * poly.positions_num[1, vnext] -
                       Sb * poly.positions_num[1, vcur]
            new_num2 = Sa * poly.positions_num[2, vnext] -
                       Sb * poly.positions_num[2, vcur]

            g = gcd(gcd(new_den, new_num1), new_num2)
            if g > one(T)
                new_den  = div(new_den,  g)
                new_num1 = div(new_num1, g)
                new_num2 = div(new_num2, g)
            end

            poly.positions_num[1, new_idx] = new_num1
            poly.positions_num[2, new_idx] = new_num2
            poly.positions_den[new_idx]    = new_den

            other = 3 - np                  # 1↔2 swap
            poly.pnbrs[other, new_idx] = Int32(vcur)
            poly.pnbrs[np, new_idx]    = Int32(0)   # filled by Step 3
            poly.pnbrs[np, vcur]       = Int32(new_idx)

            clipped[new_idx] = Int32(0)
        end
    end

    # Step 3: link new vertices in pairs by walking pnbrs[1] from each
    # new vertex with an unset slot through the kept-side cycle until
    # we hit another new vertex. Mirrors r2d.c lines 97–105.
    @inbounds for vstart in (onv+1):poly.nverts
        poly.pnbrs[2, vstart] != 0 && continue
        vcur = Int(poly.pnbrs[1, vstart])
        while vcur <= onv
            vcur = Int(poly.pnbrs[1, vcur])
        end
        poly.pnbrs[2, vstart] = Int32(vcur)
        poly.pnbrs[1, vcur]   = Int32(vstart)
    end

    # Step 4: compact, dropping clipped vertices.
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                poly.positions_num[1, numunclipped] = poly.positions_num[1, v]
                poly.positions_num[2, numunclipped] = poly.positions_num[2, v]
                poly.positions_den[numunclipped]    = poly.positions_den[v]
                poly.pnbrs[1, numunclipped]         = poly.pnbrs[1, v]
                poly.pnbrs[2, numunclipped]         = poly.pnbrs[2, v]
            end
            clipped[v] = Int32(numunclipped)
        else
            clipped[v] = Int32(0)
        end
    end
    poly.nverts = numunclipped
    @inbounds for v in 1:poly.nverts, np in 1:2
        old = Int(poly.pnbrs[np, v])
        poly.pnbrs[np, v] = old == 0 ? Int32(0) : clipped[old]
    end
    return true
end

"""
    area_exact(poly::IntFlatPolytope{2,T}, ::Type{R} = T) -> Rational{R}

Return the polygon's signed area as an exact `Rational{R}`. The
default accumulator type `R = T` keeps the calculation in the
polytope's storage type; pass `R = BigInt` to defend against
overflow in the per-edge two-fold numerator product.

Algorithm: signed shoelace on the rational vertex coordinates,
walking pnbrs[1] (CCW next) around the boundary from vertex 1.
Each consecutive pair `(v, w)` contributes
`(x_v · y_w − x_w · y_v) / (2 · den_v · den_w)` — pure-integer
numerator, integer-product denominator, no float at any step.
For a positively-oriented polytope (CCW boundary) the result is
non-negative.
"""
function area_exact(poly::IntFlatPolytope{2,T},
                    ::Type{R} = T) where {T,R<:Signed}
    nv = poly.nverts
    nv == 0 && return zero(Rational{R})

    sum_rat = zero(Rational{R})
    @inbounds begin
        v = 1
        # Walk pnbrs[1] around the polygon. We've consumed each edge
        # once when we return to vertex 1.
        for _ in 1:nv
            w = Int(poly.pnbrs[1, v])
            xv_n = R(poly.positions_num[1, v]); yv_n = R(poly.positions_num[2, v])
            xw_n = R(poly.positions_num[1, w]); yw_n = R(poly.positions_num[2, w])
            den  = R(poly.positions_den[v]) * R(poly.positions_den[w])
            twoa_num = xv_n * yw_n - xw_n * yv_n
            sum_rat += twoa_num // den
            v = w
        end
    end
    return sum_rat // 2
end

# ===========================================================================
# Polynomial moments at D ∈ {2, 3} — direct exact-rational port of the
# `R3D.Flat.moments!` Koehl recursion. Same per-edge / per-face boundary
# walk; same `Sm` / `Dm` / `Cm` polynomial-coefficient scratch; same
# normalization at the end. The float kernel uses only integer-constant
# division and multiplication / addition / subtraction in vertex
# coordinates, so the algorithm is exact-rational compatible verbatim
# once the storage type changes from `T` to `Rational{R}`.
#
# Returns `Vector{Rational{R}}` of length `R3D.num_moments(D, P)`. The
# `R` accumulator type defaults to the polytope's storage type; pass
# a wider integer type (`R = BigInt`) to defend against numerator-product
# overflow when the polytope has been clipped repeatedly.
# ===========================================================================

"""
    moments_exact(poly::IntFlatPolytope{D, T}, P::Integer;
                  R::Type{<:Signed} = T) -> Vector{Rational{R}}

Compute moments `∫ x_1^{α_1} … x_D^{α_D} dV` over `poly`, exactly
as `Rational{R}`, for every multi-index `α` with `|α| ≤ P`. Result
is in canonical lex-by-degree order (matching `R3D.Flat.moments` at
the same D).

The accumulator type `R` defaults to `T`; pass `R = BigInt` (or
another wider integer type) for overflow defense — the per-edge /
per-face Koehl recursion's polynomial-scratch denominators grow
roughly as `O(P^D)` bits per step.

Allocation: one `Rational{R}` per `out[i]` plus a small (`(P+1)^D`)
polynomial scratch per call. Exact arithmetic has an inherent
allocation cost beyond what the IntExact `clip!` hot path manages.
"""
function moments_exact(poly::IntFlatPolytope{D, T}, P::Integer;
                       R::Type{<:Signed} = T) where {D, T}
    out = Vector{Rational{R}}(undef, num_moments(D, Int(P)))
    moments_exact!(out, poly, Int(P))
    return out
end

"""
    moments_exact!(out::AbstractVector{Rational{R}},
                   poly::IntFlatPolytope{D, T}, P::Integer) -> out

In-place form of [`moments_exact`](@ref). The accumulator type
`R` is taken from `out`'s element type (`out::Vector{Rational{R}}`),
so to use BigInt accumulation, pass
`out::Vector{Rational{BigInt}}`. `length(out) ≥ num_moments(D, P)`
required. Writes the moments into `out` in canonical order.
Returns `out`.
"""
function moments_exact!(out::AbstractVector{Rational{R}},
                         poly::IntFlatPolytope{D, T},
                         P::Integer) where {D, T, R<:Signed}
    @assert length(out) >= num_moments(D, Int(P))
    fill!(out, zero(Rational{R}))
    poly.nverts <= 0 && return out
    if D == 2
        return _moments_exact_d2!(out, poly, Int(P), R)
    elseif D == 3
        return _moments_exact_d3!(out, poly, Int(P), R)
    else
        error("moments_exact!: D = $D not supported in IntExact (D ∈ {2, 3} only). ",
              "Phase 6 of d4plus_finalization_plan.md will add D ≥ 4 volume; ",
              "polynomial moments at D ≥ 4 are research-grade (sqrt-free " *
              "alternatives to Lasserre).")
    end
end

# Convert a vertex's stored shared-denominator integer position into a
# `Rational{R}` per coordinate. Allocates two `Rational{R}` per call.
@inline _vertex_rational_d2(poly::IntFlatPolytope{2, T}, v::Int,
                             ::Type{R}) where {T, R} =
    (R(poly.positions_num[1, v]) // R(poly.positions_den[v]),
     R(poly.positions_num[2, v]) // R(poly.positions_den[v]))

@inline _vertex_rational_d3(poly::IntFlatPolytope{3, T}, v::Int,
                             ::Type{R}) where {T, R} =
    (R(poly.positions_num[1, v]) // R(poly.positions_den[v]),
     R(poly.positions_num[2, v]) // R(poly.positions_den[v]),
     R(poly.positions_num[3, v]) // R(poly.positions_den[v]))

# Direct exact-rational port of `R3D.Flat.moments!(::FlatPolytope{2, T})`.
# Same per-edge boundary walk on pnbrs[1] (CCW next), same Cm/Dm
# polynomial-scratch recursion, same `out[m] /= Cv·(corder+1)·(corder+2)`
# normalization. The only differences:
# - `twoa = v0x*v1y - v0y*v1x` → exact rational product.
# - `out[1] += 0.5 * twoa` → `out[1] += twoa // 2`.
# - polynomial-scratch arrays are `Rational{R}` (heap-allocated;
#   one per call, reused across vertex iterations).
function _moments_exact_d2!(out::AbstractVector{Rational{R}},
                             poly::IntFlatPolytope{2, T}, P::Int,
                             ::Type{R}) where {T, R<:Signed}
    np1 = P + 1
    Dm = Array{Rational{R}, 3}(undef, np1, 1, 2)
    Cm = Array{Rational{R}, 3}(undef, np1, 1, 2)
    fill!(Dm, zero(Rational{R}))
    fill!(Cm, zero(Rational{R}))
    prevlayer = 1; curlayer = 2

    nv = poly.nverts
    @inbounds for vcur in 1:nv
        vnext = Int(poly.pnbrs[1, vcur])
        v0x, v0y = _vertex_rational_d2(poly, vcur,  R)
        v1x, v1y = _vertex_rational_d2(poly, vnext, R)

        twoa = v0x * v1y - v0y * v1x

        Dm[1, 1, prevlayer] = one(Rational{R})
        Cm[1, 1, prevlayer] = one(Rational{R})
        out[1] += twoa // 2

        m = 1
        for corder in 1:P
            for i in corder:-1:0
                j = corder - i
                m += 1
                ci = i + 1
                Cv = zero(Rational{R}); Dv = zero(Rational{R})
                if i > 0
                    Cv += v1x * Cm[ci - 1, 1, prevlayer]
                    Dv += v0x * Dm[ci - 1, 1, prevlayer]
                end
                if j > 0
                    Cv += v1y * Cm[ci, 1, prevlayer]
                    Dv += v0y * Dm[ci, 1, prevlayer]
                end
                Dv += Cv
                Cm[ci, 1, curlayer] = Cv
                Dm[ci, 1, curlayer] = Dv
                out[m] += twoa * Dv
            end
            curlayer  = 3 - curlayer
            prevlayer = 3 - prevlayer
        end
    end

    @inbounds Cm[1, 1, prevlayer] = one(Rational{R})
    m = 1
    for corder in 1:P
        for i in corder:-1:0
            j = corder - i
            m += 1
            ci = i + 1
            Cv = zero(Rational{R})
            i > 0 && (Cv += Cm[ci - 1, 1, prevlayer])
            j > 0 && (Cv += Cm[ci, 1, prevlayer])
            Cm[ci, 1, curlayer] = Cv
            out[m] = out[m] // (Cv * (corder + 1) * (corder + 2))
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end
    return out
end

# Direct exact-rational port of `R3D.Flat.moments!(::FlatPolytope{3, T})`.
# Same per-face boundary walk via emarks; same fan-triangulation from
# each face's first vertex; same Sm/Dm/Cm polynomial-scratch recursion;
# same `out[m] /= Cv·(corder+1)·(corder+2)·(corder+3)` normalization.
function _moments_exact_d3!(out::AbstractVector{Rational{R}},
                             poly::IntFlatPolytope{3, T}, P::Int,
                             ::Type{R}) where {T, R<:Signed}
    nv = poly.nverts
    np1 = P + 1
    Sm = Array{Rational{R}, 3}(undef, np1, np1, 2)
    Dm = Array{Rational{R}, 3}(undef, np1, np1, 2)
    Cm = Array{Rational{R}, 3}(undef, np1, np1, 2)
    fill!(Sm, zero(Rational{R}))
    fill!(Dm, zero(Rational{R}))
    fill!(Cm, zero(Rational{R}))

    emarks = falses(nv, 3)
    prevlayer = 1; curlayer = 2

    @inbounds for vstart in 1:nv, pstart in 1:3
        emarks[vstart, pstart] && continue

        pnext = pstart
        vcur = vstart
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])
        v0x, v0y, v0z = _vertex_rational_d3(poly, vcur, R)

        np = _find_back3_intexact(poly.pnbrs, vnext, vcur)
        vcur = vnext
        pnext = (np == 3) ? 1 : (np + 1)
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])

        while vnext != vstart
            v2x, v2y, v2z = _vertex_rational_d3(poly, vcur,  R)
            v1x, v1y, v1z = _vertex_rational_d3(poly, vnext, R)

            sixv = -v2x * v1y * v0z + v1x * v2y * v0z +
                    v2x * v0y * v1z - v0x * v2y * v1z -
                    v1x * v0y * v2z + v0x * v1y * v2z

            Sm[1, 1, prevlayer] = one(Rational{R})
            Dm[1, 1, prevlayer] = one(Rational{R})
            Cm[1, 1, prevlayer] = one(Rational{R})
            out[1] += sixv // 6

            m = 1
            for corder in 1:P
                for i in corder:-1:0, j in (corder - i):-1:0
                    k = corder - i - j
                    m += 1
                    ci = i + 1; cj = j + 1
                    Cv = zero(Rational{R}); Dv = zero(Rational{R}); Sv = zero(Rational{R})
                    if i > 0
                        Cv += v2x * Cm[ci - 1, cj, prevlayer]
                        Dv += v1x * Dm[ci - 1, cj, prevlayer]
                        Sv += v0x * Sm[ci - 1, cj, prevlayer]
                    end
                    if j > 0
                        Cv += v2y * Cm[ci, cj - 1, prevlayer]
                        Dv += v1y * Dm[ci, cj - 1, prevlayer]
                        Sv += v0y * Sm[ci, cj - 1, prevlayer]
                    end
                    if k > 0
                        Cv += v2z * Cm[ci, cj, prevlayer]
                        Dv += v1z * Dm[ci, cj, prevlayer]
                        Sv += v0z * Sm[ci, cj, prevlayer]
                    end
                    Dv += Cv
                    Sv += Dv
                    Cm[ci, cj, curlayer] = Cv
                    Dm[ci, cj, curlayer] = Dv
                    Sm[ci, cj, curlayer] = Sv
                    out[m] += sixv * Sv
                end
                curlayer  = 3 - curlayer
                prevlayer = 3 - prevlayer
            end

            np = _find_back3_intexact(poly.pnbrs, vnext, vcur)
            vcur = vnext
            pnext = (np == 3) ? 1 : (np + 1)
            emarks[vcur, pnext] = true
            vnext = Int(poly.pnbrs[pnext, vcur])
        end
    end

    @inbounds Cm[1, 1, prevlayer] = one(Rational{R})
    m = 1
    for corder in 1:P
        for i in corder:-1:0, j in (corder - i):-1:0
            k = corder - i - j
            m += 1
            ci = i + 1; cj = j + 1
            Cv = zero(Rational{R})
            i > 0 && (Cv += Cm[ci - 1, cj, prevlayer])
            j > 0 && (Cv += Cm[ci, cj - 1, prevlayer])
            k > 0 && (Cv += Cm[ci, cj, prevlayer])
            Cm[ci, cj, curlayer] = Cv
            out[m] = out[m] //
                     (Cv * (corder + 1) * (corder + 2) * (corder + 3))
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end
    return out
end

# Local copy of `R3D.Flat.find_back3` — kept here so this submodule
# doesn't reach into Flat's internals. The 3-regular pnbrs graph at
# D = 3 makes the by-elimination return correct.
@inline function _find_back3_intexact(pnbrs::Matrix{Int32},
                                       vnext::Int, vcur::Int)
    @inbounds begin
        pnbrs[1, vnext] == Int32(vcur) && return 1
        pnbrs[2, vnext] == Int32(vcur) && return 2
        return 3
    end
end

# ===========================================================================
# voxelize_fold! at D ∈ {2, 3} — exact-rational analog of
# `R3D.Flat.voxelize_fold!`. Bisects the polytope by axis-aligned
# integer-coordinate planes until each leaf cell is a 1×…×1 voxel,
# then computes per-leaf exact-rational moments via `moments_exact!`
# and folds them through the user callback.
#
# The grid spacing `d` is integer (`NTuple{D, T}`). For non-integer
# spacings, scale up the polytope's coordinates first — the IntExact
# representation is happiest when all geometry sits on an integer
# common-denominator grid.
# ===========================================================================

"""
    IntVoxelizeWorkspace{D, T}

Reusable scratch storage for [`voxelize_fold!`](@ref) at D ∈ {2, 3}.
Holds a stack of `IntFlatPolytope{D, T}` buffers (one per recursion
depth) plus the `(ibox_lo, ibox_hi)` index ranges for each. Allocate
once, reuse across many `voxelize_fold!` calls — the bisection-loop
inner work is then heap-free except for the per-call exact-rational
moment scratch (an unavoidable cost of `Rational{R}` arithmetic).
"""
mutable struct IntVoxelizeWorkspace{D, T<:Signed}
    polys::Vector{IntFlatPolytope{D, T}}
    iboxes::Vector{Tuple{NTuple{D, Int}, NTuple{D, Int}}}
    capacity::Int
end

function IntVoxelizeWorkspace{D, T}(capacity::Integer = 64;
                                     max_depth::Integer = 64) where {D, T<:Signed}
    @assert D == 2 || D == 3 "IntVoxelizeWorkspace: D ∈ {2, 3} supported"
    polys = [IntFlatPolytope{D, T}(Int(capacity)) for _ in 1:Int(max_depth)]
    iboxes = Vector{Tuple{NTuple{D, Int}, NTuple{D, Int}}}(undef, Int(max_depth))
    IntVoxelizeWorkspace{D, T}(polys, iboxes, Int(capacity))
end

@inline function _ensure_stack_intexact!(ws::IntVoxelizeWorkspace{D, T},
                                          n::Int) where {D, T}
    z = ntuple(_ -> 0, Val(D))
    while length(ws.polys) < n
        push!(ws.polys, IntFlatPolytope{D, T}(ws.capacity))
        push!(ws.iboxes, (z, z))
    end
    return ws
end

# Copy positions_num + positions_den + pnbrs + nverts. Used by the
# voxelize bisection loop's two-clips-per-split pattern (parallel to
# `R3D.Flat._copy_polytope!`).
@inline function _copy_polytope_intexact!(dst::IntFlatPolytope{D, T},
                                           src::IntFlatPolytope{D, T}) where {D, T}
    n = src.nverts
    @assert dst.capacity >= n
    @inbounds for v in 1:n
        for k in 1:D
            dst.positions_num[k, v] = src.positions_num[k, v]
            dst.pnbrs[k, v]         = src.pnbrs[k, v]
        end
        dst.positions_den[v] = src.positions_den[v]
    end
    dst.nverts = n
    return dst
end

"""
    voxelize_fold!(callback::F, state, poly::IntFlatPolytope{D, T},
                   ibox_lo::NTuple{D, Int}, ibox_hi::NTuple{D, Int},
                   d::NTuple{D, T}, P::Integer;
                   workspace = nothing,
                   R::Type{<:Signed} = T) where {F, D, T}

Walk the same `r3d_voxelize`-style bisection recursion as
`R3D.Flat.voxelize_fold!`, but in exact-rational arithmetic. At
each non-empty leaf cell `(i_1, …, i_D)` (1-based, relative to
`ibox_lo`) calls

    state = callback(state, idx::NTuple{D, Int}, m::Vector{Rational{R}})

where `m` is a `num_moments(D, P)`-length view into the
moment-scratch buffer (overwritten on the next leaf — consume in
the callback or copy out).

`d` is the integer grid spacing per axis. Cell `(i_1, …, i_D)`
covers `[i_k · d_k, (i_k + 1) · d_k)` per axis. Restricting to
integer `d` keeps every per-cell intersection polytope at
shared-denominator-of-product-of-d's resolution — exact rational
all the way through. For finer-than-unit grids, scale the
polytope coordinates first.

Returns the final `state`.

D ∈ {2, 3} only at this writing; D ≥ 4 lands in Phase 6.
"""
function voxelize_fold!(callback::F,
                        state,
                        poly::IntFlatPolytope{D, T},
                        ibox_lo::NTuple{D, Int}, ibox_hi::NTuple{D, Int},
                        d::NTuple{D, T}, P::Integer;
                        workspace::Union{Nothing, IntVoxelizeWorkspace{D, T}} = nothing,
                        R::Type{<:Signed} = T) where {F, D, T}
    @assert D == 2 || D == 3 "voxelize_fold! IntExact: D ∈ {2, 3} only"
    sizes = ntuple(k -> ibox_hi[k] - ibox_lo[k], Val(D))
    (poly.nverts <= 0 || any(s -> s <= 0, sizes)) && return state

    ws = workspace === nothing ?
        IntVoxelizeWorkspace{D, T}(poly.capacity) : workspace
    log2c(x) = x <= 1 ? 0 : ceil(Int, log2(x))
    max_depth = sum(log2c, sizes) + 2
    _ensure_stack_intexact!(ws, max_depth)

    moments_buf = Vector{Rational{R}}(undef, num_moments(D, Int(P)))

    _copy_polytope_intexact!(ws.polys[1], poly)
    ws.iboxes[1] = (ibox_lo, ibox_hi)
    nstack = 1

    @inbounds while nstack > 0
        cur = ws.polys[nstack]
        lo, hi = ws.iboxes[nstack]
        nstack -= 1

        cur.nverts <= 0 && continue

        # Find the longest axis to split.
        spax = 1
        dmax = hi[1] - lo[1]
        for k in 2:D
            s = hi[k] - lo[k]
            if s > dmax
                dmax = s
                spax = k
            end
        end

        if dmax == 1
            # Leaf cell: compute exact moments and call back.
            moments_exact!(moments_buf, cur, Int(P))
            idx = ntuple(k -> lo[k] - ibox_lo[k] + 1, Val(D))
            state = callback(state, idx, moments_buf)
            continue
        end

        # Bisect along spax at the integer cell boundary.
        half = dmax >> 1
        split_index = lo[spax] + half
        split_pos = T(split_index) * d[spax]

        if nstack + 2 > length(ws.polys)
            _ensure_stack_intexact!(ws, nstack + 2)
        end
        # cur === ws.polys[nstack + 1] (the just-popped slot). Save into
        # ws.polys[nstack + 2] before mutating cur in place.
        out0 = ws.polys[nstack + 1]   # alias of cur — becomes the left half
        out1 = ws.polys[nstack + 2]   # fresh slot — gets the right half

        _copy_polytope_intexact!(out1, cur)

        # Plane convention is identical to R3D.Flat: retain
        # {x : n·x + d_plane ≥ 0}.
        #   plane_neg = (-e_spax,  +split_pos) keeps x[spax] ≤ split_pos.
        #   plane_pos = (+e_spax, -split_pos) keeps x[spax] ≥ split_pos.
        n_neg = ntuple(k -> k == spax ? T(-1) : T(0), Val(D))
        n_pos = ntuple(k -> k == spax ? T( 1) : T(0), Val(D))
        plane_neg = Plane{D, T}(Vec{D, T}(n_neg),  split_pos)
        plane_pos = Plane{D, T}(Vec{D, T}(n_pos), -split_pos)
        clip!(out0, plane_neg)
        clip!(out1, plane_pos)

        hi_left  = ntuple(k -> k == spax ? split_index : hi[k], Val(D))
        lo_right = ntuple(k -> k == spax ? split_index : lo[k], Val(D))
        ws.iboxes[nstack + 1] = (lo, hi_left)
        ws.iboxes[nstack + 2] = (lo_right, hi)
        nstack += 2
    end
    return state
end

# ===========================================================================
# Affine operations + construction conveniences. Pure-integer when the
# transform's coefficients are integer; rational shared-denominator
# representation propagates naturally through `(A_num · pos_num) / (den ·
# A_den)`. The `rotate!` wrapper accepts only integer-orthogonal matrices
# (signed permutations + reflections); general SO(D) requires
# irrational entries and falls outside IntExact's scope.
# ===========================================================================

"""
    translate!(poly::IntFlatPolytope{D, T}, t::NTuple{D, T}) -> poly

Translate every vertex of `poly` by the integer offset `t`. Same
denominator-per-vertex; numerators incremented by `t .* den`.
Heap-free; integer arithmetic only.
"""
function translate!(poly::IntFlatPolytope{D, T},
                     t::NTuple{D, T}) where {D, T}
    @inbounds for v in 1:poly.nverts
        den = poly.positions_den[v]
        for k in 1:D
            poly.positions_num[k, v] += t[k] * den
        end
    end
    return poly
end

"""
    scale!(poly::IntFlatPolytope{D, T}, s::T) -> poly

Uniformly scale `poly` by the integer factor `s`. Multiplies every
numerator; denominators unchanged. Pass a negative `s` for a
reflection through the origin. Heap-free.

For non-integer scaling pass an `(s_num, s_den)` pair: `scale!(poly,
s_num, s_den)` multiplies every numerator by `s_num` and every
denominator by `s_den`, then GCD-reduces per vertex.
"""
function scale!(poly::IntFlatPolytope{D, T}, s::T) where {D, T}
    @inbounds for v in 1:poly.nverts
        for k in 1:D
            poly.positions_num[k, v] *= s
        end
    end
    return poly
end

function scale!(poly::IntFlatPolytope{D, T}, s_num::T, s_den::T) where {D, T}
    @assert s_den != zero(T) "scale! denominator must be non-zero"
    sign_den = sign(s_den)
    s_den_pos = sign_den * s_den
    s_num_signed = sign_den * s_num
    @inbounds for v in 1:poly.nverts
        for k in 1:D
            poly.positions_num[k, v] *= s_num_signed
        end
        poly.positions_den[v] *= s_den_pos
        # GCD reduce so the representation stays tight.
        g = abs(poly.positions_den[v])
        for k in 1:D
            g = gcd(g, abs(poly.positions_num[k, v]))
            g == one(T) && break
        end
        if g > one(T)
            for k in 1:D
                poly.positions_num[k, v] = div(poly.positions_num[k, v], g)
            end
            poly.positions_den[v] = div(poly.positions_den[v], g)
        end
    end
    return poly
end

"""
    affine!(poly::IntFlatPolytope{D, T}, A_num::AbstractMatrix{T},
            A_den::T = one(T)) -> poly

Apply the linear map `(A_num / A_den) * x` to every vertex. The
matrix is `D × D`; pass `A_den = 1` for an integer matrix. With a
shared denominator A_den, new vertex denominators become
`old_den * A_den` (then GCD-reduced).

For an affine transform with translation, compose
`affine!(poly, A_num, A_den)` then `translate!(poly, t)`. There's
no augmented `(D+1) × (D+1)` form — the integer-by-integer
multiplication composition is cleaner.
"""
function affine!(poly::IntFlatPolytope{D, T},
                  A_num::AbstractMatrix{T},
                  A_den::T = one(T)) where {D, T}
    @assert size(A_num) == (D, D) "affine! matrix must be D × D = $D × $D"
    @assert A_den != zero(T) "affine! denominator must be non-zero"

    # Stack-friendly row-by-row update: cache each vertex's old
    # numerators in locals before overwriting them.
    @inbounds for v in 1:poly.nverts
        if D == 2
            old1 = poly.positions_num[1, v]
            old2 = poly.positions_num[2, v]
            poly.positions_num[1, v] = A_num[1, 1] * old1 + A_num[1, 2] * old2
            poly.positions_num[2, v] = A_num[2, 1] * old1 + A_num[2, 2] * old2
        elseif D == 3
            old1 = poly.positions_num[1, v]
            old2 = poly.positions_num[2, v]
            old3 = poly.positions_num[3, v]
            poly.positions_num[1, v] = A_num[1, 1] * old1 + A_num[1, 2] * old2 + A_num[1, 3] * old3
            poly.positions_num[2, v] = A_num[2, 1] * old1 + A_num[2, 2] * old2 + A_num[2, 3] * old3
            poly.positions_num[3, v] = A_num[3, 1] * old1 + A_num[3, 2] * old2 + A_num[3, 3] * old3
        else
            # D ≥ 4 not yet supported — Phase 6 lands the generic kernel.
            error("affine! IntExact: D = $D not supported (D ∈ {2, 3} only).")
        end
        if A_den != one(T)
            poly.positions_den[v] *= A_den
        end
        # Keep representation tight — GCD-reduce against the new den.
        if A_den != one(T)
            g = abs(poly.positions_den[v])
            for k in 1:D
                g = gcd(g, abs(poly.positions_num[k, v]))
                g == one(T) && break
            end
            if g > one(T)
                for k in 1:D
                    poly.positions_num[k, v] = div(poly.positions_num[k, v], g)
                end
                poly.positions_den[v] = div(poly.positions_den[v], g)
            end
        end
    end
    return poly
end

"""
    rotate!(poly::IntFlatPolytope{D, T}, A::AbstractMatrix{T}) -> poly

Apply the integer-orthogonal map `A` to every vertex. Asserts
`A * A' == I` and `det(A) == ±1`; for integer `A` this restricts
to signed permutations (axis swaps with optional sign flips).
General SO(D) rotations involve irrational entries and aren't
representable in integer-only IntExact — use `R3D.Flat.rotate!`
for those.
"""
function rotate!(poly::IntFlatPolytope{D, T},
                  A::AbstractMatrix{T}) where {D, T}
    @assert size(A) == (D, D) "rotate! matrix must be D × D"
    # A * A' should be the identity.
    @inbounds for i in 1:D, j in 1:D
        s = zero(T)
        for k in 1:D
            s += A[i, k] * A[j, k]
        end
        expected = (i == j) ? one(T) : zero(T)
        @assert s == expected (
            "rotate! requires A * A' == I (integer-orthogonal); ",
            "row-product[$i, $j] = $s, expected $expected. ",
            "General rotations need irrational entries — use ",
            "R3D.Flat.rotate! for those.")
    end
    return affine!(poly, A, one(T))
end

# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

"""
    box(lo::NTuple{D, T}, hi::NTuple{D, T}; capacity = 64) where {D, T<:Signed}

Allocate a fresh `IntFlatPolytope{D, T}` and initialize as the
axis-aligned integer box with the given corners. D ∈ {2, 3} only.
"""
# Per-D dispatch: a single `where {D, T<:Signed}` method binds both
# parameters via `NTuple{D, T}`, which Aqua flags as unbound when
# D could be 0. Per-D explicit methods avoid the warning and match
# the existing R3D.Flat per-D `@eval` pattern.
for _D in 2:3
    @eval function box(lo::NTuple{$_D, T}, hi::NTuple{$_D, T};
                        capacity::Integer = 64) where {T<:Signed}
        poly = IntFlatPolytope{$_D, T}(Int(capacity))
        init_box!(poly, [lo...], [hi...])
        return poly
    end
end

"""
    simplex(verts::Vararg{NTuple{D, T}, M}; capacity = 64) where {D, T<:Signed, M}

Allocate a fresh `IntFlatPolytope{D, T}` and initialize as the
simplex with the given `D + 1` integer-coordinate vertices.
D ∈ {2, 3} (3 vertices for a triangle, 4 for a tetrahedron).
"""
# Per-D dispatch (mirrors the R3D.Flat per-D `@eval` pattern for Aqua's
# unbound-args check). For D = 2 the user supplies 3 NTuple{2}s; for
# D = 3 the user supplies 4 NTuple{3}s.
function simplex(v1::NTuple{2, T}, v2::NTuple{2, T}, v3::NTuple{2, T};
                  capacity::Integer = 64) where {T<:Signed}
    poly = IntFlatPolytope{2, T}(Int(capacity))
    init_simplex!(poly, [v1...], [v2...], [v3...])
    return poly
end

function simplex(v1::NTuple{3, T}, v2::NTuple{3, T}, v3::NTuple{3, T},
                  v4::NTuple{3, T};
                  capacity::Integer = 64) where {T<:Signed}
    poly = IntFlatPolytope{3, T}(Int(capacity))
    init_tet!(poly, ([v1...], [v2...], [v3...], [v4...]))
    return poly
end

# ===========================================================================
# D ≥ 4 — exact-rational kernel mirroring `R3D.Flat`'s D ≥ 4 path.
# Vertex storage stays shared-denom integer; the new fields `finds[a, b, v]`
# (2-face IDs) and `facets[k, v]` (codim-1 IDs) carry the same meaning as
# in `FlatPolytope`. The clip kernel ports `R3D.Flat.clip_plane!`'s D ≥ 4
# 5-step algorithm verbatim, swapping the cut-position formula for the
# integer shared-denominator one. `volume_exact` at D = 4 fan-triangulates
# the polytope into 4-simplices through the existing facet/2-face/edge
# walk and sums signed `det(M) / 4!` per simplex — fully sqrt-free.
# ===========================================================================

"""
    init_box!(poly::IntFlatPolytope{D,T}, lo, hi) -> poly  (D ≥ 4)

Initialize as the axis-aligned integer box with corners `lo` and `hi`.
Vertex labelling, `pnbrs`, `finds`, and `facets` tables match
`R3D.Flat.init_box!` D ≥ 4.
"""
function init_box!(poly::IntFlatPolytope{D,T},
                   lo::AbstractVector, hi::AbstractVector) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 init_box! methods for those dimensions"
    nv = 1 << D
    @assert poly.capacity >= nv "init_box! D=$D needs capacity ≥ $nv, got $(poly.capacity)"
    @assert length(lo) == D && length(hi) == D
    poly.nverts = nv
    @inbounds for v in 0:(nv - 1)
        for i in 1:D
            stride = 1 << (i - 1)
            poly.positions_num[i, v + 1] = T((v & stride) != 0 ? hi[i] : lo[i])
            poly.pnbrs[i, v + 1] = Int32((v ⊻ stride) + 1)
        end
        poly.positions_den[v + 1] = one(T)
    end

    # 2-face connectivity: one 2-face per axis pair (np, np1) with np < np1.
    f = 0
    @inbounds for np in 1:D, np1 in (np + 1):D
        f += 1
        for v in 1:nv
            poly.finds[np,  np1, v] = Int32(f)
            poly.finds[np1, np,  v] = Int32(f)
        end
    end
    poly.nfaces = f

    # 2D facets per axis-side, IDed by axis convention.
    @inbounds for v in 0:(nv - 1), k in 1:D
        stride = 1 << (k - 1)
        on_hi = (v & stride) != 0
        poly.facets[k, v + 1] = Int32(on_hi ? 2k : 2k - 1)
    end
    poly.nfacets = 2D
    return poly
end

"""
    init_simplex!(poly::IntFlatPolytope{D,T}, vertices) -> poly  (D ≥ 4)

D-simplex constructor for `D ≥ 4`. Takes a length-`(D+1)` collection of
integer vertex positions. Mirrors `R3D.Flat.init_simplex!` D ≥ 4: vertex
v's neighbours are `(v + i) mod (D + 1)` for `i ∈ 1:D` in cyclic order;
2-face IDs from `C(D+1, 3)` triples; facet IDs `pnbrs[k, v]` (facet
opposite v's slot k = facet opposite the OTHER endpoint of that edge).
"""
function init_simplex!(poly::IntFlatPolytope{D,T}, vertices) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 init_simplex! methods for those dimensions"
    nv = D + 1
    @assert length(vertices) == nv "D=$D simplex needs $(D+1) vertices, got $(length(vertices))"
    @assert poly.capacity >= nv
    poly.nverts = nv
    @inbounds for v in 1:nv
        p = vertices[v]
        for i in 1:D
            poly.positions_num[i, v] = T(p[i])
            poly.pnbrs[i, v] = Int32(((v - 1 + i) % nv) + 1)
        end
        poly.positions_den[v] = one(T)
    end

    f = 0
    @inbounds for v0 in 1:nv, v1 in (v0+1):nv, v2 in (v1+1):nv
        f += 1
        np1_at_v0 = _find_pnbr_slot(poly, v0, v1, D)
        np2_at_v0 = _find_pnbr_slot(poly, v0, v2, D)
        poly.finds[np1_at_v0, np2_at_v0, v0] = Int32(f)
        poly.finds[np2_at_v0, np1_at_v0, v0] = Int32(f)
        np0_at_v1 = _find_pnbr_slot(poly, v1, v0, D)
        np2_at_v1 = _find_pnbr_slot(poly, v1, v2, D)
        poly.finds[np0_at_v1, np2_at_v1, v1] = Int32(f)
        poly.finds[np2_at_v1, np0_at_v1, v1] = Int32(f)
        np0_at_v2 = _find_pnbr_slot(poly, v2, v0, D)
        np1_at_v2 = _find_pnbr_slot(poly, v2, v1, D)
        poly.finds[np0_at_v2, np1_at_v2, v2] = Int32(f)
        poly.finds[np1_at_v2, np0_at_v2, v2] = Int32(f)
    end
    poly.nfaces = f

    @inbounds for v in 1:nv, k in 1:D
        poly.facets[k, v] = poly.pnbrs[k, v]
    end
    poly.nfacets = nv
    return poly
end

@inline function _find_pnbr_slot(poly::IntFlatPolytope, v::Int, target::Int, D::Int)
    @inbounds for k in 1:D
        Int(poly.pnbrs[k, v]) == target && return k
    end
    error("_find_pnbr_slot: vertex $target not a neighbour of $v")
end

@inline function _find_pnbr_slot_or0_int(poly::IntFlatPolytope, v::Int, target::Int, D::Int)
    @inbounds for k in 1:D
        Int(poly.pnbrs[k, v]) == target && return k
    end
    return 0
end

@inline function _find_face_in_row0_int(poly::IntFlatPolytope, vcur::Int, target::Int, D::Int)
    @inbounds for k in 2:D
        Int(poly.finds[1, k, vcur]) == target && return k
    end
    error("_find_face_in_row0_int: face $target not in row 0 of vertex $vcur")
end

@inline function _find_face_in_finds_row_int(poly::IntFlatPolytope, vcur::Int, pprev::Int,
                                              target::Int, D::Int)
    @inbounds for k in 1:D
        k == pprev && continue
        Int(poly.finds[pprev, k, vcur]) == target && return k
    end
    error("_find_face_in_finds_row_int: face $target not in row $pprev of vertex $vcur")
end

@inline function _find_face_in_finds_col_int(poly::IntFlatPolytope, vcur::Int, pprev::Int,
                                              target::Int, D::Int)
    @inbounds for k in 1:D
        k == pprev && continue
        Int(poly.finds[k, pprev, vcur]) == target && return k
    end
    error("_find_face_in_finds_col_int: face $target not in col $pprev of vertex $vcur")
end

"""
    clip!(poly::IntFlatPolytope{D,T}, planes) -> Bool  (D ≥ 4)

Clip in-place against an iterable of integer-coefficient `Plane{D,T}`s.
Returns `false` on capacity exhaust, `true` otherwise. Same five-step
structure as `R3D.Flat.clip!` D ≥ 4 (signed-distance numerators →
trivial accept/reject → vertex insertion via shared-denom integer
linear combination → 2-face boundary walk linker → compaction).
"""
function clip!(poly::IntFlatPolytope{D,T},
               planes::AbstractVector{Plane{D,T}}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    poly.nverts <= 0 && return false
    @inbounds for plane in planes
        ok = clip_plane!(poly, plane)
        ok || return false
        poly.nverts == 0 && return true
    end
    return true
end

function clip!(poly::IntFlatPolytope{D,T}, plane::Plane{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

"""
    clip_plane!(poly::IntFlatPolytope{D,T}, plane::Plane{D,T}) -> Bool  (D ≥ 4)

Apply a single half-space clip to a `D ≥ 4` integer polytope. Direct
exact-rational port of `R3D.Flat.clip_plane!` D ≥ 4: same Step-1
signed-distance pass, Step-2 trivial accept/reject, Step-3 cut-vertex
insertion (using the shared-denom integer cut formula), Step-4 2-face
boundary walk to close new 2-faces, Step-5 compaction. The plane
convention matches Flat: keeps `{x : n · x + d ≥ 0}`. Coordinates of
`n` and `d` must be integers of type `T`.
"""
function clip_plane!(poly::IntFlatPolytope{D,T}, plane::Plane{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    sd_num  = poly.sd_num
    clipped = poly.clipped
    onv     = poly.nverts

    # --- Step 1: signed-distance numerators (denom = positions_den[v] > 0).
    any_inside = false; any_outside = false
    @inbounds for v in 1:onv
        s = plane.d * poly.positions_den[v]
        for i in 1:D
            s += poly.positions_num[i, v] * plane.n[i]
        end
        sd_num[v] = s
        if s >= zero(T)
            any_inside = true
            clipped[v] = Int32(0)
        else
            any_outside = true
            clipped[v] = Int32(1)
        end
    end

    # --- Step 2: trivial accept/reject.
    any_outside || return true
    if !any_inside
        poly.nverts = 0
        return true
    end

    # --- Step 3: insert new vertices on each cut edge. Same connectivity
    # bookkeeping as Flat's D ≥ 4 step 3 — partial finds row/column from
    # vcur, partial facet propagation, slot 1 reserved for the new cut
    # facet.
    new_facet_id = Int32(poly.nfacets + 1)
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        Sa = sd_num[vcur]      # ≥ 0
        da = poly.positions_den[vcur]
        for np in 1:D
            vnext = Int(poly.pnbrs[np, vcur])
            (vnext == 0 || clipped[vnext] == 0) && continue

            poly.nverts >= poly.capacity && return false
            new_v = (poly.nverts += 1)

            Sb = sd_num[vnext]      # < 0
            db = poly.positions_den[vnext]

            # Cut formula in shared-denom form:
            #   new_den  = Sa * db - Sb * da   (> 0 strictly, since Sa ≥ 0,
            #                                   Sb < 0, da, db > 0)
            #   new_num_i = Sa * pos_num[i, vnext] - Sb * pos_num[i, vcur]
            new_den = Sa * db - Sb * da
            g = new_den
            for i in 1:D
                num_i = Sa * poly.positions_num[i, vnext] -
                        Sb * poly.positions_num[i, vcur]
                poly.positions_num[i, new_v] = num_i
                g = gcd(g, num_i)
            end
            if g > one(T)
                new_den = div(new_den, g)
                for i in 1:D
                    poly.positions_num[i, new_v] = div(poly.positions_num[i, new_v], g)
                end
            end
            poly.positions_den[new_v] = new_den

            poly.pnbrs[1, new_v] = Int32(vcur)
            for k in 2:D
                poly.pnbrs[k, new_v] = Int32(0)
            end
            poly.pnbrs[np, vcur] = Int32(new_v)

            # Slot 1 of new_v opposite vcur ⇒ the new cut facet.
            poly.facets[1, new_v] = new_facet_id

            # Carry 2-face IDs and facets from vcur into new_v.
            np1 = 1
            for np0 in 1:D
                np0 == np && continue
                np1 += 1
                fid = poly.finds[np, np0, vcur]
                poly.finds[1,   np1, new_v] = fid
                poly.finds[np1, 1,   new_v] = fid
                poly.facets[np1, new_v] = poly.facets[np0, vcur]
            end
            for np0 in 2:D, np1b in 2:D
                poly.finds[np0, np1b, new_v] = Int32(0)
            end
            clipped[new_v] = Int32(0)
        end
    end

    poly.nfacets += 1

    # --- Step 4: walk 2-face boundaries (verbatim from Flat D ≥ 4).
    nfaces = poly.nfaces
    @inbounds for vstart in (onv + 1):poly.nverts, np0 in 2:D, np1 in (np0 + 1):D
        if poly.finds[np0, np1, vstart] != 0
            continue
        end
        fcur = Int(poly.finds[1, np0, vstart])
        fadj = Int(poly.finds[1, np1, vstart])
        vprev = vstart
        vcur  = Int(poly.pnbrs[1, vstart])
        prevnewvert = vstart
        new_fid = Int32(nfaces + 1)
        while true
            if vcur > onv
                pprev = _find_face_in_row0_int(poly, vcur, fcur, D)
                pnext = _find_face_in_row0_int(poly, prevnewvert, fcur, D)
                npx   = _find_face_in_row0_int(poly, vcur, fadj, D)
                poly.pnbrs[pprev, vcur]            = Int32(prevnewvert)
                poly.pnbrs[pnext, prevnewvert]     = Int32(vcur)
                poly.finds[pprev, npx, vcur]       = new_fid
                poly.finds[npx, pprev, vcur]       = new_fid
                fcur, fadj = fadj, fcur
                prevnewvert = vcur
                vprev = vcur
                vcur  = Int(poly.pnbrs[1, vcur])
            end
            pprev = _find_pnbr_slot_or0_int(poly, vcur, vprev, D)
            pnext = _find_face_in_finds_row_int(poly, vcur, pprev, fcur, D)
            npx   = _find_face_in_finds_col_int(poly, vcur, pprev, fadj, D)
            fadj = Int(poly.finds[npx, pnext, vcur])
            vprev = vcur
            vcur  = Int(poly.pnbrs[pnext, vcur])
            vcur == vstart && break
        end
        pprev = _find_face_in_row0_int(poly, vcur, fcur, D)
        pnext = _find_face_in_row0_int(poly, prevnewvert, fcur, D)
        npx   = _find_face_in_row0_int(poly, vcur, fadj, D)
        poly.pnbrs[pprev, vcur]        = Int32(prevnewvert)
        poly.pnbrs[pnext, prevnewvert] = Int32(vcur)
        poly.finds[pprev, npx, vcur]   = new_fid
        poly.finds[npx, pprev, vcur]   = new_fid
        nfaces += 1
    end
    poly.nfaces = nfaces

    # --- Step 5: compact, dropping clipped vertices. `facets[k, v]`
    # follows v but the ID is global (no remap).
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                for i in 1:D
                    poly.positions_num[i, numunclipped] = poly.positions_num[i, v]
                    poly.pnbrs[i, numunclipped]         = poly.pnbrs[i, v]
                    poly.facets[i, numunclipped]        = poly.facets[i, v]
                end
                poly.positions_den[numunclipped] = poly.positions_den[v]
                for i in 1:D, j in 1:D
                    poly.finds[i, j, numunclipped] = poly.finds[i, j, v]
                end
            end
            clipped[v] = Int32(numunclipped)
        else
            clipped[v] = Int32(0)
        end
    end
    poly.nverts = numunclipped
    @inbounds for v in 1:poly.nverts, np in 1:D
        old = Int(poly.pnbrs[np, v])
        poly.pnbrs[np, v] = old == 0 ? Int32(0) : clipped[old]
    end
    return true
end

# ---------------------------------------------------------------------------
# Exact volume at D = 4 via fan triangulation
# ---------------------------------------------------------------------------

# `volume_exact` at D = 4 produces an exact `Rational{R}` by fan-triangulating
# the polytope into 4-simplices and summing |det|/4! per simplex.
#
# Triangulation: pick polytope apex `v0` (vertex 1). For each facet `f` not
# containing `v0`, recursively fan-triangulate `f` (a 3-polytope) into
# tetrahedra. For each tet, form a 4-simplex with `v0` as the fifth vertex
# and add |det(M)|/4! to the running rational sum.
#
# The 3-polytope facet triangulation is done WITHOUT relying on `finds[]`'s
# 2-face IDs (which are ambiguous for boxes — `init_box!` assigns one ID
# per axis pair, conflating all opposite parallel 2-faces). Instead, we
# walk 2-face boundaries using GEOMETRIC coplanarity tests in exact
# integer arithmetic. The polytope's `pnbrs[]` graph restricted to the
# facet vertices gives the in-facet adjacency; from `w_0 = first vertex
# on f` we enumerate "link edges" (pairs of in-facet neighbours that are
# themselves connected by an in-facet edge). Each link edge bounds one
# 2-face NOT containing `w_0`; we walk that 2-face's boundary cycle by
# choosing each next vertex via a 4×3 minor-vanish coplanarity test.

# Signed volume of a 4-simplex with apex `vap` and base vertices
# (w0, z0, a, b) — all integer-coordinate IntExact vertex IDs. Returns
# the contribution as a `Rational{R}`: |det(M)| / 24 where M's columns
# are the four edges from `vap`. det is computed in the rational
# numerator system, so the denominator is the product of all five
# vertex denoms.
@inline function _vol_4simplex(poly::IntFlatPolytope{4,T}, vap::Int, w0::Int,
                                z0::Int, a::Int, b::Int,
                                ::Type{R}) where {T,R<:Signed}
    da = R(poly.positions_den[vap])
    dw = R(poly.positions_den[w0])
    dz = R(poly.positions_den[z0])
    de1 = R(poly.positions_den[a])
    de2 = R(poly.positions_den[b])

    # Edge vectors from `vap`. Numerator (vertex_num * d_apex - apex_num * d_vertex);
    # the shared denominator factors out cleanly.
    # M[i, j] for j in 1:4 corresponds to edges to (w0, z0, a, b).
    # Each entry stays as a rational (num, denom).
    e_num = ntuple(j -> ntuple(i -> begin
        v = (j == 1) ? w0 : (j == 2) ? z0 : (j == 3) ? a : b
        dv = R(poly.positions_den[v])
        R(poly.positions_num[i, v]) * da - R(poly.positions_num[i, vap]) * dv
    end, Val(4)), Val(4))
    e_den = (dw, dz, de1, de2) .* da

    # det of 4×4 via Laplace expansion along row 1.
    @inline minor3(c1, c2, c3, r1, r2, r3) = begin
        m11 = e_num[c1][r1]; m12 = e_num[c2][r1]; m13 = e_num[c3][r1]
        m21 = e_num[c1][r2]; m22 = e_num[c2][r2]; m23 = e_num[c3][r2]
        m31 = e_num[c1][r3]; m32 = e_num[c2][r3]; m33 = e_num[c3][r3]
        m11 * (m22 * m33 - m23 * m32) -
        m12 * (m21 * m33 - m23 * m31) +
        m13 * (m21 * m32 - m22 * m31)
    end

    det_num =
        e_num[1][1] * minor3(2, 3, 4, 2, 3, 4) -
        e_num[2][1] * minor3(1, 3, 4, 2, 3, 4) +
        e_num[3][1] * minor3(1, 2, 4, 2, 3, 4) -
        e_num[4][1] * minor3(1, 2, 3, 2, 3, 4)

    # Common denominator across all four columns.
    det_den = e_den[1] * e_den[2] * e_den[3] * e_den[4]
    # |det| / 24 — abs at the end so all fan pieces add up positively.
    return abs(det_num) // (det_den * R(24))
end

"""
    volume_exact(poly::IntFlatPolytope{4,T}, ::Type{R} = T) -> Rational{R}

Return the polytope's 4-volume as an exact `Rational{R}`. Sqrt-free —
fan-triangulates the polytope into 4-simplices through the existing
facet → 2-face → edge boundary walk and sums `|det(M)| / 4!` per
simplex. Each piece is a clean rational `(integer numerator) /
(product of vertex denominators × 24)`.

The default accumulator type `R = T` keeps the calculation in the
polytope's storage type; pass `R = BigInt` (or another wider integer
type) to defend against overflow at the per-4-simplex 4-fold numerator
product.

The polytope must be convex (which all `clip!`-derived polytopes are);
non-convex polytopes will give wrong answers because fan triangulation
relies on convexity.
"""
function volume_exact(poly::IntFlatPolytope{4,T},
                      ::Type{R} = T) where {T,R<:Signed}
    nv = poly.nverts
    nv == 0 && return zero(Rational{R})

    apex = 1   # polytope apex
    sum_rat = zero(Rational{R})

    # Enumerate each facet exactly once.
    seen_f = falses(poly.nfacets)
    @inbounds for v in 1:nv, k in 1:4
        fid = Int(poly.facets[k, v])
        (fid <= 0 || fid > poly.nfacets) && continue
        seen_f[fid] && continue
        seen_f[fid] = true

        # Skip if `apex` is on this facet.
        on_apex = false
        for ka in 1:4
            if Int(poly.facets[ka, apex]) == fid
                on_apex = true; break
            end
        end
        on_apex && continue

        sum_rat += _vol_facet_d4(poly, fid, apex, R)
    end
    return sum_rat
end

# Test whether 4 vertices `ref, p1, p2, p3` of a D = 4 IntExact polytope
# are coplanar (the three difference vectors `p_i - ref` lie in a 2D
# subspace of R^4). Equivalent to: every 3 × 3 minor of the 4 × 3 matrix
# of difference numerators vanishes. Uses pure integer arithmetic — no
# rationals constructed (the per-column denominators are positive scalars
# that don't change minor vanishing).
@inline function _coplanar_d4(poly::IntFlatPolytope{4,T},
                               ref::Int, p1::Int, p2::Int, p3::Int) where {T}
    dr = poly.positions_den[ref]
    d1 = poly.positions_den[p1]
    d2 = poly.positions_den[p2]
    d3 = poly.positions_den[p3]
    @inline diff_num(p::Int, dp::T, k::Int) =
        poly.positions_num[k, p] * dr - poly.positions_num[k, ref] * dp
    @inbounds for omit in 1:4
        # 3 × 3 minor over the rows other than `omit`. det formula on
        # the explicit entries of the 3 × 3 submatrix.
        ks = (omit == 1 ? (2, 3, 4) :
              omit == 2 ? (1, 3, 4) :
              omit == 3 ? (1, 2, 4) : (1, 2, 3))
        a11 = diff_num(p1, d1, ks[1])
        a12 = diff_num(p2, d2, ks[1])
        a13 = diff_num(p3, d3, ks[1])
        a21 = diff_num(p1, d1, ks[2])
        a22 = diff_num(p2, d2, ks[2])
        a23 = diff_num(p3, d3, ks[2])
        a31 = diff_num(p1, d1, ks[3])
        a32 = diff_num(p2, d2, ks[3])
        a33 = diff_num(p3, d3, ks[3])
        det = a11 * (a22 * a33 - a23 * a32) -
              a12 * (a21 * a33 - a23 * a31) +
              a13 * (a21 * a32 - a22 * a31)
        det != zero(T) && return false
    end
    return true
end

# Walk the boundary cycle of the 2-face of facet `fid` (encoded by
# `in_facet`) at vertex `v` along edges (v, n_a) and (v, n_b). Pushes
# the cycle vertices in order onto `cycle` (cleared first). Returns
# `true` on success; `false` on a degenerate walk (no coplanar next).
function _walk_2face_d4!(cycle::Vector{Int},
                          poly::IntFlatPolytope{4,T},
                          in_facet::Vector{Bool},
                          v::Int, n_a::Int, n_b::Int) where {T}
    empty!(cycle)
    push!(cycle, v); push!(cycle, n_a)
    v_prev = v; v_curr = n_a
    nv = poly.nverts
    steps = 0
    while true
        steps += 1
        steps > nv + 4 && return false
        v_next = 0
        @inbounds for k in 1:4
            cand = Int(poly.pnbrs[k, v_curr])
            cand == 0 && continue
            in_facet[cand] || continue
            cand == v_prev && continue
            if _coplanar_d4(poly, v, n_a, n_b, cand)
                v_next = cand; break
            end
        end
        v_next == 0 && return false
        if v_next == v
            return true
        end
        push!(cycle, v_next)
        v_prev = v_curr
        v_curr = v_next
    end
end

# Fan-triangulate facet `fid` of a D = 4 polytope (a 3-polytope embedded
# in a 3-hyperplane of R^4). Picks `w_0 = smallest in-facet vertex`,
# enumerates each 2-face of `fid` exactly once via "smallest-vertex of
# cycle" deduplication, fan-triangulates each 2-face NOT containing
# `w_0` into triangles, and combines with `apex` to form 4-simplices.
function _vol_facet_d4(poly::IntFlatPolytope{4,T}, fid::Int, apex::Int,
                        ::Type{R}) where {T,R<:Signed}
    fid_i = Int32(fid)
    nv = poly.nverts

    in_facet = Vector{Bool}(undef, nv)
    @inbounds for v in 1:nv
        on = false
        for k in 1:4
            if poly.facets[k, v] == fid_i
                on = true; break
            end
        end
        in_facet[v] = on
    end

    w_0 = 0
    @inbounds for v in 1:nv
        if in_facet[v]
            w_0 = v; break
        end
    end
    @assert w_0 != 0 "_vol_facet_d4: facet $fid has no vertices"

    sum_rat = zero(Rational{R})
    cycle = Int[]
    in_nbrs_v = Int[]   # reused per vertex

    # Walk every 2-face of `fid` exactly once: at each vertex `v` on
    # `fid`, enumerate ordered pairs (n_a, n_b) of v's in-facet
    # neighbours; the 2-face containing edges (v, n_a) and (v, n_b) is
    # processed iff `v` is the smallest-indexed vertex of that 2-face's
    # cycle (and additionally we use the lexicographically-smallest
    # (n_a, n_b) pair at v for the 2-face — by walking via n_a first;
    # walking via n_b would trace the same cycle in reverse).
    @inbounds for v in 1:nv
        in_facet[v] || continue

        empty!(in_nbrs_v)
        for k in 1:4
            n = Int(poly.pnbrs[k, v])
            n != 0 && in_facet[n] && push!(in_nbrs_v, n)
        end
        n_inn = length(in_nbrs_v)
        n_inn >= 2 || continue

        for ii in 1:n_inn, jj in (ii + 1):n_inn
            n_a = in_nbrs_v[ii]
            n_b = in_nbrs_v[jj]

            ok = _walk_2face_d4!(cycle, poly, in_facet, v, n_a, n_b)
            ok || continue

            # Smallest-vertex dedup. Also: at the canonical vertex `v`,
            # the 2-face determines a UNIQUE pair {n_a, n_b}; we must
            # skip if (n_a, n_b) isn't the canonical pair (e.g. swapped).
            # Since we only iterate ii < jj, each unordered pair is
            # visited once, so this is automatic. But each 2-face has 3
            # pairs at v if v has > 2 in-facet neighbours and the 2-face
            # only uses 2 of them — we must verify the cycle's 2nd vertex
            # is `n_a` (not some unrelated vertex from the same plane).
            cycle_min = cycle[1]
            for c in cycle
                c < cycle_min && (cycle_min = c)
            end
            cycle_min != v && continue

            # Skip if w_0 is on this 2-face (cone has zero 4-volume).
            on_w0 = false
            for c in cycle
                if c == w_0; on_w0 = true; break; end
            end
            on_w0 && continue

            # Fan-triangulate cycle from cycle[1].
            z_0 = cycle[1]
            for i in 2:(length(cycle) - 1)
                sum_rat += _vol_4simplex(poly, apex, w_0, z_0,
                                          cycle[i], cycle[i + 1], R)
            end
        end
    end

    return sum_rat
end

end # module IntExact
