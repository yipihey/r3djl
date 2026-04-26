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
using ..R3D: Plane, Vec

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
end

function IntFlatPolytope{D,T}(capacity::Integer = 64) where {D,T<:Signed}
    @assert D == 3 "IntFlatPolytope prototype: only D = 3 is implemented"
    capacity = Int(capacity)
    IntFlatPolytope{D,T}(zeros(T, D, capacity),
                         ones(T, capacity),
                         zeros(Int32, D, capacity),
                         0, capacity,
                         Vector{T}(undef, capacity),
                         Vector{Int32}(undef, capacity))
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

end # module IntExact
