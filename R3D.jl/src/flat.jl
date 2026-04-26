"""
    R3D.Flat

Struct-of-arrays variant of `Polytope` that mirrors r3d's C struct
exactly: positions and neighbour indices live in flat dense matrices,
not in an array of mutable `Vertex` objects.

This module is parallel to the main `Polytope` type — both implement
the same algorithm, the only difference is memory layout. Benchmarks
in `R3DBenchmarks.bench_flat_*` compare them head-to-head.

# Layout

```
mutable struct FlatPolytope{D,T}
    positions::Matrix{T}    # D × capacity, column-major (D coords per vertex)
    pnbrs::Matrix{Int32}    # D × capacity, column-major (D neighbours per vertex)
    nverts::Int
    capacity::Int

    # scratch (reused across clip/moments calls — caller-allocated buffer pattern)
    sdists::Vector{T}
    clipped::Vector{Int32}
    emarks::Matrix{Bool}
    Sm::Array{T,3}
    Dm::Array{T,3}
    Cm::Array{T,3}
    moment_order::Int
end
```

This matches the C struct's memory pattern (a packed sequence of
`(pnbrs[3], pos)` records per vertex) but uses SoA instead of AoS,
which is friendlier to modern CPU prefetching and SIMD when scanning
many vertices for one operation (e.g. computing all signed distances
at once).

# Caller-allocated buffer pattern (the fast path)

The `FlatPolytope` itself **is** the reusable buffer — exactly mirroring
how the upstream C library is used (`r3d_poly poly; r3d_init_box(&poly,
…); r3d_clip(&poly, …);` — caller owns the storage). Allocate once,
re-init and re-clip in a hot loop:

```julia
buf = R3D.Flat.FlatBuffer{3,Float64}(64)        # one allocation
for cell in cells
    R3D.Flat.init_box!(buf, cell.lo, cell.hi)   # 0 allocs
    R3D.Flat.clip!(buf, planes)                 # 0 allocs (after first call of an order)
    m = R3D.Flat.moments(buf, 1)                # only `out` allocates; use moments! for 0
end
```

`FlatBuffer` is an alias for `FlatPolytope`. The convenience constructor
`box(lo, hi)` allocates a fresh polytope each call; reach for the
buffer pattern in tight loops.
"""
module Flat

using StaticArrays
using ..R3D: Vec, Plane, signed_distance

# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

"""
    FlatPolytope{D,T}

POD-layout polytope plus pre-allocated scratch storage for clip and
moment integration. Two `Matrix` allocations for the vertex graph plus
a handful of fixed-size scratch buffers, all owned by the polytope.

This is the reusable buffer type — see the module docstring for the
caller-allocated buffer pattern.
"""
mutable struct FlatPolytope{D,T}
    positions::Matrix{T}    # D × capacity
    pnbrs::Matrix{Int32}    # D × capacity (1-based; 0 = "unset")
    nverts::Int
    capacity::Int

    # Per-clip scratch. Allocated once at construction; reused across calls.
    sdists::Vector{T}       # capacity
    clipped::Vector{Int32}  # capacity

    # Per-moments scratch. emarks is capacity-sized; S/D/C are sized by
    # the largest order ever requested and grown lazily.
    emarks::Matrix{Bool}    # capacity × D
    Sm::Array{T,3}
    Dm::Array{T,3}
    Cm::Array{T,3}
    moment_order::Int       # current allocated order for Sm/Dm/Cm; -1 = none

    # 2-face connectivity for D ≥ 4 (rNd port). For D ≤ 3 this stays at
    # the empty (0,0,0) sentinel — D=2 and D=3 clip kernels don't need
    # it (in D=3 a 2-face coincides with an edge, hence the `find_back3`
    # shortcut). For D ≥ 4 it is shape `D × D × capacity`, with
    # `finds[a, b, v]` = `finds[b, a, v]` = the index of the 2-face
    # containing v's edges to neighbours a and b. Sentinel `0` = unset.
    finds::Array{Int32,3}
    nfaces::Int

    # (D−1)-face ("facet") IDs for D ≥ 4. For D ≤ 3 this stays at the
    # empty (0, 0) sentinel. For D ≥ 4: shape `D × capacity`, with
    # `facets[k, v]` = ID of the facet OPPOSITE edge-slot k of vertex
    # v (i.e., the facet that contains all of v's edges except slot k).
    # Sentinel `0` = unset. Used by Lasserre-style higher-order moment
    # integration (added on top of this in a follow-up session).
    facets::Matrix{Int32}
    nfacets::Int

    # D × D scratch for `_reduce_nd_zeroth!`'s LTD recursion at D ≥ 4.
    # Allocated once at construction so `moments!`'s hot loop doesn't
    # heap-allocate an `MMatrix` per call (Julia's escape analysis
    # can't keep a stack-resident MMatrix alive across the recursive
    # `_reduce_helper_nd` call boundary). Empty for D ≤ 3, which use
    # `Sm`/`Dm`/`Cm` for moment integration instead.
    ltd_scratch::Matrix{T}

    # Per-facet outward unit normal and signed distance (`d` in
    # `n·x = d`) for D ≥ 4. Together they parameterize the facet's
    # supporting hyperplane. Lasserre's higher-order moment recursion
    # reads `(n_i · c_i)` for each facet — equivalently `d_i`, since
    # any point `c_i` on the facet satisfies `n_i · c_i = d_i`.
    # Shape: `D × facet_capacity` and `facet_capacity` respectively
    # for D ≥ 4; empty for D ≤ 3 which don't use Lasserre. Indexed by
    # facet ID 1..nfacets (slot 0 in `facets[k, v]` means "unset").
    facet_normals::Matrix{T}
    facet_distances::Vector{T}
end

# Initial allocation size for per-facet metadata (normals + distances)
# at D ≥ 4. Grows lazily via `_grow_facet_metadata!` if a polytope
# accumulates more facet IDs than this through clipping.
const _FACET_INITIAL_CAPACITY = 64

function FlatPolytope{D,T}(capacity::Int = 512) where {D,T}
    finds  = D >= 4 ? zeros(Int32, D, D, capacity) : Array{Int32,3}(undef, 0, 0, 0)
    facets = D >= 4 ? zeros(Int32, D,    capacity) : Matrix{Int32}(undef, 0, 0)
    ltd    = D >= 4 ? zeros(T,     D, D)           : Matrix{T}(undef, 0, 0)
    fnormals = D >= 4 ? zeros(T, D, _FACET_INITIAL_CAPACITY) : Matrix{T}(undef, 0, 0)
    fdistances = D >= 4 ? zeros(T,    _FACET_INITIAL_CAPACITY) : Vector{T}(undef, 0)
    FlatPolytope{D,T}(zeros(T, D, capacity),
                      zeros(Int32, D, capacity),
                      0, capacity,
                      Vector{T}(undef, capacity),
                      Vector{Int32}(undef, capacity),
                      Matrix{Bool}(undef, capacity, D),
                      Array{T,3}(undef, 0, 0, 0),
                      Array{T,3}(undef, 0, 0, 0),
                      Array{T,3}(undef, 0, 0, 0),
                      -1,
                      finds,
                      0,
                      facets,
                      0,
                      ltd,
                      fnormals,
                      fdistances)
end

# Ensure `poly`'s facet-metadata storage holds at least `nfacets`
# entries. Used by `_clip_plane_nd!` when a clip would push
# `nfacets + 1` past the current allocation. Grows by doubling, like
# `_ensure_stack!`.
@inline function _grow_facet_metadata!(poly::FlatPolytope{D,T}, n::Int) where {D,T}
    D >= 4 || return poly
    cur_cap = size(poly.facet_normals, 2)
    n <= cur_cap && return poly
    new_cap = max(2 * cur_cap, n)
    new_normals  = zeros(T, D, new_cap)
    new_distances = zeros(T, new_cap)
    @inbounds for j in 1:cur_cap, i in 1:D
        new_normals[i, j] = poly.facet_normals[i, j]
    end
    @inbounds for j in 1:cur_cap
        new_distances[j] = poly.facet_distances[j]
    end
    poly.facet_normals  = new_normals
    poly.facet_distances = new_distances
    return poly
end

"""
    FlatBuffer{D,T}

Alias for [`FlatPolytope`](@ref). Use this name when documenting the
caller-allocated buffer pattern; the underlying type is the same.
"""
const FlatBuffer = FlatPolytope

# ---------------------------------------------------------------------------
# Constructors (init_box, init_tet)
# ---------------------------------------------------------------------------

"""
    init_box!(poly, lo, hi)

Initialize as an axis-aligned box. Same neighbour table as r3d_init_box,
written directly into the flat matrices. Re-uses `poly`'s storage —
zero allocations.
"""
function init_box!(poly::FlatPolytope{3,T},
                   lo::AbstractVector, hi::AbstractVector) where {T}
    @assert poly.capacity >= 8
    poly.nverts = 8

    # Vertex positions
    @inbounds begin
        poly.positions[1,1] = lo[1]; poly.positions[2,1] = lo[2]; poly.positions[3,1] = lo[3]
        poly.positions[1,2] = hi[1]; poly.positions[2,2] = lo[2]; poly.positions[3,2] = lo[3]
        poly.positions[1,3] = hi[1]; poly.positions[2,3] = hi[2]; poly.positions[3,3] = lo[3]
        poly.positions[1,4] = lo[1]; poly.positions[2,4] = hi[2]; poly.positions[3,4] = lo[3]
        poly.positions[1,5] = lo[1]; poly.positions[2,5] = lo[2]; poly.positions[3,5] = hi[3]
        poly.positions[1,6] = hi[1]; poly.positions[2,6] = lo[2]; poly.positions[3,6] = hi[3]
        poly.positions[1,7] = hi[1]; poly.positions[2,7] = hi[2]; poly.positions[3,7] = hi[3]
        poly.positions[1,8] = lo[1]; poly.positions[2,8] = hi[2]; poly.positions[3,8] = hi[3]
    end

    # Neighbour table — same as in init.jl, line for line
    @inbounds begin
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

"Convenience: build a fresh box."
function box(lo::NTuple{3,Real}, hi::NTuple{3,Real}; capacity::Int = 512)
    p = FlatPolytope{3,Float64}(capacity)
    init_box!(p, [lo...], [hi...])
    return p
end

# D-generic convenience constructor. The D=2 / D=3 specializations
# above pick smaller default capacities tuned to those dimensions.
# For D ≥ 4 we default to 512 which comfortably accommodates a
# unit hypercube (2^D vertices) plus several clip cuts.
function box(lo::NTuple{D,Real}, hi::NTuple{D,Real}; capacity::Int = 512) where {D}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    p = FlatPolytope{D,Float64}(capacity)
    init_box!(p, [lo...], [hi...])
    return p
end

# ---------------------------------------------------------------------------
# Helpers — accessing a vertex's position/neighbours as views
# ---------------------------------------------------------------------------

@inline pos_view(p::FlatPolytope{D,T}, i::Int) where {D,T} =
    SVector{D,T}(ntuple(k -> @inbounds(p.positions[k, i]), Val(D)))

@inline function set_pos!(p::FlatPolytope{D,T}, i::Int, v::AbstractVector) where {D,T}
    @inbounds for k in 1:D
        p.positions[k, i] = v[k]
    end
end

@inline get_nbr(p::FlatPolytope, i::Int, k::Int) = @inbounds Int(p.pnbrs[k, i])
@inline set_nbr!(p::FlatPolytope, i::Int, k::Int, v::Integer) = @inbounds (p.pnbrs[k, i] = Int32(v))

# Find the slot k ∈ {1,2,3} such that pnbrs[k, vnext] == vcur. The C
# kernel does a 3-way linear search; unrolling lets LLVM keep the
# comparisons in registers and elide the `break`.
@inline function find_back3(pnbrs::Matrix{Int32}, vnext::Int, vcur::Int)
    @inbounds begin
        pnbrs[1, vnext] == Int32(vcur) && return 1
        pnbrs[2, vnext] == Int32(vcur) && return 2
        return 3   # by elimination — pnbrs is a 3-regular graph in D=3
    end
end

# Branchless next-index for the (np mod 3)+1 face walk. Avoids the
# integer-mod that mod1(np+1, 3) would otherwise emit.
@inline next3(np::Int) = ifelse(np == 3, 1, np + 1)

# ---------------------------------------------------------------------------
# clip!
# ---------------------------------------------------------------------------

"""
    clip!(poly, planes) -> Bool

Clip `poly` in-place against `planes`. Returns `true` on success, `false`
on capacity overflow.

Reuses `poly.sdists` / `poly.clipped` scratch — zero per-call heap allocation.
"""
function clip!(poly::FlatPolytope{3,T},
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
    clip!(poly::FlatPolytope{3,T}, plane::Plane{3,T}) -> Bool

Single-plane convenience overload — saves the `[plane]` array allocation
the multi-plane API would otherwise pay. Used by `voxelize_fold!`'s
bisection loop and exposed publicly for callers that already have a
single `Plane` in hand.
"""
function clip!(poly::FlatPolytope{3,T}, plane::Plane{3,T}) where {T}
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

function clip_plane!(poly::FlatPolytope{3,T}, plane::Plane{3,T}) where {T}
    onv = poly.nverts
    sdists = poly.sdists
    clipped = poly.clipped

    # Step 1: signed distances
    smin = T(Inf); smax = T(-Inf)
    @inbounds for v in 1:onv
        x = poly.positions[1, v]; y = poly.positions[2, v]; z = poly.positions[3, v]
        s = plane.d + plane.n[1]*x + plane.n[2]*y + plane.n[3]*z
        sdists[v] = s
        s < smin && (smin = s)
        s > smax && (smax = s)
    end

    # Step 2: trivial accept/reject
    smin >= 0 && return true
    if smax <= 0
        poly.nverts = 0
        return true
    end

    @inbounds for v in 1:onv
        clipped[v] = sdists[v] < 0 ? Int32(1) : Int32(0)
    end

    # Step 3: insert new vertices on cut edges
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:3
            vnext = Int(poly.pnbrs[np, vcur])
            vnext == 0 && continue
            clipped[vnext] == 0 && continue

            poly.nverts >= poly.capacity && return false
            new_idx = (poly.nverts += 1)

            wa = -sdists[vnext]; wb = sdists[vcur]
            inv = 1 / (wa + wb)
            poly.positions[1, new_idx] = (wa*poly.positions[1,vcur] + wb*poly.positions[1,vnext]) * inv
            poly.positions[2, new_idx] = (wa*poly.positions[2,vcur] + wb*poly.positions[2,vnext]) * inv
            poly.positions[3, new_idx] = (wa*poly.positions[3,vcur] + wb*poly.positions[3,vnext]) * inv

            poly.pnbrs[1, new_idx] = Int32(vcur)
            poly.pnbrs[2, new_idx] = Int32(0)
            poly.pnbrs[3, new_idx] = Int32(0)
            poly.pnbrs[np, vcur] = Int32(new_idx)

            clipped[new_idx] = Int32(0)
        end
    end

    # Step 4: link new vertices around faces
    @inbounds for vstart in (onv+1):poly.nverts
        vcur = vstart
        vnext = Int(poly.pnbrs[1, vcur])
        while true
            np = find_back3(poly.pnbrs, vnext, vcur)
            vcur = vnext
            vnext = Int(poly.pnbrs[next3(np), vcur])
            vcur <= onv || break
        end
        poly.pnbrs[3, vstart] = Int32(vcur)
        poly.pnbrs[2, vcur]   = Int32(vstart)
    end

    # Step 5: compact. SoA layout makes this a column-copy with no
    # mutable-struct aliasing risk — `positions[:, dst] .= positions[:, src]`
    # is a value copy.
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                poly.positions[1, numunclipped] = poly.positions[1, v]
                poly.positions[2, numunclipped] = poly.positions[2, v]
                poly.positions[3, numunclipped] = poly.positions[3, v]
                poly.pnbrs[1, numunclipped] = poly.pnbrs[1, v]
                poly.pnbrs[2, numunclipped] = poly.pnbrs[2, v]
                poly.pnbrs[3, numunclipped] = poly.pnbrs[3, v]
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
# moments — same Koehl recursion as R3D.moments!, against flat layout
# ---------------------------------------------------------------------------

import ..R3D: num_moments

function moments(poly::FlatPolytope{3,T}, order::Integer) where {T}
    out = zeros(T, num_moments(3, order))
    moments!(out, poly, order)
    return out
end

# Lazily allocate Sm/Dm/Cm to fit the requested order. After the first
# call at a given order, subsequent calls are zero-alloc.
@inline function _ensure_moment_scratch!(poly::FlatPolytope{D,T}, order::Int) where {D,T}
    if order > poly.moment_order
        np1 = order + 1
        poly.Sm = Array{T,3}(undef, np1, np1, 2)
        poly.Dm = Array{T,3}(undef, np1, np1, 2)
        poly.Cm = Array{T,3}(undef, np1, np1, 2)
        poly.moment_order = order
    end
    return nothing
end

function moments!(out::AbstractVector{T},
                  poly::FlatPolytope{3,T},
                  order::Integer) where {T}
    @assert length(out) >= num_moments(3, order)
    fill!(out, zero(T))
    poly.nverts <= 0 && return out

    nv = poly.nverts
    _ensure_moment_scratch!(poly, Int(order))

    emarks = poly.emarks
    @inbounds for k in 1:3, v in 1:nv
        emarks[v, k] = false
    end

    np1 = order + 1
    S = poly.Sm
    D = poly.Dm
    C = poly.Cm
    prevlayer = 1; curlayer = 2

    @inbounds for vstart in 1:nv, pstart in 1:3
        emarks[vstart, pstart] && continue

        pnext = pstart
        vcur = vstart
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])
        v0_x = poly.positions[1, vcur]
        v0_y = poly.positions[2, vcur]
        v0_z = poly.positions[3, vcur]

        np = find_back3(poly.pnbrs, vnext, vcur)
        vcur = vnext
        pnext = next3(np)
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])

        while vnext != vstart
            v2_x = poly.positions[1, vcur]; v2_y = poly.positions[2, vcur]; v2_z = poly.positions[3, vcur]
            v1_x = poly.positions[1, vnext]; v1_y = poly.positions[2, vnext]; v1_z = poly.positions[3, vnext]

            sixv = (-v2_x*v1_y*v0_z + v1_x*v2_y*v0_z + v2_x*v0_y*v1_z
                    - v0_x*v2_y*v1_z - v1_x*v0_y*v2_z + v0_x*v1_y*v2_z)

            S[1,1,prevlayer] = one(T)
            D[1,1,prevlayer] = one(T)
            C[1,1,prevlayer] = one(T)
            out[1] += sixv / 6

            m = 1
            for corder in 1:order
                for i in corder:-1:0, j in (corder - i):-1:0
                    k = corder - i - j
                    m += 1
                    ci = i + 1; cj = j + 1
                    Cv = zero(T); Dv = zero(T); Sv = zero(T)
                    if i > 0
                        Cv += v2_x * C[ci-1, cj, prevlayer]
                        Dv += v1_x * D[ci-1, cj, prevlayer]
                        Sv += v0_x * S[ci-1, cj, prevlayer]
                    end
                    if j > 0
                        Cv += v2_y * C[ci, cj-1, prevlayer]
                        Dv += v1_y * D[ci, cj-1, prevlayer]
                        Sv += v0_y * S[ci, cj-1, prevlayer]
                    end
                    if k > 0
                        Cv += v2_z * C[ci, cj, prevlayer]
                        Dv += v1_z * D[ci, cj, prevlayer]
                        Sv += v0_z * S[ci, cj, prevlayer]
                    end
                    Dv += Cv
                    Sv += Dv
                    C[ci, cj, curlayer] = Cv
                    D[ci, cj, curlayer] = Dv
                    S[ci, cj, curlayer] = Sv
                    out[m] += sixv * Sv
                end
                curlayer  = 3 - curlayer
                prevlayer = 3 - prevlayer
            end

            np = find_back3(poly.pnbrs, vnext, vcur)
            vcur = vnext
            pnext = next3(np)
            emarks[vcur, pnext] = true
            vnext = Int(poly.pnbrs[pnext, vcur])
        end
    end

    @inbounds C[1,1,prevlayer] = one(T)
    m = 1
    for corder in 1:order
        for i in corder:-1:0, j in (corder - i):-1:0
            k = corder - i - j
            m += 1
            ci = i + 1; cj = j + 1
            Cv = zero(T)
            i > 0 && (Cv += C[ci-1, cj, prevlayer])
            j > 0 && (Cv += C[ci, cj-1, prevlayer])
            k > 0 && (Cv += C[ci, cj, prevlayer])
            C[ci, cj, curlayer] = Cv
            out[m] /= Cv * (corder + 1) * (corder + 2) * (corder + 3)
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end
    return out
end

# ---------------------------------------------------------------------------
# split_coord! — axis-aligned bisection used by voxelize
# ---------------------------------------------------------------------------

"""
    split_coord!(in, out0, out1, coord, ax) -> Bool

Split `in` along the axis-aligned plane `x[ax] == coord`. The two halves
are written into `out0` (negative side, `x[ax] ≤ coord`) and `out1`
(positive side, `x[ax] > coord`). Mirrors `r3d_split_coord` in
`src/v3d.c`.

`in` is consumed (its state is undefined after the call). `out0` and
`out1` must be pre-allocated `FlatPolytope`s with the same `D`, `T`,
and `capacity` ≥ the number of vertices the split could produce
(typically `in.capacity` is plenty).

Returns `true` on success, `false` on capacity overflow.
"""
function split_coord!(in::FlatPolytope{3,T},
                      out0::FlatPolytope{3,T}, out1::FlatPolytope{3,T},
                      coord::T, ax::Int) where {T}
    if in.nverts <= 0
        out0.nverts = 0
        out1.nverts = 0
        return true
    end

    sdists = in.sdists
    side   = in.clipped   # reuse: 0 = left, 1 = right

    # Step 1: signed distances and side classification
    nright = 0
    @inbounds for v in 1:in.nverts
        sd = in.positions[ax, v] - coord
        sdists[v] = sd
        if sd > 0
            side[v] = Int32(1)
            nright += 1
        else
            side[v] = Int32(0)
        end
    end

    # Step 2: trivial pass-through cases
    if nright == 0
        _copy_polytope!(out0, in)
        out1.nverts = 0
        return true
    elseif nright == in.nverts
        _copy_polytope!(out1, in)
        out0.nverts = 0
        return true
    end

    # Step 3: insert pairs of new vertices on cut edges
    onv = in.nverts
    @inbounds for vcur in 1:onv
        side[vcur] != 0 && continue
        for np in 1:3
            vnext = Int(in.pnbrs[np, vcur])
            (vnext == 0 || side[vnext] == 0) && continue
            # cut edge vcur (left) -> vnext (right)

            in.nverts >= in.capacity - 1 && return false   # need 2 new slots

            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            nx = (wa*in.positions[1,vcur] + wb*in.positions[1,vnext]) * invw
            ny = (wa*in.positions[2,vcur] + wb*in.positions[2,vnext]) * invw
            nz = (wa*in.positions[3,vcur] + wb*in.positions[3,vnext]) * invw

            # New vertex on the LEFT side, replacing vnext from vcur's POV
            new_left = (in.nverts += 1)
            in.positions[1, new_left] = nx
            in.positions[2, new_left] = ny
            in.positions[3, new_left] = nz
            in.pnbrs[1, new_left] = Int32(vcur)
            in.pnbrs[2, new_left] = Int32(0)
            in.pnbrs[3, new_left] = Int32(0)
            in.pnbrs[np, vcur] = Int32(new_left)
            side[new_left] = Int32(0)

            # New vertex on the RIGHT side, replacing vcur from vnext's POV
            new_right = (in.nverts += 1)
            in.positions[1, new_right] = nx
            in.positions[2, new_right] = ny
            in.positions[3, new_right] = nz
            in.pnbrs[1, new_right] = Int32(vnext)
            in.pnbrs[2, new_right] = Int32(0)
            in.pnbrs[3, new_right] = Int32(0)
            side[new_right] = Int32(1)

            for npnxt in 1:3
                if in.pnbrs[npnxt, vnext] == Int32(vcur)
                    in.pnbrs[npnxt, vnext] = Int32(new_right)
                    break
                end
            end
        end
    end

    # Step 4: walk faces around new vertices to link them. Same structure
    # as clip_plane!'s linker — original vertices form bridges between
    # pairs of new vertices on the same face.
    @inbounds for vstart in (onv+1):in.nverts
        vcur = vstart
        vnext = Int(in.pnbrs[1, vcur])
        while true
            np = find_back3(in.pnbrs, vnext, vcur)
            vcur = vnext
            vnext = Int(in.pnbrs[next3(np), vcur])
            vcur <= onv || break
        end
        in.pnbrs[3, vstart] = Int32(vcur)
        in.pnbrs[2, vcur]   = Int32(vstart)
    end

    # Step 5: compact into out0 (side=0) and out1 (side=1). After this
    # loop side[v] holds the new index of v in whichever output it landed
    # in — used to reindex pnbrs below.
    #
    # The voxelize stack aliases in === out0 (each pushed child reuses
    # the popped slot). Cache `total_verts` before zeroing the outputs
    # so the loop bound isn't clobbered. Within the loop, copying
    # in.positions[:, v] to out0.positions[:, dst] is safe because
    # `dst ≤ v` always (left-side verts in 1..v-1 are at most v-1).
    total_verts = in.nverts
    out0.nverts = 0
    out1.nverts = 0
    @inbounds for v in 1:total_verts
        if side[v] == 0
            out0.nverts += 1
            dst = out0.nverts
            out0.positions[1, dst] = in.positions[1, v]
            out0.positions[2, dst] = in.positions[2, v]
            out0.positions[3, dst] = in.positions[3, v]
            out0.pnbrs[1, dst] = in.pnbrs[1, v]
            out0.pnbrs[2, dst] = in.pnbrs[2, v]
            out0.pnbrs[3, dst] = in.pnbrs[3, v]
            side[v] = Int32(dst)
        else
            out1.nverts += 1
            dst = out1.nverts
            out1.positions[1, dst] = in.positions[1, v]
            out1.positions[2, dst] = in.positions[2, v]
            out1.positions[3, dst] = in.positions[3, v]
            out1.pnbrs[1, dst] = in.pnbrs[1, v]
            out1.pnbrs[2, dst] = in.pnbrs[2, v]
            out1.pnbrs[3, dst] = in.pnbrs[3, v]
            side[v] = Int32(dst)
        end
    end

    @inbounds for v in 1:out0.nverts, np in 1:3
        old = Int(out0.pnbrs[np, v])
        out0.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    @inbounds for v in 1:out1.nverts, np in 1:3
        old = Int(out1.pnbrs[np, v])
        out1.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    return true
end

# Copy positions/pnbrs/nverts from `src` into `dst`. Used by the
# split_coord! pass-through cases.
@inline function _copy_polytope!(dst::FlatPolytope{3,T}, src::FlatPolytope{3,T}) where {T}
    n = src.nverts
    @assert dst.capacity >= n
    @inbounds for v in 1:n, k in 1:3
        dst.positions[k, v] = src.positions[k, v]
        dst.pnbrs[k, v]     = src.pnbrs[k, v]
    end
    dst.nverts = n
    return dst
end

# ---------------------------------------------------------------------------
# get_ibox — bounding-box index range
# ---------------------------------------------------------------------------

"""
    get_ibox(poly, d) -> (ibox_lo, ibox_hi)

Return the integer index range `(ibox_lo, ibox_hi)` of grid cells of
size `d` covering `poly`. `ibox_lo` is `floor(min/d)` and `ibox_hi` is
`ceil(max/d)`, both as `NTuple{3,Int}`. The voxel covered by indices
`(i,j,k)` spans `[i*d, (i+1)*d)` along each axis.

Mirrors `r3d_get_ibox` in `src/v3d.c`. Origin of the grid is at the
spatial origin.
"""
function get_ibox(poly::FlatPolytope{3,T}, d::NTuple{3,T}) where {T}
    poly.nverts <= 0 && return ((0,0,0), (0,0,0))
    minx = T(Inf); miny = T(Inf); minz = T(Inf)
    maxx = T(-Inf); maxy = T(-Inf); maxz = T(-Inf)
    @inbounds for v in 1:poly.nverts
        x = poly.positions[1,v]; y = poly.positions[2,v]; z = poly.positions[3,v]
        x < minx && (minx = x); x > maxx && (maxx = x)
        y < miny && (miny = y); y > maxy && (maxy = y)
        z < minz && (minz = z); z > maxz && (maxz = z)
    end
    lo = (Int(floor(minx / d[1])),
          Int(floor(miny / d[2])),
          Int(floor(minz / d[3])))
    hi = (Int(ceil(maxx / d[1])),
          Int(ceil(maxy / d[2])),
          Int(ceil(maxz / d[3])))
    return (lo, hi)
end

# ---------------------------------------------------------------------------
# voxelize! — grid voxelization via stack-based bisection
# ---------------------------------------------------------------------------

"""
    VoxelizeWorkspace{D,T}(capacity = 512)

Pre-allocated workspace for [`voxelize!`](@ref). Owns a stack of
`FlatPolytope{D,T}`s and a per-leaf moments scratch — sized so that
`voxelize!` runs allocation-free for grids up to ~256^D.

Construct once, reuse across many `voxelize!` calls. Parametric on
both dimension `D` (2 or 3) and coordinate type `T`.
"""
mutable struct VoxelizeWorkspace{D,T}
    polys::Vector{FlatPolytope{D,T}}             # depth ≥ ceil(log2 N) per axis × D + 1
    iboxes::Vector{Tuple{NTuple{D,Int},NTuple{D,Int}}}   # (lo, hi) pair per stack level
    moment_scratch::Vector{T}                    # nmom-sized leaf-moment buffer
    moment_order::Int                            # currently allocated order
    capacity::Int
end

function VoxelizeWorkspace{D,T}(capacity::Int = 512;
                                max_depth::Int = 64) where {D,T}
    polys  = [FlatPolytope{D,T}(capacity) for _ in 1:max_depth]
    iboxes = Vector{Tuple{NTuple{D,Int},NTuple{D,Int}}}(undef, max_depth)
    VoxelizeWorkspace{D,T}(polys, iboxes, T[], -1, capacity)
end

# Ensure the stack can hold at least `n` entries by growing it on demand.
function _ensure_stack!(ws::VoxelizeWorkspace{D,T}, n::Int) where {D,T}
    z = ntuple(_ -> 0, Val(D))
    while length(ws.polys) < n
        push!(ws.polys, FlatPolytope{D,T}(ws.capacity))
        push!(ws.iboxes, (z, z))
    end
    return ws
end

@inline function _ensure_voxelize_moment_scratch!(ws::VoxelizeWorkspace{D,T}, order::Int) where {D,T}
    if order > ws.moment_order
        resize!(ws.moment_scratch, num_moments(D, order))
        ws.moment_order = order
    end
    return ws.moment_scratch
end

"""
    voxelize_fold!(callback::F, state, poly::FlatPolytope{3,T},
                   ibox_lo, ibox_hi, d::NTuple{3,T}, order;
                   workspace = nothing) where {F,T} -> state

Walk the same `r3d_voxelize` bisection recursion as [`voxelize!`](@ref),
but at each touched leaf cell `(i, j, k)` (1-based, relative to
`ibox_lo`) call

    state = callback(state, i, j, k, m)

where `m` is the per-leaf moment vector of length
`R3D.num_moments(3, order)`. R3D writes nothing into a destination
grid — the consumer chooses how to fold the moment vector into
`state`.

This lets callers fuse downstream contractions (`dot(coeffs, m)`,
SpMV against a basis-projection matrix, weighted accumulation, etc.)
into the leaf step at zero overhead — the compiler specializes on the
callback's type and inlines the closure into R3D's leaf branch. For
`order ≥ 2` the bandwidth saving over `voxelize! + post-loop` is
proportional to `num_moments(3, order)` (e.g. 20× at order=3 in 3D).

# Contract for the callback

- Signature: `(state, i, j, k, m::AbstractVector{T}) -> new_state`.
- `m` is a **view into the workspace's moment scratch** — it's
  overwritten on the next leaf. Consume `m` within the callback or
  copy it out; do not retain a reference past the call.
- Return type should be invariant across calls (same as `Base.foldl`).
- Leaf visitation order is stack-LIFO from the recursion, **not**
  lexicographic over `(i,j,k)`. Folds that don't depend on order
  (`+=`, `max`, `dot-into-cell`) are fine; ordered folds need to
  sort afterward.
- Empty-leaf cells (where the polytope didn't intersect) are skipped:
  the callback is called only for non-empty leaves.

# Example — fused dot product into a scalar grid

```julia
ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
coeffs = …                                  # length-`nmom` basis projection
scalar_grid = zeros(Float64, ni, nj, nk)
R3D.Flat.voxelize_fold!(scalar_grid, poly, (0,0,0), (ni,nj,nk), d, 3;
                          workspace = ws) do acc, i, j, k, m
    @inbounds acc[i, j, k] += dot(coeffs, m)
    acc
end
```

`voxelize!` itself is now a one-liner over `voxelize_fold!`.
"""
function voxelize_fold!(callback::F,
                        state,
                        poly::FlatPolytope{3,T},
                        ibox_lo::NTuple{3,Int}, ibox_hi::NTuple{3,Int},
                        d::NTuple{3,T}, order::Int;
                        workspace::Union{Nothing,VoxelizeWorkspace{3,T}} = nothing
                        ) where {F,T}
    ni = ibox_hi[1] - ibox_lo[1]
    nj = ibox_hi[2] - ibox_lo[2]
    nk = ibox_hi[3] - ibox_lo[3]
    (poly.nverts <= 0 || ni <= 0 || nj <= 0 || nk <= 0) && return state

    ws = workspace === nothing ? VoxelizeWorkspace{3,T}(poly.capacity) : workspace
    log2c(x) = x <= 1 ? 0 : ceil(Int, log2(x))
    max_depth = log2c(ni) + log2c(nj) + log2c(nk) + 2
    _ensure_stack!(ws, max_depth)
    moments_buf = _ensure_voxelize_moment_scratch!(ws, order)

    _copy_polytope!(ws.polys[1], poly)
    ws.iboxes[1] = (ibox_lo, ibox_hi)
    nstack = 1

    @inbounds while nstack > 0
        cur = ws.polys[nstack]
        lo, hi = ws.iboxes[nstack]
        nstack -= 1

        cur.nverts <= 0 && continue

        sx = hi[1] - lo[1]
        sy = hi[2] - lo[2]
        sz = hi[3] - lo[3]
        spax = 1; dmax = sx
        if sy > dmax; dmax = sy; spax = 2; end
        if sz > dmax; dmax = sz; spax = 3; end

        if dmax == 1
            moments!(moments_buf, cur, order)
            i0 = lo[1] - ibox_lo[1] + 1
            j0 = lo[2] - ibox_lo[2] + 1
            k0 = lo[3] - ibox_lo[3] + 1
            state = callback(state, i0, j0, k0, moments_buf)
            continue
        end

        half = dmax >> 1
        split_index = lo[spax] + half
        split_pos = T(split_index) * d[spax]

        if nstack + 2 > length(ws.polys)
            _ensure_stack!(ws, nstack + 2)
        end
        out0 = ws.polys[nstack + 1]
        out1 = ws.polys[nstack + 2]
        split_coord!(cur, out0, out1, split_pos, spax)

        hi_left  = (spax == 1 ? split_index : hi[1],
                    spax == 2 ? split_index : hi[2],
                    spax == 3 ? split_index : hi[3])
        lo_right = (spax == 1 ? split_index : lo[1],
                    spax == 2 ? split_index : lo[2],
                    spax == 3 ? split_index : lo[3])
        ws.iboxes[nstack + 1] = (lo, hi_left)
        ws.iboxes[nstack + 2] = (lo_right, hi)
        nstack += 2
    end
    return state
end

"""
    voxelize!(dest_grid, poly, ibox_lo, ibox_hi, d, order;
              workspace = nothing) -> dest_grid

Voxelize `poly` onto a regular Cartesian grid covering index range
`(ibox_lo, ibox_hi)` with cell spacing `d`. `dest_grid` is laid out as
`(nmom, ni, nj, nk)` where `nmom = num_moments(3, order)` and
`ni = ibox_hi[1] - ibox_lo[1]` etc. Voxel `(i,j,k)` covers the spatial
region `[i*d_x, (i+1)*d_x) × …` (origin at the spatial origin), with
moments contiguous along the first axis for fast accumulation.

Implemented as a thin wrapper over [`voxelize_fold!`](@ref) — the
fold writes the per-leaf moment vector into the corresponding column
of `dest_grid`. Use `voxelize_fold!` directly if you can fuse a
downstream contraction (e.g. `dot(coeffs, m)` for basis projection).

`poly` is left untouched — the routine copies it into the workspace
and bisects there. Mirrors `r3d_voxelize` in `src/v3d.c`.
"""
function voxelize!(dest_grid::AbstractArray{T,4},
                   poly::FlatPolytope{3,T},
                   ibox_lo::NTuple{3,Int}, ibox_hi::NTuple{3,Int},
                   d::NTuple{3,T}, order::Int;
                   workspace::Union{Nothing,VoxelizeWorkspace{3,T}} = nothing) where {T}
    nmom = num_moments(3, order)
    @assert size(dest_grid, 1) == nmom
    @assert size(dest_grid, 2) >= ibox_hi[1] - ibox_lo[1]
    @assert size(dest_grid, 3) >= ibox_hi[2] - ibox_lo[2]
    @assert size(dest_grid, 4) >= ibox_hi[3] - ibox_lo[3]
    voxelize_fold!(dest_grid, poly, ibox_lo, ibox_hi, d, order;
                   workspace = workspace) do grid, i, j, k, m
        @inbounds for mi in 1:length(m)
            grid[mi, i, j, k] += m[mi]
        end
        return grid
    end
    return dest_grid
end

"""
    voxelize(poly, d, order; ibox = nothing, workspace = nothing) -> dest_grid

Allocate a fresh `(nmom, ni, nj, nk)` grid sized to `poly`'s
bounding-box cells (or the supplied `ibox`) and voxelize.

Convenience wrapper around [`voxelize!`](@ref); for hot loops, allocate
the destination grid once and call `voxelize!` directly.
"""
function voxelize(poly::FlatPolytope{3,T}, d::NTuple{3,T}, order::Int;
                  ibox::Union{Nothing,Tuple{NTuple{3,Int},NTuple{3,Int}}} = nothing,
                  workspace::Union{Nothing,VoxelizeWorkspace{T}} = nothing) where {T}
    if ibox === nothing
        lo, hi = get_ibox(poly, d)
    else
        lo, hi = ibox
    end
    ni = hi[1] - lo[1]; nj = hi[2] - lo[2]; nk = hi[3] - lo[3]
    nmom = num_moments(3, order)
    grid = zeros(T, nmom, max(ni, 1), max(nj, 1), max(nk, 1))
    voxelize!(grid, poly, lo, hi, d, order; workspace = workspace)
    return grid, lo, hi
end

# ===========================================================================
# D ≥ 4 voxelize_fold! / voxelize! / get_ibox.
#
# Implementation note: the per-split step uses the "two-clips" pattern
# (copy `cur` into `out0` then `clip!` it with the kept-side plane;
# repeat for `out1` with the negated plane) instead of porting
# `split_coord!` to D ≥ 4. This costs ~2× the per-split work compared
# to a true `split_coord!` port (~250 LOC of finds[][] propagation),
# but reuses the already-debugged D ≥ 4 `clip!` kernel and ships
# correctness immediately. Swap in a real `split_coord!` later if perf
# matters.
#
# Leaf moment vector at D ≥ 4 has length 1 (only order = 0 is supported
# today via `_reduce_nd_zeroth!`); higher orders raise the same
# informative error as elsewhere. Useful operations the consumer can
# fold include "sum polytope volume into each cell", "histogram cells
# by polytope mass", "max-fold for visualization", etc.
# ===========================================================================

# Copy positions + pnbrs + finds + nverts. Used by the D ≥ 4 voxelize
# split step when copying `cur` into the two output halves before
# clipping each.
@inline function _copy_polytope_nd!(dst::FlatPolytope{D,T},
                                     src::FlatPolytope{D,T}) where {D,T}
    @assert D >= 4 "use _copy_polytope! / _copy_polytope_2d! for lower dimensions"
    n = src.nverts
    @assert dst.capacity >= n
    @inbounds for v in 1:n
        for k in 1:D
            dst.positions[k, v] = src.positions[k, v]
            dst.pnbrs[k, v]     = src.pnbrs[k, v]
            dst.facets[k, v]    = src.facets[k, v]
        end
        for j in 1:D, i in 1:D
            dst.finds[i, j, v] = src.finds[i, j, v]
        end
    end
    dst.nverts  = n
    dst.nfaces  = src.nfaces
    dst.nfacets = src.nfacets

    # Per-facet metadata: grow dst's storage to fit, then copy.
    _grow_facet_metadata!(dst, src.nfacets)
    @inbounds for j in 1:src.nfacets
        for i in 1:D
            dst.facet_normals[i, j] = src.facet_normals[i, j]
        end
        dst.facet_distances[j] = src.facet_distances[j]
    end
    return dst
end

"""
    walk_facet_vertices(callback::F, poly::FlatPolytope{D,T}, fid::Integer) where {F,D,T}

Call `callback(v)` for each vertex `v` incident to facet `fid` in the
D ≥ 4 polytope `poly`. Linear scan over the polytope's vertices —
zero allocations.
"""
function walk_facet_vertices(callback::F, poly::FlatPolytope{D,T},
                             fid::Integer) where {F,D,T}
    @assert D >= 4 "walk_facet_vertices is for D ≥ 4 only"
    fid_i = Int32(fid)
    @inbounds for v in 1:poly.nverts
        # A vertex is on facet `fid` iff some slot of its facets row
        # equals `fid`.
        on_facet = false
        for k in 1:D
            if poly.facets[k, v] == fid_i
                on_facet = true
                break
            end
        end
        on_facet && callback(v)
    end
    return nothing
end

"""
    walk_facets(callback::F, poly::FlatPolytope{D,T}) where {F,D,T}

Enumerate each facet ID of the D ≥ 4 polytope exactly once, calling
`callback(fid::Int)` for each. Pair with [`walk_facet_vertices`](@ref)
to get the vertex set per facet.

Allocates a `BitVector` of length `poly.nfacets` for visited tracking;
free if `poly.nfacets` is small.
"""
function walk_facets(callback::F, poly::FlatPolytope{D,T}) where {F,D,T}
    @assert D >= 4 "walk_facets is for D ≥ 4 only"
    poly.nfacets <= 0 && return nothing
    seen = falses(poly.nfacets)
    @inbounds for v in 1:poly.nverts, k in 1:D
        fid = Int(poly.facets[k, v])
        (fid <= 0 || fid > poly.nfacets) && continue
        if !seen[fid]
            seen[fid] = true
            callback(fid)
        end
    end
    return nothing
end

"""
    get_ibox(poly::FlatPolytope{D,T}, d::NTuple{D,T}) -> (lo, hi) where {D ≥ 4}

D-generic bounding-box index range. Returns `NTuple{D,Int}` corners
`floor.(min_pos ./ d)` and `ceil.(max_pos ./ d)`.
"""
function get_ibox(poly::FlatPolytope{D,T}, d::NTuple{D,T}) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 get_ibox methods for those dimensions"
    poly.nverts <= 0 && return (ntuple(_ -> 0, Val(D)), ntuple(_ -> 0, Val(D)))
    lo = ntuple(_ -> T(Inf),  Val(D))
    hi = ntuple(_ -> T(-Inf), Val(D))
    @inbounds for v in 1:poly.nverts
        lo = ntuple(k -> min(lo[k], poly.positions[k, v]), Val(D))
        hi = ntuple(k -> max(hi[k], poly.positions[k, v]), Val(D))
    end
    lo_i = ntuple(k -> Int(floor(lo[k] / d[k])), Val(D))
    hi_i = ntuple(k -> Int(ceil(hi[k]  / d[k])), Val(D))
    return (lo_i, hi_i)
end

"""
    voxelize_fold!(callback::F, state, poly::FlatPolytope{D,T},
                   ibox_lo::NTuple{D,Int}, ibox_hi::NTuple{D,Int},
                   d::NTuple{D,T}, order::Int;
                   workspace = nothing) where {F,D,T} -> state

D-generic (D ≥ 4) leaf-callback voxelization. At each non-empty leaf
cell with index tuple `idx::NTuple{D,Int}` (1-based, relative to
`ibox_lo`), calls

    state = callback(state, idx, m)

where `m` is the per-leaf moment vector of length `num_moments(D, order)`.

**Order limitation at D ≥ 4**: only `order = 0` is supported today
(higher-order moments need Lasserre or a D-generic Koehl port — see
`docs/phase3_status.md`). At order = 0, `m` has length 1 and
`m[1]` is the polytope's intersection volume in the leaf cell.

The recursion uses a two-clips-per-split shortcut instead of a true
`split_coord!` port to D ≥ 4 — correct but ~2× the per-split work
of an in-place split. Performance-tune if a real use case needs it.
"""
function voxelize_fold!(callback::F,
                        state,
                        poly::FlatPolytope{D,T},
                        ibox_lo::NTuple{D,Int}, ibox_hi::NTuple{D,Int},
                        d::NTuple{D,T}, order::Int;
                        workspace::Union{Nothing,VoxelizeWorkspace{D,T}} = nothing
                        ) where {F,D,T}
    @assert D >= 4 "use the D=2 / D=3 voxelize_fold! methods for those dimensions"
    if D == 4
        # D = 4 supports any order via Lasserre's recursive face decomposition.
    else
        @assert order == 0 "voxelize_fold! at D = $D currently supports only order = 0 " *
                           "(D = 4 supports order ≥ 1 via Lasserre; D = 5 / D = 6 still need " *
                           "additional codim-face tracking — see docs/d4plus_finalization_plan.md)"
    end
    poly.nverts <= 0 && return state
    sizes = ntuple(k -> ibox_hi[k] - ibox_lo[k], Val(D))
    any(s -> s <= 0, sizes) && return state

    ws = workspace === nothing ? VoxelizeWorkspace{D,T}(poly.capacity) : workspace
    log2c(x) = x <= 1 ? 0 : ceil(Int, log2(x))
    max_depth = sum(log2c, sizes) + 2
    _ensure_stack!(ws, max_depth)
    moments_buf = _ensure_voxelize_moment_scratch!(ws, order)

    _copy_polytope_nd!(ws.polys[1], poly)
    ws.iboxes[1] = (ibox_lo, ibox_hi)
    nstack = 1

    @inbounds while nstack > 0
        cur = ws.polys[nstack]
        lo, hi = ws.iboxes[nstack]
        nstack -= 1

        cur.nverts <= 0 && continue

        # Find the longest axis to split. Wrapped in a helper so the
        # iteration's mutation of `spax` stays local — otherwise the
        # outer-scope `spax` gets boxed when captured by the `ntuple`
        # closures below, costing ~1 KB / cell on the hot loop.
        spax, dmax = _argmax_extent(lo, hi, Val(D))

        if dmax == 1
            moments!(moments_buf, cur, order)
            idx = _shifted_index(lo, ibox_lo, Val(D))
            state = callback(state, idx, moments_buf)
            continue
        end

        # Bisect along spax at the midpoint cell boundary using a pair
        # of axis-aligned single-plane clips. The single-plane `clip!`
        # overload (added alongside the D = 2 / D = 3 versions) avoids
        # the per-leaf `[plane]` Vector{Plane} allocation that the
        # multi-plane API would pay; combined with the `_copy_polytope_nd!`
        # buffer reuse, this makes the bisection loop fully heap-free
        # in steady state.
        half = dmax >> 1
        split_index = lo[spax] + half
        split_pos = T(split_index) * d[spax]

        if nstack + 2 > length(ws.polys)
            _ensure_stack!(ws, nstack + 2)
        end
        # `cur === ws.polys[nstack+1]` because the workspace stack reuses
        # the popped slot. Save `cur` into `out1` (a fresh slot) BEFORE
        # we clip `cur` in place to become `out0`.
        out0 = ws.polys[nstack + 1]   # alias of cur — becomes negative half
        out1 = ws.polys[nstack + 2]   # fresh slot — gets positive half

        plane_pos = Plane{D,T}(Vec{D,T}(_axis_unit(spax,  one(T), Val(D))), -split_pos)
        plane_neg = Plane{D,T}(Vec{D,T}(_axis_unit(spax, -one(T), Val(D))),  split_pos)

        _copy_polytope_nd!(out1, cur)
        clip!(out0, plane_neg)        # cur (=out0) clipped in place
        clip!(out1, plane_pos)

        hi_left  = _replace_at(hi, spax, split_index, Val(D))
        lo_right = _replace_at(lo, spax, split_index, Val(D))
        ws.iboxes[nstack + 1] = (lo, hi_left)
        ws.iboxes[nstack + 2] = (lo_right, hi)
        nstack += 2
    end
    return state
end

# Helpers for the D ≥ 4 voxelize_fold! hot loop. Factored out so the
# `ntuple` closures don't capture mutable locals from the parent
# scope (which would force boxing — the original cause of the
# per-cell ~1 KB allocation regression).
@inline function _argmax_extent(lo::NTuple{D,Int}, hi::NTuple{D,Int},
                                ::Val{D}) where {D}
    spax = 1
    dmax = hi[1] - lo[1]
    @inbounds for k in 2:D
        s = hi[k] - lo[k]
        if s > dmax
            dmax = s
            spax = k
        end
    end
    return spax, dmax
end

@inline _shifted_index(lo::NTuple{D,Int}, ibox_lo::NTuple{D,Int},
                       ::Val{D}) where {D} =
    ntuple(k -> lo[k] - ibox_lo[k] + 1, Val(D))

@inline _axis_unit(axis::Int, sign::T, ::Val{D}) where {D,T} =
    ntuple(k -> k == axis ? sign : zero(T), Val(D))

@inline _replace_at(t::NTuple{D,Int}, k::Int, v::Int, ::Val{D}) where {D} =
    ntuple(i -> i == k ? v : t[i], Val(D))

"""
    voxelize!(dest_grid::AbstractArray{T,DG}, poly::FlatPolytope{D,T},
              ibox_lo::NTuple{D,Int}, ibox_hi::NTuple{D,Int},
              d::NTuple{D,T}, order::Int; workspace = nothing) where {D ≥ 4}

D-generic voxelization (D ≥ 4). `dest_grid` is shape `(nmom, n1, n2, …, nD)`
with `DG = D + 1`; only `order = 0` is supported today, so `nmom = 1`.

Implemented as a thin wrapper over [`voxelize_fold!`](@ref).
"""
function voxelize!(dest_grid::AbstractArray{T},
                   poly::FlatPolytope{D,T},
                   ibox_lo::NTuple{D,Int}, ibox_hi::NTuple{D,Int},
                   d::NTuple{D,T}, order::Int;
                   workspace::Union{Nothing,VoxelizeWorkspace{D,T}} = nothing
                   ) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 voxelize! methods for those dimensions"
    @assert ndims(dest_grid) == D + 1 "voxelize! D=$D needs a $(D+1)-dim grid"
    nmom = num_moments(D, order)
    @assert size(dest_grid, 1) == nmom
    voxelize_fold!(dest_grid, poly, ibox_lo, ibox_hi, d, order;
                   workspace = workspace) do grid, idx, m
        @inbounds for mi in 1:length(m)
            grid[mi, idx...] += m[mi]
        end
        return grid
    end
    return dest_grid
end

"""
    voxelize(poly::FlatPolytope{D,T}, d, order; ibox=nothing, workspace=nothing)
        -> (grid, ibox_lo, ibox_hi)  where {D ≥ 4}

D-generic voxelization (D ≥ 4) allocating wrapper that mirrors the
D = 2 / D = 3 convenience signature. Allocates a fresh
`(nmom, n1, n2, …, nD)` grid sized to `poly`'s bounding-box cells (or
the supplied `ibox`) and returns it together with the index range used.

For tight loops, allocate the destination grid once and call
[`voxelize!`](@ref) directly; this wrapper exists for parity with the
lower-D API and ad-hoc one-shot calls.
"""
function voxelize(poly::FlatPolytope{D,T}, d::NTuple{D,T}, order::Int;
                  ibox::Union{Nothing,Tuple{NTuple{D,Int},NTuple{D,Int}}} = nothing,
                  workspace::Union{Nothing,VoxelizeWorkspace{D,T}} = nothing
                  ) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 voxelize wrapper for those dimensions"
    if ibox === nothing
        lo, hi = get_ibox(poly, d)
    else
        lo, hi = ibox
    end
    sizes = ntuple(k -> max(hi[k] - lo[k], 1), Val(D))
    nmom = num_moments(D, order)
    grid = zeros(T, nmom, sizes...)
    voxelize!(grid, poly, lo, hi, d, order; workspace = workspace)
    return grid, lo, hi
end

# ===========================================================================
# D = 2 — same SoA buffer machinery, parallel set of methods mirroring
# `src/r2d.c` and `src/v2d.c`. Notation in this block follows the 2D code:
# `pnbrs[k, v]` for k ∈ {1, 2}; pnbrs == 0 is the "unset" sentinel
# (the upstream C uses -1 in 0-based indexing).
# ===========================================================================

# ---------------------------------------------------------------------------
# init_box! / box (D=2)
# ---------------------------------------------------------------------------

"""
    init_box!(poly::FlatPolytope{2,T}, lo, hi)

Initialize as the axis-aligned rectangle with corners `lo` and `hi`.
Vertex labelling (CCW from lower-left) and pnbrs table mirror
`r2d_init_box` (`src/r2d.c`):

    4 — 3
    |   |
    1 — 2
"""
function init_box!(poly::FlatPolytope{2,T},
                   lo::AbstractVector, hi::AbstractVector) where {T}
    @assert poly.capacity >= 4
    poly.nverts = 4

    @inbounds begin
        poly.positions[1,1] = lo[1]; poly.positions[2,1] = lo[2]
        poly.positions[1,2] = hi[1]; poly.positions[2,2] = lo[2]
        poly.positions[1,3] = hi[1]; poly.positions[2,3] = hi[2]
        poly.positions[1,4] = lo[1]; poly.positions[2,4] = hi[2]

        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(4)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(4); poly.pnbrs[2,3] = Int32(2)
        poly.pnbrs[1,4] = Int32(1); poly.pnbrs[2,4] = Int32(3)
    end
    return poly
end

"Convenience: build a fresh 2D box (axis-aligned rectangle)."
function box(lo::NTuple{2,Real}, hi::NTuple{2,Real}; capacity::Int = 256)
    p = FlatPolytope{2,Float64}(capacity)
    init_box!(p, [lo...], [hi...])
    return p
end

# ---------------------------------------------------------------------------
# clip! (D=2)
# ---------------------------------------------------------------------------

"""
    clip!(poly::FlatPolytope{2,T}, planes) -> Bool

Clip a 2D polytope in-place against `planes`. Same algorithm as the 3D
version but with single-vertex insertion per cut edge and the
1-D face walk from `r2d.c`.
"""
function clip!(poly::FlatPolytope{2,T},
               planes::AbstractVector{Plane{2,T}}) where {T}
    poly.nverts <= 0 && return false
    @inbounds for plane in planes
        ok = clip_plane!(poly, plane)
        ok || return false
        poly.nverts == 0 && return true
    end
    return true
end

"""
    clip!(poly::FlatPolytope{2,T}, plane::Plane{2,T}) -> Bool

Single-plane convenience overload, parallel to the D = 3 / D ≥ 4
versions. Saves the `[plane]` array allocation for hot loops.
"""
function clip!(poly::FlatPolytope{2,T}, plane::Plane{2,T}) where {T}
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

function clip_plane!(poly::FlatPolytope{2,T}, plane::Plane{2,T}) where {T}
    onv = poly.nverts
    sdists = poly.sdists
    clipped = poly.clipped

    # Step 1: signed distances
    smin = T(Inf); smax = T(-Inf)
    @inbounds for v in 1:onv
        x = poly.positions[1, v]; y = poly.positions[2, v]
        s = plane.d + plane.n[1]*x + plane.n[2]*y
        sdists[v] = s
        s < smin && (smin = s)
        s > smax && (smax = s)
    end
    smin >= 0 && return true
    if smax <= 0
        poly.nverts = 0
        return true
    end

    @inbounds for v in 1:onv
        clipped[v] = sdists[v] < 0 ? Int32(1) : Int32(0)
    end

    # Step 2: insert ONE new vertex per cut edge. Mirrors r2d.c lines
    # 80-93 — note the asymmetry: pnbrs[1-np] back-links to vcur, while
    # pnbrs[np] is left "unset" (0 in our 1-based scheme) for the face
    # walk to fill in.
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:2
            vnext = Int(poly.pnbrs[np, vcur])
            vnext == 0 && continue
            clipped[vnext] == 0 && continue

            poly.nverts >= poly.capacity && return false
            new_idx = (poly.nverts += 1)

            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            poly.positions[1, new_idx] = (wa*poly.positions[1,vcur] + wb*poly.positions[1,vnext]) * invw
            poly.positions[2, new_idx] = (wa*poly.positions[2,vcur] + wb*poly.positions[2,vnext]) * invw

            other = 3 - np         # 1↔2 swap (mirrors C's 1-np with 0/1)
            poly.pnbrs[other, new_idx] = Int32(vcur)
            poly.pnbrs[np, new_idx]    = Int32(0)        # unset, filled by Step 3
            poly.pnbrs[np, vcur]       = Int32(new_idx)

            clipped[new_idx] = Int32(0)
        end
    end

    # Step 3: walk pnbrs[1] (the "next around polygon" slot) for each
    # new vertex whose [2] slot is still unset, find the matching new
    # vertex on the opposite side of the cut, and double-link them.
    # See r2d.c lines 97-105.
    @inbounds for vstart in (onv+1):poly.nverts
        poly.pnbrs[2, vstart] != 0 && continue
        vcur = Int(poly.pnbrs[1, vstart])
        # walk pnbrs[1] from vcur until we hit another new vertex (> onv)
        while vcur <= onv
            vcur = Int(poly.pnbrs[1, vcur])
        end
        poly.pnbrs[2, vstart] = Int32(vcur)
        poly.pnbrs[1, vcur]   = Int32(vstart)
    end

    # Step 4: compact (same pattern as 3D)
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                poly.positions[1, numunclipped] = poly.positions[1, v]
                poly.positions[2, numunclipped] = poly.positions[2, v]
                poly.pnbrs[1, numunclipped] = poly.pnbrs[1, v]
                poly.pnbrs[2, numunclipped] = poly.pnbrs[2, v]
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

# ---------------------------------------------------------------------------
# moments / moments! (D=2) — Koehl recursion, 2D variant
# ---------------------------------------------------------------------------

function moments(poly::FlatPolytope{2,T}, order::Integer) where {T}
    out = zeros(T, num_moments(2, order))
    moments!(out, poly, order)
    return out
end

# In 2D the Koehl recursion uses two 2-layer arrays of size (order+1, 2)
# rather than the 3D (order+1, order+1, 2) cubes. Reuse the 3D Sm/Dm/Cm
# fields' first column slice — but the indexing patterns differ enough
# that we just lazy-allocate fresh 2D scratch (cheap, capacity-bounded).
@inline function _ensure_moment_scratch_2d!(poly::FlatPolytope{2,T}, order::Int) where {T}
    if order > poly.moment_order
        np1 = order + 1
        # Reuse the same Sm/Cm/Dm fields but with shape (np1, 1, 2). We
        # ignore the middle axis in 2D; this keeps the struct layout
        # identical to the 3D path.
        poly.Sm = Array{T,3}(undef, np1, 1, 2)
        poly.Dm = Array{T,3}(undef, np1, 1, 2)
        poly.Cm = Array{T,3}(undef, np1, 1, 2)
        poly.moment_order = order
    end
    return nothing
end

function moments!(out::AbstractVector{T},
                  poly::FlatPolytope{2,T},
                  order::Integer) where {T}
    @assert length(out) >= num_moments(2, order)
    fill!(out, zero(T))
    poly.nverts <= 0 && return out

    _ensure_moment_scratch_2d!(poly, Int(order))
    D = poly.Dm     # shape (np1, 1, 2)
    C = poly.Cm
    prevlayer = 1; curlayer = 2

    nv = poly.nverts
    @inbounds for vcur in 1:nv
        vnext = Int(poly.pnbrs[1, vcur])
        v0x = poly.positions[1, vcur]; v0y = poly.positions[2, vcur]
        v1x = poly.positions[1, vnext]; v1y = poly.positions[2, vnext]

        twoa = v0x*v1y - v0y*v1x

        D[1,1,prevlayer] = one(T)
        C[1,1,prevlayer] = one(T)
        out[1] += 0.5 * twoa

        m = 1
        for corder in 1:order
            for i in corder:-1:0
                j = corder - i
                m += 1
                ci = i + 1
                Cv = zero(T); Dv = zero(T)
                if i > 0
                    Cv += v1x * C[ci-1, 1, prevlayer]
                    Dv += v0x * D[ci-1, 1, prevlayer]
                end
                if j > 0
                    Cv += v1y * C[ci, 1, prevlayer]
                    Dv += v0y * D[ci, 1, prevlayer]
                end
                Dv += Cv
                C[ci, 1, curlayer] = Cv
                D[ci, 1, curlayer] = Dv
                out[m] += twoa * Dv
            end
            curlayer  = 3 - curlayer
            prevlayer = 3 - prevlayer
        end
    end

    @inbounds C[1,1,prevlayer] = one(T)
    m = 1
    for corder in 1:order
        for i in corder:-1:0
            j = corder - i
            m += 1
            ci = i + 1
            Cv = zero(T)
            i > 0 && (Cv += C[ci-1, 1, prevlayer])
            j > 0 && (Cv += C[ci, 1, prevlayer])
            C[ci, 1, curlayer] = Cv
            out[m] /= Cv * (corder + 1) * (corder + 2)
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end
    return out
end

# ---------------------------------------------------------------------------
# split_coord! / get_ibox / voxelize! (D=2)
# ---------------------------------------------------------------------------

"""
    split_coord!(in::FlatPolytope{2,T}, out0, out1, coord, ax) -> Bool

2D analog of [`split_coord!(::FlatPolytope{3,T}, …)`](@ref). Splits
along the line `x[ax] == coord` into the negative side `out0` and the
positive side `out1`. Mirrors `r2d_split_coord` in `src/v2d.c`.

Same `in === out0` aliasing semantics as the 3D version: the voxelize
stack reuses the popped slot as `out0`, so the compaction caches
`total_verts` before zeroing the outputs.
"""
function split_coord!(in::FlatPolytope{2,T},
                      out0::FlatPolytope{2,T}, out1::FlatPolytope{2,T},
                      coord::T, ax::Int) where {T}
    if in.nverts <= 0
        out0.nverts = 0
        out1.nverts = 0
        return true
    end

    sdists = in.sdists
    side   = in.clipped

    nright = 0
    @inbounds for v in 1:in.nverts
        sd = in.positions[ax, v] - coord
        sdists[v] = sd
        if sd > 0
            side[v] = Int32(1); nright += 1
        else
            side[v] = Int32(0)
        end
    end

    if nright == 0
        _copy_polytope_2d!(out0, in)
        out1.nverts = 0
        return true
    elseif nright == in.nverts
        _copy_polytope_2d!(out1, in)
        out0.nverts = 0
        return true
    end

    # Step 3: insert TWO new vertices per cut edge (mirrors r2d_split_coord
    # — one for each side, with the unset slot filled in by the linker).
    onv = in.nverts
    @inbounds for vcur in 1:onv
        side[vcur] != 0 && continue
        for np in 1:2
            vnext = Int(in.pnbrs[np, vcur])
            (vnext == 0 || side[vnext] == 0) && continue

            in.nverts >= in.capacity - 1 && return false

            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            nx = (wa*in.positions[1,vcur] + wb*in.positions[1,vnext]) * invw
            ny = (wa*in.positions[2,vcur] + wb*in.positions[2,vnext]) * invw

            other = 3 - np

            # LEFT-side new vertex: replaces vnext from vcur's POV.
            new_left = (in.nverts += 1)
            in.positions[1, new_left] = nx
            in.positions[2, new_left] = ny
            in.pnbrs[other, new_left] = Int32(vcur)
            in.pnbrs[np,    new_left] = Int32(0)
            in.pnbrs[np, vcur] = Int32(new_left)
            side[new_left] = Int32(0)

            # RIGHT-side new vertex: replaces vcur from vnext's POV.
            new_right = (in.nverts += 1)
            in.positions[1, new_right] = nx
            in.positions[2, new_right] = ny
            in.pnbrs[other, new_right] = Int32(0)
            in.pnbrs[np,    new_right] = Int32(vnext)
            in.pnbrs[other, vnext] = Int32(new_right)
            side[new_right] = Int32(1)
        end
    end

    # Step 4: face-walk linker (same as 2D clip)
    @inbounds for vstart in (onv+1):in.nverts
        in.pnbrs[2, vstart] != 0 && continue
        vcur = Int(in.pnbrs[1, vstart])
        while vcur <= onv
            vcur = Int(in.pnbrs[1, vcur])
        end
        in.pnbrs[2, vstart] = Int32(vcur)
        in.pnbrs[1, vcur]   = Int32(vstart)
    end

    # Step 5: compact into out0 and out1. Cache total_verts before
    # zeroing — the voxelize stack aliases in === out0.
    total_verts = in.nverts
    out0.nverts = 0
    out1.nverts = 0
    @inbounds for v in 1:total_verts
        if side[v] == 0
            out0.nverts += 1; dst = out0.nverts
            out0.positions[1, dst] = in.positions[1, v]
            out0.positions[2, dst] = in.positions[2, v]
            out0.pnbrs[1, dst] = in.pnbrs[1, v]
            out0.pnbrs[2, dst] = in.pnbrs[2, v]
            side[v] = Int32(dst)
        else
            out1.nverts += 1; dst = out1.nverts
            out1.positions[1, dst] = in.positions[1, v]
            out1.positions[2, dst] = in.positions[2, v]
            out1.pnbrs[1, dst] = in.pnbrs[1, v]
            out1.pnbrs[2, dst] = in.pnbrs[2, v]
            side[v] = Int32(dst)
        end
    end

    @inbounds for v in 1:out0.nverts, np in 1:2
        old = Int(out0.pnbrs[np, v])
        out0.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    @inbounds for v in 1:out1.nverts, np in 1:2
        old = Int(out1.pnbrs[np, v])
        out1.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    return true
end

@inline function _copy_polytope_2d!(dst::FlatPolytope{2,T}, src::FlatPolytope{2,T}) where {T}
    n = src.nverts
    @assert dst.capacity >= n
    @inbounds for v in 1:n, k in 1:2
        dst.positions[k, v] = src.positions[k, v]
        dst.pnbrs[k, v]     = src.pnbrs[k, v]
    end
    dst.nverts = n
    return dst
end

"""
    get_ibox(poly::FlatPolytope{2,T}, d::NTuple{2,T}) -> (lo, hi)

2D analog of [`get_ibox`](@ref). `d` is `(dx, dy)`.
"""
function get_ibox(poly::FlatPolytope{2,T}, d::NTuple{2,T}) where {T}
    poly.nverts <= 0 && return ((0,0), (0,0))
    minx = T(Inf); miny = T(Inf)
    maxx = T(-Inf); maxy = T(-Inf)
    @inbounds for v in 1:poly.nverts
        x = poly.positions[1,v]; y = poly.positions[2,v]
        x < minx && (minx = x); x > maxx && (maxx = x)
        y < miny && (miny = y); y > maxy && (maxy = y)
    end
    lo = (Int(floor(minx / d[1])), Int(floor(miny / d[2])))
    hi = (Int(ceil(maxx / d[1])),  Int(ceil(maxy / d[2])))
    return (lo, hi)
end

"""
    voxelize_fold!(callback::F, state, poly::FlatPolytope{2,T},
                   ibox_lo, ibox_hi, d::NTuple{2,T}, order;
                   workspace = nothing) where {F,T} -> state

2D analog of [`voxelize_fold!(::F, ::Any, ::FlatPolytope{3,T}, …)`](@ref).
Walks the same `r2d_rasterize` recursion, calling
`state = callback(state, i, j, m)` at each non-empty leaf cell.

`m` is a length-`R3D.num_moments(2, order)` view into the workspace's
moment scratch — overwritten on the next leaf, so consume it within
the callback or copy it out.

Mirrors `r2d_rasterize` in `src/v2d.c`. `voxelize!` for D=2 is a
thin wrapper that writes `m` into the corresponding column of
`dest_grid`.
"""
function voxelize_fold!(callback::F,
                        state,
                        poly::FlatPolytope{2,T},
                        ibox_lo::NTuple{2,Int}, ibox_hi::NTuple{2,Int},
                        d::NTuple{2,T}, order::Int;
                        workspace::Union{Nothing,VoxelizeWorkspace{2,T}} = nothing
                        ) where {F,T}
    ni = ibox_hi[1] - ibox_lo[1]
    nj = ibox_hi[2] - ibox_lo[2]
    (poly.nverts <= 0 || ni <= 0 || nj <= 0) && return state

    ws = workspace === nothing ? VoxelizeWorkspace{2,T}(poly.capacity) : workspace
    log2c(x) = x <= 1 ? 0 : ceil(Int, log2(x))
    max_depth = log2c(ni) + log2c(nj) + 2
    _ensure_stack!(ws, max_depth)
    moments_buf = _ensure_voxelize_moment_scratch!(ws, order)

    _copy_polytope_2d!(ws.polys[1], poly)
    ws.iboxes[1] = (ibox_lo, ibox_hi)
    nstack = 1

    @inbounds while nstack > 0
        cur = ws.polys[nstack]
        lo, hi = ws.iboxes[nstack]
        nstack -= 1

        cur.nverts <= 0 && continue

        sx = hi[1] - lo[1]; sy = hi[2] - lo[2]
        spax = 1; dmax = sx
        if sy > dmax; dmax = sy; spax = 2; end

        if dmax == 1
            moments!(moments_buf, cur, order)
            i0 = lo[1] - ibox_lo[1] + 1
            j0 = lo[2] - ibox_lo[2] + 1
            state = callback(state, i0, j0, moments_buf)
            continue
        end

        half = dmax >> 1
        split_index = lo[spax] + half
        split_pos = T(split_index) * d[spax]

        if nstack + 2 > length(ws.polys)
            _ensure_stack!(ws, nstack + 2)
        end
        out0 = ws.polys[nstack + 1]
        out1 = ws.polys[nstack + 2]
        split_coord!(cur, out0, out1, split_pos, spax)

        hi_left  = (spax == 1 ? split_index : hi[1],
                    spax == 2 ? split_index : hi[2])
        lo_right = (spax == 1 ? split_index : lo[1],
                    spax == 2 ? split_index : lo[2])
        ws.iboxes[nstack + 1] = (lo, hi_left)
        ws.iboxes[nstack + 2] = (lo_right, hi)
        nstack += 2
    end
    return state
end

"""
    voxelize!(dest_grid::AbstractArray{T,3}, poly::FlatPolytope{2,T},
              ibox_lo, ibox_hi, d, order; workspace=nothing)

2D rasterization. `dest_grid` is `(nmom, ni, nj)`. Mirrors
`r2d_rasterize` in `src/v2d.c` (the 2D analog of `r3d_voxelize`).

Implemented as a thin wrapper over [`voxelize_fold!`](@ref) — the
fold writes the per-leaf moment vector into the corresponding column
of `dest_grid`. Use `voxelize_fold!` directly if you can fuse a
downstream contraction.
"""
function voxelize!(dest_grid::AbstractArray{T,3},
                   poly::FlatPolytope{2,T},
                   ibox_lo::NTuple{2,Int}, ibox_hi::NTuple{2,Int},
                   d::NTuple{2,T}, order::Int;
                   workspace::Union{Nothing,VoxelizeWorkspace{2,T}} = nothing) where {T}
    nmom = num_moments(2, order)
    @assert size(dest_grid, 1) == nmom
    @assert size(dest_grid, 2) >= ibox_hi[1] - ibox_lo[1]
    @assert size(dest_grid, 3) >= ibox_hi[2] - ibox_lo[2]
    voxelize_fold!(dest_grid, poly, ibox_lo, ibox_hi, d, order;
                   workspace = workspace) do grid, i, j, m
        @inbounds for mi in 1:length(m)
            grid[mi, i, j] += m[mi]
        end
        return grid
    end
    return dest_grid
end

"""
    voxelize(poly::FlatPolytope{2,T}, d, order; ibox=nothing, workspace=nothing)
        -> (grid, ibox_lo, ibox_hi)

2D convenience wrapper around [`voxelize!`](@ref). Returns the grid as
shape `(nmom, ni, nj)` along with the index range used.
"""
function voxelize(poly::FlatPolytope{2,T}, d::NTuple{2,T}, order::Int;
                  ibox::Union{Nothing,Tuple{NTuple{2,Int},NTuple{2,Int}}} = nothing,
                  workspace::Union{Nothing,VoxelizeWorkspace{2,T}} = nothing) where {T}
    if ibox === nothing
        lo, hi = get_ibox(poly, d)
    else
        lo, hi = ibox
    end
    ni = hi[1] - lo[1]; nj = hi[2] - lo[2]
    nmom = num_moments(2, order)
    grid = zeros(T, nmom, max(ni, 1), max(nj, 1))
    voxelize!(grid, poly, lo, hi, d, order; workspace = workspace)
    return grid, lo, hi
end

# ===========================================================================
# StaticFlatPolytope — small-cap variant backed by `MMatrix{D,N,…}`.
# When N is small (≤ 64 or so), the MMatrix data is inline within the
# struct and the compiler can unroll the per-vertex loops because N is
# a compile-time constant. `init_box!` drops from ~22 ns → ~10 ns,
# faster than upstream C. Useful for grid voxelization where each cell
# starts as a simple box.
# ===========================================================================

"""
    StaticFlatPolytope{D,T,N,DN}

Small-cap, type-level-sized FlatPolytope. `N` is the vertex capacity at
the type level; `DN = D*N` is required by `MMatrix` (StaticArrays).

Each unique `N` produces a fresh specialization of every method —
**pick one cap per use case**. For most callers, `N = 32` or `N = 64`
is plenty (a unit cube clipped by a handful of planes rarely exceeds
24 vertices).

Construct via:

    poly = R3D.Flat.StaticFlatPolytope{3,Float64,32}()

Then use the same `init_box!` / `clip!` / `moments!` API as
[`FlatPolytope`](@ref).
"""
mutable struct StaticFlatPolytope{D,T,N,DN}
    positions::MMatrix{D,N,T,DN}
    pnbrs::MMatrix{D,N,Int32,DN}
    nverts::Int
    sdists::MVector{N,T}
    clipped::MVector{N,Int32}
    emarks::MMatrix{N,D,Bool,DN}
    Sm::Array{T,3}
    Dm::Array{T,3}
    Cm::Array{T,3}
    moment_order::Int
end

@inline _capacity(::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN} = N

function StaticFlatPolytope{D,T,N}() where {D,T,N}
    DN = D * N
    StaticFlatPolytope{D,T,N,DN}(
        zeros(MMatrix{D,N,T,DN}),
        zeros(MMatrix{D,N,Int32,DN}),
        0,
        zeros(MVector{N,T}),
        zeros(MVector{N,Int32}),
        zeros(MMatrix{N,D,Bool,DN}),
        Array{T,3}(undef, 0, 0, 0),
        Array{T,3}(undef, 0, 0, 0),
        Array{T,3}(undef, 0, 0, 0),
        -1,
    )
end

# ---------------------------------------------------------------------------
# init_box! (D=3)
# ---------------------------------------------------------------------------

function init_box!(poly::StaticFlatPolytope{3,T,N,DN},
                   lo::AbstractVector, hi::AbstractVector) where {T,N,DN}
    @assert N >= 8
    poly.nverts = 8
    @inbounds begin
        poly.positions[1,1] = lo[1]; poly.positions[2,1] = lo[2]; poly.positions[3,1] = lo[3]
        poly.positions[1,2] = hi[1]; poly.positions[2,2] = lo[2]; poly.positions[3,2] = lo[3]
        poly.positions[1,3] = hi[1]; poly.positions[2,3] = hi[2]; poly.positions[3,3] = lo[3]
        poly.positions[1,4] = lo[1]; poly.positions[2,4] = hi[2]; poly.positions[3,4] = lo[3]
        poly.positions[1,5] = lo[1]; poly.positions[2,5] = lo[2]; poly.positions[3,5] = hi[3]
        poly.positions[1,6] = hi[1]; poly.positions[2,6] = lo[2]; poly.positions[3,6] = hi[3]
        poly.positions[1,7] = hi[1]; poly.positions[2,7] = hi[2]; poly.positions[3,7] = hi[3]
        poly.positions[1,8] = lo[1]; poly.positions[2,8] = hi[2]; poly.positions[3,8] = hi[3]

        poly.pnbrs[1,1]=Int32(2); poly.pnbrs[2,1]=Int32(5); poly.pnbrs[3,1]=Int32(4)
        poly.pnbrs[1,2]=Int32(3); poly.pnbrs[2,2]=Int32(6); poly.pnbrs[3,2]=Int32(1)
        poly.pnbrs[1,3]=Int32(4); poly.pnbrs[2,3]=Int32(7); poly.pnbrs[3,3]=Int32(2)
        poly.pnbrs[1,4]=Int32(1); poly.pnbrs[2,4]=Int32(8); poly.pnbrs[3,4]=Int32(3)
        poly.pnbrs[1,5]=Int32(8); poly.pnbrs[2,5]=Int32(1); poly.pnbrs[3,5]=Int32(6)
        poly.pnbrs[1,6]=Int32(5); poly.pnbrs[2,6]=Int32(2); poly.pnbrs[3,6]=Int32(7)
        poly.pnbrs[1,7]=Int32(6); poly.pnbrs[2,7]=Int32(3); poly.pnbrs[3,7]=Int32(8)
        poly.pnbrs[1,8]=Int32(7); poly.pnbrs[2,8]=Int32(4); poly.pnbrs[3,8]=Int32(5)
    end
    return poly
end

# ---------------------------------------------------------------------------
# clip! / clip_plane! (D=3) — same algorithm as FlatPolytope, MMatrix
# fields. The single shared kernel idea didn't pay off — direct
# specialization on this type lets LLVM compile its own version.
# ---------------------------------------------------------------------------

function clip!(poly::StaticFlatPolytope{3,T,N,DN},
               planes::AbstractVector{Plane{3,T}}) where {T,N,DN}
    poly.nverts <= 0 && return false
    @inbounds for plane in planes
        ok = clip_plane!(poly, plane)
        ok || return false
        poly.nverts == 0 && return true
    end
    return true
end

function clip_plane!(poly::StaticFlatPolytope{3,T,N,DN},
                     plane::Plane{3,T}) where {T,N,DN}
    onv = poly.nverts
    sdists = poly.sdists
    clipped = poly.clipped

    smin = T(Inf); smax = T(-Inf)
    @inbounds for v in 1:onv
        x = poly.positions[1, v]; y = poly.positions[2, v]; z = poly.positions[3, v]
        s = plane.d + plane.n[1]*x + plane.n[2]*y + plane.n[3]*z
        sdists[v] = s
        s < smin && (smin = s)
        s > smax && (smax = s)
    end
    smin >= 0 && return true
    if smax <= 0
        poly.nverts = 0
        return true
    end

    @inbounds for v in 1:onv
        clipped[v] = sdists[v] < 0 ? Int32(1) : Int32(0)
    end

    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:3
            vnext = Int(poly.pnbrs[np, vcur])
            vnext == 0 && continue
            clipped[vnext] == 0 && continue

            poly.nverts >= N && return false
            new_idx = (poly.nverts += 1)

            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            poly.positions[1, new_idx] = (wa*poly.positions[1,vcur] + wb*poly.positions[1,vnext]) * invw
            poly.positions[2, new_idx] = (wa*poly.positions[2,vcur] + wb*poly.positions[2,vnext]) * invw
            poly.positions[3, new_idx] = (wa*poly.positions[3,vcur] + wb*poly.positions[3,vnext]) * invw
            poly.pnbrs[1, new_idx] = Int32(vcur)
            poly.pnbrs[2, new_idx] = Int32(0)
            poly.pnbrs[3, new_idx] = Int32(0)
            poly.pnbrs[np, vcur] = Int32(new_idx)
            clipped[new_idx] = Int32(0)
        end
    end

    @inbounds for vstart in (onv+1):poly.nverts
        vcur = vstart
        vnext = Int(poly.pnbrs[1, vcur])
        while true
            np = _find_back3_static(poly.pnbrs, vnext, vcur)
            vcur = vnext
            vnext = Int(poly.pnbrs[next3(np), vcur])
            vcur <= onv || break
        end
        poly.pnbrs[3, vstart] = Int32(vcur)
        poly.pnbrs[2, vcur]   = Int32(vstart)
    end

    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                poly.positions[1, numunclipped] = poly.positions[1, v]
                poly.positions[2, numunclipped] = poly.positions[2, v]
                poly.positions[3, numunclipped] = poly.positions[3, v]
                poly.pnbrs[1, numunclipped] = poly.pnbrs[1, v]
                poly.pnbrs[2, numunclipped] = poly.pnbrs[2, v]
                poly.pnbrs[3, numunclipped] = poly.pnbrs[3, v]
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

# 3-way unrolled find_back specialized on MMatrix (mirrors find_back3
# above; separate dispatch keeps the inliner happy with the static
# pnbrs type).
@inline function _find_back3_static(pnbrs::MMatrix{3,N,Int32,DN}, vnext::Int, vcur::Int) where {N,DN}
    @inbounds begin
        pnbrs[1, vnext] == Int32(vcur) && return 1
        pnbrs[2, vnext] == Int32(vcur) && return 2
        return 3
    end
end

# ---------------------------------------------------------------------------
# moments / moments! (D=3) — same Koehl recursion as FlatPolytope
# ---------------------------------------------------------------------------

function moments(poly::StaticFlatPolytope{3,T,N,DN}, order::Integer) where {T,N,DN}
    out = zeros(T, num_moments(3, order))
    moments!(out, poly, order)
    return out
end

@inline function _ensure_moment_scratch_static!(poly::StaticFlatPolytope{D,T,N,DN}, order::Int) where {D,T,N,DN}
    if order > poly.moment_order
        np1 = order + 1
        poly.Sm = Array{T,3}(undef, np1, np1, 2)
        poly.Dm = Array{T,3}(undef, np1, np1, 2)
        poly.Cm = Array{T,3}(undef, np1, np1, 2)
        poly.moment_order = order
    end
    return nothing
end

function moments!(out::AbstractVector{T},
                  poly::StaticFlatPolytope{3,T,N,DN},
                  order::Integer) where {T,N,DN}
    @assert length(out) >= num_moments(3, order)
    fill!(out, zero(T))
    poly.nverts <= 0 && return out

    nv = poly.nverts
    _ensure_moment_scratch_static!(poly, Int(order))

    emarks = poly.emarks
    @inbounds for k in 1:3, v in 1:nv
        emarks[v, k] = false
    end

    S = poly.Sm; D = poly.Dm; C = poly.Cm
    prevlayer = 1; curlayer = 2

    @inbounds for vstart in 1:nv, pstart in 1:3
        emarks[vstart, pstart] && continue

        pnext = pstart
        vcur = vstart
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])
        v0_x = poly.positions[1, vcur]
        v0_y = poly.positions[2, vcur]
        v0_z = poly.positions[3, vcur]

        np = _find_back3_static(poly.pnbrs, vnext, vcur)
        vcur = vnext
        pnext = next3(np)
        emarks[vcur, pnext] = true
        vnext = Int(poly.pnbrs[pnext, vcur])

        while vnext != vstart
            v2_x = poly.positions[1, vcur]; v2_y = poly.positions[2, vcur]; v2_z = poly.positions[3, vcur]
            v1_x = poly.positions[1, vnext]; v1_y = poly.positions[2, vnext]; v1_z = poly.positions[3, vnext]

            sixv = (-v2_x*v1_y*v0_z + v1_x*v2_y*v0_z + v2_x*v0_y*v1_z
                    - v0_x*v2_y*v1_z - v1_x*v0_y*v2_z + v0_x*v1_y*v2_z)

            S[1,1,prevlayer] = one(T)
            D[1,1,prevlayer] = one(T)
            C[1,1,prevlayer] = one(T)
            out[1] += sixv / 6

            m = 1
            for corder in 1:order
                for i in corder:-1:0, j in (corder - i):-1:0
                    k = corder - i - j
                    m += 1
                    ci = i + 1; cj = j + 1
                    Cv = zero(T); Dv = zero(T); Sv = zero(T)
                    if i > 0
                        Cv += v2_x * C[ci-1, cj, prevlayer]
                        Dv += v1_x * D[ci-1, cj, prevlayer]
                        Sv += v0_x * S[ci-1, cj, prevlayer]
                    end
                    if j > 0
                        Cv += v2_y * C[ci, cj-1, prevlayer]
                        Dv += v1_y * D[ci, cj-1, prevlayer]
                        Sv += v0_y * S[ci, cj-1, prevlayer]
                    end
                    if k > 0
                        Cv += v2_z * C[ci, cj, prevlayer]
                        Dv += v1_z * D[ci, cj, prevlayer]
                        Sv += v0_z * S[ci, cj, prevlayer]
                    end
                    Dv += Cv
                    Sv += Dv
                    C[ci, cj, curlayer] = Cv
                    D[ci, cj, curlayer] = Dv
                    S[ci, cj, curlayer] = Sv
                    out[m] += sixv * Sv
                end
                curlayer  = 3 - curlayer
                prevlayer = 3 - prevlayer
            end

            np = _find_back3_static(poly.pnbrs, vnext, vcur)
            vcur = vnext
            pnext = next3(np)
            emarks[vcur, pnext] = true
            vnext = Int(poly.pnbrs[pnext, vcur])
        end
    end

    @inbounds C[1,1,prevlayer] = one(T)
    m = 1
    for corder in 1:order
        for i in corder:-1:0, j in (corder - i):-1:0
            k = corder - i - j
            m += 1
            ci = i + 1; cj = j + 1
            Cv = zero(T)
            i > 0 && (Cv += C[ci-1, cj, prevlayer])
            j > 0 && (Cv += C[ci, cj-1, prevlayer])
            k > 0 && (Cv += C[ci, cj, prevlayer])
            C[ci, cj, curlayer] = Cv
            out[m] /= Cv * (corder + 1) * (corder + 2) * (corder + 3)
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end
    return out
end

# ===========================================================================
# split!, is_good, shift_moments — additional polytope operations from
# upstream r3d.c / r2d.c. These are not on the perf-critical path; the
# implementations follow the C source line-for-line.
# ===========================================================================

# ---------------------------------------------------------------------------
# split! (D=3) — single polytope split at an arbitrary plane
# ---------------------------------------------------------------------------

"""
    split!(in::FlatPolytope{3,T}, plane, out_pos, out_neg) -> Bool

Split `in` at `plane` into two polytopes:

- `out_pos` receives the half where `plane.n ⋅ x + plane.d ≥ 0`
  (positive signed distance).
- `out_neg` receives the half where the signed distance is negative.

Mirrors `r3d_split` in `src/r3d.c`. `in` is consumed (its state is
undefined after the call). Returns `true` on success, `false` on
capacity overflow.

Differs from [`split_coord!`](@ref) in two ways:
1. The cutting plane is general (not axis-aligned).
2. The "positive" output corresponds to positive signed distance from
   the plane, not positive coordinate along an axis.
"""
function split!(in::FlatPolytope{3,T}, plane::Plane{3,T},
                out_pos::FlatPolytope{3,T}, out_neg::FlatPolytope{3,T}) where {T}
    if in.nverts <= 0
        out_pos.nverts = 0
        out_neg.nverts = 0
        return true
    end

    sdists = in.sdists
    side   = in.clipped     # 0 = positive side (out_pos), 1 = negative side (out_neg)

    # Step 1: signed distances; vertices with sd < 0 → side = 1 (out_neg).
    nneg = 0
    @inbounds for v in 1:in.nverts
        sd = plane.d + plane.n[1]*in.positions[1,v] +
             plane.n[2]*in.positions[2,v] + plane.n[3]*in.positions[3,v]
        sdists[v] = sd
        if sd < 0
            side[v] = Int32(1); nneg += 1
        else
            side[v] = Int32(0)
        end
    end

    if nneg == 0
        _copy_polytope!(out_pos, in)
        out_neg.nverts = 0
        return true
    elseif nneg == in.nverts
        _copy_polytope!(out_neg, in)
        out_pos.nverts = 0
        return true
    end

    # Step 2: insert pairs of new vertices on cut edges. Same pattern as
    # split_coord!; the side-classification is the only thing that
    # differs.
    onv = in.nverts
    @inbounds for vcur in 1:onv
        side[vcur] != 0 && continue   # vcur on positive side
        for np in 1:3
            vnext = Int(in.pnbrs[np, vcur])
            (vnext == 0 || side[vnext] == 0) && continue   # vnext also positive — no cut

            in.nverts >= in.capacity - 1 && return false

            # Note the reversed weighting compared to clip!: vcur is on
            # the kept (positive) side here, vnext is being moved to the
            # negative side. The weighted average is identical because
            # |sd_neg| / (|sd_neg| + sd_pos) is the parametric offset.
            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            nx = (wa*in.positions[1,vcur] + wb*in.positions[1,vnext]) * invw
            ny = (wa*in.positions[2,vcur] + wb*in.positions[2,vnext]) * invw
            nz = (wa*in.positions[3,vcur] + wb*in.positions[3,vnext]) * invw

            new_pos = (in.nverts += 1)
            in.positions[1, new_pos] = nx
            in.positions[2, new_pos] = ny
            in.positions[3, new_pos] = nz
            in.pnbrs[1, new_pos] = Int32(vcur)
            in.pnbrs[2, new_pos] = Int32(0)
            in.pnbrs[3, new_pos] = Int32(0)
            in.pnbrs[np, vcur]   = Int32(new_pos)
            side[new_pos] = Int32(0)

            new_neg = (in.nverts += 1)
            in.positions[1, new_neg] = nx
            in.positions[2, new_neg] = ny
            in.positions[3, new_neg] = nz
            in.pnbrs[1, new_neg] = Int32(vnext)
            in.pnbrs[2, new_neg] = Int32(0)
            in.pnbrs[3, new_neg] = Int32(0)
            side[new_neg] = Int32(1)

            for npnxt in 1:3
                if in.pnbrs[npnxt, vnext] == Int32(vcur)
                    in.pnbrs[npnxt, vnext] = Int32(new_neg)
                    break
                end
            end
        end
    end

    # Step 3: face-walk linker (same as clip_plane!).
    @inbounds for vstart in (onv+1):in.nverts
        vcur = vstart
        vnext = Int(in.pnbrs[1, vcur])
        while true
            np = find_back3(in.pnbrs, vnext, vcur)
            vcur = vnext
            vnext = Int(in.pnbrs[next3(np), vcur])
            vcur <= onv || break
        end
        in.pnbrs[3, vstart] = Int32(vcur)
        in.pnbrs[2, vcur]   = Int32(vstart)
    end

    # Step 4: compact into out_pos (side=0) / out_neg (side=1).
    total_verts = in.nverts
    out_pos.nverts = 0
    out_neg.nverts = 0
    @inbounds for v in 1:total_verts
        if side[v] == 0
            out_pos.nverts += 1; dst = out_pos.nverts
            out_pos.positions[1, dst] = in.positions[1, v]
            out_pos.positions[2, dst] = in.positions[2, v]
            out_pos.positions[3, dst] = in.positions[3, v]
            out_pos.pnbrs[1, dst] = in.pnbrs[1, v]
            out_pos.pnbrs[2, dst] = in.pnbrs[2, v]
            out_pos.pnbrs[3, dst] = in.pnbrs[3, v]
            side[v] = Int32(dst)
        else
            out_neg.nverts += 1; dst = out_neg.nverts
            out_neg.positions[1, dst] = in.positions[1, v]
            out_neg.positions[2, dst] = in.positions[2, v]
            out_neg.positions[3, dst] = in.positions[3, v]
            out_neg.pnbrs[1, dst] = in.pnbrs[1, v]
            out_neg.pnbrs[2, dst] = in.pnbrs[2, v]
            out_neg.pnbrs[3, dst] = in.pnbrs[3, v]
            side[v] = Int32(dst)
        end
    end
    @inbounds for v in 1:out_pos.nverts, np in 1:3
        old = Int(out_pos.pnbrs[np, v])
        out_pos.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    @inbounds for v in 1:out_neg.nverts, np in 1:3
        old = Int(out_neg.pnbrs[np, v])
        out_neg.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    return true
end

# ---------------------------------------------------------------------------
# split! (D=2)
# ---------------------------------------------------------------------------

"""
    split!(in::FlatPolytope{2,T}, plane, out_pos, out_neg) -> Bool

2D analog of [`split!(::FlatPolytope{3,T}, …)`](@ref). Mirrors
`r2d_split` in `src/r2d.c`.
"""
function split!(in::FlatPolytope{2,T}, plane::Plane{2,T},
                out_pos::FlatPolytope{2,T}, out_neg::FlatPolytope{2,T}) where {T}
    if in.nverts <= 0
        out_pos.nverts = 0
        out_neg.nverts = 0
        return true
    end

    sdists = in.sdists
    side   = in.clipped

    nneg = 0
    @inbounds for v in 1:in.nverts
        sd = plane.d + plane.n[1]*in.positions[1,v] + plane.n[2]*in.positions[2,v]
        sdists[v] = sd
        if sd < 0
            side[v] = Int32(1); nneg += 1
        else
            side[v] = Int32(0)
        end
    end

    if nneg == 0
        _copy_polytope_2d!(out_pos, in)
        out_neg.nverts = 0
        return true
    elseif nneg == in.nverts
        _copy_polytope_2d!(out_neg, in)
        out_pos.nverts = 0
        return true
    end

    onv = in.nverts
    @inbounds for vcur in 1:onv
        side[vcur] != 0 && continue
        for np in 1:2
            vnext = Int(in.pnbrs[np, vcur])
            (vnext == 0 || side[vnext] == 0) && continue

            in.nverts >= in.capacity - 1 && return false

            wa = -sdists[vnext]; wb = sdists[vcur]
            invw = 1 / (wa + wb)
            nx = (wa*in.positions[1,vcur] + wb*in.positions[1,vnext]) * invw
            ny = (wa*in.positions[2,vcur] + wb*in.positions[2,vnext]) * invw

            other = 3 - np

            new_pos = (in.nverts += 1)
            in.positions[1, new_pos] = nx
            in.positions[2, new_pos] = ny
            in.pnbrs[other, new_pos] = Int32(vcur)
            in.pnbrs[np,    new_pos] = Int32(0)
            in.pnbrs[np, vcur] = Int32(new_pos)
            side[new_pos] = Int32(0)

            new_neg = (in.nverts += 1)
            in.positions[1, new_neg] = nx
            in.positions[2, new_neg] = ny
            in.pnbrs[other, new_neg] = Int32(0)
            in.pnbrs[np,    new_neg] = Int32(vnext)
            in.pnbrs[other, vnext] = Int32(new_neg)
            side[new_neg] = Int32(1)
        end
    end

    @inbounds for vstart in (onv+1):in.nverts
        in.pnbrs[2, vstart] != 0 && continue
        vcur = Int(in.pnbrs[1, vstart])
        while vcur <= onv
            vcur = Int(in.pnbrs[1, vcur])
        end
        in.pnbrs[2, vstart] = Int32(vcur)
        in.pnbrs[1, vcur]   = Int32(vstart)
    end

    total_verts = in.nverts
    out_pos.nverts = 0
    out_neg.nverts = 0
    @inbounds for v in 1:total_verts
        if side[v] == 0
            out_pos.nverts += 1; dst = out_pos.nverts
            out_pos.positions[1, dst] = in.positions[1, v]
            out_pos.positions[2, dst] = in.positions[2, v]
            out_pos.pnbrs[1, dst] = in.pnbrs[1, v]
            out_pos.pnbrs[2, dst] = in.pnbrs[2, v]
            side[v] = Int32(dst)
        else
            out_neg.nverts += 1; dst = out_neg.nverts
            out_neg.positions[1, dst] = in.positions[1, v]
            out_neg.positions[2, dst] = in.positions[2, v]
            out_neg.pnbrs[1, dst] = in.pnbrs[1, v]
            out_neg.pnbrs[2, dst] = in.pnbrs[2, v]
            side[v] = Int32(dst)
        end
    end
    @inbounds for v in 1:out_pos.nverts, np in 1:2
        old = Int(out_pos.pnbrs[np, v])
        out_pos.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    @inbounds for v in 1:out_neg.nverts, np in 1:2
        old = Int(out_neg.pnbrs[np, v])
        out_neg.pnbrs[np, v] = old == 0 ? Int32(0) : side[old]
    end
    return true
end

# ---------------------------------------------------------------------------
# is_good — connectivity sanity check
# ---------------------------------------------------------------------------

"""
    is_good(poly::FlatPolytope{D,T}) -> Bool

Check that every vertex's neighbour list is internally consistent:

- No vertex is its own neighbour, and no vertex's slots are duplicates.
- Every neighbour index is in `1:nverts`.
- Every vertex is referenced by exactly `D` other vertices.

Mirrors `r3d_is_good` / `r2d_is_good` (`src/r3d.c`, `src/r2d.c`). Useful
as a debugging aid; not on the hot path.
"""
function is_good(poly::FlatPolytope{D,T}) where {D,T}
    nv = poly.nverts
    nv <= 0 && return true   # empty polytope is trivially good
    refcount = poly.clipped  # reuse capacity-sized scratch as a counter
    @inbounds for v in 1:nv
        refcount[v] = Int32(0)
    end
    @inbounds for v in 1:nv
        # No self-loops; no duplicate slots.
        if D == 3
            a = poly.pnbrs[1, v]; b = poly.pnbrs[2, v]; c = poly.pnbrs[3, v]
            (a == b || b == c || a == c) && return false
            (a == Int32(v) || b == Int32(v) || c == Int32(v)) && return false
            (a < 1 || a > nv || b < 1 || b > nv || c < 1 || c > nv) && return false
            refcount[a] += Int32(1)
            refcount[b] += Int32(1)
            refcount[c] += Int32(1)
        else  # D == 2
            a = poly.pnbrs[1, v]; b = poly.pnbrs[2, v]
            a == b && return false
            (a == Int32(v) || b == Int32(v)) && return false
            (a < 1 || a > nv || b < 1 || b > nv) && return false
            refcount[a] += Int32(1)
            refcount[b] += Int32(1)
        end
    end
    @inbounds for v in 1:nv
        refcount[v] == Int32(D) || return false
    end
    return true
end

function is_good(poly::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN}
    nv = poly.nverts
    nv <= 0 && return true
    refcount = poly.clipped
    @inbounds for v in 1:nv
        refcount[v] = Int32(0)
    end
    @inbounds for v in 1:nv
        a = poly.pnbrs[1, v]; b = poly.pnbrs[2, v]; c = poly.pnbrs[3, v]
        (a == b || b == c || a == c) && return false
        (a == Int32(v) || b == Int32(v) || c == Int32(v)) && return false
        (a < 1 || a > nv || b < 1 || b > nv || c < 1 || c > nv) && return false
        refcount[a] += Int32(1)
        refcount[b] += Int32(1)
        refcount[c] += Int32(1)
    end
    @inbounds for v in 1:nv
        refcount[v] == Int32(3) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# shift_moments — translate moments to a new origin via Pascal's triangle
# ---------------------------------------------------------------------------

"""
    shift_moments!(moments, polyorder, shift, ::Val{D})

Translate a polytope's `moments` (about the spatial origin) to the
moments of a polytope translated by `shift`. Implements

    ∫_{Ω+shift} x^i y^j (z^k) dV
        = ∫_Ω (x+shift.x)^i (y+shift.y)^j (z+shift.z)^k dV

via the multinomial expansion. The 0th moment (volume / area) is
unchanged. Mirrors `r3d_shift_moments` (D=3) / `r2d_shift_moments`
(D=2). `Val{D}` selects the dimensionality.

In the C source the same operation is used to "shift back" central
moments computed against a translated polytope (numerically more
accurate for high-order moments) — pass the polytope's centroid as
`shift` and `moments` as the central moments to recover the raw
moments. The `moments` vector is updated in place.
"""
function shift_moments!(moments::AbstractVector{T}, polyorder::Integer,
                        vc::NTuple{D,T}, ::Val{D}) where {T,D}
    @assert D == 2 || D == 3 "shift_moments! only supports D=2 or D=3"
    nm = num_moments(D, polyorder)
    @assert length(moments) >= nm
    polyorder <= 0 && return moments

    # Pascal's triangle B[i, corder] = binom(corder, i), 0 ≤ i ≤ corder
    Bsz = polyorder + 1
    B = zeros(T, Bsz, Bsz)
    B[1, 1] = one(T)
    @inbounds for corder in 1:polyorder
        for i in corder:-1:0
            j = corder - i
            B[i+1, corder+1] = one(T)
            if i > 0 && j > 0
                B[i+1, corder+1] = B[i+1, corder] + B[i, corder]
            end
        end
    end

    # Working copy to avoid reading back updated values.
    moments2 = zeros(T, nm)

    if D == 3
        @inbounds for corder in 1:polyorder, i in corder:-1:0, j in (corder - i):-1:0
            k = corder - i - j
            m = _moment_index_3d(i, j, k)
            acc = zero(T)
            for mcorder in 0:corder, mi in mcorder:-1:0, mj in (mcorder - mi):-1:0
                mk = mcorder - mi - mj
                if mi <= i && mj <= j && mk <= k
                    mm = _moment_index_3d(mi, mj, mk)
                    acc += B[mi+1, i+1] * B[mj+1, j+1] * B[mk+1, k+1] *
                           vc[1]^(i-mi) * vc[2]^(j-mj) * vc[3]^(k-mk) *
                           moments[mm]
                end
            end
            moments2[m] = acc
        end
    else  # D == 2
        @inbounds for corder in 1:polyorder, i in corder:-1:0
            j = corder - i
            m = _moment_index_2d(i, j)
            acc = zero(T)
            for mcorder in 0:corder, mi in mcorder:-1:0
                mj = mcorder - mi
                if mi <= i && mj <= j
                    mm = _moment_index_2d(mi, mj)
                    acc += B[mi+1, i+1] * B[mj+1, j+1] *
                           vc[1]^(i-mi) * vc[2]^(j-mj) *
                           moments[mm]
                end
            end
            moments2[m] = acc
        end
    end

    @inbounds for m in 2:nm
        moments[m] = moments2[m]
    end
    return moments
end

# Index of moment x^i y^j z^k in the 3D moment vector. The order is
# row-major over (i, j, k) with corder = i+j+k iterated outer-to-inner
# and i descending, mirroring the existing reduce loop pattern in
# `moments!` (`flat.jl`).
function _moment_index_3d(i::Int, j::Int, k::Int)
    corder = i + j + k
    corder == 0 && return 1
    # 1 + sum_{c=1..corder-1} (c+1)(c+2)/2 + (offset within corder)
    base = 1
    for c in 1:corder-1
        base += (c+1)*(c+2)÷2
    end
    # within corder: outer loop i: corder..0; inner loop j: (corder-i)..0
    off = 0
    for ii in corder:-1:i+1
        off += (corder - ii) + 1
    end
    off += (corder - i) - j + 1
    return base + off
end

function _moment_index_2d(i::Int, j::Int)
    corder = i + j
    corder == 0 && return 1
    base = 1
    for c in 1:corder-1
        base += c + 1
    end
    off = corder - i + 1
    return base + off
end

# ===========================================================================
# init_tet! / init_simplex! / init_poly! — additional constructors
# ===========================================================================

"""
    init_tet!(poly::FlatPolytope{3,T}, v1, v2, v3, v4)

Initialize as a tetrahedron with the four given vertex positions.
Mirrors `r3d_init_tet` in `src/r3d.c` line-for-line (same pnbrs table,
1-based here vs 0-based in C).
"""
function init_tet!(poly::FlatPolytope{3,T},
                   v1::AbstractVector, v2::AbstractVector,
                   v3::AbstractVector, v4::AbstractVector) where {T}
    @assert poly.capacity >= 4
    poly.nverts = 4
    @inbounds begin
        poly.positions[1,1] = v1[1]; poly.positions[2,1] = v1[2]; poly.positions[3,1] = v1[3]
        poly.positions[1,2] = v2[1]; poly.positions[2,2] = v2[2]; poly.positions[3,2] = v2[3]
        poly.positions[1,3] = v3[1]; poly.positions[2,3] = v3[2]; poly.positions[3,3] = v3[3]
        poly.positions[1,4] = v4[1]; poly.positions[2,4] = v4[2]; poly.positions[3,4] = v4[3]

        poly.pnbrs[1,1]=Int32(2); poly.pnbrs[2,1]=Int32(4); poly.pnbrs[3,1]=Int32(3)
        poly.pnbrs[1,2]=Int32(3); poly.pnbrs[2,2]=Int32(4); poly.pnbrs[3,2]=Int32(1)
        poly.pnbrs[1,3]=Int32(1); poly.pnbrs[2,3]=Int32(4); poly.pnbrs[3,3]=Int32(2)
        poly.pnbrs[1,4]=Int32(2); poly.pnbrs[2,4]=Int32(3); poly.pnbrs[3,4]=Int32(1)
    end
    return poly
end

"Convenience: build a fresh 3D tetrahedron."
function tet(v1::NTuple{3,Real}, v2::NTuple{3,Real},
             v3::NTuple{3,Real}, v4::NTuple{3,Real}; capacity::Int = 512)
    p = FlatPolytope{3,Float64}(capacity)
    init_tet!(p, [v1...], [v2...], [v3...], [v4...])
    return p
end

"""
    init_simplex!(poly::FlatPolytope{3,T}, v1, v2, v3, v4) -> poly

D = 3 alias for [`init_tet!`](@ref). Provided for API uniformity with
the D = 2 version of `init_simplex!` (triangle from 3 vertices).
"""
@inline init_simplex!(poly::FlatPolytope{3,T},
                      v1::AbstractVector, v2::AbstractVector,
                      v3::AbstractVector, v4::AbstractVector) where {T} =
    init_tet!(poly, v1, v2, v3, v4)

"""
    init_simplex!(poly::FlatPolytope{D,T}, vertices) -> poly

Initialize from a length-`(D+1)` collection of vertex positions
(`AbstractVector`s, `NTuple`s, or `SVector`s). Convenience wrapper
that splats into the per-vertex `init_simplex!` (D=2: 3 verts;
D=3: 4 verts).
"""
function init_simplex!(poly::FlatPolytope{2,T}, vertices) where {T}
    @assert length(vertices) == 3 "D=2 simplex needs 3 vertices, got $(length(vertices))"
    init_simplex!(poly, vertices[1], vertices[2], vertices[3])
end

function init_simplex!(poly::FlatPolytope{3,T}, vertices) where {T}
    @assert length(vertices) == 4 "D=3 simplex needs 4 vertices, got $(length(vertices))"
    init_simplex!(poly, vertices[1], vertices[2], vertices[3], vertices[4])
end

# Accept tuple-element vertices as well (the 2D init_simplex! signature
# above takes AbstractVector; tuples don't match it directly). These
# methods write the positions directly from the tuples — no Vector
# splat — so they're 0-alloc for hot loops.
function init_simplex!(poly::FlatPolytope{2,T},
                       v1::NTuple{2,<:Real}, v2::NTuple{2,<:Real},
                       v3::NTuple{2,<:Real}) where {T}
    @assert poly.capacity >= 3
    poly.nverts = 3
    @inbounds begin
        poly.positions[1,1] = T(v1[1]); poly.positions[2,1] = T(v1[2])
        poly.positions[1,2] = T(v2[1]); poly.positions[2,2] = T(v2[2])
        poly.positions[1,3] = T(v3[1]); poly.positions[2,3] = T(v3[2])
        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(3)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(1); poly.pnbrs[2,3] = Int32(2)
    end
    return poly
end

function init_simplex!(poly::FlatPolytope{3,T},
                       v1::NTuple{3,<:Real}, v2::NTuple{3,<:Real},
                       v3::NTuple{3,<:Real}, v4::NTuple{3,<:Real}) where {T}
    @assert poly.capacity >= 4
    poly.nverts = 4
    @inbounds begin
        poly.positions[1,1] = T(v1[1]); poly.positions[2,1] = T(v1[2]); poly.positions[3,1] = T(v1[3])
        poly.positions[1,2] = T(v2[1]); poly.positions[2,2] = T(v2[2]); poly.positions[3,2] = T(v2[3])
        poly.positions[1,3] = T(v3[1]); poly.positions[2,3] = T(v3[2]); poly.positions[3,3] = T(v3[3])
        poly.positions[1,4] = T(v4[1]); poly.positions[2,4] = T(v4[2]); poly.positions[3,4] = T(v4[3])
        poly.pnbrs[1,1]=Int32(2); poly.pnbrs[2,1]=Int32(4); poly.pnbrs[3,1]=Int32(3)
        poly.pnbrs[1,2]=Int32(3); poly.pnbrs[2,2]=Int32(4); poly.pnbrs[3,2]=Int32(1)
        poly.pnbrs[1,3]=Int32(1); poly.pnbrs[2,3]=Int32(4); poly.pnbrs[3,3]=Int32(2)
        poly.pnbrs[1,4]=Int32(2); poly.pnbrs[2,4]=Int32(3); poly.pnbrs[3,4]=Int32(1)
    end
    return poly
end

"""
    init_simplex!(poly::FlatPolytope{2,T}, v1, v2, v3)

Initialize as a triangle (2D simplex) with the three given vertices in
CCW order. Equivalent to `init_poly!` on a 3-vertex closed polygon.
"""
function init_simplex!(poly::FlatPolytope{2,T},
                       v1::AbstractVector, v2::AbstractVector,
                       v3::AbstractVector) where {T}
    @assert poly.capacity >= 3
    poly.nverts = 3
    @inbounds begin
        poly.positions[1,1] = v1[1]; poly.positions[2,1] = v1[2]
        poly.positions[1,2] = v2[1]; poly.positions[2,2] = v2[2]
        poly.positions[1,3] = v3[1]; poly.positions[2,3] = v3[2]
        poly.pnbrs[1,1] = Int32(2); poly.pnbrs[2,1] = Int32(3)
        poly.pnbrs[1,2] = Int32(3); poly.pnbrs[2,2] = Int32(1)
        poly.pnbrs[1,3] = Int32(1); poly.pnbrs[2,3] = Int32(2)
    end
    return poly
end

"Convenience: build a fresh 2D triangle."
function simplex(v1::NTuple{2,Real}, v2::NTuple{2,Real}, v3::NTuple{2,Real};
                 capacity::Int = 256)
    p = FlatPolytope{2,Float64}(capacity)
    init_simplex!(p, [v1...], [v2...], [v3...])
    return p
end

# D ≥ 4 simplex constructors — explicit per-D dispatches for Aqua
# unbound-args hygiene. Takes `D + 1` vertices as `NTuple{D,Real}`.
# The D = 3 analogue is `tet(v1, v2, v3, v4)`.
for _D in 4:6
    _M = _D + 1
    @eval function simplex(verts::Vararg{NTuple{$_D,Real}, $_M};
                            capacity::Int = 512)
        p = FlatPolytope{$_D,Float64}(capacity)
        init_simplex!(p, [collect(Float64, v) for v in verts])
        return p
    end
end

"""
    init_poly!(poly::FlatPolytope{2,T}, vertices) -> poly

Initialize a 2D polytope as a closed polygon from a CCW-ordered
iterable of `D=2` vertex positions (`AbstractVector`s, `NTuple`s, or
`SVector`s). Mirrors `r2d_init_poly` in `src/r2d.c`.
"""
function init_poly!(poly::FlatPolytope{2,T}, vertices) where {T}
    n = length(vertices)
    @assert n >= 3
    @assert poly.capacity >= n
    poly.nverts = n
    @inbounds for v in 1:n
        p = vertices[v]
        poly.positions[1, v] = T(p[1])
        poly.positions[2, v] = T(p[2])
        # CCW: pnbrs[1] = next vertex; pnbrs[2] = previous vertex
        poly.pnbrs[1, v] = Int32(v == n ? 1 : v + 1)
        poly.pnbrs[2, v] = Int32(v == 1 ? n : v - 1)
    end
    return poly
end

"""
    init_poly!(poly::FlatPolytope{3,T}, vertices, faces) -> poly

Initialize a 3D polytope from `vertices` (length-`numverts` collection
of position vectors) and `faces` (a vector of vector-of-1-based
vertex-indices, one per face, each ordered to form a closed loop).
Mirrors `r3d_init_poly` in `src/r3d.c`.

Supports both the **simple case** (every vertex has exactly 3 incident
faces — e.g. tetrahedron) and the **general case** (vertices with > 3
incident faces — e.g. cube, octahedron) via the same vertex-duplication
algorithm as the upstream code.
"""
function init_poly!(poly::FlatPolytope{3,T}, vertices, faces) where {T}
    numverts = length(vertices)
    numfaces = length(faces)
    @assert numverts >= 4 && numfaces >= 4

    # Edges per vertex
    eperv = zeros(Int32, numverts)
    for f in 1:numfaces, v in faces[f]
        eperv[v] += Int32(1)
    end
    minvperf = Int32(numverts); maxvperf = Int32(0)
    for v in 1:numverts
        e = eperv[v]
        e < minvperf && (minvperf = e)
        e > maxvperf && (maxvperf = e)
    end

    poly.nverts = 0
    minvperf < 3 && return poly  # degenerate input

    if maxvperf == 3
        # Simple case: no vertex duplication needed.
        @assert poly.capacity >= numverts
        poly.nverts = numverts
        sentinel = Int32(0)   # our 1-based "unset"
        @inbounds for v in 1:numverts
            p = vertices[v]
            poly.positions[1, v] = T(p[1])
            poly.positions[2, v] = T(p[2])
            poly.positions[3, v] = T(p[3])
            poly.pnbrs[1, v] = sentinel
            poly.pnbrs[2, v] = sentinel
            poly.pnbrs[3, v] = sentinel
        end
        @inbounds for f in 1:numfaces
            face = faces[f]
            nfv = length(face)
            for vi in 1:nfv
                vprev = face[vi]
                vcur  = face[vi == nfv ? 1 : vi + 1]
                vnext = face[vi >= nfv - 1 ? vi - nfv + 2 : vi + 2]
                set = false
                for np in 1:3
                    pn = poly.pnbrs[np, vcur]
                    if pn == Int32(vprev)
                        # CCW traversal: vnext goes opposite to vprev's slot,
                        # i.e. (np + 1) (mod 3) in 1-based: (np % 3) + 1
                        opposite_slot = (np + 1) % 3 + 1   # (np-1+2)%3 + 1
                        poly.pnbrs[opposite_slot, vcur] = Int32(vnext)
                        set = true; break
                    elseif pn == Int32(vnext)
                        # vprev goes in the slot one CCW from vnext's slot
                        opposite_slot = np % 3 + 1
                        poly.pnbrs[opposite_slot, vcur] = Int32(vprev)
                        set = true; break
                    end
                end
                if !set
                    # First time touching vcur: arbitrarily place vnext, vprev
                    poly.pnbrs[1, vcur] = Int32(vnext)
                    poly.pnbrs[2, vcur] = Int32(vprev)
                end
            end
        end
        return poly
    end

    # General case: each vertex with k > 3 incident faces is duplicated
    # k times; we link the duplicates and then collapse pairs.
    # Mirrors r3d.c lines 793-904 (port preserves indices in 1-based form;
    # the upstream `R3D_MAX_VERTS` sentinel is replaced by 0 here).
    total = sum(eperv)
    cap_needed = total
    @assert poly.capacity >= cap_needed "poly.capacity = $(poly.capacity), need ≥ $cap_needed"

    # Working buffers
    pos_tmp  = zeros(T, 3, total)
    pnb_tmp  = zeros(Int32, 3, total)
    util     = zeros(Int32, total)
    vstart   = zeros(Int, numverts)

    # Read in vertex locations with duplicates; each vertex v lives at
    # indices vstart[v] .. vstart[v]+eperv[v]-1 (1-based here).
    nv = 0
    @inbounds for v in 1:numverts
        vstart[v] = nv + 1
        p = vertices[v]
        for _ in 1:eperv[v]
            nv += 1
            pos_tmp[1, nv] = T(p[1])
            pos_tmp[2, nv] = T(p[2])
            pos_tmp[3, nv] = T(p[3])
            pnb_tmp[1, nv] = Int32(0)
            pnb_tmp[2, nv] = Int32(0)
            pnb_tmp[3, nv] = Int32(0)
        end
    end

    # Fill in connectivity for all duplicates (per-face traversal)
    @inbounds for f in 1:numfaces
        face = faces[f]
        nfv = length(face)
        for vi in 1:nfv
            vprev = face[vi]
            vcur_orig = face[vi == nfv ? 1 : vi + 1]
            vnext = face[vi >= nfv - 1 ? vi - nfv + 2 : vi + 2]
            vcur = vstart[vcur_orig] + util[vcur_orig]
            util[vcur_orig] += Int32(1)
            pnb_tmp[2, vcur] = Int32(vnext)
            pnb_tmp[3, vcur] = Int32(vprev)
        end
    end

    # Link degenerate duplicates around each original vertex
    fill!(util, Int32(0))
    @inbounds for v in 1:numverts
        for v0 in vstart[v]:vstart[v]+eperv[v]-1
            for v1 in vstart[v]:vstart[v]+eperv[v]-1
                if pnb_tmp[3, v0] == pnb_tmp[2, v1] && util[v0] == 0
                    pnb_tmp[3, v0] = Int32(v1)
                    pnb_tmp[1, v1] = Int32(v0)
                    util[v0] = Int32(1)
                end
            end
        end
    end

    # Complete vertex pairs across distinct original vertices
    fill!(util, Int32(0))
    @inbounds for v0 in 1:numverts, v1 in (v0+1):numverts
        for v00 in vstart[v0]:vstart[v0]+eperv[v0]-1
            for v11 in vstart[v1]:vstart[v1]+eperv[v1]-1
                if pnb_tmp[2, v00] == Int32(v1) && pnb_tmp[2, v11] == Int32(v0) &&
                   util[v00] == 0 && util[v11] == 0
                    pnb_tmp[2, v00] = Int32(v11)
                    pnb_tmp[2, v11] = Int32(v00)
                    util[v00] = Int32(1)
                    util[v11] = Int32(1)
                end
            end
        end
    end

    # Remove unnecessary dummy vertices (collapses each duplicate pair).
    fill!(util, Int32(0))
    @inbounds for v in 1:numverts
        v0 = vstart[v]
        v1 = pnb_tmp[1, v0]
        v00 = pnb_tmp[3, v0]
        v11 = pnb_tmp[1, v1]
        pnb_tmp[1, v00] = pnb_tmp[2, v0]
        pnb_tmp[3, v11] = pnb_tmp[2, v1]
        # Patch back-links from v0 / v1's "outside" neighbour
        target0 = pnb_tmp[2, v0]
        for np in 1:3
            if pnb_tmp[np, target0] == Int32(v0)
                pnb_tmp[np, target0] = v00
                break
            end
        end
        target1 = pnb_tmp[2, v1]
        for np in 1:3
            if pnb_tmp[np, target1] == Int32(v1)
                pnb_tmp[np, target1] = v11
                break
            end
        end
        util[v0] = Int32(1)
        util[v1] = Int32(1)
    end

    # Compact into the real polytope buffer
    numunclipped = 0
    @inbounds for v in 1:nv
        if util[v] == 0
            numunclipped += 1
            poly.positions[1, numunclipped] = pos_tmp[1, v]
            poly.positions[2, numunclipped] = pos_tmp[2, v]
            poly.positions[3, numunclipped] = pos_tmp[3, v]
            poly.pnbrs[1, numunclipped] = pnb_tmp[1, v]
            poly.pnbrs[2, numunclipped] = pnb_tmp[2, v]
            poly.pnbrs[3, numunclipped] = pnb_tmp[3, v]
            util[v] = Int32(numunclipped)
        end
    end
    poly.nverts = numunclipped
    @inbounds for v in 1:poly.nverts, np in 1:3
        old = Int(poly.pnbrs[np, v])
        poly.pnbrs[np, v] = old == 0 ? Int32(0) : util[old]
    end
    return poly
end

# ===========================================================================
# Affine transformations — translate!, scale!, rotate!, shear!, affine!
# Mirror upstream `r3d_translate`/`r3d_scale`/`r3d_rotate`/`r3d_shear`/
# `r3d_affine` (and the 2D variants). All operate in-place on
# `poly.positions`; no scratch needed.
# ===========================================================================

"""
    translate!(poly, shift)

Translate `poly` by `shift` (an `NTuple{D,T}` or `Vec{D,T}`).
"""
function translate!(poly::FlatPolytope{D,T},
                    shift::Union{NTuple{D,<:Real},AbstractVector}) where {D,T}
    @assert length(shift) == D
    @inbounds for v in 1:poly.nverts, k in 1:D
        poly.positions[k, v] += T(shift[k])
    end
    return poly
end

"""
    scale!(poly, s)

Uniform scale by scalar `s`. Mirrors `r3d_scale` / `r2d_scale`.
"""
function scale!(poly::FlatPolytope{D,T}, s::Real) where {D,T}
    sT = T(s)
    @inbounds for v in 1:poly.nverts, k in 1:D
        poly.positions[k, v] *= sT
    end
    return poly
end

"""
    shear!(poly, shear, axb, axs)

Shear axis `axb` by `shear * x[axs]`. Mirrors `r3d_shear` / `r2d_shear`.
`axb` and `axs` are 1-based axis indices (1..D).
"""
function shear!(poly::FlatPolytope{D,T}, shear::Real, axb::Int, axs::Int) where {D,T}
    @assert 1 <= axb <= D && 1 <= axs <= D
    sT = T(shear)
    @inbounds for v in 1:poly.nverts
        poly.positions[axb, v] += sT * poly.positions[axs, v]
    end
    return poly
end

"""
    rotate!(poly::FlatPolytope{2,T}, theta)

Rotate a 2D polytope by angle `theta` (radians, CCW). The upstream
`r2d_rotate` has a long-standing bug where it writes `pos.x` twice and
never updates `pos.y`; this implementation does the correct rotation.
"""
function rotate!(poly::FlatPolytope{2,T}, theta::Real) where {T}
    c = T(cos(theta)); s = T(sin(theta))
    @inbounds for v in 1:poly.nverts
        x = poly.positions[1, v]; y = poly.positions[2, v]
        poly.positions[1, v] = c * x - s * y
        poly.positions[2, v] = s * x + c * y
    end
    return poly
end

"""
    rotate!(poly::FlatPolytope{3,T}, theta, axis)

Rotate a 3D polytope by angle `theta` (radians, right-hand rule) around
the given coordinate `axis` ∈ {1, 2, 3}. Mirrors `r3d_rotate(poly, theta, axis-1)`.
"""
function rotate!(poly::FlatPolytope{3,T}, theta::Real, axis::Int) where {T}
    @assert 1 <= axis <= 3
    c = T(cos(theta)); s = T(sin(theta))
    # axes orthogonal to `axis`: (axis%3)+1 and ((axis+1)%3)+1, 1-based
    a = (axis % 3) + 1
    b = ((axis + 1) % 3) + 1
    @inbounds for v in 1:poly.nverts
        u = poly.positions[a, v]; w = poly.positions[b, v]
        poly.positions[a, v] = c * u - s * w
        poly.positions[b, v] = s * u + c * w
    end
    return poly
end

"""
    affine!(poly::FlatPolytope{2,T}, mat::AbstractMatrix)

Apply the 3×3 homogeneous affine matrix `mat` to a 2D polytope.
Last row is the homogeneous projection; if `w != 1`, divides through.
"""
function affine!(poly::FlatPolytope{2,T}, mat::AbstractMatrix) where {T}
    @assert size(mat) == (3, 3)
    @inbounds for v in 1:poly.nverts
        x = poly.positions[1, v]; y = poly.positions[2, v]
        nx = mat[1,1]*x + mat[1,2]*y + mat[1,3]
        ny = mat[2,1]*x + mat[2,2]*y + mat[2,3]
        w  = mat[3,1]*x + mat[3,2]*y + mat[3,3]
        poly.positions[1, v] = T(nx / w)
        poly.positions[2, v] = T(ny / w)
    end
    return poly
end

"""
    affine!(poly::FlatPolytope{3,T}, mat::AbstractMatrix)

Apply the 4×4 homogeneous affine matrix `mat` to a 3D polytope.
Mirrors `r3d_affine`. Includes the homogeneous-w divide for perspective
projections.
"""
function affine!(poly::FlatPolytope{3,T}, mat::AbstractMatrix) where {T}
    @assert size(mat) == (4, 4)
    @inbounds for v in 1:poly.nverts
        x = poly.positions[1, v]; y = poly.positions[2, v]; z = poly.positions[3, v]
        nx = mat[1,1]*x + mat[1,2]*y + mat[1,3]*z + mat[1,4]
        ny = mat[2,1]*x + mat[2,2]*y + mat[2,3]*z + mat[2,4]
        nz = mat[3,1]*x + mat[3,2]*y + mat[3,3]*z + mat[3,4]
        w  = mat[4,1]*x + mat[4,2]*y + mat[4,3]*z + mat[4,4]
        poly.positions[1, v] = T(nx / w)
        poly.positions[2, v] = T(ny / w)
        poly.positions[3, v] = T(nz / w)
    end
    return poly
end

# ===========================================================================
# Generic D ≥ 4 constructors. The clip + moments kernels themselves are
# NOT yet generalized to D ≥ 4 (see docs/phase3_status.md and the
# `clip!`/`moments!` D ≥ 4 stubs further down). These constructors
# exist so consumers can build the initial polytope in any D and
# inspect it; they're zero-alloc on a reused buffer like the D = 2 / 3
# versions.
# ===========================================================================

"""
    init_box!(poly::FlatPolytope{D,T}, lo::AbstractVector, hi::AbstractVector) -> poly

D-dimensional axis-aligned box constructor for `D ≥ 4`. Mirrors
`rNd_init_box` (`src/rNd.c`): `2^D` vertices indexed by a bitmask
(bit `i` set ⇒ vertex's `i`-th coordinate is `hi[i]`); the `i`-th
neighbour of vertex `v` is `v ⊻ (1 << (i-1))`.

The 2-face (`finds`) connectivity table needed by `clip!` in `D ≥ 4`
is **not** populated by this constructor yet; see Phase 3 status.
"""
function init_box!(poly::FlatPolytope{D,T},
                   lo::AbstractVector, hi::AbstractVector) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 init_box! methods for those dimensions"
    nv = 1 << D
    @assert poly.capacity >= nv "init_box! D=$D needs capacity ≥ $nv, got $(poly.capacity)"
    @assert length(lo) == D && length(hi) == D
    poly.nverts = nv
    @inbounds for v in 0:(nv - 1)
        # Bit-hack: bit i of v selects between lo[i+1] and hi[i+1] for axis i+1.
        for i in 1:D
            stride = 1 << (i - 1)
            poly.positions[i, v + 1] = T((v & stride) != 0 ? hi[i] : lo[i])
            # Neighbour along axis i: flip bit (i-1).
            poly.pnbrs[i, v + 1] = Int32((v ⊻ stride) + 1)
        end
    end

    # 2-face connectivity (rNd_init_box, src/rNd.c:568-574). The box has
    # one 2-face per axis pair (np, np1) with np < np1: any pair of axes
    # determines a 2D face that is shared by all 2^D vertices via the
    # symmetry of the hypercube. Total nfaces = D*(D-1)/2.
    f = 0
    @inbounds for np in 1:D, np1 in (np + 1):D
        f += 1
        for v in 1:nv
            poly.finds[np,  np1, v] = Int32(f)
            poly.finds[np1, np,  v] = Int32(f)
        end
    end
    poly.nfaces = f

    # Facet ((D−1)-face) IDs. A D-box has 2D facets — one per axis-side.
    # Convention: facet 2k-1 is "x[k] = lo[k]"; facet 2k is "x[k] = hi[k]".
    # For vertex v (bit pattern), the facet OPPOSITE edge slot k is the
    # facet on v's OWN side along axis k (since v is on that facet, and
    # the edge in slot k crosses to the OTHER side).
    @inbounds for v in 0:(nv - 1), k in 1:D
        stride = 1 << (k - 1)
        on_hi = (v & stride) != 0
        poly.facets[k, v + 1] = Int32(on_hi ? 2k : 2k - 1)
    end
    poly.nfacets = 2D

    # Per-facet outward normals + signed distances (n · x = d).
    # Facet 2k-1 ("x[k] = lo[k]") has outward normal -e_k and signed
    # distance d = -lo[k]; facet 2k ("x[k] = hi[k]") has outward normal
    # +e_k and signed distance d = hi[k]. Used by Lasserre's
    # higher-order moments recursion at D ≥ 4.
    _grow_facet_metadata!(poly, 2D)
    @inbounds for j in 1:(2D), i in 1:D
        poly.facet_normals[i, j] = zero(T)
    end
    @inbounds for k in 1:D
        poly.facet_normals[k, 2k - 1]  = T(-1)
        poly.facet_distances[2k - 1]   = -T(lo[k])
        poly.facet_normals[k, 2k]      = T( 1)
        poly.facet_distances[2k]       =  T(hi[k])
    end
    return poly
end

"""
    init_simplex!(poly::FlatPolytope{D,T}, vertices) -> poly

D-dimensional simplex constructor for `D ≥ 4`. Takes a length-`(D+1)`
collection of vertex positions. Mirrors `rNd_init_simplex`
(`src/rNd.c`): vertex `v`'s neighbours are `(v + i + 1) mod (D + 1)`
for `i ∈ 1:D` (i.e. every other vertex, in cyclic order).

The 2-face (`finds`) connectivity needed by `clip!` is **not** yet
populated; see Phase 3 status.
"""
function init_simplex!(poly::FlatPolytope{D,T}, vertices) where {D,T}
    @assert D >= 4 "use the D=2 / D=3 init_simplex! methods for those dimensions"
    nv = D + 1
    @assert length(vertices) == nv "D=$D simplex needs $(D+1) vertices, got $(length(vertices))"
    @assert poly.capacity >= nv
    poly.nverts = nv
    @inbounds for v in 1:nv
        p = vertices[v]
        for i in 1:D
            poly.positions[i, v] = T(p[i])
            poly.pnbrs[i, v] = Int32(((v - 1 + i) % nv) + 1)
        end
    end

    # 2-face connectivity (rNd_init_simplex, src/rNd.c:527-545).
    # Each unordered vertex triple (v0, v1, v2) defines one 2-face; the
    # face ID gets written into each of those vertices' finds matrix at
    # the (slot-of-other, slot-of-other) position. Total faces:
    # C(D+1, 3) = binomial(nv, 3).
    f = 0
    @inbounds for v0 in 1:nv, v1 in (v0+1):nv, v2 in (v1+1):nv
        f += 1
        # v0's slots pointing to v1 and v2
        np1_at_v0 = _find_pnbr_slot(poly, v0, v1, D)
        np2_at_v0 = _find_pnbr_slot(poly, v0, v2, D)
        poly.finds[np1_at_v0, np2_at_v0, v0] = Int32(f)
        poly.finds[np2_at_v0, np1_at_v0, v0] = Int32(f)
        # v1's slots pointing to v0 and v2
        np0_at_v1 = _find_pnbr_slot(poly, v1, v0, D)
        np2_at_v1 = _find_pnbr_slot(poly, v1, v2, D)
        poly.finds[np0_at_v1, np2_at_v1, v1] = Int32(f)
        poly.finds[np2_at_v1, np0_at_v1, v1] = Int32(f)
        # v2's slots pointing to v0 and v1
        np0_at_v2 = _find_pnbr_slot(poly, v2, v0, D)
        np1_at_v2 = _find_pnbr_slot(poly, v2, v1, D)
        poly.finds[np0_at_v2, np1_at_v2, v2] = Int32(f)
        poly.finds[np1_at_v2, np0_at_v2, v2] = Int32(f)
    end
    poly.nfaces = f

    # Facet IDs for the D-simplex. A D-simplex has D+1 facets — facet `u`
    # is the one OPPOSITE vertex u (i.e., the (D−1)-simplex on all other
    # D vertices). For vertex v, the facet opposite edge-slot k of v is
    # the facet opposite the OTHER endpoint of that edge: pnbrs[k, v].
    @inbounds for v in 1:nv, k in 1:D
        poly.facets[k, v] = poly.pnbrs[k, v]
    end
    poly.nfacets = nv

    # Per-facet outward normals + signed distances. Facet `u` is opposite
    # vertex u, so its supporting hyperplane is the affine span of the
    # other D vertices. Compute the outward unit normal by Gram-Schmidt
    # against (D − 1) facet edges, signed so n · v_u < n · (anything on
    # facet u). Then d_u = n · (any point on facet u), e.g. the first
    # non-u vertex.
    _grow_facet_metadata!(poly, nv)
    _init_simplex_facet_normals_nd!(poly, vertices, Val(D))
    return poly
end

# Compute outward facet normals + signed distances for a freshly
# initialized D-simplex. Each facet u is opposite vertex u; its
# supporting hyperplane passes through every other vertex. Algorithm:
# orthonormalize D − 1 edge vectors spanning facet u's tangent space
# via classical Gram–Schmidt, then orthogonalize a candidate direction
# (`base → v_u`) against the orthonormal basis, and normalize. Finally
# orient outward by flipping sign (the residual of the base→v_u
# projection points from the facet TOWARD v_u, which is INTO the
# simplex; outward is the opposite).
function _init_simplex_facet_normals_nd!(poly::FlatPolytope{D,T},
                                          vertices, ::Val{D}) where {D,T}
    nv = D + 1
    work = MVector{D,T}(ntuple(_ -> zero(T), Val(D)))
    edges = MMatrix{D, D, T}(zeros(T, D, D))   # columns 1..D-1 used
    @inbounds for u in 1:nv
        # Gather D−1 edges from a base facet vertex to the other facet
        # vertices.
        base_idx = (u == 1) ? 2 : 1
        base = vertices[base_idx]
        col = 0
        for w in 1:nv
            (w == u || w == base_idx) && continue
            col += 1
            for i in 1:D
                edges[i, col] = T(vertices[w][i]) - T(base[i])
            end
        end

        # Orthonormalize the edges: classical Gram–Schmidt.
        for c in 1:(D - 1)
            for cprev in 1:(c - 1)
                dotp = zero(T)
                for i in 1:D
                    dotp += edges[i, c] * edges[i, cprev]
                end
                for i in 1:D
                    edges[i, c] -= dotp * edges[i, cprev]
                end
            end
            len2 = zero(T)
            for i in 1:D
                len2 += edges[i, c] * edges[i, c]
            end
            len = sqrt(len2)
            @assert len > eps(T) "degenerate D-simplex facet at vertex $u: edges are linearly dependent"
            for i in 1:D
                edges[i, c] /= len
            end
        end

        # Candidate direction: base → v_u. After projecting out the
        # tangent components, `work` is orthogonal to facet u and
        # points from base toward v_u — i.e., INWARD across the facet.
        for i in 1:D
            work[i] = T(vertices[u][i]) - T(base[i])
        end
        for c in 1:(D - 1)
            dotp = zero(T)
            for i in 1:D
                dotp += work[i] * edges[i, c]
            end
            for i in 1:D
                work[i] -= dotp * edges[i, c]
            end
        end

        # Normalize and flip sign for outward orientation.
        len2 = zero(T)
        for i in 1:D
            len2 += work[i] * work[i]
        end
        len = sqrt(len2)
        @assert len > eps(T)
        for i in 1:D
            poly.facet_normals[i, u] = -work[i] / len
        end
        d_u = zero(T)
        for i in 1:D
            d_u += poly.facet_normals[i, u] * T(base[i])
        end
        poly.facet_distances[u] = d_u
    end
    return poly
end

# Linear scan over poly.pnbrs[1..D, v] for the slot containing `target`.
# D ≥ 4 simplex/box neighbour graphs are dense, so a 0-return is a bug.
@inline function _find_pnbr_slot(poly::FlatPolytope, v::Int, target::Int, D::Int)
    @inbounds for k in 1:D
        if Int(poly.pnbrs[k, v]) == target
            return k
        end
    end
    error("_find_pnbr_slot: vertex $v has no neighbour slot pointing to $target (D=$D)")
end

# ===========================================================================
# Phase 2 helpers — ergonomics for the overlap-layer style hot loop.
# ===========================================================================

"""
    aabb(poly::FlatPolytope{D,T}) -> (lo::NTuple{D,T}, hi::NTuple{D,T})

Axis-aligned bounding box of the polytope's current vertex set as a
pair of `NTuple{D,T}`s. O(`nverts`), 0 allocations. For an empty
polytope (`nverts == 0`) returns `((+Inf, …), (-Inf, …))` so callers
can use the result as the identity for `extend` reductions.
"""
function aabb(poly::FlatPolytope{2,T}) where {T}
    if poly.nverts <= 0
        return ((T(Inf), T(Inf)), (T(-Inf), T(-Inf)))
    end
    @inbounds begin
        x_lo = poly.positions[1, 1]; x_hi = x_lo
        y_lo = poly.positions[2, 1]; y_hi = y_lo
        for v in 2:poly.nverts
            x = poly.positions[1, v]; y = poly.positions[2, v]
            x < x_lo && (x_lo = x); x > x_hi && (x_hi = x)
            y < y_lo && (y_lo = y); y > y_hi && (y_hi = y)
        end
    end
    return ((x_lo, y_lo), (x_hi, y_hi))
end

function aabb(poly::FlatPolytope{3,T}) where {T}
    if poly.nverts <= 0
        return ((T(Inf), T(Inf), T(Inf)), (T(-Inf), T(-Inf), T(-Inf)))
    end
    @inbounds begin
        x_lo = poly.positions[1, 1]; x_hi = x_lo
        y_lo = poly.positions[2, 1]; y_hi = y_lo
        z_lo = poly.positions[3, 1]; z_hi = z_lo
        for v in 2:poly.nverts
            x = poly.positions[1, v]; y = poly.positions[2, v]; z = poly.positions[3, v]
            x < x_lo && (x_lo = x); x > x_hi && (x_hi = x)
            y < y_lo && (y_lo = y); y > y_hi && (y_hi = y)
            z < z_lo && (z_lo = z); z > z_hi && (z_hi = z)
        end
    end
    return ((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))
end

"""
    aabb(poly::StaticFlatPolytope) -> (lo, hi)

`StaticFlatPolytope` analog of [`aabb`](@ref).
"""
function aabb(poly::StaticFlatPolytope{3,T,N,DN}) where {T,N,DN}
    if poly.nverts <= 0
        return ((T(Inf), T(Inf), T(Inf)), (T(-Inf), T(-Inf), T(-Inf)))
    end
    @inbounds begin
        x_lo = poly.positions[1, 1]; x_hi = x_lo
        y_lo = poly.positions[2, 1]; y_hi = y_lo
        z_lo = poly.positions[3, 1]; z_hi = z_lo
        for v in 2:poly.nverts
            x = poly.positions[1, v]; y = poly.positions[2, v]; z = poly.positions[3, v]
            x < x_lo && (x_lo = x); x > x_hi && (x_hi = x)
            y < y_lo && (y_lo = y); y > y_hi && (y_hi = y)
            z < z_lo && (z_lo = z); z > z_hi && (z_hi = z)
        end
    end
    return ((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi))
end

# D-generic axis-aligned bounding box. Used at D ≥ 4 (D = 2 / D = 3
# have specialized methods above that unroll the per-axis update).
function aabb(poly::FlatPolytope{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    if poly.nverts <= 0
        return (ntuple(_ -> T( Inf), Val(D)),
                ntuple(_ -> T(-Inf), Val(D)))
    end
    lo = MVector{D,T}(ntuple(k -> @inbounds(poly.positions[k, 1]), Val(D)))
    hi = MVector{D,T}(ntuple(k -> @inbounds(poly.positions[k, 1]), Val(D)))
    @inbounds for v in 2:poly.nverts, k in 1:D
        x = poly.positions[k, v]
        x < lo[k] && (lo[k] = x)
        x > hi[k] && (hi[k] = x)
    end
    return (Tuple(lo), Tuple(hi))
end

function aabb(poly::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 3 has a specialized method above"
    if poly.nverts <= 0
        return (ntuple(_ -> T( Inf), Val(D)),
                ntuple(_ -> T(-Inf), Val(D)))
    end
    lo = MVector{D,T}(ntuple(k -> @inbounds(poly.positions[k, 1]), Val(D)))
    hi = MVector{D,T}(ntuple(k -> @inbounds(poly.positions[k, 1]), Val(D)))
    @inbounds for v in 2:poly.nverts, k in 1:D
        x = poly.positions[k, v]
        x < lo[k] && (lo[k] = x)
        x > hi[k] && (hi[k] = x)
    end
    return (Tuple(lo), Tuple(hi))
end

"""
    box_planes(lo, hi) -> Vector{Plane{D,T}}

The `2D` clipping planes for the axis-aligned box `[lo[1], hi[1]] × …`,
in the `n·x + d ≥ 0`-keeps convention used by [`clip!`](@ref).
Allocates the result; for hot loops use [`box_planes!`](@ref) into a
pre-sized buffer.

D = 2 returns 4 planes (in `+x, -x, +y, -y` order); D = 3 returns 6.
"""
function box_planes(lo::NTuple{2,T}, hi::NTuple{2,T}) where {T}
    out = Vector{Plane{2,T}}(undef, 4)
    box_planes!(out, lo, hi)
    return out
end

function box_planes(lo::NTuple{3,T}, hi::NTuple{3,T}) where {T}
    out = Vector{Plane{3,T}}(undef, 6)
    box_planes!(out, lo, hi)
    return out
end

# D ≥ 4 box_planes: explicit per-D dispatches to keep Aqua's
# unbound-args check happy (a single `where {D, T}` method binds
# both parameters only via the argument NTuple, which Aqua flags
# as ambiguous when `D == 0`). Same pattern as the existing 2D / 3D
# specializations above.
for _D in 4:6
    @eval function box_planes(lo::NTuple{$_D,T}, hi::NTuple{$_D,T}) where {T}
        out = Vector{Plane{$_D,T}}(undef, 2 * $_D)
        box_planes!(out, lo, hi)
        return out
    end
end

"""
    box_planes!(out::AbstractVector{Plane{D,T}}, lo, hi) -> out

Write the `2D` axis-aligned-box clipping planes for `[lo, hi]` into
`out` (must satisfy `length(out) == 2D`). Zero allocations.

Plane order: `(+axis_k, -axis_k)` pairs for `k = 1, …, D` — i.e.
`[+x, -x, +y, -y]` in 2D, `[+x, -x, +y, -y, +z, -z]` in 3D.
"""
function box_planes!(out::AbstractVector{Plane{2,T}},
                     lo::NTuple{2,T}, hi::NTuple{2,T}) where {T}
    @assert length(out) == 4 "box_planes! D=2 needs out of length 4, got $(length(out))"
    @inbounds begin
        out[1] = Plane{2,T}(Vec{2,T}(T( 1), T(0)), -lo[1])
        out[2] = Plane{2,T}(Vec{2,T}(T(-1), T(0)),  hi[1])
        out[3] = Plane{2,T}(Vec{2,T}(T(0),  T( 1)), -lo[2])
        out[4] = Plane{2,T}(Vec{2,T}(T(0),  T(-1)),  hi[2])
    end
    return out
end

function box_planes!(out::AbstractVector{Plane{3,T}},
                     lo::NTuple{3,T}, hi::NTuple{3,T}) where {T}
    @assert length(out) == 6 "box_planes! D=3 needs out of length 6, got $(length(out))"
    @inbounds begin
        out[1] = Plane{3,T}(Vec{3,T}(T( 1), T(0), T(0)), -lo[1])
        out[2] = Plane{3,T}(Vec{3,T}(T(-1), T(0), T(0)),  hi[1])
        out[3] = Plane{3,T}(Vec{3,T}(T(0),  T( 1), T(0)), -lo[2])
        out[4] = Plane{3,T}(Vec{3,T}(T(0),  T(-1), T(0)),  hi[2])
        out[5] = Plane{3,T}(Vec{3,T}(T(0),  T(0),  T( 1)), -lo[3])
        out[6] = Plane{3,T}(Vec{3,T}(T(0),  T(0),  T(-1)),  hi[3])
    end
    return out
end

function box_planes!(out::AbstractVector{Plane{D,T}},
                     lo::NTuple{D,T}, hi::NTuple{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    @assert length(out) == 2D "box_planes! D=$D needs out of length $(2D), got $(length(out))"
    @inbounds for k in 1:D
        n_pos = ntuple(j -> j == k ? T( 1) : T(0), Val(D))
        n_neg = ntuple(j -> j == k ? T(-1) : T(0), Val(D))
        out[2k - 1] = Plane{D,T}(Vec{D,T}(n_pos), -lo[k])
        out[2k    ] = Plane{D,T}(Vec{D,T}(n_neg),  hi[k])
    end
    return out
end

"""
    is_empty(poly::FlatPolytope) -> Bool

`true` iff the polytope has been fully clipped away (`nverts == 0`).
"""
@inline is_empty(poly::FlatPolytope) = poly.nverts <= 0
@inline is_empty(poly::StaticFlatPolytope) = poly.nverts <= 0

"""
    volume(poly::FlatPolytope{D,T}) -> T

The 0-th moment of the polytope (signed volume in 3D, signed area in
2D). Allocation-free convenience around `moments!` with a 1-element
output.
"""
function volume(poly::FlatPolytope{D,T}) where {D,T}
    poly.nverts <= 0 && return zero(T)
    out = poly.sdists       # piggy-back on the per-call scratch — bounded ≥ 1
    moments!(out, poly, 0)
    return @inbounds out[1]
end

function volume(poly::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN}
    poly.nverts <= 0 && return zero(T)
    out = poly.sdists
    moments!(out, poly, 0)
    return @inbounds out[1]
end

"""
    copy!(dst::FlatPolytope{D,T}, src::FlatPolytope{D,T}) -> dst

Copy `src`'s vertex graph (positions, pnbrs, nverts) into `dst`. Does
NOT touch `dst`'s scratch buffers. `dst.capacity` must be ≥ `src.nverts`.
"""
@inline function copy!(dst::FlatPolytope{2,T}, src::FlatPolytope{2,T}) where {T}
    return _copy_polytope_2d!(dst, src)
end

@inline function copy!(dst::FlatPolytope{3,T}, src::FlatPolytope{3,T}) where {T}
    return _copy_polytope!(dst, src)
end

@inline function copy!(dst::FlatPolytope{D,T}, src::FlatPolytope{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    return _copy_polytope_nd!(dst, src)
end

# ===========================================================================
# Multi-threaded batched voxelization. Each thread gets its own
# VoxelizeWorkspace (and its own polytope copy if it'll be consumed).
# ===========================================================================

"""
    voxelize_batch!(grids, polys, ibox_lo, ibox_hi, d, order;
                    workspaces = nothing) -> grids

Voxelize each `polys[k]` into `grids[k]` in parallel via
`Threads.@threads`. Each thread needs its own `VoxelizeWorkspace`;
pass them as `workspaces` (a vector of length `nthreads()`), or omit
to allocate a fresh one per thread.

`grids` is a vector of pre-allocated grids (each shape
`(nmom, ni, nj, nk)`); they are zeroed before voxelization.
`polys`, `ibox_lo`, `ibox_hi` are vectors of the same length.

Run with `julia --threads=N` to actually get parallelism. The
serial path falls out when `nthreads() == 1`.
"""
function voxelize_batch!(grids::AbstractVector,
                         polys::AbstractVector,
                         ibox_lo::AbstractVector,
                         ibox_hi::AbstractVector,
                         d::NTuple{3,T},
                         order::Int;
                         workspaces::Union{Nothing,AbstractVector} = nothing) where {T}
    n = length(polys)
    @assert length(grids) == n
    @assert length(ibox_lo) == n
    @assert length(ibox_hi) == n

    # `Threads.threadid()` can exceed `nthreads()` with the interactive
    # thread pool or task migration, so size by `maxthreadid()`.
    nthr = Threads.maxthreadid()
    ws_local = if workspaces === nothing
        # Each thread allocates its own workspace once.
        cap = isempty(polys) ? 64 : polys[1].capacity
        [VoxelizeWorkspace{3,T}(cap) for _ in 1:nthr]
    else
        @assert length(workspaces) >= nthr
        workspaces
    end

    Threads.@threads for k in 1:n
        tid = Threads.threadid()
        fill!(grids[k], zero(T))
        voxelize!(grids[k], polys[k], ibox_lo[k], ibox_hi[k], d, order;
                  workspace = ws_local[tid])
    end
    return grids
end

"""
    voxelize_batch!(grids, polys::AbstractVector{<:FlatPolytope{2,T}},
                    ibox_lo, ibox_hi, d::NTuple{2,T}, order;
                    workspaces = nothing)

2D variant of [`voxelize_batch!`](@ref). `grids[k]` has shape
`(nmom, ni, nj)`; `ibox_lo[k]`, `ibox_hi[k]` are `NTuple{2,Int}`.
"""
function voxelize_batch!(grids::AbstractVector,
                         polys::AbstractVector{<:FlatPolytope{2,T}},
                         ibox_lo::AbstractVector{<:NTuple{2,Int}},
                         ibox_hi::AbstractVector{<:NTuple{2,Int}},
                         d::NTuple{2,T},
                         order::Int;
                         workspaces::Union{Nothing,AbstractVector} = nothing) where {T}
    n = length(polys)
    @assert length(grids) == n

    # `Threads.threadid()` can exceed `nthreads()` with the interactive
    # thread pool or task migration, so size by `maxthreadid()`.
    nthr = Threads.maxthreadid()
    ws_local = if workspaces === nothing
        cap = isempty(polys) ? 64 : polys[1].capacity
        [VoxelizeWorkspace{2,T}(cap) for _ in 1:nthr]
    else
        @assert length(workspaces) >= nthr
        workspaces
    end

    Threads.@threads for k in 1:n
        tid = Threads.threadid()
        fill!(grids[k], zero(T))
        voxelize!(grids[k], polys[k], ibox_lo[k], ibox_hi[k], d, order;
                  workspace = ws_local[tid])
    end
    return grids
end

# ---------------------------------------------------------------------------
# Base.show — REPL pretty-printing for FlatPolytope / StaticFlatPolytope
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", p::FlatPolytope{D,T}) where {D,T}
    println(io, "R3D.Flat.FlatPolytope{", D, ",", T, "} (capacity = ", p.capacity, ")")
    println(io, "  nverts = ", p.nverts)
    if p.nverts > 0
        nshow = min(p.nverts, 8)
        println(io, "  positions[:, 1:", nshow, "] =")
        for v in 1:nshow
            print(io, "    [", v, "] (")
            for k in 1:D
                k > 1 && print(io, ", ")
                Base.print(io, p.positions[k, v])
            end
            print(io, ")  pnbrs = (")
            for k in 1:D
                k > 1 && print(io, ", ")
                print(io, Int(p.pnbrs[k, v]))
            end
            println(io, ")")
        end
        p.nverts > nshow && println(io, "    … (", p.nverts - nshow, " more)")
    end
end

function Base.show(io::IO, p::FlatPolytope{D,T}) where {D,T}
    print(io, "FlatPolytope{", D, ",", T, "}(nverts=", p.nverts,
              ", capacity=", p.capacity, ")")
end

function Base.show(io::IO, ::MIME"text/plain", p::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN}
    println(io, "R3D.Flat.StaticFlatPolytope{", D, ",", T, ",", N, "} (capacity = ", N, ", MMatrix-backed)")
    println(io, "  nverts = ", p.nverts)
    if p.nverts > 0
        nshow = min(p.nverts, 8)
        println(io, "  positions[:, 1:", nshow, "] =")
        for v in 1:nshow
            print(io, "    [", v, "] (")
            for k in 1:D
                k > 1 && print(io, ", ")
                Base.print(io, p.positions[k, v])
            end
            print(io, ")  pnbrs = (")
            for k in 1:D
                k > 1 && print(io, ", ")
                print(io, Int(p.pnbrs[k, v]))
            end
            println(io, ")")
        end
        p.nverts > nshow && println(io, "    … (", p.nverts - nshow, " more)")
    end
end

function Base.show(io::IO, p::StaticFlatPolytope{D,T,N,DN}) where {D,T,N,DN}
    print(io, "StaticFlatPolytope{", D, ",", T, ",", N, "}(nverts=", p.nverts, ")")
end

# ===========================================================================
# D ≥ 4 clip! port — translation of rNd_clip from src/rNd.c.
#
# The algorithm is the same five-step structure as the 3D `clip!`:
# (1) signed distances, (2) trivial accept/reject, (3) insert new
# vertices on cut edges, (4) walk 2-face boundaries to close newly
# created faces, (5) compact. The genuinely D-generic piece is step
# 4, which uses the `finds[D][D]` 2-face table to walk around new
# face boundaries — in D = 3 this collapses to an edge walk (handled
# by `find_back3` / `next3`), but in D ≥ 4 each face has its own
# boundary that needs explicit tracking.
#
# Indexing convention shift from the C source:
#   - C: 0-based vertex / pnbrs / face indices, sentinel `-1` for unset.
#   - Julia: 1-based throughout, sentinel `0` for unset.
# ===========================================================================

"""
    clip!(poly::FlatPolytope{D,T}, planes) where {D ≥ 4}

Clip a `D`-dimensional polytope (D ≥ 4) in place against an array of
half-spaces `planes` (`Plane{D,T}`). Mirrors `rNd_clip` (`src/rNd.c`).

Returns `true` on success, `false` on capacity overflow.

Requires the polytope's `finds[D][D]` 2-face table to be populated (this
happens automatically when the polytope is built with `init_box!` or
`init_simplex!` for `D ≥ 4`).
"""
function clip!(poly::FlatPolytope{D,T},
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

"""
    clip!(poly::FlatPolytope{D,T}, plane::Plane{D,T}) where {D ≥ 4} -> Bool

Single-plane convenience overload mirroring the multi-plane API. Avoids
the per-call `Vector{Plane}` allocation that `clip!(poly, [plane])`
would otherwise pay — important for the bisection loop in
[`voxelize_fold!`](@ref) at `D ≥ 4`.
"""
function clip!(poly::FlatPolytope{D,T}, plane::Plane{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    poly.nverts <= 0 && return false
    return clip_plane!(poly, plane)
end

"""
    clip_plane!(poly::FlatPolytope{D,T}, plane::Plane{D,T}) where {D ≥ 4} -> Bool

Apply a single half-space clip to a `D ≥ 4` polytope (translation of
one iteration of `rNd_clip`'s outer loop). Public single-plane API,
parallel to the `clip_plane!` for `D = 2` / `D = 3`. Reuses the
polytope's preallocated `sdists` / `clipped` scratch — zero per-call
heap allocation.
"""
function clip_plane!(poly::FlatPolytope{D,T}, plane::Plane{D,T}) where {D,T}
    @assert D >= 4 "this method is for D ≥ 4 only; D = 2 / D = 3 have specialized methods"
    sdists  = poly.sdists
    clipped = poly.clipped
    onv     = poly.nverts

    # --- Step 1: signed distances + clipped flags
    smin = T(Inf); smax = T(-Inf)
    @inbounds for v in 1:onv
        s = plane.d
        for i in 1:D
            s += poly.positions[i, v] * plane.n[i]
        end
        sdists[v] = s
        s < smin && (smin = s)
        s > smax && (smax = s)
        clipped[v] = s < 0 ? Int32(1) : Int32(0)
    end

    # --- Step 2: trivial accept/reject
    smin >= 0 && return true
    if smax <= 0
        poly.nverts = 0
        return true
    end

    # --- Step 3: insert new vertices on each cut edge.
    # For a kept vertex `vcur` whose neighbour `vnext` is clipped, we
    # insert a new vertex at the cut. The new vertex inherits a partial
    # `finds` connectivity from `vcur` (the row 0 / column 0 entries are
    # populated; the rest are sentinel-zero for the linker to fill).
    #
    # Facet propagation: each clip creates ONE new facet (the cut
    # hyperplane). Every new vertex's slot 1 points back to vcur (the
    # kept side) — so the facet OPPOSITE that slot is the cut. Other
    # slots inherit facets from vcur (a facet that contains both vcur
    # and vnext stays intact through the cut, so the new vertex is on
    # it too). new_v slot k (k = 2..D) inherits from vcur slot np_k,
    # using the same k_new → k_orig mapping as the finds row-0 fill
    # above (i.e., vcur's non-np slots in order).
    new_facet_id = Int32(poly.nfacets + 1)
    # ε-nudge for the cut-position formula. When sdists[vcur] is exactly
    # zero (vcur lies on the cut plane), the formula
    # (vnext * sd_vcur - vcur * sd_vnext) / (sd_vcur - sd_vnext)
    # collapses to vcur, creating duplicate vertices that break the
    # simple-polytope invariant the LTD moments recursion assumes.
    # That manifested as wrong volumes for D ≥ 4 sequential `clip!` on
    # simplex-like polytopes (e.g. axis-aligned half-plane clips at 0.5
    # of a unit D = 4 simplex; the upstream C `rNd` has the same bug,
    # producing NaN). Nudging sd_vcur up by `tol_nudge` shifts the new
    # vertex ε-close-but-distinct from vcur, preserving simplicity. The
    # volume error is O(eps(T)) — well below floating-point precision
    # in any realistic moments computation, and only triggered on the
    # measure-zero case sd_vcur == 0 (random clip planes never hit it,
    # so existing diff-tests against C remain bit-exact).
    tol_nudge = eps(T) * max(abs(smin), abs(smax), one(T)) * 256
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        sd_vcur = sdists[vcur] < tol_nudge ? tol_nudge : sdists[vcur]
        for np in 1:D
            vnext = Int(poly.pnbrs[np, vcur])
            (vnext == 0 || clipped[vnext] == 0) && continue

            poly.nverts >= poly.capacity && return false
            new_v = (poly.nverts += 1)

            # Position: weighted average between vcur and vnext.
            # Match the C formula exactly to keep diff-test agreement at
            # floating-point precision: (vnext * sd_vcur - vcur * sd_vnext) / (sd_vcur - sd_vnext)
            for i in 1:D
                poly.positions[i, new_v] =
                    (poly.positions[i, vnext] * sd_vcur -
                     poly.positions[i, vcur]  * sdists[vnext]) /
                    (sd_vcur - sdists[vnext])
            end

            # pnbrs[1] = vcur (the kept vertex side); rest sentinel-zero.
            poly.pnbrs[1, new_v] = Int32(vcur)
            for k in 2:D
                poly.pnbrs[k, new_v] = Int32(0)
            end
            # vcur's slot for vnext now points at the new vertex.
            poly.pnbrs[np, vcur] = Int32(new_v)

            # Slot 1 of new_v opposite vcur ⇒ the new cut facet.
            poly.facets[1, new_v] = new_facet_id

            # Carry 2-face IDs and facet IDs from vcur into new_v.
            # For each np0 ≠ np in vcur's pnbrs, the 2-face
            # vcur.finds[np][np0] becomes new_v.finds[0][np1] / [np1][0]
            # AND the facet vcur.facets[np0] is also incident to new_v
            # via slot np1. (np1 = 2..D in new_v, in the same enumeration
            # order as np0 = 1..D excluding np in vcur.)
            np1 = 1
            for np0 in 1:D
                np0 == np && continue
                np1 += 1
                fid = poly.finds[np, np0, vcur]
                poly.finds[1,   np1, new_v] = fid
                poly.finds[np1, 1,   new_v] = fid
                poly.facets[np1, new_v] = poly.facets[np0, vcur]
            end
            # Mark everything else (interior of the finds matrix) as 0
            # for the linker to fill.
            for np0 in 2:D, np1b in 2:D
                poly.finds[np0, np1b, new_v] = Int32(0)
            end
            clipped[new_v] = Int32(0)
        end
    end

    # If we got past the trivial accept/reject checks, at least one
    # edge was cut (the polytope is connected and has vertices on both
    # sides), so we have inserted at least one new vertex onto the cut
    # facet. Commit the new facet ID and stash its supporting hyperplane
    # for Lasserre's higher-order moments recursion. The clip-plane's
    # `n` and `d` already give the cut hyperplane: `n · x + d = 0` ⇒
    # `n · x = -d`, but our convention stores facet normals pointing
    # OUTWARD of the kept half-space. Plane `n` points INTO the kept
    # half-space (positive sdists are kept), so the outward facet
    # normal is `-n` and the matching signed distance is `-(-d) = d`?
    # Re-derive carefully: the kept set is {x : n·x + d ≥ 0}, i.e.,
    # {x : n·x ≥ -d}. The outward normal of the cut facet (pointing
    # AWAY from kept) is `-n`, and any point on the facet satisfies
    # `(-n) · x = -(-d) = d`... no, `n · x = -d` ⇒ `(-n) · x = d`.
    # So outward_n · x = d for points on the cut, i.e., the signed
    # distance is `d`. (Equivalently, the new facet's outward
    # half-space is the discarded side.)
    poly.nfacets += 1
    new_id = poly.nfacets
    _grow_facet_metadata!(poly, new_id)
    @inbounds for i in 1:D
        poly.facet_normals[i, new_id] = -plane.n[i]
    end
    @inbounds poly.facet_distances[new_id] = plane.d

    # --- Step 4: walk 2-face boundaries to close all new 2-faces.
    # For each new vertex `vstart` and each pair of unset slot-pairs
    # (np0, np1) with np0 < np1 (both ≥ 2 in our 1-based scheme),
    # walk around the boundary of a newly-created 2-face. Each step
    # patches `pnbrs` and `finds` so that the new face has consistent
    # connectivity once the walk returns to vstart.
    nfaces = poly.nfaces
    @inbounds for vstart in (onv + 1):poly.nverts, np0 in 2:D, np1 in (np0 + 1):D
        if poly.finds[np0, np1, vstart] != 0
            continue   # already closed
        end
        # The two faces incident to (vstart's edges in slots np0 and np1)
        # via row 0 of vstart's finds matrix. The walker will alternate
        # which is "current" and which is "adjacent" each time it
        # crosses through a new vertex.
        fcur = Int(poly.finds[1, np0, vstart])
        fadj = Int(poly.finds[1, np1, vstart])
        vprev = vstart
        vcur  = Int(poly.pnbrs[1, vstart])    # the original vertex on
                                                # the kept side that vstart
                                                # was inserted onto.
        prevnewvert = vstart

        # Walk until we return to vstart. The new-face ID for this walk
        # is `nfaces + 1`; every patch along the walk (both inside-loop
        # at intermediate new vertices AND the final close-loop patch)
        # uses the SAME ID — they all belong to the single 2-face the
        # walk is tracing. Only after the walk completes do we increment
        # `nfaces` once to make this ID committed and pick a fresh one
        # for the next walk.
        new_fid = Int32(nfaces + 1)
        while true
            if vcur > onv
                pprev = _find_face_in_row0(poly, vcur, fcur, D)
                pnext = _find_face_in_row0(poly, prevnewvert, fcur, D)
                npx   = _find_face_in_row0(poly, vcur, fadj, D)
                poly.pnbrs[pprev, vcur]            = Int32(prevnewvert)
                poly.pnbrs[pnext, prevnewvert]     = Int32(vcur)
                poly.finds[pprev, npx, vcur]       = new_fid
                poly.finds[npx, pprev, vcur]       = new_fid
                # Swap fcur ↔ fadj for the next leg.
                fcur, fadj = fadj, fcur
                prevnewvert = vcur
                vprev = vcur
                vcur  = Int(poly.pnbrs[1, vcur])
            end
            pprev = _find_pnbr_slot_or0(poly, vcur, vprev, D)
            pnext = _find_face_in_finds_row(poly, vcur, pprev, fcur, D)
            npx   = _find_face_in_finds_col(poly, vcur, pprev, fadj, D)
            fadj = Int(poly.finds[npx, pnext, vcur])
            vprev = vcur
            vcur  = Int(poly.pnbrs[pnext, vcur])
            vcur == vstart && break
        end
        # Final close patch (same shape as inside-loop branch, with the
        # same `new_fid`).
        pprev = _find_face_in_row0(poly, vcur, fcur, D)
        pnext = _find_face_in_row0(poly, prevnewvert, fcur, D)
        npx   = _find_face_in_row0(poly, vcur, fadj, D)
        poly.pnbrs[pprev, vcur]        = Int32(prevnewvert)
        poly.pnbrs[pnext, prevnewvert] = Int32(vcur)
        poly.finds[pprev, npx, vcur]   = new_fid
        poly.finds[npx, pprev, vcur]   = new_fid
        nfaces += 1
    end
    poly.nfaces = nfaces

    # --- Step 5: compact (same as 3D, generalized to D pnbrs slots).
    # facets[k, v] is also moved with v but does NOT need re-indexing
    # — facet IDs are global, not vertex-relative.
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                for i in 1:D
                    poly.positions[i, numunclipped] = poly.positions[i, v]
                    poly.pnbrs[i, numunclipped]     = poly.pnbrs[i, v]
                    poly.facets[i, numunclipped]    = poly.facets[i, v]
                end
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

# Linear scan of vcur's row 0 of finds for the face ID `target`. Used
# by the linker when walking through a new vertex.
@inline function _find_face_in_row0(poly::FlatPolytope, vcur::Int, target::Int, D::Int)
    @inbounds for k in 2:D
        if Int(poly.finds[1, k, vcur]) == target
            return k
        end
    end
    error("_find_face_in_row0: face $target not in row 0 of vertex $vcur")
end

# Same as _find_pnbr_slot but returns 0 instead of erroring (used in
# pnbrs-walk where 0 means "not found").
@inline function _find_pnbr_slot_or0(poly::FlatPolytope, v::Int, target::Int, D::Int)
    @inbounds for k in 1:D
        Int(poly.pnbrs[k, v]) == target && return k
    end
    return 0
end

# Find slot pnext such that finds[pprev, pnext, vcur] == target, with
# pnext ≠ pprev. Used in the linker's per-original-vertex step.
@inline function _find_face_in_finds_row(poly::FlatPolytope, vcur::Int, pprev::Int,
                                          target::Int, D::Int)
    @inbounds for k in 1:D
        k == pprev && continue
        Int(poly.finds[pprev, k, vcur]) == target && return k
    end
    error("_find_face_in_finds_row: face $target not in row $pprev of vertex $vcur")
end

# Find slot npx such that finds[npx, pprev, vcur] == target, with
# npx ≠ pprev.
@inline function _find_face_in_finds_col(poly::FlatPolytope, vcur::Int, pprev::Int,
                                          target::Int, D::Int)
    @inbounds for k in 1:D
        k == pprev && continue
        Int(poly.finds[k, pprev, vcur]) == target && return k
    end
    error("_find_face_in_finds_col: face $target not in column $pprev of vertex $vcur")
end

# ===========================================================================
# D ≥ 4 moments stub (Phase 3d). The 0th moment can be ported from
# rNd_reduce; higher orders need Lasserre-style decomposition or
# D-generic Koehl, which is a separate ~week of work.
# ===========================================================================

"""
    moments(poly::FlatPolytope{D,T}, order) where {D ≥ 4}
    moments!(out, poly::FlatPolytope{D,T}, order) where {D ≥ 4}

The 0th moment is computed via a port of `rNd_reduce` (orthogonalized
"line-tangent-distance" recursion at each vertex; `src/rNd.c:175–315`).
**Higher orders (order ≥ 1) are not yet implemented** — they require
either Lasserre's recursive decomposition or a D-generic Koehl
recursion, separate work tracked in `docs/phase3_status.md`.
"""
function moments(poly::FlatPolytope{D,T}, order::Integer) where {D,T}
    D >= 4 || error("moments reached the D ≥ 4 fallback at D = $D")
    if order == 0
        out = zeros(T, 1)
        _reduce_nd_zeroth!(out, poly)
        return out
    end
    if D == 4
        out = zeros(T, num_moments(4, Int(order)))
        _reduce_nd_higher_d4!(out, poly, Int(order))
        return out
    end
    error("R3D.Flat.moments: D = $D, order ≥ 1 not yet implemented. ",
          "Use `moments(poly, 0)` for the 0th moment, or D = 4 for higher orders. ",
          "D = 5 / D = 6 require an additional 3-face / 4-face tracking layer; ",
          "see docs/d4plus_finalization_plan.md.")
end

function moments!(out::AbstractVector{T}, poly::FlatPolytope{D,T},
                  order::Integer) where {D,T}
    D >= 4 || error("moments! reached the D ≥ 4 fallback at D = $D")
    if order == 0
        @assert length(out) >= 1
        _reduce_nd_zeroth!(out, poly)
        return out
    end
    if D == 4
        @assert length(out) >= num_moments(4, Int(order))
        _reduce_nd_higher_d4!(out, poly, Int(order))
        return out
    end
    error("R3D.Flat.moments!: D = $D, order ≥ 1 not yet implemented. ",
          "Use `moments!(out, poly, 0)` for the 0th moment, or D = 4 for higher orders. ",
          "D = 5 / D = 6 require additional codim-face tracking layers; ",
          "see docs/d4plus_finalization_plan.md.")
end

# ===========================================================================
# Phase A — Lasserre's recursive face decomposition for D = 4 moments at
# polynomial order ≥ 1. The formula:
#
#     ∫_P x^α dV  =  (1 / (D + |α|)) Σ_F d_F ∫_F x^α dA
#
# where each facet F has unit outward normal n_F, signed distance d_F =
# n_F · c_F (any c_F on F), and ∫_F x^α dA is a (D−1)-dim moment integral
# over the facet polytope. We project F into 3D coords y via x = c_F + B y
# (B is a 4×3 orthonormal basis for F's tangent plane), build a 3D
# `FlatPolytope`, call existing `moments!(_, ::FlatPolytope{3,T})`, then
# expand `(c_F + B y)^α` multinomially to map 3D y-monomials back to the
# 4D x-monomials.
#
# Validation: no C oracle exists at D ≥ 4 P ≥ 1 (upstream `rNd_reduce`'s
# higher-order branch is `#else`-blocked off). We test against closed-form
# unit D-simplex moments (`α! / (D + |α|)!`) and unit D-box moments
# (separable: `∏_j 1 / (α_j + 1)`), plus voxelize-fold consistency.
# ===========================================================================

"""
    _reduce_nd_higher_d4!(out::Vector{T}, poly::FlatPolytope{4,T}, P::Int)

Compute moments of `poly` for orders `0..P` via Lasserre. Writes
`num_moments(4, P)` entries into `out` in the canonical lex-by-degree
order returned by `_enumerate_moments_d4`.
"""
function _reduce_nd_higher_d4!(out::AbstractVector{T},
                                poly::FlatPolytope{4,T}, P::Int) where {T}
    @assert P >= 1 "_reduce_nd_higher_d4! is for P ≥ 1; use _reduce_nd_zeroth! for P = 0"
    nmom = num_moments(4, P)
    @assert length(out) >= nmom
    fill!(out, zero(T))
    poly.nverts <= 0 && return out

    # Enumerate the target 4D multi-indices α (length-nmom Vector{NTuple{4,Int}}).
    alphas = _enumerate_moments_d4(P)

    # Pre-allocate a 3D facet polytope buffer, sized for the largest facet.
    # `walk_facet_vertices` is O(nverts), and the largest facet of the unit
    # D-box has 2^(D-1) = 8 vertices; on a heavily-clipped polytope the
    # facet vertex count grows but stays bounded by `poly.nverts`.
    facet3d = FlatPolytope{3,T}(max(poly.nverts, 16))
    moments_3d = zeros(T, num_moments(3, P))

    # Per-facet scratch.
    B = zeros(T, 4, 3)
    for_facet_vmap = zeros(Int32, poly.nverts)   # 4D vertex idx → 3D vertex idx
    for_facet_slot = zeros(Int32, poly.nverts)   # 4D vertex idx → "wrong" slot
                                                  #  (slot k where facets[k,v] == fid)
    perm = zeros(Int32, 3, poly.nverts)          # 3D pnbrs slot → 4D pnbrs slot
                                                  #  per in-facet vertex
    bfs_queue = zeros(Int32, poly.nverts)        # BFS frontier scratch

    # Sum over facets.
    @inbounds for fid in 1:poly.nfacets
        # Build the 3D facet polytope. Returns false if facet is degenerate
        # (e.g. all in-facet vertices were clipped away, or the facet has
        # < 4 vertices in 4D — degenerate sub-simplex).
        ok = _build_facet3d!(facet3d, B, for_facet_vmap, for_facet_slot,
                              perm, bfs_queue, poly, fid)
        ok || continue
        c_F = (poly.positions[1, _first_facet_vertex(poly, fid)],
               poly.positions[2, _first_facet_vertex(poly, fid)],
               poly.positions[3, _first_facet_vertex(poly, fid)],
               poly.positions[4, _first_facet_vertex(poly, fid)])

        # 3D moments of the projected facet.
        moments!(moments_3d, facet3d, P)

        # If `facet3d` came out CW (negative volume), flip every entry so
        # the higher-moments use the canonical positive-area version. The
        # outward-normal sign on the parent polytope absorbs the choice.
        sgn = sign(moments_3d[1])
        sgn == zero(T) && continue   # truly degenerate facet
        if sgn < zero(T)
            for k in eachindex(moments_3d)
                moments_3d[k] = -moments_3d[k]
            end
        end

        d_F = poly.facet_distances[fid]
        # Lift each 3D y^β moment up to 4D x^α moments. For each α with
        # |α| ≤ P, ∫_F x^α dA = Σ_β coeff(α, β; c_F, B) · moments_3d[β].
        for ai in 1:nmom
            α = alphas[ai]
            order_α = α[1] + α[2] + α[3] + α[4]
            order_α == 0 && continue   # 0th moment handled separately
            integral = _expand_facet_monomial_d4(α, c_F, B, moments_3d, P)
            out[ai] += d_F * integral / (4 + order_α)
        end
    end

    # 0th moment: reuse the existing LTD-recursion zeroth-moment computation,
    # which is bit-exact to the C upstream and dodges Lasserre's accumulated
    # round-off on the volume term. Volume is at index 1 in the canonical
    # enumeration.
    zeroth = zeros(T, 1)
    _reduce_nd_zeroth!(zeroth, poly)
    out[1] = zeroth[1]
    return out
end

# Find the smallest-index vertex of `poly` that lies on facet `fid`. Used
# as the reference point `c_F` in Lasserre's per-facet projection.
@inline function _first_facet_vertex(poly::FlatPolytope{D,T},
                                      fid::Integer) where {D,T}
    @inbounds for v in 1:poly.nverts
        for k in 1:D
            if Int(poly.facets[k, v]) == Int(fid)
                return v
            end
        end
    end
    error("_first_facet_vertex: facet $fid has no incident vertices")
end

# Build a 3D `FlatPolytope` for the projected facet `fid` of `poly`. Writes:
# - `facet3d.positions[:, 1..nf]` from `B^T (poly.positions[:, v_4d] - c_F)`,
# - `facet3d.pnbrs[:, 1..nf]` from each in-facet vertex's three in-facet
#   neighbors mapped through `vmap`,
# - `B` (4×3 orthonormal basis perpendicular to the facet's outward normal),
# - `vmap[v_4d] = v_3d` for in-facet vertices, 0 otherwise.
# Returns `true` on success, `false` if the facet is too degenerate to
# project (< 4 vertices, or rank-deficient projection).
function _build_facet3d!(facet3d::FlatPolytope{3,T},
                          B::Matrix{T},
                          vmap::Vector{Int32},
                          wrong_slot::Vector{Int32},
                          perm::Matrix{Int32},
                          bfs_queue::Vector{Int32},
                          poly::FlatPolytope{4,T},
                          fid::Integer) where {T}
    fid_i = Int(fid)
    nf = 0
    fill!(vmap, Int32(0))
    fill!(wrong_slot, Int32(0))

    # Pass 1: enumerate in-facet vertices and record vmap + wrong slot.
    @inbounds for v in 1:poly.nverts
        for k in 1:4
            if Int(poly.facets[k, v]) == fid_i
                nf += 1
                vmap[v] = Int32(nf)
                wrong_slot[v] = Int32(k)
                break
            end
        end
    end
    nf < 4 && return false   # facet has < 4 vertices: degenerate
    facet3d.capacity >= nf || return false

    # Compute 3D basis B (4 × 3) ⊥ to the facet's outward normal.
    _orthonormal_perp_basis!(B, poly.facet_normals, fid_i)

    # Reference point c_F: position of the first in-facet vertex.
    v0 = _first_facet_vertex(poly, fid_i)
    c1 = poly.positions[1, v0]; c2 = poly.positions[2, v0]
    c3 = poly.positions[3, v0]; c4 = poly.positions[4, v0]

    # Pass 2: project positions and write a CONSISTENTLY-ORIENTED 3D
    # pnbrs table.
    #
    # The 3D moments code (`moments!(::FlatPolytope{3,T})`) walks faces
    # by cycling pnbrs slots via `next3`. For a face walk to close,
    # every vertex along the walk must have its 3 in-facet edges
    # ordered with consistent handedness — i.e. all "right-handed"
    # frames or all "left-handed" frames in the projected 3D
    # coordinate system.
    #
    # We achieve this purely geometrically: for each vertex v, lay
    # down the 3 in-facet edges in numerical-slot order, then check
    # `det(e_1, e_2, e_3)`. If its sign disagrees with the seed
    # vertex's sign, swap two slots to flip handedness.
    facet3d.nverts = nf
    seed_sign = zero(T)
    @inbounds for v in 1:poly.nverts
        v3d = Int(vmap[v])
        v3d == 0 && continue
        # Project position
        for j in 1:3
            facet3d.positions[j, v3d] =
                B[1, j] * (poly.positions[1, v] - c1) +
                B[2, j] * (poly.positions[2, v] - c2) +
                B[3, j] * (poly.positions[3, v] - c3) +
                B[4, j] * (poly.positions[4, v] - c4)
        end
    end
    @inbounds for v in 1:poly.nverts
        v3d = Int(vmap[v])
        v3d == 0 && continue
        # Gather the 3 in-facet edges in numerical-slot order.
        wrong = Int(wrong_slot[v])
        slot3d = 0
        nbr3d = (Int32(0), Int32(0), Int32(0))
        e = (zero(T), zero(T), zero(T))
        e1 = e2 = e3 = (zero(T), zero(T), zero(T))
        nbr_3d_a = Int32(0); nbr_3d_b = Int32(0); nbr_3d_c = Int32(0)
        for k in 1:4
            k == wrong && continue
            slot3d += 1
            nbr_4d = Int(poly.pnbrs[k, v])
            n3d = vmap[nbr_4d]
            ex = facet3d.positions[1, n3d] - facet3d.positions[1, v3d]
            ey = facet3d.positions[2, n3d] - facet3d.positions[2, v3d]
            ez = facet3d.positions[3, n3d] - facet3d.positions[3, v3d]
            if slot3d == 1
                nbr_3d_a = n3d; e1 = (ex, ey, ez)
            elseif slot3d == 2
                nbr_3d_b = n3d; e2 = (ex, ey, ez)
            else
                nbr_3d_c = n3d; e3 = (ex, ey, ez)
            end
        end
        # det([e1 e2 e3]) — signed volume of parallelepiped.
        det_v = e1[1] * (e2[2] * e3[3] - e2[3] * e3[2]) -
                e1[2] * (e2[1] * e3[3] - e2[3] * e3[1]) +
                e1[3] * (e2[1] * e3[2] - e2[2] * e3[1])
        if seed_sign == zero(T)
            seed_sign = sign(det_v)
        end
        # If this vertex's handedness disagrees with the seed, swap
        # slots 2 and 3 to flip it.
        if det_v == zero(T)
            return false   # degenerate vertex (collinear edges)
        end
        if sign(det_v) != seed_sign
            nbr_3d_b, nbr_3d_c = nbr_3d_c, nbr_3d_b
        end
        facet3d.pnbrs[1, v3d] = nbr_3d_a
        facet3d.pnbrs[2, v3d] = nbr_3d_b
        facet3d.pnbrs[3, v3d] = nbr_3d_c
    end

    # Reset moment scratch so the 3D moments! resizes lazily for `P` only.
    facet3d.moment_order = -1
    return true
end

# Compute a 4×3 orthonormal basis whose columns are perpendicular to
# `poly.facet_normals[:, fid]`. Writes the basis into `B`. The basis is
# computed by Gram-Schmidt against the standard 4D unit vectors, dropping
# the one most-parallel to the normal.
@inline function _orthonormal_perp_basis!(B::Matrix{T},
                                           facet_normals::Matrix{T},
                                           fid::Int) where {T}
    n1 = facet_normals[1, fid]; n2 = facet_normals[2, fid]
    n3 = facet_normals[3, fid]; n4 = facet_normals[4, fid]
    # Drop the unit axis most-parallel to n (largest |n_k|), to avoid a
    # near-zero residual in the Gram-Schmidt step.
    abs_n = (abs(n1), abs(n2), abs(n3), abs(n4))
    drop = argmax(abs_n)
    col = 0
    @inbounds for k in 1:4
        k == drop && continue
        col += 1
        # e_k orthogonalized against n.
        # First column of B = e_k - (n_k) * n  (since n is unit length).
        nk = facet_normals[k, fid]
        B[1, col] = (k == 1 ? T(1) : T(0)) - nk * n1
        B[2, col] = (k == 2 ? T(1) : T(0)) - nk * n2
        B[3, col] = (k == 3 ? T(1) : T(0)) - nk * n3
        B[4, col] = (k == 4 ? T(1) : T(0)) - nk * n4
    end
    # Now classical Gram-Schmidt across columns 1..3 of B.
    @inbounds for c in 1:3
        for cprev in 1:(c - 1)
            dotp = B[1, c] * B[1, cprev] + B[2, c] * B[2, cprev] +
                   B[3, c] * B[3, cprev] + B[4, c] * B[4, cprev]
            B[1, c] -= dotp * B[1, cprev]
            B[2, c] -= dotp * B[2, cprev]
            B[3, c] -= dotp * B[3, cprev]
            B[4, c] -= dotp * B[4, cprev]
        end
        len2 = B[1, c]^2 + B[2, c]^2 + B[3, c]^2 + B[4, c]^2
        len = sqrt(len2)
        @assert len > eps(T) "_orthonormal_perp_basis!: degenerate facet normal"
        B[1, c] /= len; B[2, c] /= len
        B[3, c] /= len; B[4, c] /= len
    end
    return B
end

# Enumerate all 4D multi-indices α = (α_1, α_2, α_3, α_4) with
# |α| ≤ P, in the canonical "lex by total degree, then lex within
# each degree" order matching `num_moments(4, P)`. Returned as a
# vector of NTuple{4,Int}, length = num_moments(4, P).
#
# The order matches what `moments!` writes for D = 4: index 1 is the
# zeroth moment (volume), then degree-1 moments in order
# (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), then degree-2, etc.
function _enumerate_moments_d4(P::Int)
    out = NTuple{4,Int}[]
    for total in 0:P
        for a in total:-1:0, b in (total - a):-1:0, c in (total - a - b):-1:0
            d = total - a - b - c
            push!(out, (a, b, c, d))
        end
    end
    return out
end

# Return the index of 3D multi-index β = (β_1, β_2, β_3) within the
# canonical D = 3 moments enumeration. Matches the order produced by
# `moments!(::FlatPolytope{3,T})`'s outer loop:
#   for total in 0..P
#     for i in total..0, j in (total-i)..0
#       k = total - i - j
@inline function _moment_index_d3(β::NTuple{3,Int})
    total = β[1] + β[2] + β[3]
    # Skip count: number of multi-indices with total < `total`.
    idx = 0
    for t in 0:(total - 1)
        idx += div((t + 1) * (t + 2), 2)
    end
    # Within `total`, position of β = (i, j, k) with i + j + k = total,
    # ordered by i descending, then j descending. The index of (i, j, k)
    # in the per-degree block is computed as:
    #   sum_{i' > i} (total - i' + 1)   +   (total - β[1] - β[2])
    i_cur = β[1]
    sub = 0
    for i_above in (total):-1:(i_cur + 1)
        sub += total - i_above + 1
    end
    sub += (total - β[1]) - β[2]   # (total - i) - j gives offset within row
    return idx + sub + 1
end

# Compute ∫_F x^α dA expressed in terms of 3D moments of the projected
# facet. Substitutes x_j = c_F[j] + Σ_k B[j, k] y_k into x^α and
# multinomially expands, summing each y^β monomial weighted by
# `moments_3d[_moment_index_d3(β)]`.
#
# Implementation: build the polynomial as a dense `Array{T,3}` of size
# `(P+1) × (P+1) × (P+1)` indexed by β = (b_1, b_2, b_3) where the entry
# is the coefficient. Multiply factor-by-factor. For P ≤ 3 this stays
# tiny (≤ 64 entries) so we don't sweat allocation.
function _expand_facet_monomial_d4(α::NTuple{4,Int},
                                    c::NTuple{4,T},
                                    B::Matrix{T},
                                    moments_3d::Vector{T},
                                    P::Int) where {T}
    # Polynomial in y: poly[b1+1, b2+1, b3+1] = coefficient of y_1^b1 y_2^b2 y_3^b3.
    nb = P + 1
    poly_buf = zeros(T, nb, nb, nb)
    poly_new = zeros(T, nb, nb, nb)
    @inbounds poly_buf[1, 1, 1] = one(T)   # start with constant 1

    @inbounds for j in 1:4
        αj = α[j]
        αj == 0 && continue
        # Multiply poly_buf by (c_j + B[j,1] y_1 + B[j,2] y_2 + B[j,3] y_3)^{αj}.
        # Do it αj times, multiplying by the linear factor each time.
        cj  = c[j]
        Bj1 = B[j, 1]; Bj2 = B[j, 2]; Bj3 = B[j, 3]
        for _step in 1:αj
            fill!(poly_new, zero(T))
            for b1 in 0:(P), b2 in 0:(P - b1), b3 in 0:(P - b1 - b2)
                pv = poly_buf[b1 + 1, b2 + 1, b3 + 1]
                pv == zero(T) && continue
                # constant term: c_j * y^β
                poly_new[b1 + 1, b2 + 1, b3 + 1] += cj * pv
                # B[j,1] * y_1 * y^β = B[j,1] * y^(β + e_1)
                if b1 + 1 < nb
                    poly_new[b1 + 2, b2 + 1, b3 + 1] += Bj1 * pv
                end
                if b2 + 1 < nb
                    poly_new[b1 + 1, b2 + 2, b3 + 1] += Bj2 * pv
                end
                if b3 + 1 < nb
                    poly_new[b1 + 1, b2 + 1, b3 + 2] += Bj3 * pv
                end
            end
            # Swap roles via copy (small array, fine for now).
            poly_buf, poly_new = poly_new, poly_buf
        end
    end

    # Sum poly_buf[β] * moments_3d[index_of(β)] over all β with |β| ≤ P.
    s = zero(T)
    @inbounds for b1 in 0:P, b2 in 0:(P - b1), b3 in 0:(P - b1 - b2)
        coeff = poly_buf[b1 + 1, b2 + 1, b3 + 1]
        coeff == zero(T) && continue
        s += coeff * moments_3d[_moment_index_d3((b1, b2, b3))]
    end
    return s
end

# Port of rNd_reduce (`src/rNd.c:252-315`) for the 0th moment only.
# Sums per-vertex orthogonalized LTD terms; each per-vertex term is
# itself a recursive Gram-Schmidt over the D! permutations of the
# vertex's D outgoing edges.
function _reduce_nd_zeroth!(out::AbstractVector{T},
                            poly::FlatPolytope{D,T}) where {D,T}
    out[1] = zero(T)
    poly.nverts <= 0 && return out
    # The LTD scratch is owned by the polytope so `_reduce_helper_nd`
    # can mutate it in place without per-call heap allocation. The
    # `processed` set is a UInt32 bitfield (rather than an
    # `MVector{D,Bool}`) so Julia's escape analysis sees it as a plain
    # by-value scalar across the recursive call boundary. Both matter
    # for `voxelize_fold!`'s hot loop, which calls this once per leaf.
    @assert D <= 32 "_reduce_nd_zeroth! UInt32 processed-bitfield supports up to D = 32"
    ltd = poly.ltd_scratch
    s = zero(T)
    @inbounds for v in 1:poly.nverts
        s += _reduce_helper_nd(poly, v, 0, UInt32(0), ltd, Val(D))
    end
    out[1] = s
    return out
end

# Recursive Gram-Schmidt LTD helper. `d` is the current depth (0-based,
# matching the C source); `processed` is a UInt32 bitfield where bit
# (i-1) indicates whether vertex slot `i` has already been used at a
# higher-up recursion level. When d == D we have a full orthonormal
# frame and accumulate its contribution.
function _reduce_helper_nd(poly::FlatPolytope{D,T}, v::Int, d::Int,
                           processed::UInt32,
                           ltd::Matrix{T},
                           ::Val{D}) where {D,T}
    if d == D
        # Full LTD: product of dot(ltd[:,dd], pos_v) / (dd+1) for dd in 0..D-1.
        ltdsum = one(T)
        @inbounds for dd in 1:D
            dot = zero(T)
            for j in 1:D
                dot += ltd[j, dd] * poly.positions[j, v]
            end
            ltdsum *= dot / dd
        end
        return ltdsum
    end

    s = zero(T)
    @inbounds for i in 1:D
        ((processed >> (i - 1)) & UInt32(1)) == UInt32(1) && continue
        # Edge vector: v's position minus its i-th neighbour's position
        nbr = Int(poly.pnbrs[i, v])
        for j in 1:D
            ltd[j, d + 1] = poly.positions[j, v] - poly.positions[j, nbr]
        end
        # Gram-Schmidt against prior orthonormal vectors ltd[:, 1..d]
        for dd in 1:d
            dotp = zero(T)
            for j in 1:D
                dotp += ltd[j, d + 1] * ltd[j, dd]
            end
            for j in 1:D
                ltd[j, d + 1] -= dotp * ltd[j, dd]
            end
        end
        # Normalize
        len = zero(T)
        for j in 1:D
            len += ltd[j, d + 1] * ltd[j, d + 1]
        end
        len = sqrt(len)
        # If the edge is degenerate (collinear with prior), skip this
        # branch — its contribution is zero. The C code divides anyway
        # and produces NaN; we guard against that here.
        len <= eps(T) && continue
        for j in 1:D
            ltd[j, d + 1] /= len
        end
        # Recurse into the next depth. Backtracking is implicit — the
        # caller's UInt32 is unchanged; we OR-in slot `i` only for the
        # callee's frame.
        s += _reduce_helper_nd(poly, v, d + 1,
                                processed | (UInt32(1) << (i - 1)),
                                ltd, Val(D))
    end
    return s
end

end # module Flat
