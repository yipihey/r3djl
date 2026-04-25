"""
    R3D.Types

Core type definitions for the polytope representation.

The central type is `Polytope{D,T,S}`:

- `D :: Int` — dimensionality (2, 3, 4, ...).
- `T <: Real` — coordinate type (`Float64`, `Float32`, `Dual`, ...).
- `S <: AbstractStorage` — vertex storage strategy (static or dynamic).

Each `Vertex{D,T}` carries its position and a list of `D` neighbour indices
(the doubly-linked edge graph). For `D > 3` we also carry a `D×D` 2-face
connectivity table, matching r3d's `rNd_vertex.finds`.
"""

# ---------------------------------------------------------------------------
# Vectors and planes
# ---------------------------------------------------------------------------

using StaticArrays: SVector, MVector

"""
    Vec{D,T}

A `D`-dimensional position vector. Alias for `SVector{D,T}`.

In r3d this is `r3d_rvec3` / `rNd_rvec` (a union with named fields). We use
`SVector` because it is stack-allocated, supports broadcasting, plays well
with AD, and the `.x/.y/.z` accessors aren't needed in idiomatic Julia.
"""
const Vec{D,T} = SVector{D,T}

"""
    Plane{D,T}

A hyperplane represented as `n ⋅ x + d = 0` where `n` is a unit normal and
`d` is the signed offset from the origin.

A point `x` is on the **positive** side iff `n ⋅ x + d ≥ 0`. r3d clips out
the *negative* side, so `clip!` retains the half-space `{x : n⋅x + d ≥ 0}`.
"""
struct Plane{D,T}
    n::Vec{D,T}
    d::T
end

# Convenience constructor for non-canonical inputs (e.g. plain Vector and
# Real). Uses a different name to avoid Julia's dispatch quirks around
# inner-vs-outer constructors with overlapping signatures.
"""
    plane(D, T, n, d) -> Plane{D,T}

Construct a `Plane{D,T}` from any `AbstractVector` `n` and `Real` `d`,
with explicit element-type conversion. Use this when your inputs aren't
already in canonical (`Vec{D,T}`, `T`) form.

For canonical inputs, just use `Plane{D,T}(n, d)` directly.
"""
function plane(::Val{D}, ::Type{T}, n::AbstractVector, d::Real) where {D,T}
    Plane{D,T}(Vec{D,T}(n), convert(T, d))
end

"""
    signed_distance(p::Plane, x::Vec)

Signed distance from `x` to the plane. Positive on the inside, negative on
the side that `clip!` removes. Mirrors `sdists[v] = planes[p].d + dot(...)`
in the C source.
"""
@inline signed_distance(p::Plane, x::Vec) = p.d + p.n' * x

# ---------------------------------------------------------------------------
# Vertex
# ---------------------------------------------------------------------------

"""
    Vertex{D,T}

A doubly-linked vertex. Stores the position and the indices of its `D`
neighbours in the host polytope.

# Field layout
- `pos::Vec{D,T}` — position.
- `pnbrs::MVector{D,Int32}` — neighbour vertex indices. Mutability lets
  the clip kernel rewrite connectivity in-place, matching the C code's
  `vertbuffer[v].pnbrs[np] = ...`.

We keep `pnbrs` indices as `Int32` rather than `Int` to match the C
layout and keep the struct compact. Cache-line size matters for the clip
inner loop.

# Note on rNd's `finds` table

`rNd_vertex` in the C source carries an additional `finds[D][D]` 2-face
connectivity table needed to walk face boundaries when `D ≥ 4`. For the
initial port we omit it — `D ≤ 3` doesn't use it, and the `D ≥ 4` clip
kernel is stubbed pending a faithful port of `rNd_clip`. When that work
lands, this struct will gain an optional `finds` field gated on `D`.
"""
mutable struct Vertex{D,T}
    pos::Vec{D,T}
    pnbrs::MVector{D,Int32}

    # Explicit inner constructor with concrete-typed arguments. This is
    # what the outer convenience constructor ultimately dispatches to,
    # and it cannot recurse back to the outer because the dispatch
    # signatures differ.
    Vertex{D,T}(pos::Vec{D,T}, pnbrs::MVector{D,Int32}) where {D,T} =
        new{D,T}(pos, pnbrs)
end

# Outer convenience constructor: accepts any AbstractVector and converts.
# Distinct dispatch from the inner constructor above.
function Vertex{D,T}(pos::AbstractVector, pnbrs::AbstractVector) where {D,T}
    Vertex{D,T}(Vec{D,T}(pos), MVector{D,Int32}(pnbrs))
end

# ---------------------------------------------------------------------------
# Storage traits
# ---------------------------------------------------------------------------

"""
    AbstractStorage

Vertex storage strategy for `Polytope`. Two concrete subtypes:

- `StaticStorage{N}` — fixed `N`-vertex buffer (matches r3d's
  `R3D_MAX_VERTS`). Zero allocation; clip operations that exceed `N` fail.
- `DynamicStorage` — `Vector{Vertex}` (matches PolyClipper). Allocates;
  no upper bound on complexity.

The storage is a type parameter, so dispatch is at compile time and the
two strategies generate completely separate code paths.
"""
abstract type AbstractStorage end

"""
    StaticStorage{N}

Statically-sized vertex buffer of capacity `N`. The Polytope holds an
`MVector{N,Vertex}` and tracks an `nverts` count. Matches r3d's
`R3D_MAX_VERTS` semantics; on overflow, `clip!` returns `false`.
"""
struct StaticStorage{N} <: AbstractStorage end

"""
    DynamicStorage

Dynamically-sized vertex buffer (`Vector{Vertex}`). Grows as needed during
clipping. Matches PolyClipper's storage strategy.
"""
struct DynamicStorage <: AbstractStorage end

# ---------------------------------------------------------------------------
# Polytope
# ---------------------------------------------------------------------------

"""
    Polytope{D,T,S}

A `D`-dimensional polytope with real type `T` and storage strategy `S`.

Internal representation is the same as r3d: a vertex buffer plus a
neighbour list per vertex (the dual graph of the edge structure).

For `S = StaticStorage{N}`, the buffer is preallocated to capacity `N`.
For `S = DynamicStorage`, the buffer is a growable `Vector`.
"""
mutable struct Polytope{D,T,S<:AbstractStorage}
    verts::Vector{Vertex{D,T}}   # Always a Vector internally; the storage
                                  # trait controls capacity / growth policy
                                  # rather than the container type.
    nverts::Int
    capacity::Int
end

# --- Static storage constructor ---
function Polytope{D,T,StaticStorage{N}}() where {D,T,N}
    # Preallocate exactly N undef vertices. We initialize positions to zero
    # rather than leaving truly undef so accidental reads don't segfault.
    verts = [Vertex{D,T}(zeros(Vec{D,T}), zeros(MVector{D,Int32})) for _ in 1:N]
    Polytope{D,T,StaticStorage{N}}(verts, 0, N)
end

# --- Dynamic storage constructor ---
function Polytope{D,T,DynamicStorage}(; sizehint::Int = 16) where {D,T}
    verts = Vertex{D,T}[]
    sizehint!(verts, sizehint)
    Polytope{D,T,DynamicStorage}(verts, 0, typemax(Int))
end

# Default: 3D Float64 with r3d's default cap of 512.
const DefaultPolytope = Polytope{3,Float64,StaticStorage{512}}

"""
    capacity(p::Polytope)

Return the maximum number of vertices `p` can hold without growing
(infinite for `DynamicStorage`).
"""
capacity(p::Polytope) = p.capacity

"""
    has_static_capacity(::Type{S})

Trait helper: `true` if `S` has a fixed capacity (the clip kernel checks
this to decide whether overflow should fail or grow the buffer).
"""
has_static_capacity(::Type{<:StaticStorage}) = true
has_static_capacity(::Type{DynamicStorage}) = false
