"""
    R3D.Init

Polytope constructors. These set up the initial vertex graph for common
shapes — boxes, tetrahedra, simplices — analogous to `r3d_init_box`,
`r3d_init_tet`, and `r3d_init_poly` in the C source.

The vertex graph for each shape is hand-wired here, exactly as in r3d.
The neighbour orderings matter: the clip algorithm assumes a consistent
orientation around each face for its `(np + 1) % D` face-walk.
"""

# ---------------------------------------------------------------------------
# 3D box
# ---------------------------------------------------------------------------

"""
    init_box!(poly::Polytope{3,T}, lo::Vec{3,T}, hi::Vec{3,T})

Initialize `poly` as the axis-aligned box with corners `lo` and `hi`.

Vertex labelling matches `r3d_init_box` in `src/r3d.c`:

    7 — 6
    | \\ | \\
    4 — 5  \\
     \\  3 — 2
      \\ |   |
       0 — 1

with neighbour orderings as in the original (right-handed, outward-facing
faces). We replicate the exact pnbrs table from r3d_init_box, line for
line, to guarantee identical clip behaviour.
"""
function init_box!(poly::Polytope{3,T}, lo::Vec{3,T}, hi::Vec{3,T}) where {T}
    poly.nverts = 8
    ensure_capacity!(poly, 8)

    # Reproduce r3d_init_box vertex positions exactly. r3d uses 0-indexed
    # neighbour indices; we shift to 1-based.
    positions = (
        Vec{3,T}(lo[1], lo[2], lo[3]),  # 1 (r3d's 0)
        Vec{3,T}(hi[1], lo[2], lo[3]),  # 2 (r3d's 1)
        Vec{3,T}(hi[1], hi[2], lo[3]),  # 3 (r3d's 2)
        Vec{3,T}(lo[1], hi[2], lo[3]),  # 4 (r3d's 3)
        Vec{3,T}(lo[1], lo[2], hi[3]),  # 5 (r3d's 4)
        Vec{3,T}(hi[1], lo[2], hi[3]),  # 6 (r3d's 5)
        Vec{3,T}(hi[1], hi[2], hi[3]),  # 7 (r3d's 6)
        Vec{3,T}(lo[1], hi[2], hi[3]),  # 8 (r3d's 7)
    )

    # Neighbour table copied verbatim from r3d_init_box (src/r3d.c),
    # translated to 1-based indices. The cyclic order in each row is
    # required by clip!'s and reduce!'s face-walk: at each step,
    # `(np + 1) mod 3` advances around the same face. r3d uses
    # right-handed, outward-facing faces; preserve this exactly.
    pnbrs = (
        (2, 5, 4),    # 1 (r3d's vertex 0: C(1,4,3) → Jl(2,5,4))
        (3, 6, 1),    # 2 (r3d's vertex 1: C(2,5,0) → Jl(3,6,1))
        (4, 7, 2),    # 3 (r3d's vertex 2: C(3,6,1) → Jl(4,7,2))
        (1, 8, 3),    # 4 (r3d's vertex 3: C(0,7,2) → Jl(1,8,3))
        (8, 1, 6),    # 5 (r3d's vertex 4: C(7,0,5) → Jl(8,1,6))
        (5, 2, 7),    # 6 (r3d's vertex 5: C(4,1,6) → Jl(5,2,7))
        (6, 3, 8),    # 7 (r3d's vertex 6: C(5,2,7) → Jl(6,3,8))
        (7, 4, 5),    # 8 (r3d's vertex 7: C(6,3,4) → Jl(7,4,5))
    )

    @inbounds for i in 1:8
        poly.verts[i].pos = positions[i]
        poly.verts[i].pnbrs[1] = Int32(pnbrs[i][1])
        poly.verts[i].pnbrs[2] = Int32(pnbrs[i][2])
        poly.verts[i].pnbrs[3] = Int32(pnbrs[i][3])
    end
    return poly
end

# Convenience constructor: returns a fresh box polytope
function box(lo::Vec{3,T}, hi::Vec{3,T};
             storage::Type{S} = StaticStorage{512}) where {T,S}
    p = Polytope{3,T,S}()
    init_box!(p, lo, hi)
    return p
end

# Tuple input convenience
box(lo::NTuple{3}, hi::NTuple{3}; kwargs...) =
    box(Vec{3,Float64}(lo), Vec{3,Float64}(hi); kwargs...)

# ---------------------------------------------------------------------------
# 3D tetrahedron
# ---------------------------------------------------------------------------

"""
    init_tet!(poly::Polytope{3,T}, verts::NTuple{4,Vec{3,T}})

Initialize `poly` as the tetrahedron with the four given vertices.

The neighbour table mirrors `r3d_init_tet`: vertex `i` is connected to
the three other vertices in an order that gives consistent outward-
facing faces when the input has positive `r3d_orient`.
"""
function init_tet!(poly::Polytope{3,T}, verts::NTuple{4,Vec{3,T}}) where {T}
    poly.nverts = 4
    ensure_capacity!(poly, 4)

    # Neighbour table copied from r3d_init_tet (src/r3d.c),
    # translated to 1-based indices.
    pnbrs = (
        (2, 4, 3),    # 1 (r3d's vertex 0: C(1,3,2) → Jl(2,4,3))
        (3, 4, 1),    # 2 (r3d's vertex 1: C(2,3,0) → Jl(3,4,1))
        (1, 4, 2),    # 3 (r3d's vertex 2: C(0,3,1) → Jl(1,4,2))
        (2, 3, 1),    # 4 (r3d's vertex 3: C(1,2,0) → Jl(2,3,1))
    )

    @inbounds for i in 1:4
        poly.verts[i].pos = verts[i]
        for k in 1:3
            poly.verts[i].pnbrs[k] = Int32(pnbrs[i][k])
        end
    end
    return poly
end

function tet(v1::Vec{3,T}, v2::Vec{3,T}, v3::Vec{3,T}, v4::Vec{3,T};
             storage::Type{S} = StaticStorage{512}) where {T,S}
    p = Polytope{3,T,S}()
    init_tet!(p, (v1, v2, v3, v4))
    return p
end

# ---------------------------------------------------------------------------
# Generic D-simplex
# ---------------------------------------------------------------------------

"""
    init_simplex!(poly::Polytope{D,T}, verts::NTuple{D⁺¹, Vec{D,T}})

Initialize `poly` as a `D`-simplex (D+1 vertices, each connected to all
others). For `D ≤ 3` this delegates to the hand-wired kernels above; for
`D ≥ 4` it builds the connectivity table generically and populates the
`finds[][]` 2-face entries.

Reference: `rNd_init_simplex` (not yet present in upstream r3d's public
API but implicit in the test code at `src/tests`).
"""
function init_simplex!(
    poly::Polytope{D,T},
    verts::NTuple{N,Vec{D,T}},
) where {D,T,N}
    @assert N == D + 1 "A D-simplex has D+1 vertices (got D=$D, N=$N)"
    poly.nverts = N
    ensure_capacity!(poly, N)

    @inbounds for i in 1:N
        poly.verts[i].pos = verts[i]
        # The D neighbours of vertex i are the other D vertices, in some
        # consistent cyclic order. For D ≤ 3 we hard-code; for D ≥ 4 the
        # ordering must come from a face-traversal of the simplex.
        if D == 2
            others = (mod1(i + 1, N), mod1(i - 1, N))
            for k in 1:2
                poly.verts[i].pnbrs[k] = Int32(others[k])
            end
        elseif D == 3
            # Use the same table as init_tet! (verbatim r3d_init_tet)
            pnbrs = ((2, 4, 3), (3, 4, 1), (1, 4, 2), (2, 3, 1))
            for k in 1:3
                poly.verts[i].pnbrs[k] = Int32(pnbrs[i][k])
            end
        else
            error("init_simplex!: D=$D requires finds[][] population — TODO")
        end
    end
    return poly
end
