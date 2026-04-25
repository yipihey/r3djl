"""
    R3D.Moments

Analytic integration of polynomial densities over polytopes. Implements
the recursive method of Koehl (2012), as in `r3d_reduce`.

For a polytope `P` and polynomial order `n`, computes the moments

    M[i,j,k] = ∫_P x^i y^j z^k dV

for all `(i, j, k)` with `i + j + k ≤ n`. The total number of such
moments is `(n+1)(n+2)(n+3)/6`, exposed as `num_moments(D, n)`.

# Algorithm

Decompose the polyhedron into a fan of tetrahedra rooted at one vertex
of each face, then sum tetrahedral moment integrals using Koehl's
trinomial pyramid recursion. The recursion runs in two layers (`prev`,
`cur`) so total memory is `O(n²)` rather than `O(n³)`.

The face-walk uses the same `(np + 1) % D` convention as `clip!`, with
an `emarks` table to avoid revisiting edges.

This file ports `r3d_reduce` from `src/r3d.c` lines 253–402. The
correspondence is line-for-line; comments preserve the original
references.
"""

"""
    num_moments(D, order)

Number of monomials of total degree `≤ order` in `D` variables.

For 3D this is `(n+1)(n+2)(n+3)/6` (matches `R3D_NUM_MOMENTS` macro).
For 2D, `(n+1)(n+2)/2`. Generic formula: binomial(D + order, D).
"""
num_moments(D::Integer, order::Integer) = binomial(D + order, D)

"""
    moments(poly::Polytope{D,T}, order::Integer) -> Vector{T}

Allocating wrapper around `moments!`. Returns a freshly allocated vector
of length `num_moments(D, order)`.
"""
function moments(poly::Polytope{D,T}, order::Integer) where {D,T}
    out = zeros(T, num_moments(D, order))
    moments!(out, poly, order)
    return out
end

"""
    moments!(out, poly::Polytope{3,T}, order::Integer)

In-place 3D moment integration. `out` must have length
`num_moments(3, order) = (n+1)(n+2)(n+3)/6`.

Moments are returned in the same row-major order as r3d:
`1, x, y, z, x², xy, xz, y², yz, z², x³, ...`.

# Note on shift-poly

Upstream r3d has a `SHIFT_POLY` compile flag that translates the
polytope to its centroid before integrating, then shifts the moments
back at the end. This improves accuracy for high orders on polytopes
far from the origin, at higher computational cost. We expose this via a
keyword argument rather than a compile flag.
"""
function moments!(
    out::AbstractVector{T},
    poly::Polytope{3,T},
    order::Integer;
    shift::Bool = false,
) where {T}
    @assert length(out) >= num_moments(3, order)
    fill!(out, zero(T))
    poly.nverts <= 0 && return out

    nv = poly.nverts
    vc = shift ? poly_center(poly) : zero(Vec{3,T})

    # Edge-mark table to avoid revisiting faces. emarks[v, np] = true means
    # the edge starting at vertex v through neighbour slot np has already
    # been consumed by a face triangulation.
    emarks = zeros(Bool, nv, 3)

    # Two-layer trinomial pyramid storage. Indexed as S[i+1, j+1, layer]
    # because Julia is 1-based; the C version is 0-based.
    np1 = order + 1
    S = zeros(T, np1, np1, 2)
    D = zeros(T, np1, np1, 2)
    C = zeros(T, np1, np1, 2)
    prevlayer = 1
    curlayer = 2

    @inbounds for vstart in 1:nv, pstart in 1:3
        emarks[vstart, pstart] && continue

        # Initialize face walk from this edge
        pnext = pstart
        vcur = vstart
        emarks[vcur, pnext] = true
        vnext = Int(poly.verts[vcur].pnbrs[pnext])
        v0 = poly.verts[vcur].pos - vc

        # Step to second edge of the face
        np = 0
        for k in 1:3
            if poly.verts[vnext].pnbrs[k] == vcur
                np = k
                break
            end
        end
        vcur = vnext
        pnext = mod1(np + 1, 3)
        emarks[vcur, pnext] = true
        vnext = Int(poly.verts[vcur].pnbrs[pnext])

        # Triangle fan: each iteration adds (v0, v1, v2) tetrahedron's
        # contribution to the moments
        while vnext != vstart
            v2 = poly.verts[vcur].pos - vc
            v1 = poly.verts[vnext].pos - vc

            # 6 × signed tetrahedral volume (v0, v1, v2, origin)
            sixv = (-v2[1]*v1[2]*v0[3] + v1[1]*v2[2]*v0[3]
                    + v2[1]*v0[2]*v1[3] - v0[1]*v2[2]*v1[3]
                    - v1[1]*v0[2]*v2[3] + v0[1]*v1[2]*v2[3])

            # ---- Koehl trinomial pyramid recursion ----
            S[1, 1, prevlayer] = one(T)
            Dd = D; Cc = C; Ss = S  # short aliases for the inner loop
            Dd[1, 1, prevlayer] = one(T)
            Cc[1, 1, prevlayer] = one(T)
            out[1] += sixv / 6

            m = 1
            for corder in 1:order
                for i in corder:-1:0
                    for j in (corder - i):-1:0
                        k = corder - i - j
                        m += 1
                        ci = i + 1; cj = j + 1   # 1-based indices
                        Cv = zero(T); Dv = zero(T); Sv = zero(T)
                        if i > 0
                            Cv += v2[1] * Cc[ci-1, cj, prevlayer]
                            Dv += v1[1] * Dd[ci-1, cj, prevlayer]
                            Sv += v0[1] * Ss[ci-1, cj, prevlayer]
                        end
                        if j > 0
                            Cv += v2[2] * Cc[ci, cj-1, prevlayer]
                            Dv += v1[2] * Dd[ci, cj-1, prevlayer]
                            Sv += v0[2] * Ss[ci, cj-1, prevlayer]
                        end
                        if k > 0
                            Cv += v2[3] * Cc[ci, cj, prevlayer]
                            Dv += v1[3] * Dd[ci, cj, prevlayer]
                            Sv += v0[3] * Ss[ci, cj, prevlayer]
                        end
                        Dv += Cv
                        Sv += Dv
                        Cc[ci, cj, curlayer] = Cv
                        Dd[ci, cj, curlayer] = Dv
                        Ss[ci, cj, curlayer] = Sv
                        out[m] += sixv * Sv
                    end
                end
                curlayer  = 3 - curlayer
                prevlayer = 3 - prevlayer
            end

            # Advance the face walk by one edge
            np = 0
            for k in 1:3
                if poly.verts[vnext].pnbrs[k] == vcur
                    np = k
                    break
                end
            end
            vcur = vnext
            pnext = mod1(np + 1, 3)
            emarks[vcur, pnext] = true
            vnext = Int(poly.verts[vcur].pnbrs[pnext])
        end
    end

    # Final pass: divide each moment by the multinomial coefficient
    # times (corder+1)(corder+2)(corder+3), reusing C as scratch.
    @inbounds C[1, 1, prevlayer] = one(T)
    m = 1
    for corder in 1:order
        for i in corder:-1:0
            for j in (corder - i):-1:0
                k = corder - i - j
                m += 1
                ci = i + 1; cj = j + 1
                Cv = zero(T)
                if i > 0; Cv += C[ci-1, cj, prevlayer]; end
                if j > 0; Cv += C[ci, cj-1, prevlayer]; end
                if k > 0; Cv += C[ci, cj, prevlayer]; end
                C[ci, cj, curlayer] = Cv
                out[m] /= Cv * (corder + 1) * (corder + 2) * (corder + 3)
            end
        end
        curlayer  = 3 - curlayer
        prevlayer = 3 - prevlayer
    end

    if shift
        shift_moments!(out, order, vc)
    end
    return out
end

"""
    poly_center(poly::Polytope{D,T}) -> Vec{D,T}

Mean of the vertex positions. Used as the shift origin for SHIFT_POLY.
Mirrors `r3d_poly_center`.
"""
function poly_center(poly::Polytope{D,T}) where {D,T}
    nv = poly.nverts
    nv == 0 && return zero(Vec{D,T})
    s = zero(Vec{D,T})
    @inbounds for i in 1:nv
        s += poly.verts[i].pos
    end
    return s / nv
end

"""
    shift_moments!(moments, order, vc)

Adjust the moments of a polytope that was integrated in shifted-to-origin
coordinates back to the original frame. Mirrors `r3d_shift_moments`.

NOT YET IMPLEMENTED — for moment-shift, see `r3d_shift_moments` in
`src/r3d.c` lines 404+. The full version is a few dozen lines of binomial
expansion bookkeeping.
"""
function shift_moments!(moments::AbstractVector, order::Integer, vc::Vec)
    error("shift_moments!: TODO — port from r3d_shift_moments in r3d.c")
end
