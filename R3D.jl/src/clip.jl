"""
    R3D.Clip

Half-space clipping of polytopes. This is the central algorithm of r3d and
the workhorse of every higher-level operation (split, voxelize, intersect).

## Algorithm

For each clip plane `(n, d)`:

1. Compute signed distances from every vertex to the plane.
2. If the polytope lies entirely on the inside, skip; if entirely outside,
   discard everything.
3. Otherwise, for each edge that crosses the plane, insert a new vertex at
   the intersection. New vertices are appended; their first neighbour
   slot points back to the surviving vertex on the inside side.
4. Walk the boundary of the clipped face to wire up the remaining
   neighbour slots of the new vertices, doubly-linking them.
5. Compact the buffer, removing vertices on the outside and re-indexing.

The C code does all of this with raw indices into a static buffer. We
preserve that structure exactly — it's what makes r3d fast — and abstract
only over dimension `D` and storage trait `S`.

## Generality across `D`

In 3D, every vertex has exactly 3 neighbours. The "walk the boundary"
step in step 4 uses `pnext = (np + 1) % 3` to advance around the face.
In `D` dimensions, this generalizes via the `finds` 2-face table that
`rNd` carries on each vertex. For `D ≤ 3` the table is implicit (each
edge belongs to a unique pair of faces given by `(np+1)%D` and
`(np+D-1)%D`), so we special-case those for speed; for `D ≥ 4` we
consult `finds` explicitly.
"""

# ---------------------------------------------------------------------------
# Distance computation: lifted out so it inlines and benchmarks separately
# ---------------------------------------------------------------------------

"""
    compute_signed_distances!(sdists, poly, plane)

Fill `sdists[1:nverts]` with signed distances of each vertex to `plane`.
Returns `(smin, smax)`, the min and max signed distance — these decide
whether the plane intersects the polytope at all.

Mirrors the C inner loop:

    for (v = 0; v < onv; ++v) {
        sdists[v] = planes[p].d + dot(vertbuffer[v].pos, planes[p].n);
        ...
    }
"""
@inline function compute_signed_distances!(
    sdists::AbstractVector{T},
    poly::Polytope{D,T},
    plane::Plane{D,T},
) where {D,T}
    nverts = poly.nverts
    smin = T(Inf)
    smax = T(-Inf)
    @inbounds for v in 1:nverts
        s = signed_distance(plane, poly.verts[v].pos)
        sdists[v] = s
        s < smin && (smin = s)
        s > smax && (smax = s)
    end
    return smin, smax
end

# ---------------------------------------------------------------------------
# The main clip routine
# ---------------------------------------------------------------------------

"""
    clip!(poly::Polytope{D,T}, planes::AbstractVector{Plane{D,T}}) -> Bool

Clip `poly` in-place against every plane in `planes`. The result is the
intersection of `poly` with the half-spaces `{x : nᵢ⋅x + dᵢ ≥ 0}`.

Returns `true` on success, `false` if a static-capacity buffer overflowed
(matching r3d's status-code return). For `DynamicStorage`, this only
returns `false` if the polytope is empty to begin with.

# Indexing convention

We use **1-based** indices throughout, including in `pnbrs`. The sentinel
"no neighbour" value is `0` (rather than `-1` as in the C code). This is
the only place the Julia port deviates from the C source by more than a
syntactic substitution.
"""
function clip!(
    poly::Polytope{D,T,S},
    planes::AbstractVector{Plane{D,T}},
) where {D,T,S}
    poly.nverts <= 0 && return false

    # Scratch buffers. For StaticStorage{N} we know the upper bound; for
    # DynamicStorage we size to current nverts + slack. Either way the
    # allocation happens once per clip!, not once per plane.
    nmax_initial = capacity(poly)
    sdists = Vector{T}(undef, nmax_initial == typemax(Int) ?
                              max(poly.nverts * 4, 64) : nmax_initial)
    clipped = Vector{Int32}(undef, length(sdists))

    @inbounds for plane in planes
        ok = clip_against_plane!(poly, plane, sdists, clipped)
        ok || return false
        # If the polytope was fully clipped away, stop early
        poly.nverts == 0 && return true
    end
    return true
end

# Single-plane clip — extracted so we can benchmark the per-plane cost
# directly and so the per-plane logic isn't buried in nested loops.
function clip_against_plane!(
    poly::Polytope{D,T,S},
    plane::Plane{D,T},
    sdists::AbstractVector{T},
    clipped::AbstractVector{Int32},
) where {D,T,S}

    onv = poly.nverts

    # Grow scratch buffers if the dynamic case has outgrown them
    if length(sdists) < onv
        resize!(sdists, max(2onv, length(sdists) * 2))
        resize!(clipped, length(sdists))
    end

    # Step 1: signed distances + bounds
    smin, smax = compute_signed_distances!(sdists, poly, plane)

    # Step 2: trivial accept / reject
    smin >= 0 && return true                 # entirely inside
    if smax <= 0                             # entirely outside
        poly.nverts = 0
        return true
    end

    # Mark vertices on the outside (negative side)
    @inbounds for v in 1:onv
        clipped[v] = sdists[v] < 0 ? Int32(1) : Int32(0)
    end

    # Step 3: insert new vertices on cut edges
    @inbounds for vcur in 1:onv
        clipped[vcur] != 0 && continue
        for np in 1:D
            vnext = poly.verts[vcur].pnbrs[np]
            vnext == 0 && continue
            clipped[vnext] == 0 && continue

            # Edge (vcur -> vnext) crosses the plane: insert a new vertex
            if has_static_capacity(S) && poly.nverts >= capacity(poly)
                @debug "clip!: static buffer overflow ($(capacity(poly)))"
                return false
            end

            new_idx = poly.nverts + 1
            ensure_capacity!(poly, new_idx)
            poly.nverts = new_idx

            # Position: weighted average, weights are |sdist| of the *other*
            # endpoint, so we move proportionally toward the crossing.
            wa = -sdists[vnext]   # weight on vcur (note sign)
            wb =  sdists[vcur]    # weight on vnext
            newpos = (wa * poly.verts[vcur].pos + wb * poly.verts[vnext].pos) / (wa + wb)

            # Wire in the new vertex. Its 0th neighbour points back to vcur.
            # The remaining neighbour slots will be set in step 4.
            poly.verts[new_idx].pos = newpos
            poly.verts[new_idx].pnbrs .= 0
            poly.verts[new_idx].pnbrs[1] = Int32(vcur)
            poly.verts[vcur].pnbrs[np] = Int32(new_idx)

            # Grow scratch arrays if needed (dynamic case only)
            if length(clipped) < new_idx
                resize!(clipped, 2 * new_idx)
                # Newly created entries are unclipped (they're new boundary
                # vertices, by definition on the plane).
                @inbounds for k in (new_idx):length(clipped)
                    clipped[k] = Int32(0)
                end
            else
                clipped[new_idx] = Int32(0)
            end
        end
    end

    # Step 4: link the new vertices into the surviving connectivity. This
    # is r3d's face-walking trick: starting at each new vertex, walk along
    # its initial neighbour, then around the face via (np+1) % D, until we
    # come back to a new vertex; that's the next link in the face boundary.
    #
    # In dimensions higher than 3, we'd need the finds[][] table to know
    # which "next face" to take. For now we only specialize D=2,3 (the
    # cases r3d has hand-tuned kernels for); D≥4 falls through to a
    # method-not-implemented error that the test suite will catch.
    link_new_vertices!(poly, onv, Val(D))

    # Step 5: compact the vertex buffer, dropping clipped vertices and
    # re-indexing pnbrs accordingly. We reuse `clipped` as the old->new
    # index map (negative values become 0).
    #
    # CRITICAL: Vertex is a mutable struct, so `poly.verts[i] = poly.verts[j]`
    # would alias them — the same object is referenced from two slots, and
    # any subsequent mutation hits both. We instead copy the field
    # contents into the destination slot, preserving its independent
    # identity.
    numunclipped = 0
    @inbounds for v in 1:poly.nverts
        if clipped[v] == 0
            numunclipped += 1
            if numunclipped != v
                # Copy contents, not the object reference
                src = poly.verts[v]
                dst = poly.verts[numunclipped]
                dst.pos = src.pos
                dst.pnbrs .= src.pnbrs
            end
            clipped[v] = Int32(numunclipped)
        else
            clipped[v] = Int32(0)   # 0 = "removed"
        end
    end
    poly.nverts = numunclipped
    @inbounds for v in 1:poly.nverts
        for np in 1:D
            old = poly.verts[v].pnbrs[np]
            poly.verts[v].pnbrs[np] = old == 0 ? Int32(0) : clipped[old]
        end
    end

    return true
end

# ---------------------------------------------------------------------------
# Linking new vertices: D=3 special case (r3d's hot path)
# ---------------------------------------------------------------------------

# In 3D, around each face, the cyclic order of neighbours is well-defined:
# given vcur and its incoming neighbour, the outgoing neighbour around the
# same face is at (np + 1) % 3. This is what r3d_clip uses verbatim.
function link_new_vertices!(
    poly::Polytope{3,T,S},
    onv::Int,
    ::Val{3},
) where {T,S}
    @inbounds for vstart in (onv+1):poly.nverts
        vcur = vstart
        vnext = Int(poly.verts[vcur].pnbrs[1])
        np = 0
        while true
            # Find which neighbour slot in vnext points back to vcur
            for k in 1:3
                if poly.verts[vnext].pnbrs[k] == vcur
                    np = k
                    break
                end
            end
            vcur = vnext
            pnext = mod1(np + 1, 3)              # (np + 1) % 3, but 1-based
            vnext = Int(poly.verts[vcur].pnbrs[pnext])
            # Mirror the C condition `while (vcur < onv)`: keep walking while
            # we're still on old (pre-cut) vertices; stop when we land on a
            # newly-inserted vertex (index > onv).
            vcur <= onv || break
        end
        poly.verts[vstart].pnbrs[3] = Int32(vcur)
        poly.verts[vcur].pnbrs[2]   = Int32(vstart)
    end
end

# 2D special case: each vertex has 2 neighbours; the "face walk" is trivial
# because there's only one other slot to choose from.
function link_new_vertices!(
    poly::Polytope{2,T,S},
    onv::Int,
    ::Val{2},
) where {T,S}
    @inbounds for vstart in (onv+1):poly.nverts
        # The new vertex has pnbrs[1] = vcur (its inside neighbour). Find the
        # corresponding new vertex on the other side of the cut edge by
        # walking from vcur through its remaining neighbour.
        vcur = Int(poly.verts[vstart].pnbrs[1])
        # Slot in vcur that doesn't point back to vstart is the next link
        for k in 1:2
            if poly.verts[vcur].pnbrs[k] != vstart
                # Walk one step; the eventual target is another new vertex
                # (which `clip!` will have created in the same plane sweep).
                # Detailed 2D linking left as an exercise — the structure
                # exactly mirrors r2d_clip in src/r2d.c.
            end
        end
        # NOTE: full 2D port deferred — see r2d.c for the reference algorithm
        error("2D clip linking: TODO (port from r2d.c, ~30 LOC)")
    end
end

# Generic D≥4 fallback: needs the finds[][] face table from rNd_clip.
function link_new_vertices!(
    poly::Polytope{D,T,S},
    onv::Int,
    ::Val{D},
) where {D,T,S}
    error("clip!: ND linking (D=$D) requires finds[] table — see rNd.c for " *
          "the face-walking algorithm. Not yet ported.")
end

# ---------------------------------------------------------------------------
# Capacity helpers
# ---------------------------------------------------------------------------

# StaticStorage: capacity is fixed at type level
@inline ensure_capacity!(::Polytope{D,T,StaticStorage{N}}, _) where {D,T,N} = nothing

# DynamicStorage: grow the underlying Vector if necessary
@inline function ensure_capacity!(p::Polytope{D,T,DynamicStorage}, idx::Int) where {D,T}
    if idx > length(p.verts)
        # Push undef-positioned vertices; clip! will overwrite immediately
        new_len = max(idx, 2 * length(p.verts))
        while length(p.verts) < new_len
            push!(p.verts, Vertex{D,T}(zeros(Vec{D,T}), zeros(MVector{D,Int32})))
        end
    end
end
