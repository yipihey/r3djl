"""
    R3D_C

Thin `ccall` wrapper around the upstream r3d C library. Used as a ground-
truth oracle for differential testing of `R3D.jl` and as a baseline for
performance comparison.

Two ways to point this at a libr3d build:

1. Set `ENV["R3D_LIB"] = "/path/to/libr3d.so"` before `using R3D_C`.
2. Use the `r3d_jll` artifact (when published).

The C structs are mirrored exactly — same field order, same primitive
types — so we can pass `Ref(jl_poly)` through `ccall` directly. The
critical compile-time constant is `R3D_MAX_VERTS`, which controls the
size of `r3d_poly`. We default to 512 (the upstream default); override
via `ENV["R3D_MAX_VERTS"]` if your local build differs.
"""
module R3D_C

using Libdl

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

const R3D_MAX_VERTS = parse(Int, get(ENV, "R3D_MAX_VERTS", "512"))

const libr3d = Ref{String}("")

function __init__()
    libpath = get(ENV, "R3D_LIB", "")
    if isempty(libpath)
        @warn "R3D_C: ENV[\"R3D_LIB\"] not set; ccalls will fail until you " *
              "set it to the path of libr3d.{so,dylib,dll}."
        return
    end
    if !isfile(libpath)
        error("R3D_C: ENV[\"R3D_LIB\"] = $libpath does not exist")
    end
    libr3d[] = libpath
    try
        Libdl.dlopen(libpath)
    catch e
        @error "R3D_C: failed to dlopen $libpath" exception=e
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# Mirrored C structs (must match src/r3d.h exactly)
# ---------------------------------------------------------------------------

struct RVec3
    x::Float64
    y::Float64
    z::Float64
end
RVec3(v::NTuple{3,<:Real}) = RVec3(v[1], v[2], v[3])
RVec3(x::Real, y::Real, z::Real) = RVec3(Float64(x), Float64(y), Float64(z))

struct DVec3
    i::Int32
    j::Int32
    k::Int32
end

struct Plane
    n::RVec3
    d::Float64
end

# Field layout for r3d_vertex:
#   r3d_int pnbrs[3]   = 12 bytes
#   r3d_rvec3 pos      = 24 bytes (8-byte aligned)
# Padding makes total size 40 bytes.
struct Vertex
    pnbrs::NTuple{3,Int32}
    pos::RVec3
end

struct Poly{N}
    verts::NTuple{N,Vertex}
    nverts::Int32
end

const PolyDefault = Poly{R3D_MAX_VERTS}

"""
    new_poly() -> (ptr, buf)

Allocate a zero-initialized C poly buffer. Returns `(ptr, buf)` where
`ptr` is a `Ptr{Poly{N}}` aliasing into a GC-rooted `Vector{UInt8}`. The
caller MUST keep `buf` alive while using `ptr`.
"""
function new_poly()
    buf = zeros(UInt8, sizeof(Poly{R3D_MAX_VERTS}))
    ptr = Base.unsafe_convert(Ptr{Poly{R3D_MAX_VERTS}}, pointer(buf))
    return ptr, buf
end

# ---------------------------------------------------------------------------
# ccall wrappers — runtime-resolved library path
# ---------------------------------------------------------------------------

function init_box!(poly_ptr::Ptr{Poly{N}}, lo::RVec3, hi::RVec3) where {N}
    bounds = Ref((lo, hi))
    GC.@preserve bounds begin
        ccall((:r3d_init_box, libr3d[]), Cvoid,
              (Ptr{Poly{N}}, Ptr{NTuple{2,RVec3}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,RVec3}}, bounds))
    end
    return poly_ptr
end

function init_tet!(poly_ptr::Ptr{Poly{N}},
                   v1::RVec3, v2::RVec3, v3::RVec3, v4::RVec3) where {N}
    verts = Ref((v1, v2, v3, v4))
    GC.@preserve verts begin
        ccall((:r3d_init_tet, libr3d[]), Cvoid,
              (Ptr{Poly{N}}, Ptr{NTuple{4,RVec3}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{4,RVec3}}, verts))
    end
    return poly_ptr
end

function clip!(poly_ptr::Ptr{Poly{N}}, planes::Vector{Plane}) where {N}
    nplanes = Int32(length(planes))
    return ccall((:r3d_clip, libr3d[]), Cint,
                 (Ptr{Poly{N}}, Ptr{Plane}, Int32),
                 poly_ptr, planes, nplanes)
end

function reduce!(poly_ptr::Ptr{Poly{N}},
                 moments::Vector{Float64},
                 polyorder::Integer) where {N}
    ccall((:r3d_reduce, libr3d[]), Cvoid,
          (Ptr{Poly{N}}, Ptr{Float64}, Int32),
          poly_ptr, moments, Int32(polyorder))
    return moments
end

function is_good(poly_ptr::Ptr{Poly{N}}) where {N}
    rv = ccall((:r3d_is_good, libr3d[]), Int32, (Ptr{Poly{N}},), poly_ptr)
    return rv != 0
end

"""
    split!(in_ptr, plane, out_pos_ptr, out_neg_ptr) -> Bool

Single-polytope wrapper around `r3d_split` (which is variadic over an
array of inputs; we always pass `npolys = 1`).
"""
function split!(in_ptr::Ptr{Poly{N}}, plane::Plane,
                out_pos_ptr::Ptr{Poly{N}}, out_neg_ptr::Ptr{Poly{N}}) where {N}
    rv = ccall((:r3d_split, libr3d[]), Cint,
               (Ptr{Poly{N}}, Int32, Plane, Ptr{Poly{N}}, Ptr{Poly{N}}),
               in_ptr, Int32(1), plane, out_pos_ptr, out_neg_ptr)
    return rv != 0
end

function rotate!(poly_ptr::Ptr{Poly{N}}, theta::Float64, axis::Integer) where {N}
    # axis is 1-based here; the C side wants 0-based.
    ccall((:r3d_rotate, libr3d[]), Cvoid,
          (Ptr{Poly{N}}, Float64, Int32),
          poly_ptr, theta, Int32(axis - 1))
    return poly_ptr
end

function translate!(poly_ptr::Ptr{Poly{N}}, shift::RVec3) where {N}
    ccall((:r3d_translate, libr3d[]), Cvoid,
          (Ptr{Poly{N}}, RVec3),
          poly_ptr, shift)
    return poly_ptr
end

function scale!(poly_ptr::Ptr{Poly{N}}, s::Float64) where {N}
    ccall((:r3d_scale, libr3d[]), Cvoid,
          (Ptr{Poly{N}}, Float64),
          poly_ptr, s)
    return poly_ptr
end

function affine!(poly_ptr::Ptr{Poly{N}}, mat::AbstractMatrix{Float64}) where {N}
    @assert size(mat) == (4, 4)
    # r3d expects a row-major 4x4. Julia matrices are column-major, so
    # we transpose into a temporary buffer the C code can read.
    rowmajor = Matrix{Float64}(undef, 4, 4)
    @inbounds for i in 1:4, j in 1:4
        rowmajor[j, i] = mat[i, j]   # Julia(j, i) memory == C(i, j)
    end
    ccall((:r3d_affine, libr3d[]), Cvoid,
          (Ptr{Poly{N}}, Ptr{Float64}),
          poly_ptr, rowmajor)
    return poly_ptr
end

function shift_moments!(moments::Vector{Float64}, polyorder::Integer, vc::RVec3)
    ccall((:r3d_shift_moments, libr3d[]), Cvoid,
          (Ptr{Float64}, Int32, RVec3),
          moments, Int32(polyorder), vc)
    return moments
end

# ---------------------------------------------------------------------------
# v3d.h — voxelization
# ---------------------------------------------------------------------------

"""
    get_ibox(poly_ptr, d) -> ((lo_i, lo_j, lo_k), (hi_i, hi_j, hi_k))

Mirror of `r3d_get_ibox`. `d` is the per-axis cell spacing as
`(dx, dy, dz)`.
"""
function get_ibox(poly_ptr::Ptr{Poly{N}}, d::NTuple{3,Float64}) where {N}
    ibox = Ref((DVec3(0,0,0), DVec3(0,0,0)))
    drv = RVec3(d[1], d[2], d[3])
    GC.@preserve ibox begin
        ccall((:r3d_get_ibox, libr3d[]), Cvoid,
              (Ptr{Poly{N}}, Ptr{NTuple{2,DVec3}}, RVec3),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,DVec3}}, ibox), drv)
    end
    lo, hi = ibox[]
    return ((Int(lo.i), Int(lo.j), Int(lo.k)),
            (Int(hi.i), Int(hi.j), Int(hi.k)))
end

"""
    voxelize!(dest_grid, poly_ptr, ibox_lo, ibox_hi, d, polyorder) -> dest_grid

Mirror of `r3d_voxelize`. `dest_grid` is a flat `Vector{Float64}` of
length `(hi[1]-lo[1]) * (hi[2]-lo[2]) * (hi[3]-lo[3]) * num_moments`,
laid out row-major as the C library expects:
`dest_grid[(i*nj*nk + j*nk + k)*nmom + m + 1]` (1-based) for moment
`m ∈ 0:nmom-1` of voxel `(i,j,k) ∈ 0:ni-1 × …`.
"""
function voxelize!(dest_grid::Vector{Float64},
                   poly_ptr::Ptr{Poly{N}},
                   ibox_lo::NTuple{3,Int}, ibox_hi::NTuple{3,Int},
                   d::NTuple{3,Float64},
                   polyorder::Integer) where {N}
    ibox = Ref((DVec3(Int32(ibox_lo[1]), Int32(ibox_lo[2]), Int32(ibox_lo[3])),
                DVec3(Int32(ibox_hi[1]), Int32(ibox_hi[2]), Int32(ibox_hi[3]))))
    drv = RVec3(d[1], d[2], d[3])
    GC.@preserve ibox begin
        ccall((:r3d_voxelize, libr3d[]), Cvoid,
              (Ptr{Poly{N}}, Ptr{NTuple{2,DVec3}}, Ptr{Float64}, RVec3, Int32),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,DVec3}}, ibox),
              dest_grid, drv, Int32(polyorder))
    end
    return dest_grid
end

# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

function nverts(poly_ptr::Ptr{Poly{N}}) where {N}
    offset = fieldoffset(Poly{N}, 2)
    return unsafe_load(reinterpret(Ptr{Int32}, poly_ptr + offset))
end

function vertex_positions(poly_ptr::Ptr{Poly{N}}) where {N}
    n = nverts(poly_ptr)
    out = Vector{NTuple{3,Float64}}(undef, n)
    vsize = sizeof(Vertex)
    pos_off = fieldoffset(Vertex, 2)
    base = reinterpret(Ptr{UInt8}, poly_ptr)
    @inbounds for i in 1:n
        ptr = reinterpret(Ptr{Float64}, base + (i-1) * vsize + pos_off)
        out[i] = (unsafe_load(ptr, 1),
                  unsafe_load(ptr, 2),
                  unsafe_load(ptr, 3))
    end
    return out
end

# ===========================================================================
# 2D — r2d.h / v2d.h wrappers
# ===========================================================================

const R2D_MAX_VERTS = parse(Int, get(ENV, "R2D_MAX_VERTS", "256"))

struct RVec2
    x::Float64
    y::Float64
end
RVec2(v::NTuple{2,<:Real}) = RVec2(v[1], v[2])
RVec2(x::Real, y::Real) = RVec2(Float64(x), Float64(y))

struct DVec2
    i::Int32
    j::Int32
end

struct Plane2
    n::RVec2
    d::Float64
end

# r2d_vertex layout: pnbrs[2] = 8 bytes, pos = 16 bytes, total 24 bytes.
struct Vertex2
    pnbrs::NTuple{2,Int32}
    pos::RVec2
end

struct Poly2{N}
    verts::NTuple{N,Vertex2}
    nverts::Int32
end

const Poly2Default = Poly2{R2D_MAX_VERTS}

"""
    new_poly2() -> (ptr, buf)

Allocate a zero-initialized 2D `r2d_poly` buffer.
"""
function new_poly2()
    buf = zeros(UInt8, sizeof(Poly2{R2D_MAX_VERTS}))
    ptr = Base.unsafe_convert(Ptr{Poly2{R2D_MAX_VERTS}}, pointer(buf))
    return ptr, buf
end

function init_box2!(poly_ptr::Ptr{Poly2{N}}, lo::RVec2, hi::RVec2) where {N}
    bounds = Ref((lo, hi))
    GC.@preserve bounds begin
        ccall((:r2d_init_box, libr3d[]), Cvoid,
              (Ptr{Poly2{N}}, Ptr{NTuple{2,RVec2}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,RVec2}}, bounds))
    end
    return poly_ptr
end

function clip2!(poly_ptr::Ptr{Poly2{N}}, planes::Vector{Plane2}) where {N}
    nplanes = Int32(length(planes))
    ccall((:r2d_clip, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Ptr{Plane2}, Int32),
          poly_ptr, planes, nplanes)
    return poly_ptr
end

function reduce2!(poly_ptr::Ptr{Poly2{N}}, moments::Vector{Float64},
                  polyorder::Integer) where {N}
    ccall((:r2d_reduce, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Ptr{Float64}, Int32),
          poly_ptr, moments, Int32(polyorder))
    return moments
end

function get_ibox2(poly_ptr::Ptr{Poly2{N}}, d::NTuple{2,Float64}) where {N}
    ibox = Ref((DVec2(0,0), DVec2(0,0)))
    drv = RVec2(d[1], d[2])
    GC.@preserve ibox begin
        ccall((:r2d_get_ibox, libr3d[]), Cvoid,
              (Ptr{Poly2{N}}, Ptr{NTuple{2,DVec2}}, RVec2),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,DVec2}}, ibox), drv)
    end
    lo, hi = ibox[]
    return ((Int(lo.i), Int(lo.j)), (Int(hi.i), Int(hi.j)))
end

"""
    rasterize2!(dest_grid, poly_ptr, ibox_lo, ibox_hi, d, polyorder) -> dest_grid

Mirror of `r2d_rasterize`. `dest_grid` is a flat row-major
`Vector{Float64}` of length `(hi[1]-lo[1]) * (hi[2]-lo[2]) * num_moments`.
"""
function is_good2(poly_ptr::Ptr{Poly2{N}}) where {N}
    rv = ccall((:r2d_is_good, libr3d[]), Int32, (Ptr{Poly2{N}},), poly_ptr)
    return rv != 0
end

function split2!(in_ptr::Ptr{Poly2{N}}, plane::Plane2,
                 out_pos_ptr::Ptr{Poly2{N}}, out_neg_ptr::Ptr{Poly2{N}}) where {N}
    ccall((:r2d_split, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Int32, Plane2, Ptr{Poly2{N}}, Ptr{Poly2{N}}),
          in_ptr, Int32(1), plane, out_pos_ptr, out_neg_ptr)
    return true
end

function rotate2!(poly_ptr::Ptr{Poly2{N}}, theta::Float64) where {N}
    # Note: upstream r2d_rotate (in r2d.c) has a known bug — it sets
    # pos.x twice and never updates pos.y. We expose the C entry point
    # for completeness but do NOT use it as a diff oracle in tests.
    ccall((:r2d_rotate, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Float64),
          poly_ptr, theta)
    return poly_ptr
end

function translate2!(poly_ptr::Ptr{Poly2{N}}, shift::RVec2) where {N}
    ccall((:r2d_translate, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, RVec2),
          poly_ptr, shift)
    return poly_ptr
end

function scale2!(poly_ptr::Ptr{Poly2{N}}, s::Float64) where {N}
    ccall((:r2d_scale, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Float64),
          poly_ptr, s)
    return poly_ptr
end

function affine2!(poly_ptr::Ptr{Poly2{N}}, mat::AbstractMatrix{Float64}) where {N}
    @assert size(mat) == (3, 3)
    rowmajor = Matrix{Float64}(undef, 3, 3)
    @inbounds for i in 1:3, j in 1:3
        rowmajor[j, i] = mat[i, j]
    end
    ccall((:r2d_affine, libr3d[]), Cvoid,
          (Ptr{Poly2{N}}, Ptr{Float64}),
          poly_ptr, rowmajor)
    return poly_ptr
end

function shift_moments2!(moments::Vector{Float64}, polyorder::Integer, vc::RVec2)
    ccall((:r2d_shift_moments, libr3d[]), Cvoid,
          (Ptr{Float64}, Int32, RVec2),
          moments, Int32(polyorder), vc)
    return moments
end

function rasterize2!(dest_grid::Vector{Float64},
                     poly_ptr::Ptr{Poly2{N}},
                     ibox_lo::NTuple{2,Int}, ibox_hi::NTuple{2,Int},
                     d::NTuple{2,Float64},
                     polyorder::Integer) where {N}
    ibox = Ref((DVec2(Int32(ibox_lo[1]), Int32(ibox_lo[2])),
                DVec2(Int32(ibox_hi[1]), Int32(ibox_hi[2]))))
    drv = RVec2(d[1], d[2])
    GC.@preserve ibox begin
        ccall((:r2d_rasterize, libr3d[]), Cvoid,
              (Ptr{Poly2{N}}, Ptr{NTuple{2,DVec2}}, Ptr{Float64}, RVec2, Int32),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,DVec2}}, ibox),
              dest_grid, drv, Int32(polyorder))
    end
    return dest_grid
end

end # module R3D_C
