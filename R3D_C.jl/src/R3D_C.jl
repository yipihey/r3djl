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
# Per-dimension rNd libraries (set via ENV["R3D_LIB_4D"] etc., dlopen'd
# lazily). Each one is a separate compilation of rNd.c with -DRND_DIM=N.
const libr3d_4d = Ref{String}("")
const libr3d_5d = Ref{String}("")
const libr3d_6d = Ref{String}("")

function __init__()
    # 1. Main r3d (D = 2 / D = 3) library: ENV["R3D_LIB"] → r3d_jll → warn.
    libpath = get(ENV, "R3D_LIB", "")
    main_loaded = false
    if !isempty(libpath)
        if !isfile(libpath)
            error("R3D_C: ENV[\"R3D_LIB\"] = $libpath does not exist")
        end
        libr3d[] = libpath
        main_loaded = true
    else
        try
            @eval Main using r3d_jll
            libr3d[] = Main.r3d_jll.libr3d
            main_loaded = true
        catch
            # JLL not available — fall through.
        end
    end

    if main_loaded
        try
            Libdl.dlopen(libr3d[])
        catch e
            @error "R3D_C: failed to dlopen $(libr3d[])" exception=e
            rethrow()
        end
    else
        @warn "R3D_C: ENV[\"R3D_LIB\"] not set and `r3d_jll` not available; " *
              "D=2 / D=3 ccalls will fail until you set R3D_LIB or `Pkg.add(\"r3d_jll\")`."
    end

    # 2. Per-dimension rNd libraries (optional, independent of the main
    # r3d library). Each is a separate compilation of rNd.c with
    # -DRND_DIM=N. When missing, the *_4d / *_5d / *_6d ccalls fail
    # with a clear pointer at the corresponding ENV var.
    for (D, ref, var) in ((4, libr3d_4d, "R3D_LIB_4D"),
                          (5, libr3d_5d, "R3D_LIB_5D"),
                          (6, libr3d_6d, "R3D_LIB_6D"))
        p = get(ENV, var, "")
        isempty(p) && continue
        if !isfile(p)
            @warn "R3D_C: ENV[\"$var\"] = $p does not exist; D=$D ccalls disabled"
            continue
        end
        ref[] = p
        try
            Libdl.dlopen(p)
        catch e
            @warn "R3D_C: failed to dlopen $p" exception=e
            ref[] = ""
        end
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

# ===========================================================================
# rNd (D ≥ 4) wrappers — one set per dimension. Each set targets a
# separately-built libr3d_${D}d.dylib (rNd.c compiled with -DRND_DIM=N),
# so the function names collide across dimensions but live in different
# libraries.
#
# Layout note for `Vertex{D}`: the C struct is
#   pnbrs[D]   (D × Int32)
#   finds[D][D] (D*D × Int32, row-major)
#   pos[D]      (D × Float64)
# Float64 alignment is 8 bytes. For D ∈ {4, 5, 6}, the int32 prefix
# (D + D²) * 4 bytes is already 8-byte aligned, so no padding before
# `pos`. Verify with `sizeof(Vertex4) == 112` etc.
# ===========================================================================

struct RVec4; xyz::NTuple{4, Float64}; end
struct Plane4; n::RVec4; d::Float64; end
struct Vertex4
    pnbrs::NTuple{4, Int32}
    finds::NTuple{16, Int32}
    pos::RVec4
end
struct Poly4{N}
    verts::NTuple{N, Vertex4}
    nverts::Int32
    nfaces::Int32
end
const RND_4D_MAX_VERTS = 1024
const Poly4Default = Poly4{RND_4D_MAX_VERTS}

struct RVec5; xyz::NTuple{5, Float64}; end
struct Plane5; n::RVec5; d::Float64; end
struct Vertex5
    pnbrs::NTuple{5, Int32}
    finds::NTuple{25, Int32}
    pos::RVec5
end
struct Poly5{N}
    verts::NTuple{N, Vertex5}
    nverts::Int32
    nfaces::Int32
end
const RND_5D_MAX_VERTS = 1024
const Poly5Default = Poly5{RND_5D_MAX_VERTS}

struct RVec6; xyz::NTuple{6, Float64}; end
struct Plane6; n::RVec6; d::Float64; end
struct Vertex6
    pnbrs::NTuple{6, Int32}
    finds::NTuple{36, Int32}
    pos::RVec6
end
struct Poly6{N}
    verts::NTuple{N, Vertex6}
    nverts::Int32
    nfaces::Int32
end
const RND_6D_MAX_VERTS = 1024
const Poly6Default = Poly6{RND_6D_MAX_VERTS}

# Buffer allocation helpers (mirror new_poly() / new_poly2()).
function new_poly4()
    buf = zeros(UInt8, sizeof(Poly4Default))
    return Base.unsafe_convert(Ptr{Poly4Default}, pointer(buf)), buf
end
function new_poly5()
    buf = zeros(UInt8, sizeof(Poly5Default))
    return Base.unsafe_convert(Ptr{Poly5Default}, pointer(buf)), buf
end
function new_poly6()
    buf = zeros(UInt8, sizeof(Poly6Default))
    return Base.unsafe_convert(Ptr{Poly6Default}, pointer(buf)), buf
end

function _check_lib(libref::Ref{String}, var::String, D::Int)
    isempty(libref[]) && error(
        "R3D_C: D=$D rNd library not loaded. Set ENV[\"$var\"] to a libr3d_$(D)d.{so,dylib} " *
        "built with `gcc -O3 -fPIC -shared -DRND_DIM=$D rNd.c -lm` before `using R3D_C`.")
    return libref[]
end

# --- D = 4 ccall wrappers --------------------------------------------------

function init_box4!(poly_ptr::Ptr{Poly4{N}}, lo::RVec4, hi::RVec4) where {N}
    bounds = Ref((lo, hi))
    GC.@preserve bounds begin
        ccall((:rNd_init_box, _check_lib(libr3d_4d, "R3D_LIB_4D", 4)),
              Cvoid, (Ptr{Poly4{N}}, Ptr{NTuple{2,RVec4}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,RVec4}}, bounds))
    end
    return poly_ptr
end

function init_simplex4!(poly_ptr::Ptr{Poly4{N}}, verts::NTuple{5,RVec4}) where {N}
    vertsref = Ref(verts)
    GC.@preserve vertsref begin
        ccall((:rNd_init_simplex, _check_lib(libr3d_4d, "R3D_LIB_4D", 4)),
              Cvoid, (Ptr{Poly4{N}}, Ptr{NTuple{5,RVec4}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{5,RVec4}}, vertsref))
    end
    return poly_ptr
end

function clip4!(poly_ptr::Ptr{Poly4{N}}, planes::Vector{Plane4}) where {N}
    nplanes = Int32(length(planes))
    ccall((:rNd_clip, _check_lib(libr3d_4d, "R3D_LIB_4D", 4)),
          Cvoid, (Ptr{Poly4{N}}, Ptr{Plane4}, Int32),
          poly_ptr, planes, nplanes)
    return poly_ptr
end

function reduce4!(poly_ptr::Ptr{Poly4{N}}, moments::Vector{Float64},
                  polyorder::Integer) where {N}
    ccall((:rNd_reduce, _check_lib(libr3d_4d, "R3D_LIB_4D", 4)),
          Cvoid, (Ptr{Poly4{N}}, Ptr{Float64}, Int32),
          poly_ptr, moments, Int32(polyorder))
    return moments
end

function nverts4(poly_ptr::Ptr{Poly4{N}}) where {N}
    offset = fieldoffset(Poly4{N}, 2)
    return unsafe_load(reinterpret(Ptr{Int32}, poly_ptr + offset))
end

# --- D = 5 ccall wrappers --------------------------------------------------

function init_box5!(poly_ptr::Ptr{Poly5{N}}, lo::RVec5, hi::RVec5) where {N}
    bounds = Ref((lo, hi))
    GC.@preserve bounds begin
        ccall((:rNd_init_box, _check_lib(libr3d_5d, "R3D_LIB_5D", 5)),
              Cvoid, (Ptr{Poly5{N}}, Ptr{NTuple{2,RVec5}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,RVec5}}, bounds))
    end
    return poly_ptr
end

function init_simplex5!(poly_ptr::Ptr{Poly5{N}}, verts::NTuple{6,RVec5}) where {N}
    vertsref = Ref(verts)
    GC.@preserve vertsref begin
        ccall((:rNd_init_simplex, _check_lib(libr3d_5d, "R3D_LIB_5D", 5)),
              Cvoid, (Ptr{Poly5{N}}, Ptr{NTuple{6,RVec5}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{6,RVec5}}, vertsref))
    end
    return poly_ptr
end

function clip5!(poly_ptr::Ptr{Poly5{N}}, planes::Vector{Plane5}) where {N}
    ccall((:rNd_clip, _check_lib(libr3d_5d, "R3D_LIB_5D", 5)),
          Cvoid, (Ptr{Poly5{N}}, Ptr{Plane5}, Int32),
          poly_ptr, planes, Int32(length(planes)))
    return poly_ptr
end

function reduce5!(poly_ptr::Ptr{Poly5{N}}, moments::Vector{Float64},
                  polyorder::Integer) where {N}
    ccall((:rNd_reduce, _check_lib(libr3d_5d, "R3D_LIB_5D", 5)),
          Cvoid, (Ptr{Poly5{N}}, Ptr{Float64}, Int32),
          poly_ptr, moments, Int32(polyorder))
    return moments
end

function nverts5(poly_ptr::Ptr{Poly5{N}}) where {N}
    offset = fieldoffset(Poly5{N}, 2)
    return unsafe_load(reinterpret(Ptr{Int32}, poly_ptr + offset))
end

# --- D = 6 ccall wrappers --------------------------------------------------

function init_box6!(poly_ptr::Ptr{Poly6{N}}, lo::RVec6, hi::RVec6) where {N}
    bounds = Ref((lo, hi))
    GC.@preserve bounds begin
        ccall((:rNd_init_box, _check_lib(libr3d_6d, "R3D_LIB_6D", 6)),
              Cvoid, (Ptr{Poly6{N}}, Ptr{NTuple{2,RVec6}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{2,RVec6}}, bounds))
    end
    return poly_ptr
end

function init_simplex6!(poly_ptr::Ptr{Poly6{N}}, verts::NTuple{7,RVec6}) where {N}
    vertsref = Ref(verts)
    GC.@preserve vertsref begin
        ccall((:rNd_init_simplex, _check_lib(libr3d_6d, "R3D_LIB_6D", 6)),
              Cvoid, (Ptr{Poly6{N}}, Ptr{NTuple{7,RVec6}}),
              poly_ptr, Base.unsafe_convert(Ptr{NTuple{7,RVec6}}, vertsref))
    end
    return poly_ptr
end

function clip6!(poly_ptr::Ptr{Poly6{N}}, planes::Vector{Plane6}) where {N}
    ccall((:rNd_clip, _check_lib(libr3d_6d, "R3D_LIB_6D", 6)),
          Cvoid, (Ptr{Poly6{N}}, Ptr{Plane6}, Int32),
          poly_ptr, planes, Int32(length(planes)))
    return poly_ptr
end

function reduce6!(poly_ptr::Ptr{Poly6{N}}, moments::Vector{Float64},
                  polyorder::Integer) where {N}
    ccall((:rNd_reduce, _check_lib(libr3d_6d, "R3D_LIB_6D", 6)),
          Cvoid, (Ptr{Poly6{N}}, Ptr{Float64}, Int32),
          poly_ptr, moments, Int32(polyorder))
    return moments
end

function nverts6(poly_ptr::Ptr{Poly6{N}}) where {N}
    offset = fieldoffset(Poly6{N}, 2)
    return unsafe_load(reinterpret(Ptr{Int32}, poly_ptr + offset))
end

end # module R3D_C
