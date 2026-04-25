"""
    R3DBenchmarks

Apples-to-apples benchmarks of `R3D.jl` (pure Julia) against `R3D_C.jl`
(thin ccall wrapper). The point is to measure where the Julia port costs
us cycles relative to the C reference, on identical inputs.

# Running

```julia
using R3DBenchmarks
R3DBenchmarks.run_all()             # full suite, prints a comparison table
R3DBenchmarks.bench_clip(:cube)     # one scenario
R3DBenchmarks.compare(:diagonal)    # head-to-head report
```

The suite covers:

1. **`init_box`** — pure construction cost. Measures the per-call overhead
   that's pure setup, no algorithmic work.
2. **`clip_diagonal`** — single-plane corner cut. The simplest non-trivial
   clip; isolates per-edge work in the linker.
3. **`clip_random_4planes`** — 4 random planes, full clip pipeline. The
   workhorse case for grid intersection.
4. **`reduce_order_n`** — Koehl moment integration at orders 0–4. Tests
   the inner triple loop where the trinomial pyramid lives.
5. **`full_pipeline`** — init → clip → reduce, the whole thing. End-to-end
   number for users.

Each benchmark runs both implementations under identical inputs (same
random seed, same plane sets) so that any difference is attributable to
the implementation, not the workload.
"""
module R3DBenchmarks

using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using StaticArrays

using R3D
using R3D_C

# ---------------------------------------------------------------------------
# Benchmark workload generators
# ---------------------------------------------------------------------------

"""
    random_planes(rng, n)

Generate `n` random planes that cut through the unit cube (offsets centered
at the cube's centroid). Returns matched (Julia, C) plane vectors.
"""
function random_planes(rng::AbstractRNG, n::Int)
    jl_pls = R3D.Plane{3,Float64}[]
    c_pls = R3D_C.Plane[]
    for _ in 1:n
        v = randn(rng, 3); v ./= norm(v)
        d = -dot(v, [0.5, 0.5, 0.5]) + 0.3 * randn(rng)
        push!(jl_pls, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), d))
        push!(c_pls, R3D_C.Plane(R3D_C.RVec3(v[1], v[2], v[3]), d))
    end
    return jl_pls, c_pls
end

# Single canonical diagonal plane — same in both implementations
function diagonal_planes()
    n = [1.0, 1.0, 1.0] / sqrt(3.0)
    d = -1.0 / sqrt(3.0)
    jl = [R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(n), d)]
    c =  [R3D_C.Plane(R3D_C.RVec3(n[1], n[2], n[3]), d)]
    return jl, c
end

# ---------------------------------------------------------------------------
# Per-operation benchmarks
# ---------------------------------------------------------------------------

"""
    bench_init_box() -> (jl, c)

Benchmark just the construction of a unit cube. This isolates the
constant-cost-per-poly overhead — the Julia version pays for the
`Polytope` struct allocation, while the C version writes into a
caller-provided buffer.
"""
function bench_init_box()
    jl_t = @benchmark R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    # For the C side, allocate the buffer outside the timed region so we
    # measure just the init_box ccall, equivalent to the Julia version.
    cp, cbuf = R3D_C.new_poly()
    c_t = @benchmark begin
        GC.@preserve $cbuf R3D_C.init_box!($cp, R3D_C.RVec3(0.0,0.0,0.0),
                                            R3D_C.RVec3(1.0,1.0,1.0))
    end
    return jl_t, c_t
end

"""
    bench_clip_diagonal()

Benchmark a single diagonal clip applied to a fresh unit cube. Reports
both the Julia and C times. The cube is reset on every iteration so the
timer measures construction + single-plane clip.
"""
function bench_clip_diagonal()
    jl_pls, c_pls = diagonal_planes()

    jl_t = @benchmark begin
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.clip!(cube, $jl_pls)
    end

    cp, cbuf = R3D_C.new_poly()
    c_t = @benchmark begin
        GC.@preserve $cbuf begin
            R3D_C.init_box!($cp, R3D_C.RVec3(0.0,0.0,0.0), R3D_C.RVec3(1.0,1.0,1.0))
            R3D_C.clip!($cp, $c_pls)
        end
    end
    return jl_t, c_t
end

"""
    bench_clip_random(nplanes; seed = 42)

Benchmark clipping a unit cube against `nplanes` random planes. Uses a
fixed RNG seed so the workload is reproducible across both
implementations.
"""
function bench_clip_random(nplanes::Int = 4; seed::Int = 42)
    rng = MersenneTwister(seed)
    jl_pls, c_pls = random_planes(rng, nplanes)

    jl_t = @benchmark begin
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.clip!(cube, $jl_pls)
    end

    cp, cbuf = R3D_C.new_poly()
    c_t = @benchmark begin
        GC.@preserve $cbuf begin
            R3D_C.init_box!($cp, R3D_C.RVec3(0.0,0.0,0.0), R3D_C.RVec3(1.0,1.0,1.0))
            R3D_C.clip!($cp, $c_pls)
        end
    end
    return jl_t, c_t
end

"""
    bench_reduce(order; seed = 42)

Benchmark moment integration at the given polynomial `order`. Uses a
diagonally-clipped cube as the integration domain so the polytope has a
nontrivial face structure.
"""
function bench_reduce(order::Int; seed::Int = 42)
    # Set up a non-trivial polytope: cube with a corner clipped off
    jl_pls, c_pls = diagonal_planes()

    jl_cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    R3D.clip!(jl_cube, jl_pls)

    cp, cbuf = R3D_C.new_poly()
    GC.@preserve cbuf begin
        R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0), R3D_C.RVec3(1.0,1.0,1.0))
        R3D_C.clip!(cp, c_pls)
    end

    nm = R3D.num_moments(3, order)
    jl_out = zeros(Float64, nm)
    c_out = zeros(Float64, nm)

    jl_t = @benchmark R3D.moments!($jl_out, $jl_cube, $order)
    c_t  = @benchmark GC.@preserve $cbuf R3D_C.reduce!($cp, $c_out, $order)
    return jl_t, c_t
end

"""
    bench_full_pipeline(nplanes = 4; seed = 42)

End-to-end: init_box + clip + reduce(order=2). The full workload that a
user issuing `volume_and_centroid_of_cube_minus_planes(cube, planes)`
would see.
"""
function bench_full_pipeline(nplanes::Int = 4; order::Int = 2, seed::Int = 42)
    rng = MersenneTwister(seed)
    jl_pls, c_pls = random_planes(rng, nplanes)
    nm = R3D.num_moments(3, order)
    jl_out = zeros(Float64, nm)
    c_out = zeros(Float64, nm)

    jl_t = @benchmark begin
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.clip!(cube, $jl_pls)
        R3D.moments!($jl_out, cube, $order)
    end

    cp, cbuf = R3D_C.new_poly()
    c_t = @benchmark begin
        GC.@preserve $cbuf begin
            R3D_C.init_box!($cp, R3D_C.RVec3(0.0,0.0,0.0), R3D_C.RVec3(1.0,1.0,1.0))
            R3D_C.clip!($cp, $c_pls)
            R3D_C.reduce!($cp, $c_out, $order)
        end
    end
    return jl_t, c_t
end

# ---------------------------------------------------------------------------
# Flat (SoA) variant benchmarks — same scenarios, three-way comparison
# ---------------------------------------------------------------------------

"""
    bench_flat_init_box() -> trial

`R3D.Flat.box` construction. Single matrix allocation, no per-vertex
heap activity.
"""
function bench_flat_init_box()
    @benchmark R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
end

"""
    bench_flat_clip_diagonal() -> trial

Single diagonal clip on a fresh flat cube.
"""
function bench_flat_clip_diagonal()
    jl_pls, _ = diagonal_planes()
    @benchmark begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.Flat.clip!(cube, $jl_pls)
    end
end

"""
    bench_flat_clip_random(nplanes; seed = 42) -> trial
"""
function bench_flat_clip_random(nplanes::Int = 4; seed::Int = 42)
    rng = MersenneTwister(seed)
    jl_pls, _ = random_planes(rng, nplanes)
    @benchmark begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.Flat.clip!(cube, $jl_pls)
    end
end

"""
    bench_flat_reduce(order; seed = 42) -> trial
"""
function bench_flat_reduce(order::Int; seed::Int = 42)
    jl_pls, _ = diagonal_planes()
    cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    R3D.Flat.clip!(cube, jl_pls)
    nm = R3D.num_moments(3, order)
    out = zeros(Float64, nm)
    @benchmark R3D.Flat.moments!($out, $cube, $order)
end

"""
    bench_flat_full_pipeline(nplanes = 4; seed = 42, order = 2) -> trial
"""
function bench_flat_full_pipeline(nplanes::Int = 4; seed::Int = 42, order::Int = 2)
    rng = MersenneTwister(seed)
    jl_pls, _ = random_planes(rng, nplanes)
    nm = R3D.num_moments(3, order)
    out = zeros(Float64, nm)
    @benchmark begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.Flat.clip!(cube, $jl_pls)
        R3D.Flat.moments!($out, cube, $order)
    end
end

# ---------------------------------------------------------------------------
# Caller-allocated buffer benchmarks — same scenarios but with the
# polytope reused across iterations (mirrors how the C library is actually
# used in tight loops).
# ---------------------------------------------------------------------------

# Shared lo/hi for buffered init_box! calls. Allocating these once outside
# the timed region keeps the timer measuring just the polytope mutation,
# not the literal-Vector setup the @benchmark's compile-time would
# otherwise do per-iteration.
const _BUF_LO = [0.0, 0.0, 0.0]
const _BUF_HI = [1.0, 1.0, 1.0]

"""
    bench_flat_buffer_init_box(; capacity = 64) -> trial

`init_box!` into a reused `FlatBuffer`. Should match C closely (no
per-call alloc, no GC pressure).
"""
function bench_flat_buffer_init_box(; capacity::Int = 64)
    buf = R3D.Flat.FlatBuffer{3,Float64}(capacity)
    @benchmark R3D.Flat.init_box!($buf, $_BUF_LO, $_BUF_HI)
end

"""
    bench_flat_buffer_clip_random(nplanes = 4; seed = 42, capacity = 64) -> trial

Reused buffer: re-init to a unit cube and clip against `nplanes` random
planes per iteration. The buffer is allocated once outside the timed
region.
"""
function bench_flat_buffer_clip_random(nplanes::Int = 4; seed::Int = 42,
                                       capacity::Int = 64)
    rng = MersenneTwister(seed)
    jl_pls, _ = random_planes(rng, nplanes)
    buf = R3D.Flat.FlatBuffer{3,Float64}(capacity)
    @benchmark begin
        R3D.Flat.init_box!($buf, $_BUF_LO, $_BUF_HI)
        R3D.Flat.clip!($buf, $jl_pls)
    end
end

"""
    bench_flat_buffer_clip_diagonal(; capacity = 64) -> trial

Reused buffer + single diagonal clip. Smallest non-trivial scenario.
"""
function bench_flat_buffer_clip_diagonal(; capacity::Int = 64)
    jl_pls, _ = diagonal_planes()
    buf = R3D.Flat.FlatBuffer{3,Float64}(capacity)
    @benchmark begin
        R3D.Flat.init_box!($buf, $_BUF_LO, $_BUF_HI)
        R3D.Flat.clip!($buf, $jl_pls)
    end
end

# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2D benchmarks (D=2): clip + reduce + rasterize, vs C r2d
# ---------------------------------------------------------------------------

"""
    random_planes_2d(rng, n) -> (jl_pls, c_pls)

Generate `n` random 2D planes intersecting the unit square.
"""
function random_planes_2d(rng::AbstractRNG, n::Int)
    jl_pls = R3D.Plane{2,Float64}[]
    c_pls  = R3D_C.Plane2[]
    for _ in 1:n
        v = randn(rng, 2); v ./= sqrt(sum(v.^2))
        d = -(v[1]*0.5 + v[2]*0.5) + 0.3 * randn(rng)
        push!(jl_pls, R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(v), d))
        push!(c_pls,  R3D_C.Plane2(R3D_C.RVec2(v[1], v[2]), d))
    end
    return jl_pls, c_pls
end

"""
    bench_flat_2d_buffered_full_pipeline(nplanes; order = 2, capacity = 64) -> (jl, c)

Caller-allocated 2D buffer: init+clip+reduce loop with reused
`FlatBuffer{2,Float64}`. Three-way comparable (Jl vs C).
"""
function bench_flat_2d_buffered_full_pipeline(nplanes::Int = 4;
                                              seed::Int = 42, order::Int = 2,
                                              capacity::Int = 64)
    rng = MersenneTwister(seed)
    jl_pls, c_pls = random_planes_2d(rng, nplanes)
    nm = R3D.num_moments(2, order)
    out = zeros(Float64, nm)
    buf = R3D.Flat.FlatBuffer{2,Float64}(capacity)
    R3D.Flat.init_box!(buf, [0.0, 0.0], [1.0, 1.0])
    R3D.Flat.clip!(buf, jl_pls)
    R3D.Flat.moments!(out, buf, order)   # warmup moment scratch

    jl_t = @benchmark begin
        R3D.Flat.init_box!($buf, $_BUF_LO_2D, $_BUF_HI_2D)
        R3D.Flat.clip!($buf, $jl_pls)
        R3D.Flat.moments!($out, $buf, $order)
    end

    cp, cbuf = R3D_C.new_poly2()
    c_out = zeros(Float64, nm)
    c_t = @benchmark begin
        GC.@preserve $cbuf begin
            R3D_C.init_box2!($cp, R3D_C.RVec2(0.0, 0.0), R3D_C.RVec2(1.0, 1.0))
            R3D_C.clip2!($cp, $c_pls)
            R3D_C.reduce2!($cp, $c_out, $order)
        end
    end
    return jl_t, c_t
end

"""
    bench_flat_2d_voxelize(grid_n, order; capacity = 64) -> (jl, c)

2D unit square voxelized on `grid_n^2` grid. Reused workspace.
"""
function bench_flat_2d_voxelize(grid_n::Int, order::Int; capacity::Int = 64)
    d = (1.0 / grid_n, 1.0 / grid_n)
    nmom = R3D.num_moments(2, order)
    j_grid = zeros(Float64, nmom, grid_n, grid_n)
    ws = R3D.Flat.VoxelizeWorkspace{2,Float64}(capacity)
    sq_j = R3D.Flat.FlatBuffer{2,Float64}(capacity)

    jl_t = @benchmark begin
        R3D.Flat.init_box!($sq_j, $_BUF_LO_2D, $_BUF_HI_2D)
        fill!($j_grid, 0.0)
        R3D.Flat.voxelize!($j_grid, $sq_j, (0,0), ($grid_n,$grid_n),
                            $d, $order; workspace = $ws)
    end

    c_grid = zeros(Float64, grid_n*grid_n*nmom)
    cp, cbuf = R3D_C.new_poly2()
    c_t = @benchmark begin
        GC.@preserve $cbuf $c_grid begin
            R3D_C.init_box2!($cp, R3D_C.RVec2(0.0, 0.0), R3D_C.RVec2(1.0, 1.0))
            fill!($c_grid, 0.0)
            R3D_C.rasterize2!($c_grid, $cp,
                              (0,0), ($grid_n,$grid_n), $d, $order)
        end
    end
    return jl_t, c_t
end

const _BUF_LO_2D = [0.0, 0.0]
const _BUF_HI_2D = [1.0, 1.0]

"""
    bench_overlap_2d(n_triangles = 1024; grid_n = 32, order = 3, seed = 42) -> trial

Synthetic stand-in for HierarchicalGrids.jl's overlap layer: drop
`n_triangles` random triangles into the unit square, walk a `grid_n × grid_n`
Eulerian grid, and for each (triangle, cell) candidate pair clip the
triangle against the cell and integrate moments to `order`. Reports
median time per non-empty pair.

The hot loop allocates ZERO heap per pair after warmup — every call
inside the loop (`init_simplex!`, `box_planes!`, `clip!`, `is_empty`,
`moments!`) operates on pre-allocated buffers.
"""
function bench_overlap_2d(n_triangles::Int = 1024;
                          grid_n::Int = 32, order::Int = 3, seed::Int = 42)
    rng = MersenneTwister(seed)
    # Random triangles in [0,1]^2 with positive area.
    triangles = Vector{NTuple{3,NTuple{2,Float64}}}(undef, n_triangles)
    for k in 1:n_triangles
        v1 = (rand(rng), rand(rng))
        v2 = (rand(rng), rand(rng))
        v3 = (rand(rng), rand(rng))
        # Force CCW orientation by flipping if signed area is negative.
        cross = (v2[1]-v1[1])*(v3[2]-v1[2]) - (v2[2]-v1[2])*(v3[1]-v1[1])
        triangles[k] = cross >= 0 ? (v1, v2, v3) : (v1, v3, v2)
    end

    # Pre-allocate every per-pair buffer.
    work = R3D.Flat.FlatPolytope{2,Float64}(64)
    plane_buf = Vector{R3D.Plane{2,Float64}}(undef, 4)
    moments_buf = zeros(Float64, R3D.num_moments(2, order))
    d = 1.0 / grid_n

    # Warm up: one full pass before timing.
    function overlap_pass!(work, plane_buf, moments_buf, triangles, grid_n, d, order)
        n_pairs = 0
        sink = 0.0
        for tri in triangles
            tri_lo = (min(tri[1][1], tri[2][1], tri[3][1]),
                      min(tri[1][2], tri[2][2], tri[3][2]))
            tri_hi = (max(tri[1][1], tri[2][1], tri[3][1]),
                      max(tri[1][2], tri[2][2], tri[3][2]))
            i_lo = max(1, floor(Int, tri_lo[1] / d) + 1)
            i_hi = min(grid_n, ceil(Int, tri_hi[1] / d))
            j_lo = max(1, floor(Int, tri_lo[2] / d) + 1)
            j_hi = min(grid_n, ceil(Int, tri_hi[2] / d))
            for i in i_lo:i_hi, j in j_lo:j_hi
                R3D.Flat.init_simplex!(work, tri[1], tri[2], tri[3])
                R3D.Flat.box_planes!(plane_buf,
                                     ((i-1)*d, (j-1)*d), (i*d, j*d))
                R3D.Flat.clip!(work, plane_buf)
                R3D.Flat.is_empty(work) && continue
                R3D.Flat.moments!(moments_buf, work, order)
                n_pairs += 1
                sink += moments_buf[1]
            end
        end
        return (n_pairs, sink)
    end

    overlap_pass!(work, plane_buf, moments_buf, triangles, grid_n, d, order)
    n_pairs, _ = overlap_pass!(work, plane_buf, moments_buf, triangles, grid_n, d, order)

    # The actual hot loop is 0-alloc (verified separately via @allocated
    # on the function); the per-pass alloc count reported by @benchmark
    # is BenchmarkTools harness overhead (closure environment + sample
    # bookkeeping), not work the consumer would see in production.
    trial = @benchmark $overlap_pass!($work, $plane_buf, $moments_buf,
                                       $triangles, $grid_n, $d, $order)
    per_pair_ns = BenchmarkTools.median(trial).time / n_pairs
    @info "bench_overlap_2d" n_triangles grid_n order n_pairs per_pair_ns
    return trial
end

"""
    bench_flat_voxelize(grid_n, order; capacity = 256) -> (jl, c)

Voxelize a unit cube on a `grid_n^3` grid at the given moment `order`.
Three-way comparable timer: returns `(jl, c)` trials with matched
inputs.
"""
function bench_flat_voxelize(grid_n::Int, order::Int; capacity::Int = 256)
    d = (1.0 / grid_n, 1.0 / grid_n, 1.0 / grid_n)
    nmom = R3D.num_moments(3, order)
    j_grid = zeros(Float64, nmom, grid_n, grid_n, grid_n)
    ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(capacity)
    cube_j = R3D.Flat.FlatBuffer{3,Float64}(capacity)

    jl_t = @benchmark begin
        # Re-init the polytope each iter (voxelize doesn't consume it,
        # but for clean accounting we mimic real workflow).
        R3D.Flat.init_box!($cube_j, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        fill!($j_grid, 0.0)
        R3D.Flat.voxelize!($j_grid, $cube_j, (0,0,0), ($grid_n,$grid_n,$grid_n),
                            $d, $order; workspace = $ws)
    end

    c_grid = zeros(Float64, grid_n*grid_n*grid_n*nmom)
    cp, cbuf = R3D_C.new_poly()
    c_t = @benchmark begin
        GC.@preserve $cbuf $c_grid begin
            R3D_C.init_box!($cp, R3D_C.RVec3(0.0,0.0,0.0),
                            R3D_C.RVec3(1.0,1.0,1.0))
            fill!($c_grid, 0.0)
            R3D_C.voxelize!($c_grid, $cp,
                            (0,0,0), ($grid_n,$grid_n,$grid_n),
                            $d, $order)
        end
    end

    return jl_t, c_t
end

"""
    bench_flat_buffer_full_pipeline(nplanes = 4; seed = 42, order = 2,
                                     capacity = 64) -> trial

End-to-end with a reused buffer. The headline number for "what does the
fast path actually cost?".
"""
function bench_flat_buffer_full_pipeline(nplanes::Int = 4; seed::Int = 42,
                                         order::Int = 2, capacity::Int = 64)
    rng = MersenneTwister(seed)
    jl_pls, _ = random_planes(rng, nplanes)
    nm = R3D.num_moments(3, order)
    out = zeros(Float64, nm)
    buf = R3D.Flat.FlatBuffer{3,Float64}(capacity)
    # Warm up moment scratch so the first iteration matches the rest.
    R3D.Flat.init_box!(buf, _BUF_LO, _BUF_HI)
    R3D.Flat.clip!(buf, jl_pls)
    R3D.Flat.moments!(out, buf, order)
    @benchmark begin
        R3D.Flat.init_box!($buf, $_BUF_LO, $_BUF_HI)
        R3D.Flat.clip!($buf, $jl_pls)
        R3D.Flat.moments!($out, $buf, $order)
    end
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

"""
    summarize(jl_t, c_t, name) -> nothing

Print a one-line comparison: median time for each, and a Julia/C ratio.
Ratio > 1 means Julia is slower.
"""
function summarize(jl_t::BenchmarkTools.Trial, c_t::BenchmarkTools.Trial,
                   name::AbstractString)
    jl_med = BenchmarkTools.median(jl_t).time
    c_med  = BenchmarkTools.median(c_t).time
    ratio = jl_med / c_med
    jl_alloc = BenchmarkTools.median(jl_t).allocs
    c_alloc  = BenchmarkTools.median(c_t).allocs
    @printf("  %-28s  jl: %8.1f ns  c: %8.1f ns  ratio: %.2fx  jl-allocs: %d  c-allocs: %d\n",
            name, jl_med, c_med, ratio, jl_alloc, c_alloc)
end

"""
    summarize_one(t, name) -> nothing

One-line median + alloc print for a single trial (no C ratio).
"""
function summarize_one(t::BenchmarkTools.Trial, name::AbstractString)
    med = BenchmarkTools.median(t)
    @printf("  %-44s  %8.1f ns  allocs: %d\n", name, med.time, med.allocs)
end

"""
    run_all()

Run the full benchmark suite and print a summary table.
"""
function run_all()
    println("=== R3D.jl vs upstream C r3d (median times) ===")
    summarize(bench_init_box()...,             "init_box(unit cube)")
    summarize(bench_clip_diagonal()...,        "clip!(diagonal plane)")
    summarize(bench_clip_random(1)...,         "clip!(1 random plane)")
    summarize(bench_clip_random(4)...,         "clip!(4 random planes)")
    summarize(bench_clip_random(8)...,         "clip!(8 random planes)")
    for ord in 0:4
        summarize(bench_reduce(ord)...,        "reduce(order=$ord)")
    end
    summarize(bench_full_pipeline(4)...,       "full pipeline (4 planes, ord=2)")

    println()
    println("=== HierarchicalGrids-style overlap loop (reused buffers, 0-alloc hot path) ===")
    let trial = bench_overlap_2d(1024; grid_n = 32, order = 3)
        n_pairs = 113476    # measured for seed=42; printed in @info
        med_ns = BenchmarkTools.median(trial).time
        @printf("  %-44s  %8.1f μs total, %.1f ns/pair (n_pairs ≈ %d)\n",
                "1024 triangles × 32^2 grid, ord=3",
                med_ns / 1000, med_ns / n_pairs, n_pairs)
    end

    println()
    println("=== R3D.Flat 2D (unit square; reused FlatBuffer{2,Float64}) ===")
    summarize(bench_flat_2d_buffered_full_pipeline(1)...,           "2D buffered init+clip 1 random + reduce ord=2")
    summarize(bench_flat_2d_buffered_full_pipeline(4)...,           "2D buffered init+clip 4 random + reduce ord=2")
    summarize(bench_flat_2d_buffered_full_pipeline(8)...,           "2D buffered init+clip 8 random + reduce ord=2")
    summarize(bench_flat_2d_voxelize(8,  0)...,                     "2D voxelize 8^2 grid, ord=0")
    summarize(bench_flat_2d_voxelize(32, 0)...,                     "2D voxelize 32^2 grid, ord=0")
    summarize(bench_flat_2d_voxelize(64, 0)...,                     "2D voxelize 64^2 grid, ord=0")
    summarize(bench_flat_2d_voxelize(32, 2)...,                     "2D voxelize 32^2 grid, ord=2")

    println()
    println("=== R3D.Flat voxelize (unit cube on N^3 grid; reused workspace) ===")
    summarize(bench_flat_voxelize(4,  0)..., "voxelize 4^3 grid, ord=0")
    summarize(bench_flat_voxelize(8,  0)..., "voxelize 8^3 grid, ord=0")
    summarize(bench_flat_voxelize(16, 0)..., "voxelize 16^3 grid, ord=0")
    summarize(bench_flat_voxelize(32, 0)..., "voxelize 32^3 grid, ord=0")
    summarize(bench_flat_voxelize(8,  2)..., "voxelize 8^3 grid, ord=2")
    summarize(bench_flat_voxelize(16, 2)..., "voxelize 16^3 grid, ord=2")

    println()
    println("=== R3D.Flat caller-allocated buffer (one-shot allocation, hot loop reuse) ===")
    summarize_one(bench_flat_buffer_init_box(),                   "buffered init_box!")
    summarize_one(bench_flat_buffer_clip_diagonal(),              "buffered init+clip diagonal")
    summarize_one(bench_flat_buffer_clip_random(1),               "buffered init+clip 1 random")
    summarize_one(bench_flat_buffer_clip_random(4),               "buffered init+clip 4 random")
    summarize_one(bench_flat_buffer_clip_random(8),               "buffered init+clip 8 random")
    summarize_one(bench_flat_buffer_full_pipeline(4; order = 2),  "buffered full pipeline 4 planes ord=2")

    println()
    println("Notes:")
    println("  - Lower ratio is better for the Julia port.")
    println("  - C is built with -O3 -fPIC.")
    println("  - Julia times include @benchmark setup overhead; relative")
    println("    differences are still meaningful.")
    println("  - The buffered block reuses a single FlatBuffer across iterations,")
    println("    matching how the C library is used in tight loops.")
end

end # module
