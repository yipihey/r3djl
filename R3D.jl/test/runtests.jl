using Test
using R3D
using ForwardDiff
using Aqua

# Slack for `@allocated f(x)` style heap-pressure assertions.
# Julia 1.11+ is consistently 0 bytes for the operations we test
# (small Tuple/NTuple returns and stack-resident scratch are all
# fully optimized). Julia 1.10's escape analysis is weaker — small
# return tuples and kwargs-call machinery sometimes spill, with
# observed per-call overhead in the 16–256 byte range depending on
# how the call is shaped (kwargs, bisection iboxes, captured
# locals). We assert 0 on 1.11+ to catch real regressions, and
# ≤ 256 on 1.10 to keep the LTS slot in the CI matrix green
# without losing signal on the important hot paths. The underlying
# operations are truly 0-alloc on 1.10 too — wrap the call in a
# nested function and the spillover vanishes; the residual is
# purely @allocated / @testset scope overhead that's not present
# in real consumer code.
const ALLOC_TOLERANCE = VERSION >= v"1.11" ? 0 : 256

@testset "R3D pure-Julia" begin

    @testset "Construction" begin
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test cube.nverts == 8
        @test R3D.capacity(cube) == 512  # default static cap

        # Each vertex of a 3D box has 3 neighbours
        for i in 1:8
            @test count(!iszero, cube.verts[i].pnbrs) == 3
        end

        # Dynamic storage variant
        cube_dyn = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0);
                           storage = R3D.DynamicStorage)
        @test cube_dyn.nverts == 8
        @test R3D.capacity(cube_dyn) == typemax(Int)
    end

    @testset "Volumes — closed forms" begin
        # Unit cube
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        m = R3D.moments(cube, 0)
        @test m[1] ≈ 1.0

        # 2×3×4 box
        big = R3D.box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        @test R3D.moments(big, 0)[1] ≈ 24.0

        # Tetrahedron with corners at origin and unit axes: V = 1/6
        t = R3D.tet(R3D.Vec{3,Float64}(0, 0, 0),
                    R3D.Vec{3,Float64}(1, 0, 0),
                    R3D.Vec{3,Float64}(0, 1, 0),
                    R3D.Vec{3,Float64}(0, 0, 1))
        @test R3D.moments(t, 0)[1] ≈ 1/6
    end

    @testset "Centroid via first moments" begin
        # Cube [0,1]³: centroid (0.5, 0.5, 0.5)
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        m = R3D.moments(cube, 1)
        V = m[1]
        @test m[2] / V ≈ 0.5
        @test m[3] / V ≈ 0.5
        @test m[4] / V ≈ 0.5

        # Off-center cube
        cube2 = R3D.box((1.0, 2.0, 3.0), (3.0, 5.0, 7.0))
        m2 = R3D.moments(cube2, 1)
        V2 = m2[1]
        @test V2 ≈ 24.0
        @test m2[2] / V2 ≈ 2.0
        @test m2[3] / V2 ≈ 3.5
        @test m2[4] / V2 ≈ 5.0
    end

    @testset "Single-plane clip" begin
        # Cube clipped by x ≥ 0.5 should give half the volume
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(1, 0, 0), -0.5)
        @test R3D.clip!(cube, [plane])
        @test R3D.moments(cube, 0)[1] ≈ 0.5

        # The remaining centroid should be at x = 0.75
        m = R3D.moments(cube, 1)
        @test m[2] / m[1] ≈ 0.75
    end

    @testset "Diagonal corner clip" begin
        # Cube ∩ {x + y + z ≥ 1} = cube minus origin-corner tet ⇒ 5/6
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        n = [1.0, 1.0, 1.0] ./ sqrt(3.0)
        d = -1.0 / sqrt(3.0)
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(n), d)
        @test R3D.clip!(cube, [plane])
        @test R3D.moments(cube, 0)[1] ≈ 5/6
    end

    @testset "Empty result on full clip" begin
        cube = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        # Plane way outside, retaining the negative half
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(1, 0, 0), -100.0)
        @test R3D.clip!(cube, [plane])
        @test cube.nverts == 0
        @test R3D.moments(cube, 0)[1] == 0.0
    end

    @testset "Type stability" begin
        # 32-bit version
        cube_f32 = R3D.box(R3D.Vec{3,Float32}(0, 0, 0),
                           R3D.Vec{3,Float32}(1, 1, 1))
        @test eltype(cube_f32.verts[1].pos) == Float32
        m = R3D.moments(cube_f32, 0)
        @test eltype(m) == Float32
        @test m[1] ≈ 1.0f0
    end
end

# Differential tests live in their own file; only run if a libr3d is available
include("differential.jl")

# Flat (SoA) variant: smoke + differential tests
@testset "R3D.Flat (SoA variant)" begin

    @testset "Construction and volume" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test cube.nverts == 8
        @test R3D.Flat.moments(cube, 0)[1] ≈ 1.0

        big = R3D.Flat.box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        @test R3D.Flat.moments(big, 0)[1] ≈ 24.0
    end

    @testset "Clip and centroid" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(1, 0, 0), -0.5)
        ok = R3D.Flat.clip!(cube, [plane])
        @test ok
        m = R3D.Flat.moments(cube, 1)
        @test m[1] ≈ 0.5
        @test m[2] / m[1] ≈ 0.75
    end

    @testset "Diagonal clip" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        n = [1.0, 1.0, 1.0] / sqrt(3.0); d = -1/sqrt(3.0)
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(n), d)
        @test R3D.Flat.clip!(cube, [plane])
        @test R3D.Flat.moments(cube, 0)[1] ≈ 5/6
    end

    @testset "Cross-validation: AoS == Flat == C on random clips" begin
        if !HAVE_C
            @info "skipping cross-validation; ENV[R3D_LIB] not set"
            return
        end
        rng = Random.MersenneTwister(20260425)
        for trial in 1:200
            # Make matched cubes in all three implementations
            jl = R3D.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            flat = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            cp, cbuf = R3D_C.new_poly()
            GC.@preserve cbuf R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                               R3D_C.RVec3(1.0,1.0,1.0))

            # Same plane set in all three
            nplanes = rand(rng, 1:6)
            jl_pls = R3D.Plane{3,Float64}[]
            c_pls = R3D_C.Plane[]
            for _ in 1:nplanes
                v = randn(rng, 3); v ./= sqrt(sum(v.^2))
                d = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.3*randn(rng)
                push!(jl_pls, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), d))
                push!(c_pls, R3D_C.Plane(R3D_C.RVec3(v[1],v[2],v[3]), d))
            end

            R3D.clip!(jl, jl_pls)
            R3D.Flat.clip!(flat, jl_pls)
            GC.@preserve cbuf R3D_C.clip!(cp, c_pls)

            # All three volumes must agree
            v_aos  = R3D.moments(jl, 0)[1]
            v_flat = R3D.Flat.moments(flat, 0)[1]
            v_c = zeros(Float64, 1)
            GC.@preserve cbuf R3D_C.reduce!(cp, v_c, 0)

            @test isapprox(v_aos,  v_c[1]; atol=1e-12, rtol=1e-10)
            @test isapprox(v_flat, v_c[1]; atol=1e-12, rtol=1e-10)
        end
    end
end

# Caller-allocated buffer reuse: a single FlatBuffer driven through many
# clips must produce the same results as a fresh buffer per clip. Catches
# state-leakage bugs that single-shot tests miss.
@testset "R3D.Flat buffer reuse" begin
    rng = Random.MersenneTwister(20260425)

    @testset "Reused buffer matches fresh buffer (1000 random clips)" begin
        buf = R3D.Flat.FlatBuffer{3,Float64}(64)
        out_reused = zeros(Float64, R3D.num_moments(3, 1))
        out_fresh  = zeros(Float64, R3D.num_moments(3, 1))
        for trial in 1:1000
            nplanes = rand(rng, 1:6)
            planes = R3D.Plane{3,Float64}[]
            for _ in 1:nplanes
                v = randn(rng, 3); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.3 * randn(rng)
                push!(planes, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd))
            end

            R3D.Flat.init_box!(buf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            R3D.Flat.clip!(buf, planes)
            R3D.Flat.moments!(out_reused, buf, 1)

            fresh = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            R3D.Flat.clip!(fresh, planes)
            R3D.Flat.moments!(out_fresh, fresh, 1)

            for k in 1:length(out_fresh)
                @test isapprox(out_reused[k], out_fresh[k]; atol=1e-12, rtol=1e-10)
            end
        end
    end

    @testset "Buffered hot loop is allocation-free (after warmup)" begin
        buf = R3D.Flat.FlatBuffer{3,Float64}(64)
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0]/sqrt(3.0)),
                                      -1/sqrt(3.0))
        planes = [plane]
        out = zeros(Float64, R3D.num_moments(3, 2))

        # warm up (first call at a given moment order allocates Sm/Dm/Cm)
        R3D.Flat.init_box!(buf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        R3D.Flat.clip!(buf, planes)
        R3D.Flat.moments!(out, buf, 2)

        a_init = @allocated R3D.Flat.init_box!(buf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        a_clip = @allocated R3D.Flat.clip!(buf, planes)
        a_mom  = @allocated R3D.Flat.moments!(out, buf, 2)
        # `init_box!` accepts AbstractVector, so the literal `[…]` arg
        # itself allocates; the polytope mutation does not. We only assert
        # zero allocation in the hot ops that write into `buf`.
        @test a_clip <= ALLOC_TOLERANCE
        @test a_mom  <= ALLOC_TOLERANCE
        @test a_init <= 256  # only the literal Vector{Float64}(...) the call site builds
    end

    @testset "Buffered run agrees with C (HAVE_C)" begin
        if !HAVE_C
            @info "skipping buffered-vs-C cross-check; ENV[R3D_LIB] not set"
            return
        end
        rng2 = Random.MersenneTwister(20260426)
        buf = R3D.Flat.FlatBuffer{3,Float64}(64)
        out = zeros(Float64, R3D.num_moments(3, 0))
        cp, cbuf = R3D_C.new_poly()
        for trial in 1:1000
            nplanes = rand(rng2, 1:6)
            jl_pls = R3D.Plane{3,Float64}[]
            c_pls = R3D_C.Plane[]
            for _ in 1:nplanes
                v = randn(rng2, 3); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.3 * randn(rng2)
                push!(jl_pls, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd))
                push!(c_pls, R3D_C.Plane(R3D_C.RVec3(v[1], v[2], v[3]), dd))
            end

            R3D.Flat.init_box!(buf, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            R3D.Flat.clip!(buf, jl_pls)
            R3D.Flat.moments!(out, buf, 0)

            GC.@preserve cbuf begin
                R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                R3D_C.RVec3(1.0,1.0,1.0))
                R3D_C.clip!(cp, c_pls)
            end
            v_c = zeros(Float64, 1)
            GC.@preserve cbuf R3D_C.reduce!(cp, v_c, 0)

            @test isapprox(out[1], v_c[1]; atol=1e-12, rtol=1e-10)
        end
    end
end

# Voxelization (R3D.Flat.voxelize!) — closed-form + differential vs C.
@testset "R3D.Flat voxelize" begin

    @testset "voxelize_fold! basis-agnostic leaf hook (D=3)" begin
        # 1. Sum-fold equals voxelize! exactly. The point of refactoring
        #    voxelize! over voxelize_fold! is that this round-trips bit
        #    for bit, but assert it explicitly so future regressions
        #    show up here rather than buried in the diff-vs-C testset.
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0]/sqrt(3)),
                                      -1/sqrt(3))
        R3D.Flat.clip!(cube, [plane])
        d = (0.1, 0.1, 0.1)
        lo, hi = R3D.Flat.get_ibox(cube, d)

        ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
        grid_via_voxelize = zeros(Float64, R3D.num_moments(3, 2),
                                   hi[1]-lo[1], hi[2]-lo[2], hi[3]-lo[3])
        R3D.Flat.voxelize!(grid_via_voxelize, cube, lo, hi, d, 2; workspace = ws)

        grid_via_fold = zeros(Float64, R3D.num_moments(3, 2),
                               hi[1]-lo[1], hi[2]-lo[2], hi[3]-lo[3])
        R3D.Flat.voxelize_fold!(grid_via_fold, cube, lo, hi, d, 2;
                                  workspace = ws) do g, i, j, k, m
            @inbounds for mi in 1:length(m)
                g[mi, i, j, k] += m[mi]
            end
            return g
        end
        @test grid_via_voxelize == grid_via_fold

        # 2. Total volume from a fold matches the polytope's analytic
        #    volume. Sums independent of leaf order.
        cube2 = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        total = R3D.Flat.voxelize_fold!(0.0, cube2, (0,0,0), (4,4,4),
                                          (0.25, 0.25, 0.25), 0;
                                          workspace = ws) do acc, i, j, k, m
            return acc + m[1]
        end
        @test isapprox(total, 1.0; atol=1e-12)

        # 3. Fused dot-product fold equals two-pass voxelize! + post-dot.
        cube3 = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.Flat.clip!(cube3, [plane])
        nmom = R3D.num_moments(3, 3)
        coeffs = collect(1.0:nmom)
        # Two-pass reference
        full_grid = zeros(Float64, nmom, hi[1]-lo[1], hi[2]-lo[2], hi[3]-lo[3])
        R3D.Flat.voxelize!(full_grid, cube3, lo, hi, d, 3; workspace = ws)
        scalar_ref = zeros(Float64, hi[1]-lo[1], hi[2]-lo[2], hi[3]-lo[3])
        for k in 1:size(full_grid, 4), j in 1:size(full_grid, 3), i in 1:size(full_grid, 2)
            s = 0.0
            for mi in 1:nmom
                s += coeffs[mi] * full_grid[mi, i, j, k]
            end
            scalar_ref[i, j, k] = s
        end
        # Fused
        cube3b = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        R3D.Flat.clip!(cube3b, [plane])
        scalar_fused = zeros(Float64, hi[1]-lo[1], hi[2]-lo[2], hi[3]-lo[3])
        R3D.Flat.voxelize_fold!(scalar_fused, cube3b, lo, hi, d, 3;
                                  workspace = ws) do acc, i, j, k, m
            s = 0.0
            @inbounds for mi in 1:length(m)
                s += coeffs[mi] * m[mi]
            end
            @inbounds acc[i, j, k] += s
            return acc
        end
        @test all(isapprox.(scalar_fused, scalar_ref; atol=1e-12))

        # 4. Hot-loop allocation check: warmed-up fold is 0-alloc.
        # The callback must be hoisted out of the @allocated expression
        # — a do-block evaluated INSIDE @allocated would create a new
        # anonymous-function type at that source location and trigger
        # compilation, which @allocated charges to the call. In real
        # consumer code the do-block's type is fixed at the call site
        # and compiled only once, so the per-call cost is what we
        # measure here.
        cube4 = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        scalar2 = zeros(Float64, 4, 4, 4)
        sum_first_moment = (acc, i, j, k, m) -> (@inbounds acc[i,j,k] += m[1]; acc)
        R3D.Flat.voxelize_fold!(sum_first_moment, scalar2, cube4, (0,0,0), (4,4,4),
                                 (0.25,0.25,0.25), 1; workspace = ws)   # warmup
        fill!(scalar2, 0.0)
        a = @allocated R3D.Flat.voxelize_fold!(sum_first_moment, scalar2, cube4,
                                                (0,0,0), (4,4,4), (0.25,0.25,0.25), 1;
                                                workspace = ws)
        @test a <= ALLOC_TOLERANCE
    end

    @testset "voxelize_fold! basis-agnostic leaf hook (D=2)" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        d2 = (0.25, 0.25)
        ws = R3D.Flat.VoxelizeWorkspace{2,Float64}(64)
        grid_via_voxelize = zeros(Float64, R3D.num_moments(2, 1), 4, 4)
        R3D.Flat.voxelize!(grid_via_voxelize, sq, (0,0), (4,4), d2, 1;
                            workspace = ws)
        grid_via_fold = zeros(Float64, R3D.num_moments(2, 1), 4, 4)
        R3D.Flat.voxelize_fold!(grid_via_fold, sq, (0,0), (4,4), d2, 1;
                                  workspace = ws) do g, i, j, m
            @inbounds for mi in 1:length(m)
                g[mi, i, j] += m[mi]
            end
            return g
        end
        @test grid_via_voxelize == grid_via_fold

        # 0-alloc fused fold for D=2 too. Same gotcha as D=3: hoist the
        # callback so @allocated doesn't time the closure's compile.
        sq2 = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        scalar = zeros(Float64, 4, 4)
        sum_first_2d = (acc, i, j, m) -> (@inbounds acc[i,j] += m[1]; acc)
        R3D.Flat.voxelize_fold!(sum_first_2d, scalar, sq2, (0,0), (4,4), d2, 0;
                                 workspace = ws)   # warmup
        fill!(scalar, 0.0)
        a = @allocated R3D.Flat.voxelize_fold!(sum_first_2d, scalar, sq2,
                                                (0,0), (4,4), d2, 0; workspace = ws)
        @test a <= ALLOC_TOLERANCE
    end

    @testset "Aligned cube tiles its grid exactly" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = (0.25, 0.25, 0.25)
        grid, lo, hi = R3D.Flat.voxelize(cube, d, 0)
        @test lo == (0,0,0)
        @test hi == (4,4,4)
        @test size(grid) == (1, 4, 4, 4)
        # Every voxel exactly d^3 = 1/64
        @test all(isapprox.(grid, 1/64; atol=1e-12))
        @test isapprox(sum(grid), 1.0; atol=1e-12)
    end

    @testset "Off-axis cube total mass = volume" begin
        cube = R3D.Flat.box((0.13, 0.27, 0.41), (1.62, 1.45, 1.78))
        for d in [(0.1, 0.1, 0.1), (0.2, 0.15, 0.07), (0.3, 0.3, 0.3)]
            grid, _, _ = R3D.Flat.voxelize(cube, d, 0)
            expected = (1.62-0.13) * (1.45-0.27) * (1.78-0.41)
            @test isapprox(sum(grid), expected; atol=1e-10)
        end
    end

    @testset "Order-1 grid sums recover the polytope's moments" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = (0.25, 0.25, 0.25)
        grid, _, _ = R3D.Flat.voxelize(cube, d, 1)
        # Sum over voxels gives ∫ x^a y^b z^c dV over the polytope
        @test isapprox(sum(@view grid[1, :, :, :]), 1.0;  atol=1e-12)
        @test isapprox(sum(@view grid[2, :, :, :]), 0.5;  atol=1e-12)
        @test isapprox(sum(@view grid[3, :, :, :]), 0.5;  atol=1e-12)
        @test isapprox(sum(@view grid[4, :, :, :]), 0.5;  atol=1e-12)
    end

    @testset "Single-voxel grid equals moments(poly)" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        # Choose d so the cube exactly fits in 1 voxel
        d = (1.0, 1.0, 1.0)
        for order in 0:3
            grid, _, _ = R3D.Flat.voxelize(cube, d, order)
            m = R3D.Flat.moments(cube, order)
            for k in 1:length(m)
                @test isapprox(grid[k, 1, 1, 1], m[k]; atol=1e-12)
            end
        end
    end

    @testset "Workspace reuse is allocation-free (after warmup)" begin
        cube = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = (0.25, 0.25, 0.25)
        order = 1
        nmom = R3D.num_moments(3, order)
        ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
        grid = zeros(Float64, nmom, 4, 4, 4)

        # warmup
        R3D.Flat.voxelize!(grid, cube, (0,0,0), (4,4,4), d, order; workspace=ws)
        fill!(grid, 0.0)
        a = @allocated R3D.Flat.voxelize!(grid, cube, (0,0,0), (4,4,4), d, order; workspace=ws)
        @test a <= ALLOC_TOLERANCE
        @test isapprox(sum(@view grid[1, :, :, :]), 1.0; atol=1e-12)
    end

    @testset "Differential vs C r3d_voxelize" begin
        if !HAVE_C
            @info "skipping voxelize differential; ENV[R3D_LIB] not set"
            return
        end
        rng = Random.MersenneTwister(20260427)
        cp, cbuf = R3D_C.new_poly()

        # Sweep over a range of polytope shapes and grid configurations.
        for trial in 1:50
            nplanes = rand(rng, 0:3)
            jl_pls = R3D.Plane{3,Float64}[]
            c_pls = R3D_C.Plane[]
            for _ in 1:nplanes
                v = randn(rng, 3); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.3 * randn(rng)
                push!(jl_pls, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd))
                push!(c_pls, R3D_C.Plane(R3D_C.RVec3(v[1], v[2], v[3]), dd))
            end

            cube_j = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            GC.@preserve cbuf R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                              R3D_C.RVec3(1.0,1.0,1.0))
            R3D.Flat.clip!(cube_j, jl_pls)
            GC.@preserve cbuf R3D_C.clip!(cp, c_pls)
            cube_j.nverts == 0 && continue

            d = (rand(rng)*0.2 + 0.05, rand(rng)*0.2 + 0.05, rand(rng)*0.2 + 0.05)
            order = rand(rng, 0:2)

            lo, hi = R3D.Flat.get_ibox(cube_j, d)
            lo_c, hi_c = GC.@preserve cbuf R3D_C.get_ibox(cp, d)
            @test lo == lo_c
            @test hi == hi_c
            (lo == hi) && continue   # skip degenerate empty grids

            ni = hi[1] - lo[1]; nj = hi[2] - lo[2]; nk = hi[3] - lo[3]
            nmom = R3D.num_moments(3, order)
            c_grid = zeros(Float64, ni*nj*nk*nmom)
            GC.@preserve cbuf c_grid R3D_C.voxelize!(c_grid, cp, lo, hi, d, order)

            j_grid, _, _ = R3D.Flat.voxelize(cube_j, d, order; ibox=(lo, hi))

            for i in 1:ni, j in 1:nj, k in 1:nk, m in 1:nmom
                c_idx = ((i-1)*nj*nk + (j-1)*nk + (k-1))*nmom + m
                @test isapprox(j_grid[m, i, j, k], c_grid[c_idx];
                               atol=1e-12, rtol=1e-10)
            end
        end
    end
end

# Need LinearAlgebra for the affine identity matrix in the testset below.
using LinearAlgebra: I

# Tier-2 polish tests: affine ops, init_tet/simplex/poly, AD compat,
# Base.show.
@testset "R3D.Flat new constructors and transforms" begin

    @testset "init_tet! D=3 closed forms" begin
        t = R3D.Flat.tet((0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0))
        @test t.nverts == 4
        @test R3D.Flat.is_good(t)
        @test isapprox(R3D.Flat.moments(t, 0)[1], 1/6; atol=1e-12)
    end

    @testset "init_simplex! D=2 closed forms" begin
        tri = R3D.Flat.simplex((0.0,0.0), (1.0,0.0), (0.0,1.0))
        @test tri.nverts == 3
        @test R3D.Flat.is_good(tri)
        @test isapprox(R3D.Flat.moments(tri, 0)[1], 0.5; atol=1e-12)
    end

    @testset "init_poly! D=2 (pentagon area via shoelace)" begin
        p = R3D.Flat.FlatPolytope{2,Float64}(64)
        verts = [(0.0,0.0), (1.0,0.0), (1.5,1.0), (0.5,1.5), (-0.5,1.0)]
        R3D.Flat.init_poly!(p, verts)
        @test p.nverts == 5
        @test R3D.Flat.is_good(p)
        @test isapprox(R3D.Flat.moments(p, 0)[1], 2.0; atol=1e-12)
    end

    @testset "init_poly! D=3 simple-case (tet) matches init_tet!" begin
        verts = [(0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
        faces = [[1,3,2], [1,2,4], [2,3,4], [1,4,3]]
        p = R3D.Flat.FlatPolytope{3,Float64}(64)
        R3D.Flat.init_poly!(p, verts, faces)
        @test p.nverts == 4
        @test R3D.Flat.is_good(p)
        @test isapprox(abs(R3D.Flat.moments(p, 0)[1]), 1/6; atol=1e-12)
    end

    @testset "Affine ops D=2 closed forms" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        R3D.Flat.translate!(sq, (10.0, 20.0))
        m = R3D.Flat.moments(sq, 1)
        @test m[1] ≈ 1.0
        @test m[2] / m[1] ≈ 10.5
        @test m[3] / m[1] ≈ 20.5

        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        R3D.Flat.scale!(sq, 3.0)
        @test R3D.Flat.moments(sq, 0)[1] ≈ 9.0

        sq = R3D.Flat.box((-0.5, -0.5), (0.5, 0.5))
        R3D.Flat.rotate!(sq, π/4)
        @test isapprox(R3D.Flat.moments(sq, 0)[1], 1.0; atol=1e-12)
        m = R3D.Flat.moments(sq, 1)
        @test isapprox(m[2] / m[1], 0.0; atol=1e-12)
        @test isapprox(m[3] / m[1], 0.0; atol=1e-12)

        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        R3D.Flat.shear!(sq, 0.5, 1, 2)
        @test isapprox(R3D.Flat.moments(sq, 0)[1], 1.0; atol=1e-12)

        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        R3D.Flat.affine!(sq, Float64[1 0 0; 0 1 0; 0 0 1])
        @test isapprox(R3D.Flat.moments(sq, 1)[2] / R3D.Flat.moments(sq, 0)[1], 0.5;
                       atol=1e-12)
    end

    @testset "Affine ops D=3 closed forms" begin
        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        R3D.Flat.translate!(cube, (1.0, 2.0, 3.0))
        m = R3D.Flat.moments(cube, 1)
        @test m[2] / m[1] ≈ 1.5
        @test m[3] / m[1] ≈ 2.5
        @test m[4] / m[1] ≈ 3.5

        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        R3D.Flat.scale!(cube, 2.0)
        @test R3D.Flat.moments(cube, 0)[1] ≈ 8.0

        cube = R3D.Flat.box((-0.5,-0.5,-0.5), (0.5,0.5,0.5))
        R3D.Flat.rotate!(cube, π/2, 3)
        @test isapprox(R3D.Flat.moments(cube, 0)[1], 1.0; atol=1e-12)

        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        R3D.Flat.shear!(cube, 0.3, 1, 3)
        @test isapprox(R3D.Flat.moments(cube, 0)[1], 1.0; atol=1e-12)

        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        R3D.Flat.affine!(cube, Matrix{Float64}(I, 4, 4))
        @test isapprox(R3D.Flat.moments(cube, 0)[1], 1.0; atol=1e-12)
    end

    @testset "Affine ops D ≥ 4" begin
        # Identity transform on a unit 4-cube must preserve volume.
        b4 = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        R3D.Flat.affine!(b4, Matrix{Float64}(I, 4, 4))
        @test isapprox(R3D.Flat.moments(b4, 0)[1], 1.0; atol=1e-12)

        # Axis-permutation matrix has det = ±1 and preserves volume of any
        # symmetric box. We use a cyclic shift (rows permuted) which has
        # det = +1 for D = 4 (even number of transpositions).
        b4 = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        Pmat = zeros(Float64, 4, 4)
        # Cyclic shift: row i gets column ((i mod 4) + 1).
        for i in 1:4
            Pmat[i, (i % 4) + 1] = 1.0
        end
        R3D.Flat.affine!(b4, Pmat)
        @test isapprox(R3D.Flat.moments(b4, 0)[1], 1.0; atol=1e-12)

        # 90° rotation in (x_1, x_2) plane, identity on (x_3, x_4). Acts on
        # a centered hypercube => volume invariant.
        b4 = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4, [-0.5, -0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5])
        Rmat = Matrix{Float64}(I, 4, 4)
        Rmat[1, 1] = 0.0; Rmat[1, 2] = -1.0
        Rmat[2, 1] = 1.0; Rmat[2, 2] =  0.0
        R3D.Flat.rotate!(b4, Rmat)
        @test isapprox(R3D.Flat.moments(b4, 0)[1], 1.0; atol=1e-12)

        # Anisotropic scaling via affine! with diag(2, 1, 1, 1) doubles vol.
        b4 = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        Dmat = Matrix{Float64}(I, 4, 4)
        Dmat[1, 1] = 2.0
        R3D.Flat.affine!(b4, Dmat)
        @test isapprox(R3D.Flat.moments(b4, 0)[1], 2.0; atol=1e-12)

        # rotate! must reject non-orthogonal matrices.
        b4 = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        bad = Matrix{Float64}(I, 4, 4); bad[1, 1] = 2.0
        @test_throws AssertionError R3D.Flat.rotate!(b4, bad)

        # D = 5: a coordinate cycle preserves volume on a unit 5-cube.
        b5 = R3D.Flat.FlatPolytope{5,Float64}(1 << 5)
        R3D.Flat.init_box!(b5, [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        P5 = zeros(Float64, 5, 5)
        for i in 1:5
            P5[i, (i % 5) + 1] = 1.0
        end
        R3D.Flat.affine!(b5, P5)
        @test isapprox(R3D.Flat.moments(b5, 0)[1], 1.0; atol=1e-12)

        # Steady-state allocation check: after a warmup, a single affine!
        # call on a 4-cube must not allocate.
        b4w = R3D.Flat.FlatPolytope{4,Float64}(1 << 4)
        R3D.Flat.init_box!(b4w, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        Mw = Matrix{Float64}(I, 4, 4)
        R3D.Flat.affine!(b4w, Mw)  # warmup
        @test (@allocated R3D.Flat.affine!(b4w, Mw)) <= ALLOC_TOLERANCE
    end

    @testset "Affine ops differential vs C r3d" begin
        if !HAVE_C
            @info "skipping affine differential; ENV[R3D_LIB] not set"
            return
        end
        rng = Random.MersenneTwister(20260434)
        cp, cbuf = R3D_C.new_poly()

        for trial in 1:50
            jl = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            GC.@preserve cbuf R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                              R3D_C.RVec3(1.0,1.0,1.0))

            shift = (randn(rng), randn(rng), randn(rng))
            R3D.Flat.translate!(jl, shift)
            GC.@preserve cbuf R3D_C.translate!(cp, R3D_C.RVec3(shift...))

            theta = randn(rng); axis = rand(rng, 1:3)
            R3D.Flat.rotate!(jl, theta, axis)
            GC.@preserve cbuf R3D_C.rotate!(cp, theta, axis)

            s = abs(randn(rng)) + 0.1
            R3D.Flat.scale!(jl, s)
            GC.@preserve cbuf R3D_C.scale!(cp, s)

            for ord in 0:2
                nm = R3D.num_moments(3, ord)
                m_jl = R3D.Flat.moments(jl, ord)
                m_c  = zeros(Float64, nm)
                GC.@preserve cbuf R3D_C.reduce!(cp, m_c, ord)
                for k in 1:nm
                    @test isapprox(m_jl[k], m_c[k]; atol=1e-9, rtol=1e-7)
                end
            end
        end

        # Affine matrix vs C
        for trial in 1:20
            jl = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            GC.@preserve cbuf R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                              R3D_C.RVec3(1.0,1.0,1.0))
            mat = Matrix{Float64}(I, 4, 4) + 0.1 * randn(rng, 4, 4)
            mat[4, 1:3] .= 0.0
            mat[4, 4] = 1.0
            R3D.Flat.affine!(jl, mat)
            GC.@preserve cbuf R3D_C.affine!(cp, mat)
            m_jl = R3D.Flat.moments(jl, 1)
            m_c  = zeros(Float64, R3D.num_moments(3, 1))
            GC.@preserve cbuf R3D_C.reduce!(cp, m_c, 1)
            for k in eachindex(m_jl)
                @test isapprox(m_jl[k], m_c[k]; atol=1e-9, rtol=1e-7)
            end
        end
    end

    @testset "ForwardDiff over clip+moment pipeline" begin
        # Allocate the polytope with element type T so ForwardDiff Dual
        # numbers propagate through it cleanly. The same pattern works
        # for any T <: Real (Float64, Float32, ForwardDiff.Dual, ...).
        function half_vol(p::T) where {T<:Real}
            cube = R3D.Flat.FlatPolytope{3,T}(64)
            R3D.Flat.init_box!(cube, T[0, 0, 0], T[1, 1, 1])
            plane = R3D.Plane{3,T}(R3D.Vec{3,T}(T[1, 0, 0]), -p)
            R3D.Flat.clip!(cube, [plane])
            return R3D.Flat.moments(cube, 0)[1]
        end
        g = ForwardDiff.derivative(half_vol, 0.5)
        # Closed-form: clipping the unit cube at x ≥ p keeps a slab of
        # volume (1-p), so dvol/dp = -1 exactly.
        @test isapprox(g, -1.0; atol=1e-10)

        # 2D centroid is a smooth function of plane offset away from
        # degeneracies — gradient should match finite-difference.
        function centroid_y_2d(p::T) where {T<:Real}
            sq = R3D.Flat.FlatPolytope{2,T}(64)
            R3D.Flat.init_box!(sq, T[0, 0], T[1, 1])
            plane = R3D.Plane{2,T}(R3D.Vec{2,T}(T[0, 1]), -p)
            R3D.Flat.clip!(sq, [plane])
            m = R3D.Flat.moments(sq, 1)
            return m[3] / m[1]
        end
        g2 = ForwardDiff.derivative(centroid_y_2d, 0.3)
        eps = 1e-6
        fd = (centroid_y_2d(0.3 + eps) - centroid_y_2d(0.3 - eps)) / (2eps)
        @test isapprox(g2, fd; atol=1e-6)
    end

    @testset "Phase-2 overlap-layer helpers (aabb / box_planes / volume / is_empty / copy!)" begin
        # aabb closed-form
        sq = R3D.Flat.box((0.13, 0.27), (1.62, 1.45))
        @test R3D.Flat.aabb(sq) == ((0.13, 0.27), (1.62, 1.45))
        cube = R3D.Flat.box((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0))
        @test R3D.Flat.aabb(cube) == ((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0))

        # aabb of an empty polytope returns identity tuples
        sq_empty = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        plane_far = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(1.0, 0.0), -100.0)
        R3D.Flat.clip!(sq_empty, [plane_far])
        @test R3D.Flat.is_empty(sq_empty)
        @test R3D.Flat.aabb(sq_empty) == ((Inf, Inf), (-Inf, -Inf))

        # 0-alloc check
        sq2 = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        cube2 = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        R3D.Flat.aabb(sq2); R3D.Flat.aabb(cube2)   # warmup
        @test (@allocated R3D.Flat.aabb(sq2)) <= ALLOC_TOLERANCE
        @test (@allocated R3D.Flat.aabb(cube2)) <= ALLOC_TOLERANCE

        # box_planes 2D + 3D
        ps2 = R3D.Flat.box_planes((0.0, 0.0), (1.0, 1.0))
        @test length(ps2) == 4
        ps3 = R3D.Flat.box_planes((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test length(ps3) == 6

        # Round-trip: clipping a larger box by another box's planes equals
        # the inner box (up to floating-point).
        big2 = R3D.Flat.box((-0.5, -0.5), (1.5, 1.5))
        R3D.Flat.clip!(big2, ps2)
        @test isapprox(R3D.Flat.moments(big2, 0)[1], 1.0; atol=1e-12)

        big3 = R3D.Flat.box((-0.5,-0.5,-0.5), (1.5,1.5,1.5))
        R3D.Flat.clip!(big3, ps3)
        @test isapprox(R3D.Flat.moments(big3, 0)[1], 1.0; atol=1e-12)

        # box_planes! 0-alloc
        out2 = Vector{R3D.Plane{2,Float64}}(undef, 4)
        R3D.Flat.box_planes!(out2, (0.0, 0.0), (1.0, 1.0))   # warmup
        @test (@allocated R3D.Flat.box_planes!(out2, (0.0, 0.0), (1.0, 1.0))) <= ALLOC_TOLERANCE
        out3 = Vector{R3D.Plane{3,Float64}}(undef, 6)
        R3D.Flat.box_planes!(out3, (0.0,0.0,0.0), (1.0,1.0,1.0))
        @test (@allocated R3D.Flat.box_planes!(out3, (0.0,0.0,0.0), (1.0,1.0,1.0))) <= ALLOC_TOLERANCE

        # is_empty
        sq3 = R3D.Flat.box((0.0,0.0), (1.0,1.0))
        @test !R3D.Flat.is_empty(sq3)
        R3D.Flat.clip!(sq3, [plane_far])
        @test R3D.Flat.is_empty(sq3)

        # volume — matches moments(., 0)[1] exactly, 0-alloc after warmup
        sq4 = R3D.Flat.box((0.0, 0.0), (3.0, 5.0))
        @test R3D.Flat.volume(sq4) == 15.0
        R3D.Flat.volume(sq4)   # warmup
        @test (@allocated R3D.Flat.volume(sq4)) <= ALLOC_TOLERANCE
        cube4 = R3D.Flat.box((0.0,0.0,0.0), (1.0,2.0,3.0))
        R3D.Flat.volume(cube4)
        @test R3D.Flat.volume(cube4) == 6.0
        @test (@allocated R3D.Flat.volume(cube4)) <= ALLOC_TOLERANCE

        # volume on empty polytope
        @test R3D.Flat.volume(sq3) == 0.0

        # copy! 0-alloc, faithful
        src2 = R3D.Flat.box((1.0, 2.0), (4.0, 7.0))
        dst2 = R3D.Flat.FlatPolytope{2,Float64}(64)
        R3D.Flat.copy!(dst2, src2)
        @test dst2.nverts == src2.nverts
        @test isapprox(R3D.Flat.moments(dst2, 0)[1], 15.0; atol=1e-12)
        @test (@allocated R3D.Flat.copy!(dst2, src2)) <= ALLOC_TOLERANCE

        src3 = R3D.Flat.box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        dst3 = R3D.Flat.FlatPolytope{3,Float64}(64)
        R3D.Flat.copy!(dst3, src3)
        @test dst3.nverts == src3.nverts
        @test isapprox(R3D.Flat.moments(dst3, 0)[1], 8.0; atol=1e-12)
        @test (@allocated R3D.Flat.copy!(dst3, src3)) <= ALLOC_TOLERANCE
    end

    @testset "R3D.Flat facet ((D−1)-face) tracking" begin
        # Box: 2D facets, each shared by 2^(D-1) vertices.
        for D in 4:6
            box = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(box, zeros(D), ones(D))
            @test box.nfacets == 2D
            counts = zeros(Int, box.nfacets)
            for v in 1:box.nverts, k in 1:D
                counts[box.facets[k, v]] += 1
            end
            # Every (vertex, slot) entry contributes 1 occurrence; total
            # = nverts × D = 2^D × D = 2D × 2^(D-1) ⇒ each facet
            # appears in exactly 2^(D-1) vertices.
            @test all(counts .== 2^(D - 1))
            @test all(box.facets[k, v] != 0 for v in 1:box.nverts, k in 1:D)
            # walk_facets visits each ID exactly once.
            visited = Set{Int}()
            R3D.Flat.walk_facets(box) do fid; push!(visited, fid); end
            @test visited == Set(1:2D)
            # walk_facet_vertices returns the right cardinality.
            for fid in 1:2D
                count = 0
                R3D.Flat.walk_facet_vertices(box, fid) do v; count += 1; end
                @test count == 2^(D - 1)
            end
        end

        # Simplex: D+1 facets, each shared by D vertices (everyone except
        # the vertex it's opposite).
        for D in 4:6
            verts = [ntuple(j -> j == i ? 1.0 : 0.0, D) for i in 0:D]
            sim = R3D.Flat.FlatPolytope{D,Float64}(64)
            R3D.Flat.init_simplex!(sim, verts)
            @test sim.nfacets == D + 1
            counts = zeros(Int, sim.nfacets)
            for v in 1:sim.nverts, k in 1:D
                counts[sim.facets[k, v]] += 1
            end
            @test all(counts .== D)
            visited = Set{Int}()
            R3D.Flat.walk_facets(sim) do fid; push!(visited, fid); end
            @test visited == Set(1:(D + 1))
        end

        # Single clip on a D=4 unit box: facet count = 2D + 1 = 9. The
        # new facet ID = 9 appears in exactly the new vertices' slot 1.
        buf = R3D.Flat.FlatPolytope{4,Float64}(64)
        R3D.Flat.init_box!(buf, zeros(4), ones(4))
        nverts_pre = buf.nverts
        R3D.Flat.clip!(buf, [R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(1.0,0,0,0), -0.5)])
        @test buf.nfacets == 9
        cut_count = sum(buf.facets[1, v] == 9 for v in 1:buf.nverts)
        # The kept-side edge (slot 1) of each NEW vertex points back
        # at an original kept vertex; the facet opposite that edge is
        # the cut. Half of the post-clip vertices are new (the cut
        # creates one new vertex per cut edge ≈ half the box's vertices).
        @test cut_count > 0
        # Cut facet's vertex set: all newly-inserted vertices.
        cut_verts = Int[]
        R3D.Flat.walk_facet_vertices(buf, 9) do v; push!(cut_verts, v); end
        @test length(cut_verts) == 8   # the 8 new vertices on the x[1]=0.5 cut

        # Sequential clips: 3 orthogonal cuts at D = 4, 5, 6 → nfacets += 3.
        for D in 4:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            base = buf.nfacets
            for axis in 1:3
                R3D.Flat.clip!(buf, [R3D.Plane{D,Float64}(
                    R3D.Vec{D,Float64}(ntuple(k -> k == axis ? 1.0 : 0.0, D)),
                    -0.5)])
                @test buf.nfacets == base + axis
            end
        end
    end

    @testset "Phase A: D = 4 Lasserre moments (P ≥ 1) closed-form validation" begin
        # Closed-form unit D-simplex moments: ∫_{Δ_D} x^α dV = α! / (D + |α|)!
        # Closed-form unit D-box moments:     ∫_{[0,1]^D} x^α dV = ∏ 1 / (α_j + 1)
        factorial_int(n) = prod(1:n; init = 1)
        function expected_simplex(α::NTuple{4,Int})
            num = prod(factorial_int(a) for a in α; init = 1)
            return num / factorial_int(4 + sum(α))
        end
        expected_box(α::NTuple{4,Int}) = prod(1.0 / (a + 1) for a in α; init = 1.0)

        # Up through P = 3 — all 35 multi-indices must match.
        for P in 1:3
            sim = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                    (1.0, 0.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0, 0.0),
                                    (0.0, 0.0, 1.0, 0.0),
                                    (0.0, 0.0, 0.0, 1.0))
            m_sim = R3D.Flat.moments(sim, P)
            box = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
            m_box = R3D.Flat.moments(box, P)
            for (i, α) in enumerate(R3D.Flat._enumerate_moments_d4(P))
                @test isapprox(m_sim[i], expected_simplex(α); atol = 1e-10)
                @test isapprox(m_box[i], expected_box(α); atol = 1e-10)
            end
        end

        # Coordinate-permutation symmetry on the unit hypercube: any
        # moment with α invariant under a coordinate permutation must
        # equal the moment with the permuted α.
        let
            box = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
            m = R3D.Flat.moments(box, 3)
            alphas = R3D.Flat._enumerate_moments_d4(3)
            idx = Dict(α => i for (i, α) in enumerate(alphas))
            for α in alphas
                # Try a coordinate swap (1, 2): if α has α[1] != α[2],
                # the swapped α is a different multi-index and must
                # have the same moment value (since the unit hypercube
                # is invariant under that swap).
                α_swap = (α[2], α[1], α[3], α[4])
                @test isapprox(m[idx[α]], m[idx[α_swap]]; atol = 1e-12)
            end
        end

        # Clipped simplex: ∫ x_1 dV checked against analytic decomposition.
        # P_clip = unit-D=4 simplex ∩ {x_1 ≤ 0.5}. By inclusion-exclusion:
        #   vol(P_clip) = 1/24 - vol(corner)
        #   ∫_{P_clip} x_1 dV = ∫_simplex x_1 - ∫_corner x_1
        # corner = {x_1 ∈ [0.5, 1], rest in (1/2)-scaled simplex translate}.
        # Substituting u = (x - 0.5*e_1)/0.5 (so x_1 = 0.5 + u_1/2, others
        # = u_i/2), the corner integral closes via standard simplex
        # moments: vol(corner) = 1/384, ∫_corner x_1 = 1/640.
        let
            poly = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                     (1.0, 0.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0, 0.0),
                                     (0.0, 0.0, 1.0, 0.0),
                                     (0.0, 0.0, 0.0, 1.0))
            R3D.Flat.clip!(poly,
                R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(-1.0, 0.0, 0.0, 0.0), 0.5))
            m = R3D.Flat.moments(poly, 1)
            @test isapprox(m[1], 1/24 - 1/384; atol = 1e-12)
            @test isapprox(m[2], 1/120 - 1/640; atol = 1e-10)
            # By symmetry under (x_2, x_3, x_4) permutation, the
            # other three first-moments are equal.
            @test isapprox(m[3], m[4]; atol = 1e-12)
            @test isapprox(m[3], m[5]; atol = 1e-12)
        end

        # Voxelize-fold consistency: per-cell moment sum equals whole
        # polytope moment to fp precision.
        let
            poly = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                     (1.0, 0.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0, 0.0),
                                     (0.0, 0.0, 1.0, 0.0),
                                     (0.0, 0.0, 0.0, 1.0))
            whole = R3D.Flat.moments(poly, 2)
            ws = R3D.Flat.VoxelizeWorkspace{4,Float64}(64)
            n = 2
            d_grid  = ntuple(_ -> 1.0 / n, Val(4))
            ibox_lo = ntuple(_ -> 0,       Val(4))
            ibox_hi = ntuple(_ -> n,       Val(4))
            sum_cells = zeros(Float64, length(whole))
            R3D.Flat.voxelize_fold!(sum_cells, poly, ibox_lo, ibox_hi, d_grid,
                                    2; workspace = ws) do acc, idx, m
                @inbounds for k in eachindex(acc)
                    acc[k] += m[k]
                end
                acc
            end
            for k in eachindex(whole)
                @test isapprox(sum_cells[k], whole[k]; atol = 1e-10, rtol = 1e-10)
            end
        end
    end

    @testset "R3D.Flat facet normals + signed distances (D ≥ 4, Phase A foundation)" begin
        # Box facets: outward axis-aligned, signed-distance matches
        # the corresponding lo/hi coordinate.
        for D in 4:6
            lo = ntuple(k -> 0.1 * k, D)
            hi = ntuple(k -> 1.0 + 0.1 * k, D)
            buf = R3D.Flat.FlatPolytope{D,Float64}(256)
            R3D.Flat.init_box!(buf, [lo...], [hi...])
            for k in 1:D
                # Facet 2k-1: x[k] = lo[k], outward normal -e_k, d = -lo[k]
                @test buf.facet_normals[k, 2k - 1]   == -1.0
                @test buf.facet_distances[2k - 1]    == -lo[k]
                for j in 1:D
                    j == k && continue
                    @test buf.facet_normals[j, 2k - 1] == 0.0
                end
                # Facet 2k: x[k] = hi[k], outward normal +e_k, d = hi[k]
                @test buf.facet_normals[k, 2k]       ==  1.0
                @test buf.facet_distances[2k]        ==  hi[k]
            end
            # Sanity: every facet vertex satisfies n·v == d; non-facet
            # vertices satisfy n·v < d (outward).
            for f in 1:buf.nfacets, v in 1:buf.nverts
                np = sum(buf.facet_normals[k, f] * buf.positions[k, v] for k in 1:D)
                on_facet = any(k -> buf.facets[k, v] == f, 1:D)
                if on_facet
                    @test isapprox(np, buf.facet_distances[f]; atol = 1e-12)
                else
                    @test np < buf.facet_distances[f] - 1e-12
                end
            end
        end

        # Simplex facets: opposite vertex u, outward unit normal computed
        # via Gram-Schmidt against the D − 1 in-facet edges.
        for D in 4:6
            verts = [ntuple(j -> j == i ? 1.0 : 0.0, D) for i in 0:D]
            sim = R3D.Flat.FlatPolytope{D,Float64}(64)
            R3D.Flat.init_simplex!(sim, verts)
            for f in 1:sim.nfacets
                # Unit length
                norm2 = sum(sim.facet_normals[k, f]^2 for k in 1:D)
                @test isapprox(norm2, 1.0; atol = 1e-12)
                # On-facet vertices satisfy n·v == d; the opposite vertex
                # (vertex f, since facet u is opposite vertex u) is
                # strictly inside the kept half-space (n·v < d).
                for v in 1:sim.nverts
                    np = sum(sim.facet_normals[k, f] * sim.positions[k, v] for k in 1:D)
                    on_facet = any(k -> sim.facets[k, v] == f, 1:D)
                    if on_facet
                        @test isapprox(np, sim.facet_distances[f]; atol = 1e-12)
                    else
                        @test np < sim.facet_distances[f] - 1e-12
                    end
                end
            end
            # Spot check the unit D-simplex (origin + D unit basis vectors):
            # the facet OPPOSITE the origin (vertex 1 in our 1-indexed
            # convention) is `x_1 + x_2 + ... + x_D = 1`, so its outward
            # unit normal is (1,1,…,1)/√D and d = 1/√D.
            invsqrtD = 1.0 / sqrt(Float64(D))
            for k in 1:D
                @test isapprox(sim.facet_normals[k, 1], invsqrtD; atol = 1e-12)
            end
            @test isapprox(sim.facet_distances[1], invsqrtD; atol = 1e-12)
        end

        # `clip!` propagation: a single cut on a unit D-box adds one
        # facet whose stored (n, d) matches the clip plane's
        # outward-normal-of-discarded-side / signed-distance pair.
        # For `n·x + d ≥ 0` (kept set), the cut facet's outward
        # normal is `-n` and its signed distance is `d` (since the
        # facet sits at `n·x = -d`, equivalently `(-n)·x = d`).
        for D in 4:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(256)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            n = R3D.Vec{D,Float64}(ntuple(k -> k == 1 ? 1.0 : 0.0, D))
            R3D.Flat.clip!(buf, [R3D.Plane{D,Float64}(n, -0.5)])
            new_id = buf.nfacets
            # Outward normal points to the discarded side (x[1] < 0.5),
            # i.e. `-e_1`. Signed distance of facet at x[1] = 0.5 in
            # outward direction: (-1) * 0.5 = -0.5. That matches
            # plane.d = -0.5 by our convention.
            @test buf.facet_normals[1, new_id] == -1.0
            for k in 2:D
                @test buf.facet_normals[k, new_id] == 0.0
            end
            @test buf.facet_distances[new_id] == -0.5
            # Sanity: all post-clip vertices satisfy
            # outward_n · v ≤ d_new (kept side).
            for v in 1:buf.nverts
                np = sum(buf.facet_normals[k, new_id] * buf.positions[k, v]
                         for k in 1:D)
                @test np <= buf.facet_distances[new_id] + 1e-12
            end
        end

        # `_copy_polytope_nd!` carries facet metadata across copies
        # (used by voxelize_fold!'s two-clips-per-split pattern).
        src = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
        R3D.Flat.clip!(src, [R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(1.0, 0, 0, 0), -0.5)])
        dst = R3D.Flat.FlatPolytope{4,Float64}(256)
        R3D.Flat.copy!(dst, src)
        @test dst.nfacets == src.nfacets
        for f in 1:src.nfacets
            for k in 1:4
                @test dst.facet_normals[k, f] == src.facet_normals[k, f]
            end
            @test dst.facet_distances[f] == src.facet_distances[f]
        end
    end

    @testset "R3D.Flat ND (D ≥ 4) — clip + 0th-moment closed forms + diff vs C" begin
        # Closed-form: unit D-simplex volume = 1/D!, unit D-box volume = 1.
        for D in 4:6
            sim = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_simplex!(sim,
                [ntuple(j -> j == i ? 1.0 : 0.0, D) for i in 0:D])
            @test isapprox(R3D.Flat.volume(sim), 1 / factorial(D);
                           atol=1e-12, rtol=1e-10)

            box = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(box, zeros(D), ones(D))
            @test isapprox(R3D.Flat.volume(box), 1.0; atol=1e-12, rtol=1e-10)

            # Clip with x[1] ≥ 0.5 → slab volume 0.5
            R3D.Flat.clip!(box, [R3D.Plane{D,Float64}(
                R3D.Vec{D,Float64}(ntuple(i -> i == 1 ? 1.0 : 0.0, D)), -0.5)])
            @test isapprox(R3D.Flat.volume(box), 0.5; atol=1e-12, rtol=1e-10)
        end

        # D = 4 order ≥ 1 now lands via Lasserre (Phase A). D = 5 / D = 6
        # still require additional codim-face tracking layers and stay
        # stubbed with an informative error.
        buf4 = R3D.Flat.FlatPolytope{4,Float64}(64)
        R3D.Flat.init_box!(buf4, zeros(4), ones(4))
        m4 = R3D.Flat.moments(buf4, 1)
        @test length(m4) == R3D.num_moments(4, 1)
        @test isapprox(m4[1], 1.0; atol = 1e-12)   # zeroth moment = vol = 1
        # First moments of unit hypercube = 0.5 each.
        for i in 2:5
            @test isapprox(m4[i], 0.5; atol = 1e-12)
        end
        for D in 5:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(64)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            @test_throws ErrorException R3D.Flat.moments(buf, 1)
            @test_throws ErrorException R3D.Flat.moments!(zeros(D + 1), buf, 1)
        end

        # Differential vs C (only when the per-dimension rNd library is loaded).
        if HAVE_C && !isempty(R3D_C.libr3d_4d[])
            rng = Random.MersenneTwister(20260425)
            for (D, new_poly_fn, init_box_fn, clip_fn, reduce_fn, RVecD, PlaneD) in [
                (4, R3D_C.new_poly4, R3D_C.init_box4!, R3D_C.clip4!,
                 R3D_C.reduce4!, R3D_C.RVec4, R3D_C.Plane4),
            ]
                cp, cbuf = new_poly_fn()
                for trial in 1:100
                    v = randn(rng, D); v ./= sqrt(sum(v.^2))
                    dd = -sum(v) / 2 + 0.3 * randn(rng)

                    bufjl = R3D.Flat.FlatPolytope{D,Float64}(256)
                    R3D.Flat.init_box!(bufjl, zeros(D), ones(D))
                    R3D.Flat.clip!(bufjl, [
                        R3D.Plane{D,Float64}(R3D.Vec{D,Float64}(v), dd)])

                    GC.@preserve cbuf begin
                        init_box_fn(cp,
                            RVecD(ntuple(_ -> 0.0, D)),
                            RVecD(ntuple(_ -> 1.0, D)))
                        clip_fn(cp, [PlaneD(RVecD(ntuple(i -> v[i], D)), dd)])
                        out = zeros(Float64, 1)
                        reduce_fn(cp, out, 0)
                        v_jl = bufjl.nverts > 0 ? R3D.Flat.volume(bufjl) : 0.0
                        @test isapprox(v_jl, out[1]; atol=1e-12, rtol=1e-10)
                    end
                end
            end
        else
            @info "skipping D≥4 differential tests; ENV[\"R3D_LIB_4D\"] etc. not set"
        end
    end

    @testset "voxelize_fold! / voxelize! for D ≥ 4 (order = 0)" begin
        # Closed-form: unit D-box voxelized over an N^D grid sums to 1
        # and every cell equals 1 / N^D.
        for (D, N) in [(4, 4), (5, 2), (6, 2)]
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            ws = R3D.Flat.VoxelizeWorkspace{D,Float64}(128)
            d = ntuple(_ -> 1.0 / N, D)
            ibox_lo = ntuple(_ -> 0, D)
            ibox_hi = ntuple(_ -> N, D)
            grid = zeros(Float64, 1, ntuple(_ -> N, D)...)
            R3D.Flat.voxelize!(grid, buf, ibox_lo, ibox_hi, d, 0; workspace = ws)
            cell_vol = inv(N^D)
            @test isapprox(sum(grid), 1.0; atol=1e-12, rtol=1e-10)
            @test all(isapprox.(grid, cell_vol; atol=1e-12, rtol=1e-10))
        end

        # D = 4 voxelize_fold! at order ≥ 1 lands via Lasserre. D = 5 / D = 6
        # still throw.
        buf4 = R3D.Flat.FlatPolytope{4,Float64}(64)
        R3D.Flat.init_box!(buf4, zeros(4), ones(4))
        ws4 = R3D.Flat.VoxelizeWorkspace{4,Float64}(64)
        # Sum-over-voxels of order-1 moments == whole-polytope order-1 moment.
        whole = R3D.Flat.moments(buf4, 1)
        sum_over_cells = zeros(Float64, 5)
        R3D.Flat.voxelize_fold!(sum_over_cells, buf4, (0,0,0,0),
            (4,4,4,4), (0.25,0.25,0.25,0.25), 1; workspace = ws4) do acc, idx, m
            @inbounds for k in 1:5
                acc[k] += m[k]
            end
            acc
        end
        for k in 1:5
            @test isapprox(sum_over_cells[k], whole[k]; atol = 1e-10, rtol = 1e-10)
        end
        # D = 5 / D = 6 still throw on order ≥ 1 (no Lasserre yet).
        for D in 5:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(64)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            ws = R3D.Flat.VoxelizeWorkspace{D,Float64}(64)
            d_grid = ntuple(_ -> 0.5, D)
            ibox_lo = ntuple(_ -> 0, D)
            ibox_hi = ntuple(_ -> 2, D)
            @test_throws AssertionError R3D.Flat.voxelize_fold!(0.0, buf, ibox_lo,
                ibox_hi, d_grid, 1; workspace = ws) do acc, idx, m
                acc + m[1]
            end
        end
    end

    @testset "nD clip! / voxelize_fold! finalization (D ≥ 4)" begin
        # 1) Single-plane `clip!` overload (no `[plane]` array allocation).
        #    Result must agree with the multi-plane API and with the
        #    private `clip_plane!` helper.
        for D in 4:6
            buf_a = R3D.Flat.FlatPolytope{D,Float64}(128)
            buf_b = R3D.Flat.FlatPolytope{D,Float64}(128)
            buf_c = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf_a, zeros(D), ones(D))
            R3D.Flat.init_box!(buf_b, zeros(D), ones(D))
            R3D.Flat.init_box!(buf_c, zeros(D), ones(D))

            n_axis = R3D.Vec{D,Float64}(ntuple(k -> k == 1 ? 1.0 : 0.0, D))
            plane = R3D.Plane{D,Float64}(n_axis, -0.5)

            R3D.Flat.clip!(buf_a, [plane])               # array form
            R3D.Flat.clip!(buf_b, plane)                 # single-plane overload
            R3D.Flat.clip_plane!(buf_c, plane)           # internal helper
            @test buf_a.nverts == buf_b.nverts == buf_c.nverts
            @test isapprox(R3D.Flat.volume(buf_a), R3D.Flat.volume(buf_b);
                           atol=1e-14, rtol=1e-12)
            @test isapprox(R3D.Flat.volume(buf_a), R3D.Flat.volume(buf_c);
                           atol=1e-14, rtol=1e-12)
            @test isapprox(R3D.Flat.volume(buf_b), 0.5; atol=1e-12)
        end

        # 2) `voxelize` allocating wrapper for D ≥ 4 — agrees with the
        #    in-place `voxelize!` and gives sum == polytope volume.
        for D in 4:5
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            d_grid = ntuple(_ -> 0.5, D)
            grid_alloc, lo, hi = R3D.Flat.voxelize(buf, d_grid, 0)
            @test lo == ntuple(_ -> 0, D)
            @test hi == ntuple(_ -> 2, D)
            @test isapprox(sum(grid_alloc), 1.0; atol=1e-12)

            ws = R3D.Flat.VoxelizeWorkspace{D,Float64}(128)
            grid_inplace = zeros(Float64, 1, ntuple(_ -> 2, D)...)
            R3D.Flat.voxelize!(grid_inplace, buf, lo, hi, d_grid, 0; workspace = ws)
            @test grid_alloc ≈ grid_inplace
        end

        # 3) Non-axis-aligned-clipped box voxelizes consistently with
        #    `volume`: sum of leaf cell volumes must match the polytope's
        #    own moment-based volume to floating-point precision. This
        #    is the strongest correctness check that doesn't depend on a
        #    closed form — any bug in the bisection loop shows up here.
        for D in 4:5
            rng = Random.MersenneTwister(20260426 + D)
            for trial in 1:5
                buf = R3D.Flat.FlatPolytope{D,Float64}(256)
                R3D.Flat.init_box!(buf, zeros(D), ones(D))
                # Random oblique cut that retains a non-trivial fraction.
                v = randn(rng, D); v ./= sqrt(sum(v.^2))
                dd = -0.6 * sum(v) / D + 0.05 * randn(rng)
                ok = R3D.Flat.clip!(buf,
                    R3D.Plane{D,Float64}(R3D.Vec{D,Float64}(v), dd))
                @test ok
                buf.nverts == 0 && continue   # empty after clip; nothing to check
                vol_direct = R3D.Flat.volume(buf)

                d_grid = ntuple(_ -> 0.5, D)
                grid, lo, hi = R3D.Flat.voxelize(buf, d_grid, 0)
                vol_voxel = sum(grid)
                @test isapprox(vol_voxel, vol_direct; atol=1e-12, rtol=1e-10)
            end
        end

        # 4) D-simplex voxelization sums to closed-form 1/D!.
        for D in 4:5
            sim = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_simplex!(sim,
                [ntuple(j -> j == i ? 1.0 : 0.0, D) for i in 0:D])
            d_grid = ntuple(_ -> 0.5, D)
            grid, _, _ = R3D.Flat.voxelize(sim, d_grid, 0)
            @test isapprox(sum(grid), 1 / factorial(D); atol=1e-12, rtol=1e-10)
        end

        # 5) `voxelize_fold!` hot loop is allocation-free for D ≥ 4 once
        #    the workspace is warm. The `[plane_neg]` / `[plane_pos]`
        #    array allocations the previous implementation paid per leaf
        #    have been removed via the single-plane `clip!` overload.
        D = 4
        buf = R3D.Flat.FlatPolytope{D,Float64}(128)
        R3D.Flat.init_box!(buf, zeros(D), ones(D))
        ws = R3D.Flat.VoxelizeWorkspace{D,Float64}(128)
        d_grid = ntuple(_ -> 0.25, D)
        ibox_lo = ntuple(_ -> 0, D)
        ibox_hi = ntuple(_ -> 4, D)

        # Wrap in a function so the closure type stays the same across
        # warmup and measurement — each toplevel `do ... end` is a
        # distinct anonymous-function type, which would otherwise
        # trigger fresh compilation inside `@allocated` and dominate
        # the measurement (millions of bytes of compiler internals).
        function _vfold_sumvol(buf, ibox_lo, ibox_hi, d_grid, ws)
            R3D.Flat.voxelize_fold!(0.0, buf, ibox_lo, ibox_hi, d_grid,
                                    0; workspace = ws) do acc, idx, m
                acc + m[1]
            end
        end
        # warmup pass — first call may allocate workspace stack growth /
        # moment scratch resize; we measure the second call.
        _vfold_sumvol(buf, ibox_lo, ibox_hi, d_grid, ws)
        a = @allocated _vfold_sumvol(buf, ibox_lo, ibox_hi, d_grid, ws)
        # Hot loop is now fully heap-free: the per-leaf `[plane]`
        # array allocations were removed via the single-plane `clip!`
        # overload, and the LTD scratch backing the per-leaf
        # `_reduce_nd_zeroth!` call lives on the polytope itself.
        @test a <= ALLOC_TOLERANCE
    end

    @testset "D ≥ 4 sequential clips don't corrupt finds[][]" begin
        # Regression for the linker bug found while wiring voxelize_fold!:
        # earlier the inside-loop branch incremented `nfaces` per cross
        # instead of per walk, leaving finds[][] in a state that broke
        # the next clip. Three sequential clips reduce volume by 2× each
        # — so vol = 1 → 0.5 → 0.25 → 0.125 across three orthogonal
        # half-space clips of a unit D-box.
        for D in 4:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            for axis in 1:3
                R3D.Flat.clip!(buf, [R3D.Plane{D,Float64}(
                    R3D.Vec{D,Float64}(ntuple(k -> k == axis ? 1.0 : 0.0, D)),
                    -0.5)])
                @test isapprox(R3D.Flat.volume(buf), 1.0 / 2^axis;
                               atol=1e-12, rtol=1e-10)
            end
        end
    end

    @testset "D ≥ 4 split_coord! API parity + voxelize_fold! integration" begin
        # split_coord! at D ≥ 4 — public API parity with the lower-D
        # versions. Internally a wrapper over `clip_plane!` (Phase B.1).
        # voxelize_fold! D ≥ 4 now uses this wrapper, so any bug here
        # would fail the existing voxelize tests too.
        for D in 4:6
            poly = R3D.Flat.box(ntuple(_ -> 0.0, Val(D)),
                                ntuple(_ -> 1.0, Val(D)))
            out0 = R3D.Flat.FlatPolytope{D,Float64}(256)
            out1 = R3D.Flat.FlatPolytope{D,Float64}(256)
            ok = R3D.Flat.split_coord!(poly, out0, out1, 0.5, 1)
            @test ok
            @test isapprox(R3D.Flat.volume(out0), 0.5; atol = 1e-12)
            @test isapprox(R3D.Flat.volume(out1), 0.5; atol = 1e-12)
            @test isapprox(R3D.Flat.volume(out0) + R3D.Flat.volume(out1),
                           1.0; atol = 1e-12)
        end

        # Aliasing pattern: in === out0. Same result, no extra copy.
        let
            poly = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
            out1 = R3D.Flat.FlatPolytope{4,Float64}(256)
            ok = R3D.Flat.split_coord!(poly, poly, out1, 0.25, 3)
            @test ok
            # poly itself is now the "x_3 ≤ 0.25" half.
            @test isapprox(R3D.Flat.volume(poly), 0.25; atol = 1e-12)
            @test isapprox(R3D.Flat.volume(out1), 0.75; atol = 1e-12)
        end

        # Simplex split: volume conservation.
        for c in (0.5, 1/3, 0.25, 0.7)
            sim = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                    (1.0, 0.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0, 0.0),
                                    (0.0, 0.0, 1.0, 0.0),
                                    (0.0, 0.0, 0.0, 1.0))
            out0 = R3D.Flat.FlatPolytope{4,Float64}(256)
            out1 = R3D.Flat.FlatPolytope{4,Float64}(256)
            R3D.Flat.split_coord!(sim, out0, out1, c, 1)
            @test isapprox(R3D.Flat.volume(out0) + R3D.Flat.volume(out1),
                           1/24; rtol = 1e-12)
        end

        # Empty input returns two empty halves.
        let
            empty4 = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
            far = R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(1.0, 0, 0, 0), -100.0)
            R3D.Flat.clip!(empty4, far)
            @test empty4.nverts == 0
            out0 = R3D.Flat.FlatPolytope{4,Float64}(64)
            out1 = R3D.Flat.FlatPolytope{4,Float64}(64)
            R3D.Flat.split_coord!(empty4, out0, out1, 0.5, 1)
            @test out0.nverts == 0
            @test out1.nverts == 0
        end
    end

    @testset "D ≥ 4 clip! handles FP-boundary clipped vertex (sd_vnext ≈ 0)" begin
        # Regression for the symmetric case of the c4496ab boundary-vertex
        # bug: a previously-interpolated vertex's coordinate may exactly
        # match a later clip plane, but FP rounding places it 1 ULP on
        # the clipped side (sd_vnext ≈ -eps). The cut-position formula
        # then collapses the new vertex onto vnext, breaking the
        # simple-polytope invariant the LTD recursion assumes.
        # Manifests at D = 4 when 4 axis-aligned clips at irrational
        # constants (e.g. 1/3) accumulate FP noise.
        for c in (1/3, 1/3 + 1e-9, 1/3 - 1e-9, 0.333, 0.4)
            poly = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                     (1.0, 0.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0, 0.0),
                                     (0.0, 0.0, 1.0, 0.0),
                                     (0.0, 0.0, 0.0, 1.0))
            for k in 1:4
                nrm = ntuple(j -> j == k ? -1.0 : 0.0, Val(4))
                R3D.Flat.clip!(poly,
                    R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(nrm), c))
            end
            # Closed-form volume of simplex ∩ [0, c]^4 (D = 4):
            #   vol = sum_{k=0}^{floor(1/c)} (-1)^k C(4,k) (1 - k*c)^4 / 4!
            expected = sum((-1)^k * binomial(4, k) * max(0, 1 - k * c)^4
                           for k in 0:4) / 24
            # Relative tolerance reflects that the ε-nudge fix produces
            # O(eps(T)*scale) error per cut and 4 cuts compound to ~1e-7.
            # Pre-fix, the c=1/3 case returned ~0 here.
            @test isapprox(R3D.Flat.volume(poly), expected; rtol = 1e-7)
        end

        # Voxelize-fold across small grid sizes — at non-power-of-2
        # n the bisection produces clip planes at coordinates that
        # coincide with previously-interpolated vertices, hitting the
        # symmetric boundary case. Pre-fix, n = 3 gave 5 % of expected
        # total. Tolerance reflects accumulated ε-nudge error: each
        # nudge perturbs a cell's volume by O(eps(T)*scale), and the
        # bisection produces O(n^D) cells, giving total error
        # ~ n^D * eps. Relative-tolerance budget 1e-3 covers up to
        # n = 8 with margin to spare.
        ws = R3D.Flat.VoxelizeWorkspace{4,Float64}(64)
        for n in 2:8
            poly = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                                     (1.0, 0.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0, 0.0),
                                     (0.0, 0.0, 1.0, 0.0),
                                     (0.0, 0.0, 0.0, 1.0))
            d_grid  = ntuple(_ -> 1.0 / n, Val(4))
            ibox_lo = ntuple(_ -> 0,       Val(4))
            ibox_hi = ntuple(_ -> n,       Val(4))
            total = R3D.Flat.voxelize_fold!(0.0, poly, ibox_lo, ibox_hi,
                                            d_grid, 0; workspace = ws) do acc, idx, m
                acc + m[1]
            end
            @test isapprox(total, 1/24; rtol = 1e-3)
        end
    end

    @testset "D ≥ 4 sequential simplex clips preserve volume + symmetry" begin
        # Regression for the boundary-vertex degenerate cut bug: when a
        # second clip plane passes exactly through a vertex created by
        # the first clip (sdists[vcur] == 0), the cut-position formula
        # collapsed to vcur, creating duplicate vertices that broke the
        # LTD moments recursion. Repro from the bug report: two-axis
        # quadrant decomposition of a unit D = 4 simplex must respect
        # coordinate symmetry and sum to 1/24.
        function clip2_simplex(planes)
            poly = R3D.Flat.FlatPolytope{4, Float64}(512)
            verts = [[0.0, 0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]
            R3D.Flat.init_simplex!(poly, verts)
            for p in planes
                R3D.Flat.clip!(poly, p)
            end
            R3D.Flat.volume(poly)
        end
        plane_neg(axis, c) = R3D.Plane{4, Float64}(
            R3D.Vec{4, Float64}(ntuple(k -> k == axis ? -1.0 : 0.0, Val(4))), c)
        plane_pos(axis, c) = R3D.Plane{4, Float64}(
            R3D.Vec{4, Float64}(ntuple(k -> k == axis ?  1.0 : 0.0, Val(4))), -c)

        # Two-axis quadrant volumes for axis pair (1, 2). The four
        # quadrants must tile the simplex and obey coordinate symmetry
        # under x ↔ y swap.
        v_nn = clip2_simplex((plane_neg(1, 0.5), plane_neg(2, 0.5)))
        v_np = clip2_simplex((plane_neg(1, 0.5), plane_pos(2, 0.5)))
        v_pn = clip2_simplex((plane_pos(1, 0.5), plane_neg(2, 0.5)))
        v_pp = clip2_simplex((plane_pos(1, 0.5), plane_pos(2, 0.5)))
        @test isapprox(v_np, v_pn; atol = 1e-10)
        @test isapprox(v_nn + v_np + v_pn + v_pp, 1/24; atol = 1e-10)
        # Geometrically: v_pp = ∅ (x ≥ 0.5 + y ≥ 0.5 forces x + y ≥ 1
        # which leaves no interior of the unit simplex), v_np = v_pn =
        # 1/384 (one-corner sub-simplex), v_nn = 1/24 - 2/384.
        @test isapprox(v_pp, 0.0;        atol = 1e-12)
        @test isapprox(v_np, 1/384;      atol = 1e-10)
        @test isapprox(v_nn, 1/24 - 2/384; atol = 1e-10)

        # voxelize_fold! over a 2^D grid recovers the per-cell volumes
        # with correct symmetry: the four "single-axis" corner cells
        # (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2) all equal 1/384,
        # the origin cell (1,1,1,1) holds the bulk 1/24 - 4/384, and
        # all other cells are empty. Pre-fix this returned an
        # increasing 0, 1/384, 2/384, 3/384, 4/384 sequence and a 0
        # origin cell.
        poly = R3D.Flat.FlatPolytope{4, Float64}(512)
        verts = [[0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
        R3D.Flat.init_simplex!(poly, verts)
        d_grid  = ntuple(_ -> 0.5, Val(4))
        ibox_lo = ntuple(_ -> 0,   Val(4))
        ibox_hi = ntuple(_ -> 2,   Val(4))
        cells = Tuple{NTuple{4, Int}, Float64}[]
        R3D.Flat.voxelize_fold!(cells, poly, ibox_lo, ibox_hi,
                                d_grid, 0) do acc, idx, m
            push!(acc, (idx, m[1]))
            acc
        end
        cellvol = Dict(idx => v for (idx, v) in cells)
        for axis in 1:4
            idx = ntuple(k -> k == axis ? 2 : 1, Val(4))
            @test isapprox(cellvol[idx], 1/384; atol = 1e-10)
        end
        @test isapprox(cellvol[(1, 1, 1, 1)], 1/24 - 4/384; atol = 1e-10)
        @test isapprox(sum(values(cellvol)), 1/24; atol = 1e-10)
    end

    @testset "D ≥ 4 API parity: box / simplex / aabb / box_planes / copy!" begin
        # Generic `box(lo, hi)` / `simplex(verts...)` constructors at D ≥ 4
        # mirror the D = 2 / D = 3 conveniences. Phase B.5.
        for D in 4:6
            lo = ntuple(_ -> 0.0, Val(D))
            hi = ntuple(_ -> 1.0, Val(D))
            b = R3D.Flat.box(lo, hi)
            @test b isa R3D.Flat.FlatPolytope{D,Float64}
            @test b.nverts == 1 << D
            @test isapprox(R3D.Flat.volume(b), 1.0; atol = 1e-12)
        end

        # Splat-of-vertices simplex constructor.
        s4 = R3D.Flat.simplex((0.0, 0.0, 0.0, 0.0),
                              (1.0, 0.0, 0.0, 0.0),
                              (0.0, 1.0, 0.0, 0.0),
                              (0.0, 0.0, 1.0, 0.0),
                              (0.0, 0.0, 0.0, 1.0))
        @test s4.nverts == 5
        @test isapprox(R3D.Flat.volume(s4), 1/24; atol = 1e-12)
        s5 = R3D.Flat.simplex(ntuple(_ -> 0.0, 5),
                              ntuple(k -> k == 1 ? 1.0 : 0.0, 5),
                              ntuple(k -> k == 2 ? 1.0 : 0.0, 5),
                              ntuple(k -> k == 3 ? 1.0 : 0.0, 5),
                              ntuple(k -> k == 4 ? 1.0 : 0.0, 5),
                              ntuple(k -> k == 5 ? 1.0 : 0.0, 5))
        @test s5.nverts == 6
        @test isapprox(R3D.Flat.volume(s5), 1/120; atol = 1e-12)

        # aabb (Phase B.4): D-generic FlatPolytope variant + D ≥ 4
        # StaticFlatPolytope variant.
        b4 = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0, 4.0))
        @test R3D.Flat.aabb(b4) ==
            ((0.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0, 4.0))
        # Empty polytope (clipped away) returns the canonical sentinel.
        empty4 = R3D.Flat.box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
        far_plane = R3D.Plane{4,Float64}(R3D.Vec{4,Float64}(1.0, 0.0, 0.0, 0.0), -100.0)
        R3D.Flat.clip!(empty4, far_plane)
        @test R3D.Flat.aabb(empty4) ==
            (ntuple(_ -> Inf, 4), ntuple(_ -> -Inf, 4))

        # box_planes / box_planes! (Phase B.2)
        for D in 4:6
            lo = ntuple(_ -> 0.0, Val(D))
            hi = ntuple(_ -> 1.0, Val(D))
            ps = R3D.Flat.box_planes(lo, hi)
            @test length(ps) == 2D
            # Round-trip: clipping a larger box by these planes equals the
            # inner unit hypercube to floating-point precision.
            big = R3D.Flat.box(ntuple(_ -> -0.5, Val(D)),
                               ntuple(_ ->  1.5, Val(D)))
            R3D.Flat.clip!(big, ps)
            @test isapprox(R3D.Flat.volume(big), 1.0; atol = 1e-12)
        end
        # In-place variant fills the right slots.
        out4 = Vector{R3D.Plane{4,Float64}}(undef, 8)
        R3D.Flat.box_planes!(out4, (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
        # Plane 1 is +x at lo[1] = 0; plane 2 is -x at hi[1] = 1.
        @test out4[1].n == R3D.Vec{4,Float64}(1.0, 0.0, 0.0, 0.0)
        @test out4[1].d == 0.0
        @test out4[2].n == R3D.Vec{4,Float64}(-1.0, 0.0, 0.0, 0.0)
        @test out4[2].d == 1.0
        # Length-mismatch errors out cleanly.
        wrong = Vector{R3D.Plane{4,Float64}}(undef, 6)
        @test_throws AssertionError R3D.Flat.box_planes!(wrong,
            (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))

        # copy! public overload (Phase B.3): drop-in for the existing
        # internal `_copy_polytope_nd!`.
        for D in 4:6
            src = R3D.Flat.box(ntuple(_ -> 0.0, Val(D)),
                               ntuple(_ -> 1.0, Val(D)))
            dst = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.copy!(dst, src)
            @test dst.nverts == src.nverts
            @test isapprox(R3D.Flat.volume(dst), R3D.Flat.volume(src);
                           atol = 1e-12)
            # Mutating dst's positions must not affect src.
            dst.positions[1, 1] = 999.0
            @test src.positions[1, 1] != 999.0
        end
    end

    @testset "Phase 3 groundwork: D ≥ 4 constructors + num_moments" begin
        # init_box for D = 4, 5, 6 — bit-hack vertex enumeration
        for D in 4:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            @test buf.nverts == 1 << D
            # vertex 0 should be all zeros, vertex (2^D - 1) all ones
            @test all(buf.positions[i, 1] == 0.0 for i in 1:D)
            @test all(buf.positions[i, 1 << D] == 1.0 for i in 1:D)
        end

        # init_simplex for D = 4, 5, 6 — D+1 vertices, cyclic neighbours
        for D in 4:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(128)
            verts = [ntuple(j -> j == i ? 1.0 : 0.0, D) for i in 0:D]
            R3D.Flat.init_simplex!(buf, verts)
            @test buf.nverts == D + 1
            # Spot-check pnbrs cyclic structure
            @test buf.pnbrs[1, 1] == ((0 + 1) % (D + 1)) + 1
        end

        # num_moments(D, P) = binomial(D+P, P) for D = 4..6, P = 0..3
        for D in 4:6, P in 0:3
            @test R3D.num_moments(D, P) == binomial(D + P, P)
        end

        # `clip!` and `moments(., 0)` are now real for D ≥ 4 (Phase 3c
        # + 3d). D = 4 higher-order moments now via Lasserre (Phase A);
        # D = 5 / D = 6 still stubbed.
        buf4 = R3D.Flat.FlatPolytope{4,Float64}(64)
        R3D.Flat.init_box!(buf4, zeros(4), ones(4))
        @test isapprox(R3D.Flat.moments(buf4, 0)[1], 1.0; atol=1e-12)
        @test isapprox(R3D.Flat.moments(buf4, 1)[1], 1.0; atol=1e-12)
        for D in 5:6
            buf = R3D.Flat.FlatPolytope{D,Float64}(64)
            R3D.Flat.init_box!(buf, zeros(D), ones(D))
            @test_throws ErrorException R3D.Flat.moments(buf, 1)
        end
    end

    @testset "Hot-loop overlap pattern is 0-alloc end-to-end" begin
        # Mirror exactly what the HierarchicalGrids overlap layer does:
        # init_simplex! + box_planes! + clip! + moments!. Every call must
        # be 0-alloc once warmed up.
        work = R3D.Flat.FlatPolytope{2,Float64}(64)
        plane_buf = Vector{R3D.Plane{2,Float64}}(undef, 4)
        moments_buf = zeros(Float64, R3D.num_moments(2, 3))

        verts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        leaf_lo = (0.25, 0.0); leaf_hi = (1.0, 0.5)

        # warmup
        R3D.Flat.init_simplex!(work, verts)
        R3D.Flat.box_planes!(plane_buf, leaf_lo, leaf_hi)
        R3D.Flat.clip!(work, plane_buf)
        R3D.Flat.moments!(moments_buf, work, 3)

        # Now a clean iteration measured.
        a_init = @allocated R3D.Flat.init_simplex!(work, verts)
        a_box  = @allocated R3D.Flat.box_planes!(plane_buf, leaf_lo, leaf_hi)
        a_clip = @allocated R3D.Flat.clip!(work, plane_buf)
        a_mom  = @allocated R3D.Flat.moments!(moments_buf, work, 3)
        @test a_init <= 256   # only the literal `[(0.0,0.0), …]` from the call site
        @test a_box  <= ALLOC_TOLERANCE
        @test a_clip <= ALLOC_TOLERANCE
        @test a_mom  <= ALLOC_TOLERANCE
        @test isapprox(moments_buf[1], 0.25; atol=1e-12)
    end

    @testset "voxelize_batch! parallel agrees with serial" begin
        # Build a small batch of clipped boxes
        rng = Random.MersenneTwister(20260435)
        N = 16
        polys = R3D.Flat.FlatPolytope{3,Float64}[]
        for _ in 1:N
            p = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            nplanes = rand(rng, 0:2)
            pls = R3D.Plane{3,Float64}[]
            for _ in 1:nplanes
                v = randn(rng, 3); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5+v[2]*0.5+v[3]*0.5) + 0.3*randn(rng)
                push!(pls, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd))
            end
            R3D.Flat.clip!(p, pls)
            push!(polys, p)
        end
        nmom = R3D.num_moments(3, 1)
        grids_par = [zeros(Float64, nmom, 8, 8, 8) for _ in 1:N]
        grids_ser = [zeros(Float64, nmom, 8, 8, 8) for _ in 1:N]
        los = [(0,0,0) for _ in 1:N]
        his = [(8,8,8) for _ in 1:N]
        d = (1/8, 1/8, 1/8)

        R3D.Flat.voxelize_batch!(grids_par, polys, los, his, d, 1)

        # Serial reference
        ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)
        for k in 1:N
            R3D.Flat.voxelize!(grids_ser[k], polys[k], los[k], his[k], d, 1;
                                workspace = ws)
        end

        for k in 1:N
            for I in eachindex(grids_par[k])
                @test isapprox(grids_par[k][I], grids_ser[k][I];
                               atol=1e-12, rtol=1e-10)
            end
            # Also: grid total mass equals direct moments(poly, 0)[1]
            @test isapprox(sum(@view grids_par[k][1, :, :, :]),
                           R3D.Flat.moments(polys[k], 0)[1];
                           atol=1e-10, rtol=1e-9)
        end
    end

    @testset "Aqua quality checks" begin
        # Standard hygiene: no method ambiguities, no piracy of Base
        # methods, all declared deps used, no unbound type parameters,
        # no stale-pin compat bounds. We allow Project-extras compat
        # warnings since Aqua < v0.8 didn't always recognize the test
        # extras pattern across Julia versions.
        Aqua.test_all(R3D;
                      ambiguities = (recursive = false,),
                      piracies = true,
                      deps_compat = (check_extras = false,),
                      project_extras = true,
                      stale_deps = true,
                      unbound_args = true,
                      undefined_exports = true)
    end

    @testset "Base.show outputs include nverts" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        s = sprint(show, MIME("text/plain"), sq)
        @test occursin("nverts = 4", s)
        @test occursin("FlatPolytope", s)
        s2 = sprint(show, sq)
        @test occursin("nverts=4", s2)

        sfp = R3D.Flat.StaticFlatPolytope{3,Float64,32}()
        R3D.Flat.init_box!(sfp, [0.0,0.0,0.0], [1.0,1.0,1.0])
        s3 = sprint(show, MIME("text/plain"), sfp)
        @test occursin("StaticFlatPolytope", s3)
        @test occursin("nverts = 8", s3)
    end
end

# Additional polytope ops: split!, is_good, shift_moments — closed-form
# checks plus differential vs C r3d / r2d.
@testset "R3D.Flat additional ops" begin

    @testset "split! D=3 conserves total volume" begin
        rng = Random.MersenneTwister(20260430)
        out_pos = R3D.Flat.FlatBuffer{3,Float64}(64)
        out_neg = R3D.Flat.FlatBuffer{3,Float64}(64)
        for trial in 1:200
            buf = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            v_in = R3D.Flat.moments(buf, 0)[1]
            v = randn(rng, 3); v ./= sqrt(sum(v.^2))
            dd = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.4 * randn(rng)
            plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd)
            ok = R3D.Flat.split!(buf, plane, out_pos, out_neg)
            @test ok
            v_pos = out_pos.nverts > 0 ? R3D.Flat.moments(out_pos, 0)[1] : 0.0
            v_neg = out_neg.nverts > 0 ? R3D.Flat.moments(out_neg, 0)[1] : 0.0
            @test isapprox(v_pos + v_neg, v_in; atol=1e-12, rtol=1e-10)
        end
    end

    @testset "split! D=2 conserves total area" begin
        rng = Random.MersenneTwister(20260431)
        op2 = R3D.Flat.FlatBuffer{2,Float64}(32)
        on2 = R3D.Flat.FlatBuffer{2,Float64}(32)
        for trial in 1:200
            sq = R3D.Flat.box((0.0,0.0), (1.0,1.0))
            a_in = R3D.Flat.moments(sq, 0)[1]
            v = randn(rng, 2); v ./= sqrt(sum(v.^2))
            dd = -(v[1]*0.5 + v[2]*0.5) + 0.4 * randn(rng)
            plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(v), dd)
            R3D.Flat.split!(sq, plane, op2, on2)
            a_pos = op2.nverts > 0 ? R3D.Flat.moments(op2, 0)[1] : 0.0
            a_neg = on2.nverts > 0 ? R3D.Flat.moments(on2, 0)[1] : 0.0
            @test isapprox(a_pos + a_neg, a_in; atol=1e-12, rtol=1e-10)
        end
    end

    @testset "is_good on fresh and clipped polytopes" begin
        @test R3D.Flat.is_good(R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0)))
        @test R3D.Flat.is_good(R3D.Flat.box((0.0,0.0), (1.0,1.0)))

        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0,1.0,1.0]/sqrt(3)), -1/sqrt(3))
        R3D.Flat.clip!(cube, [plane])
        @test R3D.Flat.is_good(cube)

        # StaticFlatPolytope path
        sfp = R3D.Flat.StaticFlatPolytope{3,Float64,32}()
        R3D.Flat.init_box!(sfp, [0.0,0.0,0.0], [1.0,1.0,1.0])
        @test R3D.Flat.is_good(sfp)
    end

    @testset "shift_moments! identity (shift by 0)" begin
        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        m = R3D.Flat.moments(cube, 3)
        m_orig = copy(m)
        R3D.Flat.shift_moments!(m, 3, (0.0, 0.0, 0.0), Val(3))
        @test all(isapprox.(m, m_orig; atol=1e-14))

        sq = R3D.Flat.box((0.0,0.0), (1.0,1.0))
        ms = R3D.Flat.moments(sq, 3)
        ms_orig = copy(ms)
        R3D.Flat.shift_moments!(ms, 3, (0.0, 0.0), Val(2))
        @test all(isapprox.(ms, ms_orig; atol=1e-14))
    end

    @testset "shift_moments! shifts polytope (closed forms)" begin
        # Cube [0,1]^3 → shifted by (1.5, 0, 0) → cube [1.5, 2.5] × [0,1]²
        cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        m = R3D.Flat.moments(cube, 1)
        R3D.Flat.shift_moments!(m, 1, (1.5, 0.0, 0.0), Val(3))
        # Volume unchanged; M[x] = ∫(x+1.5) over cube = 0.5 + 1.5 = 2.0
        @test m[1] ≈ 1.0
        @test m[2] ≈ 2.0   # ∫x for shifted cube
        @test m[3] ≈ 0.5   # ∫y unchanged
        @test m[4] ≈ 0.5   # ∫z unchanged

        # Compare against direct moments of the shifted cube
        cube2 = R3D.Flat.box((1.5, 0.0, 0.0), (2.5, 1.0, 1.0))
        m_ref = R3D.Flat.moments(cube2, 2)
        cube  = R3D.Flat.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        m = R3D.Flat.moments(cube, 2)
        R3D.Flat.shift_moments!(m, 2, (1.5, 0.0, 0.0), Val(3))
        for k in eachindex(m)
            @test isapprox(m[k], m_ref[k]; atol=1e-12, rtol=1e-10)
        end
    end

    @testset "shift_moments! 2D matches direct re-computation" begin
        rng = Random.MersenneTwister(20260432)
        for trial in 1:50
            shift = (randn(rng), randn(rng))
            sq = R3D.Flat.box((0.13, 0.27), (1.4, 1.6))
            sq_shifted = R3D.Flat.box((0.13 + shift[1], 0.27 + shift[2]),
                                       (1.4 + shift[1], 1.6 + shift[2]))
            m  = R3D.Flat.moments(sq, 3)
            m_ref = R3D.Flat.moments(sq_shifted, 3)
            R3D.Flat.shift_moments!(m, 3, shift, Val(2))
            for k in eachindex(m)
                @test isapprox(m[k], m_ref[k]; atol=1e-10, rtol=1e-8)
            end
        end
    end

    @testset "split!, is_good, shift_moments! differential vs C" begin
        if !HAVE_C
            @info "skipping additional-ops differential; ENV[R3D_LIB] not set"
            return
        end
        rng = Random.MersenneTwister(20260433)
        out_pos = R3D.Flat.FlatBuffer{3,Float64}(64)
        out_neg = R3D.Flat.FlatBuffer{3,Float64}(64)
        cp_in,  buf_in  = R3D_C.new_poly()
        cp_pos, buf_pos = R3D_C.new_poly()
        cp_neg, buf_neg = R3D_C.new_poly()

        for trial in 1:100
            jl_buf = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            GC.@preserve buf_in R3D_C.init_box!(cp_in, R3D_C.RVec3(0.0,0.0,0.0),
                                                 R3D_C.RVec3(1.0,1.0,1.0))
            v = randn(rng, 3); v ./= sqrt(sum(v.^2))
            dd = -(v[1]*0.5 + v[2]*0.5 + v[3]*0.5) + 0.3 * randn(rng)
            jl_plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd)
            c_plane  = R3D_C.Plane(R3D_C.RVec3(v[1], v[2], v[3]), dd)

            R3D.Flat.split!(jl_buf, jl_plane, out_pos, out_neg)
            GC.@preserve buf_in buf_pos buf_neg R3D_C.split!(cp_in, c_plane, cp_pos, cp_neg)

            for ord in 0:1
                nm = R3D.num_moments(3, ord)
                # Positive side
                m_jp = out_pos.nverts > 0 ? R3D.Flat.moments(out_pos, ord) : zeros(Float64, nm)
                m_cp = zeros(Float64, nm)
                GC.@preserve buf_pos R3D_C.reduce!(cp_pos, m_cp, ord)
                for k in 1:nm
                    @test isapprox(m_jp[k], m_cp[k]; atol=1e-12, rtol=1e-10)
                end
                # Negative side
                m_jn = out_neg.nverts > 0 ? R3D.Flat.moments(out_neg, ord) : zeros(Float64, nm)
                m_cn = zeros(Float64, nm)
                GC.@preserve buf_neg R3D_C.reduce!(cp_neg, m_cn, ord)
                for k in 1:nm
                    @test isapprox(m_jn[k], m_cn[k]; atol=1e-12, rtol=1e-10)
                end
            end
        end

        # is_good: cross-check with C
        cube_j = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
        cp, cbuf = R3D_C.new_poly()
        GC.@preserve cbuf R3D_C.init_box!(cp, R3D_C.RVec3(0.0,0.0,0.0),
                                          R3D_C.RVec3(1.0,1.0,1.0))
        @test R3D.Flat.is_good(cube_j) == GC.@preserve cbuf R3D_C.is_good(cp)

        # shift_moments: compare to C r3d_shift_moments
        for trial in 1:30
            cube = R3D.Flat.box((0.0,0.0,0.0), (1.0,1.0,1.0))
            m_jl = R3D.Flat.moments(cube, 3)
            m_c  = copy(m_jl)
            shift = (randn(rng), randn(rng), randn(rng))
            R3D.Flat.shift_moments!(m_jl, 3, shift, Val(3))
            R3D_C.shift_moments!(m_c, 3, R3D_C.RVec3(shift[1], shift[2], shift[3]))
            for k in eachindex(m_jl)
                @test isapprox(m_jl[k], m_c[k]; atol=1e-9, rtol=1e-8)
            end
        end
    end
end

# StaticFlatPolytope: small-cap MMatrix variant. Same algorithm as
# FlatPolytope, just type-level capacity. Confirm it agrees on every
# operation.
@testset "R3D.Flat StaticFlatPolytope" begin
    sfp = R3D.Flat.StaticFlatPolytope{3,Float64,32}()
    fbuf = R3D.Flat.FlatBuffer{3,Float64}(32)

    @testset "init_box! and basic moments agree with FlatPolytope" begin
        for (lo, hi) in [((0.0,0.0,0.0), (1.0,1.0,1.0)),
                         ((0.13,0.27,0.41), (1.62,1.45,1.78)),
                         ((-1.0,-2.0,-3.0), (0.5,0.5,0.5))]
            R3D.Flat.init_box!(sfp, [lo...], [hi...])
            R3D.Flat.init_box!(fbuf, [lo...], [hi...])
            for ord in 0:2
                @test R3D.Flat.moments(sfp, ord) ≈ R3D.Flat.moments(fbuf, ord)
            end
        end
    end

    @testset "clip! agrees with FlatPolytope on random clip sets" begin
        rng = Random.MersenneTwister(20260429)
        for trial in 1:200
            R3D.Flat.init_box!(sfp, [0.0,0.0,0.0], [1.0,1.0,1.0])
            R3D.Flat.init_box!(fbuf, [0.0,0.0,0.0], [1.0,1.0,1.0])
            nplanes = rand(rng, 0:4)
            planes = R3D.Plane{3,Float64}[]
            for _ in 1:nplanes
                v = randn(rng, 3); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5+v[2]*0.5+v[3]*0.5) + 0.3*randn(rng)
                push!(planes, R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(v), dd))
            end
            R3D.Flat.clip!(sfp, planes)
            R3D.Flat.clip!(fbuf, planes)
            @test sfp.nverts == fbuf.nverts
            @test R3D.Flat.moments(sfp, 1) ≈ R3D.Flat.moments(fbuf, 1)
        end
    end

    @testset "Hot loop is allocation-free (after warmup)" begin
        sfp = R3D.Flat.StaticFlatPolytope{3,Float64,32}()
        plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0,1.0,1.0]./sqrt(3.0)), -1/sqrt(3.0))
        out = zeros(Float64, R3D.num_moments(3, 2))
        R3D.Flat.init_box!(sfp, [0.0,0.0,0.0], [1.0,1.0,1.0])
        R3D.Flat.clip!(sfp, [plane])
        R3D.Flat.moments!(out, sfp, 2)   # warmup
        a_clip = @allocated R3D.Flat.clip!(sfp, [plane])
        a_mom  = @allocated R3D.Flat.moments!(out, sfp, 2)
        @test a_clip <= 256   # only the literal `[plane]` vector
        @test a_mom <= ALLOC_TOLERANCE
    end
end

# 2D port (D=2): closed-form + differential vs C r2d.
@testset "R3D.Flat 2D" begin

    @testset "Construction and area" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        @test sq.nverts == 4
        @test R3D.Flat.moments(sq, 0)[1] ≈ 1.0
        big = R3D.Flat.box((0.0, 0.0), (2.0, 3.0))
        @test R3D.Flat.moments(big, 0)[1] ≈ 6.0
    end

    @testset "Centroid via first moments" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        m = R3D.Flat.moments(sq, 1)
        @test m ≈ [1.0, 0.5, 0.5]
        # Off-center
        r = R3D.Flat.box((1.0, 2.0), (3.0, 5.0))
        m2 = R3D.Flat.moments(r, 1)
        V = m2[1]
        @test V ≈ 6.0
        @test m2[2] / V ≈ 2.0
        @test m2[3] / V ≈ 3.5
    end

    @testset "Order-2 closed forms on unit square" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        m = R3D.Flat.moments(sq, 2)
        @test m[1] ≈ 1.0
        @test m[2] ≈ 0.5      # ∫x
        @test m[3] ≈ 0.5      # ∫y
        @test m[4] ≈ 1/3      # ∫x²
        @test m[5] ≈ 1/4      # ∫xy
        @test m[6] ≈ 1/3      # ∫y²
    end

    @testset "Single-plane clip" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(1, 0), -0.5)
        @test R3D.Flat.clip!(sq, [plane])
        @test R3D.Flat.moments(sq, 0)[1] ≈ 0.5
        m = R3D.Flat.moments(sq, 1)
        @test m[2] / m[1] ≈ 0.75
    end

    @testset "Diagonal cut → triangle" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        n = [1.0, 1.0] ./ sqrt(2.0); d = -1.5/sqrt(2.0)
        plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(n), d)
        R3D.Flat.clip!(sq, [plane])
        @test sq.nverts == 3
        @test R3D.Flat.moments(sq, 0)[1] ≈ 0.125
    end

    @testset "Empty result on full clip" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(1, 0), -100.0)
        @test R3D.Flat.clip!(sq, [plane])
        @test sq.nverts == 0
    end

    @testset "Voxelize unit square exactly tiles its grid" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        d = (0.25, 0.25)
        grid, lo, hi = R3D.Flat.voxelize(sq, d, 0)
        @test lo == (0, 0)
        @test hi == (4, 4)
        @test size(grid) == (1, 4, 4)
        @test all(isapprox.(grid, 1/16; atol=1e-12))
    end

    @testset "Voxelize total mass = area for off-axis rect" begin
        r = R3D.Flat.box((0.13, 0.27), (1.62, 1.45))
        for d in [(0.1, 0.1), (0.2, 0.15), (0.07, 0.3)]
            grid, _, _ = R3D.Flat.voxelize(r, d, 0)
            @test isapprox(sum(grid), (1.62-0.13) * (1.45-0.27); atol=1e-10)
        end
    end

    @testset "2D voxelize workspace is 0-alloc" begin
        sq = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
        ws = R3D.Flat.VoxelizeWorkspace{2,Float64}(64)
        grid = zeros(Float64, R3D.num_moments(2, 1), 4, 4)
        d = (0.25, 0.25)
        R3D.Flat.voxelize!(grid, sq, (0,0), (4,4), d, 1; workspace=ws)   # warmup
        fill!(grid, 0.0)
        a = @allocated R3D.Flat.voxelize!(grid, sq, (0,0), (4,4), d, 1; workspace=ws)
        @test a <= ALLOC_TOLERANCE
        @test isapprox(sum(@view grid[1, :, :]), 1.0; atol=1e-12)
    end

    @testset "Differential vs C r2d (clip + reduce + rasterize)" begin
        if !HAVE_C
            @info "skipping 2D differential; ENV[R3D_LIB] not set"
            return
        end
        rng = Random.MersenneTwister(20260428)
        cp, cbuf = R3D_C.new_poly2()

        for trial in 1:200
            sq_j = R3D.Flat.box((0.0, 0.0), (1.0, 1.0))
            GC.@preserve cbuf R3D_C.init_box2!(cp, R3D_C.RVec2(0.0,0.0),
                                                R3D_C.RVec2(1.0,1.0))

            nplanes = rand(rng, 0:4)
            jl_pls = R3D.Plane{2,Float64}[]
            c_pls = R3D_C.Plane2[]
            for _ in 1:nplanes
                v = randn(rng, 2); v ./= sqrt(sum(v.^2))
                dd = -(v[1]*0.5 + v[2]*0.5) + 0.3 * randn(rng)
                push!(jl_pls, R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(v), dd))
                push!(c_pls, R3D_C.Plane2(R3D_C.RVec2(v[1], v[2]), dd))
            end

            R3D.Flat.clip!(sq_j, jl_pls)
            GC.@preserve cbuf R3D_C.clip2!(cp, c_pls)

            # Compare moments at order 2
            order = 2
            m_j = R3D.Flat.moments(sq_j, order)
            m_c = zeros(Float64, R3D.num_moments(2, order))
            GC.@preserve cbuf R3D_C.reduce2!(cp, m_c, order)
            for k in eachindex(m_j)
                @test isapprox(m_j[k], m_c[k]; atol=1e-12, rtol=1e-10)
            end

            # Skip voxelize on emptied or degenerate polytopes
            sq_j.nverts == 0 && continue
            d = (rand(rng)*0.2 + 0.05, rand(rng)*0.2 + 0.05)
            ord_v = rand(rng, 0:1)
            lo, hi = R3D.Flat.get_ibox(sq_j, d)
            lo == hi && continue
            ni = hi[1] - lo[1]; nj = hi[2] - lo[2]
            nmom = R3D.num_moments(2, ord_v)
            c_grid = zeros(Float64, ni*nj*nmom)
            GC.@preserve cbuf c_grid R3D_C.rasterize2!(c_grid, cp, lo, hi, d, ord_v)
            j_grid, _, _ = R3D.Flat.voxelize(sq_j, d, ord_v; ibox=(lo, hi))

            for i in 1:ni, j in 1:nj, m in 1:nmom
                c_idx = ((i-1)*nj + (j-1))*nmom + m
                @test isapprox(j_grid[m, i, j], c_grid[c_idx];
                               atol=1e-12, rtol=1e-10)
            end
        end
    end
end
