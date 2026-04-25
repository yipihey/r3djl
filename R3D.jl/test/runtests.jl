using Test
using R3D
using ForwardDiff
using Aqua

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
        @test a_clip == 0
        @test a_mom  == 0
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
        @test a == 0
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
        @test a_mom == 0
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
        @test a == 0
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
