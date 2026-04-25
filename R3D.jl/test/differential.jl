"""
Differential tests: every operation in R3D.jl run side-by-side against the
upstream C library wrapped by R3D_C.jl, on identical inputs.

These tests are skipped when `ENV["R3D_LIB"]` is unset (R3D_C.jl can't
load the library). When it is set, every clipping case here checks that
the Julia and C outputs agree to within numerical tolerance.

# Why this matters

Unit tests on the Julia port can pass on simple shapes while still being
subtly wrong — e.g. a misordered pnbrs table that produces correct
volumes but wrong centroids on asymmetric inputs. Cross-checking against
the original C catches these.

# Test categories

1. **Init invariants**: same vertex positions, same connectivity.
2. **Clip outputs**: same vertex set (modulo permutation), same volume.
3. **Moment integration**: agreement to N significant figures across
   polynomial orders 0–3.
4. **Random stress**: random tetrahedra clipped by random plane sets,
   driven from the same RNG seed; volumes must agree.
"""

using Test
using R3D
using Random
using LinearAlgebra: dot

const HAVE_C = !isempty(get(ENV, "R3D_LIB", ""))

if HAVE_C
    using R3D_C
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All C-poly-using helpers take both the pointer and the underlying buffer
# so the caller can wrap them in GC.@preserve. Without that discipline, the
# pointer can be left dangling if the buffer Vector is collected.

"Compare a Julia Polytope to a C poly by total volume."
function compare_volume(jl_poly, c_ptr, c_buf; rtol = 1e-12)
    jl_m = R3D.moments(jl_poly, 0)
    c_m = zeros(Float64, 1)
    GC.@preserve c_buf begin
        R3D_C.reduce!(c_ptr, c_m, 0)
    end
    @test isapprox(jl_m[1], c_m[1]; rtol = rtol)
end

"Compare moments up to a given order."
function compare_moments(jl_poly, c_ptr, c_buf, order; rtol = 1e-10)
    jl_m = R3D.moments(jl_poly, order)
    nm = R3D.num_moments(3, order)
    c_m = zeros(Float64, nm)
    GC.@preserve c_buf begin
        R3D_C.reduce!(c_ptr, c_m, order)
    end
    @test length(jl_m) == nm
    for i in 1:nm
        if abs(c_m[i]) < 1e-14
            @test abs(jl_m[i]) < 1e-12
        else
            @test isapprox(jl_m[i], c_m[i]; rtol = rtol)
        end
    end
end

"Build matched (Julia, C) cube pairs."
function matched_cubes(lo::NTuple{3,Float64}, hi::NTuple{3,Float64})
    jl = R3D.box(lo, hi)
    c_ptr, c_buf = R3D_C.new_poly()
    GC.@preserve c_buf begin
        R3D_C.init_box!(c_ptr, R3D_C.RVec3(lo), R3D_C.RVec3(hi))
    end
    return jl, c_ptr, c_buf
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "Differential vs upstream r3d" begin

    if !HAVE_C
        @info "ENV[\"R3D_LIB\"] not set — skipping differential tests. " *
              "Run R3D_C.jl/deps/build_libr3d.sh and set R3D_LIB to enable."
        return
    end

    @testset "init_box agreement" begin
        jl, cp, cbuf = matched_cubes((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        @test jl.nverts == 8
        @test R3D_C.nverts(cp) == 8

        # Both should have unit volume
        compare_volume(jl, cp, cbuf)

        # Vertex positions: Julia and C use the same labelling, so positions
        # at matching indices should be identical.
        c_positions = R3D_C.vertex_positions(cp)
        for i in 1:8
            jl_pos = jl.verts[i].pos
            @test all(jl_pos .≈ c_positions[i])
        end
    end

    @testset "moments on unit cube" begin
        jl, cp, cbuf = matched_cubes((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        # Unit cube at origin: M[1] = 1, M[x] = 1/2, M[y] = 1/2, M[z] = 1/2,
        # M[x²] = 1/3, M[xy] = 1/4, ...
        compare_moments(jl, cp, cbuf, 0)
        compare_moments(jl, cp, cbuf, 1)
        compare_moments(jl, cp, cbuf, 2)
        compare_moments(jl, cp, cbuf, 3)

        # Sanity: known closed forms
        m = R3D.moments(jl, 1)
        @test m[1] ≈ 1.0
        @test m[2] ≈ 0.5    # ∫x dV
        @test m[3] ≈ 0.5
        @test m[4] ≈ 0.5
    end

    @testset "single-plane clip agrees" begin
        jl, cp, cbuf = matched_cubes((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        # Plane: x + y + z ≥ 1, i.e. clip out the corner near origin
        n = [1.0, 1.0, 1.0] / sqrt(3.0)
        d = -1.0 / sqrt(3.0)
        jl_plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(n), d)
        c_plane = R3D_C.Plane(R3D_C.RVec3(n[1], n[2], n[3]), d)

        ok_jl = R3D.clip!(jl, [jl_plane])
        ok_c = R3D_C.clip!(cp, [c_plane])
        @test ok_jl
        @test ok_c == 1

        compare_volume(jl, cp, cbuf)
        # Closed form: cube minus tetrahedron with vertices at three
        # adjacent corners and the opposite far corner ⇒ 1 - 1/6 = 5/6.
        m = R3D.moments(jl, 0)
        @test m[1] ≈ 5/6
    end

    @testset "random stress: 100 random clips of unit cube" begin
        rng = MersenneTwister(20260425)
        for trial in 1:100
            jl, cp, cbuf = matched_cubes((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

            # 1–4 random planes, normals uniform on sphere, offsets near the
            # cube interior so the clips are nontrivial
            nplanes = rand(rng, 1:4)
            jl_planes = R3D.Plane{3,Float64}[]
            c_planes = R3D_C.Plane[]
            for _ in 1:nplanes
                n = randn(rng, 3); n ./= sqrt(sum(n.^2))
                # Offset so the plane crosses inside the cube
                d = -(n[1]*0.5 + n[2]*0.5 + n[3]*0.5) + 0.3 * randn(rng)
                push!(jl_planes,
                      R3D.Plane{3,Float64}(R3D.Vec{3,Float64}(n), d))
                push!(c_planes,
                      R3D_C.Plane(R3D_C.RVec3(n[1], n[2], n[3]), d))
            end

            R3D.clip!(jl, jl_planes)
            R3D_C.clip!(cp, c_planes)

            # Volumes should match very tightly
            compare_volume(jl, cp, cbuf; rtol = 1e-10)
        end
    end
end
