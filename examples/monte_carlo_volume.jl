# Monte Carlo volume estimate vs analytic moments.
#
# Demonstrates that R3D's analytic moment integration converges to the
# true volume orders of magnitude faster than uniform sampling.
#
# Run:
#   julia --project=R3D.jl examples/monte_carlo_volume.jl

using R3D
using Random

# Build a clipped tet — analytic volume comes from moments(poly, 0)[1].
poly = R3D.Flat.tet((0.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (0.0, 2.0, 0.0),
                    (0.0, 0.0, 2.0))
plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0] / sqrt(3)),
                              -1.2 / sqrt(3))
R3D.Flat.clip!(poly, [plane])

vol_exact = R3D.Flat.moments(poly, 0)[1]
println("R3D analytic volume:       ", vol_exact)

# Bounding box for MC sampling.
lo = (0.0, 0.0, 0.0); hi = (2.0, 2.0, 2.0)
box_vol = (hi[1] - lo[1]) * (hi[2] - lo[2]) * (hi[3] - lo[3])

function inside(x, y, z, plane)
    s = plane.d + plane.n[1]*x + plane.n[2]*y + plane.n[3]*z
    if s < 0; return false; end
    # tet: x ≥ 0, y ≥ 0, z ≥ 0, x + y + z ≤ 2
    return x >= 0 && y >= 0 && z >= 0 && x + y + z <= 2
end

rng = MersenneTwister(20260425)
println()
println("Monte Carlo samples vs error:")
for nsamp in [10^3, 10^4, 10^5, 10^6, 10^7]
    hits = 0
    for _ in 1:nsamp
        x = lo[1] + rand(rng) * (hi[1] - lo[1])
        y = lo[2] + rand(rng) * (hi[2] - lo[2])
        z = lo[3] + rand(rng) * (hi[3] - lo[3])
        inside(x, y, z, plane) && (hits += 1)
    end
    vol_mc = hits / nsamp * box_vol
    println("  N = $nsamp:  vol ≈ $vol_mc   (error ", abs(vol_mc - vol_exact), ")")
end

println()
println("Analytic moments converge in O(nverts) work, no sampling noise.")
