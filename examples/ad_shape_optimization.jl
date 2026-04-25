# ForwardDiff-driven shape optimization.
#
# Find the plane offset that splits a unit cube into a target volume
# fraction. Demonstrates that R3D's clip + reduce pipeline is
# AD-compatible end-to-end (positions can be ForwardDiff.Dual numbers).
#
# Run:
#   julia --project=R3D.jl examples/ad_shape_optimization.jl

using R3D
using ForwardDiff

# Side fraction f(p) = volume of cube ∩ {x ≥ p}, as a function of p.
# Allocate the polytope with the input element type so ForwardDiff Dual
# numbers propagate through positions cleanly.
function side_fraction(p::T) where {T<:Real}
    cube  = R3D.Flat.FlatPolytope{3,T}(64)
    R3D.Flat.init_box!(cube, T[0, 0, 0], T[1, 1, 1])
    plane = R3D.Plane{3,T}(R3D.Vec{3,T}(T[1, 0, 0]), -p)
    R3D.Flat.clip!(cube, [plane])
    return R3D.Flat.moments(cube, 0)[1]
end

# Newton iteration to find p* such that side_fraction(p*) = 0.31.
function newton_solve(target; p0 = 0.5, iters = 8)
    p = p0
    for iter in 1:iters
        f = side_fraction(p) - target
        df = ForwardDiff.derivative(side_fraction, p)
        @info "iteration" iter p f df
        p -= f / df
    end
    return p
end

p_star = newton_solve(0.31)
println("Converged p* = ", p_star)
println("side_fraction(p*) = ", side_fraction(p_star), "   (target = 0.31)")
