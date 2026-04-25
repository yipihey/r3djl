# Overlap of a Lagrangian triangle with an Eulerian axis-aligned box,
# with polynomial moments to order 3. The runnable counterpart of
# docs/overlap_example.md.
#
# Run:
#   julia --project=R3D.jl examples/overlap_triangle_box.jl

using R3D

# 1. Build both polytopes.
verts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
tri = R3D.Flat.FlatPolytope{2,Float64}(64)
R3D.Flat.init_simplex!(tri, verts)

box_lo = (0.25, 0.0)
box_hi = (1.0,  0.5)

# 2. Build the box's clipping planes.
# Convention: each plane (n, d) keeps the half-space n·x + d ≥ 0.
planes = [
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 1.0,  0.0), -box_lo[1]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}(-1.0,  0.0),  box_hi[1]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 0.0,  1.0), -box_lo[2]),
    R3D.Plane{2,Float64}(R3D.Vec{2,Float64}( 0.0, -1.0),  box_hi[2]),
]

# 3. Clip in place; integrate moments to order 3.
ok = R3D.Flat.clip!(tri, planes)
@assert ok
m = R3D.Flat.moments(tri, 3)

# 4. Closed-form check.
# T ∩ B is a quadrilateral with vertices
#   (0.25, 0), (1, 0), (0.5, 0.5), (0.25, 0.5)
# (walk CCW: along y=0 to T's hypotenuse, along x+y=1 to y=0.5, along
# y=0.5 left to x=0.25, then down x=0.25 back to start).
# Shoelace gives area = 0.25 = 1/4.
expected_area = 0.25
println("Intersection vertices: ", tri.nverts)
println("∫ 1 dV  = ", m[1], "   (expect 1/4 = $expected_area)")
println("∫ x dV  = ", m[2], "   (centroid x = ", m[2] / m[1], ")")
println("∫ y dV  = ", m[3], "   (centroid y = ", m[3] / m[1], ")")
println("∫ x² dV = ", m[4])
println("∫ xy dV = ", m[5])
println("∫ y² dV = ", m[6])
println("...higher moments through y³ are at indices 7..10")

@assert isapprox(m[1], expected_area; atol=1e-12) "got $(m[1]), expected $expected_area"
println()
println("✓ closed-form area matches.")
