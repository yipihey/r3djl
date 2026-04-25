# 2D rasterization: render a clipped polygon onto a grid.
#
# Run:
#   julia --project=R3D.jl examples/rasterize_2d.jl

using R3D

# Pentagon, then clip away the upper-left corner.
verts = [(0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (0.5, 1.5), (-0.5, 1.0)]
poly = R3D.Flat.FlatPolytope{2,Float64}(64)
R3D.Flat.init_poly!(poly, verts)

plane = R3D.Plane{2,Float64}(R3D.Vec{2,Float64}([-1.0, 1.0] / sqrt(2)), -0.6)
R3D.Flat.clip!(poly, [plane])

ws = R3D.Flat.VoxelizeWorkspace{2,Float64}(64)

# Pick the integer grid that bounds the polygon.
d = (0.05, 0.05)
lo, hi = R3D.Flat.get_ibox(poly, d)
ni, nj = hi[1] - lo[1], hi[2] - lo[2]
nmom = R3D.num_moments(2, 0)
grid = zeros(Float64, nmom, ni, nj)

R3D.Flat.voxelize!(grid, poly, lo, hi, d, 0; workspace = ws)

println("Rasterized $(ni)×$(nj) cells, total area = ", sum(grid))
println("Direct area from moments = ", R3D.Flat.moments(poly, 0)[1])

# ASCII visualization of the cell coverage (each cell = area fraction).
println()
println("Cell coverage (∎ ≥ 75%, ▣ ≥ 25%, · partial):")
cell_area = d[1] * d[2]
for j in nj:-1:1
    line = ""
    for i in 1:ni
        frac = grid[1, i, j] / cell_area
        line *= frac >= 0.75 ? "∎" : frac >= 0.25 ? "▣" : frac > 0.0 ? "·" : " "
    end
    println("  ", line)
end
