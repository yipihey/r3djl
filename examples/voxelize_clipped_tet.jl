# Voxelize a clipped tetrahedron onto a 3D grid.
#
# Run from the repo root:
#   julia --project=R3D.jl examples/voxelize_clipped_tet.jl

using R3D

# Build a tet, clip it by an oblique plane to make the geometry interesting.
poly = R3D.Flat.tet((0.0, 0.0, 0.0),
                    (1.5, 0.0, 0.0),
                    (0.0, 1.5, 0.0),
                    (0.0, 0.0, 1.5))
plane = R3D.Plane{3,Float64}(R3D.Vec{3,Float64}([1.0, 1.0, 1.0] / sqrt(3)),
                              -0.7 / sqrt(3))
R3D.Flat.clip!(poly, [plane])

# Pre-allocate a workspace + destination grid for the hot loop.
ws = R3D.Flat.VoxelizeWorkspace{3,Float64}(64)

ni = nj = nk = 32
d = (1/ni, 1/nj, 1/nk)
nmom = R3D.num_moments(3, 1)        # volume + first moments per voxel
grid = zeros(Float64, nmom, ni, nj, nk)

R3D.Flat.voxelize!(grid, poly, (0, 0, 0), (ni, nj, nk), d, 1; workspace = ws)

println("Total volume on grid:    ", sum(@view grid[1, :, :, :]))
m = R3D.Flat.moments(poly, 1)
println("Total volume from moments: ", m[1], "  (should match)")

# Centroid recovered by summing first moments / volume:
total_vol = sum(@view grid[1, :, :, :])
total_x   = sum(@view grid[2, :, :, :])
total_y   = sum(@view grid[3, :, :, :])
total_z   = sum(@view grid[4, :, :, :])
println("Recovered centroid: ($(total_x/total_vol), $(total_y/total_vol), $(total_z/total_vol))")
println("Direct centroid:    ($(m[2]/m[1]), $(m[3]/m[1]), $(m[4]/m[1]))")
