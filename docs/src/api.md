# API reference

## Types

```@docs
R3D.Plane
R3D.Vec
R3D.Polytope
R3D.Vertex
R3D.StaticStorage
R3D.DynamicStorage
```

## Flat (recommended) variant

```@docs
R3D.Flat.FlatPolytope
R3D.Flat.FlatBuffer
R3D.Flat.StaticFlatPolytope
R3D.Flat.VoxelizeWorkspace
```

### Constructors

```@docs
R3D.Flat.box
R3D.Flat.tet
R3D.Flat.simplex
R3D.Flat.init_box!
R3D.Flat.init_tet!
R3D.Flat.init_simplex!
R3D.Flat.init_poly!
```

### Clipping and integration

```@docs
R3D.Flat.clip!
R3D.Flat.split!
R3D.Flat.split_coord!
R3D.Flat.moments
R3D.Flat.moments!
R3D.Flat.shift_moments!
R3D.Flat.is_good
```

### Voxelization / rasterization

```@docs
R3D.Flat.get_ibox
R3D.Flat.voxelize
R3D.Flat.voxelize!
```

### Affine transformations

```@docs
R3D.Flat.translate!
R3D.Flat.scale!
R3D.Flat.rotate!
R3D.Flat.shear!
R3D.Flat.affine!
```

## Module-level API (AoS reference path)

```@docs
R3D.box
R3D.tet
R3D.clip!
R3D.moments
R3D.moments!
R3D.num_moments
R3D.poly_center
R3D.signed_distance
```
