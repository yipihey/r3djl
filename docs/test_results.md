# Verified test results

Run on Linux x64, Julia 1.11.4, libr3d built with gcc 13.3.0 -O3 -fPIC,
R3D_MAX_VERTS=512, R3D_LIB=$libr3d/libr3d.so.

```
$ julia --project=test_env R3D.jl/test/runtests.jl

Test Summary:  | Pass  Total  Time
R3D pure-Julia |   33     33  0.1s
Test Summary:                | Pass  Total  Time
Differential vs upstream r3d |  158    158  0.0s
Test Summary:          | Pass  Total  Time
R3D.Flat (SoA variant) |  408    408  0.0s
```

**599 tests, all passing.**

Test breakdown:

- **R3D pure-Julia (33)**: closed-form sanity checks of the AoS reference
  implementation. Construction, volumes, centroids, single-plane and
  diagonal clips, type stability across `Float64`/`Float32`, capacity
  trait dispatch.
- **Differential vs upstream r3d (158)**: same operations replayed
  against the C library via `R3D_C.jl`. Vertex-position agreement after
  `init_box`, moments orders 0 through 3 agreement (each order
  contributes (n+1)(n+2)(n+3)/6 individual moment comparisons), 100
  random clip stress trials with up to 4 planes each.
- **R3D.Flat SoA variant (408)**: closed-form tests of the flat
  implementation, plus 200 random three-way (AoS = Flat = C) cross-
  validation trials with up to 6 planes per trial. Each trial
  contributes ~2 assertions (AoS vs C and Flat vs C).
