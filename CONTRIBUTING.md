# Contributing

Thanks for your interest. This repo holds three Julia packages developed
together (`R3D`, `R3D_C`, `R3DBenchmarks`) and a BinaryBuilder recipe
(`r3d_jll`).

## Getting set up

```bash
git clone https://github.com/yipihey/r3djl.git
cd r3djl

# Build the C library (until r3d_jll is published):
./R3D_C.jl/deps/build_libr3d.sh /path/to/local/r3d-source $HOME/.local/r3d
export R3D_LIB=$HOME/.local/r3d/lib/libr3d.dylib   # or .so on Linux

# Test
cd R3D.jl
julia --project=. -e 'using Pkg; Pkg.develop(path="../R3D_C.jl"); Pkg.add(["Test", "Random", "ForwardDiff"])'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Without `R3D_LIB`, the differential tests skip silently and the
pure-Julia tests still run.

## Running benchmarks

```bash
cd /path/to/r3djl
julia --project=R3DBenchmarks.jl -e 'using R3DBenchmarks; R3DBenchmarks.run_all()'
```

## Code style

- Run `julia --project=. -e 'using JuliaFormatter; format(".")'` before
  committing. The repo's `.JuliaFormatter.toml` is the source of truth.
- Default to **no comments** in code. Add a comment only when the *why*
  is non-obvious — a hidden invariant, a subtle algorithmic choice, a
  workaround for a specific upstream bug.
- Match the existing line-for-line port pattern: when porting a piece of
  `r3d.c` / `r2d.c` / `v3d.c` / `v2d.c`, name your function the same way
  and reference the upstream source line in the docstring.

## What needs differential tests

Anything that touches `clip!`, `moments!`, `split!`, `voxelize!`, the
moment recursion, or vertex-graph connectivity must add a randomized
differential test against the C library. The pattern is established in
`R3D.jl/test/runtests.jl` — copy from one of the existing testsets.

## Reporting bugs

If you find a numerical disagreement with the C library, please attach
a minimal reproducer (polytope, plane(s), order) and the output of both
implementations. The fastest fix path is to add the failing case to the
existing differential testset and let CI demonstrate the bug.
