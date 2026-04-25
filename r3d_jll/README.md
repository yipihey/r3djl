# r3d_jll

This directory contains a [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil)
[BinaryBuilder](https://docs.binarybuilder.org/stable/) recipe for the upstream
C library [`r3d`](https://github.com/devonmpowell/r3d) (Powell & Abel's
exact polyhedral clipping / moments routines).

Once this recipe is merged into Yggdrasil and the resulting JLL is
auto-registered, users will be able to install the binary with

```julia
pkg> add r3d_jll
```

instead of cloning `r3d` and running `gcc` by hand. The JLL exposes a
single library product, `libr3d`, configured with `R3D_MAX_VERTS = 512`
and `R2D_MAX_VERTS = 256` (the upstream default).

## Files

- `build_tarballs.jl` — the BinaryBuilder recipe. Pins upstream r3d to
  a specific commit, writes the small `r3d-config.h` header, and
  compiles the four C sources (`r3d.c`, `v3d.c`, `r2d.c`, `v2d.c`)
  into a single shared library installed at `${libdir}/libr3d.${dlext}`.

## Testing the recipe locally

You need Docker (Linux/macOS) running and `BinaryBuilder.jl` installed
into a temporary environment:

```bash
mkdir -p /tmp/r3d-bb && cd /tmp/r3d-bb
julia --project=. -e 'using Pkg; Pkg.add("BinaryBuilder")'
cp /path/to/R3D.jl/r3d_jll/build_tarballs.jl .
julia --project=. build_tarballs.jl --verbose --debug
```

To restrict to one platform while iterating (much faster):

```bash
julia --project=. build_tarballs.jl --verbose --debug x86_64-linux-gnu
```

Successful builds drop tarballs in `./products/` and a deploy log under
`./build/`. To smoke-test the resulting `.so`, `dlopen` it and confirm
that `r3d_clip` and friends resolve.

## Submitting to Yggdrasil

1. Fork [`JuliaPackaging/Yggdrasil`](https://github.com/JuliaPackaging/Yggdrasil).
2. Add the recipe at `R/r3d/build_tarballs.jl` in your fork.
   Yggdrasil groups recipes alphabetically into single-letter top-level
   directories, with the case of the first character preserved
   (e.g. `R/RDKit/`, `R/RNAstructure/`). Lowercase project names like
   `r3d` still go under `R/` — there is no separate `r/` tree.
3. Open a PR titled e.g. `[r3d] new builder, version 0.1.0`. Yggdrasil's
   CI will build the recipe on every supported platform; once merged
   the bot opens a registration PR for `r3d_jll` against the General
   registry automatically.

Useful references:

- Yggdrasil README contributor section
- BinaryBuilder docs: https://docs.binarybuilder.org/stable/
- A similar pure-C, header-and-four-files recipe to use as a template:
  any of the small libraries already living in `R/` (e.g. `R/RE2/`'s
  layout, ignoring its much larger build script).

## Switching `R3D_C.jl` to use `r3d_jll`

`R3D_C.jl` currently locates the shared library through an environment
variable:

```julia
const libr3d = ENV["R3D_LIB"]
```

Once `r3d_jll` is published this becomes a one-line change:

```julia
using r3d_jll: libr3d
```

and `R3D_C.jl`'s `Project.toml` gains

```toml
[deps]
r3d_jll = "<UUID assigned by the registrator bot>"

[compat]
r3d_jll = "0.1"
```

The hand-build instructions can then be deleted from the top-level
`R3D.jl` README, and CI no longer needs to compile r3d before running
tests — the JLL artifact will be downloaded automatically by `Pkg`.
