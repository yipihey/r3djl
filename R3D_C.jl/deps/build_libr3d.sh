#!/usr/bin/env bash
# build_libr3d.sh — build libr3d as a shared library for R3D_C.jl ccall use.
#
# Upstream r3d's CMake produces a static archive by default. We need a
# shared object so Julia can dlopen it. This script does a minimal build
# with -fPIC and links it as a shared library.
#
# Usage:
#   ./build_libr3d.sh /path/to/r3d/source [/path/to/install/prefix]
#
# After running, set:
#   export R3D_LIB=$INSTALL_PREFIX/lib/libr3d.so   (Linux)
#   export R3D_LIB=$INSTALL_PREFIX/lib/libr3d.dylib (macOS)
#
# This is a stopgap until we have a proper r3d_jll BinaryBuilder recipe.

set -euo pipefail

SRC="${1:-}"
PREFIX="${2:-$HOME/.local/r3d}"

if [[ -z "$SRC" ]]; then
    echo "Usage: $0 <r3d_source_dir> [install_prefix]" >&2
    echo "  Default install prefix: $HOME/.local/r3d" >&2
    exit 1
fi

if [[ ! -d "$SRC/src" ]] || [[ ! -f "$SRC/src/r3d.c" ]]; then
    echo "Error: $SRC does not look like an r3d source tree (missing src/r3d.c)" >&2
    exit 1
fi

BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "==> Building r3d shared library"
echo "    source:  $SRC"
echo "    build:   $BUILD_DIR"
echo "    install: $PREFIX"

cd "$BUILD_DIR"
cmake "$SRC" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fPIC -O3" \
    -DBUILD_SHARED_LIBS=ON \
    -DR3D_MAX_VERTS=512

make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make install

# Detect platform
case "$(uname)" in
    Linux)  EXT=so ;;
    Darwin) EXT=dylib ;;
    *)      EXT=so ;;
esac

LIB="$PREFIX/lib/libr3d.$EXT"
if [[ ! -f "$LIB" ]]; then
    # Some platforms install under lib64/
    LIB="$PREFIX/lib64/libr3d.$EXT"
fi

if [[ -f "$LIB" ]]; then
    echo
    echo "==> Built successfully."
    echo "    Set: export R3D_LIB=$LIB"
else
    echo "==> Build completed but libr3d.$EXT not found in $PREFIX/lib*" >&2
    exit 1
fi
