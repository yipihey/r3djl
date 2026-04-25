using Documenter
using R3D

DocMeta.setdocmeta!(R3D, :DocTestSetup, :(using R3D); recursive=true)

makedocs(;
    modules = [R3D],
    authors = "Tom Abel <tabel@stanford.edu>",
    sitename = "R3D.jl",
    repo = "https://github.com/yipihey/r3djl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://yipihey.github.io/r3djl",
        edit_link = "main",
        assets = String[],
        size_threshold_warn = 200_000,
        size_threshold = 400_000,
    ),
    pages = [
        "Home" => "index.md",
        "2D rasterization" => "2d.md",
        "3D voxelization" => "voxelize.md",
        "Performance" => "performance.md",
        "Internals" => "internals.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(;
    repo = "github.com/yipihey/r3djl.git",
    devbranch = "main",
    push_preview = true,
)
