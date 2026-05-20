using Documenter, Lava

DocMeta.setdocmeta!(Lava, :DocTestSetup, :(using Lava); recursive=true)

makedocs(
    modules  = [Lava],
    sitename = "Lava.jl",
    authors  = "Simon Danisch and contributors",
    format   = Documenter.HTML(;
        canonical = "https://SimonDanisch.github.io/Lava.jl",
        edit_link = "main",
        assets    = String[],
    ),
    pages = [
        "Home"               => "index.md",
        "Installation"       => "installation.md",
        "Compute"            => "compute.md",
        "Graphics"           => "graphics.md",
        "Ray Tracing"        => "raytracing.md",
        "Performance"        => "performance.md",
        "Architecture"       => "architecture.md",
        "Debugging"          => "debugging.md",
        "Known Issues"       => "known_issues.md",
        "API Reference"      => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references],
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/SimonDanisch/Lava.jl.git",
    devbranch = "main",
    push_preview = true,
)
