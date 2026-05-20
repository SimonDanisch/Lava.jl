using Documenter, Lava

DocMeta.setdocmeta!(Lava, :DocTestSetup, :(using Lava); recursive=true)

# Copy benchmark images into the docs build tree so relative <img> links
# resolve.  Anything outside `docs/src/` would otherwise be rejected by
# Documenter as a link pointing outside the build directory.
let src_bench = joinpath(@__DIR__, "..", "benchmarks"),
    dst_bench = joinpath(@__DIR__, "src", "assets", "benchmarks")
    if isdir(src_bench)
        isdir(dst_bench) || mkpath(dst_bench)
        for f in readdir(src_bench)
            endswith(f, ".png") || continue
            cp(joinpath(src_bench, f), joinpath(dst_bench, f); force=true)
        end
    end
end

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
    # Treat structural problems as warnings on first build so the deploy
    # is not gated on every dangling cross-reference or missing docstring.
    # Tighten this back to a list once the docs settle.
    warnonly = true,
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/SimonDanisch/Lava.jl.git",
    devbranch = "main",
    push_preview = true,
)
