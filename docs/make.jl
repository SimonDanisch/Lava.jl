using Documenter, Lava

DocMeta.setdocmeta!(Lava, :DocTestSetup, :(using Lava); recursive = true)

makedocs(
    modules  = [Lava],
    sitename = "Lava.jl",
    authors  = "Simon Danisch and contributors",
    format   = Documenter.HTML(;
        canonical = "https://SimonDanisch.github.io/Lava.jl",
        edit_link = "main",
    ),
    pages = [
        "Home"                 => "index.md",
        "Compiling"            => "compiling.md",
        "Device code"          => "device.md",
        "Device vocabulary"    => "vocabulary.md",
        "Graphics stages"      => "graphics.md",
        "Architecture"         => "architecture.md",
        "Debugging"            => "debugging.md",
        "API reference"        => "api.md",
        "Vocabulary reference" => "vocabulary_api.md",
    ],
    # Every public docstring has to appear in the manual.
    checkdocs = :public,
)

deploydocs(
    repo         = "github.com/SimonDanisch/Lava.jl.git",
    devbranch    = "main",
    push_preview = true,
)
