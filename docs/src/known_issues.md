# Known Issues

This page mirrors the canonical [`KNOWN_ISSUES.md`](https://github.com/SimonDanisch/Lava.jl/blob/main/KNOWN_ISSUES.md) at the repo root. Open issues are tracked there with reproducers.

```@eval
using Markdown
candidates = (
    joinpath(@__DIR__, "..", "..", "KNOWN_ISSUES.md"),
    joinpath(@__DIR__, "..", "KNOWN_ISSUES.md"),
)
path = nothing
for c in candidates
    if isfile(c)
        path = c; break
    end
end
path === nothing ?
    Markdown.parse("`KNOWN_ISSUES.md` could not be located.") :
    Markdown.parse(read(path, String))
```
