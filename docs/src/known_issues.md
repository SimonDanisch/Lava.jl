# Known Issues

This page mirrors the canonical [`KNOWN_ISSUES.md`](https://github.com/SimonDanisch/Lava.jl/blob/main/KNOWN_ISSUES.md) at the repo root. Open issues are tracked there with reproducers.

```@eval
using Markdown
path = joinpath(dirname(pathof(Lava)), "..", "KNOWN_ISSUES.md")
isfile(path) ? Markdown.parse(read(path, String)) : Markdown.parse("KNOWN_ISSUES.md not found.")
```
