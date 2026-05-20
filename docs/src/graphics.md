# Graphics

Vertex, fragment, geometry, and tessellation shaders are written as plain Julia functions and compiled to SPIR-V. There is no GLSL, no shader file format, no external preprocessor — your shaders live in the same module as your CPU code, share types and helpers with the rest of your codebase, and are subject to the same Julia tooling.

!!! note "API stability"
    The graphics surface area is functional and exercised by real-world renderers, but the public API is still being refined. Expect minor breaking changes between 0.x releases.

## Pipelines

The high-level entry points are `Rasterizer`, `TrianglePipeline`, and `LinePipeline`. Each takes a vertex and fragment shader plus optional state (blend mode, cull face, depth test, topology):

```julia
using Lava, GeometryBasics

vertex(pos, mvp) = mvp * pos                       # → vec4 clip-space position
fragment(color)  = color                            # → vec4 RGBA

pipeline = Rasterizer(
    vertex   = vertex,
    fragment = fragment,
    blend    = :alpha,
    cull     = :back,
    depth    = :less,
    topology = :triangle_list,
)
```

The shader functions are regular Julia: they can call each other, dispatch on types, use packages (`GeometryBasics`, `LinearAlgebra`, `StaticArrays`, …), and are compiled per-instantiation just like compute kernels.

## Render targets and present

```julia
window = RenderWindow(width=1280, height=720, title="demo")
framebuffer = window.framebuffer

while window_should_close(window) == false
    draw!(pipeline, framebuffer; vertices=mesh, uniforms=params)
    present_frame!(window)
end
```

Offscreen rendering uses `LavaFramebuffer(width, height)` directly without opening a window:

```julia
fb = LavaFramebuffer(1920, 1080)
draw!(pipeline, fb; vertices=mesh, uniforms=params)
img = Array(fb)   # → Matrix{RGBA{N0f8}}
```

## Textures and samplers

```julia
img = LavaTexture2D(load("brick.png"))
sampler = LavaSampler(; filter=:linear, wrap=:clamp_to_edge)

fragment(uv, tex, smp) = sample(tex, smp, uv)
```

Textures and samplers participate in the same BDA arg-packing as buffer arguments; descriptor set management is hidden from the user.

## Geometry and tessellation

Geometry and tessellation control / evaluation stages are wired in the same way: you pass `geometry=...` and `tess_control=...` / `tess_eval=...` to `Rasterizer`. The custom emitter handles all five graphics stages — Lava does not delegate to `llc`, which still crashes on geometry shaders today.

## Pixel readback

For most workflows you'll just call `Array(framebuffer)` to grab the final image after `vk_flush!`. The conversion handles the GPU → CPU staging, pixel format, and row-pitch unswizzling.
