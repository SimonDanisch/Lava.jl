# Collapsing the material type explosion

What it is, why it was the dominant compile cost, and what it measured.

## The cost model

Hikari compiles ONE closest-hit shader per concrete material type that reaches
the device (`VPClosesthitTyped{T}`), plus one per-material compute shading
kernel (`vp_shade_material_kernel!{T}`). So the compile time of a scene scales
with its number of Julia material TYPES, not with its number of named
materials.

Crown has ~50 named pbrt materials, 5 material classes, and used to produce 12
concrete types. The other 7 were not modelling distinctions:

* **Constant vs texture.** A material field held whatever the caller passed —
  `Float32` for a constant, `Raycore.TextureRef{…}` for an image map,
  `CheckerboardTexture` for a procedural. Those are different types, so
  `Conductor{…,Float32,…}` and `Conductor{…,TextureRef{…},…}` were different
  materials. Crown alone had three Conductors that differ only in this.
* **The bump wrapper.** `BumpMapped{M, T}` wrapped the material it decorated,
  so a bumped and an unbumped gold were two types.
* **Mix pairs.** `MixMaterial{M1, M2, AmountTex}` embedded its two
  sub-materials, so every distinct PAIR was its own type. This one is the
  expensive case: a mix chit dispatches over every material type in the scene
  (rt-pipeline.jl, `T <: MixMaterial`), so it inlines the whole shading system.
  Crown had two of them.

pbrt-v4 has none of these problems, for reasons that port directly:
`FloatTexture` / `SpectrumTexture` are `TaggedPointer`s and a constant is a
`FloatConstantTexture` behind the same handle; `displacement` is a plain field
on `Material`; and `MixMaterial` holds `Material materials[2]`, i.e. two tagged
pointers rather than two embedded materials.

## The port

### `TexHandle`

A 32-byte isbits tagged union: `kind`, an inline `Float32`, an inline
`RGBSpectrum`, and a `(slot, idx)` pair for the out-of-line kinds
(`IMAGE`, `CHECKER`, `VERTEX_COLOR`). It replaces `TextureRef` in material
fields because `TextureRef` encodes its slot in its TYPE — which is precisely
the thing being erased. `Raycore.with_texture` supplies the missing piece: a
runtime `(slot, idx)` dispatch over the texture tuple, generated as an
if-elseif chain whose arms are each monomorphic.

The tag switch is paid once per hit inside `get_bxdf`, never in a BSDF inner
loop. That ordering is what makes it affordable, and it is why the `get_bxdf`
split (pbrt-v4's `Material::GetBxDF`) had to land first: without it the switch
would be inlined at each of the ~87 texture reads.

### Host and device forms

A material is routinely built before any scene exists — RayMakie constructs
`Diffuse(color_texture, …)` inside a plot's argument-conversion node, and the
Hikari scene only appears when the screen renders. A texture parameter
therefore has to survive on the host as the texture itself, and become a
handle only at push time, which is the first moment a texture store exists.

So the texture-carrying fields are TYPE PARAMETERS, and
`to_device_material` (hooked into `Raycore.maybe_convert_field` for
`::Material`) rewrites every one of them to a `TexHandle`. A field counts as
texture-carrying exactly when its declared type is a free type parameter of the
struct, so this needs no per-material table — concrete fields (`eta::Float32`,
`remap_roughness::Bool`, `max_depth::Int32`, `SetKey`) are left alone.

The pbrt path builds its handles once in `build_pbrt_textures` instead, so an
image map referenced by ten materials is uploaded once rather than ten times.

### `displacement` and `MixMaterialSpec`

`BumpMapped` is deleted; `displacement` is a field, defaulting to a NONE
handle, and `perturb_shading_frame_impl` is one implementation for every
material with a single predictable branch. The perturbation still happens at
intersection time, not inside the BSDF, so the integrator's own
`cos_theta = dot(wi, ns)` sees the bumped normal.

`MixMaterial` holds two `SetKey`s. Because keys only exist once the
sub-materials are pushed, the host-side form is `MixMaterialSpec`, which
carries the sub-materials until `resolve_material` pushes them — the same
two-stage shape pbrt-v4 has between its scene-entity and its `Material`.

## Results

### Type counts

| scene                       | before | after |
|-----------------------------|--------|-------|
| Crown                       | 12     | 5     |
| RayMakie materials benchmark| 11     | 10    |

Crown's 5 are exactly its 5 classes: `Diffuse`, `Conductor`, `CoatedDiffuse`,
`Dielectric`, `MixMaterial`. The RayMakie scene barely moves because it is a
material SAMPLER — it deliberately instantiates one of everything, so almost
every type there is a real class.

### Correctness

The pbrt reference suite at its normal settings (256 spp, tile p95 < 0.07,
energy ratio in [0.95, 1.05]):

| path | scenes | pass |
|------|--------|------|
| SW   | 151    | 151  |
| HW   | 151    | 151  |

The four scenes closest to the tile threshold were re-measured against a
pre-refactor checkout and came out identical to four decimal places
(`shadow_bumpgold_dome_over_velvet` 0.0699 both ways,
`mat_dielectric_rough_light_spot` 0.0465 both ways,
`mat_dielectric_rough_high_light_spot` 0.0419,
`medium_null_interface_homog` 0.0391) — so the refactor is numerically neutral
and those margins were already there.

RayMakie's materials benchmark renders unchanged, including the Perlin-textured
emissive and the roughness-mapped gold, which exercise the scene-less host path
where a material holds a raw `Texture` until push.

### Compile and render time

See TIMINGS section appended below, measured one scene per process — rendering
anything in the same session lets the second run reuse GPUCompiler's inference
cache for every shader the two have in common, which measures cache residency
rather than compilation.

## TIMINGS

Crown (`RayDemo/Crown/crown.pbrt`), NVIDIA RTX 4000 Ada, driver 595.80,
Julia 1.12.6, GPUCompiler 2.1.1. Hikari `bf0ffcc` (before) against `b69b1ad`
(after); Lava `cdfcf66` in both. One scene per process, cold cache.

### Compile

`cold_seconds` is the first `render_pbrt` after the scene is built.
`first_seconds` is the same quantity measured by the other harness, which
builds the scene itself first.

**These are not interchangeable, and the difference is not noise.**
`render_pbrt` re-parses and REBUILDS the scene, so `cold_seconds` is
`second_build + compile`. On crown the second build is ~20 s and the two
metrics agree to within 10 % (232.9 vs 212.5 s on 2026-08-18). On
`RayDemo/Materials/materials.pbrt` the build is ~394 s — the scene is 20
spheres, each tessellated at `segments=512` into ~524 k triangles
(`scene_builder.jl:925`), so ~10.5 M triangles get a BVH built twice — and
`cold_seconds` reads 632.7 s against a true compile time of 238.6 s. Any
cross-revision comparison on a sphere-heavy scene must use `first_seconds`;
`cold_seconds` there is mostly measuring tessellation.

| | before (12 types) | after (5 types) | change |
|---|---|---|---|
| HW RT, cold render   | 714.6 s | 212.2 s | **−70.3%**  (3.37×) |
| HW RT, first render! | 705.3 s | 190.5 s | **−73.0%**  (3.70×) |
| SW, cold render      | 194.1 s | 128.7 s | **−33.7%**  (1.51×) |
| SW, first render!    | 161.8 s | 103.1 s | **−36.3%**  (1.57×) |

Phase attribution, HW RT:

| phase   | before   | after    | change |
|---------|----------|----------|--------|
| pass    | 532.2 s  | 124.8 s  | −76.5% |
| stage   |  21.5 s  |  21.5 s  |   0    |
| emitfn  |   9.4 s  |   6.6 s  | −29.8% |
| emit    |   3.1 s  |   3.3 s  |  +4.5% |

`pass` is the LLVM pass pipeline the RT shaders run through, and it absorbs
essentially the whole win — which is what the type count predicts, since each
material type is another chit through that pipeline. `stage` (the compute-side
phases) is identical to the tenth of a second, a useful control: the change did
not make individual kernels cheaper, it made there be fewer of them.

SW moves less and differently — `stage` 104.7 s → 60.2 s, `pass` 40.8 s →
30.2 s — because the SW path's cost is spread over per-material COMPUTE kernels
rather than concentrated in chits.

### Render

Per-sample `render!`, best and median of five, after compilation:

| | before | after | change |
|---|---|---|---|
| HW RT, best   | 0.1220 s | 0.1108 s | **−9.2%** |
| HW RT, median | 0.1238 s | 0.1117 s | **−9.8%** |
| SW, best      | 0.2356 s | 0.2390 s | +1.4% |
| SW, median    | 0.2385 s | 0.2398 s | +0.5% |

The handle indirection was expected to cost a little and instead HW got faster.
The plausible reason is that it is paid once per hit in `get_bxdf` while what
it removes is per-shader: 12 chits collapse to 5, so the shader binding table
and the instruction footprint both shrink, and the bump path becomes one
predictable branch instead of a separate shader. SW is inside run-to-run noise
either way (the 5-sample spread is ~0.03 s on its slowest sample).

Two caveats worth keeping with these numbers:

* Scene build time is unchanged (66–83 s for Crown either way) and is NOT in
  the compile figures. It is pbrt parsing and BVH construction.
* The `warm_seconds_8spp` column in the raw logs includes a full scene rebuild
  per call, because `render_pbrt` reparses. Use the per-sample table above for
  render cost; that is what `bench_render_time.jl` exists for.
