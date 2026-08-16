# Open defect: hoisting Dielectric's BxDF blacks out HW-RT media scenes

Status: reproducible, localized, NOT fixed. The change is kept out of the tree;
`Dielectric` and `ThinDielectric` are the only materials that still resolve
their textures/IOR inside `sample_bsdf_spectral` rather than in `get_bxdf`.

## Symptom

Converting `Dielectric` to the pbrt-v4 `Material::GetBxDF` pattern (a
`DielectricEvaluated` carrier built once per hit) renders correctly everywhere
EXCEPT hardware ray tracing on scenes that also have participating media:

| scene (256 spp)              | SW energy | HW energy |
|------------------------------|-----------|-----------|
| mat_dielectric_light_point   | 0.9995    | 0.9995    |
| mat_dielectric_rough_*       | 1.015     | 1.015     |
| medium_null_interface_homog  | 0.9925    | 0.9919    |
| **medium_cloud_point**       | ok        | **0.035** |
| **medium_coffee_point**      | ok        | **0.684** |
| **medium_milk_point**        | ok        | **0.186** |
| **medium_smoke_point**       | 1.003     | **0.037** |

The failing scenes are exactly the ones whose medium boundary is
`Material "dielectric" "float eta" 1.0`. `medium_null_interface_homog` uses
`Material "interface"` (a null material) and is unaffected. The HW image is
essentially black — which is consistent with the contribution being carried
entirely by the continuation ray that the boundary hit is supposed to push,
since a specular BSDF contributes nothing to direct lighting.

## Reproducer

```julia
include("dev/Hikari/test/pbrt/suite.jl")
rec = render_scene("medium_smoke_point"; samples=32, hw_accel=true)
energy_ratio(ensure_reference("medium_smoke_point"), rec)   # 0.0369, want ~1.0
```

with the `get_bxdf(::Dielectric, ...)` version of `materials/dielectric.jl`
(kept at `/tmp/dielectric_get_bxdf_attempt.jl` during the session; it is a
mechanical port, identical in structure to `ConductorEvaluated`).

## What the bisection established

Each step below is a full rebuild + render of `medium_smoke_point` on HW.

1. **Not the maths, not the struct.** Keeping `DielectricEvaluated` and its
   BSDF methods but rebuilding the carrier INSIDE `sample_bsdf_spectral`
   (rather than hoisting into `get_bxdf`) renders correctly: energy 0.999.

2. **Not the compute path.** Hoisting in `vp_shade_material_kernel!`
   (surface-eval.jl) while leaving the chit on the raw material renders
   correctly: energy 0.999. So the defect is in the closest-hit shader.

3. **Not direct lighting.** Handing the hoisted carrier to
   `surface_direct_lighting_inner_typed!` but the raw material to
   `evaluate_material_inner_typed!` renders correctly: energy 0.999. So it is
   specifically the BSDF-sampling/continuation-push call.

4. **Not the struct layout.** Replacing the trailing `is_dispersive::Bool`
   with a `Float32` changes nothing — energy stays 0.0369, bit for bit.

5. **Not reading the struct at all.** Replacing every `bxdf.field` read in
   `sample_bsdf_spectral` with the literal constants that scene actually has
   (Kr = Kt = 1, eta = 1, roughness = 0) STILL gives 0.0369 on HW and 1.0027 on
   SW. At that point the sampling code is arithmetically identical to the
   working raw-material version; the only remaining difference is that
   `evaluate_material_inner_typed!` is specialized on `DielectricEvaluated`
   instead of on `Dielectric`, with the carrier passed in and unused.

So the trigger is the specialization itself, not any value it carries — which
points at the emitter rather than at Hikari. The reproducible bit-identical
energy across variants 4 and 5 argues against uninitialised memory and for a
deterministic miscompile.

## Where to look next

`evaluate_material_inner_typed!` is where the continuation ray is pushed
(`push!(next_ray_queue, ray_item)`), and a lost push explains the symptom
exactly. The next step is to dump the two chit modules — one specialized on
`Dielectric`, one on `DielectricEvaluated` — with `LAVA_DUMP_KERNELS=1` and
diff the SPIR-V around the work-queue append, rather than to keep bisecting
from the Hikari side.

Note this is NOT the same as the older AMD/RADV "HW RT volumetric trips
DEVICE_LOST" note: this reproduces on NVIDIA RTX 4000 Ada, driver 595.80, with
no device loss and no validation error.
