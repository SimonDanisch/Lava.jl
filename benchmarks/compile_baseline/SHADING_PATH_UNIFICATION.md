# RESOLVED: two surface-shading implementations, one of them stale

Status: **fixed** in Hikari by deleting the second implementation
(`integrators/volpath/surface-eval.jl`, `volpath.jl`, `materials/dispatch.jl`).
`Dielectric` and `ThinDielectric` now use the pbrt-v4 `Material::GetBxDF`
pattern like the other eight materials, so Phase 1 is 10/10.

This file first recorded the bug as a SPIR-V emitter defect, then as a missing
`get_bxdf` call. Both were symptoms. Keeping the wrong turns because the way
they were wrong is the useful part — see the two sections at the bottom.

## Symptom

Converting `Dielectric` to `get_bxdf` + `DielectricEvaluated` rendered
correctly everywhere EXCEPT hardware RT on scenes with participating media
whose boundary is `Material "dielectric" "float eta" 1.0`:

| scene (16 spp)               | SW energy | HW energy |
|------------------------------|-----------|-----------|
| mat_dielectric_light_point   | 1.005     | 1.005     |
| medium_null_interface_homog  | 0.994     | 0.995     |
| **medium_smoke_point**       | 1.004     | **0.037** |
| **medium_cloud_point**       | 0.971     | **0.036** |
| **medium_milk_point**        | 1.012     | **0.185** |
| **medium_coffee_point**      | 1.006     | **0.681** |

## Root cause

Surface shading had TWO implementations of the same logic:

* **`vp_shade_material_kernel!{T}`** (surface-eval.jl) — one kernel per concrete
  material type, `material_of_type` + `get_bxdf` + BSDF on the carrier. Used by
  software BVH, and the closest-hit shaders call the same inner functions.
* **`vp_shade_surface_hits_kernel!`** — a `with_index` switch over the whole
  material set, calling `sample_bsdf_spectral` on the RAW material.

`diff`ing the two direct-lighting bodies gave back comment rewording, a `mat::M`
parameter and the material lookup; Sobol dimensions, light selection, MIS and
the shadow-ray push were duplicated line for line, ~320 lines against ~200.

Which one ran was decided at volpath.jl:701 by `accel isa Lava.HWAdaptedAccel`,
inside `if !chit_owns_surface`. So: SW → typed, HW without media → neither (the
chit shades inline), **HW with media → the `with_index` copy**, and nothing
else.

Phase 1 moved every material's BSDF onto a `get_bxdf` carrier and updated the
typed copy. The `with_index` copy kept calling the raw material, which after
conversion matched only a `sample_bsdf_spectral(::Material, ...)` catch-all —
**a gray Lambertian with albedo 0.5**. It stayed invisible because a catch-all
is a method, not an error, and because `Dielectric` was converted last: as long
as it kept a raw BSDF method the medium boundary itself still shaded correctly.
Converting it made a specular straight-through boundary opaque, and every path
was trapped inside the medium. 0.037 is the camera-visible floor.

## The fix

Delete the second implementation. `vp_shade_surfaces!` (formerly
`vp_shade_typed!`) now drains the per-material queues on every backend, and the
`isa` branch is gone.

This is a drop-in, not a port: both producers already fill the per-material
queues through the same `enqueue_after_intersection!` — the surface trace for
non-medium rays (intersection.jl) and delta tracking's survive-to-surface path
(delta-tracking.jl:273) — and the typed queues carry 4-byte `TypedHitRef{T}`
indices into the very `hit_surface_queue` the deleted kernel was draining. Both
producers also resolve MixMaterial and handle the null-material medium swap at
the push site, which is everything the deleted kernel did beyond shading. It was
redundant as well as divergent.

Also deleted, same pass: `vp_trace_rays_kernel!` and its dispatchers (a second
tracing implementation, dead since cd02567 fused tracing into
`vp_trace_and_shade_kernel!`), and the gray Lambertian itself. `MixMaterial` and
`Emissive` still need BSDF methods — `foreach_type` generates a kernel for every
type in the set — but they now return `SpectralBSDFSample()` and
`(SpectralRadiance(), 0f0)`. An unreachable path that renders black is one you
notice.

`accel isa Lava.HWAdaptedAccel` became `shades_surfaces_inline(accel)`: generic
method `false` in volpath.jl, `HWAdaptedAccel` method in hw-rt.jl beside the
other accel-axis overrides. What survives as a boolean is
`chit_owns_surface = shades_surfaces_inline(accel) && isempty(media)`, which is
not an implementation choice but "did the trace already do this work" — and
media presence is scene data, not a type.

Net: **849 deletions, 252 insertions.**

## How the duplication arose

Four commits, each locally reasonable:

1. **b917d89** (Jun 5) — per-material chit slots land.
   `surface_direct_lighting_inner_typed!` / `evaluate_material_inner_typed!` are
   added as a COPY of the existing bodies with `mat::M` threaded in. The
   original stays. Two implementations from here on.
2. **0ca4c20** (Jun 5) — the chit turns out not to own surface shading when
   media are present; medium hits were being dropped (3–6 % of reference on
   cloud/smoke/coffee/milk). Gate becomes
   `chit_owns_surface = hw && isempty(media)`, and HW-with-media drains through
   `vp_shade_typed!` — still one implementation in use.
3. **cd02567** (Jun 10) — crown dispatch-overhead overhaul, item 3: "on the HW
   chit path with media, the 12 per-material vp_shade_typed dispatches +
   emitters ran every round only for the rare medium-survive-to-surface hits.
   Those hits now shade in ONE monolithic vp_shade_surface_hits! dispatch". The
   `isa` branch appears; the copy goes live for one configuration.
4. **Phase 1** (Jun 4 →) — materials move to `get_bxdf` one at a time. Each
   conversion updates the copy that SW and the chit exercise. The other one is
   never touched.

A copy made for a good reason, revived for a good reason in one narrow
configuration, then starved of the maintenance the original got.

## Why the first diagnosis was wrong

The first investigation bisected from the Hikari side inside ONE long-lived
`bt_julia_eval` session, and concluded "the trigger is the specialization
itself, not any value it carries — which points at the emitter".

Every step after the first was measuring the same compiled shader. Revise does
not reliably apply an edit before the next eval in that session — it was
observed lagging by minutes. `Lava.clear_kernel_cache!()` +
`empty!(Hikari._VP_RT_PIPELINES)` cannot help with code that was never revised.
Different variants all returned `energy = 0.03690232956015418` — bit-identical,
including one that replaced every carrier field with a literal constant, and
including a revert to plain HEAD. That bit-identity across changes that must
alter the image was the tell, and it was read as "deterministic miscompile"
instead of "nothing recompiled".

Rendering each variant in a FRESH process fixed the loop. An unconditional
marker written into `pixel_L` then showed up (energy 1 → 121), proving the probe
worked and the earlier runs had not been rebuilding. From there `@lava_printf`
on one pixel walked chit → medium queue → delta tracking → enqueue → shade to
the exact divergence.

Two rules for the next one:

* Verify the BASELINE in the same harness before attributing a failure to a
  change. HEAD on HW was never measured for these scenes; had it been, the four
  "regressions" would have shown up as pre-existing on the first run.
* If a code change cannot alter the output, stop and prove the code is running
  before theorising about why the output is wrong.

## Why the second diagnosis was wrong

The second fix taught the `with_index` copy to call `get_bxdf` too. It was
correct — 450/450 SW and 450/450 HW at 256 spp, and the four medium scenes back
at ~1.0 — and it was still the wrong fix: it made the two copies agree again
rather than removing the reason they could disagree, and it cost compile time,
because that kernel then inlined five real BSDFs where it had been inlining five
gray-Lambertian stubs. Crown HW cold compile went 212.2 s → 293.0 s while crown
SW, which never ran that kernel, stayed put (128.7 → 131.0 s).

That asymmetry is what exposed it. A fix whose cost lands on exactly one
configuration is a fix applied to a copy.
