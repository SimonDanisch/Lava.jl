# Contributing to Lava

Lava produces **one SPIR-V module that has to run correctly on every Vulkan driver** — NVIDIA, RADV, AMDVLK, lavapipe, MoltenVK. The rules below exist to keep that promise as we accumulate driver workarounds.

## The cardinal rule: no vendor branching

* **No vendor-conditional code in `src/`.** No `if vendor == "NVIDIA" … else …`, no `if is_radv …`, no per-driver `@static`. When a driver miscompiles a pattern (e.g. NVIDIA dropping the `I[1]` term from a nested `dims[1]*dims[2]` stride product, or RADV miscompiling shared multi-use PSB `OpPtrAccessChain`), the fix is to change how the **emitter** produces that pattern so every vendor handles it. Driver names belong in comments as historical attribution, never in code paths.
* **No vendor-conditional tests.** No `if backend.device_name contains "AMD" skip ...`. The test identity is the source pattern; the assertion is the same on every platform. If the test fails on a new vendor, that is a real correctness bug — investigate it the same way as the original.
* **No vendor in test filenames.** A file called `test_nvidia_thing.jl` implies NVIDIA-only relevance; it isn't. Use names that describe the pattern (`test_psb_chain_fold.jl`, `test_repeat_inner_3d.jl`, `test_loop_unswitch_miscompile.jl`).

If you genuinely need to skip a test on a platform (e.g. a feature the device does not support, not a bug), gate on the **capability** (`ctx.ray_query_available`, `Vulkan.SUBGROUP_FEATURE_SHUFFLE_BIT`, …), never on the vendor string.

## Testing rules

* The full suite is `julia --project=/sim/Programmieren/VulkanDev dev/Lava/test/runtests.jl` (FULL_MODE is the default, no flag). It is the authoritative cross-platform check. Headline at the bottom must read `0 failed, 0 errors`.
* `LAVA_CI=1` runs a fast subset (~5 min on lavapipe / NVIDIA) for tight iteration; the full suite is still the merge gate.
* **Tier 3d (`SPIR-V Pattern Correctness & Stress`) is the firewall.** Each file in it pins a SPIR-V emitter pattern that has miscompiled at least once. If you touch the emitter or the LLVM pipeline, those four files must keep passing on every platform you build for.
* **Add a regression test for every emitter / pipeline bug you fix.** Put it in Tier 3d, name it for the pattern, include a comment block explaining: the symptom, which driver(s) surfaced it first (historical), and what the fix does. Pattern-level MWE plus an end-to-end check is ideal.
* Tests run on a real Vulkan device. CPU-only validation (compile + spirv-val, no execution) is fine for emission-shape tests and is preferred where it works — see `test/spirv/`.

## When a new driver surfaces a miscompile

The drill we used for the NVIDIA fixes this cycle:

1. **Bisect to the SPIR-V pattern.** Build a minimal MWE that fails (often a few-line kernel). Compare the failing emission to a working variant of the same operation.
2. **Decide whether the SPIR-V is valid.** Run it through `spirv-val` and the Vulkan validation layer. If it is invalid, the emitter is wrong; fix the emitter. If it is valid and the driver miscompiles it, the emitter still has to change — emit a different (also-valid) pattern that the driver handles. The user-visible behavior is correctness, on every platform.
3. **Pick a fix that is uniformly safe.** Loop-unswitch disabled at `opt_level≥2` (NVIDIA workaround) is also a no-op-or-improvement on AMD because unswitching is a pure optimization. Horner-form `linear_index` is just a rewrite of an integer expression; every driver evaluates it the same. The chain-fold replaces a shared intermediate `OpPtrAccessChain` with a fold-from-root; no driver loses anything by it. Look for fixes with that property.
4. **Pin it with a Tier 3d test.** Pattern in the name, attribution in the comment, assertion vendor-neutral. Include the MWE plus, ideally, an end-to-end check that exercises the path in real code (an AK op, a render, etc.).
5. **Re-run the full suite on every platform you can.** The merge bar is green on every platform. If it is now green on the driver that exposed the bug AND on every other driver, ship it.

## Recovering from `LavaCompilationError` on a new platform

If on a fresh device a kernel that builds elsewhere fails to compile with messages like `heap allocation (GC)` or `dynamic dispatch (type instability)`:

* First retry from a fresh Julia session. Revise stale state and Julia 1.12 world-age issues (look for `Detected access to binding ... in a world prior to its definition world`) routinely produce one-shot false positives that don't reproduce on rerun.
* If it really reproduces, dump the LLVM IR with `Lava.lava_compile_to_llvm(f, tt)` (or `Lava.lava_debug_path("lava_last.ll")`) and look for what survived. The error chain reports the **deepest user-code line that led to** the runtime call; the actual `julia.gc_alloc_obj` or `jl_apply_generic` is in the IR a few inlinings up.
* Type instability in the Hikari/Raycore call chain is *almost always* the real cause; fix it where it lives. Lava's job is to compile correct, type-stable Julia.

## What handoff is not

Handoff is *not* a list of "NVIDIA-specific fixes the next person should be careful around." There are no NVIDIA-specific fixes. There are general SPIR-V correctness fixes that NVIDIA happens to need most often, plus general AMD/RADV correctness fixes that AMD/RADV needs most often. The test suite, run on every platform, is what enforces them on everyone. A regression on AMD of one of "the NVIDIA fixes" surfaces as a Tier 3d failure on AMD, exactly as if the fix had never been made — and you fix it the same way.
