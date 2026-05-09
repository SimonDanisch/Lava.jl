# Known Issues

## RADV RT Crash After Many Compute Dispatches

**Status**: Unresolved (RADV driver bug)
**GPU**: Radeon 8060S (RADV STRIX_HALO)
**Mesa version**: 25.x (openSUSE Tumbleweed, April 2026)

### Symptom

GPUVM PERMISSION_FAULT (`VK_ERROR_DEVICE_LOST`) on the first RT dispatch (`vkCmdTraceRaysIndirectKHR`) after many prior compute dispatches from Hikari's VolPath integrator. The crash occurs during `vkQueueSubmit`, not during recording.

### Reproduction

```julia
using Hikari, GeometryBasics, Colors, Adapt
import Lava

backend = Lava.LavaBackend()

# Create and render ~150+ SW scenes (each allocates GPU buffers, dispatches
# ~2800 compute kernels via VolPath, then buffers are GC'd)
for i in 1:150
    scene = Hikari.Scene(backend=backend, hw_accel=false)
    # ... add lights, geometry, materials ...
    Hikari.sync!(scene)
    film = Adapt.adapt(backend, Hikari.Film(Point2f(128, 128)))
    camera = Hikari.PerspectiveCamera(...)
    Hikari.VolPath(samples=16, max_depth=8)(scene, film, camera)
end

# This RT render crashes:
scene = Hikari.Scene(backend=backend, hw_accel=true)
# ... add lights, geometry ...
Hikari.sync!(scene)
Hikari.VolPath(samples=4, max_depth=5, hw_accel=true)(scene, film, camera)
# -> GPUVM fault at rt_indirect dispatch
```

### Investigation Summary

Extensively debugged. All of the following were verified:

- **TLAS keepalive**: TLAS is correctly kept alive in `batch.data_refs` during RT dispatch (proven by MWE)
- **Buffer keepalive**: All `LavaArray` args pushed to `batch.data_refs` via the refactored dispatch path
- **Deferred frees**: Deferred free mechanism correctly defers VkBuffer destruction during recording
- **Pipeline caches**: Clearing all pipeline/kernel/descriptor caches does not fix the crash
- **Pool blocks**: Pool block VkBuffers are never freed during normal operation; pool chunk recycling is correct
- **AS memory**: BLAS/TLAS backing VkBuffers from old scenes are correctly freed by Vulkan.jl finalizers; 200+ RT build/dispatch/GC cycles without SW scenes do NOT crash
- **Vulkan validation layers**: No API-level validation errors detected
- **GPU-assisted validation**: Enabled `VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT`. No errors detected, but GPU-AV does NOT instrument `PhysicalStorageBuffer` (BDA) access - it only validates descriptor-based buffer access (`StorageBuffer`). Since Lava uses BDA exclusively (no descriptors for compute/RT buffer access), GPU-AV cannot catch out-of-bounds BDA pointer arithmetic. An intentional OOB BDA write also produces zero validation messages, confirming this limitation.

### What Triggers It

- Requires many Hikari VolPath SW compute renders (not simple KA kernels)
- Requires subsequent HW RT render
- The exact threshold depends on GC timing and allocation patterns (~50-150 SW scenes)
- Does NOT reproduce with: simple compute kernels, RT-only workloads, or if old scene references are kept alive (preventing GC)

### Workarounds

- `Lava.vk_reset_device!()` before the HW RT render (destroys all GPU state, requires scene rebuild)
- Keep old scene references alive (prevents buffer recycling)
- Render HW RT scenes before SW scenes in a session

### Update: Unaligned BDA Access Fixed (April 2026)

GPU-assisted validation initially reported "zero shader access violations" because the
`VK_EXT_validation_features` extension was only being checked in global instance extensions,
not in layer-provided extensions. After fixing that, GPU-AV caught unaligned BDA pointer
access (`OpStore at buffer device address ... is not aligned to Aligned operand of 4`).

**Root cause**: The SPIR-V emitter's `use_ptr_arithmetic` GEP path computed struct field
byte offsets by summing `_compute_type_size` of preceding fields without accounting for
alignment padding. For structs with `Bool` (i8) fields followed by 4-byte-aligned fields
(e.g. `VPRayWorkItem` with `specular_bounce::Bool` at offset 152 followed by
`any_non_specular_bounces::Bool` at 153 and then `[2 x i32]` at 156), the computed offset
was 2 bytes too small (154 instead of 156), causing misaligned stores.

**Fix**: Use `LLVM.API.LLVMOffsetOfElement` (the same API used for `MemberOffset` SPIR-V
decorations) to compute struct field offsets, guaranteeing they match LLVM's data layout.

The unaligned access was benign on AMD RDNA3/4 hardware (render completed correctly) but
violated the SPIR-V spec. Whether this also caused the RT crash after many dispatches
remains unconfirmed.

### Conclusion

The unaligned BDA access bug has been fixed. The RT crash after many SW dispatches followed
by HW RT may still occur on RADV. If it does, it is a driver bug - all known shader-level
issues have been resolved.
