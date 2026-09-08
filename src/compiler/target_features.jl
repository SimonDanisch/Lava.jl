# What the target device lets a SPIR-V module declare.
#
# The emitter has to know a few things about the hardware it is emitting for,
# because a capability declared on a device that lacks it is a validation error
# rather than a slow path. It used to learn them from a process-global record the
# runtime pushed when it bound a device, which answered for the BOUND device and
# not for the one a kernel was being compiled for: a raygen compiled for a second
# device while an NVIDIA card was bound declared `ShaderInvocationReorderNV` on
# hardware without it, and the frozen SPIR-V it produced was served to any device
# afterwards, because the key knew nothing about features either.
#
# So there is no global. The record is a field of `LavaCompilerParams`: part of
# the compile job, part of every cache key derived from the job, and filled by
# the runtime from the context it is compiling for. An emitter test with no
# device passes `TargetFeatures()`, which is the module that is valid everywhere.
#
# NOT `DeviceCaps`. That is `KernelInterface`'s and is deliberately portable:
# subgroups, workgroups, matrix shapes, things every GPU reports. These are
# Vulkan extensions that gate SPIR-V emission, so they are Lava's.

"""
    TargetFeatures(; ser = false, ray_query = false)

Vulkan extensions the emitter may declare capabilities for.

    ser         SPV_NV_shader_invocation_reorder: `OpHitObject*` and
                `OpReorderThreadWithHitObjectNV`. NVIDIA only; declaring it
                elsewhere fails validation, so a module emits the implicit
                `OpTraceRayKHR` fallback instead.
    ray_query   VK_KHR_ray_query: `OpRayQueryInitializeKHR` and friends, for
                tracing from a compute shader rather than a ray-tracing pipeline.

Both default to `false`: an emitter test with no device running gets the
conservative module, which is the one that is valid everywhere. A compile for a
device passes the device's record through `lava_compiler_config(; features)` or
the `features` keyword of `lava_compile_gpu` and `lava_compile_rt_shader`; the
Vulkan runtime keeps it on the context as `ctx.features`.
"""
Base.@kwdef struct TargetFeatures
    ser::Bool = false
    ray_query::Bool = false
end

# Content-hashed, so a frozen key that mixes it in is the same across sessions.
Base.hash(f::TargetFeatures, h::UInt) =
    hash(f.ray_query, hash(f.ser, hash(:TargetFeatures, h)))
