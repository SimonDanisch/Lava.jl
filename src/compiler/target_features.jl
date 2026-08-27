# What the target device lets a SPIR-V module declare.
#
# The emitter has to know a few things about the hardware it is emitting for,
# because a capability declared on a device that lacks it is a validation error
# rather than a slow path. It used to learn them by reading `VK_CONTEXT_REF[]`
# and reaching into a `VkContext` — which made the compiler depend on the Vulkan
# runtime for two booleans, and was the only reason `compiler/` named `VkContext`
# at all.
#
# So the two booleans are their own record, owned here, and the runtime sets it
# when it binds a device. Lava's half of the split then has no Vulkan in it.
#
# **Process-global, deliberately and unchanged.** `VK_CONTEXT_REF` was already
# the process-wide current context, and the emitter is reached from compile paths
# that carry no device argument, so the scope of the answer is exactly what it
# was. It is one `Ref` holding one immutable record rather than a live handle,
# which is the smaller thing to have global. Two devices with different feature
# sets were already wrong here and still are — see `vk_reset_device!`, which
# rebinds both together.
#
# NOT `DeviceCaps`. That is `KernelInterface`'s and is deliberately portable —
# subgroups, workgroups, matrix shapes, things every GPU reports. These are
# Vulkan extensions that gate SPIR-V emission, so they are Lava's.

"""
    TargetFeatures(; ser = false, ray_query = false)

Vulkan extensions the emitter may declare capabilities for.

    ser         SPV_NV_shader_invocation_reorder — `OpHitObject*` and
                `OpReorderThreadWithHitObjectNV`. NVIDIA only; declaring it
                elsewhere fails validation, so a module emits the implicit
                `OpTraceRayKHR` fallback instead.
    ray_query   VK_KHR_ray_query — `OpRayQueryInitializeKHR` and friends, for
                tracing from a compute shader rather than a ray-tracing pipeline.

Both default to `false`: an emitter test with no device running gets the
conservative module, which is the one that is valid everywhere.
"""
Base.@kwdef struct TargetFeatures
    ser::Bool = false
    ray_query::Bool = false
end

const TARGET_FEATURES = Ref(TargetFeatures())

"""
    targetfeatures() -> TargetFeatures

What the bound device allows. All-`false` when no device is bound, which is the
documented case for emitter tests and is the only one that may answer `false`
without being asked: a device that exists but cannot report a flag is a bug, and
treating that as "unsupported" would silently emit the slower module for ever.
"""
targetfeatures() = TARGET_FEATURES[]

"""
    targetfeatures!(f::TargetFeatures)

Tell the emitter what the device allows. Called by the runtime when it binds or
releases a device, and by tests that want to emit for hardware they do not have.
"""
targetfeatures!(f::TargetFeatures) = (TARGET_FEATURES[] = f; f)
