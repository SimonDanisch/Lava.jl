#
# Lava's implementation of `KernelInterface` (KI).
#
# KI is the cross-backend device/host/launch contract from
# `KernelAbstractions/lib/KernelInterface`. Every backend implements it, and
# kernels written against it compile on all of them — that is the whole point,
# and it is why these names, not Lava's own, are what Lava uses internally.
#
# The cooperative-matrix vocabulary that used to be a second package called
# `KernelInterfaces` — plural, UUID ce25d451 — is part of KI now; `Lava.jl`
# imports those names at the top. `matrix_shapes` below is the backend hook it
# gained in the move.
#
# ── Conventions this file reconciles ────────────────────────────────────────
#
# KI returns **1-based** indices, as `@NamedTuple{x::Int, y::Int, z::Int}` for
# the 3D queries. SPIR-V's builtins are 0-based vectors of `u32`. So every id
# gains one here, and every *size* does not — a count is a count in both.
#
# The raw intrinsics stay 0-based on purpose: they feed `subgroup_shuffle` and
# friends, which are specified in SPIR-V's numbering, and `shuffle(v, lane())`
# would read the neighbouring lane if `lane()` were shifted. The offset belongs
# at this boundary and nowhere else.

import KernelInterface as KI

# ── Indexing ────────────────────────────────────────────────────────────────

@inline KI.get_global_id() = (x = Int(lava_global_invocation_id(1)) + 1,
                              y = Int(lava_global_invocation_id(2)) + 1,
                              z = Int(lava_global_invocation_id(3)) + 1)

@inline KI.get_local_id() = (x = Int(lava_local_invocation_id(1)) + 1,
                             y = Int(lava_local_invocation_id(2)) + 1,
                             z = Int(lava_local_invocation_id(3)) + 1)

@inline KI.get_group_id() = (x = Int(lava_workgroup_id(1)) + 1,
                             y = Int(lava_workgroup_id(2)) + 1,
                             z = Int(lava_workgroup_id(3)) + 1)

@inline KI.get_num_groups() = (x = Int(lava_num_workgroups(1)),
                               y = Int(lava_num_workgroups(2)),
                               z = Int(lava_num_workgroups(3)))

# `get_local_size` and `get_global_size` are NOT implemented yet, and the reason
# is a bug they uncovered rather than an oversight.
#
# Both need the workgroup size, and `lava_workgroup_size` (BUILTIN_3D in
# runtime/intrinsics.jl) emits it as an Input *variable*. Vulkan does not allow
# that: spirv-val rejects the module with
#
#     BuiltIn decoration on target '%__spirv_BuiltInWorkgroupSize'
#     must be a constant for WorkgroupSize
#
# The workgroup size is fixed when the kernel is compiled — the launch plan
# picks it — so SPIR-V wants it as an `OpConstantComposite` carrying the
# decoration, not a load. `lava_workgroup_size` was declared and never called by
# anything in src/ or test/, so it has produced invalid SPIR-V unnoticed; this
# was its first caller. Emitting the constant is the fix, and it belongs with
# the emitter rather than smuggled in here.
#
# `get_global_size` is then `num_groups * local_size` per dimension, which is
# exact: a dispatch covers whole workgroups, and that is what makes the tail
# real for a kernel to bound itself against.

# ── Subgroups ───────────────────────────────────────────────────────────────

@inline KI.get_sub_group_size()     = lava_subgroup_size()
@inline KI.get_num_sub_groups()     = lava_num_subgroups()
@inline KI.get_sub_group_id()       = lava_subgroup_id() + UInt32(1)
@inline KI.get_sub_group_local_id() = lava_subgroup_local_id() + UInt32(1)

# `get_max_sub_group_size` is missing on purpose: SPIR-V puts SubgroupMaxSize
# under the Kernel (OpenCL) capability and spirv-val rejects it in a Vulkan
# module. Getting the device's maximum into a shader means a specialization
# constant fed from `subgroup_size_control(ctx).max` — its own task.

# `shfl_down` is KI's name for the down-shuffle. Lava generates the family for
# six element types; KI's `shfl_down_types` is what tells its own test suite
# which ones to exercise, so the two lists are derived from one place.
const KI_SHFL_TYPES = (Float32, Float64, Int32, UInt32, Int64, UInt64)

for T in KI_SHFL_TYPES
    @eval @inline KI.shfl_down(val::$T, offset::Integer) = subgroup_shuffle_down(val, offset)
end

# `KI.shfl_down_types(::LavaBackend)` reads this list but dispatches on a
# backend, which is a host handle — so it is in the host half, and this is the
# one list both sides derive from.

# `sub_group_reduce_add` is KI's name for the reduce-add, and Lava's
# `subgroup_add` is the SPIR-V `OpGroupNonUniformFAdd`/`IAdd` behind it.
#
# Its own type list rather than a reuse of `KI_SHFL_TYPES`, even though the six
# coincide today: the shuffle family and the reductions are separate SPIR-V
# capabilities, and a backend can have one without the other — Metal's
# `simd_sum` has no `Float64` at all. Tying them to one tuple would make that
# divergence unrepresentable.
const KI_REDUCE_ADD_TYPES = (Float32, Float64, Int32, UInt32, Int64, UInt64)

for T in KI_REDUCE_ADD_TYPES
    @eval @inline KI.sub_group_reduce_add(val::$T) = subgroup_add(val)
end

# ── Barriers ────────────────────────────────────────────────────────────────
#
# `@lava_device_override`, not a plain method, and the difference matters here in
# a way it does not above. KI gives `barrier` a HOST method that errors, so a
# plain method would shadow it and a host-side call would reach an `llvmcall` of
# a SPIR-V intrinsic on the CPU. The index queries above are bare stubs with no
# host method, so a plain method shadows nothing.
#
# Same instruction `KA.@synchronize` lowers to: `OpControlBarrier Workgroup
# Workgroup` with AcquireRelease | WorkgroupMemory | MakeAvailable | MakeVisible
# (emit.jl:7529). One barrier, two portable spellings.
@lava_device_override @inline KI.barrier() = lava_workgroup_barrier()

# `KI.sub_group_barrier` is DELIBERATELY not implemented. A subgroup-scoped
# barrier is a different instruction — `OpControlBarrier Subgroup Subgroup` — and
# lowering it means teaching that path a scope argument.
#
# Aliasing it to the workgroup barrier would "pass" while synchronising the
# wrong set of invocations, so the method is left missing: KI's own convention
# is that a capability a backend lacks is a missing method, not a wrong one, and
# KI's host method then reports it by name.

# ── Workgroup memory ────────────────────────────────────────────────────────
#
# `KI.localmemory` is NOT implemented, and the reason is KI's signature rather
# than anything about Vulkan.
#
# Lava's workgroup memory is an LLVM global in addrspace(3), created by
# `lava_alloc_shared(::Val{Id}, ::Type{T}, ::Val{N})`, and the global is NAMED
# `lava_shared_$Id`. `KA.SharedMemory(T, Val(Dims), Val(Id))` carries that `Id`
# because `@localmem` mints one per call site; `KI.localmemory(T, Val(Dims))`
# has no such parameter, so the only key available is `(T, Dims)`.
#
# Two `localmemory(Float32, (16, 16))` calls in one kernel would then be one
# buffer. That is not an error anything can raise — the second write silently
# lands on the first tile — so the method is left missing rather than
# implemented wrong. A generated function cannot mint the id itself: its body is
# cached per signature, and a counter inside it would not be pure.
#
# `KA.@localmem` is the working path on Lava and is unaffected. Closing this
# needs the id in KI's signature, which is an upstream change.

# ── Printing ────────────────────────────────────────────────────────────────
#
# The same builder `KernelAbstractions.__print` uses (device/printf.jl), so the
# two portable spellings cannot print differently. `@lava_device_override` for
# the reason `barrier` is: KI's `_print` has a host fallback that prints with
# `Base.print`, and that has to keep working off-device.
@lava_device_override @inline KI._print(items...) = lava_device_print(items...)

# The HOST half of KI — `synchronize`, `allocate`, the device queries and the
# launch — is in `array/kernelinterface_host.jl`. It needs a queue, a pool and a
# batch; nothing above this line does. See that file's header for the split.
