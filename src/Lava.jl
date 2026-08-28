module Lava

# ── what a Julia→SPIR-V compiler exposes ─────────────────────────────────────
#
# 115 names left this list on 2026-08-27 when the runtime moved to Mantle:
# `LavaArray`, `LavaBackend`, `BatchQueue`, every graphics pipeline and texture
# type, all of ray tracing, `fft`/`gemv`, `vk_reset_device!`. They are Mantle's
# exports now. What is left is the compiler, the code a kernel is compiled
# against, and the vocabulary both sides name.
#
# `CoopMat`, `Scalar` and `matriximpl` went too, and are not in Mantle either:
# they were exported here and defined nowhere, left over from an earlier
# refactor. Checking which of 208 exports still resolved is what surfaced them.

# The device-side array — a `(pointer, dims)` pair, which is what a kernel gets.
# The host-side `LavaArray` that fills one in is Mantle's.
export LavaDeviceArray

# Compilation.
export CompilationResult, lava_compile, optimize_spirv, LavaGfxShader

# `export dump_state, gpu_memory_usage` was here, and both went to Mantle with
# the runtime — they read a `VkContext`. Exporting a name this module does not
# define is legal and silent, and worse than useless: `using Lava` then brings an
# undefined binding into scope which SHADOWS Mantle's real export, so
# `gpu_memory_usage()` in a file that had loaded both was an UndefVarError with
# no indication that the working definition was one `using` away. Pinned by
# `test_no_stale_exports.jl`.

# `@setup_workload` is PrecompileTools', re-exported. `@compile_workload` is NOT
# here any more: the version-taking one freezes kernels into the on-disk cache,
# which needs a device to compile them for, so it moved to Mantle with
# `runtime/workload.jl`. Exporting a macro this package no longer defines is how
# `CoopMat` and `Scalar` survived as dead names for months.
export @setup_workload

# Cooperative matrices: the TYPE a kernel is compiled against, and the operand
# positions. `MatrixUse`/`MatrixScope`/`MatrixShape`/`DeviceCaps` come from
# `KernelInterface` and are re-exported so a kernel library needs one import.
export AcceleratedMatrix, WorkgroupMatrix, MatrixA, MatrixB, Accumulator

# Ray-tracing device intrinsics. The pipeline that dispatches them is Mantle's;
# these are what a shader body calls.
export lava_rt_ignore_intersection, lava_rt_terminate_ray
# SER (SPV_NV_shader_invocation_reorder)
export lava_rt_hit_object_trace_ray, lava_rt_reorder_thread, lava_rt_hit_object_execute_shader

# Re-export Raycore.Ray so `using Lava` users get Ray without ambiguity
export Ray

# Graphics: the configuration types a pipeline is described with, and the
# shader-stage intrinsics. Pure Julia — `graphics/types.jl` has no Vulkan in it,
# which is why it stayed when the pipeline that consumes it left.
export ShaderStage, VertexStage, FragmentStage, GeometryStage, TessControlStage, TessEvalStage
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied
export CullFace, NoCull, CullBack, CullFront
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList, PatchList
export LineListAdjacency, LineStripAdjacency
export DepthMode, DepthLess, DepthLessEq, DepthGreater, DepthAlways, DepthOff
export GeometryConfig, TessConfig
export TessSpacing, EqualSpacing, FractionalEvenSpacing, FractionalOddSpacing
export TessWinding, WindingCW, WindingCCW
export TessDomain, TessTriangles, TessQuads, TessIsolines
export vertex_index, instance_index
export frag_coord, frag_coord_x, frag_coord_y, frag_coord_xy
export dFdx, dFdy
export front_facing
export set_position!, set_point_size!
export gfx_output, gfx_input, gfx_output_flat, gfx_input_flat
export emit_vertex!, end_primitive!, invocation_id, primitive_id_in
export geom_input, geom_input_position
export tess_coord, tess_coord_uvw, set_tess_level_outer!, set_tess_level_inner!
export sample_texture_2d, GfxTexture2D

# Device capability vocabulary, not exported — a kernel library reaches these as
# `Lava.caps(backend)`. `caps` is the one to reach for: a kernel deciding a
# tiling wants several of these and they must all describe the same device.
# `DeviceCaps` and `caps` are `KernelInterface`'s; the METHOD that fills one in
# from a `VkContext` is Mantle's.
public caps, DeviceCaps

import Serialization
import PrecompileTools
using PrecompileTools: @setup_workload
# NOT `using Vulkan`, and that is the headline of the 2026-08-27 split: a
# compiler does not need a driver. GLFW, GPUArraysCore and LinearAlgebra left
# with it — windows, host array machinery and `mul!` are the runtime's, and the
# runtime is `Mantle/src/vulkan/` now.
using GPUCompiler
using Raycore: Ray
using LLVM
using LLVM: API
using GPUArrays
using KernelAbstractions
using Adapt
using Atomix
using SPIRV_Tools_jll
using StaticArrays
using GeometryBasics
# The device vocabulary, shared with Mantle. It lives in `KernelInterface`
# alongside the rest of the backend contract; names are listed one by one because
# that module exports nothing on purpose.
#
# `DeviceCaps` and the matrix types were each written twice, here and in Mantle,
# and bridged by a POSITIONAL copy — both definitions carried a comment warning
# that a field inserted anywhere but the end would misalign it silently. Both
# copies are deleted: there is one type, and Mantle's `caps(ctx::VkContext)` is
# the method that fills it in.
using KernelInterface: MatrixUse, MatrixA, MatrixB, Accumulator,
                       MatrixScope, SubgroupScope, WorkgroupScope, MatrixShape,
                       DeviceCaps
# `import`, not `using ... :` — `device/acceleratedmatrix.jl` and the emitter
# read these. The METHODS that fill a `DeviceCaps` in from a device are Mantle's;
# what is here is the vocabulary a kernel is compiled against.
import KernelInterface: supports, bestshape, caps

# ---- Graphics types (pure Julia, no Vulkan dependency) ----
include("graphics/types.jl")

# ---- Error infrastructure ----
include("runtime/errors.jl")

# ---- SPIR-V module builder ----
include("compiler/spirv/module.jl")
include("compiler/spirv/content_hash.jl")   # a module's identity: every byte
# What the bound device lets a module declare. The runtime pushes it; the
# emitter reads it, and so names no Vulkan type.
include("compiler/target_features.jl")

# ---- SPIR-V emitter stages ----
include("compiler/spirv/types.jl")
include("compiler/spirv/emit.jl")

# ---- Compiler phase timing ----
include("compiler/phase_timer.jl")

# ---- LLVM passes ----
include("compiler/passes/lift_geps.jl")
include("compiler/passes/retype_allocas.jl")
include("compiler/passes/cfg_utils.jl")
include("compiler/passes/structurize_cfg.jl")
include("compiler/passes/prepare_vulkan.jl")
include("compiler/passes/lower_intrinsics.jl")
# shared_memory.jl — consolidated into prepare_vulkan.jl

# ---- Compiler (depends on emitter + passes) ----
include("compiler/target.jl")
include("compiler/entry_wrapper.jl")
include("compiler/compilation.jl")
# The frozen cache's COMPILER half: keys, paths, eligibility and the
# ray-tracing entries, which `compilation.jl` consults before it compiles. AFTER
# it, and that ordering is real rather than tidiness: `FROZEN_LAYOUT` is a
# `const` that hashes `fieldnames(LavaGPUKernel)` at LOAD time, so the struct has
# to exist. The device half is `runtime/frozen_pipeline.jl`.
include("compiler/frozen_spirv.jl")
# Planned files consolidated into emit.jl (4431 lines) and compilation.jl:
#   sourcemap.jl, intrinsics.jl, control_flow.jl, decorations.jl, compute.jl
include("compiler/spirv/raytracing.jl")    # RT stages, OpTraceRayKHR, payload handling
include("compiler/spirv/rayquery.jl")     # VK_KHR_ray_query capabilities + emission
include("compiler/spirv/coopmat.jl")      # SPV_KHR_cooperative_matrix emission
include("compiler/spirv/graphics.jl")     # Graphics stages, I/O variables, emit_vertex

# ---- Device functions ----
# device/array.jl — LavaDeviceArray defined in array/lavaarray.jl
include("device/math.jl")
include("device/quirks.jl")
include("device/rt_intrinsics.jl")  # RT builtins + trace_ray intrinsic
include("device/ray_query_intrinsics.jl")  # lava_ray_query_init
include("device/gfx_intrinsics.jl")  # Graphics builtins + shader I/O intrinsics
include("device/printf.jl")          # @lava_printf → NonSemantic.DebugPrintf
# atomics.jl is included after lavaarray.jl (needs LavaDeviceArray)

# ---- Vulkan runtime ----
# Types before the context that names them. `VkContext` holds its per-device
# state in concrete fields, so every struct in those fields has to exist first —
# and only the `struct` blocks moved, because include order constrains types and
# not methods.
# The driver's cooperative-matrix table, read by `caps(ctx)` to fill
# `DeviceCaps.shapes`. AFTER device.jl, which is where `VkContext` is defined —
# these are type ANNOTATIONS, evaluated at load, while `caps` calling
# `matrixshapes` resolves at call time and so does not constrain the order.
include("runtime/intrinsics.jl")

# ---- Array interface (before launch.jl because it references LavaArray) ----
include("device/devicearray.jl")  # LavaDeviceArray: what a kernel receives
include("device/atomics.jl")  # needs LavaDeviceArray from devicearray.jl
include("device/subgroup.jl") # subgroup / group-non-uniform intrinsics
# KI's DEVICE half: index queries, barriers, shuffles, printing. It needs the
# intrinsics above and nothing else — no queue, no pool, no context. The HOST
# half is `array/kernelinterface_host.jl`, further down with the runtime it
# depends on, and the two are split along exactly that line.
include("device/kernelinterface.jl")
# KA's device half, the counterpart of the above: how `@index` recovers a global
# index from the dispatch builtins, plus `@synchronize` and `@private`. After
# `device/subgroup.jl` for the builtins and after `devicearray.jl` because
# `__index_Global_Linear` indexes a `LinearIndices` over the ndrange.
include("device/ndrange.jl")
include("device/acceleratedmatrix.jl")     # AcceleratedMatrix: the user-facing type
include("device/coopmat_intrinsics.jl")    # its llvmcall stubs
# `@localmem`: addrspace(3) globals and `LavaSharedArray`. AFTER the two above,
# because it also carries the `CoopMatrix` constructors that load a matrix
# straight out of a shared tile — the type has to exist first.
include("device/sharedmemory.jl")
include("device/tensor_intrinsics.jl")     # SPV_NV_tensor_addressing layouts

# LAST. The frozen WORLD (not the frozen cache): `invoke_frozen`, which runs the
# compilation pipeline in the world captured at `__init__` so a later-loaded
# package cannot invalidate its precompiled native code — and the workload that
# puts that code in the package image.
#
# Last because the workload COMPILES A KERNEL at precompile time, so it needs
# `LavaDeviceArray` and the intrinsics that kernel calls to already exist. That
# is a load-order constraint, not tidiness: it fails with an `UndefVarError`
# anywhere earlier.
include("compiler/frozen_world.jl")

# ---- Launch API (depends on LavaArray / LavaDeviceArray) ----
# Frozen kernel cache — `launch.jl` calls into it, and it calls `link_kernel`
# back; both are resolved at call time, so the include order is free.
# The workload macros sit on top of the frozen cache and PrecompileTools.
# Pipeline cache persistence — depends on lava_disk_cache_dir from launch.jl;
# referenced by VkContext constructor (forward at include time, resolved at call time)
# runtime/sync.jl — sync handled via vk_flush!() in launch.jl

# ---- KernelAbstractions backend ----
# KI's HOST half: `synchronize`, `allocate`, the device queries and the launch.
# After ka_backend.jl because it needs `LavaBackend`, `caps`, `ka_launch!` and
# `vk_context(::LavaBackend)`.

# ---- Debug / validation API (needs LavaArray + KA backend) ----

# ---- Hardware video decode ----
#
# `decode_h264_gpu`, `decode_h264_nv12` and `decode_h264_luma` moved to
# `Mantle/src/vulkan/runtime/videoapi.jl` on 2026-08-27. Each one supplies
# `vk_context()` so its caller does not have to hold one, which is precisely the
# thing a compiler cannot do.
# ---- Runtime diagnostics ----
#
# `gpu_memory_usage`, `dump_state`, `verify_gpu_av` and the rest moved to
# `Mantle/src/vulkan/runtime/diagnostics.jl` on 2026-08-27, with
# `GPUArraysCore.allowscalar(false)`. Every one read a `VkContext`.


function __init__()
    # Nothing to reset here any more. The counters and logs this used to zero
    # were module-level `Ref`s and `Vector`s, which meant a device crash during
    # precompilation serialised its wreckage into the pkgimage and poisoned every
    # later session. They are `ctx.diag` fields now, built fresh with the context,
    # so there is nothing that can survive into the image to clear.
    # Frozen kernel cache ON by default. The key already mixes in
    # `Base.module_build_id` of both the kernel's defining module and Lava, so a
    # changed kernel body produces a different key; `frozen_eligible` restricts
    # it to package modules, whose build ids actually move (Main's does not --
    # measured: an edited Main kernel was served stale SPIR-V).
    #
    # This is what makes a package's SECOND session cheap: Hikari's 45-kernel
    # scene goes 31.5 s -> 20.1 s, and per frozen_cache.jl crown's hw_accel
    # startup ~1063 s -> ~123 s. Recording costs nothing measurable (31.9 s vs
    # 31.5 s), so both halves are on.
    FROZEN_VERSION[] = "1"
    FROZEN_RECORDING[] = true

    # Capture BEFORE any other package loads. The precompile workload above put
    # the pipeline's native code in THIS package image; a later-loaded package
    # defining methods can invalidate it, and then the first compile re-JITs the
    # compiler. Freezing here keeps that precompiled code live.
    _initialization_world[] = Base.get_world_counter()

    # No `init_pipeline_thread!` and no `atexit` hook here any more. Both were
    # about a live device — a background thread that builds pipelines, and
    # marking the device lost before Julia's finalizer sweep so a finalizer
    # cannot call into a torn-down driver. A compiler has no device to lose, so
    # they moved with `runtime/device.jl` and run from Mantle's `__init__`.
end

end # module Lava
