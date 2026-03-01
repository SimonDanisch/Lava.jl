module Lava

export LavaArray, LavaBackend, LavaBuffer, LavaDeviceArray

using Vulkan
using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix
using SPIRV_Tools_jll
using LinearAlgebra

# ---- Error infrastructure ----
include("runtime/errors.jl")

# ---- SPIR-V module builder ----
include("compiler/spirv/module.jl")

# ---- SPIR-V emitter stages ----
include("compiler/spirv/types.jl")
include("compiler/spirv/emit.jl")

# ---- LLVM passes ----
include("compiler/passes/lift_geps.jl")
include("compiler/passes/structurize_cfg.jl")
include("compiler/passes/prepare_vulkan.jl")
include("compiler/passes/lower_intrinsics.jl")
# include("compiler/passes/shared_memory.jl")

# ---- Compiler (depends on emitter + passes) ----
include("compiler/target.jl")
include("compiler/entry_wrapper.jl")
include("compiler/compilation.jl")
# include("compiler/sourcemap.jl")
# include("compiler/intrinsics.jl")
# include("compiler/spirv/control_flow.jl")
# include("compiler/spirv/decorations.jl")
# include("compiler/spirv/compute.jl")
# include("compiler/spirv/graphics.jl")
# include("compiler/spirv/raytracing.jl")

# ---- Device functions ----
# include("device/array.jl")
include("device/math.jl")
include("device/quirks.jl")
# atomics.jl is included after lavaarray.jl (needs LavaDeviceArray)

# ---- Vulkan runtime ----
include("runtime/device.jl")
include("runtime/memory.jl")
include("runtime/pipeline.jl")
include("runtime/command.jl")
include("runtime/intrinsics.jl")

# ---- Array interface (before launch.jl because it references LavaArray) ----
include("array/lavaarray.jl")
include("device/atomics.jl")  # needs LavaDeviceArray from lavaarray.jl

# ---- Launch API (depends on LavaArray and LavaBuffer) ----
include("runtime/launch.jl")
# include("runtime/sync.jl")

# ---- KernelAbstractions backend ----
include("array/ka_backend.jl")
include("array/gpuarrays.jl")
include("array/mapreduce.jl")

# ---- Graphics ----
# include("graphics/shaders.jl")
# include("graphics/pipeline.jl")
# include("graphics/textures.jl")

# ---- Ray Tracing ----
# include("raytracing/shaders.jl")
# include("raytracing/acceleration.jl")
# include("raytracing/sbt.jl")
# include("raytracing/pipeline.jl")

# ---- Default settings ----
# Disable scalar indexing by default (GPU arrays should not be accessed element-by-element)
GPUArraysCore.allowscalar(false)

end # module Lava
