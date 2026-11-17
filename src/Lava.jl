module Lava

# A Julia to SPIR-V compiler for Vulkan. Devices, memory, launches and pipelines
# are Mantle's; its Vulkan backend calls into this module to compile.

# The device-side array: a `(pointer, dims)` pair, which is what a kernel receives.
export LavaDeviceArray

# Compilation.
export CompilationResult, lava_compile, LavaGfxShader

# PrecompileTools' macro, for packages that precompile kernels through Lava.
export @setup_workload

# Operand positions of a cooperative matrix, written directly when a tile is
# instantiated. The matrix types themselves are re-exported by Mantle.
export MatrixA, MatrixB, Accumulator

# Stage configuration: topologies, geometry and tessellation settings.
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList, PatchList
export LineListAdjacency, LineStripAdjacency
export GeometryConfig, TessConfig
export TessSpacing, EqualSpacing, FractionalEvenSpacing, FractionalOddSpacing
export TessWinding, WindingCW, WindingCCW
export TessDomain, TessTriangles, TessQuads, TessIsolines

# The stage lowering: interface variables by location, geometry-stage inputs,
# tessellation levels and the stage wrappers. The builtins a shader author writes
# (`vertex_index`, `frag_coord`, `discard`, ...) are the vocabulary's and are
# exported by Mantle, so `using Mantle, Lava` binds each name once.
export front_facing
export set_position!
export gfx_output, gfx_input, gfx_output_flat, gfx_input_flat
export invocation_id
export geom_input, geom_input_position
export tess_coord, tess_coord_uvw, set_tess_level_outer!, set_tess_level_inner!
export GfxTexture2D
export VertexWrapper, FragmentWrapper, GeometryWrapper

import Serialization
using Base.ScopedValues: ScopedValue, with
import PrecompileTools
using PrecompileTools: @setup_workload

# The device vocabulary: functions and types every backend implements by overlay.
# Outside the gate below, so it loads on a machine without a Vulkan loader.
import KernelInterface as KI
include("vocabulary/vocabulary.jl")

import Vulkan

# The compiler, built only where a Vulkan loader exists.
@static if Vulkan.HAS_LOADER

using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
using KernelAbstractions
using Adapt
using Atomix
using SPIRV_Tools_jll
using StaticArrays
using GeometryBasics
# The module binding itself: the device files add methods to `KernelInterface.<name>`.
import KernelInterface

include("graphics/types.jl")                # stage configuration types
include("runtime/errors.jl")                # LavaCompilationError and its report

# SPIR-V module builder and emitter.
include("compiler/spirv/module.jl")
include("compiler/spirv/content_hash.jl")   # a module's identity: every byte
include("compiler/target_features.jl")      # what the bound device lets a module declare
include("compiler/spirv/types.jl")
include("compiler/spirv/emit.jl")
include("compiler/spirv/foldedoffset.jl")   # keeps negative constant offsets off loads

include("compiler/phase_timer.jl")

# LLVM passes run ahead of the emitter.
include("compiler/passes/lift_geps.jl")
include("compiler/passes/retype_allocas.jl")
include("compiler/passes/cfg_utils.jl")
include("compiler/passes/structurize_cfg.jl")
include("compiler/passes/prepare_vulkan.jl")
include("compiler/passes/private_arguments.jl")
include("compiler/passes/lower_intrinsics.jl")
include("compiler/passes/exact_narrowing.jl")
include("compiler/passes/wide_division.jl")
include("compiler/passes/odd_integers.jl")

# The GPUCompiler target, the entry wrapper and the compilation pipeline.
include("compiler/target.jl")
include("compiler/entry_wrapper.jl")
include("compiler/passes/readonly_loads.jl")  # loads of read-only arguments, by provenance
include("compiler/compilation.jl")
# Compiled kernels and stages, kept with their `CodeInstance`s.
include("compiler/cache.jl")
include("compiler/optimization_cache.jl")
include("compiler/inference_cache.jl")
include("compiler/function_cache.jl")
include("compiler/spirv/rayquery.jl")       # VK_KHR_ray_query
include("compiler/spirv/coopmat.jl")        # cooperative matrices
include("compiler/spirv/graphics.jl")       # graphics stages and their interface variables

# Device functions.
include("device/math.jl")
include("device/math_f64.jl")
include("device/quirks.jl")
include("device/ray_query_intrinsics.jl")
include("device/gfx_intrinsics.jl")         # graphics builtins and stage I/O
include("device/printf.jl")                 # @lava_printf through NonSemantic.DebugPrintf
include("runtime/intrinsics.jl")
include("device/devicearray.jl")            # LavaDeviceArray
include("device/constarray.jl")             # Const: a LavaDeviceArray the kernel only reads
include("device/atomics.jl")
include("device/subgroup.jl")
include("device/kernelinterface.jl")        # KernelInterface's device half
include("device/ndrange.jl")                # KernelAbstractions' index and barrier lowering
include("device/coopmat_intrinsics.jl")     # the vocabulary's matrices on OpCooperativeMatrix*
include("device/sharedmemory.jl")           # workgroup memory
include("device/tensor_intrinsics.jl")      # coopmat2 tensor layouts

# Last: the precompile workload in it compiles a kernel, which needs everything above.
include("compiler/frozen_world.jl")

function __init__()
    reset_compile_stats!()

    # Compile in the world of this point, before later-loaded packages can
    # invalidate the compiler's precompiled native code.
    _initialization_world[] = Base.get_world_counter()

    # Register every LLVM target on one thread before the first compile: two
    # threads registering concurrently corrupt LLVM's target registry list.
    LLVM.InitializeAllTargets()
    LLVM.InitializeAllTargetInfos()
    LLVM.InitializeAllAsmPrinters()
    LLVM.InitializeAllAsmParsers()
    LLVM.InitializeAllTargetMCs()
end

end # @static if Vulkan.HAS_LOADER

end # module Lava
