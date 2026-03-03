module Lava

export LavaArray, LavaBackend, LavaBuffer, LavaDeviceArray

# Graphics exports
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export RenderWindow, LavaFramebuffer, WindowTarget, OffscreenTarget
export CompiledGraphicsPipeline, LavaGfxShader
export draw!, blit!, present_frame!, acquire_next_image!, readback_framebuffer
export vk_begin_pass!, vk_draw_in_pass!, vk_end_pass!
export LavaTexture2D, LavaTexture1D, LavaSampler, SampledTexture, LavaTexture
export TextureBindings, bind_textures
# Graphics types
export ShaderStage, VertexStage, FragmentStage, GeometryStage, TessControlStage, TessEvalStage
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied
export CullFace, NoCull, CullBack, CullFront
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList, PatchList
export DepthMode, DepthLess, DepthLessEq, DepthGreater, DepthAlways, DepthOff
export GeometryConfig, TessConfig
export TessSpacing, EqualSpacing, FractionalEvenSpacing, FractionalOddSpacing
export TessWinding, WindingCW, WindingCCW
export TessDomain, TessTriangles, TessQuads, TessIsolines
# Graphics device intrinsics
export _lava_gfx_vertex_index, _lava_gfx_instance_index
export _lava_gfx_frag_coord, _lava_gfx_frag_coord_x, _lava_gfx_frag_coord_y
export _lava_gfx_front_facing
export _lava_gfx_set_position, _lava_gfx_set_point_size
export _lava_gfx_output_f32, _lava_gfx_output_vec2, _lava_gfx_output_vec3, _lava_gfx_output_vec4
export _lava_gfx_input_f32, _lava_gfx_input_vec2, _lava_gfx_input_vec3, _lava_gfx_input_vec4
export _lava_gfx_emit_vertex, _lava_gfx_end_primitive
export _lava_gfx_tess_coord, _lava_gfx_set_tess_level_outer, _lava_gfx_set_tess_level_inner

# Ray tracing exports
export HardwareAccel, RTRay, RTHitResult
export trace_closest_hits!, trace_closest_hits_indirect!, RayTracingPipeline, trace_rays!, trace_rays_indirect!
export set_anyhit_pipeline!, trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export _lava_rt_ignore_intersection, _lava_rt_terminate_ray

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
import GLFW

# ---- Graphics types (pure Julia, no Vulkan dependency) ----
include("graphics/types.jl")

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
# shared_memory.jl — consolidated into prepare_vulkan.jl

# ---- Compiler (depends on emitter + passes) ----
include("compiler/target.jl")
include("compiler/entry_wrapper.jl")
include("compiler/compilation.jl")
# Planned files consolidated into emit.jl (4431 lines) and compilation.jl:
#   sourcemap.jl, intrinsics.jl, control_flow.jl, decorations.jl, compute.jl
include("compiler/spirv/raytracing.jl")    # RT stages, OpTraceRayKHR, payload handling
include("compiler/spirv/graphics.jl")     # Graphics stages, I/O variables, emit_vertex

# ---- Device functions ----
# device/array.jl — LavaDeviceArray defined in array/lavaarray.jl
include("device/math.jl")
include("device/quirks.jl")
include("device/rt_intrinsics.jl")  # RT builtins + trace_ray intrinsic
include("device/gfx_intrinsics.jl")  # Graphics builtins + shader I/O intrinsics
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
# runtime/sync.jl — sync handled via vk_flush!() in launch.jl

# ---- KernelAbstractions backend ----
include("array/ka_backend.jl")
include("array/gpuarrays.jl")
include("array/mapreduce.jl")

# ---- Phase 2: Graphics ----
include("graphics/window.jl")         # RenderWindow, swapchain, present
include("graphics/framebuffer.jl")   # LavaFramebuffer, RenderTarget, image memory helpers
include("graphics/pipeline.jl")      # CompiledGraphicsPipeline, draw recording
include("graphics/textures.jl")      # LavaTexture2D, LavaSampler, descriptor sets
include("graphics/api.jl")           # GraphicsPipeline, draw!, blit!, present_frame!

# ---- Phase 2: Ray Tracing ----
include("raytracing/acceleration.jl")  # BLAS/TLAS build
include("raytracing/pipeline.jl")      # RT pipeline + SBT + dispatch
include("raytracing/shaders.jl")       # High-level RayTracingPipeline API
include("raytracing/raycore_compat.jl") # HardwareAccel + trace_closest_hits!

# ---- Default settings ----
# Disable scalar indexing by default (GPU arrays should not be accessed element-by-element)
GPUArraysCore.allowscalar(false)

end # module Lava
