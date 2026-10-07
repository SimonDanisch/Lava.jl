# Compiled kernels and shader stages, kept with the code they were compiled from.
#
# GPUCompiler attaches a results struct to each kernel's `CodeInstance`
# (`GPUCompiler.cached_results`), which is how CUDA.jl caches its kernels. An
# edit to a kernel or to anything inlined into it (Revise) invalidates the
# `CodeInstance`, and the next lookup compiles again; an unrelated method
# definition costs nothing. Code compiled while a package precompiles goes into
# its image along with the `CodeInstance`, because Lava's code is relocatable
# (`relocation_lowering`, target.jl): a package's `@compile_workload` leaves its
# kernels compiled for every session after.
#
# Only SPIR-V lives here. What a device builds from it, a `VkPipeline`, is the
# runtime's, keyed on these objects by identity.

"""
    CompileStats

Kernel and shader lookups since the last `reset_compile_stats!`: `hits` found
SPIR-V compiled earlier, in this session or while a package precompiled;
`misses` compiled it. The names and shape are Metal.jl's `compile_stats`.
"""
mutable struct CompileStats
    hits::Int
    misses::Int
end

const COMPILE_STATS = CompileStats(0, 0)

"""
    compile_stats() -> (; hits, misses)

How many kernel and shader lookups found compiled SPIR-V, and how many compiled
it, since the last [`reset_compile_stats!`](@ref). A workload that leaves
nothing to compile shows `misses == 0` when run again in a new session.
"""
compile_stats() = (; hits = COMPILE_STATS.hits, misses = COMPILE_STATS.misses)

"""Reset what [`compile_stats`](@ref) counts."""
reset_compile_stats!() = (COMPILE_STATS.hits = 0; COMPILE_STATS.misses = 0; nothing)

"""Whether this process is writing a package image."""
generating_output() = ccall(:jl_generating_output, Cint, ()) == 1

"""
What goes into a package image of a compiled stage: its SPIR-V without the LLVM
IR, which only a debugging session reads and which is the largest field by far.
"""
portable(s::LavaRTShader) = LavaRTShader(s.spirv_bytes, s.stage, s.push_info, "")
portable(s::LavaGfxShader) = LavaGfxShader(s.spirv_bytes, s.stage, s.push_info, "")

"""
Before compiling anything in a package's precompilation: infer the job with the
GPU interpreter, which is what enrolls its `CodeInstance`, and so the results
kept with it, in the package image.
"""
enroll!(job) = (generating_output() && precompile(job); nothing)

# ── Compute kernels ──────────────────────────────────────────────────────────

"""
    LavaKernelResults

What GPUCompiler keeps with a compute kernel's `CodeInstance`: its compiled
SPIR-V, or `nothing` before the first compile.
"""
mutable struct LavaKernelResults
    kernel::Union{Nothing,LavaGPUKernel}
    LavaKernelResults() = new(nothing)
end

"""
    lava_kernel_job(f, tt; workgroup_size, enable_ray_query=false, features=TargetFeatures()) -> CompilerJob

The compile job of compute kernel `f` taking arguments of types `tt`, in the
current world.
"""
lava_kernel_job(@nospecialize(f), @nospecialize(tt); workgroup_size::NTuple{3,Int},
                enable_ray_query::Bool = false, features::TargetFeatures = TargetFeatures()) =
    GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(f), tt),
                            lava_compiler_config(; workgroup_size, enable_ray_query, features))

"""
    compile_or_lookup(job) -> LavaGPUKernel

The compute kernel `job` describes, compiled for the code it has now: kept with
its `CodeInstance` when it was compiled before, compiled otherwise. The compile
runs in the world Lava was loaded in (`invoke_frozen`), so methods defined since
cannot invalidate the compiler's own code.
"""
@noinline function compile_or_lookup(job::LavaCompilerJob)::LavaGPUKernel
    res = GPUCompiler.cached_results(LavaKernelResults, job)
    if res !== nothing && res.kernel !== nothing && GPUCompiler.compile_hook[] === nothing
        COMPILE_STATS.hits += 1
        return res.kernel
    end
    COMPILE_STATS.misses += 1
    enroll!(job)
    kernel = invoke_frozen(lava_compile_gpu_from_job, job)::LavaGPUKernel
    # Compiling the job made its `CodeInstance`, so this finds one.
    res = @something res GPUCompiler.cached_results(LavaKernelResults, job)
    res.kernel = kernel
    return kernel
end

# ── Graphics stages ──────────────────────────────────────────────────────────

"""
    GfxShaders

The graphics stages compiled from one function at one point in its history,
by `(stage, config)`, kept with its `CodeInstance`.
"""
mutable struct GfxShaders
    stages::Dict{Any,LavaGfxShader}
    GfxShaders() = new(Dict{Any,LavaGfxShader}())
end

"""
    cached_gfx_shader(f, tt, stage; config=nothing) -> LavaGfxShader

Stage `stage` of `f` with arguments of types `tt`, compiled for the code `f` has
now. Keyed on the function and its argument types for the life of a device, as
it was before this, the first SPIR-V of a session was drawn for the rest of it,
whatever the source said by then.
"""
function cached_gfx_shader(@nospecialize(f), @nospecialize(tt), stage::Symbol; config = nothing)
    job = lava_gfx_job(f, tt)
    key = (stage, config)
    shaders = GPUCompiler.cached_results(GfxShaders, job)
    if shaders !== nothing
        shader = get(shaders.stages, key, nothing)
        if shader !== nothing
            COMPILE_STATS.hits += 1
            return shader
        end
    end
    COMPILE_STATS.misses += 1
    enroll!(job)
    shader = lava_compile_gfx_shader(f, tt; stage, config, job)
    shaders = @something shaders GPUCompiler.cached_results(GfxShaders, job)
    shaders.stages[key] = generating_output() ? portable(shader) : shader
    return shader
end

# ── Ray-tracing stages ───────────────────────────────────────────────────────

"""
    RTShaders

The ray-tracing stages compiled from one function at one point in its history,
by `(stage, payload_type, push_constant_size)`, kept with its `CodeInstance`.
"""
mutable struct RTShaders
    stages::Dict{Any,LavaRTShader}
    RTShaders() = new(Dict{Any,LavaRTShader}())
end

"""
    cached_rt_shader(f, tt; stage, push_constant_size, payload_type, features) -> LavaRTShader

Ray-tracing stage `stage` of `f`, compiled for the code `f` has now. With
`LAVA_CAPTURE_RT_JOBS=1` every lookup is recorded (`captured_rt_jobs`), found or
compiled: the capture is about which stages a scene uses, not about timing them.
"""
function cached_rt_shader(@nospecialize(f), @nospecialize(tt); stage::Symbol,
                          push_constant_size::Integer, payload_type::Symbol,
                          features::TargetFeatures)
    if get(ENV, "LAVA_CAPTURE_RT_JOBS", "") == "1"
        push!(RT_JOB_CAPTURE, RTShaderJob(f, tt, stage, payload_type, Int(push_constant_size)))
    end
    job = lava_rt_job(f, tt, features)
    key = (stage, payload_type, Int(push_constant_size))
    shaders = GPUCompiler.cached_results(RTShaders, job)
    if shaders !== nothing
        shader = get(shaders.stages, key, nothing)
        if shader !== nothing
            COMPILE_STATS.hits += 1
            return shader
        end
    end
    COMPILE_STATS.misses += 1
    enroll!(job)
    shader = lava_compile_rt_shader(f, tt; stage, push_constant_size, payload_type,
                                    validate = true, features, job)
    shaders = @something shaders GPUCompiler.cached_results(RTShaders, job)
    shaders.stages[key] = generating_output() ? portable(shader) : shader
    return shader
end
