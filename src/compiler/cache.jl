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
# Kernels compiled at run time outlive the session too, kept by what they were
# compiled from (`persisted`, below): Mantle guarantees of every backend that a
# kernel compiled once is not compiled again in a later session.
#
# Only SPIR-V lives here. What a device builds from it, a `VkPipeline`, is the
# runtime's, keyed on these objects by identity.

using GPUCompiler.CompilerCaching: ObjCache

"""
    CompileStats

Kernel and shader lookups since the last `reset_compile_stats!`: `hits` found
SPIR-V compiled earlier, in this session, in an earlier one or while a package
precompiled; `misses` compiled it. The names and shape are Metal.jl's
`compile_stats`.
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
portable(s::LavaGfxShader) = LavaGfxShader(s.spirv_bytes, s.stage, s.push_info, "")
portable(k::LavaGPUKernel) = LavaGPUKernel(k.spirv_bytes, k.entry_name, k.workgroup_size, k.push_info, "",
                                           k.enable_ray_query, k.source_name)

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
    # reflection (`@device_code_*`) sees cached kernels too, without recompiling
    GPUCompiler.run_compile_hook(job)
    res = GPUCompiler.cached_results(LavaKernelResults, job)
    if res !== nothing && res.kernel !== nothing
        COMPILE_STATS.hits += 1
        return res.kernel
    end
    enroll!(job)
    kernel = invoke_frozen(persisted_kernel, job)::LavaGPUKernel
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
    enroll!(job)
    shader = persisted_gfx_shader(job; stage, config)
    shaders = @something shaders GPUCompiler.cached_results(GfxShaders, job)
    shaders.stages[key] = generating_output() ? portable(shader) : shader
    return shader
end

# ── Across sessions ──────────────────────────────────────────────────────────
#
# A kernel compiled at run time has a `CodeInstance` of this session only, and no
# other process can check that it is still the code the kernel was compiled from.
# What another process can compare is the LLVM module GPUCompiler makes of it
# (`lava_llvm`): everything Lava does from there, its passes, SPIR-V emission,
# spirv-opt and spirv-val, four fifths of a compile, depends on that module, the
# stage's settings and the compiler, and on nothing else. So a stage is kept in
# CompilerCaching's object cache under those three, and a later session that makes
# the same module finds it there.

"""Bump to drop every stage kept under an earlier layout of what is stored."""
const PERSISTED_SCHEMA = 1

"""
    canonical_ir(mod) -> Vector{UInt8}

The text of `mod` without what differs between two sessions compiling the same code. Julia
numbers the functions and constants it names in the order it compiles them (`julia_f_1234`), and
the numbers reach the symbols, the names of inlined values, the debug info and the names of the
alias scopes inlining makes (`!"julia_f_1234: %sret_return"`). A copy of `mod` without debug info
and value names, its own symbols numbered in module order and the numbers dropped from metadata
strings, prints the same in every session. A scope is `distinct`, so its name tells it from no
other. No debug info reaches the SPIR-V Lava emits; the names that do, the entry's, are those of
the session that compiled the stage, which keeps them with it.
"""
function canonical_ir(mod::LLVM.Module)
    c = copy(mod)
    LLVM.strip_debuginfo!(c)
    i = 0
    for f in LLVM.functions(c)
        LLVM.isdeclaration(f) && continue
        LLVM.name!(f, "f$(i += 1)")
        foreach(a -> LLVM.name!(a, ""), LLVM.parameters(f))
        for b in LLVM.blocks(f)
            LLVM.name!(b, "")
            foreach(ins -> LLVM.name!(ins, ""), LLVM.instructions(b))
        end
    end
    for g in LLVM.globals(c)
        LLVM.isdeclaration(g) || LLVM.name!(g, "g$(i += 1)")
    end
    text = replace(string(c), r"!\"[^\"]*\"" => name -> replace(name, r"_\d+" => ""))
    # Bytes, not the `String`: SHA reads a `String` a thousand times slower.
    text = Vector{UInt8}(text)
    LLVM.dispose(c)
    return text
end

"""
What a stage depends on besides its LLVM module and its settings: Lava's build, Julia's, LLVM's
and SPIRV-Tools' versions, and every switch that changes what the pipeline emits.
"""
compiler_identity() = (string(Base.module_build_id(@__MODULE__)), string(VERSION), string(LLVM.version()),
                       string(pkgversion(SPIRV_Tools_jll)), string(SPIRV_OPT_ENABLED[]), string(LOOP_UNROLL[]),
                       get(ENV, "LAVA_OUTLINE_ENABLED", "0"), get(ENV, "LAVA_OUTLINE_BB_THRESHOLD", "50"),
                       get(ENV, "LAVA_SKIP_PASSES", ""))

function serialized(x)
    io = IOBuffer()
    Serialization.serialize(io, x)
    return take!(io)
end

"""
    persisted(compile, ns, mod, settings...) -> stage

The stage `compile()` makes of the module `mod` with `settings`: kept in namespace `ns` of
CompilerCaching's object cache, under `mod`'s canonical text (`canonical_ir`), the settings and
the compiler (`compiler_identity`), and found there when a session compiled it before. Counted in
`compile_stats` either way. Nothing is kept while a package precompiles, whose image keeps it
instead, nor while kernels are dumped for debugging, which only compiling does.
"""
function persisted(compile, ns::String, mod::LLVM.Module, settings...)
    keep = !generating_output() && !kernel_dump_wanted()
    key = keep ? ObjCache.keyhash(PERSISTED_SCHEMA, canonical_ir(mod),
                                  Vector{UInt8}(join((compiler_identity()..., map(repr, settings)...), '\0'))) :
                 UInt8[]
    found = keep ? ObjCache.get(ns, key) : nothing
    if found !== nothing
        COMPILE_STATS.hits += 1
        return Serialization.deserialize(IOBuffer(found))
    end
    COMPILE_STATS.misses += 1
    stage = compile()
    keep && ObjCache.put!(ns, key, serialized(portable(stage)))
    return stage
end

"""The compute kernel of `job`, kept from an earlier session or compiled (`persisted`)."""
function persisted_kernel(job::LavaCompilerJob)
    GPUCompiler.JuliaContext() do ctx
        mod, meta = lava_llvm(job)
        p = job.config.params
        return persisted("Lava/kernel", mod, p.workgroup_size, p.enable_ray_query, p.features) do
            lava_gpu_kernel(mod, meta, job)
        end
    end
end

"""Graphics stage `stage` of `job`, kept from an earlier session or compiled (`persisted`)."""
function persisted_gfx_shader(job::GPUCompiler.CompilerJob; stage::Symbol, config)
    GPUCompiler.JuliaContext() do ctx
        mod, meta = lava_llvm(job)
        return persisted("Lava/gfx", mod, stage, config) do
            gfx_shader(mod, meta; stage, config, validate = true)
        end
    end
end
