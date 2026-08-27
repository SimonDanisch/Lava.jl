# Keeping the compiler's own precompiled code alive, and putting it there.
#
# Both halves are about the COMPILATION PIPELINE, not about a device, which is
# why they stayed in Lava when the runtime moved out on 2026-08-27. They were the
# tail of `Lava.jl` and briefly travelled with the diagnostics next to them,
# which was adjacency rather than a reason.
#
# `invoke_frozen` is called from the launch path, which is Mantle's now — one of
# the names it imports from here.

# ── Frozen world ────────────────────────────────────────────────────────────
#
# Same construction as CUDACore (`CUDACore/src/initialization.jl`) and cuTile
# (`cuTile.jl:invoke_frozen`). The compilation pipeline — GPUCompiler's
# typeinf/codegen AND Lava's own SPIR-V emitter — is precompiled into package
# images, but a method defined by any later-loaded package can invalidate that
# native code, so the first kernel compile in a session pays to re-JIT the
# COMPILER before it even starts compiling the kernel. Running the pipeline in
# the world captured at `__init__` keeps the precompiled code live.
#
# Defaults to `typemax(UInt)` so that during precompilation, before `__init__`
# has run, `invoke_in_world` clamps to the current world and behaves normally.
const _initialization_world = Ref{UInt}(typemax(UInt))

"""
    invoke_frozen(f, args...; kwargs...)

Invoke `f(args...; kwargs...)` in the world captured at `__init__` time, so
precompiled native code for the compilation pipeline stays usable across method
insertions in later-loaded packages.

`invoke_in_world` is not inferable, so callers should annotate the result with a
concrete return type where it matters.
"""
function invoke_frozen(f, args...; kwargs...)
    @inline
    kwargs = merge(NamedTuple(), kwargs)
    if isempty(kwargs)
        return Base.invoke_in_world(_initialization_world[], f, args...)
    end
    return Base.invoke_in_world(_initialization_world[], Core.kwcall, kwargs, f, args...)
end

# ── Precompile workload ─────────────────────────────────────────────────────
#
# Compile one representative kernel at PRECOMPILE time so the pipeline's native
# code — GPUCompiler's typeinf/codegen and Lava's own SPIR-V emitter — lands in
# Lava's package image. Without it, `using Lava` leaves the pipeline cold and the
# first kernel compile in a session spends ~24 s JITting the COMPILER before it
# starts on the kernel.
#
# DEVICE-FREE BY CONSTRUCTION: `lava_compile_gpu` only reaches `vk_context()`
# when `enable_ray_query=true`. Precompilation must never touch the driver.
function _precompile_warmup_kernel!(out::LavaDeviceArray{Float32, 1},
                                    a::LavaDeviceArray{Float32, 1})
    i = Int(lava_global_invocation_id_x()) + 1
    if i <= length(out)
        acc = 0f0
        for k in 1:4
            @inbounds acc += a[i] * Float32(k)
        end
        @inbounds out[i] = acc
    end
    return nothing
end

# A second shape: struct-valued args, mixed int/float, and an atomic — the
# emitter specialises differently on each, and real kernels (wavefront queues,
# reductions) hit these paths far more than the scalar loop above.
struct _PrecompileWarmupParams
    scale::Float32
    count::Int32
end

function _precompile_warmup_kernel2!(out::LavaDeviceArray{Float32, 1},
                                     idx::LavaDeviceArray{Int32, 1},
                                     p::_PrecompileWarmupParams)
    i = Int(lava_global_invocation_id_x()) + 1
    if i <= length(out)
        @inbounds j = idx[i]
        if j > Int32(0) && j <= Int32(length(out))
            @inbounds out[j] = out[j] * p.scale + Float32(p.count)
        end
    end
    return nothing
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        lava_compile_gpu(_precompile_warmup_kernel!,
                         Tuple{LavaDeviceArray{Float32, 1}, LavaDeviceArray{Float32, 1}};
                         validate = false)
        lava_compile_gpu(_precompile_warmup_kernel2!,
                         Tuple{LavaDeviceArray{Float32, 1}, LavaDeviceArray{Int32, 1},
                               _PrecompileWarmupParams};
                         validate = false)
    end
end