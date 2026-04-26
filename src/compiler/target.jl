# GPUCompiler target definition for Lava.jl
#
# Uses GPUCompiler.SPIRVCompilerTarget with backend=:llvm to get the
# spirv64-unknown-unknown-unknown triple. We only use GPUCompiler for
# LLVM IR generation (compile(:llvm, ...)), never for SPIR-V codegen —
# that's handled by our custom emitter.

# ── Compiler parameters ──

struct LavaCompilerParams <: GPUCompiler.AbstractCompilerParams
    workgroup_size::NTuple{3,Int}
    enable_ray_query::Bool
end
LavaCompilerParams() = LavaCompilerParams((64, 1, 1), false)
LavaCompilerParams(workgroup_size::NTuple{3,Int}) = LavaCompilerParams(workgroup_size, false)

const LavaCompilerConfig = GPUCompiler.CompilerConfig{GPUCompiler.SPIRVCompilerTarget, LavaCompilerParams}
const LavaCompilerJob = GPUCompiler.CompilerJob{GPUCompiler.SPIRVCompilerTarget, LavaCompilerParams}

# ── Device runtime stubs ──
#
# GPUCompiler's lower_throw! pass converts `throw(...)` into calls to these
# runtime functions. On GPU we can't throw, so they're no-ops. The pass then
# inserts `llvm.trap` + `unreachable` after these calls, which our LLVM passes
# (rm_trap!, replace_unreachable!) clean up before SPIR-V emission.

module LavaRuntime
    signal_exception() = return
    report_exception(ex) = return
    report_oom(sz) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return

    # GPUCompiler's runtime expects malloc/free for dynamic allocation.
    # On GPU these are no-ops that return null — dynamic allocation is not
    # supported in Vulkan compute shaders. The compiler should eliminate
    # these via dead code elimination in practice.
    malloc(sz) = reinterpret(Ptr{Nothing}, UInt(0))
    free(ptr) = return
end

# ── Method table for device overrides ──
#
# @lava_device_override puts methods on this table, which GPUCompiler uses
# instead of the regular method table during compilation. This lets us
# replace CPU functions with GPU-safe versions (e.g., intrinsics, math).

Base.Experimental.@MethodTable lava_method_table

macro lava_device_override(ex)
    esc(quote
        Base.Experimental.@overlay($lava_method_table, $ex)
    end)
end

# ── GPUCompiler hooks ──

GPUCompiler.runtime_module(::LavaCompilerJob) = LavaRuntime

GPUCompiler.method_table(::LavaCompilerJob) = lava_method_table

# Vulkan doesn't use GPUCompiler's kernel state mechanism
GPUCompiler.kernel_state_type(::LavaCompilerJob) = Nothing

# Allow SPIR-V intrinsic function calls to pass IR validation.
# llvm.spv.* intrinsics are used by the LLVM SPIR-V backend.
# OpenCL-mangled names (_Z*) are registered dynamically by our intrinsics
# module when @builtin_ccall is used.
const KNOWN_INTRINSICS = String[]

function GPUCompiler.isintrinsic(::LavaCompilerJob, fn::String)
    # LLVM SPIR-V intrinsics (used by the SPIR-V backend for workgroup/subgroup ops)
    startswith(fn, "llvm.spv.") && return true
    # Custom _lava_glsl_* externals → SPIR-V emitter maps to GLSL.std.450
    startswith(fn, "_lava_glsl_") && return true
    # RT intrinsics → SPIR-V emitter maps to OpTraceRayKHR, payload load/store
    startswith(fn, "_lava_rt_") && return true
    # OpenCL C++ mangled builtins (thread indices, barriers, math)
    fn in KNOWN_INTRINSICS && return true
    return false
end

# Override check_invocation to allow types with zero-sized non-isbits fields.
# Julia's broadcast wraps functions in RefValue{typeof(f)} which is not isbits
# but has sizeof == 0 and gets elided by LLVM. GPUCompiler's default check
# rejects these, but they're safe to pass to GPU since they carry no data.
function GPUCompiler.check_invocation(job::LavaCompilerJob)
    sig = job.source.specTypes
    for (arg_i, dt) in enumerate(sig.parameters)
        GPUCompiler.isghosttype(dt) && continue
        Core.Compiler.isconstType(dt) && continue
        fieldcount(dt) == 0 && continue
        if !isbitstype(dt) && !is_gpu_compatible(dt)
            throw(GPUCompiler.KernelError(job, "passing non-bitstype argument",
                """Argument $arg_i to your kernel function is of type $dt, which is not a bitstype:
                   $(GPUCompiler.explain_nonisbits(dt))

                   Only bitstypes can be used in GPU kernels."""))
        end
    end
    return
end

# Check if a non-isbits type is still GPU-compatible by verifying all
# non-isbits fields are zero-sized (will be elided by LLVM).
function is_gpu_compatible(@nospecialize(dt::Type))
    isbitstype(dt) && return true
    isabstracttype(dt) && return false
    try
        sizeof(dt) == 0 && return true
    catch
        return false
    end
    # Check recursively: all fields must be either isbits or zero-sized
    for i in 1:fieldcount(dt)
        ft = fieldtype(dt, i)
        isbitstype(ft) && continue
        try
            sizeof(ft) == 0 && continue
        catch
            return false
        end
        # Recurse into non-isbits, non-zero-sized fields
        is_gpu_compatible(ft) || return false
    end
    return true
end

# ── Compiler configuration cache ──

const COMPILER_CONFIGS = Dict{UInt, LavaCompilerConfig}()

"""
    lava_compiler_config(; workgroup_size=(64,1,1)) -> LavaCompilerConfig

Get or create a cached compiler configuration. The configuration sets up
GPUCompiler's SPIRVCompilerTarget with:
- backend=:llvm for the spirv64 triple (required for LLVM IR generation)
- validate=false (we validate SPIR-V ourselves after our custom emitter)
- supports_fp64=true (AMD RX 7900 XTX supports Float64)
- always_inline=true (SPIR-V requires all functions inlined for compute)
- kernel=true (entry point is a kernel, not a regular function)
"""
function lava_compiler_config(; workgroup_size::NTuple{3,Int} = (64, 1, 1),
                               enable_ray_query::Bool = false,
                               kwargs...)
    h = hash((workgroup_size, enable_ray_query, kwargs))
    config = get(COMPILER_CONFIGS, h, nothing)
    if config !== nothing
        return config
    end
    config = lava_full_compiler_config(; workgroup_size, enable_ray_query, kwargs...)
    COMPILER_CONFIGS[h] = config
    return config
end

@noinline function lava_full_compiler_config(;
        kernel = true,
        name = nothing,
        always_inline = true,
        workgroup_size::NTuple{3,Int} = (64, 1, 1),
        enable_ray_query::Bool = false,
        kwargs...)
    target = GPUCompiler.SPIRVCompilerTarget(;
        backend = :llvm,       # gives spirv64-unknown-unknown-unknown triple
        validate = false,      # we validate after our custom SPIR-V emitter
        supports_fp64 = true,  # AMD RX 7900 XTX supports Float64
    )
    params = LavaCompilerParams(workgroup_size, enable_ray_query)
    GPUCompiler.CompilerConfig(target, params; kernel, name, always_inline)
end
