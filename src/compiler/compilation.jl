# Main compilation pipeline for Lava.jl
#
# Pipeline: Julia function → GPUCompiler → LLVM IR → LLVM passes → custom SPIR-V emitter
#
# Two entry points:
#   lava_compile_to_llvm()  — returns LLVM IR string (for debugging)
#   lava_compile_to_spirv() — returns validated SPIR-V binary

# Debug counter for unique kernel file naming
const _KERNEL_DEBUG_COUNTER = Ref(0)

"""
Wrap GPUCompiler.InvalidIRError with Lava-specific context and actionable suggestions.
Called from compilation entry points to provide better user-facing errors.
"""
function _wrap_gpu_compiler_error(@nospecialize(e), @nospecialize(f), @nospecialize(tt))
    e isa GPUCompiler.InvalidIRError || rethrow(e)

    fname = try string(nameof(typeof(f))) catch; string(f) end
    err_str = sprint(showerror, e)
    suggestions = String[]

    if occursin("dynamic function invocation", err_str) || occursin("runtime call", err_str)
        push!(suggestions, "Type instability: the function contains dynamic dispatch. Ensure all types are inferrable — use @code_warntype to check.")
    end
    if occursin("jl_f_throw_methoderror", err_str) || occursin("jl_f__apply_iterate", err_str)
        push!(suggestions, "Method lookup at runtime: an operation doesn't have a concrete method for the given types. Check that all operations are GPU-compatible.")
    end
    if occursin("gc_pool_alloc", err_str) || occursin("jl_gc", err_str) || occursin("jl_alloc", err_str) || occursin("get_pgcstack", err_str)
        push!(suggestions, "Heap allocation: GPU kernels cannot allocate memory. Avoid Arrays, Strings, or mutable containers. Use NTuple or StaticArrays instead.")
    end
    if occursin("undefined name", err_str) || occursin("jl_f_getglobal", err_str)
        push!(suggestions, "Global variable access: GPU kernels cannot access non-const globals. Pass data as arguments or use `const`.")
    end
    if occursin("jl_f_tuple", err_str)
        push!(suggestions, "Dynamic tuple construction from type instability. Check for type instabilities in the call chain.")
    end

    suggestion = isempty(suggestions) ?
        "Check the stacktrace in the raw error below. Use @code_typed to inspect type inference." :
        join(suggestions, "\n  ")

    # Extract deduplicated call chains showing which user functions are problematic
    call_chain_summary = _extract_call_chains(err_str)

    throw(LavaCompilationError(
        "kernel compilation",
        "Cannot compile $fname($(join(tt.parameters, ", "))) to GPU code",
        suggestion;
        raw_error=err_str,
        call_chains=call_chain_summary
    ))
end

"""
    _extract_call_chains(err_str) -> String

Parse GPUCompiler's InvalidIRError output to extract unique call chains through user code.
Deduplicates the repeated stacktraces (GPUCompiler emits one per illegal call, but many
share the same user-code path). Returns a compact summary showing:
- Each unique user-code call chain (deepest user function → kernel entry)
- The reason the deepest function is GPU-incompatible
"""
function _extract_call_chains(err_str::String)
    # Parse each "Reason: ...\nStacktrace:\n [1] ...\n [2] ..." block
    blocks = _parse_error_blocks(err_str)
    isempty(blocks) && return ""

    # For each block, extract the chain of user functions (skip Base/stdlib internals)
    seen_chains = Set{String}()
    unique_chains = Vector{Tuple{String, Vector{String}}}()  # (reason, [user_funcs...])

    for (reason, stack_entries) in blocks
        user_funcs = String[]
        for entry in stack_entries
            # Skip Base/stdlib internals — keep user code and Lava code
            is_internal = _is_base_internal(entry)
            if !is_internal
                push!(user_funcs, entry)
            end
        end
        isempty(user_funcs) && continue

        # Deduplicate by the user function chain
        chain_key = join(user_funcs, " → ")
        if chain_key ∉ seen_chains
            push!(seen_chains, chain_key)
            push!(unique_chains, (reason, user_funcs))
        end
    end

    isempty(unique_chains) && return ""

    # Format: show each unique chain compactly
    lines = String[]
    push!(lines, "Call chain(s) with GPU-incompatible code:")
    for (i, (reason, funcs)) in enumerate(unique_chains)
        # Show as: kernel! → helper1 → helper2 → [problem] (reason)
        short_reason = _shorten_reason(reason)
        chain = join(reverse(funcs), " → ")
        push!(lines, "  $i. $chain")
        push!(lines, "     Problem: $short_reason")
    end

    return join(lines, "\n")
end

function _parse_error_blocks(err_str::String)
    blocks = Vector{Tuple{String, Vector{String}}}()
    # Split by "Reason:" — each is one error
    parts = split(err_str, "Reason: ")
    for part in parts[2:end]  # skip the header before first Reason
        lines = split(part, '\n')
        reason = strip(String(lines[1]))

        # Parse stacktrace entries: " [N] func_name\n   @ file:line"
        stack_entries = String[]
        i = 1
        while i <= length(lines)
            m = match(r"^\s*\[(\d+)\]\s+(.+)$", lines[i])
            if m !== nothing
                func_name = strip(String(m.captures[2]))
                # Next line has location
                loc = ""
                if i + 1 <= length(lines)
                    lm = match(r"^\s+@ (.+)$", lines[i+1])
                    if lm !== nothing
                        loc = strip(String(lm.captures[1]))
                        i += 1
                    end
                end
                entry = isempty(loc) ? func_name : "$func_name @ $loc"
                push!(stack_entries, entry)
            end
            i += 1
        end
        push!(blocks, (reason, stack_entries))
    end
    return blocks
end

# Base/stdlib paths to filter out of call chains
const _BASE_INTERNAL_PATTERNS = [
    "/share/julia/", "boot.jl", "Base.jl", "array.jl", "strings/",
    "ryu/", "iobuffer.jl", "pointer.jl", "float.jl", "int.jl",
    "abstractarray.jl", "essentials.jl", "promotion.jl", "math.jl",
    "number.jl", "operators.jl", "reduce.jl", "dict.jl", "set.jl",
    "range.jl", "simdloop.jl", "refvalue.jl", "iterators.jl",
]

function _is_base_internal(entry::String)
    for pat in _BASE_INTERNAL_PATTERNS
        occursin(pat, entry) && return true
    end
    return false
end

function _shorten_reason(reason::String)
    if occursin("gc_pool_alloc", reason) || occursin("get_pgcstack", reason) ||
       occursin("new_gc_frame", reason) || occursin("push_gc_frame", reason) ||
       occursin("pop_gc_frame", reason) || occursin("gc_frame_slot", reason)
        return "heap allocation (GC)"
    elseif occursin("jl_alloc", reason) || occursin("ijl_alloc", reason)
        return "heap allocation"
    elseif occursin("ijl_pchar_to_string", reason) || occursin("jl_string", reason) ||
           occursin("genericmemory_to_string", reason) || occursin("string_to_generic", reason)
        return "String allocation"
    elseif occursin("dynamic function invocation", reason)
        return "dynamic dispatch (type instability)"
    elseif occursin("runtime call", reason)
        return "runtime function call"
    elseif occursin("jl_f_getglobal", reason) || occursin("undefined name", reason)
        return "global variable access"
    elseif occursin("jl_argument_error", reason)
        return "argument validation (allocates)"
    elseif occursin("jl_f_throw_methoderror", reason)
        return "method lookup failure"
    else
        # Return first ~80 chars
        return length(reason) > 80 ? reason[1:80] * "..." : reason
    end
end

# SPIR-V optimization: enabled automatically on NVIDIA to work around
# Always run spirv-opt: produces cleaner SPIR-V, helps with driver bugs
# (NVIDIA Xid 31 MMU faults, AMD Windows shader compiler hangs).
const _spirv_opt_enabled = Ref(true)

"""
    _run_spirv_opt(spirv_bytes::Vector{UInt8}) -> Vector{UInt8}

Run spirv-opt on the SPIR-V binary to optimize it. Uses SPIRV_Tools_jll.
This can help NVIDIA's shader compiler handle complex shaders that would
otherwise cause miscompilation (Xid 31 MMU faults with large kernels).
"""
function _run_spirv_opt(spirv_bytes::Vector{UInt8})
    spirv_opt = SPIRV_Tools_jll.spirv_opt()
    in_path = tempname() * ".spv"
    out_path = tempname() * ".spv"
    try
        write(in_path, spirv_bytes)
        p = run(pipeline(`$spirv_opt --target-env=vulkan1.3 -O $in_path -o $out_path`;
                          stderr=devnull, stdout=devnull); wait=true)
        if p.exitcode == 0 && isfile(out_path)
            return read(out_path)
        end
    catch
        # Fall through to return original bytes
    finally
        rm(in_path; force=true)
        rm(out_path; force=true)
    end
    return spirv_bytes  # fallback: return unoptimized
end

# ── Compilation result types ──

struct LavaLLVMResult
    ir::String
    entry_name::String
    workgroup_size::NTuple{3,Int}
end

struct LavaSPIRVResult
    spirv_bytes::Vector{UInt8}
    entry_name::String
    workgroup_size::NTuple{3,Int}
    ir::String  # LLVM IR for debugging
end

"""
GPU-ready compilation result with BDA argument buffer layout.
"""
struct LavaGPUKernel
    spirv_bytes::Vector{UInt8}
    entry_name::String
    workgroup_size::NTuple{3,Int}
    push_info::PushConstantInfo
    ir::String
end

# ── Unified introspection result ──

"""
    CompilationResult

Unified compilation result capturing every pipeline stage for introspection
and cross-platform validation. Returned by `lava_compile()`.
"""
struct CompilationResult
    pre_pass_ir::String           # Raw GPUCompiler LLVM IR (before our passes)
    post_pass_ir::String          # LLVM IR after all passes (what the emitter sees)
    spirv_bytes::Vector{UInt8}    # Raw SPIR-V binary
    spirv_disasm::String          # spirv-dis output
    entry_name::String
    stage::Symbol                 # :compute, :vertex, :fragment, :raygen, etc.
    workgroup_size::NTuple{3,Int}
    push_info::Union{Nothing, PushConstantInfo}
    source_map::Dict{UInt32, Tuple{String, Int}}  # SPIR-V ID → (julia_file, julia_line)
end

"""
    lava_compile(f, tt; stage=:compute, workgroup_size=(64,1,1),
                 config=nothing, payload_type=:f32, validate=true) -> CompilationResult

Compile a Julia function through the full Lava pipeline, returning a `CompilationResult`
with IR at every stage for introspection and validation.

Supports all shader stages: `:compute`, `:vertex`, `:fragment`, `:geometry`,
`:tess_control`, `:tess_eval`, `:raygen`, `:closesthit`, `:miss`, `:anyhit`,
`:intersection`, `:callable`.
"""
function lava_compile(@nospecialize(f), @nospecialize(tt);
                      stage::Symbol=:compute,
                      workgroup_size::NTuple{3,Int}=(64, 1, 1),
                      config=nothing,
                      payload_type::Symbol=:f32,
                      validate::Bool=true)
    if stage == :compute
        result = _lava_compile_full(f, tt; workgroup_size, validate)
        return result
    elseif stage in (:vertex, :fragment, :geometry, :tess_control, :tess_eval)
        return _lava_compile_gfx_full(f, tt; stage, config, validate)
    elseif stage in (:raygen, :closesthit, :miss, :anyhit, :intersection, :callable)
        return _lava_compile_rt_full(f, tt; stage, payload_type, validate)
    else
        error("Unknown stage: $stage")
    end
end

"""Compile compute kernel, capturing pre/post IR."""
function _lava_compile_full(@nospecialize(f), @nospecialize(tt);
                            workgroup_size::NTuple{3,Int}=(64, 1, 1),
                            validate::Bool=true)
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # Capture pre-pass IR
        pre_pass_ir = string(mod)

        # BDA entry wrapper
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size)
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # LLVM passes
        _run_llvm_passes!(mod, wrapper_fn)
        post_pass_ir = string(mod)

        # SPIR-V emission
        spirv_bytes, source_map = _emit_spirv_from_llvm(mod, wrapper_name, workgroup_size)

        # Validation
        write("/tmp/lava_last.spv", spirv_bytes)
        write("/tmp/lava_last.ll", post_pass_ir)
        if validate
            _validate_spirv(spirv_bytes, post_pass_ir, source_map)
        end

        spirv_disasm = disassemble_spirv(spirv_bytes)

        return CompilationResult(pre_pass_ir, post_pass_ir, spirv_bytes, spirv_disasm,
                                 wrapper_name, :compute, workgroup_size, push_info, source_map)
    end
end

"""Compile graphics shader, capturing pre/post IR."""
function _lava_compile_gfx_full(@nospecialize(f), @nospecialize(tt);
                                 stage::Symbol=:vertex,
                                 config=nothing,
                                 validate::Bool=true)
    config_wg = lava_compiler_config(; workgroup_size=(1, 1, 1))
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config_wg)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        pre_pass_ir = string(mod)

        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        _run_llvm_passes!(mod, wrapper_fn)
        post_pass_ir = string(mod)

        spirv_bytes, source_map = _emit_spirv_from_llvm_gfx(mod, wrapper_name, stage; config=config)

        write("/tmp/lava_last.spv", spirv_bytes)
        write("/tmp/lava_last.ll", post_pass_ir)
        if validate
            _validate_spirv(spirv_bytes, post_pass_ir, source_map)
        end

        spirv_disasm = disassemble_spirv(spirv_bytes)

        return CompilationResult(pre_pass_ir, post_pass_ir, spirv_bytes, spirv_disasm,
                                 wrapper_name, stage, (1, 1, 1), push_info, source_map)
    end
end

"""Compile RT shader, capturing pre/post IR."""
function _lava_compile_rt_full(@nospecialize(f), @nospecialize(tt);
                                stage::Symbol=:raygen,
                                payload_type::Symbol=:f32,
                                validate::Bool=true)
    config = lava_compiler_config(; workgroup_size=(1, 1, 1))
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        pre_pass_ir = string(mod)

        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        _run_llvm_passes!(mod, wrapper_fn)
        post_pass_ir = string(mod)

        spirv_bytes, source_map = _emit_spirv_from_llvm_rt(mod, wrapper_name, stage;
                                                payload_type=payload_type)

        write("/tmp/lava_last.spv", spirv_bytes)
        write("/tmp/lava_last.ll", post_pass_ir)
        if validate
            _validate_spirv(spirv_bytes, post_pass_ir, source_map)
        end

        spirv_disasm = disassemble_spirv(spirv_bytes)

        return CompilationResult(pre_pass_ir, post_pass_ir, spirv_bytes, spirv_disasm,
                                 wrapper_name, stage, (1, 1, 1), push_info, source_map)
    end
end

"""    disassemble(r::CompilationResult) -> String

Return the SPIR-V disassembly text."""
disassemble(r::CompilationResult) = r.spirv_disasm

"""
    optimize_spirv(spirv_bytes::Vector{UInt8}; passes="-O") -> Vector{UInt8}

Run spirv-opt on SPIR-V binary. Returns optimized bytes.
Validates output with spirv-val.
"""
function optimize_spirv(spirv_bytes::Vector{UInt8}; passes::String="-O")
    spirv_opt = SPIRV_Tools_jll.spirv_opt()
    spirv_val_cmd = SPIRV_Tools_jll.spirv_val()

    in_path = tempname() * ".spv"
    out_path = tempname() * ".spv"
    try
        write(in_path, spirv_bytes)
        run(`$spirv_opt --target-env=vulkan1.3 --scalar-block-layout $passes $in_path -o $out_path`)
        optimized = read(out_path)
        # Validate optimized output
        run(`$spirv_val_cmd --target-env vulkan1.3 --scalar-block-layout $out_path`)
        return optimized
    finally
        rm(in_path; force=true)
        rm(out_path; force=true)
    end
end

# ── LLVM IR compilation (GPUCompiler → LLVM Module) ──

"""
    lava_compile_to_llvm(f, tt; workgroup_size=(64,1,1)) -> LavaLLVMResult

Compile a Julia function to LLVM IR via GPUCompiler. Returns IR string.
"""
function lava_compile_to_llvm(@nospecialize(f), @nospecialize(tt);
                              workgroup_size::NTuple{3,Int} = (64, 1, 1))
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_name = LLVM.name(meta.entry)
        ir = string(mod)
        return LavaLLVMResult(ir, entry_name, workgroup_size)
    end
end

# ── Full pipeline: Julia → SPIR-V ──

"""
    lava_compile_to_spirv(f, tt; workgroup_size=(64,1,1), validate=true) -> LavaSPIRVResult

Full compilation pipeline: Julia function → LLVM IR → LLVM passes → SPIR-V emission → validation.

Returns `LavaSPIRVResult` with validated SPIR-V binary, entry name, and LLVM IR for debugging.
"""
function lava_compile_to_spirv(@nospecialize(f), @nospecialize(tt);
                               workgroup_size::NTuple{3,Int} = (64, 1, 1),
                               validate::Bool = true)
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # ── Stage 1: LLVM passes ──
        _run_llvm_passes!(mod, entry_fn)

        # Save IR for debugging (after passes, before emission)
        ir = string(mod)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes, source_map = _emit_spirv_from_llvm(mod, entry_name, workgroup_size)

        # ── Stage 3: Validation ──
        if validate
            _validate_spirv(spirv_bytes, "", source_map)
        end

        return LavaSPIRVResult(spirv_bytes, entry_name, workgroup_size, ir)
    end
end

"""
    lava_compile_gpu(f, tt; workgroup_size=(64,1,1), validate=true) -> LavaGPUKernel

Full GPU-ready compilation pipeline with BDA entry wrapper.
The resulting SPIR-V has a void() entry point that loads arguments from a BDA buffer.

The `push_info` field describes the argument buffer layout for `pack_kernel_args`.
"""
function lava_compile_gpu(@nospecialize(f), @nospecialize(tt);
                           workgroup_size::NTuple{3,Int} = (64, 1, 1),
                           validate::Bool = true)
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # ── Stage 0: BDA entry wrapper ──
        # Must happen BEFORE passes — the wrapper becomes the new entry point
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size)
        wrapper_name = push_info.wrapper_name

        # The wrapper function is now the entry point
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # ── Stage 1: LLVM passes ──
        _run_llvm_passes!(mod, wrapper_fn)

        # Save IR for debugging
        ir = string(mod)
        _KERNEL_DEBUG_COUNTER[] += 1
        _kidx = _KERNEL_DEBUG_COUNTER[]
        _dbg_dir = joinpath(@__DIR__, "..", "..", "..", "tmp_kernels")
        mkpath(_dbg_dir)
        write(joinpath(_dbg_dir, "kernel_$(_kidx)_$(replace(wrapper_name, r"[^a-zA-Z0-9_]" => "_")).ll"), ir)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes, source_map = _emit_spirv_from_llvm(mod, wrapper_name, workgroup_size)

        # ── Stage 2.5: SPIR-V optimization (optional, helps NVIDIA) ──
        if _spirv_opt_enabled[]
            spirv_bytes = _run_spirv_opt(spirv_bytes)
        end

        # Save SPIR-V for debugging
        write(joinpath(_dbg_dir, "kernel_$(_kidx)_$(replace(wrapper_name, r"[^a-zA-Z0-9_]" => "_")).spv"), spirv_bytes)

        # ── Stage 3: Validation ──
        if validate
            _validate_spirv(spirv_bytes, ir, source_map)
        end

        return LavaGPUKernel(spirv_bytes, wrapper_name, workgroup_size, push_info, ir)
    end
end

# ── RT Shader Compilation ──

"""
    LavaRTShader

Compilation result for a ray tracing shader stage.
"""
struct LavaRTShader
    spirv_bytes::Vector{UInt8}
    stage::Symbol
    push_info::PushConstantInfo
    ir::String
end

"""
    lava_compile_rt_shader(f, tt; stage=:raygen, push_constant_size=8,
                            payload_type=:f32, validate=true) -> LavaRTShader

Compile a Julia function to a ray tracing shader stage (raygen, closesthit, miss).

The function receives its arguments via a BDA push constant buffer (same as compute),
except that RT-specific data (TLAS, payload, builtins) is handled automatically by
the emitter.

# Stages
- `:raygen` — Ray generation shader. Can call `lava_rt_trace_ray!()`.
- `:closesthit` — Closest hit shader. Can read hit builtins, write payload.
- `:miss` — Miss shader. Can write payload.

# Payload
Currently only Float32 payload supported (`payload_type=:f32`).
"""
function lava_compile_rt_shader(@nospecialize(f), @nospecialize(tt);
                                 stage::Symbol=:raygen,
                                 push_constant_size::Integer=8,
                                 payload_type::Symbol=:f32,
                                 validate::Bool=true)
    config = lava_compiler_config(; workgroup_size=(1, 1, 1))
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # BDA entry wrapper (same as compute — args via push constant buffer)
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # LLVM passes (same as compute)
        _run_llvm_passes!(mod, wrapper_fn)

        ir = string(mod)
        write("/tmp/lava_last_rt.ll", ir)

        # RT-specific SPIR-V emission
        spirv_bytes, source_map = _emit_spirv_from_llvm_rt(mod, wrapper_name, stage;
                                                payload_type=payload_type)

        write("/tmp/lava_last.spv", spirv_bytes)

        if validate
            _validate_spirv(spirv_bytes, ir, source_map)
        end

        return LavaRTShader(spirv_bytes, stage, push_info, ir)
    end
end

# ── Graphics Shader Compilation ──

"""
    LavaGfxShader

Compilation result for a graphics shader stage.
"""
struct LavaGfxShader
    spirv_bytes::Vector{UInt8}
    stage::Symbol
    push_info::PushConstantInfo
    ir::String
end

"""
    lava_compile_gfx_shader(f, tt; stage=:vertex, config=nothing, validate=true) -> LavaGfxShader

Compile a Julia function to a graphics shader stage (vertex, fragment, geometry,
tess_control, tess_eval).

The function receives its arguments via a BDA push constant buffer (same as compute/RT).
Graphics-specific builtins and I/O variables are handled automatically by the emitter.

# Stages
- `:vertex` — Vertex shader. Can use `_lava_gfx_vertex_index()`, `_lava_gfx_set_position()`.
- `:fragment` — Fragment shader. Can use `_lava_gfx_frag_coord()`, `_lava_gfx_output_vec4()`.
- `:geometry` — Geometry shader. Requires `config::GeometryConfig`.
- `:tess_control` — Tessellation control shader. Requires `config::TessConfig`.
- `:tess_eval` — Tessellation evaluation shader. Requires `config::TessConfig`.
"""
function lava_compile_gfx_shader(@nospecialize(f), @nospecialize(tt);
                                   stage::Symbol=:vertex,
                                   config=nothing,
                                   validate::Bool=true)
    config_wg = lava_compiler_config(; workgroup_size=(1, 1, 1))
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config_wg)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            _wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # BDA entry wrapper (same as compute — args via push constant buffer)
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # LLVM passes (same as compute)
        _run_llvm_passes!(mod, wrapper_fn)

        ir = string(mod)
        write("/tmp/lava_last_gfx_$(stage).ll", ir)

        # Graphics-specific SPIR-V emission
        spirv_bytes, source_map = _emit_spirv_from_llvm_gfx(mod, wrapper_name, stage; config=config)

        write("/tmp/lava_last_gfx_$(stage).spv", spirv_bytes)

        if validate
            _validate_spirv(spirv_bytes, ir, source_map)
        end

        return LavaGfxShader(spirv_bytes, stage, push_info, ir)
    end
end

# ── Stage 1: LLVM Pass Pipeline ──

function _run_llvm_passes!(mod::LLVM.Module, entry_fn::LLVM.Function)
    # ── CFG cleanup ──
    # Remove constructs that SPIR-V can't handle
    # Replace freeze before optimization: GPU kernel arguments are never undef,
    # so freeze is unnecessary. Removing it early lets LLVM produce simpler IR.
    _replace_freeze!(mod)
    _strip_assume!(mod)

    # Remove trap/unreachable from error paths (GPUCompiler's lower_throw!)
    GPUCompiler.rm_trap!(mod)
    _replace_unreachable!(mod)
    _strip_noreturn!(mod)

    # ── Force-inline all internal functions ──
    # GPU shaders are single-function programs. GPUCompiler generates helper
    # functions (error throwing, boxing, etc.) that must be inlined into the
    # entry function. After inlining, the error paths become dead code.
    _force_inline_all!(mod, entry_fn)

    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write("/tmp/lava_ir_1_postinline.ll", string(mod))
    end

    # ── Fix barrier-skipping error paths ──
    # _replace_unreachable! (pre-inlining) converts error paths to early returns.
    # After inlining, these returns may skip barriers that other invocations reach,
    # causing undefined behavior (deadlock on CPU/software Vulkan implementations).
    # Redirect barrier-skipping paths to the barrier-containing continuation.
    _fix_barrier_skipping_paths!(entry_fn)

    # ── Post-inlining optimization ──
    LLVM.run!(LLVM.InstCombinePass(), mod)
    LLVM.run!(LLVM.SROAPass(), mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)

    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write("/tmp/lava_ir_2_postsroa.ll", string(mod))
    end

    # ── Fix inttoptr address spaces after SROA ──
    # SROA eliminates allocas and creates `inttoptr i64 %bda_val to ptr` (addrspace 0)
    # for BDA pointer fields. These should be addrspace 1 (PhysicalStorageBuffer)
    # for correct SPIR-V emission. Convert them and update all downstream uses.
    _fix_inttoptr_addrspace!(mod)

    # ── Remove Julia runtime artifacts from inlined error paths ──
    # After force-inlining, error/boxing helpers may reference Julia runtime:
    # - `load i64, ptr @jl_int64_type` (type tags for boxing)
    # - `store i64, ptr inttoptr(1)` (GC tag slot writes)
    # These are dead error paths that will never execute on GPU. Remove them
    # so the SPIR-V emitter doesn't need to handle runtime declarations.
    _remove_julia_runtime_artifacts!(mod)

    # ── Lower LLVM intrinsics unsupported by SPIR-V ──
    # memcpy → typed loads/stores, lifetime markers → removed
    _lower_unsupported_intrinsics!(mod)

    # ── Fix GEPs with mismatched source types on allocas ──
    # After SROA + inlining, some GEPs reference the original full tuple type
    # through a smaller alloca pointer. Convert these to byte-offset GEPs so the
    # lift_byte_geps pass can properly convert them using the alloca's type.
    _fix_gep_alloca_type_mismatches!(mod)

    # ── Flatten chained GEPs on allocas ──
    # Pattern: gep i8 alloca -4 → gep i32 result %idx  →  gep i8 alloca (-4 + idx*4)
    # This handles Julia's 1-based MArray indexing where the base is shifted.
    _flatten_chained_geps_on_allocas!(mod)

    # ── Lift byte-offset GEPs to typed GEPs ──
    # Julia accesses struct fields via `getelementptr i8, ptr %p, i64 <offset>`.
    # SPIR-V needs typed GEPs into the struct. Run 3x as later passes may create more.
    _lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    _lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    _lift_byte_geps_on_allocas!(mod)

    # ── Combine consecutive same-type GEPs ──
    # Patterns like `gep T, (gep T, p, i), j` → `gep T, p, add(i, j)`.
    # This avoids chained OpPtrAccessChain which some drivers handle incorrectly.
    _combine_chained_geps!(mod)

    # ── Structured control flow ──
    # SPIR-V requires structured CF. Run the full structurize pipeline.
    run_structurize_cfg_pipeline!(mod)

    # ── Flatten nested workgroup array globals ──
    # Replace [32 x [2 x float]] → [64 x float] in addrspace(3).
    # Without VK_KHR_workgroup_memory_explicit_layout, SPIR-V Workgroup arrays
    # cannot have ArrayStride decorations, and NVIDIA miscomputes the stride
    # for nested arrays. Flattening to scalar arrays avoids this.
    _flatten_nested_workgroup_arrays!(mod)

    # ── Lift byte-offset GEPs on workgroup globals ──
    # Convert `gep i8, @shared, <offset>` ConstantExpr to typed struct-member GEPs.
    # Must run before decompose passes so the emitter sees proper typed access patterns.
    dl = LLVM.datalayout(mod)
    # ── Decompose workgroup typepun copies ──
    # LLVM may optimize shared memory struct copies (shared[i] = shared[j]) into raw
    # integer block copies. Detect and replace with per-field typed copies.
    _decompose_workgroup_typepun_copies!(mod, dl)

    # ── Decompose composite workgroup accesses ──
    # Struct loads/stores on addrspace(3) must be decomposed into scalar ops
    # because shared memory is flattened to scalar arrays in SPIR-V.
    _decompose_composite_workgroup_accesses!(mod, dl)

    # ── Decompose type-punned alloca loads ──
    # LLVM memcpy lowering may create `load i64, ptr %alloca_of_struct` which
    # reads raw bytes from a padded struct. For workgroup stores, decompose
    # the entire copy into per-field struct stores. For other uses, replace
    # with first-field load + zext.
    _decompose_typepun_alloca_loads!(mod, dl)

    # Re-run composite workgroup decomposition: the typepun pass may have created
    # new struct stores to workgroup memory that need field-by-field decomposition.
    _decompose_composite_workgroup_accesses!(mod, dl)

    # Decompose type-punned loads through GEPs into struct allocas.
    # LLVM memcpy optimization creates `load i64, ptr %gep_to_float_field` —
    # reads crossing struct field boundaries. Decompose into per-field loads + pack.
    _decompose_typepun_gep_loads!(mod, dl)

    # Re-combine chained byte-offset GEPs that were left undecomposed above
    # (because the chain contains dynamic indices). E.g.:
    #   gep i8, (gep i8, %alloca, %dynamic), 224  →  gep i8, %alloca, add(%dynamic, 224)
    # The emitter can then decompose the combined dynamic byte offset into proper
    # OpAccessChain indices via _decompose_flat_index_for_composite!.
    _combine_chained_geps!(mod)

    # LLVM may create loads where the type differs from the alloca type
    # (e.g., `load i16` from `alloca { [2 x i8] }`). SPIR-V requires strict
    # type matching, so rewrite these to byte-by-byte extraction.
    _fix_alloca_type_mismatched_loads!(mod)

    # LLVM SROA/memcpy lowering creates `store i32, ptr %alloca_of_[16 x i64]`.
    # SPIR-V requires the stored value type to match the pointer's pointee type.
    # Rewrite these stores to drill into the alloca type via GEP.
    _fix_alloca_type_mismatched_stores!(mod, dl)

    # Fold type-punned scalar allocas where constant partial stores reconstruct a value.
    # SROA decomposes e.g. `zero(Float64)` into `store float 0.0` at offset 0 +
    # `store i32 0` at offset 4, then `load double`. Fold to a direct constant.
    _fold_typepun_scalar_alloca_constants!(mod, dl)

    # Lower chained mismatched-type GEPs on allocas.
    # Julia's MArray/StaticArray patterns create chains like:
    #   %base = getelementptr i32, ptr %alloca_[16 x i64], i64 -1
    #   %elem = getelementptr i32, ptr %base, i64 %var
    #   store i32 %val, ptr %elem
    # Lower to proper element-level access with runtime index computation.
    _lower_chained_mismatched_geps!(mod)

    # ── Convert typed GEPs on mismatched allocas to byte-offset GEPs ──
    # When InstCombine (inside StructurizeCFG) creates typed GEPs like
    #   gep i32, ptr %alloca_[8xi64], i64 %dynamic
    # the emitter can't handle the type mismatch. Convert to byte GEPs:
    #   gep i8, ptr %alloca, i64 (%dynamic * 4)
    # so _lower_byte_gep_chain_on_allocas! can apply shift/mask extraction.
    _convert_typepunned_geps_to_byte_geps!(mod)

    # ── Lower byte-offset GEP chains on MArray allocas ──
    # After InstCombine splits flattened byte-offset GEPs back into chains:
    #   %gep1 = gep i8, ptr %alloca_[16xi64], %dynamic
    #   %gep2 = gep i8, ptr %gep1, -4
    #   store i32 %val, ptr %gep2
    # Lower to proper element-level access with shift/mask (no integer divide).
    _lower_byte_gep_chain_on_allocas!(mod)

    # ── Lower PHI-chained typepunned loads on array allocas ──
    # When byte-offset GEPs flow through PHI chains (from StructurizeCFG) before being
    # loaded, _lower_byte_gep_chain_on_allocas! can't handle them (it requires direct
    # GEP→load). This pass "lifts" the load to each leaf GEP site with shift/mask
    # extraction and creates parallel value PHI chains.
    # Critical for MVector{32,UInt32} stored as [16 x i64] in BVH stack traversal.
    _lower_phi_typepunned_loads!(mod)

    # ── Lift byte-offset GEPs on workgroup globals ──
    # The decompose passes above may create byte-offset ConstantExpr GEPs like
    # `gep i8, @shared, <offset>` when splitting struct loads from workgroup globals.
    # Convert these to typed struct-member GEPs so the emitter produces proper OpAccessChain.
    _lift_byte_geps_on_workgroup_globals!(mod, dl)

    # ── Final cleanup: fix GEPs with mismatched source types on allocas ──
    # LLVM's SROA/memcpy lowering can create typed GEPs using wrong source types
    # (e.g., `gep [3 x float]` on `alloca [3 x i32]`). Convert these to byte-offset
    # GEPs followed by the lift pass to normalize types.
    _fix_gep_alloca_type_mismatches!(mod)
    _lift_byte_geps_on_allocas!(mod)

    # The final _lift_byte_geps_on_allocas! may convert byte GEPs (gep i8, alloca, <off>)
    # to typed GEPs (gep [2 x double], alloca, 0, 0), but the load users still have the
    # wrong type (e.g., load i32 from a double*). Run the typepun GEP load decomposition
    # one more time to fix these.
    _decompose_typepun_gep_loads!(mod, dl)

    return nothing
end

"""
Force-inline all internal functions into the entry function.

Marks all non-entry, non-declaration functions as `alwaysinline`, runs the
AlwaysInliner pass, then removes dead functions with GlobalDCE.
After this, only the entry function (and LLVM intrinsic declarations) remain.
"""
function _force_inline_all!(mod::LLVM.Module, entry_fn::LLVM.Function)
    entry_name = LLVM.name(entry_fn)

    for fn in LLVM.functions(mod)
        fn_name = LLVM.name(fn)

        # Skip entry function, declarations (no body), and LLVM intrinsics
        fn_name == entry_name && continue
        isempty(LLVM.blocks(fn)) && continue
        startswith(fn_name, "llvm.") && continue

        # Remove noinline, add alwaysinline
        attrs = LLVM.function_attributes(fn)
        delete!(attrs, LLVM.EnumAttribute("noinline"))
        push!(attrs, LLVM.EnumAttribute("alwaysinline"))
    end

    # Run inliner
    LLVM.run!(LLVM.AlwaysInlinerPass(), mod)

    # Remove now-dead internal functions
    LLVM.run!(LLVM.GlobalDCEPass(), mod)

    return nothing
end

# ── Stage 2: SPIR-V Emission ──

"""
Emit SPIR-V binary from an LLVM module. Handles type mapping, instruction emission,
entry point setup, and serialization.
"""
function _emit_spirv_from_llvm(llvm_mod::LLVM.Module, entry_name::String,
                                workgroup_size::NTuple{3,Int})
    # Build pointee type map (opaque pointer → typed pointer recovery)
    ptm = build_pointee_type_map(llvm_mod)

    # Create SPIR-V module and type context
    spirv_mod = SPIRVModule()
    type_ctx = SPIRVTypeContext(spirv_mod, ptm)

    # Setup module header
    setup_memory_model!(spirv_mod; physical_storage_buffer=true)
    require_capability!(spirv_mod, Cap.Shader)
    require_capability!(spirv_mod, Cap.VariablePointers)
    require_extension!(spirv_mod, "SPV_KHR_variable_pointers")

    # Build struct pointer member type map (resolves ptr members in structs)
    build_struct_ptr_member_types!(type_ctx, llvm_mod)

    # Pre-collect all types used in the module
    collect_module_types!(type_ctx, llvm_mod)

    # Create emitter state
    state = SPIRVEmitterState(spirv_mod, type_ctx)

    # Find the entry function
    entry_fn = LLVM.functions(llvm_mod)[entry_name]

    # Emit global variables (if any — needed for builtin inputs, etc.)
    interface_ids = _emit_globals!(state, llvm_mod)

    # Check if entry function has parameters
    fn_ty = LLVM.function_type(entry_fn)
    n_params = length(collect(LLVM.parameters(fn_ty)))

    if n_params == 0
        # No parameters — emit directly as entry point
        func_id = emit_function!(state, entry_fn; is_entry=true)
    else
        # Entry functions in Vulkan SPIR-V cannot have parameters.
        # Create a parameterless wrapper that calls the kernel.
        # For now, pass undef values for parameters (real BDA wrapper comes later).
        func_id = _emit_entry_wrapper!(state, entry_fn)
    end

    # Setup entry point and execution modes
    emit_entry_point!(spirv_mod, ExecModel.GLCompute, func_id, "main", interface_ids)
    emit_execution_mode!(spirv_mod, func_id, ExecMode.LocalSize,
                         UInt32(workgroup_size[1]), UInt32(workgroup_size[2]), UInt32(workgroup_size[3]))

    # Add debug name
    emit_name!(spirv_mod, func_id, entry_name)

    # Add Block + MemberOffset decorations for PSB-pointed struct types
    decorate_psb_struct_layouts!(type_ctx, llvm_mod)

    # Serialize to binary, return bytes + source map
    return serialize(spirv_mod), spirv_mod.source_locations
end

"""
Emit a parameterless entry wrapper that calls the kernel function.
For now, passes undef values for parameters. The real BDA entry wrapper will
load arguments from a device-memory buffer via PhysicalStorageBuffer.
"""
function _emit_entry_wrapper!(state::SPIRVEmitterState, entry_fn::LLVM.Function)
    mod = state.mod

    # First, emit the original function as a non-entry function
    inner_func_id = emit_function!(state, entry_fn; is_entry=false)

    # Create wrapper function type: void()
    void_ty = emit_type_void!(mod)
    wrapper_fn_ty = emit_type_function!(mod, void_ty)

    # Create wrapper function
    wrapper_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpFunction, void_ty, wrapper_id, FuncControl.None, wrapper_fn_ty)

    # Entry block
    label_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLabel, label_id)

    # Build undef arguments for each parameter
    arg_ids = UInt32[]
    for param in LLVM.parameters(entry_fn)
        param_ty = LLVM.value_type(param)
        if param_ty isa LLVM.PointerType
            param_spirv_ty = map_pointer_type_for_value!(state.type_ctx, param)
        else
            param_spirv_ty = map_type!(state.type_ctx, param_ty)
        end
        # Create OpUndef for each parameter
        undef_id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, UInt16(1), param_spirv_ty, undef_id)  # OpUndef
        push!(arg_ids, undef_id)
    end

    # Call the inner function
    word_count = UInt32(4 + length(arg_ids))
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpFunctionCall))
    push!(mod.functions, void_ty)
    call_result_id = fresh_id!(mod)
    push!(mod.functions, call_result_id)
    push!(mod.functions, inner_func_id)
    append!(mod.functions, arg_ids)

    # Return
    encode_instruction!(mod.functions, Op.OpReturn)
    encode_instruction!(mod.functions, Op.OpFunctionEnd)

    return wrapper_id
end

"""
Emit global variables from the LLVM module. Returns interface variable IDs for OpEntryPoint.
Handles push constant globals (addrspace 2) and builtin globals (addrspace 7).
"""
function _emit_globals!(state::SPIRVEmitterState, llvm_mod::LLVM.Module)
    interface_ids = UInt32[]

    for gv in LLVM.globals(llvm_mod)
        gv_ty = LLVM.value_type(gv)
        gv_ty isa LLVM.PointerType || continue
        as = LLVM.addrspace(gv_ty)

        if as == 2
            # Push constant global (addrspace 2)
            var_id = _emit_push_constant_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 3
            # Workgroup/shared memory global (addrspace 3)
            var_id = _emit_workgroup_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 7
            # Input global (addrspace 7) — SPIR-V builtins
            var_id = _emit_builtin_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 1
            # Julia constant global (addrspace 1) — lookup tables like _j_const_N
            var_id = _emit_constant_global!(state, gv)
            if var_id !== nothing
                push!(interface_ids, var_id)  # SPIR-V 1.4+ requires all globals in interface
            end
        end
    end

    return interface_ids
end

"""
Emit a push constant global variable in SPIR-V.
Creates the struct type with Block decoration, pointer type, and OpVariable.
"""
function _emit_push_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)

    # Map the struct type
    struct_spirv_id = map_type!(state.type_ctx, gv_value_ty)

    # Add Block decoration to the struct type (required for PushConstant)
    emit_decorate!(mod, struct_spirv_id, Dec.Block)

    # Add MemberOffset decorations
    if gv_value_ty isa LLVM.StructType
        members = LLVM.elements(gv_value_ty)
        offset = UInt32(0)
        for (i, member_ty) in enumerate(members)
            emit_member_decorate!(mod, struct_spirv_id, UInt32(i - 1), Dec.Offset, offset)
            offset += UInt32(_compute_type_size(member_ty))
        end
    end

    # Create pointer type: OpTypePointer PushConstant %struct_type
    ptr_ty = map_pointer_type!(state.type_ctx, struct_spirv_id, SC.PushConstant)

    # Create OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.PushConstant)

    # Register in value_map so loads can find it
    state.value_map[gv] = var_id

    # Register pointee type in PTM for downstream pointer type resolution
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    return var_id
end

"""
Emit a workgroup (shared memory) global variable in SPIR-V.
Creates OpVariable with Workgroup storage class for addrspace(3) globals.
These are used by KA's @localmem for shared memory within a workgroup.

Workgroup variables must NOT have explicit layout decorations (ArrayStride,
Offset, Block) unless VK_KHR_workgroup_memory_explicit_layout is enabled.
We create FRESH type IDs (via `map_workgroup_type!`) separate from the main
cache, so PSB types keep their decorations while workgroup types stay clean.
"""
function _emit_workgroup_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)

    # Check if this workgroup type contains a struct (needs explicit layout).
    # Without explicit layout, NVIDIA drivers may use incorrect memory layout
    # for struct types in workgroup storage, corrupting mixed-type tuple data.
    needs_explicit_layout = _wg_type_contains_struct(gv_value_ty)

    # Create FRESH type IDs for workgroup usage.
    # map_workgroup_type! now adds MemberOffset/ArrayStride decorations when
    # the type contains structs (for VK_KHR_workgroup_memory_explicit_layout).
    pointee_spirv = map_workgroup_type!(state.type_ctx, gv_value_ty)

    if needs_explicit_layout
        # Wrap in a Block-decorated outer struct for explicit layout.
        # SPIR-V requires: Block on outermost struct, Offset on its members,
        # ArrayStride on arrays of structs, MemberOffset on inner structs.
        #
        # Structure: %outer_block = OpTypeStruct %pointee_type
        #   with Block decoration and member 0 at Offset 0
        block_struct_id = fresh_id!(mod)
        word_count = UInt32(3)  # OpTypeStruct + result + 1 member
        push!(mod.types_constants, (word_count << 16) | UInt32(Op.OpTypeStruct))
        push!(mod.types_constants, block_struct_id)
        push!(mod.types_constants, pointee_spirv)

        emit_decorate!(mod, block_struct_id, Dec.Block)
        emit_member_decorate!(mod, block_struct_id, UInt32(0), Dec.Offset, UInt32(0))

        # Create pointer type for the Block wrapper
        ptr_ty = map_pointer_type!(state.type_ctx, block_struct_id, SC.Workgroup)

        # Require the extension and capability
        require_capability!(mod, Cap.WorkgroupMemoryExplicitLayoutKHR)
        require_extension!(mod, "SPV_KHR_workgroup_memory_explicit_layout")

        # Require 8-bit/16-bit access capabilities if the type contains such elements
        if _wg_type_contains_width(gv_value_ty, 8)
            require_capability!(mod, Cap.WorkgroupMemoryExplicitLayout8BitAccessKHR)
        end
        if _wg_type_contains_width(gv_value_ty, 16)
            require_capability!(mod, Cap.WorkgroupMemoryExplicitLayout16BitAccessKHR)
        end

        # Create OpVariable
        var_id = fresh_id!(mod)
        encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Workgroup)

        # Register the wrapping so the function preamble can emit unwrapping AccessChains.
        # The unwrapping AccessChain will drill through the Block struct to get a pointer
        # to the inner array/type, which all existing GEP handlers expect.
        state.wg_wrapped_vars[gv] = (var_id, pointee_spirv, gv_value_ty)
    else
        # No struct types — use simple undecorated workgroup variable (original path)
        ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, SC.Workgroup)

        var_id = fresh_id!(mod)
        encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Workgroup)

        # Register directly in value_map
        state.value_map[gv] = var_id
    end

    # Register pointee type in PTM for downstream GEP/load/store resolution
    # (always the ORIGINAL type, not the Block wrapper)
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    # Debug name
    gv_name = LLVM.name(gv)
    if !isempty(gv_name)
        emit_name!(mod, var_id, gv_name)
    end

    return var_id
end

"""Check if an LLVM type contains an integer element of the given bit width."""
function _wg_type_contains_width(ty::LLVM.LLVMType, bits::Int)
    if ty isa LLVM.IntegerType
        return LLVM.width(ty) == bits
    elseif ty isa LLVM.StructType
        return any(mt -> _wg_type_contains_width(mt, bits), LLVM.elements(ty))
    elseif ty isa LLVM.ArrayType
        return _wg_type_contains_width(LLVM.eltype(ty), bits)
    else
        return false
    end
end

"""Check if an LLVM type contains a struct at any nesting level."""
function _wg_type_contains_struct(ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        return true
    elseif ty isa LLVM.ArrayType
        return _wg_type_contains_struct(eltype(ty))
    else
        return false
    end
end

# ── Builtin name → SPIR-V BuiltIn decoration mapping ──
const _SPIRV_BUILTIN_MAP = Dict{String, UInt32}(
    "__spirv_BuiltInGlobalInvocationId"   => BuiltIn.GlobalInvocationId,
    "__spirv_BuiltInLocalInvocationId"    => BuiltIn.LocalInvocationId,
    "__spirv_BuiltInWorkgroupId"          => BuiltIn.WorkgroupId,
    "__spirv_BuiltInNumWorkgroups"        => BuiltIn.NumWorkgroups,
    "__spirv_BuiltInWorkgroupSize"        => BuiltIn.WorkgroupSize,
    "__spirv_BuiltInLocalInvocationIndex" => BuiltIn.LocalInvocationIndex,
)

"""
Emit a builtin Input global variable in SPIR-V.
Creates the appropriate type (vec3<u32> or u32), pointer type, OpVariable,
and BuiltIn decoration based on the global's name.
"""
function _emit_builtin_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_name = LLVM.name(gv)
    gv_value_ty = LLVM.global_value_type(gv)

    # Look up builtin decoration from name
    builtin_id = get(_SPIRV_BUILTIN_MAP, gv_name, nothing)
    if builtin_id === nothing
        error("Unknown SPIR-V builtin global: $gv_name")
    end

    # Map the value type (e.g., <3 x i32> → OpTypeVector(i32, 3), or i32)
    value_spirv_id = map_type!(state.type_ctx, gv_value_ty)

    # Create pointer type: OpTypePointer Input %value_type
    ptr_ty = map_pointer_type!(state.type_ctx, value_spirv_id, SC.Input)

    # Create OpVariable in Input storage class
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Input)

    # Add BuiltIn decoration
    emit_decorate!(mod, var_id, Dec.BuiltIn, builtin_id)

    # Add debug name
    emit_name!(mod, var_id, gv_name)

    # Register in value_map so loads can find it
    state.value_map[gv] = var_id

    # Register pointee type in PTM
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    return var_id
end

"""
Emit a Julia constant global (addrspace 1) as a Private-scope SPIR-V variable
with a constant initializer. These are lookup tables generated by Julia's
math library (e.g., polynomial coefficients for log/exp/pow).

Strategy: Create an OpConstantComposite for the array data, then declare
an OpVariable in Private storage class with the composite as initializer.
"""
function _emit_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)
    gv_name = LLVM.name(gv)

    # Must be an array type
    if !(gv_value_ty isa LLVM.ArrayType)
        @warn "Skipping non-array constant global: $gv_name (type: $gv_value_ty)"
        return
    end

    # Must have an initializer
    init = LLVM.initializer(gv)
    if init === nothing
        @warn "Skipping constant global without initializer: $gv_name"
        return
    end

    # Map the array type to SPIR-V
    arr_spirv_ty = map_type!(state.type_ctx, gv_value_ty)

    # Build the composite constant recursively
    composite_id = _emit_llvm_constant!(state, init, gv_value_ty)

    # Create pointer type: OpTypePointer Private %array_type
    ptr_ty = map_pointer_type!(state.type_ctx, arr_spirv_ty, SC.Private)

    # Create OpVariable with Private SC and initializer
    var_id = fresh_id!(mod)
    word_count = UInt32(5)  # result_type, result_id, storage_class, initializer
    push!(mod.global_vars, (word_count << 16) | UInt32(Op.OpVariable))
    push!(mod.global_vars, ptr_ty)
    push!(mod.global_vars, var_id)
    push!(mod.global_vars, UInt32(SC.Private))
    push!(mod.global_vars, composite_id)

    # Register in value_map
    state.value_map[gv] = var_id

    # Register pointee type in PTM
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    # Debug name
    emit_name!(mod, var_id, gv_name)

    return var_id
end

"""
Emit an LLVM constant value as a SPIR-V constant, returning its ID.
Handles ConstantInt, ConstantFP, ConstantDataArray, ConstantArray, ConstantStruct.
"""
function _emit_llvm_constant!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.LLVMType)
    # Scalar constants
    if val isa LLVM.ConstantInt || val isa LLVM.ConstantFP ||
       val isa LLVM.UndefValue || val isa LLVM.PoisonValue ||
       val isa LLVM.ConstantAggregateZero
        return map_constant!(state.type_ctx, val)
    end

    # ConstantDataArray/ConstantDataVector — packed scalar data
    if val isa LLVM.ConstantDataSequential
        return _emit_constant_data_array!(state, val, ty)
    end

    # ConstantArray — array of aggregate elements
    if ty isa LLVM.ArrayType
        return _emit_constant_array!(state, val, ty)
    end

    # ConstantStruct
    if ty isa LLVM.StructType
        return _emit_constant_struct!(state, val, ty)
    end

    error("Unsupported LLVM constant type for emission: $(typeof(val)), LLVM type: $ty")
end

"""Emit a ConstantDataArray (packed scalar data) as OpConstantComposite."""
function _emit_constant_data_array!(state::SPIRVEmitterState, val::LLVM.ConstantDataSequential, ty::LLVM.ArrayType)
    n = LLVM.length(ty)
    elem_ty = LLVM.eltype(ty)
    arr_spirv_ty = map_type!(state.type_ctx, ty)

    # Extract each element via LLVMGetElementAsConstant
    elem_ids = UInt32[]
    for i in 0:(n-1)
        elem_ref = LLVM.API.LLVMGetElementAsConstant(val, UInt32(i))
        elem_val = LLVM.Value(elem_ref)::LLVM.Constant
        elem_id = map_constant!(state.type_ctx, elem_val)
        push!(elem_ids, elem_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(elem_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, arr_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, elem_ids)

    return composite_id
end

"""Emit a ConstantArray (array of aggregate elements) as OpConstantComposite."""
function _emit_constant_array!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.ArrayType)
    n = LLVM.length(ty)
    elem_ty = LLVM.eltype(ty)
    arr_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    elem_ids = UInt32[]
    for i in 1:n
        elem_id = _emit_llvm_constant!(state, ops[i]::LLVM.Constant, elem_ty)
        push!(elem_ids, elem_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(elem_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, arr_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, elem_ids)

    return composite_id
end

"""Emit a ConstantStruct as OpConstantComposite."""
function _emit_constant_struct!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.StructType)
    members = LLVM.elements(ty)
    struct_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    member_ids = UInt32[]
    for (i, member_ty) in enumerate(members)
        member_id = _emit_llvm_constant!(state, ops[i]::LLVM.Constant, member_ty)
        push!(member_ids, member_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(member_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, struct_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, member_ids)

    return composite_id
end

# ── Stage 3: SPIR-V Validation ──

"""
Validate SPIR-V binary using spirv-val. On failure, writes debug artifacts
to /tmp/lava_last.{spv,dis,ll} and throws with a focused error excerpt.
"""
function _validate_spirv(spirv_bytes::Vector{UInt8}, llvm_ir::String="",
                          source_map::Dict{UInt32, Tuple{String, Int}}=Dict{UInt32, Tuple{String, Int}}())
    spirv_val = SPIRV_Tools_jll.spirv_val()
    spirv_dis_cmd = SPIRV_Tools_jll.spirv_dis()
    spv_path = "/tmp/lava_last.spv"

    # Write SPIR-V binary so spirv-val can read it
    write(spv_path, spirv_bytes)

    # Validate — capture stderr via temp file (spirv-val writes errors to stderr)
    val_err_file = tempname()
    p = run(pipeline(`$spirv_val --target-env vulkan1.3 --scalar-block-layout $spv_path`;
                      stderr=val_err_file, stdout=devnull); wait=false)
    wait(p)
    val_errors = isfile(val_err_file) ? read(val_err_file, String) : ""
    rm(val_err_file; force=true)
    p.exitcode == 0 && return nothing

    # ── Validation failed — build a useful error message ──

    # Disassemble and save
    dis = try
        read(`$spirv_dis_cmd --no-color $spv_path`, String)
    catch
        ""
    end
    if !isempty(dis)
        write("/tmp/lava_last.dis", dis)
    end

    io = IOBuffer()

    # ── Extract SPIR-V IDs from error messages and map to Julia source ──
    # spirv-val errors reference IDs as %<num>, e.g. "error: ... %42 ..."
    error_ids = UInt32[]
    for m in eachmatch(r"%(\d+)", val_errors)
        push!(error_ids, parse(UInt32, m.captures[1]))
    end
    unique!(error_ids)

    # Build a reverse map: disassembly line → SPIR-V ID (for context display)
    # spirv-dis output has lines like "  %42 = OpFAdd %float %40 %41"
    dis_lines = isempty(dis) ? String[] : split(dis, '\n')
    id_to_dis_line = Dict{UInt32, Int}()
    for (i, line) in enumerate(dis_lines)
        m = match(r"^\s*%(\d+)\b", line)
        if m !== nothing
            id_to_dis_line[parse(UInt32, m.captures[1])] = i
        end
    end

    # ── Julia source locations for error IDs ──
    julia_locations = Tuple{UInt32, String, Int}[]  # (spirv_id, file, line)
    for id in error_ids
        loc = get(source_map, id, nothing)
        if loc !== nothing
            push!(julia_locations, (id, loc[1], loc[2]))
        end
    end

    if !isempty(julia_locations)
        println(io, "\n  Julia source locations for problematic SPIR-V instructions:")
        seen_files = Set{Tuple{String, Int}}()
        for (id, file, line) in julia_locations
            key = (file, line)
            key in seen_files && continue
            push!(seen_files, key)
            # Shorten path for readability
            short_file = _shorten_path(file)
            println(io, "    %$id → $short_file:$line")
            # Try to show the Julia source line
            _print_julia_source_context(io, file, line)
        end
    end

    # ── SPIR-V disassembly context around error sites ──
    error_line_nums = Int[]
    for m in eachmatch(r"line (\d+):", val_errors)
        push!(error_line_nums, parse(Int, m.captures[1]))
    end
    # Also add lines for error IDs found in disassembly
    for id in error_ids[1:min(5, end)]
        lnum = get(id_to_dis_line, id, 0)
        lnum > 0 && push!(error_line_nums, lnum)
    end
    unique!(error_line_nums)

    if !isempty(error_line_nums) && !isempty(dis_lines)
        for lnum in error_line_nums[1:min(3, end)]
            lo = max(1, lnum - 5)
            hi = min(length(dis_lines), lnum + 3)
            println(io, "  ┌─ SPIR-V around line $lnum:")
            for j in lo:hi
                marker = j == lnum ? " >> " : "    "
                # Annotate with Julia source if this line has a mapped ID
                m = match(r"^\s*%(\d+)\b", dis_lines[j])
                annotation = ""
                if m !== nothing
                    id = parse(UInt32, m.captures[1])
                    loc = get(source_map, id, nothing)
                    if loc !== nothing
                        annotation = "  ← $(_shorten_path(loc[1])):$(loc[2])"
                    end
                end
                println(io, "  │$marker$j: ", dis_lines[j], annotation)
            end
            println(io, "  └───")
        end
    end

    excerpt = String(take!(io))

    error("""
    SPIR-V validation failed!

    spirv-val:
    $(strip(val_errors))
    $excerpt
    Debug files:
      LLVM IR:     /tmp/lava_last.ll
      SPIR-V bin:  /tmp/lava_last.spv
      SPIR-V dis:  /tmp/lava_last.dis
    """)
end

"""Shorten a file path for error display."""
function _shorten_path(path::String)
    # Try to make paths relative to common prefixes
    for prefix in ("/home/sim/programmieren/VulkanDev/dev/",
                    "/home/sim/.julia/packages/")
        if startswith(path, prefix)
            return path[length(prefix)+1:end]
        end
    end
    # For Julia stdlib, show just the filename
    m = match(r"/share/julia/stdlib/[^/]+/(.+)$", path)
    m !== nothing && return "julia/" * m.captures[1]
    # For boot/base files
    m = match(r"([^/]+\.jl)$", path)
    m !== nothing && return m.captures[1]
    return path
end

"""Print Julia source context around a line for error messages."""
function _print_julia_source_context(io::IO, file::String, line::Int)
    isfile(file) || return
    try
        src_lines = readlines(file)
        lo = max(1, line - 1)
        hi = min(length(src_lines), line + 1)
        for j in lo:hi
            marker = j == line ? " >> " : "    "
            println(io, "      │$marker$j: ", src_lines[j])
        end
    catch
    end
end

"""
Disassemble SPIR-V binary to text for debugging.
"""
function disassemble_spirv(spirv_bytes::Vector{UInt8})
    spirv_dis = SPIRV_Tools_jll.spirv_dis()
    tmpfile = tempname() * ".spv"
    try
        write(tmpfile, spirv_bytes)
        return read(`$spirv_dis --no-color $tmpfile`, String)
    finally
        rm(tmpfile; force=true)
    end
end

# ── Lower unsupported LLVM intrinsics ──

"""
Lower LLVM intrinsics that SPIR-V cannot represent:
- llvm.memcpy → typed load + store (using destination alloca's type)
- llvm.lifetime.start/end → removed (no-op)
"""
function _lower_unsupported_intrinsics!(mod::LLVM.Module)
    to_erase = LLVM.Instruction[]

    for fn in LLVM.functions(mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                called = LLVM.called_operand(inst)
                called isa LLVM.Function || continue
                fname = LLVM.name(called)

                if startswith(fname, "llvm.memcpy")
                    _lower_memcpy!(inst)
                    push!(to_erase, inst)
                elseif startswith(fname, "llvm.memset")
                    _lower_memset!(inst)
                    push!(to_erase, inst)
                elseif startswith(fname, "llvm.lifetime")
                    push!(to_erase, inst)
                end
            end
        end
    end

    for inst in to_erase
        LLVM.erase!(inst)
    end

    # Remove dead intrinsic declarations
    for fn in collect(LLVM.functions(mod))
        fname = LLVM.name(fn)
        if (startswith(fname, "llvm.memcpy") || startswith(fname, "llvm.memset") ||
            startswith(fname, "llvm.lifetime")) && isempty(LLVM.uses(fn))
            LLVM.erase!(fn)
        end
    end
end

"""
Lower a single memcpy call to a typed load + store.
If the destination is an alloca, use the alloca's type for the load/store
so SPIR-V doesn't need to bitcast structs.
"""
function _lower_memcpy!(inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dst = ops[1]
    src = ops[2]

    # Try to find the alloca's element type for typed load/store
    copy_type = nothing
    if dst isa LLVM.AllocaInst
        copy_type = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(dst))
    end

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        if copy_type !== nothing
            # Typed load/store using the alloca's type
            val = LLVM.load!(builder, copy_type, src, "memcpy_val")
            LLVM.store!(builder, val, dst)
        else
            # Fallback: copy using i32 chunks (NOT i64 — avoids misaligned i64 stores
            # on NVIDIA when the destination is a PSB pointer into a non-8-byte-aligned
            # struct array, e.g. 12-byte or 52-byte structs where odd-indexed elements
            # are only 4-aligned).
            len_val = ops[3]
            if !(len_val isa LLVM.ConstantInt)
                error("Cannot lower memcpy with non-constant length: $inst")
            end
            nbytes = convert(Int, len_val)
            T_i32 = LLVM.Int32Type()
            T_i64 = LLVM.Int64Type()
            T_i8 = LLVM.Int8Type()
            offset = 0
            n_words = nbytes ÷ 4
            for i in 0:(n_words-1)
                off = i * 4
                s_ptr = off == 0 ? src : LLVM.gep!(builder, T_i8, src, [LLVM.ConstantInt(T_i64, off)])
                d_ptr = off == 0 ? dst : LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, off)])
                val = LLVM.load!(builder, T_i32, s_ptr)
                LLVM.alignment!(val, 4)
                st = LLVM.store!(builder, val, d_ptr)
                LLVM.alignment!(st, 4)
            end
            # Handle tail bytes
            for i in (n_words*4):(nbytes-1)
                src_ptr = LLVM.gep!(builder, T_i8, src, [LLVM.ConstantInt(T_i64, i)])
                dst_ptr = LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, i)])
                val = LLVM.load!(builder, T_i8, src_ptr)
                LLVM.store!(builder, val, dst_ptr)
            end
        end
    end
end

"""
Lower a single memset call to explicit stores.
SPIR-V has no memset intrinsic, so we replace with i32 stores (4-byte chunks)
plus i8 tail stores. For addrspace 0 (allocas), uses GEP-based addressing.
"""
function _lower_memset!(inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dst = ops[1]
    fill_val = ops[2]  # i8
    len_val = ops[3]

    if !(len_val isa LLVM.ConstantInt)
        error("Cannot lower memset with non-constant length: $inst")
    end
    nbytes = convert(Int, len_val)

    T_i8 = LLVM.Int8Type()
    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        # Build fill word: replicate i8 val to i32
        val32 = LLVM.zext!(builder, fill_val, T_i32)
        v1 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 8))
        val32 = LLVM.or!(builder, val32, v1)
        v2 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 16))
        val32 = LLVM.or!(builder, val32, v2)

        # Store i32 chunks via byte-offset GEPs
        n_words = nbytes ÷ 4
        for i in 0:(n_words-1)
            off = i * 4
            ptr = if off == 0
                dst
            else
                LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, off)])
            end
            st = LLVM.store!(builder, val32, ptr)
            LLVM.alignment!(st, 4)
        end

        # Handle tail bytes
        for i in (n_words*4):(nbytes-1)
            ptr = LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, i)])
            st = LLVM.store!(builder, fill_val, ptr)
            LLVM.alignment!(st, 1)
        end
    end
end

# ── Compiler caches ──

const _compiler_cache = Dict{Any, Any}()
const _kernel_cache = Dict{UInt, Any}()
const _compile_lock = ReentrantLock()
