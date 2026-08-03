# Main compilation pipeline for Lava.jl
#
# Pipeline: Julia function → GPUCompiler → LLVM IR → LLVM passes → custom SPIR-V emitter
#
# Two entry points:
#   lava_compile_to_llvm()  — returns LLVM IR string (for debugging)
#   lava_compile_to_spirv() — returns validated SPIR-V binary

# Debug counter for unique kernel file naming
# Genuinely process-level, and staying: it indexes the numbered IR files the
# compiler writes to `tmp_kernels/`, and compilation happens before a device is
# chosen — there is no context to hang it on. Not device state.
const KERNEL_DEBUG_COUNTER = Ref(0)

"""
Wrap GPUCompiler.InvalidIRError with Lava-specific context and actionable suggestions.
Called from compilation entry points to provide better user-facing errors.
"""
function wrap_gpu_compiler_error(@nospecialize(e), @nospecialize(f), @nospecialize(tt))
    e isa GPUCompiler.InvalidIRError || rethrow(e)

    # `nameof` fails only for a callable without a name (a closure instance);
    # anything else here is a bug in the error formatter and must not be masked
    # while it is formatting somebody else's error.
    fname = try
        string(nameof(typeof(f)))
    catch ex
        ex isa Union{ArgumentError, MethodError} || rethrow()
        string(f)
    end
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
    call_chain_summary = extract_call_chains(err_str)

    throw(LavaCompilationError(
        "kernel compilation",
        "Cannot compile $fname($(join(tt.parameters, ", "))) to GPU code",
        suggestion;
        raw_error=err_str,
        call_chains=call_chain_summary
    ))
end

"""
    extract_call_chains(err_str) -> String

Parse GPUCompiler's InvalidIRError output to extract unique call chains through user code.
Deduplicates the repeated stacktraces (GPUCompiler emits one per illegal call, but many
share the same user-code path). Returns a compact summary showing:
- Each unique user-code call chain (deepest user function → kernel entry)
- The reason the deepest function is GPU-incompatible
"""
function extract_call_chains(err_str::String)
    # Parse each "Reason: ...\nStacktrace:\n [1] ...\n [2] ..." block
    blocks = parse_error_blocks(err_str)
    isempty(blocks) && return ""

    # For each block, extract the chain of user functions (skip Base/stdlib internals)
    seen_chains = Set{String}()
    unique_chains = Vector{Tuple{String, Vector{String}}}()  # (reason, [user_funcs...])

    for (reason, stack_entries) in blocks
        user_funcs = String[]
        for entry in stack_entries
            # Skip Base/stdlib internals — keep user code and Lava code
            is_internal = is_base_internal(entry)
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
        short_reason = shorten_reason(reason)
        chain = join(reverse(funcs), " → ")
        push!(lines, "  $i. $chain")
        push!(lines, "     Problem: $short_reason")
    end

    return join(lines, "\n")
end

function parse_error_blocks(err_str::String)
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
const BASE_INTERNAL_PATTERNS = [
    "/share/julia/", "boot.jl", "Base.jl", "array.jl", "strings/",
    "ryu/", "iobuffer.jl", "pointer.jl", "float.jl", "int.jl",
    "abstractarray.jl", "essentials.jl", "promotion.jl", "math.jl",
    "number.jl", "operators.jl", "reduce.jl", "dict.jl", "set.jl",
    "range.jl", "simdloop.jl", "refvalue.jl", "iterators.jl",
]

function is_base_internal(entry::String)
    for pat in BASE_INTERNAL_PATTERNS
        occursin(pat, entry) && return true
    end
    return false
end

function shorten_reason(reason::String)
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
const SPIRV_OPT_ENABLED = Ref(true)  # enabled: cleans up StructurizeCFG's redundant phis
                                      # that RADV miscompiles for loops containing `continue`

# Directory for the unconditional "last compiled kernel" debug dumps
# (LLVM IR / SPIR-V / disasm).  Defaults to the OS temp dir — `/tmp` on Unix,
# `%TEMP%` on Windows — so the writes are portable; hardcoded `/tmp/` crashed
# on Windows where it doesn't exist.  Override with LAVA_DEBUG_DIR.
lava_debug_dir() = get(ENV, "LAVA_DEBUG_DIR", tempdir())
lava_debug_path(name::AbstractString) = joinpath(lava_debug_dir(), name)

"""
    dump_spirv_to_disk(spirv_bytes::Vector{UInt8}, post_pass_ir::AbstractString,
                       kernel_name::AbstractString;
                       entry_name::AbstractString="")

If `ENV["LAVA_SPIRV_DUMP_DIR"]` is set, write the SPIR-V binary + post-pass
LLVM IR to `<dir>/<entry>__<sanitised_kernel>__<hash>.{spv,ll}`.  No-op
when the env var is unset (the compile-path fast-path stays free of file
I/O).

Used to capture every SPIR-V module Lava compiles in a session — essential
when triaging GPU-AV alignment errors so we can `spirv-dis` the exact
shader that tripped the validator and trace the offset back through the
post-pass IR.

`entry_name` is the original (pre-wrapper) function name, e.g. the Julia
kernel symbol; if non-empty it is prepended to the filename for human
readability.  `kernel_name` is the wrapper symbol (typically `"main"`).

The hash is over `spirv_bytes` (not the IR) so structurally-identical
shaders share a file across runs.
"""
function dump_spirv_to_disk(spirv_bytes::Vector{UInt8},
                            post_pass_ir::AbstractString,
                            kernel_name::AbstractString;
                            entry_name::AbstractString="")
    dir = get(ENV, "LAVA_SPIRV_DUMP_DIR", "")
    isempty(dir) && return nothing
    isdir(dir) || mkpath(dir)
    # `spirv_content_hash`, not `hash` — the sampling hash collides for modules
    # that differ in a few bytes, so two distinct dumps used to overwrite each
    # other and a triage session would compare a file with itself.
    h = string(spirv_content_hash(spirv_bytes); base=16, pad=16)
    sanitize(s) = replace(s, r"[^A-Za-z0-9_]" => "_")
    # Mangled GPUCompiler names can run to several hundred chars (every
    # type parameter is encoded in the symbol).  Filesystems cap filenames
    # at ~255 bytes, so trim entry_name to a readable prefix.  The sanitized
    # `kernel_name` (typically "main") + 16-char SPIR-V hash are unique
    # enough; entry_name is human-readable hint only.
    prefix = ""
    if !isempty(entry_name)
        s = sanitize(entry_name)
        s_trim = length(s) > 80 ? s[1:80] : s
        prefix = s_trim * "__"
    end
    base = joinpath(dir, "$(prefix)$(sanitize(kernel_name))__$(h)")
    write(base * ".spv", spirv_bytes)
    write(base * ".ll", post_pass_ir)
    return base * ".spv"
end

"""
    lava_run(cmd; timeout=180.0, label="subprocess") -> Base.Process

Spawn `cmd` and wait for it WITHOUT `wait(p)`. Once a Vulkan device is up, the
GPU driver breaks Julia's libuv SIGCHLD reaping: an external process (spirv-opt,
spirv-val, spirv-dis) exits and is reaped (no zombie), but the libuv exit
notification is lost, so `wait`/`success`/`read(cmd)` block forever. Poll the
already-updated `process_exited` state instead, with a hard timeout that kills a
genuinely-stuck child. Always check `process_exited(p) && p.exitcode == 0`
before trusting the result.
"""
function lava_run(cmd::Base.AbstractCmd; timeout::Float64=180.0,
                  label::AbstractString="subprocess")
    p = run(cmd; wait=false)
    deadline = time() + timeout
    while !process_exited(p) && time() < deadline
        sleep(0.005)
    end
    if !process_exited(p)
        @warn "Lava: $label did not report exit within $(round(Int, timeout))s — killing"
        # The process may exit between the check and the signal, which is the
        # only tolerated race; a permissions failure is real.
        try
            kill(p)
        catch ex
            ex isa Union{Base.IOError, ProcessFailedException} || rethrow()
            @warn "Lava: could not kill $label after timeout" exception = ex
        end
    end
    return p
end

"""
    run_spirv_opt(spirv_bytes::Vector{UInt8}) -> Vector{UInt8}

Run spirv-opt on the SPIR-V binary to optimize it. Uses SPIRV_Tools_jll.
This can help NVIDIA's shader compiler handle complex shaders that would
otherwise cause miscompilation (Xid 31 MMU faults with large kernels).
"""
function run_spirv_opt(spirv_bytes::Vector{UInt8})
    # `-O` runs spirv-opt's default optimization set. Of the passes this
    # triggers, `--if-conversion` is the one that specifically fixes the
    # class of RADV miscompile seen with `while true / if / continue / break`
    # walk patterns (post-StructurizeCFG, RADV sometimes mis-evaluates OpPhi
    # predecessor selection; replacing phi with OpSelect dodges it). The
    # other `-O` passes give additional cleanup without known regressions
    # in the Lava test suite, so we keep the full pipeline.
    spirv_opt = SPIRV_Tools_jll.spirv_opt()
    in_path = tempname() * ".spv"
    out_path = tempname() * ".spv"
    try
        write(in_path, spirv_bytes)
        p = lava_run(pipeline(`$spirv_opt --target-env=vulkan1.3 -O $in_path -o $out_path`;
                              stderr=devnull, stdout=devnull); label="spirv-opt")
        if process_exited(p) && p.exitcode == 0 && isfile(out_path)
            return read(out_path)
        end
        @debug "Lava: spirv-opt non-zero exit; returning unoptimized SPIR-V" exitcode=p.exitcode
    catch ex
        # An external binary that is missing or unrunnable is the tolerated case;
        # anything else is our bug. @warn, not @debug: silently shipping every
        # module unoptimised is a large, invisible regression.
        ex isa Union{Base.IOError, SystemError, Base.ProcessFailedException} || rethrow()
        @warn "Lava: spirv-opt invocation failed; returning unoptimized SPIR-V" exception=ex
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
    enable_ray_query::Bool
end
# Backward-compat constructor for disk-cache deserialization (old entries lack the field).
LavaGPUKernel(spirv_bytes, entry_name, workgroup_size, push_info, ir) =
    LavaGPUKernel(spirv_bytes, entry_name, workgroup_size, push_info, ir, false)

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
        result = lava_compile_full(f, tt; workgroup_size, validate)
        return result
    elseif stage in (:vertex, :fragment, :geometry, :tess_control, :tess_eval)
        return lava_compile_gfx_full(f, tt; stage, config, validate)
    elseif stage in (:raygen, :closesthit, :miss, :anyhit, :intersection, :callable)
        return lava_compile_rt_full(f, tt; stage, payload_type, validate)
    else
        error("Unknown stage: $stage")
    end
end

"""Compile compute kernel, capturing pre/post IR."""
function lava_compile_full(@nospecialize(f), @nospecialize(tt);
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
            wrap_gpu_compiler_error(e, f, tt)
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
        run_llvm_passes!(mod, wrapper_fn)
        post_pass_ir = string(mod)

        # SPIR-V emission
        spirv_bytes, source_map = emit_spirv_from_llvm(mod, wrapper_name, workgroup_size)

        # Validation
        write(lava_debug_path("lava_last.spv"), spirv_bytes)
        write(lava_debug_path("lava_last.ll"), post_pass_ir)
        dump_spirv_to_disk(spirv_bytes, post_pass_ir, wrapper_name; entry_name)
        if validate
            validate_spirv(spirv_bytes, post_pass_ir, source_map)
        end

        spirv_disasm = disassemble_spirv(spirv_bytes)

        return CompilationResult(pre_pass_ir, post_pass_ir, spirv_bytes, spirv_disasm,
                                 wrapper_name, :compute, workgroup_size, push_info, source_map)
    end
end

"""Compile graphics shader, capturing pre/post IR."""
function lava_compile_gfx_full(@nospecialize(f), @nospecialize(tt);
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
            wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        pre_pass_ir = string(mod)

        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # GFX shader emission doesn't have the multi-OpFunction walker yet;
        # keep the old single-OpFunction behavior for now.
        run_llvm_passes!(mod, wrapper_fn; force_inline_all=true)
        post_pass_ir = string(mod)

        spirv_bytes, source_map = emit_spirv_from_llvm_gfx(mod, wrapper_name, stage; config=config)

        write(lava_debug_path("lava_last.spv"), spirv_bytes)
        write(lava_debug_path("lava_last.ll"), post_pass_ir)
        dump_spirv_to_disk(spirv_bytes, post_pass_ir, wrapper_name; entry_name)
        if validate
            validate_spirv(spirv_bytes, post_pass_ir, source_map)
        end

        spirv_disasm = disassemble_spirv(spirv_bytes)

        return CompilationResult(pre_pass_ir, post_pass_ir, spirv_bytes, spirv_disasm,
                                 wrapper_name, stage, (1, 1, 1), push_info, source_map)
    end
end

"""Compile RT shader, capturing pre/post IR."""
function lava_compile_rt_full(@nospecialize(f), @nospecialize(tt);
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
            wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        pre_pass_ir = string(mod)

        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # RT shader emission doesn't have the multi-OpFunction walker yet;
        # keep the old single-OpFunction behavior for now.
        run_llvm_passes!(mod, wrapper_fn; force_inline_all=true)
        post_pass_ir = string(mod)

        spirv_bytes, source_map = emit_spirv_from_llvm_rt(mod, wrapper_name, stage;
                                                payload_type=payload_type)

        write(lava_debug_path("lava_last.spv"), spirv_bytes)
        write(lava_debug_path("lava_last.ll"), post_pass_ir)
        dump_spirv_to_disk(spirv_bytes, post_pass_ir, wrapper_name; entry_name)
        if validate
            validate_spirv(spirv_bytes, post_pass_ir, source_map)
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
        po = lava_run(`$spirv_opt --target-env=vulkan1.3 --scalar-block-layout $passes $in_path -o $out_path`; label="spirv-opt")
        (process_exited(po) && po.exitcode == 0) || error("spirv-opt failed (exit $(process_exited(po) ? po.exitcode : "timeout"))")
        optimized = read(out_path)
        # Validate optimized output
        pv = lava_run(`$spirv_val_cmd --target-env vulkan1.3 --scalar-block-layout $out_path`; label="spirv-val")
        (process_exited(pv) && pv.exitcode == 0) || error("spirv-val failed on optimized SPIR-V")
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
            wrap_gpu_compiler_error(e, f, tt)
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
            wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # ── Stage 1: LLVM passes ──
        run_llvm_passes!(mod, entry_fn)

        # Save IR for debugging (after passes, before emission)
        ir = string(mod)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes, source_map = emit_spirv_from_llvm(mod, entry_name, workgroup_size)

        # ── Stage 3: Validation ──
        if validate
            validate_spirv(spirv_bytes, "", source_map)
        end

        return LavaSPIRVResult(spirv_bytes, entry_name, workgroup_size, ir)
    end
end

"""
    lava_compile_gpu_from_job(job::CompilerJob; validate=true) -> LavaGPUKernel

Run the full LLVM → SPIR-V pipeline on a pre-built `CompilerJob`.  This is the
`compiler` function passed to `GPUCompiler.cached_compilation` in `launch.jl`,
so the cache keys it off of Julia's `MethodInstance` (type-based) and gets
proper world-age tracking for free.
"""
function lava_compile_gpu_from_job(job::GPUCompiler.CompilerJob;
                                    enable_ray_query::Bool = job.config.params.enable_ray_query,
                                    validate::Bool = true,
                                    force_inline_all::Bool = false)
    workgroup_size = job.config.params.workgroup_size
    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            wrap_gpu_compiler_error(e, job.source.def.sig, job.source.specTypes)
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
        run_llvm_passes!(mod, wrapper_fn; force_inline_all)

        # Save IR for debugging
        ir = string(mod)
        KERNEL_DEBUG_COUNTER[] += 1
        _kidx = KERNEL_DEBUG_COUNTER[]
        _dbg_dir = joinpath(@__DIR__, "..", "..", "..", "tmp_kernels")
        mkpath(_dbg_dir)
        write(joinpath(_dbg_dir, "kernel_$(_kidx)_$(replace(wrapper_name, r"[^a-zA-Z0-9_]" => "_")).ll"), ir)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes, source_map = emit_spirv_from_llvm(mod, wrapper_name, workgroup_size;
                                                        enable_ray_query)

        # ── Stage 2.5: SPIR-V optimization (optional, helps NVIDIA) ──
        if SPIRV_OPT_ENABLED[]
            spirv_bytes = run_spirv_opt(spirv_bytes)
        end

        # Save SPIR-V for debugging
        write(joinpath(_dbg_dir, "kernel_$(_kidx)_$(replace(wrapper_name, r"[^a-zA-Z0-9_]" => "_")).spv"), spirv_bytes)
        dump_spirv_to_disk(spirv_bytes, ir, wrapper_name; entry_name)

        # ── Stage 3: Validation ──
        if validate
            validate_spirv(spirv_bytes, ir, source_map)
        end

        return LavaGPUKernel(spirv_bytes, wrapper_name, workgroup_size, push_info, ir, enable_ray_query)
    end
end

"""
    lava_compile_gpu(f, tt; workgroup_size=(64,1,1), validate=true) -> LavaGPUKernel

Full GPU-ready compilation pipeline with BDA entry wrapper.  Thin wrapper
around `lava_compile_gpu_from_job` that builds the `CompilerJob` from
`(f, tt, workgroup_size)`.  Kept as a public convenience for callers that
don't want to construct a job manually.
"""
function lava_compile_gpu(@nospecialize(f), @nospecialize(tt);
                           workgroup_size::NTuple{3,Int} = (64, 1, 1),
                           enable_ray_query::Bool = false,
                           validate::Bool = true,
                           force_inline_all::Bool = false)
    if enable_ray_query && !vk_context().ray_query_available
        error("lava_compile_gpu: enable_ray_query=true requested, but the " *
              "active Vulkan device does not support VK_KHR_ray_query. " *
              "Either run on a device that supports ray_query (e.g. RADV, " *
              "lavapipe) or call lava_compile_gpu without enable_ray_query.")
    end
    config = lava_compiler_config(; workgroup_size, enable_ray_query)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)
    return lava_compile_gpu_from_job(job; enable_ray_query, validate, force_inline_all)
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
    # Frozen SPIR-V, before any of the compiler runs.  Everything below —
    # GPUCompiler, the LLVM pass pipeline, structurize, the SPIR-V emitter — is
    # what an hw_accel=true scene pays in every session, and it dwarfs rendering
    # (crown: ~610 s of compile against ~7.8 s of frames).  `ctx.pipeline_cache`
    # does not help here; it caches the driver's SPIR-V → ISA step, which cannot
    # start until this function has produced the SPIR-V.
    let hit = frozen_rt_load(f, tt, stage, payload_type, push_constant_size)
        hit === nothing || return hit
    end

    # Per-material chit shaders may do inline shadow-ray traces via ray query
    # (`surface_direct_lighting_inner_typed!`), so the chit shader needs
    # `enable_ray_query=true` to bring in the TLAS variable + the rayQuery
    # capability. The cost of enabling on RT shaders that don't actually use
    # ray query is one unused descriptor binding (the driver strips dead code).
    config = lava_compiler_config(; workgroup_size=(1, 1, 1), enable_ray_query=true)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        local mod, meta
        checkpoint = PhaseTimer("[phase] "; threshold=1.0)
        try
            mod, meta = GPUCompiler.compile(:llvm, job)
        catch e
            wrap_gpu_compiler_error(e, f, tt)
        end
        checkpoint("GPUCompiler.compile(:llvm)")
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # BDA entry wrapper (same as compute — args via push constant buffer)
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]
        checkpoint("wrap_entry_for_vulkan!")

        # LLVM passes. The RT emitter now has the multi-OpFunction walker, so
        # keep callees as separate OpFunctions (force_inline_all=false): a fat
        # per-material chit stays one architecture but compiles as many small
        # functions instead of one giant inlined one (minutes -> seconds).
        run_llvm_passes!(mod, wrapper_fn; force_inline_all=false)
        checkpoint("run_llvm_passes!")

        ir = string(mod)
        write(lava_debug_path("lava_last_rt.ll"), ir)
        # Own checkpoint: on a fat chit `string(mod)` is seconds by itself, and
        # folding it into the emitter's number would point the finger at the
        # wrong phase.
        checkpoint("string(mod) + IR dump")

        # RT-specific SPIR-V emission
        spirv_bytes, source_map = emit_spirv_from_llvm_rt(mod, wrapper_name, stage;
                                                payload_type=payload_type)
        checkpoint("emit_spirv_from_llvm_rt")

        write(lava_debug_path("lava_last.spv"), spirv_bytes)
        dump_spirv_to_disk(spirv_bytes, ir, wrapper_name; entry_name)

        if validate
            validate_spirv(spirv_bytes, ir, source_map)
            checkpoint("validate_spirv")
        end

        shader = LavaRTShader(spirv_bytes, stage, push_info, ir)
        frozen_rt_store(f, tt, stage, payload_type, push_constant_size, shader)
        return shader
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
            wrap_gpu_compiler_error(e, f, tt)
        end
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # BDA entry wrapper (same as compute — args via push constant buffer)
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size=(1, 1, 1))
        wrapper_name = push_info.wrapper_name
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # LLVM passes (GFX emit doesn't have the multi-OpFunction walker yet;
        # keep the old single-OpFunction behavior for now.)
        run_llvm_passes!(mod, wrapper_fn; force_inline_all=true)

        ir = string(mod)
        write(lava_debug_path("lava_last_gfx_$(stage).ll"), ir)

        # Graphics-specific SPIR-V emission
        spirv_bytes, source_map = emit_spirv_from_llvm_gfx(mod, wrapper_name, stage; config=config)

        write(lava_debug_path("lava_last_gfx_$(stage).spv"), spirv_bytes)

        if validate
            validate_spirv(spirv_bytes, ir, source_map)
        end

        return LavaGfxShader(spirv_bytes, stage, push_info, ir)
    end
end

# ── Stage 1: LLVM Pass Pipeline ──

"""
    LOOP_UNROLL[] :: Bool

Whether `run_llvm_passes!` runs `unroll_loops!`. **Off**, because as written it
does not actually unroll anything — see `unroll_loops!` for why, and for what
would have to change. Left in place because the diagnosis behind it is the
largest known deficit in Lava's generated code and the scaffolding is the
starting point for fixing it.
"""
const LOOP_UNROLL = Ref(false)

"""
    unroll_loops!(mod)

Rotate, canonicalise and unroll every loop in `mod`.

**Unrolling is the single largest known win in Lava's generated code, and this
pass does not currently deliver it.**

Measured, same dependent `muladd` chain, 4.19M threads, source-level unrolling
via `Base.Cartesian.@nexprs` rather than any compiler flag:

| loop body | Lava | CUDA.jl |
|---|---|---|
| 1 fma per iteration | 3905 GFLOP/s | 23445 |
| 8 fma per iteration | **13699** | 23587 |

3.5x, and it closes the Lava/CUDA gap on this kernel from 5.6x to 1.7x. LLVM
unrolls for NVPTX and does not for SPIR-V, and that difference alone accounts
for most of the flat ~2.5x deficit Lava shows on identical KernelAbstractions
source.

(An earlier note here claimed forcing the unroll made things *worse*, based on
`LLVM.clopts("--unroll-count=8", "--unroll-allow-partial")` measuring
4200 -> 2103 GFLOP/s. That was wrong: those are process-global LLVM options and
they perturb every other compilation in the session. The source-level
measurement above is the trustworthy one.)

Nothing in Lava's pipeline unrolls, and that shows up as a flat ~2.5x deficit
against identical KernelAbstractions source on CUDA.jl — uniform across shapes,
which is the signature of a per-iteration cost rather than a tiling problem. A
dependent `muladd` chain with a compile-time trip count of 256 measures
4.2 TFLOP/s here against 24.5 on CUDA.jl (this card's fp32 peak is ~26.7). The
SPIR-V is not the problem: it emits a correct `GLSL.std.450 Fma`. It wraps it in
an `OpLoopMerge` — one FMA per iteration behind a compare and a branch.

The IR reaching this point is a clean counted loop (`icmp eq i64 %iv, 256`) on a
function with no blocking attributes, so the loop is unrollable in principle.
LLVM still declines, with or without a `TargetMachine`:

  * `LoopFullUnrollPass` won't: 256 iterations is over its full-unroll threshold.
  * `LoopUnrollPass` won't unroll *partially* either, because partial unrolling
    is opt-in per target via `TargetTransformInfo::UnrollingPreferences`, and a
    SPIR-V pipeline has no GPU TTI to ask. Handing it a host `JITTargetMachine`
    does not help — verified, the fma count is unchanged.

So the fix is one of: attach explicit `llvm.loop.unroll.count` / `.full`
metadata to the latch before running the pass (which overrides the cost model
outright), or supply a TTI that reports GPU-like unrolling preferences. Both are
real work; neither is done. Until then this is a no-op that costs compile time,
hence `LOOP_UNROLL[] = false`.

Note the pass plumbing itself is correct and worth keeping: loop passes need a
`NewPMLoopPassManager` nested in a function pass manager, and the loop must be
rotated with a canonical induction variable before the trip count is visible.
"""
function unroll_loops!(mod::LLVM.Module)
    LOOP_UNROLL[] || return mod
    @dispose pb = LLVM.NewPMPassBuilder() begin
        LLVM.add!(pb, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.LoopSimplifyPass())
            LLVM.add!(fpm, LLVM.NewPMLoopPassManager()) do lpm
                LLVM.add!(lpm, LLVM.LoopRotatePass())
                LLVM.add!(lpm, LLVM.IndVarSimplifyPass())
            end
            LLVM.add!(fpm, LLVM.LoopFullUnrollPass())
            LLVM.add!(fpm, LLVM.LoopUnrollPass())
            LLVM.add!(fpm, LLVM.InstCombinePass())
            LLVM.add!(fpm, LLVM.SimplifyCFGPass())
        end
        LLVM.run!(pb, mod)
    end
    return mod
end

function run_llvm_passes!(mod::LLVM.Module, entry_fn::LLVM.Function;
                           force_inline_all::Bool=false)
    # ── CFG cleanup ──
    # Verify IR after each custom pass to catch corruption early.
    # Only in debug mode — verify is cheap but adds up across 30+ passes.
    verify_passes = get(ENV, "LAVA_VERIFY_PASSES", "") == "1"
    function verify_ir!(label)
        verify_passes || return
        try
            LLVM.verify(mod)
        catch e
            ir = string(mod)
            write(lava_debug_path("lava_broken_$(label).ll"), ir)
            error("LLVM IR verification failed after $label — dumped to /tmp/lava_broken_$(label).ll\n$(sprint(showerror, e))")
        end
    end

    checkpoint = PhaseTimer("    [pass] ")

    # Remove constructs that SPIR-V can't handle
    # Replace freeze before optimization: GPU kernel arguments are never undef,
    # so freeze is unnecessary. Removing it early lets LLVM produce simpler IR.
    replace_freeze!(mod)
    strip_assume!(mod)

    # Remove trap/unreachable from error paths (GPUCompiler's lower_throw!).
    # Vendored locally (was GPUCompiler.rm_trap!, removed in GPUCompiler 1.13.3).
    rm_trap!(mod)
    # Pass entry_fn so the pass warns (rather than silently miscompiles) if it
    # ever lowers an `unreachable` in a non-entry helper — see replace_unreachable!.
    replace_unreachable!(mod, entry_fn)
    strip_noreturn!(mod)
    verify_ir!("pre_inline")

    # ── Apply inline policy ──
    # Default: respect Julia/GPUCompiler's inlining. Only Julia-marked
    # alwaysinline (throw/box wrappers) get inlined; other helpers survive
    # as their own functions and are emitted as separate OpFunctions.
    # Pass `force_inline_all=true` for the old single-OpFunction behavior.
    checkpoint("pre_inline_cleanup")
    force_inline_all!(mod, entry_fn; force_inline_all)
    verify_ir!("force_inline")
    checkpoint("force_inline_all!")

    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write(lava_debug_path("lava_ir_1_postinline.ll"), string(mod))
    end

    # ── Outline oversized functions ──
    # AMDVLK Windows segfaults on `vkCreateComputePipelines` if any OpFunction
    # exceeds LLPC's per-function complexity limit (roughly 100-150 basic
    # blocks for vp_trace_rays-class kernels). When Julia/GPUCompiler aggressively
    # inlines a kernel into a single oversized function, run LLVM's LoopExtractor
    # to peel out loops as separate helpers. The Step 6 multi-OpFunction walker
    # emits the helpers as separate OpFunctions, keeping each one under the limit.
    outline_oversized!(mod; force_inline_all)
    verify_ir!("outline_oversized")
    checkpoint("outline_oversized!")

    # ── Fix barrier-skipping error paths ──
    # replace_unreachable! (pre-inlining) converts error paths to early returns.
    # After inlining, these returns may skip barriers that other invocations reach,
    # causing undefined behavior (deadlock on CPU/software Vulkan implementations).
    # Redirect barrier-skipping paths to the barrier-containing continuation.
    fix_barrier_skipping_paths!(entry_fn)
    verify_ir!("barrier_fix")
    checkpoint("fix_barrier_skipping_paths!")

    # ── Post-inlining optimization ──
    # One round is enough: SROA fully promotes `@private` scratchpads (no allocas
    # survive into lava_ir_2_postsroa.ll). Adding rounds was measured a net loss —
    # the model went 20.6 -> 18.6 steps/s, because extra InstCombine pessimises
    # more than extra SROA recovers.
    LLVM.run!(LLVM.InstCombinePass(), mod)
    LLVM.run!(LLVM.SROAPass(), mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    checkpoint("InstCombine+SROA+InstCombine")

    # ── Loop optimisation ──
    # Nothing here used to unroll, and it showed up as a flat ~2.5x deficit
    # against the same KernelAbstractions source on CUDA.jl — uniform across
    # shapes, which is the signature of a per-iteration cost rather than a tiling
    # problem. A dependent `muladd` chain with a compile-time trip count measured
    # 4.2 TFLOP/s here against 24.5 on CUDA (the card's fp32 peak is ~26.7): the
    # SPIR-V was emitting a correct `GLSL.std.450 Fma` but wrapping it in an
    # `OpLoopMerge`, one FMA per iteration behind a compare and a branch.
    #
    # Order matters: rotate into do-while form and canonicalise the induction
    # variable first, or the unroller cannot see the trip count. This runs before
    # structurization, which needs to see the final CFG.
    unroll_loops!(mod)
    verify_ir!("loop_unroll")

    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write(lava_debug_path("lava_ir_2_postsroa.ll"), string(mod))
    end

    # ── Fix inttoptr address spaces after SROA ──
    # SROA eliminates allocas and creates `inttoptr i64 %bda_val to ptr` (addrspace 0)
    # for BDA pointer fields. These should be addrspace 1 (PhysicalStorageBuffer)
    # for correct SPIR-V emission. Convert them and update all downstream uses.
    fix_inttoptr_addrspace!(mod)
    # Note: IR is temporarily invalid here — addrspace(1) inttoptr results are used by
    # GEPs that still reference the original addrspace(0) type. The SPIR-V emitter
    # handles this correctly, and subsequent passes don't depend on address space
    # consistency in GEP source types. Skipping verify here.

    # ── Remove Julia runtime artifacts from inlined error paths ──
    # After force-inlining, error/boxing helpers may reference Julia runtime:
    # - `load i64, ptr @jl_int64_type` (type tags for boxing)
    # - `store i64, ptr inttoptr(1)` (GC tag slot writes)
    # These are dead error paths that will never execute on GPU. Remove them
    # so the SPIR-V emitter doesn't need to handle runtime declarations.
    remove_julia_runtime_artifacts!(mod)
    # After fix_inttoptr_addrspace!, IR has addrspace mismatches in GEPs (addrspace(1)
    # pointers used by addrspace(0) typed GEPs). This is handled correctly by the SPIR-V
    # emitter but makes LLVM.verify fail. Disable per-pass verification for the rest.
    # The final SPIR-V output is validated by spirv-val instead.
    verify_passes = false

    # ── Lower LLVM intrinsics unsupported by SPIR-V ──
    # memcpy → typed loads/stores, lifetime markers → removed
    lower_unsupported_intrinsics!(mod)
    verify_ir!("lower_intrinsics")
    checkpoint("addrspace+runtime+lower_intrinsics")

    # ── Fix GEPs with mismatched source types on allocas ──
    # After SROA + inlining, some GEPs reference the original full tuple type
    # through a smaller alloca pointer. Convert these to byte-offset GEPs so the
    # lift_byte_geps pass can properly convert them using the alloca's type.
    fix_gep_alloca_type_mismatches!(mod)
    verify_ir!("fix_gep_alloca_types")

    # ── Flatten chained GEPs on allocas ──
    # Pattern: gep i8 alloca -4 → gep i32 result %idx  →  gep i8 alloca (-4 + idx*4)
    # This handles Julia's 1-based MArray indexing where the base is shifted.
    flatten_chained_geps_on_allocas!(mod)
    verify_ir!("flatten_chained_geps")

    # ── Lift byte-offset GEPs to typed GEPs ──
    # Julia accesses struct fields via `getelementptr i8, ptr %p, i64 <offset>`.
    # SPIR-V needs typed GEPs into the struct. Run 3x as later passes may create more.
    lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    lift_byte_geps_on_allocas!(mod)
    verify_ir!("lift_byte_geps")
    checkpoint("fix_gep+flatten+lift_byte_geps x3")

    # ── Combine consecutive same-type GEPs ──
    # Patterns like `gep T, (gep T, p, i), j` → `gep T, p, add(i, j)`.
    # This avoids chained OpPtrAccessChain which some drivers handle incorrectly.
    combine_chained_geps!(mod)
    verify_ir!("combine_geps")
    checkpoint("combine_chained_geps! #1")

    # ── Retype uniformly-typed Function allocas ──
    # Run BEFORE structurize_cfg, while the IR is still simple — structurize
    # inserts pointer PHIs for cross-block dataflow that the conservative
    # version of this pass bails on.  Retyping here aligns the alloca's
    # storage type with its uniform access pattern (e.g. MVector{N, Vec3f}
    # alloca [N x i64] → [3N x float]), eliminating the per-access type-pun
    # fixups the emitter would otherwise need.
    retype_uniform_typed_allocas!(mod, LLVM.datalayout(mod))
    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write(lava_debug_path("lava_ir_2_post_retype.ll"), string(mod))
    end
    verify_ir!("retype_allocas")
    checkpoint("retype_uniform_typed_allocas!")

    # ── Structured control flow ──
    # SPIR-V requires structured CF. Run the full structurize pipeline.
    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write(lava_debug_path("lava_ir_3_pre_structurize.ll"), string(mod))
    end
    run_structurize_cfg_pipeline!(mod)
    if get(ENV, "LAVA_DEBUG_PASSES", "") == "1"
        write(lava_debug_path("lava_ir_4_post_structurize.ll"), string(mod))
    end
    verify_ir!("structurize_cfg")
    checkpoint("run_structurize_cfg_pipeline!")

    # ── Flatten nested workgroup array globals ──
    # Replace [32 x [2 x float]] → [64 x float] in addrspace(3).
    # Without VK_KHR_workgroup_memory_explicit_layout, SPIR-V Workgroup arrays
    # cannot have ArrayStride decorations, and NVIDIA miscomputes the stride
    # for nested arrays. Flattening to scalar arrays avoids this.
    flatten_nested_workgroup_arrays!(mod)
    verify_ir!("flatten_wg_arrays")

    # ── Lift byte-offset GEPs on workgroup globals ──
    # Convert `gep i8, @shared, <offset>` ConstantExpr to typed struct-member GEPs.
    # Must run before decompose passes so the emitter sees proper typed access patterns.
    dl = LLVM.datalayout(mod)
    # ── Decompose workgroup typepun copies ──
    # LLVM may optimize shared memory struct copies (shared[i] = shared[j]) into raw
    # integer block copies. Detect and replace with per-field typed copies.
    decompose_workgroup_typepun_copies!(mod, dl)

    # ── Decompose composite workgroup accesses ──
    # Struct loads/stores on addrspace(3) must be decomposed into scalar ops
    # because shared memory is flattened to scalar arrays in SPIR-V.
    decompose_composite_workgroup_accesses!(mod, dl)
    verify_ir!("decompose_wg_accesses_1")

    # ── Decompose type-punned alloca loads ──
    # LLVM memcpy lowering may create `load i64, ptr %alloca_of_struct` which
    # reads raw bytes from a padded struct. For workgroup stores, decompose
    # the entire copy into per-field struct stores. For other uses, replace
    # with first-field load + zext.
    decompose_typepun_alloca_loads!(mod, dl)
    verify_ir!("decompose_typepun_alloca")

    # Re-run composite workgroup decomposition: the typepun pass may have created
    # new struct stores to workgroup memory that need field-by-field decomposition.
    decompose_composite_workgroup_accesses!(mod, dl)

    # Decompose type-punned loads through GEPs into struct allocas.
    # LLVM memcpy optimization creates `load i64, ptr %gep_to_float_field` —
    # reads crossing struct field boundaries. Decompose into per-field loads + pack.
    decompose_typepun_gep_loads!(mod, dl)
    verify_ir!("decompose_typepun_gep")
    checkpoint("workgroup+typepun decompose block")

    # Re-combine chained byte-offset GEPs that were left undecomposed above
    # (because the chain contains dynamic indices). E.g.:
    #   gep i8, (gep i8, %alloca, %dynamic), 224  →  gep i8, %alloca, add(%dynamic, 224)
    # The emitter can then decompose the combined dynamic byte offset into proper
    # OpAccessChain indices via decompose_flat_index_for_composite!.
    combine_chained_geps!(mod)

    # LLVM may create loads where the type differs from the alloca type
    # (e.g., `load i16` from `alloca { [2 x i8] }`). SPIR-V requires strict
    # type matching, so rewrite these to byte-by-byte extraction.
    fix_alloca_type_mismatched_loads!(mod)

    # LLVM SROA/memcpy lowering creates `store i32, ptr %alloca_of_[16 x i64]`.
    # SPIR-V requires the stored value type to match the pointer's pointee type.
    # Rewrite these stores to drill into the alloca type via GEP.
    fix_alloca_type_mismatched_stores!(mod, dl)

    # Fix ConstantExpr GEPs on flattened workgroup globals that have negative indices
    # (from Julia's 1-based pointer adjustment). Decompose passes above may create new
    # uses of these CEs that weren't present during the flattening pass.
    fixup_negative_wg_constexprs!(mod)

    # Fold type-punned scalar allocas where constant partial stores reconstruct a value.
    # SROA decomposes e.g. `zero(Float64)` into `store float 0.0` at offset 0 +
    # `store i32 0` at offset 4, then `load double`. Fold to a direct constant.
    fold_typepun_scalar_alloca_constants!(mod, dl)

    # Lower chained mismatched-type GEPs on allocas.
    # Julia's MArray/StaticArray patterns create chains like:
    #   %base = getelementptr i32, ptr %alloca_[16 x i64], i64 -1
    #   %elem = getelementptr i32, ptr %base, i64 %var
    #   store i32 %val, ptr %elem
    # Lower to proper element-level access with runtime index computation.
    lower_chained_mismatched_geps!(mod)

    # ── Convert typed GEPs on mismatched allocas to byte-offset GEPs ──
    # When InstCombine (inside StructurizeCFG) creates typed GEPs like
    #   gep i32, ptr %alloca_[8xi64], i64 %dynamic
    # the emitter can't handle the type mismatch. Convert to byte GEPs:
    #   gep i8, ptr %alloca, i64 (%dynamic * 4)
    # so lower_byte_gep_chain_on_allocas! can apply shift/mask extraction.
    convert_typepunned_geps_to_byte_geps!(mod)

    # ── Lower byte-offset GEP chains on MArray allocas ──
    # After InstCombine splits flattened byte-offset GEPs back into chains:
    #   %gep1 = gep i8, ptr %alloca_[16xi64], %dynamic
    #   %gep2 = gep i8, ptr %gep1, -4
    #   store i32 %val, ptr %gep2
    # Lower to proper element-level access with shift/mask (no integer divide).
    lower_byte_gep_chain_on_allocas!(mod)

    # ── Lower PHI-chained typepunned loads on array allocas ──
    # When byte-offset GEPs flow through PHI chains (from StructurizeCFG) before being
    # loaded, lower_byte_gep_chain_on_allocas! can't handle them (it requires direct
    # GEP→load). This pass "lifts" the load to each leaf GEP site with shift/mask
    # extraction and creates parallel value PHI chains.
    # Critical for MVector{32,UInt32} stored as [16 x i64] in BVH stack traversal.
    lower_phi_typepunned_loads!(mod)
    verify_ir!("lower_phi_typepun")
    checkpoint("combine#2..lower_phi_typepunned_loads!")

    # ── Lower phi/select of Function-storage pointers to value-level ──
    # SROA can't promote allocas whose addresses flow through `phi ptr` /
    # `select ptr` (the canonical case is GPUArrays' findfirstlast_reduction,
    # whose conditional tuple swap leaves a select+phi over three small
    # allocas).  RADV's `nir_lower_vars_to_ssa` and lavapipe's equivalent both
    # bail on the resulting `OpPhi`/`OpSelect %_ptr_Function_*`, leaving a
    # `@store_deref` intrinsic that aborts pipeline creation
    # ("Unimplemented intrinsic instr: @store_deref").  This pass rewrites
    # `load Ty, ptr (phi/select of allocas)` into a parallel value-level
    # phi/select that loads `Ty` at each leaf alloca site, side-stepping the
    # backend SSA promoter altogether.
    lower_phi_select_function_ptrs!(mod)
    verify_ir!("lower_phi_select_func_ptrs")
    checkpoint("lower_phi_select_function_ptrs!")

    # ── Lift byte-offset GEPs on workgroup globals ──
    # The decompose passes above may create byte-offset ConstantExpr GEPs like
    # `gep i8, @shared, <offset>` when splitting struct loads from workgroup globals.
    # Convert these to typed struct-member GEPs so the emitter produces proper OpAccessChain.
    lift_byte_geps_on_workgroup_globals!(mod, dl)

    # ── Final cleanup: fix GEPs with mismatched source types on allocas ──
    # LLVM's SROA/memcpy lowering can create typed GEPs using wrong source types
    # (e.g., `gep [3 x float]` on `alloca [3 x i32]`). Convert these to byte-offset
    # GEPs followed by the lift pass to normalize types.
    fix_gep_alloca_type_mismatches!(mod)
    lift_byte_geps_on_allocas!(mod)

    # The final lift_byte_geps_on_allocas! may convert byte GEPs (gep i8, alloca, <off>)
    # to typed GEPs (gep [2 x double], alloca, 0, 0), but the load users still have the
    # wrong type (e.g., load i32 from a double*). Run the typepun GEP load decomposition
    # one more time to fix these.
    decompose_typepun_gep_loads!(mod, dl)
    verify_ir!("final")
    checkpoint("final lift+decompose block")

    # ── Fix Workgroup (shared memory) load/store alignment ──
    # Julia/GPUCompiler emits `align 1` for all addrspace(3) accesses regardless of
    # element type. Correct to natural type alignment so downstream tools and drivers
    # see accurate alignment metadata.
    fix_workgroup_alignment!(mod, dl)

    # ── Propagate PSB alignment from BDA loads ──
    # GPUCompiler emits `align 1` for all addrspace(1) accesses after SROA splits
    # struct loads into individual field loads. The wrapper sets `align 8` on the
    # initial BDA load, but that doesn't propagate through inttoptr → field load.
    # Walk the IR and propagate alignment using Julia ABI invariants (loaded i64
    # used via inttoptr is 8-aligned, GEPs preserve via gcd).
    propagate_psb_alignment!(mod, dl)
    checkpoint("workgroup_align+propagate_psb_alignment!")

    return nothing
end

"""
Fix call instructions where the call-site return type is `i8` but the callee is
defined to return `i1`. Julia emits these when a `Bool`-returning `llvmcall` stub
is wrapped by a thunk that uses `i8` at ABI boundaries. The LLVM AlwaysInliner
refuses to inline across such a type mismatch, so we fix the call site first:
replace `%v = call i8 @f()` with `%v1 = call i1 @f(); %v = zext i1 %v1 to i8`.
"""
function fix_bool_call_mismatches!(mod::LLVM.Module)
    i1_ty  = LLVM.Int1Type()
    i8_ty  = LLVM.Int8Type()
    to_fix = Tuple{LLVM.CallInst, LLVM.Function}[]

    for fn in LLVM.functions(mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                called = LLVM.called_operand(inst)
                called isa LLVM.Function || continue
                # Check: call site returns i8 but definition returns i1
                LLVM.value_type(inst) == i8_ty || continue
                def_ret = LLVM.return_type(LLVM.function_type(called))
                def_ret == i1_ty || continue
                push!(to_fix, (inst, called))
            end
        end
    end

    for (call_inst, callee) in to_fix
        # Build replacement: call i1 @callee(); zext i1 to i8
        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call_inst)
            fn_ty = LLVM.FunctionType(i1_ty, LLVM.LLVMType[])
            new_call = LLVM.call!(builder, fn_ty, callee)
            zext_val = LLVM.zext!(builder, new_call, i8_ty)
            LLVM.replace_uses!(call_inst, zext_val)
            LLVM.erase!(call_inst)
        end
    end

    return nothing
end

"""
Apply Lava's inline policy to the module.

Default (`force_inline_all=false`): respect Julia/GPUCompiler's inlining
decisions. Only functions Julia explicitly marked `alwaysinline` (typically
compiler-generated throw/box wrappers from `lower_throw!` etc.) get inlined
via the AlwaysInliner pass. Surviving helpers stay as their own functions
and Step 6's emission walker emits each as a separate SPIR-V OpFunction.

Escape hatch (`force_inline_all=true`): the original blanket-inline behavior
— mark every non-entry, non-declaration function `alwaysinline` and collapse
the whole module into a single OpFunction. Kept so any regression can be
A/B compared against the old path with a single flag flip.
"""
function outline_oversized!(mod::LLVM.Module; force_inline_all::Bool=false)
    # When force_inline_all is on we want everything in a single OpFunction
    # by design; outlining would defeat that. Also a useful escape hatch when
    # debugging whether outlining caused a regression.
    force_inline_all && return nothing

    # OPT-IN. Enable via env var:
    #   LAVA_OUTLINE_ENABLED=1
    # Outlining is structurally correct (Step 6 walker emits the helpers,
    # FuncControl.DontInline keeps spirv-opt from collapsing them), but the
    # PTM infers different pointee types for the helper's parameter (from
    # uses inside the helper) vs the argument at the call site (from where
    # the value was created). Without correct cross-boundary type coercion,
    # this produces SPIR-V that fails spirv-val for any moderately-sized
    # kernel where LoopExtractor or IROutliner finds something to extract.
    # Leaving this off until the type coercion is solved keeps medium-sized
    # Hikari/Makie kernels (which compiled fine as single OpFunctions) from
    # regressing into validation failures.
    get(ENV, "LAVA_OUTLINE_ENABLED", "0") == "1" || return nothing

    # AMDVLK chokes somewhere around 100-150 BBs in a single OpFunction. 50
    # leaves comfortable margin and matches the threshold the spirv_bisect
    # work used to characterize the chokepoint. Override via env var when
    # tuning for a specific kernel.
    threshold = parse(Int, get(ENV, "LAVA_OUTLINE_BB_THRESHOLD", "50"))

    function max_bbs_per_function(mod)
        m = 0
        for fn in LLVM.functions(mod)
            isempty(LLVM.blocks(fn)) && continue
            n = count(_ -> true, LLVM.blocks(fn))
            m = max(m, n)
        end
        m
    end

    # Try multiple LLVM outlining passes in succession. Each pass is iterated
    # to a fixed point because LLVM's outliners typically invalidate analyses
    # after one extraction and bail rather than recomputing.
    #
    # Pass selection:
    #   - LoopExtractorPass: extracts top-level loops (works well for loop-heavy
    #     kernels). vp_trace_rays-class kernels are queue-dispatch dominated so
    #     this only picks off a couple of small loops.
    #   - IROutlinerPass: finds syntactically-similar code sequences and
    #     factors them into a shared function (LLVM's `-Oz` optimizer uses this).
    #     Useful for kernels with repeated patterns (queue push/pop, etc.).
    function iterate_pass!(pass_ctor)
        for _ in 1:8
            max_bbs_per_function(mod) <= threshold && return true
            before_n = count(_ -> true, LLVM.functions(mod))
            LLVM.run!(pass_ctor(), mod)
            after_n = count(_ -> true, LLVM.functions(mod))
            after_n == before_n && return false
        end
        return false
    end

    iterate_pass!(LLVM.LoopExtractorPass)
    iterate_pass!(LLVM.IROutlinerPass)

    return nothing
end

function collect_reachable_callees(entry_fn::LLVM.Function)
    visited = Set{LLVM.Function}()
    order = LLVM.Function[]
    function visit(fn::LLVM.Function)
        fn in visited && return
        push!(visited, fn)
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = LLVM.called_operand(inst)
            callee isa LLVM.Function || continue
            isempty(LLVM.blocks(callee)) && continue
            startswith(LLVM.name(callee), "llvm.") && continue
            visit(callee)
        end
        push!(order, fn)  # post-order: callees pushed before caller
    end
    visit(entry_fn)
    return order
end

function strongly_connected_components(fns::Vector{LLVM.Function})
    indices = Dict{LLVM.Function, Int}()
    lowlinks = Dict{LLVM.Function, Int}()
    on_stack = Set{LLVM.Function}()
    stack = LLVM.Function[]
    sccs = Vector{LLVM.Function}[]
    next_index = Ref(0)
    fnset = Set(fns)

    function strongconnect(v::LLVM.Function)
        indices[v] = next_index[]
        lowlinks[v] = next_index[]
        next_index[] += 1
        push!(stack, v)
        push!(on_stack, v)
        for bb in LLVM.blocks(v), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            w = LLVM.called_operand(inst)
            w isa LLVM.Function || continue
            w in fnset || continue
            if !haskey(indices, w)
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elseif w in on_stack
                lowlinks[v] = min(lowlinks[v], indices[w])
            end
        end
        if lowlinks[v] == indices[v]
            scc = LLVM.Function[]
            while true
                w = pop!(stack)
                delete!(on_stack, w)
                push!(scc, w)
                w === v && break
            end
            push!(sccs, scc)
        end
    end

    for v in fns
        haskey(indices, v) || strongconnect(v)
    end
    return sccs
end

"""
Trace a pointer-typed LLVM value back to its source alloca. Returns the
alloca's allocated type, or `nothing` if the source isn't a unique alloca.
Walks through GEPs, bitcasts, addrspacecasts, and PHIs (PHIs return
`nothing` if their inputs disagree — caller treats that as "uncertain
type" and force-inlines).
"""
function trace_alloca_allocated_type(value::LLVM.Value,
                                      visited::Set{LLVM.Value}=Set{LLVM.Value}())
    value in visited && return nothing
    push!(visited, value)
    if value isa LLVM.AllocaInst
        return LLVM.LLVMType(API.LLVMGetAllocatedType(value))
    elseif value isa LLVM.GetElementPtrInst
        return trace_alloca_allocated_type(LLVM.operands(value)[1], visited)
    elseif value isa LLVM.BitCastInst || value isa LLVM.AddrSpaceCastInst
        return trace_alloca_allocated_type(LLVM.operands(value)[1], visited)
    elseif value isa LLVM.PHIInst
        incoming_types = LLVM.LLVMType[]
        for (val, _) in LLVM.incoming(value)
            t = trace_alloca_allocated_type(val, visited)
            t === nothing && return nothing
            push!(incoming_types, t)
        end
        isempty(incoming_types) && return nothing
        all(t -> t == incoming_types[1], incoming_types) || return nothing
        return incoming_types[1]
    else
        return nothing
    end
end

"""
Collect the set of "pointee shapes" the helper's body uses for a pointer
parameter. For a `getelementptr T, ptr %p, ...`, the helper's PTM
inference picks T as the pointee. For a direct `load T, ptr %p`, the
pointee is T. These types must match the caller's alloca type exactly,
because SPIR-V's Logical addressing requires OpFunctionCall argument
types to match the parameter type — no implicit coercion.
"""
function helper_gep_element_types(fn::LLVM.Function, param::LLVM.Argument)
    types = Set{LLVM.LLVMType}()
    for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        if inst isa LLVM.GetElementPtrInst
            ops = LLVM.operands(inst)
            ops[1] === param || continue
            src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(inst))
            push!(types, src_ty)
        elseif inst isa LLVM.LoadInst || inst isa LLVM.StoreInst
            ptr_op = inst isa LLVM.LoadInst ? LLVM.operands(inst)[1] : LLVM.operands(inst)[2]
            ptr_op === param || continue
            loaded_ty = inst isa LLVM.LoadInst ? LLVM.value_type(inst) :
                                                  LLVM.value_type(LLVM.operands(inst)[1])
            push!(types, loaded_ty)
        end
    end
    return types
end

"""
True when `fn` has at least one pointer parameter whose helper-side usage
indicates a type-pun across the function boundary. A type-pun is detected
when:

  * The helper's body accesses the pointer with a scalar element type T,
    AND
  * At least one caller passes an alloca whose innermost element type ≠ T,
    OR multiple callers pass allocas with mismatching element types.

Type-puns can't be expressed across an OpFunctionCall in SPIR-V Logical
addressing. The fix is to force-inline the helper so the type-pun stays
inside one OpFunction, where Lava's PTM resolves the pointer's type
context-locally per access.
"""
function has_alloca_type_mismatch_at_callsites(fn::LLVM.Function, mod::LLVM.Module)
    params = collect(LLVM.parameters(fn))
    ptr_params = [(i, p) for (i, p) in enumerate(params) if LLVM.value_type(p) isa LLVM.PointerType]
    isempty(ptr_params) && return false

    # For each pointer param, find the helper's expected element type(s).
    helper_types_per_param = Dict{Int, Set{LLVM.LLVMType}}()
    for (i, p) in ptr_params
        helper_types_per_param[i] = helper_gep_element_types(fn, p)
    end

    # Collect caller alloca types per param index — keeping the EXACT
    # allocated type (no array unwrapping), since SPIR-V requires exact
    # match at OpFunctionCall.
    caller_types_per_param = Dict{Int, Set{LLVM.LLVMType}}()
    for (i, _) in ptr_params
        caller_types_per_param[i] = Set{LLVM.LLVMType}()
    end
    has_any_callsite = false
    for caller in LLVM.functions(mod)
        isempty(LLVM.blocks(caller)) && continue
        for bb in LLVM.blocks(caller), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = LLVM.called_operand(inst)
            callee === fn || continue
            has_any_callsite = true
            ops = LLVM.operands(inst)
            n_args = length(ops) - 1
            for (i, _) in ptr_params
                i <= n_args || continue
                t = trace_alloca_allocated_type(ops[i])
                t === nothing && continue
                push!(caller_types_per_param[i], t)
            end
        end
    end
    has_any_callsite || return false

    # Mismatch if (a) callers disagree on the alloca's type for this param,
    # or (b) the alloca's type doesn't match any pointee shape the helper
    # uses internally — both are unrecoverable in Logical-addressing
    # OpFunctionCall.
    for (i, _) in ptr_params
        caller_set = caller_types_per_param[i]
        helper_set = helper_types_per_param[i]
        length(caller_set) > 1 && return true
        isempty(helper_set) && continue
        for ct in caller_set
            ct in helper_set || return true
        end
    end
    return false
end

"""
Debug knob: substrings of entry-function names that should be compiled
with the `force_inline_all=true` escape hatch. When the entry_fn's name
contains any of these substrings, all helpers get inlined into it,
regardless of @noinline. Used to bisect multi-OpFunction miscompiles.

Set from outside Lava via e.g.
    push!(Lava.FORCE_INLINE_KERNEL_PATTERNS, "vp_trace_rays_kernel")
"""
const FORCE_INLINE_KERNEL_PATTERNS = String[]

function force_inline_all!(mod::LLVM.Module, entry_fn::LLVM.Function;
                            force_inline_all::Bool=false)
    entry_name = LLVM.name(entry_fn)
    # Promote to force_inline_all if entry_fn matches any debug pattern
    if !force_inline_all && !isempty(FORCE_INLINE_KERNEL_PATTERNS)
        if any(p -> occursin(p, entry_name), FORCE_INLINE_KERNEL_PATTERNS)
            force_inline_all = true
        end
    end

    if force_inline_all
        for fn in LLVM.functions(mod)
            fn_name = LLVM.name(fn)
            fn === entry_fn && continue
            isempty(LLVM.blocks(fn)) && continue
            startswith(fn_name, "llvm.") && continue
            attrs = LLVM.function_attributes(fn)
            delete!(attrs, LLVM.EnumAttribute("noinline"))
            push!(attrs, LLVM.EnumAttribute("alwaysinline"))
        end
    end

    # Fix i1/i8 call-site mismatches before running the inliner so that
    # Bool-returning llvmcall thunks (e.g. lava_ray_query_proceed wrappers)
    # are eligible for inlining.
    fix_bool_call_mismatches!(mod)

    # When force_inline_all=false (the default), let GPUCompiler-marked
    # helpers survive as separate SPIR-V OpFunctions so AMDVLK doesn't choke
    # on oversized single-OpFunction kernels. Strip `alwaysinline` from every
    # eligible function, EXCEPT:
    #   (a) compiler-generated junk (throw/box/safepoint helpers) — these
    #       reference symbols Lava's SPIR-V emitter can't handle outside the
    #       entry function and Julia counts on them being inlined away.
    #   (b) the wrapper's DIRECT callees (the original kernel functions called
    #       from the BDA entry wrapper) — `emit_entry_wrapper!` passes OpUndef
    #       for args; if the kernel survives as a separate function, spirv-opt
    #       sees writes-to-undef-pointers and DCEs the entire computation.
    # Indirect helpers (callees-of-callees) survive, providing the
    # multi-OpFunction split that satisfies AMDVLK's per-OpFunction limit.
    if !force_inline_all
        must_inline_prefixes = ("julia_throw", "julia_box", "jl_box",
                                "gpu_report_exception", "gpu_signal_exception",
                                "julia.gc_", "julia.safepoint", "ijl_throw",
                                "jl_throw")

        wrapper_direct_callees = Set{String}()
        for bb in LLVM.blocks(entry_fn), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = LLVM.called_operand(inst)
            callee isa LLVM.Function || continue
            isempty(LLVM.blocks(callee)) && continue
            push!(wrapper_direct_callees, LLVM.name(callee))
        end

        for fn in LLVM.functions(mod)
            isempty(LLVM.blocks(fn)) && continue
            startswith(LLVM.name(fn), "llvm.") && continue
            fname = LLVM.name(fn)
            any(p -> occursin(p, fname), must_inline_prefixes) && continue
            fname in wrapper_direct_callees && continue
            attrs = LLVM.function_attributes(fn)
            delete!(attrs, LLVM.EnumAttribute("alwaysinline"))
        end

        # Structural rule (runs AFTER the strip pass above so it isn't undone):
        # any function whose RETURN type is a pointer must be inlined into its
        # callers.  SPIR-V Logical addressing has no concept of a contextless
        # pointer crossing an OpFunction boundary — `OpTypePointer` always
        # carries a `StorageClass`, and the storage class is determined by
        # the call site, not the callee.  GPUCompiler/Julia inject several
        # such helpers (`gpu_malloc`, `ijl_box_*` internal allocators, etc.)
        # as part of allocation/exception paths.  If they survive non-inlined,
        # `emit_function!` aborts with "function X has pointer return type"
        # because the emitter has no way to pick a storage class.  Re-mark
        # `alwaysinline` (and strip `noinline`) so the AlwaysInlinerPass below
        # eats them and DCE removes the surrounding error path entirely.
        for fn in LLVM.functions(mod)
            isempty(LLVM.blocks(fn)) && continue
            startswith(LLVM.name(fn), "llvm.") && continue
            fn === entry_fn && continue
            fn_ty = LLVM.function_type(fn)
            LLVM.return_type(fn_ty) isa LLVM.PointerType || continue
            attrs = LLVM.function_attributes(fn)
            delete!(attrs, LLVM.EnumAttribute("noinline"))
            push!(attrs, LLVM.EnumAttribute("alwaysinline"))
        end

        # Detect cross-boundary pointer-type mismatch. When a noinline helper
        # has a pointer param, and callers pass pointers to allocas of
        # DIFFERENT LLVM types (e.g. one caller passes [21 x i64], another
        # passes [5 x i64]), Lava's PTM picks one pointee type for the param
        # and the OpFunctionCall mismatches at the other site. SPIR-V can't
        # express this kind of type-punning across a function boundary in
        # Logical addressing. Re-mark such helpers `alwaysinline` so the
        # type-pun stays inside one OpFunction where Lava's PTM resolves it
        # context-locally.
        for fn in LLVM.functions(mod)
            isempty(LLVM.blocks(fn)) && continue
            startswith(LLVM.name(fn), "llvm.") && continue
            fname = LLVM.name(fn)
            fn === entry_fn && continue
            any(p -> occursin(p, fname), must_inline_prefixes) && continue
            # NOTE: do NOT skip wrapper_direct_callees here. Even kernels called
            # directly from the wrapper must be force-inlined when their callers
            # disagree on pointer-arg alloca types — the SPIR-V type mismatch is
            # a hard validation error.

            has_ptr_param = any(p -> LLVM.value_type(p) isa LLVM.PointerType,
                                LLVM.parameters(fn))
            has_ptr_param || continue

            if has_alloca_type_mismatch_at_callsites(fn, mod)
                attrs = LLVM.function_attributes(fn)
                delete!(attrs, LLVM.EnumAttribute("noinline"))
                push!(attrs, LLVM.EnumAttribute("alwaysinline"))
            end
        end
    end

    # Force-inline functions whose ray-query state would otherwise cross a
    # SPIR-V function boundary.
    #
    # A ray query is a Function-scope `OpVariable %ray_query OpTypeRayQueryKHR`:
    # `OpRayQueryInitializeKHR` allocates the implicit per-function variable
    # and all subsequent `OpRayQueryProceedKHR` / `OpRayQueryGet*KHR` /
    # `OpRayQueryConfirm*KHR` / `OpRayQueryTerminateKHR` calls in the SAME
    # SPIR-V function operate on that same variable.
    #
    # The original Lava rule was "any function with any `lava_ray_query_*`
    # call gets `alwaysinline`". That is correct but overly conservative: it
    # force-inlines self-contained helpers (e.g. `trace_shadow_transmittance`
    # which runs a complete init+proceed+get+terminate session every call)
    # into their callers, ballooning the caller's register pressure for no
    # SPIR-V-correctness reason — the helper would get its own implicit
    # ray-query OpVariable if left alone, which is what it wants.
    #
    # Correctness rule (refined): a SPIR-V function's ray-query state must be
    # SELF-CONTAINED. A function F is self-contained when EITHER
    #   (a) F has zero `lava_ray_query_*` calls, OR
    #   (b) F has BOTH a `lava_ray_query_init` AND at least one other
    #       `lava_ray_query_*` call (proceed/get/confirm/terminate).
    # Case (b) covers the common shadow-trace pattern where a helper runs the
    # entire ray-query lifecycle internally and exposes only ordinary scalar
    # / vector returns to its caller.
    #
    # The case the old rule was guarding against — Julia hoisting a
    # `while lava_ray_query_proceed() ... end` loop body into a separate
    # function — is detected: the hoisted helper has proceed/get calls but no
    # init, so it falls into the INCOMPLETE bucket and gets `alwaysinline`'d
    # back into its caller. Meanwhile, helpers whose lifecycle is closed
    # (init AND proceed) survive as separate OpFunctions and get their own
    # register frame.
    rayquery_init_callers = Set{LLVM.Function}()
    rayquery_other_callers = Set{LLVM.Function}()
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = LLVM.called_operand(inst)
            callee isa LLVM.Function || continue
            cname = LLVM.name(callee)
            startswith(cname, "lava_ray_query_") || continue
            if cname == "lava_ray_query_init"
                push!(rayquery_init_callers, fn)
            else
                push!(rayquery_other_callers, fn)
            end
        end
    end
    # symmetric difference = functions with init xor (proceed/get/...).
    # Either side alone is an incomplete lifecycle; force-inline so the
    # surviving entry-side function has a complete one.
    incomplete_rayquery = symdiff(rayquery_init_callers, rayquery_other_callers)
    for fn in incomplete_rayquery
        fn === entry_fn && continue
        attrs = LLVM.function_attributes(fn)
        delete!(attrs, LLVM.EnumAttribute("noinline"))
        push!(attrs, LLVM.EnumAttribute("alwaysinline"))
    end

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
function emit_spirv_from_llvm(llvm_mod::LLVM.Module, entry_name::String,
                                workgroup_size::NTuple{3,Int};
                                enable_ray_query::Bool=false)
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
    if enable_ray_query
        setup_ray_query_capabilities!(spirv_mod)
    end

    # Build struct pointer member type map (resolves ptr members in structs)
    build_struct_ptr_member_types!(type_ctx, llvm_mod)

    # Pre-collect all types used in the module
    collect_module_types!(type_ctx, llvm_mod)

    # Create emitter state
    state = SPIRVEmitterState(spirv_mod, type_ctx)
    state.data_layout = LLVM.datalayout(llvm_mod)

    # Find the entry function
    entry_fn = LLVM.functions(llvm_mod)[entry_name]

    # Pre-allocate SPIR-V function IDs for every function with a body so that
    # OpFunctionCall can forward-reference callees regardless of emission order.
    # No effect today (single-function case after force_inline_all!), but enables
    # the multi-function emission walker in Step 6 and avoids needing a strict
    # topological sort. SPIR-V allows forward references to function IDs.
    for fn in LLVM.functions(llvm_mod)
        isempty(LLVM.blocks(fn)) && continue
        get!(state.value_map, fn) do
            fresh_id!(spirv_mod)
        end
    end

    collect_perelement_callbacks!(state, llvm_mod)

    # Emit global variables (if any — needed for builtin inputs, etc.)
    interface_ids = emit_globals!(state, llvm_mod)

    # For ray-query compute kernels: emit the TLAS descriptor variable and
    # merge its id into the interface list for OpEntryPoint.
    if enable_ray_query
        emit_compute_tlas_descriptor!(state)
        append!(interface_ids, state.entry_interface_ids)
    end

    # Multi-OpFunction emission: walk the call graph from the entry function and
    # emit every reachable helper as its own OpFunction BEFORE the entry. The
    # function IDs were pre-allocated above so OpFunctionCall to forward
    # references is well-defined; emission order here is for source-map locality.
    # Surviving helpers exist only when force_inline_all=false (the new default);
    # under the old escape hatch only the entry survives and this loop is a no-op.
    let reachable = collect_reachable_callees(entry_fn),
        sccs = strongly_connected_components(reachable)
        for scc in sccs
            if length(scc) > 1
                names = join(LLVM.name.(scc), " -> ")
                error("Mutual recursion is not supported in SPIR-V multi-OpFunction emission: cycle through $names")
            end
        end
        for fn in reachable
            fn === entry_fn && continue
            isempty(LLVM.blocks(fn)) && continue
            emit_function!(state, fn; is_entry=false)
        end
    end

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
        func_id = emit_entry_wrapper!(state, entry_fn)
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
function emit_entry_wrapper!(state::SPIRVEmitterState, entry_fn::LLVM.Function)
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
function emit_globals!(state::SPIRVEmitterState, llvm_mod::LLVM.Module)
    interface_ids = UInt32[]
    wg_globals = LLVM.GlobalVariable[]

    for gv in LLVM.globals(llvm_mod)
        gv_ty = LLVM.value_type(gv)
        gv_ty isa LLVM.PointerType || continue
        as = LLVM.addrspace(gv_ty)

        if as == 2
            # Push constant global (addrspace 2)
            var_id = emit_push_constant_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 3
            # Workgroup/shared memory global (addrspace 3) — collected and emitted
            # as members of ONE combined Block struct after the loop (see below).
            push!(wg_globals, gv)
        elseif as == 7
            # Input global (addrspace 7) — SPIR-V builtins
            var_id = emit_builtin_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 1
            # Julia constant global (addrspace 1) — lookup tables like _j_const_N
            var_id = emit_constant_global!(state, gv)
            if var_id !== nothing
                push!(interface_ids, var_id)  # SPIR-V 1.4+ requires all globals in interface
            end
        end
    end

    # Workgroup (@localmem) globals: one plain variable each where their types allow
    # it, otherwise all of them as members of a SINGLE Block struct — separate
    # *Block-decorated* Workgroup variables must each carry `Aliased`
    # (SPV_KHR_workgroup_memory_explicit_layout spec rule) and are then permitted to
    # overlap the same storage, which lavapipe does. See the note on `emit_workgroup_globals!`.
    #
    # Every variable goes in the interface list, not just the first: SPIR-V 1.4+
    # requires all globals an entry point uses to be listed, and a plain path with
    # two `@localmem` buffers otherwise fails validation on the second.
    if !isempty(wg_globals)
        append!(interface_ids, emit_workgroup_globals!(state, wg_globals))
    end

    return interface_ids
end

"""
Emit a push constant global variable in SPIR-V.
Creates the struct type with Block decoration, pointer type, and OpVariable.
"""
function emit_push_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
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
            offset += UInt32(compute_type_size(member_ty, state.data_layout))
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
    emit_workgroup_globals!(state, wg_globals) -> Vector{UInt32}

Emit every `@localmem` global, and return the ids for the entry point's interface.

# The plain glslang form: tried, and it does not work here

glslang emits one *undecorated* `Workgroup` `OpVariable` per `shared` array, where
Lava puts them all in one `Block`-decorated struct under
`SPV_KHR_workgroup_memory_explicit_layout`. That difference was the leading
suspect for the store-dropping bug in `test_shared_index_division.jl`, so the
plain form was implemented — `wg_needs_explicit_layout(ty)` on the array types,
plain variables when nothing needed a spelled-out layout, and a `WG_VAR_UNWRAPPED`
marker so the preamble used the variable directly instead of drilling into a
struct member. **It changed nothing** — the cause was `OpUDiv`, not the layout.

It also does not validate, and the reason is worth keeping. Explicit-layout
decorations are *invalid* on a plain `Workgroup` variable's type, so the plain
path needs a second, undecorated SPIR-V type id for the same LLVM type — and
every other lookup for that type still resolves to the decorated one. The two
meet, and `spirv-val` says so:

    OpSelect %_ptr_Workgroup__arr_float_uint_128_0 ... %lava_shared___static_shmem
      error: Expected both objects to be of Result Type
    OpAccessChain %_ptr_Workgroup__arr_double_uint_3_1 ...
      error: result type does not match the type that results from indexing the base

Doing it properly means making explicit-layout a **module-wide** decision that
every `map_workgroup_type!` in that module honours, not a per-call-site one.
That is a real change with, on the evidence above, no benefit — so this stayed
with the Block struct and the note stayed here.
"""
function emit_workgroup_globals!(state::SPIRVEmitterState,
                                 wg_globals::Vector{<:LLVM.GlobalVariable})
    member_lly = LLVM.LLVMType[LLVM.global_value_type(gv) for gv in wg_globals]
    return UInt32[emit_combined_workgroup_block!(state, wg_globals, member_lly)]
end

"""
Emit ALL workgroup (shared memory / @localmem) globals as members of a single
Block-decorated struct backed by ONE Workgroup OpVariable.

Why combined rather than one variable per global: under
`SPV_KHR_workgroup_memory_explicit_layout`, the spec requires that if more than
one Workgroup variable points to a Block-decorated type, *all* of them be
decorated `Aliased` — and `Aliased` then permits them to share storage.
Conformant software rasterizers (lavapipe) take that permission and overlap the
variables, so a kernel with two `@localmem` buffers has its second buffer alias
the first (silent corruption; RADV happened to keep them separate, masking the
bug). The rule is about *Block-decorated* types, so it does not reach the plain
variables `emit_plain_workgroup_vars!` emits.

A single Block struct with one member per `@localmem`, each at a distinct
explicit Offset, gives every buffer its own storage with no `Aliased`. The
function preamble drills to member `i` (instead of member 0) for the i-th global.
Fresh workgroup type IDs (`map_workgroup_type!`) keep these types separate from
the PSB-decorated main cache.
"""
function emit_combined_workgroup_block!(state::SPIRVEmitterState,
                                        wg_globals::Vector{<:LLVM.GlobalVariable},
                                        member_lly::Vector{LLVM.LLVMType})
    mod = state.mod

    member_spirv = UInt32[map_workgroup_type!(state.type_ctx, ty) for ty in member_lly]

    # OpTypeStruct with one member per @localmem buffer.
    block_id = fresh_id!(mod)
    word_count = UInt32(2 + length(member_spirv))
    push!(mod.types_constants, (word_count << 16) | UInt32(Op.OpTypeStruct))
    push!(mod.types_constants, block_id)
    append!(mod.types_constants, member_spirv)
    emit_decorate!(mod, block_id, Dec.Block)

    # Member offsets: running offset, each member aligned to its natural alignment,
    # so members occupy disjoint storage.
    running_offset = UInt32(0)
    for (i, ty) in enumerate(member_lly)
        member_align = UInt32(wg_compute_type_alignment(ty))
        running_offset = (running_offset + member_align - UInt32(1)) & ~(member_align - UInt32(1))
        emit_member_decorate!(mod, block_id, UInt32(i - 1), Dec.Offset, running_offset)
        running_offset += UInt32(wg_compute_type_size(ty))
    end

    ptr_ty = map_pointer_type!(state.type_ctx, block_id, SC.Workgroup)

    require_capability!(mod, Cap.WorkgroupMemoryExplicitLayoutKHR)
    require_extension!(mod, "SPV_KHR_workgroup_memory_explicit_layout")
    for ty in member_lly
        wg_type_contains_width(ty, 8)  && require_capability!(mod, Cap.WorkgroupMemoryExplicitLayout8BitAccessKHR)
        wg_type_contains_width(ty, 16) && require_capability!(mod, Cap.WorkgroupMemoryExplicitLayout16BitAccessKHR)
    end

    var_id = fresh_id!(mod)
    # Zero-initialize the shared Block.  Vulkan leaves Workgroup memory undefined
    # at the start of a dispatch, so a kernel that reads a slot it did not write
    # first sees garbage — which is exactly why AK's block-level merge produced
    # wrong results for `len < 2 * block_size` and had to be worked around.
    # An OpConstantNull initializer is the spec-sanctioned fix and applies to
    # every kernel, rather than each caller having to pre-fill its scratch.
    # Requires shaderZeroInitializeWorkgroupMemory (Vulkan 1.3 core), which
    # `init_vulkan!` already enables.
    null_init = emit_constant_null!(mod, block_id)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Workgroup, null_init)
    # Single Block Workgroup variable → no `Aliased` required; members are disjoint.

    for (i, gv) in enumerate(wg_globals)
        # (var, inner_type_spirv, inner_llvm_type, member_index) — preamble drills to member i.
        state.wg_wrapped_vars[gv] = (var_id, member_spirv[i], member_lly[i], UInt32(i - 1))
        set_pointee_type!(state.type_ctx.ptm, gv, member_lly[i]; priority=5)
        gv_name = LLVM.name(gv)
        isempty(gv_name) || emit_name!(mod, var_id, gv_name)
    end

    return var_id
end

"""Check if an LLVM type contains an integer element of the given bit width."""
function wg_type_contains_width(ty::LLVM.LLVMType, bits::Int)
    if ty isa LLVM.IntegerType
        return LLVM.width(ty) == bits
    elseif ty isa LLVM.StructType
        return any(mt -> wg_type_contains_width(mt, bits), LLVM.elements(ty))
    elseif ty isa LLVM.ArrayType
        return wg_type_contains_width(LLVM.eltype(ty), bits)
    else
        return false
    end
end

"""Check if an LLVM type contains a struct at any nesting level."""
function wg_type_contains_struct(ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        return true
    elseif ty isa LLVM.ArrayType
        return wg_type_contains_struct(eltype(ty))
    else
        return false
    end
end

# ── Builtin name → SPIR-V BuiltIn decoration mapping ──
const SPIRV_BUILTIN_MAP = Dict{String, UInt32}(
    "__spirv_BuiltInGlobalInvocationId"   => BuiltIn.GlobalInvocationId,
    "__spirv_BuiltInLocalInvocationId"    => BuiltIn.LocalInvocationId,
    "__spirv_BuiltInWorkgroupId"          => BuiltIn.WorkgroupId,
    "__spirv_BuiltInNumWorkgroups"        => BuiltIn.NumWorkgroups,
    "__spirv_BuiltInWorkgroupSize"        => BuiltIn.WorkgroupSize,
    "__spirv_BuiltInLocalInvocationIndex" => BuiltIn.LocalInvocationIndex,
    # Both need Cap.GroupNonUniform, added by `emit_builtin_global!` below.
    "__spirv_BuiltInSubgroupSize"              => BuiltIn.SubgroupSize,
    "__spirv_BuiltInSubgroupLocalInvocationId" => BuiltIn.SubgroupLocalInvocationId,
)

# Builtins that are not Vulkan 1.0 core and carry a capability requirement.
const SPIRV_BUILTIN_CAPABILITY = Dict{String, UInt32}(
    "__spirv_BuiltInSubgroupSize"              => Cap.GroupNonUniform,
    "__spirv_BuiltInSubgroupLocalInvocationId" => Cap.GroupNonUniform,
)

"""
Emit a builtin Input global variable in SPIR-V.
Creates the appropriate type (vec3<u32> or u32), pointer type, OpVariable,
and BuiltIn decoration based on the global's name.
"""
function emit_builtin_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_name = LLVM.name(gv)
    gv_value_ty = LLVM.global_value_type(gv)

    # Look up builtin decoration from name
    builtin_id = get(SPIRV_BUILTIN_MAP, gv_name, nothing)
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

    cap = get(SPIRV_BUILTIN_CAPABILITY, gv_name, nothing)
    cap === nothing || require_capability!(mod, cap)

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
function emit_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)
    gv_name = LLVM.name(gv)

    # Must be an aggregate type Julia can lower as a constant composite.
    # Arrays cover lookup-table / polynomial-coefficient globals; structs
    # cover constant return values (e.g. an immutable result struct that
    # multiple branches share, like an EPA cap-out / "no contact" sentinel).
    if !(gv_value_ty isa LLVM.ArrayType || gv_value_ty isa LLVM.StructType)
        @warn "Skipping non-aggregate constant global: $gv_name (type: $gv_value_ty)"
        return
    end

    # Must have an initializer
    init = LLVM.initializer(gv)
    if init === nothing
        @warn "Skipping constant global without initializer: $gv_name"
        return
    end

    # Map the aggregate type to SPIR-V
    arr_spirv_ty = map_type!(state.type_ctx, gv_value_ty)

    # Build the composite constant recursively
    composite_id = emit_llvm_constant!(state, init, gv_value_ty)

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
function emit_llvm_constant!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.LLVMType)
    # Scalar constants
    if val isa LLVM.ConstantInt || val isa LLVM.ConstantFP ||
       val isa LLVM.UndefValue || val isa LLVM.PoisonValue ||
       val isa LLVM.ConstantAggregateZero
        return map_constant!(state.type_ctx, val)
    end

    # ConstantDataArray/ConstantDataVector — packed scalar data
    if val isa LLVM.ConstantDataSequential
        return emit_constant_data_array!(state, val, ty)
    end

    # ConstantArray — array of aggregate elements
    if ty isa LLVM.ArrayType
        return emit_constant_array!(state, val, ty)
    end

    # ConstantStruct
    if ty isa LLVM.StructType
        return emit_constant_struct!(state, val, ty)
    end

    error("Unsupported LLVM constant type for emission: $(typeof(val)), LLVM type: $ty")
end

"""Emit a ConstantDataArray (packed scalar data) as OpConstantComposite."""
function emit_constant_data_array!(state::SPIRVEmitterState, val::LLVM.ConstantDataSequential, ty::LLVM.ArrayType)
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
function emit_constant_array!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.ArrayType)
    n = LLVM.length(ty)
    elem_ty = LLVM.eltype(ty)
    arr_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    elem_ids = UInt32[]
    for i in 1:n
        elem_id = emit_llvm_constant!(state, ops[i]::LLVM.Constant, elem_ty)
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
function emit_constant_struct!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.StructType)
    members = LLVM.elements(ty)
    struct_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    member_ids = UInt32[]
    for (i, member_ty) in enumerate(members)
        member_id = emit_llvm_constant!(state, ops[i]::LLVM.Constant, member_ty)
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
function validate_spirv(spirv_bytes::Vector{UInt8}, llvm_ir::String="",
                          source_map::Dict{UInt32, Tuple{String, Int}}=Dict{UInt32, Tuple{String, Int}}())
    spirv_val = SPIRV_Tools_jll.spirv_val()
    spirv_dis_cmd = SPIRV_Tools_jll.spirv_dis()
    spv_path = lava_debug_path("lava_last.spv")

    # Write SPIR-V binary so spirv-val can read it
    write(spv_path, spirv_bytes)

    # Validate — capture stderr via temp file (spirv-val writes errors to stderr)
    val_err_file = tempname()
    # `lava_run` (not `wait(p)`): the wait condition can be lost once a Vulkan
    # device is up. Fail open on timeout — Vulkan rejects invalid SPIR-V anyway.
    p = lava_run(pipeline(`$spirv_val --target-env vulkan1.3 --scalar-block-layout $spv_path`;
                          stderr=val_err_file, stdout=devnull); label="spirv-val")
    if !process_exited(p)
        rm(val_err_file; force=true)
        return nothing
    end
    val_errors = isfile(val_err_file) ? read(val_err_file, String) : ""
    rm(val_err_file; force=true)
    p.exitcode == 0 && return nothing

    # ── Validation failed — build a useful error message ──

    # Disassemble and save (spawn + poll, not the deadlock-prone read(cmd))
    dis = try
        dis_out = tempname() * ".dis"
        pd = lava_run(pipeline(`$spirv_dis_cmd --no-color $spv_path`; stdout=dis_out); label="spirv-dis")
        txt = (process_exited(pd) && pd.exitcode == 0 && isfile(dis_out)) ? read(dis_out, String) : ""
        rm(dis_out; force=true)
        txt
    catch ex
        # Disassembly decorates a diagnostic; a missing or broken `spirv-dis` must
        # not replace the message being built. Narrowed so an internal fault here
        # still surfaces rather than silently producing an empty listing.
        ex isa Union{Base.IOError, SystemError, ProcessFailedException} || rethrow()
        @warn "Lava: spirv-dis failed; diagnostic will omit the disassembly" exception = ex
        ""
    end
    if !isempty(dis)
        write(lava_debug_path("lava_last.dis"), dis)
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
            short_file = shorten_path(file)
            println(io, "    %$id → $short_file:$line")
            # Try to show the Julia source line
            print_julia_source_context(io, file, line)
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
                        annotation = "  ← $(shorten_path(loc[1])):$(loc[2])"
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
function shorten_path(path::String)
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
function print_julia_source_context(io::IO, file::String, line::Int)
    isfile(file) || return
    try
        src_lines = readlines(file)
        lo = max(1, line - 1)
        hi = min(length(src_lines), line + 1)
        for j in lo:hi
            marker = j == line ? " >> " : "    "
            println(io, "      │$marker$j: ", src_lines[j])
        end
    catch ex
        # This is decorating an error message with source context. Failing to read
        # the file must not replace the diagnostic the caller is about to see, so
        # the fallback is real — but it is narrowed to the file being unreadable,
        # and it says so instead of printing nothing.
        ex isa Union{SystemError, Base.IOError, ArgumentError, BoundsError} || rethrow()
        println(io, "      │    (source unavailable: ", sprint(showerror, ex), ")")
    end
end

"""
Disassemble SPIR-V binary to text for debugging.
"""
function disassemble_spirv(spirv_bytes::Vector{UInt8})
    spirv_dis = SPIRV_Tools_jll.spirv_dis()
    tmpfile = tempname() * ".spv"
    out_path = tempname() * ".dis"
    try
        write(tmpfile, spirv_bytes)
        # `read(cmd, String)` blocks on the deadlock-prone wait; spawn + poll.
        p = lava_run(pipeline(`$spirv_dis --no-color $tmpfile`; stdout=out_path); label="spirv-dis")
        return (process_exited(p) && p.exitcode == 0 && isfile(out_path)) ? read(out_path, String) : ""
    finally
        rm(tmpfile; force=true)
        rm(out_path; force=true)
    end
end

# ── Lower unsupported LLVM intrinsics ──

"""
Lower LLVM intrinsics that SPIR-V cannot represent:
- llvm.memcpy → typed load + store (using destination alloca's type)
- llvm.lifetime.start/end → removed (no-op)
"""
function lower_unsupported_intrinsics!(mod::LLVM.Module)
    to_erase = LLVM.Instruction[]

    for fn in LLVM.functions(mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                called = LLVM.called_operand(inst)
                called isa LLVM.Function || continue
                fname = LLVM.name(called)

                if startswith(fname, "llvm.memcpy")
                    lower_memcpy!(inst)
                    push!(to_erase, inst)
                elseif startswith(fname, "llvm.memset")
                    lower_memset!(inst)
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
function lower_memcpy!(inst::LLVM.CallInst)
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
function lower_memset!(inst::LLVM.CallInst)
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

