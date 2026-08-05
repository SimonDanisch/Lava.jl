# Lava.jl error infrastructure
# Context-rich errors for Vulkan operations, SPIR-V validation, and compilation failures.

"""
    LavaError <: Exception

Base error type for all Lava.jl failures. Every error answers three questions:
1. What were you trying to do?
2. What went wrong?
3. What can you do about it?
"""
struct LavaError <: Exception
    operation::String
    message::String
    suggestion::String
end

function Base.showerror(io::IO, e::LavaError)
    print(io, "LavaError during ", e.operation, ":\n")
    print(io, "  ", e.message, "\n")
    if !isempty(e.suggestion)
        print(io, "  Suggestion: ", e.suggestion)
    end
end

"""
    LavaCompilationError <: Exception

Error during Julia → SPIR-V compilation with source mapping.
"""
struct LavaCompilationError <: Exception
    operation::String
    message::String
    suggestion::String
    # Source mapping: which Julia source location caused the error
    julia_file::String
    julia_line::Int
    # Optional: SPIR-V instruction ID for spirv-val errors
    spirv_id::Union{Nothing, UInt32}
    # Optional: raw spirv-val or driver error output
    raw_error::String
    # Optional: deduplicated call chains through user code
    call_chains::String
end

function LavaCompilationError(operation, message, suggestion;
    julia_file="", julia_line=0, spirv_id=nothing, raw_error="", call_chains="")
    LavaCompilationError(operation, message, suggestion, julia_file, julia_line, spirv_id, raw_error, call_chains)
end

function Base.showerror(io::IO, e::LavaCompilationError)
    print(io, "LavaCompilationError during ", e.operation, ":\n")
    print(io, "  ", e.message, "\n")
    if !isempty(e.julia_file) && e.julia_line > 0
        print(io, "  at ", e.julia_file, ":", e.julia_line, "\n")
    end
    if e.spirv_id !== nothing
        print(io, "  SPIR-V instruction ID: %", e.spirv_id, "\n")
    end
    if !isempty(e.call_chains)
        print(io, "\n  ", e.call_chains, "\n")
    end
    if !isempty(e.suggestion)
        print(io, "\n  Suggestion: ", e.suggestion, "\n")
    end
    if !isempty(e.raw_error)
        print(io, "\n  Raw error (full GPUCompiler output):\n  ", e.raw_error)
    end
end

"""
    LavaVulkanError <: Exception

Error from a Vulkan API call with context.
"""
struct LavaVulkanError <: Exception
    call::String
    vk_result::Int32
    message::String
    suggestion::String
end

function Base.showerror(io::IO, e::LavaVulkanError)
    print(io, "LavaVulkanError in ", e.call, ":\n")
    print(io, "  VkResult: ", e.vk_result, " — ", e.message, "\n")
    if !isempty(e.suggestion)
        print(io, "  Suggestion: ", e.suggestion)
    end
end

"""
    safe_fin_log(msg)

Finalizer-safe logging. Uses `jl_safe_printf` which is a raw ccall that
cannot yield — the only form of diagnostic output allowed inside a Julia
finalizer.  `msg` must be a `String` that already contains any trailing
newline; no interpolation at call time (that would trigger dispatch /
world-age / allocation).  Prefer short fixed strings built by the caller.
"""
# `msg` goes through a "%s" FORMAT, not as the format string itself.
# `jl_safe_printf(const char *fmt, ...)` interprets its first argument as a
# printf format, so a message containing `%` — a path, a driver string, an
# exception text like "100% of heap" — was undefined behaviour: `%s` would read
# a nonexistent vararg off the stack, inside a finalizer, where there is no way
# to recover. Passing the string as an ARGUMENT makes any content safe, and
# costs nothing.
@inline safe_fin_log(msg::String) =
    ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "%s", msg)

"""
    @vk_checked site_str vk_call

Wrap a Vulkan API call that returns a `ResultTypes.Result`: `unwrap` it
(propagating VkResult errors as exceptions), then flush accumulated
validation-layer messages via `check_validation_errors!(site_str)`.  Ensures
no validation error survives past the create/allocate call that produced
it and leaks into an unrelated later frame's diagnostic.

Usage:
    pipeline = @vk_checked "create_graphics_pipeline" Vulkan.create_graphics_pipelines(dev, [ci])[1]
"""
macro vk_checked(site, call)
    quote
        r = unwrap($(esc(call)))
        check_validation_errors!($(esc(site)))
        r
    end
end

# ── Source Mapping ──

"""
    SourceMap

Maps SPIR-V instruction IDs back to Julia source locations through LLVM IR.
Populated during compilation by the SPIR-V emitter.
"""
struct SourceMap
    spirv_to_julia::Dict{UInt32, Tuple{String, Int}}  # SPIR-V ID → (file, line)
end

SourceMap() = SourceMap(Dict{UInt32, Tuple{String, Int}}())

"""Look up Julia source location for a SPIR-V instruction ID."""
function lookup_source(sm::SourceMap, spirv_id::UInt32)
    get(sm.spirv_to_julia, spirv_id, ("", 0))
end

"""Record a mapping from SPIR-V instruction ID to Julia source location."""
function record_source!(sm::SourceMap, spirv_id::UInt32, file::String, line::Int)
    sm.spirv_to_julia[spirv_id] = (file, line)
end

# ── SPIR-V Validation ──

"""
    validate_spirv(spirv_binary::Vector{UInt8}) -> Nothing

Run spirv-val on SPIR-V binary. Throws LavaCompilationError on failure.
"""
function validate_spirv(spirv_binary::AbstractVector{UInt8})
    mktemp() do path, io
        write(io, spirv_binary)
        close(io)
        spirv_val = SPIRV_Tools_jll.spirv_val()
        cmd = `$spirv_val --target-env vulkan1.3 $path`
        errbuf = IOBuffer()
        proc = run(pipeline(cmd; stderr=errbuf, stdout=errbuf); wait=false)
        wait(proc)
        output = String(take!(errbuf))
        if proc.exitcode != 0
            throw(LavaCompilationError(
                "SPIR-V validation",
                "spirv-val reported errors",
                "Check the SPIR-V output for invalid instructions";
                raw_error=output
            ))
        end
    end
end

"""
    disassemble_spirv(spirv_binary::Vector{UInt8}) -> String

Run spirv-dis on SPIR-V binary. Returns disassembly text.
"""
function disassemble_spirv(spirv_binary::AbstractVector{UInt8})
    mktemp() do path, io
        write(io, spirv_binary)
        close(io)
        spirv_dis = SPIRV_Tools_jll.spirv_dis()
        out = tempname() * ".dis"
        # `read(cmd, String)` blocks on the pipe forever once a Vulkan device is
        # up (lost SIGCHLD); spawn + poll via lava_run, read the output file.
        p = lava_run(pipeline(`$spirv_dis --no-color $path`; stdout=out); label="spirv-dis")
        txt = (process_exited(p) && p.exitcode == 0 && isfile(out)) ? read(out, String) : ""
        rm(out; force=true)
        return txt
    end
end
