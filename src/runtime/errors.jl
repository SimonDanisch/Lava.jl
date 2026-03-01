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
end

function LavaCompilationError(operation, message, suggestion;
    julia_file="", julia_line=0, spirv_id=nothing, raw_error="")
    LavaCompilationError(operation, message, suggestion, julia_file, julia_line, spirv_id, raw_error)
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
    if !isempty(e.raw_error)
        print(io, "  Raw error: ", e.raw_error, "\n")
    end
    if !isempty(e.suggestion)
        print(io, "  Suggestion: ", e.suggestion)
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
        return read(`$spirv_dis --no-color $path`, String)
    end
end
