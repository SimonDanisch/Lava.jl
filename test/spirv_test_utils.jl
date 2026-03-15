# spirv_test_utils.jl — SPIR-V disassembly pattern matching and golden file helpers
#
# Adapts clspv's FileCheck pattern checks and rust-gpu's golden file comparison
# to Julia's Test.jl. Used by Tier 1 (pattern checks) and Tier 2 (golden files).
#
# Include once at top level; test files use `if !@isdefined(SPIRVTestUtils)` guard.

module SPIRVTestUtils

using Test
using Lava
using SPIRV_Tools_jll
const HAS_LLC = try
    using SPIRV_LLVM_Backend_jll
    true
catch
    false
end

export check, check_not, check_dag, check_sequence, check_count, check_regex,
       normalize_spirv, compare_golden, compile_and_disasm,
       spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

# ── Pattern Matching (inspired by clspv FileCheck) ──

"""Check that pattern `p` appears somewhere in disassembly `d`."""
function check(d::AbstractString, p::AbstractString; msg="")
    @test occursin(p, d)
end

"""Check that pattern `p` does NOT appear in disassembly `d`."""
function check_not(d::AbstractString, p::AbstractString; msg="")
    @test !occursin(p, d)
end

"""Check that ALL patterns in `ps` appear (any order) — like FileCheck CHECK-DAG."""
function check_dag(d::AbstractString, ps::AbstractVector{<:AbstractString})
    for p in ps
        @test occursin(p, d)
    end
end

"""Check that patterns appear in order (not necessarily consecutive)."""
function check_sequence(d::AbstractString, ps::AbstractVector{<:AbstractString})
    pos = 1
    for (i, p) in enumerate(ps)
        idx = findnext(p, d, pos)
        @test idx !== nothing
        if idx !== nothing
            pos = last(idx) + 1
        else
            break
        end
    end
end

"""Check that pattern `p` appears exactly `n` times."""
function check_count(d::AbstractString, p::AbstractString, n::Int)
    c = 0
    pos = 1
    while true
        idx = findnext(p, d, pos)
        idx === nothing && break
        c += 1
        pos = last(idx) + 1
    end
    @test c == n
end

"""Check that regex pattern `p` matches somewhere in disassembly `d`."""
function check_regex(d::AbstractString, p::AbstractString)
    @test occursin(Regex(p), d)
end

# ── Golden File Comparison (inspired by rust-gpu --bless) ──

"""Normalize SPIR-V disassembly for stable comparison."""
function normalize_spirv(text::AbstractString)
    lines = split(text, '\n')
    filtered = String[]
    for line in lines
        stripped = rstrip(line)
        startswith(stripped, "               OpLine") && continue
        startswith(stripped, "               OpSource") && continue
        startswith(stripped, "               OpString") && continue
        startswith(stripped, "               OpModuleProcessed") && continue
        push!(filtered, stripped)
    end
    while !isempty(filtered) && isempty(filtered[1])
        popfirst!(filtered)
    end
    while !isempty(filtered) && isempty(filtered[end])
        pop!(filtered)
    end
    return join(filtered, '\n') * '\n'
end

"""Compare SPIR-V bytes against a golden `.spvasm` file."""
function compare_golden(spirv_bytes::Vector{UInt8}, golden_path::AbstractString)
    disasm = Lava.disassemble_spirv(spirv_bytes)
    normalized = normalize_spirv(disasm)

    if get(ENV, "LAVA_BLESS", "") == "1"
        mkpath(dirname(golden_path))
        write(golden_path, normalized)
        @info "Blessed golden file: $golden_path"
        @test true
    else
        if !isfile(golden_path)
            @error "Golden file not found: $golden_path\nRun with LAVA_BLESS=1 to create it."
            @test false
        else
            expected = read(golden_path, String)
            if normalized != expected
                @error "Golden file mismatch: $golden_path"
                for (i, (a, e)) in enumerate(zip(split(normalized, '\n'), split(expected, '\n')))
                    if a != e
                        @error "First diff at line $i" actual=a expected=e
                        break
                    end
                end
                @test normalized == expected
            else
                @test true
            end
        end
    end
    return normalized
end

# ── Convenience Compilation + Disassembly ──

"""Compile a function and return `(disasm::String, spirv_bytes::Vector{UInt8})`."""
function compile_and_disasm(@nospecialize(f), @nospecialize(tt);
                            stage::Symbol=:compute,
                            workgroup_size::NTuple{3,Int}=(64,1,1),
                            config=nothing,
                            payload_type::Symbol=:f32,
                            validate::Bool=true)
    if stage == :compute
        result = Lava.lava_compile_gpu(f, tt; workgroup_size, validate)
        bytes = result.spirv_bytes
    elseif stage in (:vertex, :fragment, :geometry, :tess_control, :tess_eval)
        result = Lava.lava_compile_gfx_shader(f, tt; stage, config, validate)
        bytes = result.spirv_bytes
    elseif stage in (:raygen, :closesthit, :miss, :anyhit, :intersection, :callable)
        result = Lava.lava_compile_rt_shader(f, tt; stage, payload_type, validate)
        bytes = result.spirv_bytes
    else
        error("Unknown stage: $stage")
    end
    disasm = Lava.disassemble_spirv(bytes)
    return (disasm, bytes)
end

# ── spirv-opt Roundtrip Validation ──

"""
    spirv_opt_roundtrip(spirv_bytes; passes="-O") -> Vector{UInt8}

Run spirv-opt on SPIR-V binary and re-validate with spirv-val.
Proves the SPIR-V is semantically well-formed enough for a production optimizer.
Returns optimized bytes or throws on failure.
"""
function spirv_opt_roundtrip(spirv_bytes::Vector{UInt8}; passes::String="-O")
    spirv_opt_cmd = SPIRV_Tools_jll.spirv_opt()
    spirv_val_cmd = SPIRV_Tools_jll.spirv_val()

    in_path = tempname() * ".spv"
    out_path = tempname() * ".spv"
    try
        write(in_path, spirv_bytes)
        # Run optimizer with validation after every pass
        opt_err = tempname()
        p = run(pipeline(`$spirv_opt_cmd --target-env=vulkan1.3 --scalar-block-layout
                          --validate-after-all $passes $in_path -o $out_path`;
                         stderr=opt_err); wait=false)
        wait(p)
        err_msg = isfile(opt_err) ? read(opt_err, String) : ""
        rm(opt_err; force=true)
        if p.exitcode != 0
            error("spirv-opt failed (exit $(p.exitcode)):\n$err_msg")
        end

        optimized = read(out_path)

        # Re-validate the optimized output
        val_err = tempname()
        p2 = run(pipeline(`$spirv_val_cmd --target-env vulkan1.3 --scalar-block-layout $out_path`;
                          stderr=val_err); wait=false)
        wait(p2)
        val_msg = isfile(val_err) ? read(val_err, String) : ""
        rm(val_err; force=true)
        if p2.exitcode != 0
            error("spirv-val failed on optimized SPIR-V:\n$val_msg")
        end

        return optimized
    finally
        rm(in_path; force=true)
        rm(out_path; force=true)
    end
end

# ── Vendor Safety Checks ──

"""
    check_vendor_safety(disasm::String)

Run all known vendor-critical safety checks on SPIR-V disassembly.
Tests with @test — call inside a @testset.

Checks:
- NVIDIA: PSB loads/stores must have Aligned decoration
- All vendors: no CrossDevice scope in atomics
- All vendors: no OpUnreachable (crashes some drivers)
- All vendors: structured CF merges paired with branches
"""
function check_vendor_safety(disasm::String)
    lines = split(disasm, '\n')

    # PSB alignment: every PhysicalStorageBuffer load/store must have Aligned
    for line in lines
        if occursin("PhysicalStorageBuffer", line) &&
           (occursin("OpLoad", line) || occursin("OpStore", line))
            @test occursin("Aligned", line)
        end
    end

    # No CrossDevice scope in atomics
    @test !occursin("CrossDevice", disasm)

    # No OpUnreachable
    @test !occursin("OpUnreachable", disasm)

    # Structured CF: every OpSelectionMerge followed by branch within 3 lines
    for (i, line) in enumerate(lines)
        if occursin("OpSelectionMerge", line)
            found = false
            for j in (i+1):min(i+3, length(lines))
                if occursin("OpBranchConditional", lines[j]) || occursin("OpSwitch", lines[j])
                    found = true
                    break
                end
            end
            @test found
        end
    end

    # Structured CF: every OpLoopMerge followed by branch within 3 lines
    for (i, line) in enumerate(lines)
        if occursin("OpLoopMerge", line)
            found = false
            for j in (i+1):min(i+3, length(lines))
                if occursin("OpBranch", lines[j])
                    found = true
                    break
                end
            end
            @test found
        end
    end
end

# ── llc Reference Compilation ──

"""
    compile_with_llc(llvm_ir::String) -> (compiled::Bool, disasm::String)

Compile LLVM IR to SPIR-V using llc (SPIRV_LLVM_Backend_jll) as a reference oracle.
Returns (true, disasm) on success, (false, error_msg) on failure.

Uses `spirv64-unknown-unknown` triple (OpenCL flavor). The Vulkan triple
(`spirv-unknown-vulkan1.3`) crashes llc on BDA kernels due to missing
hlsl.shader attributes. The OpenCL output is structurally comparable:
same types, same arithmetic ops, same entry point pattern.
"""
function compile_with_llc(llvm_ir::String)
    HAS_LLC || return (false, "SPIRV_LLVM_Backend_jll not available")
    llc_cmd = SPIRV_LLVM_Backend_jll.llc()
    spirv_dis_cmd = SPIRV_Tools_jll.spirv_dis()

    ll_path = tempname() * ".ll"
    spv_path = tempname() * ".spv"
    try
        write(ll_path, llvm_ir)

        # Compile with llc — spirv64 triple, binary output
        err_file = tempname()
        p = run(pipeline(`$llc_cmd -mtriple=spirv64-unknown-unknown -O0
                          -filetype=obj -o $spv_path $ll_path`;
                         stderr=err_file); wait=false)
        wait(p)
        err_msg = isfile(err_file) ? read(err_file, String) : ""
        rm(err_file; force=true)
        if p.exitcode != 0
            return (false, "llc failed: $err_msg")
        end

        # Disassemble (don't validate with vulkan target — it's OpenCL flavor)
        disasm = try
            read(`$spirv_dis_cmd --no-color $spv_path`, String)
        catch e
            return (false, "disassembly failed: $e")
        end

        return (true, disasm)
    finally
        rm(ll_path; force=true)
        rm(spv_path; force=true)
    end
end

end # module SPIRVTestUtils
