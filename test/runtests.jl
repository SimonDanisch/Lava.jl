# Lava.jl Master Test Runner
#
# 3-tier testing:
#   Tier 1: Pattern checks on SPIR-V disassembly (no GPU needed)
#   Tier 2: Golden file comparison (no GPU needed)
#   Tier 3: GPU execution tests (requires Vulkan device)

using Test
using Lava

# Load test utilities once — individual files guard with @isdefined(SPIRVTestUtils)
include(joinpath(@__DIR__, "spirv_test_utils.jl"))
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

# Print driver info for test logs (helps distinguish RADV vs lavapipe vs others)
let ctx = Lava.vk_context()
    @info "Lava test suite running on: $(ctx.device_name)" has_rt=(ctx.rt_pipeline_properties !== nothing)
end

@testset "Lava.jl" begin

    # ── Tier 1: SPIR-V Emission Pattern Checks ──
    @testset "Tier 1: SPIR-V Emission" begin
        spirv_dir = joinpath(@__DIR__, "spirv")
        for f in sort(readdir(spirv_dir; join=true))
            endswith(f, ".jl") || continue
            @info "Running $(basename(f))..."
            include(f)
        end
    end

    # ── Tier 2: Golden File Comparison ──
    @testset "Tier 2: Golden Files" begin
        include(joinpath(@__DIR__, "test_golden.jl"))
    end

    # ── Tier 3: GPU Execution ──
    @testset "Tier 3: GPU Execution" begin
        include(joinpath(@__DIR__, "test_handwritten_spirv.jl"))
        include(joinpath(@__DIR__, "test_handwritten_rt.jl"))
    end

    # ── Tier 3b: Struct Broadcast Regression Tests ──
    @testset "Tier 3b: Struct Broadcast" begin
        include(joinpath(@__DIR__, "test_struct_broadcast.jl"))
    end

    # ── Tier 3c: Atomics & Batched Dispatch ──
    @testset "Tier 3c: Atomics & Dispatch" begin
        include(joinpath(@__DIR__, "test_atomics_and_dispatch.jl"))
    end

    # ── Tier 4: GPUArrays TestSuite ──
    @testset "Tier 4: GPUArrays TestSuite" begin
        include(joinpath(@__DIR__, "test_gpuarrays.jl"))
    end
end

