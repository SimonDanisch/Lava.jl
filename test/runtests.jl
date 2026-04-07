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
    # Skipped in CI — SPIR-V IDs differ across platforms (LLVM CSE).
    # Run locally with: LAVA_BLESS=1 julia --project test/runtests.jl
    if get(ENV, "CI", "") != "true"
        @testset "Tier 2: Golden Files" begin
            include(joinpath(@__DIR__, "test_golden.jl"))
        end
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

    # ── Tier 3d: NVIDIA Regression & Stress Tests ──
    @testset "Tier 3d: NVIDIA Regression & Stress" begin
        include(joinpath(@__DIR__, "test_nvidia_regression.jl"))
    end

    # ── Tier 3e: GPU Memory Safety ──
    @testset "Tier 3e: GPU Memory Safety" begin
        include(joinpath(@__DIR__, "test_gpu_memory_safety.jl"))
    end

    # ── Tier 3f: Caching & Allocations ──
    @testset "Tier 3f: Caching & Allocations" begin
        include(joinpath(@__DIR__, "test_caching_and_allocations.jl"))
    end

    # ── Tier 3h: Disk Cache & Two-Tier Caching ──
    @testset "Tier 3h: Kernel Cache" begin
        include(joinpath(@__DIR__, "test_disk_cache.jl"))
    end

    # ── Tier 3g: Graphics Pipeline ──
    @testset "Tier 3g: Graphics Pipeline" begin
        include(joinpath(@__DIR__, "test_graphics_pipeline.jl"))
    end

    # ── Tier 4: GPUArrays TestSuite ──
    @testset "Tier 4: GPUArrays TestSuite" begin
        include(joinpath(@__DIR__, "test_gpuarrays.jl"))
    end
end

