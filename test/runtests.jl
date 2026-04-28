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
        include(joinpath(@__DIR__, "test_rayquery_vs_cpu.jl"))
        include(joinpath(@__DIR__, "test_aabb_blas_overlap.jl"))
        include(joinpath(@__DIR__, "test_instance_masks.jl"))
    end

    # ── Tier 3b: Struct Broadcast Regression Tests ──
    @testset "Tier 3b: Struct Broadcast" begin
        include(joinpath(@__DIR__, "test_struct_broadcast.jl"))
    end

    # ── Narrow phase (CPU) -- ConvexShape / GJK / EPA ──
    # Pure-CPU reference implementations for the P4 narrow-phase pipeline; no GPU.
    @testset "Narrow phase (CPU)" begin
        include(joinpath(@__DIR__, "test_convex_shape.jl"))
        include(joinpath(@__DIR__, "test_gjk.jl"))
        include(joinpath(@__DIR__, "test_epa.jl"))
        include(joinpath(@__DIR__, "test_narrow_phase_kernel.jl"))
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

    # ── Tier 3e2: BAR-memory copy_buffer! sync regression ──
    @testset "Tier 3e2: BAR Memcpy Sync" begin
        include(joinpath(@__DIR__, "test_bar_memcpy_sync.jl"))
    end

    # ── Tier 3e3: Raycore MultiTypeSet surgical-per-mutation invariants ──
    @testset "Tier 3e3: MultiTypeSet Surgical" begin
        include(joinpath(@__DIR__, "test_multitypeset_surgical.jl"))
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

    # ── Tier 3i: HW TLAS (Lava.HWTLAS) ──
    @testset "HW TLAS — stress + correctness" begin
        include(joinpath(@__DIR__, "test_hwtlas_stress.jl"))
    end

    @testset "HW TLAS — mesh update" begin
        include(joinpath(@__DIR__, "test_hwtlas_mesh_update.jl"))
    end

    @testset "HW TLAS — UAF safety" begin
        include(joinpath(@__DIR__, "test_hwtlas_uaf_safety.jl"))
    end

    @testset "HW TLAS — nonblocking sync!" begin
        include(joinpath(@__DIR__, "test_hwtlas_nonblocking_sync.jl"))
    end

    # ── Tier 3j: Phase-M alloc/free regression matrix ──────────────────
    # Five MWEs that progressively approach Hikari's render workload.
    # All five MUST stay clean across 20+ iters with per-iter alloc/free.
    # If one starts crashing, that narrows the failing pattern — see
    # docs/specs/2026-04-25-iter6-cascade-investigation.md.
    @testset "Phase-M alloc/free MWE matrix" begin
        @testset "compute" begin
            include(joinpath(@__DIR__, "mwe_alloc_dispatch_free_loop.jl"))
        end
        @testset "RT direct" begin
            include(joinpath(@__DIR__, "mwe_rt_alloc_dispatch_free_loop.jl"))
        end
        @testset "RT indirect + busy" begin
            include(joinpath(@__DIR__, "mwe_indirect_rt_busy_loop.jl"))
        end
        @testset "12 distinct kernels" begin
            include(joinpath(@__DIR__, "mwe_distinct_kernels_per_iter.jl"))
        end
        @testset "SoA workqueue" begin
            include(joinpath(@__DIR__, "mwe_soa_workqueue_per_iter.jl"))
        end
        @testset "VolPath shape" begin
            include(joinpath(@__DIR__, "mwe_volpath_shape_per_iter.jl"))
        end
    end

    # ── Tier 4: GPU-AV regression (gated on LAVA_GPU_AV=1; ~minutes) ──
    if get(ENV, "LAVA_GPU_AV", "0") == "1"
        @testset "GPU-AV clean" begin
            include(joinpath(@__DIR__, "test_gpuav_clean.jl"))
        end
    end
end

