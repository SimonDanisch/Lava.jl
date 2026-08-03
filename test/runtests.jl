# Lava.jl Master Test Runner
#
# Tiered testing:
#   Tier 1: Pattern checks on SPIR-V disassembly (no GPU needed) — semantic
#           assertions via check/check_not/check_dag, robust across GPUCompiler
#           and LLVM versions (unlike byte-exact golden files, which we dropped).
#   Tier 3: GPU execution tests (requires Vulkan device)

using Test
using Lava

# Load test utilities once — individual files guard with @isdefined(SPIRVTestUtils)
include(joinpath(@__DIR__, "spirv_test_utils.jl"))
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

# ── Fast mode ──
# A ~3-minute subset for local CI iteration (`act`, smoke-checking workflow
# changes, fast loop after touching the emitter).  Triggered by either:
#
#     julia> Pkg.test("Lava"; test_args=["fast"])
#     $  LAVA_FAST=1 julia --project -e 'using Pkg; Pkg.test()'
#     $  julia --project test/runtests.jl fast
#
# Includes the categories that have historically caught regressions:
#   Tier 1 SPIR-V emission, Tier 3 GPU execution, Tier 3c atomics & dispatch,
#   Tier 3g graphics pipeline, Tier 3h kernel cache, Phase-M alloc/free MWE.
# Skips the heavy ones: Tier 4 (GPUArrays, ~12 min) and the stress / per-vendor
# suites.  Full runs are still the default locally.
#
# ── CI mode (`ci`) ──
# The fast subset PLUS the full Tier 4 GPUArrays correctness suite (~15-20 min).
# This is what the lavapipe CI runs: the GPUArrays suite is the bulk of real
# correctness coverage, and it runs on lavapipe with the few process-crashing
# groups skipped (see test_gpuarrays.jl `LAVAPIPE_CRASH_SKIP`). The RADV-only /
# per-vendor stress tiers stay FULL-only. Trigger with test_args=["ci"] or LAVA_CI=1.
const CI_MODE   = "ci"   in ARGS || get(ENV, "LAVA_CI",   "0") == "1"
const FAST_MODE = ("fast" in ARGS || get(ENV, "LAVA_FAST", "0") == "1") && !CI_MODE
const FULL_MODE = !FAST_MODE && !CI_MODE
FAST_MODE && @info "Lava: FAST mode — running essential tiers only (~2-3 min)."
CI_MODE   && @info "Lava: CI mode — fast subset + Tier 4 GPUArrays (~15-20 min)."

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

    # ── Tier 1b: Compiler IR passes (no GPU) ──
    @testset "Tier 1b: Compiler IR passes" begin
        include(joinpath(@__DIR__, "test_replace_unreachable.jl"))
    end

    # ── Tier 3a: Workgroup barrier-skip fix (GPU; catches lavapipe deadlock) ──
    @testset "Tier 3a: Barrier skip fix" begin
        include(joinpath(@__DIR__, "test_barrier_skip.jl"))
    end

    # ── Tier 3a2: norm FTZ rescaling regression (GPU; Float64/ComplexF64) ──
    @testset "Tier 3a2: norm FTZ rescaling" begin
        include(joinpath(@__DIR__, "test_norm_overflow.jl"))
    end

    # ── Tier 3a3: workgroup (shared) memory stress (GPU) ──
    @testset "Tier 3a3: shared-memory stress" begin
        include(joinpath(@__DIR__, "test_shared_memory_stress.jl"))
    end

    # ── Tier 3a4: OpSelect-of-Workgroup-pointers type-dedup regression (GPU) ──
    @testset "Tier 3a4: OpSelect Workgroup pointer dedup" begin
        include(joinpath(@__DIR__, "test_select_width_mismatch.jl"))
    end

    # ── Tier 3: GPU Execution ──
    @testset "Tier 3: GPU Execution" begin
        include(joinpath(@__DIR__, "test_handwritten_spirv.jl"))
        include(joinpath(@__DIR__, "test_handwritten_rt.jl"))
        include(joinpath(@__DIR__, "test_rayquery_vs_cpu.jl"))
        include(joinpath(@__DIR__, "test_aabb_blas_overlap.jl"))
        include(joinpath(@__DIR__, "test_instance_masks.jl"))
    end

    # ── Tier 3b: Struct Broadcast Regression Tests (full only) ──
    if FULL_MODE
        @testset "Tier 3b: Struct Broadcast" begin
            include(joinpath(@__DIR__, "test_struct_broadcast.jl"))
        end
    end

    # ── Narrow phase (CPU) -- ConvexShape / GJK / EPA (full only) ──
    # Pure-CPU reference implementations for the P4 narrow-phase pipeline; no GPU.
    if FULL_MODE
        @testset "Narrow phase (CPU)" begin
            include(joinpath(@__DIR__, "test_convex_shape.jl"))
            include(joinpath(@__DIR__, "test_gjk.jl"))
            include(joinpath(@__DIR__, "test_epa.jl"))
            include(joinpath(@__DIR__, "test_narrow_phase_kernel.jl"))
            include(joinpath(@__DIR__, "test_contact_record.jl"))
            include(joinpath(@__DIR__, "test_narrow_phase_contacts.jl"))
        end
    end

    # ── Tier 3c: Atomics & Batched Dispatch ──
    @testset "Tier 3c: Atomics & Dispatch" begin
        include(joinpath(@__DIR__, "test_atomics_and_dispatch.jl"))
    end

    # ── Tier 3c2: On-kernel printf (DebugPrintf SPIR-V emission) ──
    # Only the spirv-val emission test runs here; the live-output test resets the
    # device and is opt-in via LAVA_PRINTF_LIVE=1.
    @testset "Tier 3c2: @lava_printf" begin
        include(joinpath(@__DIR__, "test_lava_printf.jl"))
    end

    # ── Tier 3v: hardware H.264 video decode (skips without a video-decode queue) ──
    @testset "Tier 3v: H.264 hardware decode" begin
        include(joinpath(@__DIR__, "test_video_decode.jl"))
    end

    @testset "video decode layout" begin
        include(joinpath(@__DIR__, "test_video_decode_layout.jl"))
    end

    @testset "double-indirect MVector access" begin
        include(joinpath(@__DIR__, "test_double_indirect.jl"))
    end

    @testset "Int32 CartesianIndex into Broadcasted" begin
        include(joinpath(@__DIR__, "test_int32_cartesian_miscompile.jl"))
    end

    @testset "workgroup size limit" begin
        include(joinpath(@__DIR__, "test_workgroup_limit.jl"))
    end

    @testset "fastdiv index decomposition" begin
        include(joinpath(@__DIR__, "test_fastdiv.jl"))
    end

    @testset "cooperative-matrix epilogue" begin
        include(joinpath(@__DIR__, "test_coopmat_epilogue.jl"))
    end

    @testset "shared stores through a divided index" begin
        include(joinpath(@__DIR__, "test_shared_index_division.jl"))
    end

    @testset "staged GEMM" begin
        include(joinpath(@__DIR__, "test_gemm_staged.jl"))
    end

    @testset "whole-struct copy alignment" begin
        include(joinpath(@__DIR__, "test_struct_copy_alignment.jl"))
    end

    @testset "systematic struct alignment" begin
        include(joinpath(@__DIR__, "test_struct_alignment_systematic.jl"))
    end

    @testset "pipeline cache avoids driver compilation" begin
        include(joinpath(@__DIR__, "test_pipeline_cache_no_compile.jl"))
    end

    # Straight after the reset above, and before `test_static_workgroup.jl` —
    # which is where this crashed, because that file calls `GC.gc()` explicitly
    # and so ran whichever finalizer the reset had stranded.
    @testset "a buffer may outlive a device reset" begin
        include(joinpath(@__DIR__, "test_device_reset_finalizer.jl"))
    end

    @testset "static workgroup indexing" begin
        include(joinpath(@__DIR__, "test_static_workgroup.jl"))
    end

    # The one field the multi-device work rests on. Needs no second GPU: what it
    # pins is that "this device" and "whichever is current" stay distinguishable.
    @testset "a backend knows its device" begin
        include(joinpath(@__DIR__, "test_backend_context.jl"))
    end

    # Two live devices in one process — the real GPU and lavapipe, which the
    # loader enumerates together, so this needs no second card. Asserts both
    # compute correctly AND that one kernel compiles twice: a shared pipeline
    # can still give the right answer by luck. See the file for the five
    # separate pieces of module-scope device state it found.
    @testset "two devices in one process" begin
        include(joinpath(@__DIR__, "twodevice_probe.jl"))
        probe()
    end

    # Catches an invalid module that the driver accepts and runs correctly, so it
    # is invisible to every other test unless LAVA_VALIDATION=1 is set.
    @testset "no OpBitcast on a logical pointer" begin
        include(joinpath(@__DIR__, "test_logical_pointer_bitcast.jl"))
    end

    # Only meaningful on a device whose subgroup width is not fixed — on wave32
    # hardware "requested" and "default" are the same number and nothing is proved.
    @testset "pinned subgroup width" begin
        include(joinpath(@__DIR__, "test_subgroup_size_pinning.jl"))
    end

    # Both halves of the coopmat 32-lane pin: that it is not needed here, and
    # that the refusal fires when it would be. The second half is unreachable on
    # any device present, so it drives the capability cache instead.
    @testset "coopmat 32-lane pin" begin
        include(joinpath(@__DIR__, "test_coopmat_subgroup_pin.jl"))
    end

    # A handled allocation failure must absorb its own validation messages, or it
    # aborts whatever unrelated code calls check_validation_errors! next.
    @testset "tolerated alloc failure" begin
        include(joinpath(@__DIR__, "test_tolerated_alloc_failure.jl"))
    end

    # The frozen path is the one the runners ship, and the profiler could not see
    # it: 0 kernels reported against 45 live dispatches.
    @testset "frozen kernels are visible to the profiler" begin
        include(joinpath(@__DIR__, "test_frozen_kernels_visible.jl"))
    end

    # `vk_context` had methods for three of AnyLavaArray's six wrappers.
    @testset "vk_context through array wrappers" begin
        include(joinpath(@__DIR__, "test_vk_context_wrappers.jl"))
    end

    # The buffer-lifetime case behind the intermittent flush hang. Asserted on
    # the state machine, not by provoking the hang — see the file.
    @testset "free during recording" begin
        include(joinpath(@__DIR__, "test_free_during_recording.jl"))
    end

    # ── Previously unregistered test files ──
    #
    # These existed in test/ but were never included here. That is not neutral:
    # of the files found unregistered, three characterised bugs that had since been
    # FIXED without anyone noticing, one exercised an API deleted three months
    # earlier, one never waited on its own spawned task, and two had gone stale
    # against a refactor. Registered so they cannot rot again.
    #
    # Still deliberately out: the heavy stress/CI entry points.
    # (test_struct_alignment_systematic.jl was excluded here while the
    # whole-struct-copy bug was open; that is fixed, and it is registered above.)
    @testset "barrier elision" begin
        include(joinpath(@__DIR__, "test_barrier_elision.jl"))
    end
    @testset "broadcast paths" begin
        include(joinpath(@__DIR__, "test_broadcast_paths.jl"))
    end
    @testset "closest_hit via ray query" begin
        include(joinpath(@__DIR__, "test_closesthit_via_rayquery.jl"))
    end
    @testset "coopmat shared memory" begin
        include(joinpath(@__DIR__, "test_coopmat_shared.jl"))
    end
    @testset "batched coopmat GEMM" begin
        include(joinpath(@__DIR__, "test_gemm_batched.jl"))
    end
    @testset "HWAdaptedAccel via ray query" begin
        include(joinpath(@__DIR__, "test_hwadapted_via_rayquery.jl"))
    end
    @testset "hwtlas batch delete" begin
        include(joinpath(@__DIR__, "test_hwtlas_batch_delete.jl"))
    end
    @testset "hwtlas batch triangles" begin
        include(joinpath(@__DIR__, "test_hwtlas_batch_triangles.jl"))
    end
    @testset "hwtlas instance buffer" begin
        include(joinpath(@__DIR__, "test_hwtlas_instance_buffer.jl"))
    end
    @testset "hwtlas push instances" begin
        include(joinpath(@__DIR__, "test_hwtlas_push_instances.jl"))
    end
    @testset "hwtlas refit" begin
        include(joinpath(@__DIR__, "test_hwtlas_refit.jl"))
    end
    @testset "hwtlas sync batch" begin
        include(joinpath(@__DIR__, "test_hwtlas_sync_batch.jl"))
    end
    @testset "instance record" begin
        include(joinpath(@__DIR__, "test_instance_record.jl"))
    end
    @testset "permutedims" begin
        include(joinpath(@__DIR__, "test_permutedims.jl"))
    end
    @testset "phase 1 — pin/sync_access" begin
        include(joinpath(@__DIR__, "test_phase1_pin.jl"))
    end
    @testset "phase 2 — error surfacing" begin
        include(joinpath(@__DIR__, "test_phase2_errors.jl"))
    end
    @testset "phase 3 — lifecycle" begin
        include(joinpath(@__DIR__, "test_phase3_lifecycle.jl"))
    end
    @testset "phase 4 — single writer" begin
        include(joinpath(@__DIR__, "test_phase4_singlethread.jl"))
    end
    @testset "phase 5 — transfer path" begin
        include(joinpath(@__DIR__, "test_phase5_copy.jl"))
    end
    @testset "phase 6 — graphics dispatch" begin
        include(joinpath(@__DIR__, "test_phase6_graphics.jl"))
    end
    @testset "pool size classes" begin
        include(joinpath(@__DIR__, "test_pool_sizeclass.jl"))
    end
    @testset "source mapping" begin
        include(joinpath(@__DIR__, "test_source_mapping.jl"))
    end
    @testset "tlas allow_update" begin
        include(joinpath(@__DIR__, "test_tlas_allow_update.jl"))
    end
    @testset "tlas refit" begin
        include(joinpath(@__DIR__, "test_tlas_refit.jl"))
    end
    @testset "tlas refit integration" begin
        include(joinpath(@__DIR__, "test_tlas_refit_integration.jl"))
    end
    @testset "trace cull mask" begin
        include(joinpath(@__DIR__, "test_trace_cull_mask.jl"))
    end
    @testset "typepun trunc/bitcast" begin
        include(joinpath(@__DIR__, "test_typepun_trunc_bitcast.jl"))
    end

    # ── Tier 3d: SPIR-V emitter pattern correctness & stress (full only) ──
    #
    # Patterns here have each, at some point, miscompiled on a specific driver
    # (NVIDIA / RADV / AMDVLK / lavapipe). The pattern is the test identity, not
    # the vendor. Every test runs the same way on every platform — if it fails
    # somewhere new, that is a real correctness regression. There are no
    # vendor-conditional code paths in src/, and so there must be no
    # vendor-conditional tests here either.
    if FULL_MODE
        @testset "Tier 3d: SPIR-V Pattern Correctness & Stress" begin
            include(joinpath(@__DIR__, "test_spirv_pattern_correctness.jl"))
            include(joinpath(@__DIR__, "test_loop_unswitch_miscompile.jl"))
            include(joinpath(@__DIR__, "test_psb_chain_fold.jl"))
            include(joinpath(@__DIR__, "test_repeat_inner_3d.jl"))
            include(joinpath(@__DIR__, "test_multiindex_getindex.jl"))
        end
    end

    # ── Tier 3e: GPU Memory Safety (full only) ──
    if FULL_MODE
        @testset "Tier 3e: GPU Memory Safety" begin
            include(joinpath(@__DIR__, "test_gpu_memory_safety.jl"))
        end
    end

    # ── Tier 3e2: BAR-memory copy_buffer! sync regression (full only) ──
    if FULL_MODE
        @testset "Tier 3e2: BAR Memcpy Sync" begin
            include(joinpath(@__DIR__, "test_bar_memcpy_sync.jl"))
        end
    end

    # ── Tier 3e3: Raycore MultiTypeSet surgical-per-mutation invariants (full only) ──
    if FULL_MODE
        @testset "Tier 3e3: MultiTypeSet Surgical" begin
            include(joinpath(@__DIR__, "test_multitypeset_surgical.jl"))
        end
    end

    # ── Tier 3f: Caching & Allocations (full only) ──
    if FULL_MODE
        @testset "Tier 3f: Caching & Allocations" begin
            include(joinpath(@__DIR__, "test_caching_and_allocations.jl"))
        end
    end

    # ── Tier 3h: Disk Cache & Two-Tier Caching ──
    @testset "Tier 3h: Kernel Cache" begin
        include(joinpath(@__DIR__, "test_disk_cache.jl"))
    end

    # ── Tier 3g: Graphics Pipeline ──
    @testset "Tier 3g: Graphics Pipeline" begin
        include(joinpath(@__DIR__, "test_graphics_pipeline.jl"))
    end

    # ── Tier 4: GPUArrays TestSuite (~12 min) — full suite AND CI mode ──
    # The bulk of real correctness coverage. Runs on lavapipe CI with the few
    # process-crashing groups skipped (test_gpuarrays.jl LAVAPIPE_CRASH_SKIP).
    if FULL_MODE || CI_MODE
        @testset "Tier 4: GPUArrays TestSuite" begin
            include(joinpath(@__DIR__, "test_gpuarrays.jl"))
        end
    end

    # ── Tier 3i: HW TLAS (Lava.HWTLAS) — full only ──
    if FULL_MODE
        @testset "HW TLAS — stress + correctness" begin
            include(joinpath(@__DIR__, "test_hwtlas_stress.jl"))
        end

        @testset "HW TLAS — mesh update" begin
            include(joinpath(@__DIR__, "test_hwtlas_mesh_update.jl"))
        end

        @testset "HW TLAS — UAF safety" begin
            include(joinpath(@__DIR__, "test_hwtlas_uaf_safety.jl"))
        end

        @testset "pinned buffer lifetime" begin
            include(joinpath(@__DIR__, "test_pinned_buffer_lifetime.jl"))
        end

        @testset "workgroup zero-init" begin
            include(joinpath(@__DIR__, "test_workgroup_zero_init.jl"))
        end

        @testset "pool trim" begin
            include(joinpath(@__DIR__, "test_pool_trim.jl"))
        end

        @testset "frozen RT cache" begin
            include(joinpath(@__DIR__, "test_frozen_rt_cache.jl"))
        end

        @testset "Diagonal mul! disambiguation" begin
            include(joinpath(@__DIR__, "test_diagonal_mul.jl"))
        end

        @testset "device compute properties" begin
            include(joinpath(@__DIR__, "test_device_compute.jl"))
        end

        @testset "coopmat shape query" begin
            include(joinpath(@__DIR__, "test_coopmat_shape.jl"))
        end

        @testset "coopmat GEMM subgroup guard" begin
            include(joinpath(@__DIR__, "test_coopmat_gemm_subgroup.jl"))
        end

        @testset "subgroup shuffle family" begin
            include(joinpath(@__DIR__, "test_subgroup_shuffle.jl"))
        end

        @testset "coopmat per-element and component-wise ops" begin
            include(joinpath(@__DIR__, "test_coopmat_perelement.jl"))
        end

        @testset "coopmat component-wise add" begin
            include(joinpath(@__DIR__, "test_coopmat_add.jl"))
        end

        @testset "coopmat reductions (NV)" begin
            include(joinpath(@__DIR__, "test_coopmat_reduce.jl"))
        end

        # test_frozen_cache.jl shipped unregistered, so the compute-side frozen
        # cache had no coverage in CI. It restores FROZEN_VERSION by plain
        # assignment, so a throw part-way through would leave the cache ENABLED
        # for every later testset here — silently, since a frozen hit looks like
        # a normal launch. Restore it from a finally block instead.
        @testset "frozen kernel cache (compute)" begin
            _fv, _fr = Lava.FROZEN_VERSION[], Lava.FROZEN_RECORDING[]
            try
                include(joinpath(@__DIR__, "test_frozen_cache.jl"))
            finally
                Lava.FROZEN_VERSION[]   = _fv
                Lava.FROZEN_RECORDING[] = _fr
            end
        end

        @testset "HW TLAS — nonblocking sync!" begin
            include(joinpath(@__DIR__, "test_hwtlas_nonblocking_sync.jl"))
        end
    end

    # ── Tier 3i2: batch/pool ordering regressions ───────────────────────
    # Two latent races surfaced by Hikari's volpath bounce-loop early-exit
    # (mid-pipeline synchronize): arg-slab cursor reset while the active
    # batch holds recorded dispatches, and barrier elision between a
    # prepare-indirect and its own vkCmdDispatchIndirect inside a
    # concurrent_dispatch_group.
    @testset "Batch/pool ordering" begin
        @testset "arg-slab mid-recording sweep" begin
            include(joinpath(@__DIR__, "test_argslab_midrecording_sweep.jl"))
        end
        @testset "indirect in concurrent group" begin
            include(joinpath(@__DIR__, "test_indirect_in_concurrent_group.jl"))
        end
        include(joinpath(@__DIR__, "test_crossqueue_sync.jl"))
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

