# Lava's test suite: a Julia→SPIR-V compiler, tested without a GPU.
#
# This file drove 136 includes across three tiers until 2026-08-27, when the
# Vulkan runtime moved to `Mantle/src/vulkan/`. 128 of them went with it, to
# `Mantle/test/vulkan/` — every test that needed a device, a queue, a pool or a
# `LavaArray`. What is left needs none of those, which is the point: **this suite
# runs on a machine with no Vulkan driver.**
#
#   Tier 1   SPIR-V emission. Compile a kernel, disassemble it, check the
#            instructions. `spirv/` holds one file per stage and feature.
#   Tier 1b  Compiler IR passes, on LLVM IR, before any SPIR-V exists.
#   Tier 2   Validation and structure: every builtin through `spirv-val`, the
#            emitter's capability decisions, and the compiler's own frozen cache.
#
# There is no Tier 3. It was GPU execution, and it is Mantle's now.
#
# `compile_and_disasm` reaches `spirv-val` and `spirv-dis` through
# `SPIRV_Tools_jll`, which are binaries rather than a driver — that is what makes
# a compiler suite runnable in CI on a box with no GPU.

using Test
using Lava

# Loaded once; each test file guards with `@isdefined(SPIRVTestUtils)` so it also
# runs standalone, which is how a file gets run while a feature is being added.
include(joinpath(@__DIR__, "spirv_test_utils.jl"))
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count,
    check_regex, normalize_spirv, compare_golden, compile_and_disasm,
    spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

@testset "Lava.jl" begin

    # ── Tier 1: SPIR-V emission ──────────────────────────────────────────────
    #
    # One file per stage and feature: compute, vertex, fragment, geometry,
    # tessellation, raygen, closesthit, anyhit, miss, ray query, math
    # intrinsics, control flow, memory, types. Each compiles a kernel and checks
    # the disassembly, so a wrong instruction fails here rather than on a device.
    @testset "Tier 1: SPIR-V Emission" begin
        spirv_dir = joinpath(@__DIR__, "spirv")
        for f in sort(readdir(spirv_dir; join = true))
            endswith(f, ".jl") || continue
            @info "Running $(basename(f))..."
            include(f)
        end
    end

    # ── Tier 1b: compiler IR passes ──────────────────────────────────────────
    # On LLVM IR, before any SPIR-V exists.
    @testset "Tier 1b: Compiler IR passes" begin
        include(joinpath(@__DIR__, "test_replace_unreachable.jl"))
    end

    # ── Tier 2: validation and structure ─────────────────────────────────────

    # An `llvmcall` intrinsic emits nothing until a kernel calls it, so a builtin
    # with no caller has never been through spirv-val. That is not hypothetical:
    # `lava_workgroup_size` emits a module Vulkan rejects and 26955 passing tests
    # said nothing, because not one built a module containing it.
    @testset "SPIR-V builtin validation" begin
        include(joinpath(@__DIR__, "test_builtin_validation.jl"))
    end

    # What the emitter may declare, and who tells it. The compiler used to read
    # `VK_CONTEXT_REF[]` for two booleans; it reads a record the runtime pushes,
    # which is the last thing that had to go before the move was possible.
    # Asserted on the emitted SPIR-V: flip the record, the capability changes.
    @testset "target features" begin
        include(joinpath(@__DIR__, "test_target_features.jl"))
    end

    # The boundary itself, from the source: no file in this package names
    # `Vulkan`, `VkContext` or `vk_context` in code, and the package does not
    # depend on Vulkan.jl. Costs no device time and is what keeps the split real.
    @testset "compiler / runtime split" begin
        include(joinpath(@__DIR__, "test_compiler_runtime_split.jl"))
    end

    # The ray-tracing half of the frozen cache takes no context — that asymmetry
    # is what let it stay with the compiler when the compute half left.
    @testset "frozen RT cache" begin
        include(joinpath(@__DIR__, "test_frozen_rt_cache.jl"))
    end

    # SPV_NV_tensor_addressing layout emission.
    @testset "tensor opcodes" begin
        include(joinpath(@__DIR__, "test_tensor_opcodes.jl"))
    end

    # Two fixed per-kernel overheads that were pure waste. Pinned so they do not
    # come back; measures compile time, not run time, so no device is involved.
    @testset "compile overhead" begin
        include(joinpath(@__DIR__, "test_compile_overhead.jl"))
    end
end
