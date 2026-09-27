# Tier 1: `!llvm.loop` unroll hints reach `OpLoopMerge`'s Loop Control.
#
# Lava wrote `None` on every loop until 2026-09-27, so a Vulkan driver decided
# the unroll alone. RADV fully unrolled the implicit-GEMM convolution's FMA loop
# (16 steps of 64 FMAs) and the kernel went from 120 VGPRs to 156, and from
# 281 ms to 324 on the Qwen-Image VAE's largest convolution. The hint is written
# the way Julia writes any loop hint, `Expr(:loopinfo, …)`, and it has to
# survive StructurizeCFG, which replaces the latch branch that carries it.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_count, compile_and_disasm

const LOOPCTL_TT = Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.LavaDeviceArray{Float32,1}}

# One kernel per hint, over the same 16-step dependent chain. `nothing` is no hint.
for (name, hint) in ((:loopctl_none, nothing),
                     (:loopctl_disable, (Symbol("llvm.loop.unroll.disable"),)),
                     (:loopctl_disable_ka, (Symbol("llvm.loop.unroll.disable"), 1)),
                     (:loopctl_full, (Symbol("llvm.loop.unroll.full"),)),
                     (:loopctl_count4, (Symbol("llvm.loop.unroll.count"), 4)),
                     (:loopctl_count1, (Symbol("llvm.loop.unroll.count"), 1)))
    info = hint === nothing ? nothing : Expr(:loopinfo, hint)
    @eval function $name(out, x)
        i = Lava.lava_global_invocation_id_x()
        acc = 0f0
        for k in 1:16
            @inbounds acc = muladd(x[k], acc, 1f0)
            $info
        end
        @inbounds out[i] = acc
        return nothing
    end
end

# A loop whose header is not its latch: the `while` test is its own block.
@eval function loopctl_while_disable(out, x)
    i = Lava.lava_global_invocation_id_x()
    acc = 0f0
    k = 1
    while k <= 16
        @inbounds v = x[k]
        v > 100f0 && break
        acc = muladd(v, acc, 1f0)
        k += 1
        $(Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
    end
    @inbounds out[i] = acc
    return nothing
end

@testset "Loop Control from llvm.loop hints" begin
    @testset "no hint: LLVM unrolls the short loop, nothing to mark" begin
        d, _ = compile_and_disasm(loopctl_none, LOOPCTL_TT)
        check_not(d, "OpLoopMerge")
        check_count(d, " Fma ", 16)
    end

    # The header IS the latch here, so this is the case StructurizeCFG's
    # replacement of the latch branch would lose.
    @testset "unroll.disable on a single-block loop -> DontUnroll" begin
        d, _ = compile_and_disasm(loopctl_disable, LOOPCTL_TT)
        check_count(d, "OpLoopMerge", 1)
        check(d, "DontUnroll")
        check_count(d, " Fma ", 1)
    end

    @testset "KernelAbstractions' spelling, with an operand -> DontUnroll" begin
        d, _ = compile_and_disasm(loopctl_disable_ka, LOOPCTL_TT)
        check(d, "DontUnroll")
    end

    @testset "unroll.disable on a loop with a separate latch -> DontUnroll" begin
        d, _ = compile_and_disasm(loopctl_while_disable, LOOPCTL_TT)
        @test count("OpLoopMerge", d) >= 1
        check(d, "DontUnroll")
    end

    @testset "unroll.full: LLVM unrolls it, no loop left" begin
        d, _ = compile_and_disasm(loopctl_full, LOOPCTL_TT)
        check_not(d, "OpLoopMerge")
    end

    # LLVM applies an explicit count itself and marks the result
    # `unroll.disable`, so the driver must not unroll the four-wide body again.
    @testset "unroll.count 4: unrolled by LLVM, then DontUnroll" begin
        d, _ = compile_and_disasm(loopctl_count4, LOOPCTL_TT)
        check_count(d, "OpLoopMerge", 1)
        check(d, "DontUnroll")
        check_count(d, " Fma ", 4)
    end

    @testset "unroll.count 1 -> DontUnroll" begin
        d, _ = compile_and_disasm(loopctl_count1, LOOPCTL_TT)
        check(d, "DontUnroll")
    end

    # Every loop in these kernels carries a hint, so one without must still be
    # emitted as `None` rather than inheriting a neighbour's.
    @testset "an unhinted loop stays None" begin
        function loopctl_runtime(out, x, n)
            i = Lava.lava_global_invocation_id_x()
            acc = 0f0
            for k in Int32(1):n
                @inbounds acc = muladd(x[k], acc, 1f0)
            end
            @inbounds out[i] = acc
            return nothing
        end
        d, _ = compile_and_disasm(loopctl_runtime, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                          Lava.LavaDeviceArray{Float32,1}, Int32})
        check(d, "OpLoopMerge")
        check_not(d, "DontUnroll")
        check_not(d, " Unroll")
    end

    # `unroll.count N > 1` only reaches the emitter when LLVM declined to apply
    # it, which Julia source cannot force, so the mapping is checked directly.
    @testset "loop_control_words" begin
        Lava.LLVM.Context() do _
            L = Lava.LLVM
            loopid(ops...) = L.MDNode([L.MDNode(collect(L.Metadata, ops))])
            str(s) = L.MDString(s)
            int(n) = L.Metadata(L.ConstantInt(Int32(n)))
            @test Lava.loop_control_words(loopid(str("llvm.loop.unroll.disable"))) == [0x2]
            @test Lava.loop_control_words(loopid(str("llvm.loop.unroll.enable"))) == [0x1]
            @test Lava.loop_control_words(loopid(str("llvm.loop.unroll.full"))) == [0x1]
            @test Lava.loop_control_words(loopid(str("llvm.loop.unroll.count"), int(4))) == [0x100, 0x4]
            @test Lava.loop_control_words(loopid(str("llvm.loop.unroll.count"), int(1))) == [0x2]
            @test Lava.loop_control_words(loopid(str("llvm.loop.mustprogress"))) === nothing
        end
    end
end
