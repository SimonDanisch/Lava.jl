# Tier 1: byte swap and bit reverse at every integer width.
#
# `llvm.bswap` had no lowering at all, and LLVM forms one from byte-shuffling code
# of its own accord: Hikari's Sobol sampler failed to compile on every Vulkan
# device. `llvm.bitreverse` was `OpBitReverse` at any width, which Vulkan allows
# on 32-bit integers only (VUID-RuntimeSpirv-None-10824), so `bitreverse` of a
# `UInt16` or a `UInt64` failed validation. The values are checked on a device by
# Mantle's `test_bit_intrinsics.jl`.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, compile_and_disasm

function bswap_kernel(A, B)
    i = Lava.lava_global_invocation_id_x()
    @inbounds B[i] = bswap(A[i])
    return nothing
end

function bitreverse_kernel(A, B)
    i = Lava.lava_global_invocation_id_x()
    @inbounds B[i] = bitreverse(A[i])
    return nothing
end

@testset "Bit intrinsics" begin
    for T in (UInt16, UInt32, UInt64)
        tt = Tuple{Lava.LavaDeviceArray{T, 1}, Lava.LavaDeviceArray{T, 1}}
        @testset "bswap of a $T is shifts and masks" begin
            d, _ = compile_and_disasm(bswap_kernel, tt)   # validated by spirv-val
            check(d, "OpShiftLeftLogical")
            check(d, "OpShiftRightLogical")
            check(d, "OpBitwiseAnd")
        end
        @testset "bitreverse of a $T reverses 32-bit integers only" begin
            d, _ = compile_and_disasm(bitreverse_kernel, tt)   # validated by spirv-val
            check(d, "OpBitReverse")
        end
    end
end
