# `KI.dot4x8` on Lava is `OpSDotKHR` on two packed words. KernelInterface's own
# definition is four sign extensions and multiply-adds, which RADV does not turn
# back into `v_dot4`, so the instruction has to be in the module, with the
# capabilities and the extension SPIR-V 1.4 needs for it. `compile_and_disasm`
# runs the validator, so an operand the instruction does not take fails here.

using Test
import KernelInterface as KI
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, compile_and_disasm

@testset "packed integer dot product" begin
    function dot4x8_kernel(y, a, b)
        i = Lava.lava_global_invocation_id_x() + UInt32(1)
        @inbounds y[i] = KI.dot4x8(a[i], b[i])
        return nothing
    end
    d, _ = compile_and_disasm(dot4x8_kernel, Tuple{Lava.LavaDeviceArray{Int32,1},
                                                   Lava.LavaDeviceArray{UInt32,1},
                                                   Lava.LavaDeviceArray{UInt32,1}})
    check(d, "OpSDot")
    check(d, "PackedVectorFormat4x8Bit")
    check(d, "OpCapability DotProductInput4x8BitPacked")
    check(d, "OpCapability DotProduct\n")
    check(d, "OpExtension \"SPV_KHR_integer_dot_product\"")
end
