using Test, Lava
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, compile_and_disasm

const PRIVATE_FLOAT_PAIRS = ntuple(i -> (Float32(i), Float32(i + 13)), 24)

function loop_private_float_pairs!(dst)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    s = 0f0
    @inbounds for (x, y) in PRIVATE_FLOAT_PAIRS
        s += x * y * dst[i]
        i += 1
    end
    @inbounds dst[1] = s
    return nothing
end

@testset "packed loads from a private tuple table" begin
    # The iteration phi carries the pair as a single i64. The load itself
    # still has to use the element type of the private float array.
    args = Tuple{Lava.HeapDeviceArray{Float32,1}}
    disasm, _ = compile_and_disasm(loop_private_float_pairs!, args;
                                   features = Lava.TargetFeatures(webgpu = true))
    check(disasm, "OpLoad %_arr_float")
    check(disasm, "OpCompositeExtract %float")
end
