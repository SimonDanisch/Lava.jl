# Tier 1: a run-time index into a tuple whose elements wrap a device array.
#
# A KernelAbstractions kernel that reads `ml[l][T]` from a tuple of device arrays
# gets the tuple spilled to the stack as `[N x [1 x {ptr, [1 x i64]}]]` and steps
# it in bytes, `(l - 1) * 16`. The emitter maps the offset back onto the type:
# array index `l - 1`, then index 0 of the `[1 x ...]` around each array. A chain
# that stops above that level is typed as the struct but points at the
# one-element array, which spirv-val rejects ("OpAccessChain result type
# (OpTypeStruct) does not match"). BiotSavartBCs' FMM kernel reads its multigrid
# levels this way.
#
# A plain `NTuple{N,LavaDeviceArray}` spills as `[N x {ptr, [1 x i64]}]` and
# never had the wrapper; a tuple of one-element tuples has it, so that is what
# this compiles. The values are checked on a device in Mantle's
# `test_dynamic_tuple_getindex.jl`.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_regex, compile_and_disasm

function wrapped_level_read!(out, idx, ml)
    i = Lava.lava_global_invocation_id_x()
    (l, T) = @inbounds idx[i]
    @inbounds out[i] = ml[l][1][T]
    return nothing
end

function wrapped_level_store!(ml, idx, vals)
    i = Lava.lava_global_invocation_id_x()
    (l, T) = @inbounds idx[i]
    @inbounds ml[l][1][T] = vals[i]
    return nothing
end

const WRAPPED_LEVELS = NTuple{7,Tuple{Lava.LavaDeviceArray{Float32,1}}}
const LEVEL_INDEX = Lava.LavaDeviceArray{Tuple{Int64,Int64},1}

@testset "dynamic index into a tuple of wrapped arrays" begin
    @testset "read validates" begin
        d, _ = compile_and_disasm(wrapped_level_read!,
                                  Tuple{Lava.LavaDeviceArray{Float32,1}, LEVEL_INDEX, WRAPPED_LEVELS})
        # Down to the array struct: the dynamic level, then the wrapper's 0.
        check_regex(d, raw"OpAccessChain %_ptr_Function__struct_\d+ %\d+ %\d+ %uint_0")
        check(d, "OpLoad %float")
    end
    @testset "store validates" begin
        d, _ = compile_and_disasm(wrapped_level_store!,
                                  Tuple{WRAPPED_LEVELS, LEVEL_INDEX, Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpStore")
    end
end
