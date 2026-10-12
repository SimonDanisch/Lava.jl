# Tier 1b: `narrow_in_integers!`'s rounding is `Float16(::Float32)`, for every input.
#
# The pass replaces a narrowing the kernel keeps computing with by integer
# operations (see `src/compiler/passes/exact_narrowing.jl` for why). Those
# operations are built here into a one-function LLVM module, run on the host
# through `llvmcall`, and compared with Julia's conversion on all 2^32 Float32
# bit patterns: the IR the pass emits, not a transcription of it. NaNs compare by
# being NaN. The device side, the same sequence after SPIR-V, is Mantle's
# `test_float_conversion_exact.jl`.

using Test
using Lava
import LLVM

"The pass's narrowing as LLVM IR: `define i16 @narrow(float)`, the half's bits."
function narrowing_ir()
    LLVM.@dispose ctx = LLVM.Context() begin
        mod = LLVM.Module("narrow")
        fn = LLVM.Function(mod, "narrow", LLVM.FunctionType(LLVM.IntType(16), [LLVM.FloatType()]))
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline"))
        entry = LLVM.BasicBlock(fn, "entry")
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            LLVM.position!(builder, LLVM.at_end(entry))
            h = Lava.narrow_bits!(builder, only(LLVM.parameters(fn)))
            LLVM.ret!(builder, LLVM.bitcast!(builder, h, LLVM.IntType(16)))
        end
        string(mod)
    end
end

@eval narrowed(x::Float32) = Base.llvmcall(($(narrowing_ir()), "narrow"), UInt16, Tuple{Float32}, x)

"How many of the 2^32 inputs `narrowed` gets wrong, one high half-word per task."
function narrowing_errors()
    bad = Threads.Atomic{Int}(0)
    Threads.@threads for hi in UInt32(0):UInt32(0xffff)
        n = 0
        for lo in UInt32(0):UInt32(0xffff)
            x = reinterpret(Float32, (hi << 16) | lo)
            got = reinterpret(Float16, narrowed(x))
            n += !(isnan(x) ? isnan(got) : got === Float16(x))
        end
        Threads.atomic_add!(bad, n)
    end
    return bad[]
end

@testset "the pass narrows like Float16(::Float32)" begin
    # The edges by name first, so a failure says which.
    for x in (0f0, -0f0, 1f0, -1f0, 65504f0, 65519.996f0, 65520f0, -65520f0, Inf32, -Inf32,
              floatmin(Float16) |> Float32, 2f0^-24, 2f0^-25, nextfloat(2f0^-25), 3f0 * 2f0^-26,
              nextfloat(0f0), floatmin(Float32), 1f0 + 2f0^-11, 1f0 + 3f0 * 2f0^-11)
        @test reinterpret(Float16, narrowed(x)) === Float16(x)
    end
    @test isnan(reinterpret(Float16, narrowed(NaN32)))
    @test narrowing_errors() == 0
end
