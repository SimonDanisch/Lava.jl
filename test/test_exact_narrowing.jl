# Tier 1b: `narrow_in_integers!`'s rounding is `Float16(::Float32)`, for every input.
#
# The pass replaces a narrowing the kernel keeps computing with by integer
# operations (see `src/compiler/passes/exact_narrowing.jl` for why). Those
# operations are built here into a one-function LLVM module, run on the host
# through `llvmcall`, and compared with Julia's conversion on all 2^32 Float32
# bit patterns: the IR the pass emits, not a transcription of it. NaNs compare by
# being NaN. The device side, the same sequence after SPIR-V, is Mantle's
# `test_float_conversion_exact.jl`.

using Test, Random
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

# Half arithmetic used for more than storing is computed in Float32 and narrowed
# by the same sequence (`narrowed!`): one rounding of the Float32 result is the
# correctly rounded half result, which is what Julia's Float16 operations
# compute. The edges against each other, then random pairs.
"The pass's half `op` as LLVM IR: `define i16 @halfop(i16, i16)`, the result's bits."
function half_arithmetic_ir(op)
    LLVM.@dispose ctx = LLVM.Context() begin
        mod = LLVM.Module("halfop")
        i16, half = LLVM.IntType(16), LLVM.HalfType()
        fn = LLVM.Function(mod, "halfop", LLVM.FunctionType(i16, [i16, i16]))
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline"))
        entry = LLVM.BasicBlock(fn, "entry")
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            LLVM.position!(builder, LLVM.at_end(entry))
            a, b = (LLVM.bitcast!(builder, p, half) for p in LLVM.parameters(fn))
            t = LLVM.binop!(builder, op, a, b)
            h = Lava.narrowed!(builder, t)
            LLVM.erase!(t)
            LLVM.ret!(builder, LLVM.bitcast!(builder, h, i16))
        end
        string(mod)
    end
end

const HALF_OPS = ((+, LLVM.API.LLVMFAdd), (-, LLVM.API.LLVMFSub), (*, LLVM.API.LLVMFMul), (/, LLVM.API.LLVMFDiv))
for (f, op) in HALF_OPS
    @eval halfop(::typeof($f), a::Float16, b::Float16) = reinterpret(Float16,
        Base.llvmcall(($(half_arithmetic_ir(op)), "halfop"), UInt16, Tuple{UInt16, UInt16},
                      reinterpret(UInt16, a), reinterpret(UInt16, b)))
end

@testset "half arithmetic is narrowed like Julia's Float16 operations" begin
    same(a, b) = isnan(a) ? isnan(b) : a === b
    edges = Float16[0, -0.0, 1, -1, 0.5, 3, 65504, -65504, floatmin(Float16), nextfloat(Float16(0)),
                    prevfloat(floatmin(Float16)), 1 + eps(Float16), Inf16, -Inf16, NaN16, 2049, 0.1]
    rng = Random.MersenneTwister(16)
    as = reinterpret(Float16, rand(rng, UInt16, 1 << 20)); bs = reinterpret(Float16, rand(rng, UInt16, 1 << 20))
    for (f, _) in HALF_OPS
        @test all(same(halfop(f, a, b), f(a, b)) for a in edges, b in edges)
        @test count(i -> !same(halfop(f, as[i], bs[i]), f(as[i], bs[i])), eachindex(as)) == 0
    end
end
