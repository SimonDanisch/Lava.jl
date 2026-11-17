# Tier 1b: `lower_wide_divisions!`'s division by a constant is Julia's `÷` and `%`.
#
# The sequence `divide_by_constant!` emits (shifts for a power of two, the
# multiply-high otherwise) is built into a one-function LLVM module per operation
# and divisor, run on the host through `llvmcall`, and compared with Julia over
# the edges of each path and random bits. The device side is Mantle's
# `test_integer_division.jl`.

using Test, Random
using Lava
import LLVM

const DIVISION_OPS = (sdiv = LLVM.API.LLVMSDiv, srem = LLVM.API.LLVMSRem,
                      udiv = LLVM.API.LLVMUDiv, urem = LLVM.API.LLVMURem)

"The pass's `a op d` as LLVM IR: `define i64 @divide(i64)`."
function division_ir(op, d::Int64)
    LLVM.@dispose ctx = LLVM.Context() begin
        mod = LLVM.Module("divide")
        i64 = LLVM.IntType(64)
        fn = LLVM.Function(mod, "divide", LLVM.FunctionType(i64, [i64]))
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline"))
        entry = LLVM.BasicBlock(fn, "entry")
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            LLVM.position!(builder, LLVM.at_end(entry))
            q = Lava.divide_by_constant!(builder, op, only(LLVM.parameters(fn)), LLVM.ConstantInt(i64, d))
            LLVM.ret!(builder, q)
        end
        string(mod)
    end
end

const DIVIDED = Dict{Tuple{Symbol,Int64},Function}()
"`x op d` through the pass's IR, compiled once per operation and divisor."
function divided(op::Symbol, d::Int64, x::Int64)
    f = get!(DIVIDED, (op, d)) do
        @eval x -> Base.llvmcall(($(division_ir(DIVISION_OPS[op], d)), "divide"), Int64, Tuple{Int64}, x)
    end
    return Base.invokelatest(f, x)
end

"Dividends: small, both sides of 2^32 and 2^63, the extremes, and random bits."
function division_inputs(rng)
    edges = Int64[0, 1, 2, 3, 4, 5, 7, 8, 95, 96, 97, 1000, 2^31 - 1, 2^31, 2^32 - 1, 2^32, 2^32 + 1,
                  2^62, typemax(Int64), typemax(Int64) - 1]
    return unique([edges; -edges; typemin(Int64); typemin(Int64) + 1; -2000:2000;
                   rand(rng, Int64, 2000); rand(rng, Int64, 2000) .>> rand(rng, 0:63, 2000)])
end

# Julia's operation on the bits the IR sees.
reference(op::Symbol, x::Int64, d::Int64) =
    op === :sdiv ? x ÷ d : op === :srem ? x % d :
    reinterpret(Int64, op === :udiv ? reinterpret(UInt64, x) ÷ reinterpret(UInt64, d) :
                                      reinterpret(UInt64, x) % reinterpret(UInt64, d))

@testset "division by a constant is Julia's" begin
    x = division_inputs(MersenneTwister(14))
    powers = [2^k for k in (1, 2, 3, 4, 5, 16, 31, 32, 33, 48, 62)]
    signed = [powers; -powers; [3, -3, 7, 96, -96, 641, 10^12]]
    unsigned = [powers; [typemin(Int64), 3, 7, 96, 641, 10^12, -1]]   # typemin, -1: 2^63, 2^64 - 1
    for (ops, ds) in (((:sdiv, :srem), signed), ((:udiv, :urem), unsigned)), op in ops, d in ds
        xs = filter(a -> !(op in (:sdiv, :srem) && d == -1 && a == typemin(Int64)), x)
        bad = count(a -> divided(op, d, a) != reference(op, a, d), xs)
        @test (op, d, bad) == (op, d, 0)
    end
end
