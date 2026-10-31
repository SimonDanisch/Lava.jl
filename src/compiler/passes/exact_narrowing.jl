# LLVM pass: a float-to-half narrowing a driver would change is done in integers.
#
# `Float16(x)` rounds `x` to half precision, and Julia means that rounding of that
# value. Two drivers compute something else from the hardware conversion:
#
#   - NVIDIA drops the rounding when the narrowed value is used again in
#     registers. Measured on an RTX 4000 Ada (driver 595.99): `Float32(Float16(x))`
#     comes back as `x` in every element whatever the module says: `NoContraction`
#     on both conversions; `FPFastMathMode None` from
#     VK_KHR_shader_float_controls2, with the extension and feature enabled; the
#     same on the multiplies around them plus `FPFastMathDefault None` for both
#     float types; `RoundingModeRTE` 16/32; `DenormPreserve` 16; a bitcast round
#     trip through `i16`; the half through a `Volatile` Function variable; the
#     pair as two-lane vectors. DNNKernels' fused norm + rotary lost it where the
#     narrowed value fed a half multiply.
#   - RADV fuses a multiply into the conversion: `Float16(a * b)`, stored, is the
#     exact product rounded once, not `a * b` rounded to Float32 and then to
#     Float16 (Radeon 8060S, Mesa 26.x: 68 of 2^20 random products differ from
#     Julia, 0 from the single rounding), with `NoContraction` on the multiply and
#     on the conversion.
#   - The same NVIDIA fold reaches half arithmetic. InstCombine turns
#     `Float16(Float32(a) * Float32(b))` of two halves into `fmul half a, b` (the
#     product is exact in Float32, so that is the same value), and the widened
#     product, added to anything, came back unrounded on the Ada: 65166 of 65536
#     elements, and DNNKernels' SwiGLU fused into a ConvRot pass. So half
#     arithmetic used for more than storing is computed in Float32 and narrowed
#     here as well: one rounding of the Float32 result is the correctly rounded
#     half sum, difference and product, since 24 >= 2 * 11 + 2 bits; a quotient
#     is as exact as the device's Float32 division.
#
# What neither compiler sees through is a half built from integer operations: it
# is a 16-bit pattern like one loaded from memory. So a narrowing is spelled as
# the IEEE 754 round-to-nearest-even in integer operations, on every device, when
# either case can apply to it: its value is used for more than storing it, or its
# input is float arithmetic a conversion can absorb. A loaded value or a
# transcendental narrowed and stored stays one instruction.
#
# Doing EVERY narrowing in integers was measured and is too slow: SAM 2's encode +
# decode on the Radeon 8060S (benchmarks/jobs.jl, median of 20 per session, one
# fresh session each) took 160.1 and 160.1 ms before these passes, 169.4, 188.6
# and 176.5 ms with every narrowing in integers, and 155.7 and 167.3 ms with this
# rule. A flash-attention kernel narrows `exp` results inside its loop; a
# memory-bound kernel storing halves does not show the twenty integer operations.
#
# Lava's `test_exact_narrowing.jl` runs this pass's IR on the host against
# `Float16(::Float32)` for all 2^32 inputs.

"""
    narrow_in_integers!(mod::LLVM.Module) -> Bool

Replace every `fptrunc float to half` (scalar or vector) that is used for more
than storing, or whose input is float arithmetic, with the same rounding written
in integer operations on the float's bits; and every half add, subtract,
multiply or divide used for more than storing with the operation in Float32,
narrowed the same way. Returns whether anything changed.
"""
function narrow_in_integers!(mod::LLVM.Module)
    found = LLVM.Instruction[]
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        narrowable(inst) && push!(found, inst)
    end
    isempty(found) && return false
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        for t in found
            LLVM.position!(builder, LLVM.before(t))
            h = narrowed!(builder, t)
            LLVM.replace_uses!(t, h)
            LLVM.erase!(t)
        end
    end
    return true
end

const HalfArithmetic = Union{LLVM.FAddInst, LLVM.FSubInst, LLVM.FMulInst, LLVM.FDivInst}

"Whether `narrow_in_integers!` replaces `inst`."
narrowable(::LLVM.Instruction) = false
function narrowable(inst::LLVM.FPTruncInst)
    scalartype(LLVM.value_type(inst)) isa LLVM.HalfType || return false
    x = LLVM.operands(inst)[1]
    scalartype(LLVM.value_type(x)) isa LLVM.FloatType || return false
    return absorbable(x) || !onlystored(inst)
end
narrowable(inst::HalfArithmetic) =
    scalartype(LLVM.value_type(inst)) isa LLVM.HalfType && !onlystored(inst)

"`inst`'s half result, rounded in integer operations."
narrowed!(builder::LLVM.IRBuilder, t::LLVM.FPTruncInst) = narrow_bits!(builder, LLVM.operands(t)[1])
function narrowed!(builder::LLVM.IRBuilder, t::HalfArithmetic)
    F = shaped(LLVM.FloatType(), LLVM.value_type(t))
    a, b = (LLVM.fpext!(builder, x, F) for x in LLVM.operands(t))
    return narrow_bits!(builder, LLVM.binop!(builder, LLVM.opcode(t), a, b))
end

"""
Whether every use of `v` only puts its bits into memory: a store of it, or a
packing (insert, shuffle, bitcast, phi, select) whose own uses only store.
"""
onlystored(v::LLVM.Value, seen = Set{LLVM.Value}()) =
    all(u -> storing(LLVM.user(u), v, seen), LLVM.uses(v))
# The value stored, not the address stored to.
storing(u::LLVM.StoreInst, v, seen) = LLVM.operands(u)[1] == v
function storing(u::Union{LLVM.InsertElementInst, LLVM.InsertValueInst, LLVM.ShuffleVectorInst,
                          LLVM.BitCastInst, LLVM.PHIInst, LLVM.SelectInst}, v, seen)
    u in seen && return true
    push!(seen, u)
    return onlystored(u, seen)
end
storing(::LLVM.Value, v, seen) = false

"""
Whether `x` is float arithmetic a driver can fold into the conversion that
narrows it: an add, subtract, multiply, divide (a multiply by a reciprocal once
lowered) or fused multiply-add, through a negation or absolute value, which
conversion instructions take as modifiers.
"""
absorbable(::LLVM.Value) = false
absorbable(x::LLVM.Instruction) =
    LLVM.opcode(x) in (LLVM.API.LLVMFAdd, LLVM.API.LLVMFSub, LLVM.API.LLVMFMul, LLVM.API.LLVMFDiv) ||
    (LLVM.opcode(x) == LLVM.API.LLVMFNeg && absorbable(LLVM.operands(x)[1]))
function absorbable(x::LLVM.CallInst)
    callee = LLVM.called_operand(x)
    callee isa LLVM.Function || return false
    n = LLVM.name(callee)
    startswith(n, "llvm.fma.") || startswith(n, "llvm.fmuladd.") ||
        (startswith(n, "llvm.fabs.") && absorbable(LLVM.operands(x)[1]))
end

scalartype(t::LLVM.VectorType) = eltype(t)
scalartype(t::LLVM.LLVMType) = t

"`s` in the shape of `like`: itself for a scalar, a vector of as many lanes."
shaped(s::LLVM.LLVMType, ::LLVM.LLVMType) = s
shaped(s::LLVM.LLVMType, like::LLVM.VectorType) = LLVM.VectorType(s, length(like))

splatted(t::LLVM.IntegerType, x::Integer) = LLVM.ConstantInt(t, x)
splatted(t::LLVM.VectorType, x::Integer) = LLVM.const_splat(t, splatted(eltype(t), x))

"""
    narrow_bits!(builder, x) -> LLVM.Value

`fptrunc x to half`, rounding to nearest even, from `x`'s bits. Three results,
one selected by the magnitude `f`: a normal half (`f` rebiased, rounded by adding
`0xfff` plus the kept fraction's lowest bit, which also carries into the exponent
and up to infinity), a subnormal or zero (the fraction with its hidden bit,
shifted right by `126 - e` and rounded on the dropped bits), and infinity or a
quiet NaN with the top of the payload.
"""
function narrow_bits!(builder::LLVM.IRBuilder, x::LLVM.Value)
    xt = LLVM.value_type(x)
    I32 = shaped(LLVM.IntType(32), xt)
    k(c) = splatted(I32, c)
    and(a, b) = LLVM.and!(builder, a, b)
    or(a, b) = LLVM.or!(builder, a, b)
    add(a, b) = LLVM.add!(builder, a, b)
    sub(a, b) = LLVM.sub!(builder, a, b)
    shr(a, b) = LLVM.lshr!(builder, a, b)
    shl(a, b) = LLVM.shl!(builder, a, b)
    sel(c, a, b) = LLVM.select!(builder, c, a, b)
    cmp(p, a, b) = LLVM.icmp!(builder, p, a, b)

    b = LLVM.bitcast!(builder, x, I32)
    sign = and(shr(b, k(16)), k(0x8000))
    f = and(b, k(0x7fff_ffff))

    normal = shr(add(add(sub(f, k(112 << 23)), k(0x0fff)), and(shr(f, k(13)), k(1))), k(13))

    e = shr(f, k(23))
    # Lanes this result is not selected for still must not shift by 32 or more.
    s = sel(cmp(LLVM.API.LLVMIntULT, e, k(95)), k(31), and(sub(k(126), e), k(31)))
    m = or(and(f, k(0x007f_ffff)), k(0x0080_0000))
    q = shr(m, s)
    r = and(m, sub(shl(k(1), s), k(1)))
    halfway = shl(k(1), and(sub(s, k(1)), k(31)))
    up = or(cmp(LLVM.API.LLVMIntUGT, r, halfway),
            LLVM.and!(builder, cmp(LLVM.API.LLVMIntEQ, r, halfway),
                      cmp(LLVM.API.LLVMIntEQ, and(q, k(1)), k(1))))
    subnormal = add(q, LLVM.zext!(builder, up, I32))

    special = sel(cmp(LLVM.API.LLVMIntUGT, f, k(0x7f80_0000)),
                  or(k(0x7e00), and(shr(f, k(13)), k(0x03ff))), k(0x7c00))
    magnitude = sel(cmp(LLVM.API.LLVMIntUGE, f, k(143 << 23)), special,
                    sel(cmp(LLVM.API.LLVMIntULT, f, k(113 << 23)), subnormal, normal))
    bits16 = LLVM.trunc!(builder, or(magnitude, sign), shaped(LLVM.IntType(16), xt))
    return LLVM.bitcast!(builder, bits16, shaped(LLVM.HalfType(), xt))
end
