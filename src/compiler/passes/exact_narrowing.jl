# LLVM pass: a narrowing the kernel keeps computing with is done in integers.
#
# `Float16(x)` rounds `x` to half precision, and Julia means the rounding. NVIDIA's
# compiler drops it whenever the rounded value is used again in registers.
# Measured on an RTX 4000 Ada (driver 595.99):
#
#   - `Float32(Float16(x))` comes back as `x` in every element, whatever the
#     module says: `NoContraction` on both conversions (what Lava emits, and what
#     Mesa honours); `FPFastMathMode None` from VK_KHR_shader_float_controls2,
#     with the extension and feature enabled; the same on the multiplies around
#     them plus `FPFastMathDefault None` for both float types; `RoundingModeRTE`
#     16/32; `DenormPreserve` 16; a bitcast round trip through `i16`; the half
#     through a `Volatile` Function variable; the pair as two-lane vectors.
#   - DNNKernels' fused head norm + rotary (`test_fusermsrope.jl`) differs from
#     the same chain run as separate kernels, where the narrowed value feeds a
#     half multiply whose product is widened, after the direct pair above was
#     already fixed.
#
# A narrowing that only goes to memory is rounded correctly, and so is a half
# product of two loaded halves widened. What the compiler cannot see through is a
# half built from integer operations: it is a 16-bit pattern like one loaded from
# memory. So a narrowing whose value is computed with (widened, multiplied,
# compared...) is spelled as the IEEE 754 round-to-nearest-even in integers, on
# every device. `narrowbits` in Lava's `test_exact_narrowing.jl` is the same
# sequence in Julia and matches `Float16(::Float32)` on all 2^32 inputs; a
# narrowing stored straight to memory stays one instruction.

"""
    narrow_in_integers!(mod::LLVM.Module) -> Bool

Replace every `fptrunc float to half` (scalar or vector) whose value is used for
anything but storing it with the same rounding written in integer operations on
the float's bits. Returns whether anything changed.
"""
function narrow_in_integers!(mod::LLVM.Module)
    found = LLVM.FPTruncInst[]
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        inst isa LLVM.FPTruncInst || continue
        scalartype(LLVM.value_type(inst)) isa LLVM.HalfType || continue
        scalartype(LLVM.value_type(LLVM.operands(inst)[1])) isa LLVM.FloatType || continue
        onlystored(inst) || push!(found, inst)
    end
    isempty(found) && return false
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        for t in found
            LLVM.position!(builder, LLVM.before(t))
            h = narrow_bits!(builder, LLVM.operands(t)[1])
            LLVM.replace_uses!(t, h)
            LLVM.erase!(t)
        end
    end
    return true
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
