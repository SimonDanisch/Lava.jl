# LLVM pass: widen a value that was just narrowed, in integer operations.
#
# `Float32(Float16(x))` rounds `x` to half precision, and Julia means the
# rounding. NVIDIA's compiler drops it: a narrowing `OpFConvert` followed by a
# widening one comes back as `x`. Measured on an RTX 4000 Ada (driver 595.99),
# where every one of these still returned the unrounded value in every element:
#
#   - `NoContraction` on both conversions (what Lava emits, and what Mesa honours)
#   - `FPFastMathMode None` on both, from VK_KHR_shader_float_controls2, with the
#     extension and feature enabled
#   - the same on the multiplies around them, and `FPFastMathDefault None` for
#     both float types
#   - `RoundingModeRTE` for 16 and 32 bit, `DenormPreserve` for 16 bit
#   - a bitcast round trip through `i16` between the two conversions
#   - the half stored to and loaded from a Function variable, both `Volatile`
#   - the pair as two-lane vector conversions
#
# A narrowing that is stored is rounded correctly, and so is a half product
# widened (`Float32(a * b)` with half `a` and `b`): only the pair is folded. What
# the compiler does not see through is integer arithmetic on the narrowed BITS,
# so the widening is spelled that way, on every device. It is exact by
# construction (Mantle's `test_float_conversion_exact.jl` sweeps all 65,536 half
# patterns through it), and costs a dozen integer operations where a kernel
# rounds through a narrower type on purpose, which is rare.

"""
    widen_narrowed_floats!(mod::LLVM.Module) -> Bool

Replace every `fpext` whose operand was narrowed by an `fptrunc` (directly, or
through `phi` and `select`) with the IEEE 754 widening written in integer
operations on the narrowed value's bits. Returns whether anything changed.
"""
function widen_narrowed_floats!(mod::LLVM.Module)
    exts = LLVM.FPExtInst[]
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        inst isa LLVM.FPExtInst || continue
        narrowed(LLVM.operands(inst)[1], Set{LLVM.Value}()) && push!(exts, inst)
    end
    isempty(exts) && return false
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        for ext in exts
            LLVM.position!(builder, LLVM.before(ext))
            wide = widen_bits!(builder, LLVM.operands(ext)[1], LLVM.value_type(ext))
            LLVM.replace_uses!(ext, wide)
            LLVM.erase!(ext)
        end
    end
    return true
end

"Whether `v` is an `fptrunc`, or a `phi`/`select` one of whose inputs is."
narrowed(::LLVM.FPTruncInst, seen) = true
narrowed(::LLVM.Value, seen) = false
function narrowed(v::LLVM.PHIInst, seen)
    v in seen && return false
    push!(seen, v)
    return any(((x, _),) -> narrowed(x, seen), LLVM.incoming(v))
end
function narrowed(v::LLVM.SelectInst, seen)
    v in seen && return false
    push!(seen, v)
    ops = LLVM.operands(v)
    return narrowed(ops[2], seen) || narrowed(ops[3], seen)
end

"Total and fraction bits of an IEEE 754 binary format."
ieeeformat(::LLVM.HalfType) = (bits = 16, fraction = 10)
ieeeformat(::LLVM.BFloatType) = (bits = 16, fraction = 7)
ieeeformat(::LLVM.FloatType) = (bits = 32, fraction = 23)
ieeeformat(::LLVM.DoubleType) = (bits = 64, fraction = 52)

scalartype(t::LLVM.VectorType) = eltype(t)
scalartype(t::LLVM.LLVMType) = t

"`s` in the shape of `like`: itself for a scalar, a vector of as many lanes."
shaped(s::LLVM.LLVMType, ::LLVM.LLVMType) = s
shaped(s::LLVM.LLVMType, like::LLVM.VectorType) = LLVM.VectorType(s, length(like))

splatted(t::LLVM.IntegerType, x::Integer) = LLVM.ConstantInt(t, x)
splatted(t::LLVM.FloatingPointType, x::Real) = LLVM.ConstantFP(t, x)
splatted(t::LLVM.VectorType, x::Real) = LLVM.const_splat(t, splatted(eltype(t), x))

"""
    widen_bits!(builder, h, wide) -> LLVM.Value

`fpext h to wide`, from `h`'s bits: sign, exponent and fraction moved into the
wider format, a subnormal `h` as its fraction times a power of two (exact, and
normal in the wider format), infinities and NaNs with the all-ones exponent.
"""
function widen_bits!(builder::LLVM.IRBuilder, h::LLVM.Value, wide::LLVM.LLVMType)
    narrow = LLVM.value_type(h)
    n, w = ieeeformat(scalartype(narrow)), ieeeformat(scalartype(wide))
    nexp, wexp = n.bits - 1 - n.fraction, w.bits - 1 - w.fraction
    nbias, wbias = (1 << (nexp - 1)) - 1, (1 << (wexp - 1)) - 1
    WI = shaped(LLVM.IntType(w.bits), narrow)
    k(x) = splatted(WI, x)

    bits = LLVM.zext!(builder, LLVM.bitcast!(builder, h, shaped(LLVM.IntType(n.bits), narrow)), WI)
    # Same exponent field (bfloat to float): every pattern, subnormals included,
    # is the narrow one with zeros appended.
    nexp == wexp && return LLVM.bitcast!(builder, LLVM.shl!(builder, bits, k(w.bits - n.bits)), wide)
    sign = LLVM.shl!(builder, LLVM.and!(builder, bits, k(1 << (n.bits - 1))), k(w.bits - n.bits))
    e = LLVM.and!(builder, LLVM.lshr!(builder, bits, k(n.fraction)), k((1 << nexp) - 1))
    m = LLVM.and!(builder, bits, k((1 << n.fraction) - 1))
    fraction = LLVM.shl!(builder, m, k(w.fraction - n.fraction))

    normal = LLVM.or!(builder, LLVM.shl!(builder, LLVM.add!(builder, e, k(wbias - nbias)), k(w.fraction)), fraction)
    special = LLVM.or!(builder, k(((1 << wexp) - 1) << w.fraction), fraction)
    # `m` has at most `n.fraction` bits, so converting it is exact, and the scale
    # is a power of two: the product is the subnormal's value, rounded nowhere.
    subnormal = LLVM.bitcast!(builder,
        LLVM.fmul!(builder, LLVM.uitofp!(builder, m, wide),
                   splatted(wide, 2.0^(1 - nbias - n.fraction))), WI)

    magnitude = LLVM.select!(builder, LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, e, k((1 << nexp) - 1)),
        special,
        LLVM.select!(builder, LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, e, k(0)), subnormal, normal))
    return LLVM.bitcast!(builder, LLVM.or!(builder, magnitude, sign), wide)
end
