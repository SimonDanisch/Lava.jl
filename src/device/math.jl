# Device-side math function overrides for Lava.jl
#
# Strategy: PREFER LLVM intrinsics / GLSL.std.450 wherever possible.
#
# Three tiers:
#   1. LLVM intrinsics → SPIR-V emitter maps to GLSL.std.450
#   2. Custom _lava_glsl_* externals → SPIR-V emitter maps to GLSL.std.450 (for ops LLVM lacks)
#   3. Float64 transcendentals: exp, log, asin, acos, atan and atan2 in Float64 by
#      fdlibm's table-free algorithms (`math_f64.jl`); the rest through Float32
#
# GLSL.std.450 Float64 support:
#   - SUPPORTED (f64): Sqrt, FAbs, Floor, Ceil, Round, Trunc, FMin, FMax, Fma
#   - NOT SUPPORTED (f32/f16 only): Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh,
#     Exp, Exp2, Log, Log2, Pow, Atan2
#   For unsupported f64 transcendentals, we cast to f32 and back (GPU f64 transcendentals
#   are rare; if high precision is needed, add polynomial implementations later).

# ── Helpers: create llvmcall for LLVM intrinsics ──

macro _lava_unary_intrinsic(jl_func, llvm_base, T, llvm_ty)
    suffix = T == Float32 ? "f32" : T == Float16 ? "f16" : "f64"
    intrinsic = "$llvm_base.$suffix"
    ir = """
        declare $llvm_ty @$intrinsic($llvm_ty)
        define $llvm_ty @entry($llvm_ty %x) #0 {
            %r = call $llvm_ty @$intrinsic($llvm_ty %x)
            ret $llvm_ty %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::$T)
            Base.llvmcall(($ir, "entry"), $T, Tuple{$T}, x)
        end
    end |> esc
end

macro _lava_binary_intrinsic(jl_func, llvm_base, T, llvm_ty)
    suffix = T == Float32 ? "f32" : T == Float16 ? "f16" : "f64"
    intrinsic = "$llvm_base.$suffix"
    ir = """
        declare $llvm_ty @$intrinsic($llvm_ty, $llvm_ty)
        define $llvm_ty @entry($llvm_ty %x, $llvm_ty %y) #0 {
            %r = call $llvm_ty @$intrinsic($llvm_ty %x, $llvm_ty %y)
            ret $llvm_ty %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::$T, y::$T)
            Base.llvmcall(($ir, "entry"), $T, Tuple{$T, $T}, x, y)
        end
    end |> esc
end

macro _lava_ternary_intrinsic(jl_func, llvm_base, T, llvm_ty)
    suffix = T == Float32 ? "f32" : T == Float16 ? "f16" : "f64"
    intrinsic = "$llvm_base.$suffix"
    ir = """
        declare $llvm_ty @$intrinsic($llvm_ty, $llvm_ty, $llvm_ty)
        define $llvm_ty @entry($llvm_ty %x, $llvm_ty %y, $llvm_ty %z) #0 {
            %r = call $llvm_ty @$intrinsic($llvm_ty %x, $llvm_ty %y, $llvm_ty %z)
            ret $llvm_ty %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::$T, y::$T, z::$T)
            Base.llvmcall(($ir, "entry"), $T, Tuple{$T, $T, $T}, x, y, z)
        end
    end |> esc
end

# Helper for GLSL.std.450 ops that have no LLVM intrinsic.
# Uses custom function names (_lava_*) that the SPIR-V emitter recognizes.
macro _lava_glsl_unary(jl_func, lava_name, T, llvm_ty)
    ir = """
        declare $llvm_ty @$lava_name($llvm_ty)
        define $llvm_ty @entry($llvm_ty %x) #0 {
            %r = call $llvm_ty @$lava_name($llvm_ty %x)
            ret $llvm_ty %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::$T)
            Base.llvmcall(($ir, "entry"), $T, Tuple{$T}, x)
        end
    end |> esc
end

macro _lava_glsl_binary(jl_func, lava_name, T, llvm_ty)
    ir = """
        declare $llvm_ty @$lava_name($llvm_ty, $llvm_ty)
        define $llvm_ty @entry($llvm_ty %x, $llvm_ty %y) #0 {
            %r = call $llvm_ty @$lava_name($llvm_ty %x, $llvm_ty %y)
            ret $llvm_ty %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::$T, y::$T)
            Base.llvmcall(($ir, "entry"), $T, Tuple{$T, $T}, x, y)
        end
    end |> esc
end

# ══════════════════════════════════════════════════════════════════════
# Tier 1a: LLVM intrinsics that GLSL.std.450 supports for ALL float types
# These work for both Float32 and Float64.
# ══════════════════════════════════════════════════════════════════════

for (T, llvm_ty) in ((Float32, "float"), (Float64, "double"))
    @eval begin
        @_lava_unary_intrinsic  Base.sqrt  "llvm.sqrt"  $T $llvm_ty
        @_lava_unary_intrinsic  Base.abs   "llvm.fabs"  $T $llvm_ty
        @_lava_unary_intrinsic  Base.floor "llvm.floor" $T $llvm_ty
        @_lava_unary_intrinsic  Base.ceil  "llvm.ceil"  $T $llvm_ty
        # No `Base.round` here: Base's `round(x)` is `llvm.rint`, which the emitter maps to
        # RoundEven, Julia's ties-to-even. `llvm.round` is GLSL Round, whose ties go in a
        # direction the implementation chooses.
        @_lava_unary_intrinsic  Base.trunc "llvm.trunc" $T $llvm_ty
        @_lava_ternary_intrinsic Base.fma    "llvm.fma"     $T $llvm_ty
        @_lava_ternary_intrinsic Base.muladd "llvm.fma"     $T $llvm_ty
    end
end

# ── min / max ──
#
# `@fastmath min` (`Base.FastMath.min_fast`) is GLSL.std.450 NMin/NMax: IEEE minNum,
# where a NaN operand loses to a number. `Base.min`/`max` are Julia's: a NaN operand
# gives NaN and -0.0 < 0.0. GLSL.std.450 has no instruction for that, so they are
# NMin/NMax plus two selects; the emitter rejects `llvm.minimum`/`maximum`.
for (T, U, llvm_ty) in ((Float16, UInt16, "half"), (Float32, UInt32, "float"), (Float64, UInt64, "double"))
    @eval begin
        @_lava_binary_intrinsic Base.FastMath.min_fast "llvm.minnum" $T $llvm_ty
        @_lava_binary_intrinsic Base.FastMath.max_fast "llvm.maxnum" $T $llvm_ty
        @lava_device_override @inline function Base.min(x::$T, y::$T)
            r = Base.FastMath.min_fast(x, y)
            # equal operands are identical or ±0, where the negative one is the minimum
            r = ifelse(x == y, reinterpret($T, reinterpret($U, x) | reinterpret($U, y)), r)
            return ifelse(isnan(x) | isnan(y), ifelse(isnan(x), x, y), r)
        end
        @lava_device_override @inline function Base.max(x::$T, y::$T)
            r = Base.FastMath.max_fast(x, y)
            r = ifelse(x == y, reinterpret($T, reinterpret($U, x) & reinterpret($U, y)), r)
            return ifelse(isnan(x) | isnan(y), ifelse(isnan(x), x, y), r)
        end
    end
end

# ══════════════════════════════════════════════════════════════════════
# Tier 1b: LLVM intrinsics for transcendentals — Float32 ONLY
# GLSL.std.450 restricts these to 16/32-bit floats.
# ══════════════════════════════════════════════════════════════════════

@_lava_unary_intrinsic  Base.sin   "llvm.sin"   Float32 "float"
@_lava_unary_intrinsic  Base.cos   "llvm.cos"   Float32 "float"
@_lava_unary_intrinsic  Base.exp   "llvm.exp"   Float32 "float"
@_lava_unary_intrinsic  Base.exp2  "llvm.exp2"  Float32 "float"
@_lava_unary_intrinsic  Base.log   "llvm.log"   Float32 "float"
@_lava_unary_intrinsic  Base.log2  "llvm.log2"  Float32 "float"

# ── Powers ──
#
# `Base.:^` keeps Julia's results; only `Base.FastMath.pow_fast` (`@fastmath x^y`) is the
# native GLSL.std.450 Pow, which is undefined for x < 0 and for x = 0, y <= 0. The special
# cases follow Base's `^(::Float32, ::Float32)`. A domain error (negative base, non-integer
# exponent) is NaN, since a kernel cannot throw.

# GLSL.std.450 Pow, defined for x > 0
@inline native_pow(x::Float32, y::Float32) = Base.llvmcall(("""
        declare float @llvm.pow.f32(float, float)
        define float @entry(float %x, float %y) #0 {
            %r = call float @llvm.pow.f32(float %x, float %y)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32, Float32}, x, y)

# Small integer exponents by multiplication, the rest as |x|^n with the sign of x for odd
# n, as CUDA.jl's `^(::Float32, ::Integer)`: Base squares in Float64, which runs at a
# fraction of the Float32 rate on GPUs. The parity comes from the integer, since Float32(n)
# may round to an even number.
@lava_device_override @inline function Base.:^(x::Float32, n::Integer)
    n == 0 && return 1.0f0
    n == 1 && return x
    n == 2 && return x * x
    n == 3 && return x * x * x   # Base: compatible with literal_pow
    n == -1 && return inv(x)
    n == -2 && return (i = inv(x); i * i)
    r = x == 0.0f0 ? (n > 0 ? 0.0f0 : Inf32) : native_pow(abs(x), Float32(n))
    return ifelse(isodd(n), copysign(r, x), r)
end

@lava_device_override @inline function Base.:^(x::Float32, y::Float32)
    x == 1.0f0 && return 1.0f0
    # beyond this every result over- or underflows, and the integer test below stays exact
    if !(abs(y) < 0x1.Ap30)
        isnan(y) && return y
        y = copysign(Float32(0x1.Ap30), y)
    end
    n = unsafe_trunc(Int32, y)
    if y == n
        # Base answers an integral exponent with `x^n`; small ones multiply, as there
        n == 0 && return 1.0f0
        n == 1 && return x
        n == 2 && return x * x
        n == 3 && return x * x * x
        n == -1 && return inv(x)
        n == -2 && return (i = inv(x); i * i)
        r = x == 0.0f0 ? (n > 0 ? 0.0f0 : Inf32) : native_pow(abs(x), y)
        return ifelse(isodd(n), copysign(r, x), r)
    end
    x < 0.0f0 && return NaN32
    isfinite(x) || return x * (y > 0.0f0 || isnan(x))
    x == 0.0f0 && return abs(y) * Inf32 * !(y > 0.0f0)
    return native_pow(x, y)
end

@lava_device_override @inline Base.FastMath.pow_fast(x::Float32, y::Float32) = native_pow(x, y)
# Base lowers these to `llvm.powi`, which SPIR-V has no instruction for
@lava_device_override @inline Base.FastMath.pow_fast(x::Float32, n::Integer) = x^n
@lava_device_override @inline Base.FastMath.pow_fast(x::Float64, n::Integer) = x^n

# ══════════════════════════════════════════════════════════════════════
# Tier 2: GLSL.std.450 ops via custom _lava_glsl_* function names
# LLVM has no intrinsics for these, so we use custom function names
# that the SPIR-V emitter recognizes and maps to GLSL.std.450.
# Float32 only (GLSL.std.450 transcendentals are 16/32-bit).
# ══════════════════════════════════════════════════════════════════════

@_lava_glsl_unary  Base.tan  "_lava_glsl_tan_f32"  Float32 "float"
@_lava_glsl_unary  Base.asin "_lava_glsl_asin_f32" Float32 "float"
# Native GLSL Acos permits implementation-dependent accuracy. Use the small
# Float32 fdlibm rational approximation also used by Julia Base, keeping all
# arithmetic in Float32 (no dependency on Float64 device support).
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
@inline function _f32_arc_tRt(t::Float32)
    p = t * muladd(t, muladd(t, -8.6563630030f-03, -4.2743422091f-02), 1.6666586697f-01)
    return p / muladd(t, -7.0662963390f-01, 1.0f0)
end

@inline function _f32_acos(x::Float32)
    ax = abs(x)
    if ax >= 1.0f0
        return ax == 1.0f0 ? (x > 0.0f0 ? 0.0f0 : Float32(pi)) : NaN32
    elseif ax < 0.5f0
        ax < 1.4901161193847656f-08 && return Float32(pi)/2
        return 1.5707962513f0 - (x - (7.5497894159f-08 - x * _f32_arc_tRt(x*x)))
    end
    z = (1.0f0 - ax) * 0.5f0
    r, s = _f32_arc_tRt(z), sqrt(z)
    if x < 0.0f0
        return 3.1415925026f0 - 2.0f0 * (s + (r*s - 7.5497894159f-08))
    end
    hi = reinterpret(Float32, reinterpret(UInt32,s) & 0xfffff000)
    correction = (z - hi*hi)/(s + hi)
    return 2.0f0 * (hi + (r*s + correction))
end

@lava_device_override @inline Base.acos(x::Float32) = _f32_acos(x)
@_lava_glsl_unary  Base.atan "_lava_glsl_atan_f32" Float32 "float"
@_lava_glsl_unary  Base.sinh  "_lava_glsl_sinh_f32"  Float32 "float"
@_lava_glsl_unary  Base.cosh  "_lava_glsl_cosh_f32"  Float32 "float"
@_lava_glsl_unary  Base.tanh  "_lava_glsl_tanh_f32"  Float32 "float"
@_lava_glsl_unary  Base.asinh "_lava_glsl_asinh_f32" Float32 "float"
@_lava_glsl_unary  Base.acosh "_lava_glsl_acosh_f32" Float32 "float"
@_lava_glsl_unary  Base.atanh "_lava_glsl_atanh_f32" Float32 "float"

# atan(y, x) — atan2, GLSL.std.450 Atan2 (op 25), Float32 only
@_lava_glsl_binary Base.atan "_lava_glsl_atan2_f32" Float32 "float"

# ══════════════════════════════════════════════════════════════════════
# Tier 3: Float64 transcendentals — downcast to Float32, compute, upcast
# GLSL.std.450 does not support f64 for these operations.
# This provides ~7 decimal digits of precision (Float32 mantissa).
# exp, exp2, log, log2, log10, asin, acos, atan and atan2 are computed in
# Float64 instead, in `math_f64.jl`.
# ══════════════════════════════════════════════════════════════════════

@lava_device_override @inline Base.sin(x::Float64)  = Float64(sin(Float32(x)))
@lava_device_override @inline Base.cos(x::Float64)  = Float64(cos(Float32(x)))
@lava_device_override @inline Base.tan(x::Float64)  = Float64(tan(Float32(x)))
@lava_device_override @inline Base.sinh(x::Float64)  = Float64(sinh(Float32(x)))
@lava_device_override @inline Base.cosh(x::Float64)  = Float64(cosh(Float32(x)))
@lava_device_override @inline Base.tanh(x::Float64)  = Float64(tanh(Float32(x)))
@lava_device_override @inline Base.asinh(x::Float64) = Float64(asinh(Float32(x)))
@lava_device_override @inline Base.acosh(x::Float64) = Float64(acosh(Float32(x)))
@lava_device_override @inline Base.atanh(x::Float64) = Float64(atanh(Float32(x)))
# `^(::Float64, ...)` is in Float64, in `math_f64.jl`

# ══════════════════════════════════════════════════════════════════════
# Tier 3b: Float16 transcendentals — same downcast as Float64
#
# Without these, `exp(::Float16)` and friends fall through to Julia's generic
# implementations, which index lookup tables (`Base.Math.J_TABLE`). Those become
# Private-storage global arrays and the helper that reads them takes a pointer
# parameter the emitter types as PhysicalStorageBuffer, so spirv-val rejects the
# module:
#     OpFunctionCall Argument <id> '%_j_const_1's type does not match
#     Function <id>'s parameter type
# It is not an `exp` bug as such — every transcendental on Float16 hit it, which
# meant NO fp16 autocast kernel using sigmoid/exp/tanh could compile at all.
#
# GLSL.std.450 does define these for 16-bit floats, so a native half path is
# possible later; going through Float32 costs nothing in accuracy here since
# Float16 has 11 mantissa bits.
# ══════════════════════════════════════════════════════════════════════

@lava_device_override @inline Base.sin(x::Float16)   = Float16(sin(Float32(x)))
@lava_device_override @inline Base.cos(x::Float16)   = Float16(cos(Float32(x)))
@lava_device_override @inline Base.tan(x::Float16)   = Float16(tan(Float32(x)))
@lava_device_override @inline Base.exp(x::Float16)   = Float16(exp(Float32(x)))
@lava_device_override @inline Base.exp2(x::Float16)  = Float16(exp2(Float32(x)))
@lava_device_override @inline Base.log(x::Float16)   = Float16(log(Float32(x)))
@lava_device_override @inline Base.log2(x::Float16)  = Float16(log2(Float32(x)))
@lava_device_override @inline Base.log10(x::Float16) = Float16(log10(Float32(x)))
@lava_device_override @inline Base.asin(x::Float16)  = Float16(asin(Float32(x)))
@lava_device_override @inline Base.acos(x::Float16)  = Float16(acos(Float32(x)))
@lava_device_override @inline Base.atan(x::Float16)  = Float16(atan(Float32(x)))
@lava_device_override @inline Base.sinh(x::Float16)  = Float16(sinh(Float32(x)))
@lava_device_override @inline Base.cosh(x::Float16)  = Float16(cosh(Float32(x)))
@lava_device_override @inline Base.tanh(x::Float16)  = Float16(tanh(Float32(x)))
@lava_device_override @inline Base.asinh(x::Float16) = Float16(asinh(Float32(x)))
@lava_device_override @inline Base.acosh(x::Float16) = Float16(acosh(Float32(x)))
@lava_device_override @inline Base.atanh(x::Float16) = Float16(atanh(Float32(x)))
@lava_device_override @inline Base.sqrt(x::Float16)  = Float16(sqrt(Float32(x)))
@lava_device_override @inline Base.atan(y::Float16, x::Float16) = Float16(atan(Float32(y), Float32(x)))
@lava_device_override @inline Base.:(^)(x::Float16, y::Float16) = Float16(Float32(x) ^ Float32(y))
@lava_device_override @inline Base.sincos(x::Float16) = (sin(x), cos(x))

# sincos — composition
@lava_device_override @inline Base.sincos(x::Float32) = (sin(x), cos(x))
@lava_device_override @inline Base.sincos(x::Float64) = (sin(x), cos(x))

# log10 — composition from log2
@lava_device_override @inline Base.log10(x::Float32) = log2(x) * Float32(0.30102999566398114)

# No `clamp` override: Base's comparisons pass a NaN through, `min(max(x, lo), hi)` did not.

# copysign — IEEE bitwise sign-copy. Base's default lowers to `llvm.copysign`,
# which the SPIR-V emitter cannot answer as `FAbs(x) * FSign(y)`: GLSL.std.450
# FSign(0) == 0, so that produces 0 for any zero y and silently breaks
# ray-tracing `safe_invdir` clamps and other IEEE-dependent code.
# Mirrors the structure of Base's `flipsign` (base/floatfuncs.jl).
@lava_device_override @inline function Base.copysign(x::Float32, y::Float32)
    reinterpret(Float32,
        (reinterpret(UInt32, x) & 0x7FFFFFFF) | (reinterpret(UInt32, y) & 0x80000000))
end
@lava_device_override @inline function Base.copysign(x::Float64, y::Float64)
    reinterpret(Float64,
        (reinterpret(UInt64, x) & 0x7FFFFFFFFFFFFFFF) | (reinterpret(UInt64, y) & 0x8000000000000000))
end
@lava_device_override @inline function Base.copysign(x::Float16, y::Float16)
    reinterpret(Float16,
        (reinterpret(UInt16, x) & UInt16(0x7FFF)) | (reinterpret(UInt16, y) & UInt16(0x8000)))
end


# isfinite / isinf: Base's definitions are wrong on device, so these override them.
#
# `isfinite(NaN32)`, `isfinite(Inf32)` and `isfinite(-Inf32)` all returned
# `true` inside a kernel on RADV. Base writes it as
#
#     isfinite(x) = !isnan(x - x)      # fsub x, x  +  fcmp ord
#
# and the `x - x` is the problem: SPIR-V does not guarantee NaN/Inf semantics
# unless the module asks for `SignedZeroInfNanPreserve`, so the driver is
# entitled to fold `x - x` to zero, and ACO does. `fcmp ord 0.0, 0.0` is then
# true and every non-finite value reports itself finite. `isnan` is unaffected
# because it is a bare `fcmp uno`, which the emitter turns into `OpIsNan`
# directly, with no arithmetic in front of it to fold.
#
# Tested rather than reasoned about: on device `isnan` gave [0,1,0,0,0] over
# [1, NaN, Inf, -Inf, 0] (correct) while `isfinite` gave [1,1,1,1,1].
#
# Fixed here rather than by requesting NaN preservation for the whole module:
# the execution mode constrains EVERY float operation in the kernel, to repair
# two predicates that have an exact bitwise spelling. The exponent field being
# all ones is what "not finite" MEANS in IEEE 754, so these are not
# approximations of the Base definitions — they are the definitions.
#
# Any device code testing for a NaN should use `isnan`, which works; this is
# about the two predicates that need the *exponent*, not a comparison.

@lava_device_override @inline Base.isfinite(x::Float32) =
    (reinterpret(UInt32, x) & 0x7f800000) != 0x7f800000
@lava_device_override @inline Base.isfinite(x::Float64) =
    (reinterpret(UInt64, x) & 0x7ff0000000000000) != 0x7ff0000000000000
@lava_device_override @inline Base.isfinite(x::Float16) =
    (reinterpret(UInt16, x) & UInt16(0x7c00)) != UInt16(0x7c00)

@lava_device_override @inline Base.isinf(x::Float32) =
    (reinterpret(UInt32, x) & 0x7fffffff) == 0x7f800000
@lava_device_override @inline Base.isinf(x::Float64) =
    (reinterpret(UInt64, x) & 0x7fffffffffffffff) == 0x7ff0000000000000
@lava_device_override @inline Base.isinf(x::Float16) =
    (reinterpret(UInt16, x) & UInt16(0x7fff)) == UInt16(0x7c00)

# ══════════════════════════════════════════════════════════════════════
# Integer overrides
# ══════════════════════════════════════════════════════════════════════

@lava_device_override @inline function Base.abs(x::Int32)
    Base.llvmcall(("""
        define i32 @entry(i32 %x) #0 {
            %neg = sub i32 0, %x
            %cmp = icmp slt i32 %x, 0
            %r = select i1 %cmp, i32 %neg, i32 %x
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Int32, Tuple{Int32}, x)
end

@lava_device_override @inline function Base.abs(x::Int64)
    Base.llvmcall(("""
        define i64 @entry(i64 %x) #0 {
            %neg = sub i64 0, %x
            %cmp = icmp slt i64 %x, 0
            %r = select i1 %cmp, i64 %neg, i64 %x
            ret i64 %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Int64, Tuple{Int64}, x)
end

# ── Literal powers ──

for T in (Float32, Float64)
    @eval begin
        @lava_device_override @inline Base.literal_pow(::typeof(^), x::$T, ::Val{2}) = x * x
        @lava_device_override @inline Base.literal_pow(::typeof(^), x::$T, ::Val{3}) = x * x * x
        @lava_device_override @inline Base.literal_pow(::typeof(^), x::$T, ::Val{4}) = (x * x) * (x * x)
    end
end

# No complex `abs` override: Base's `hypot(real(z), imag(z))` scales, so large and
# small moduli neither overflow nor underflow.

# ── Functions that ccall libm and must be overridden to avoid GPU crashes ──

# openlibm's `log1pf` (FreeBSD `s_log1pf.c`), not `log(1 + x)`: that loses every
# digit of `x` below the ulp of 1 — at `x = 1e-7` it is 19% off, and TRELLIS.2's
# `softplus` split weights, `log1p(exp(logit))`, came out 24% wrong on a third
# of the voxels. Kahan's `log(u) x / (u - 1)` does not help here: the driver
# folds `(1 + x) - 1` to `x` (the same algebra that folded `x - x` to 0 under
# `isfinite`), which turns it back into `log(1 + x)`.
#
# This form is robust to that folding. Below `sqrt(2) - 1` the polynomial runs
# on `f = x` itself, which is exact; above, `1 + x` is normalised through its
# bits, which no algebra sees through. What the folding does cost is the
# rounding correction `c = 1 - (u - x)`, at most a couple of ulp where it
# applies (`k > 0`, where `log` is at least 0.35), so it is left out.
@lava_device_override @inline function Base.Math.log1p(x::Float32)
    hx = reinterpret(Int32, x)
    ax = hx & Int32(0x7fffffff)
    k = Int32(1)
    f = x
    hu = Int32(1)
    if hx < Int32(0x3ed413d0)                  # 1 + x < sqrt(2)
        ax >= Int32(0x3f800000) && return x == -1.0f0 ? -Inf32 : NaN32   # x <= -1
        if ax < Int32(0x38000000)              # |x| < 2^-15
            return ax < Int32(0x33800000) ? x : x - x * x * 0.5f0
        end
        if hx > Int32(0) || hx <= reinterpret(Int32, 0xbe95f619)
            k = Int32(0)                       # sqrt(2)/2 <= 1 + x < sqrt(2)
        end
    end
    hx >= Int32(0x7f800000) && return x + x    # Inf or NaN
    if k != Int32(0)
        u = hx < Int32(0x5a000000) ? 1.0f0 + x : x
        hu = reinterpret(Int32, u)
        k = (hu >> 23) - Int32(127)
        hu &= Int32(0x007fffff)
        if hu < Int32(0x3504f4)                # u < sqrt(2)
            u = reinterpret(Float32, hu | Int32(0x3f800000))
        else
            k += Int32(1)
            u = reinterpret(Float32, hu | Int32(0x3f000000))
            hu = (Int32(0x00800000) - hu) >> 2
        end
        f = u - 1.0f0
    end
    hfsq = 0.5f0 * f * f
    kf = Float32(k)
    ln2_hi, ln2_lo = 6.9313812256f-1, 9.0580006145f-6
    if hu == Int32(0)                          # |f| < 2^-20
        if f == 0.0f0
            return k == Int32(0) ? 0.0f0 : kf * ln2_hi + kf * ln2_lo
        end
        R = hfsq * (1.0f0 - 0.66666666666666666f0 * f)
        return k == Int32(0) ? f - R : kf * ln2_hi - ((R - kf * ln2_lo) - f)
    end
    s = f / (2.0f0 + f)
    z = s * s
    R = z * (6.6666668653f-1 + z * (4.0000000596f-1 + z * (2.8571429849f-1 + z * (2.2222198546f-1 +
        z * (1.8183572590f-1 + z * (1.5313838422f-1 + z * 1.4798198640f-1))))))
    return k == Int32(0) ? f - (hfsq - s * (hfsq + R)) : kf * ln2_hi - ((hfsq - (s * (hfsq + R) + kf * ln2_lo)) - f)
end
@lava_device_override @inline Base.Math.log1p(x::Float64) = Float64(log1p(Float32(x)))

@lava_device_override @inline function Base.Math.cbrt(x::Float32)
    s = sign(x)
    s * abs(x)^(1.0f0 / 3.0f0)
end
@lava_device_override @inline function Base.Math.cbrt(x::Float64)
    s = sign(x)
    Float64(Float32(s) * Float32(abs(x))^(1.0f0 / 3.0f0))
end

# No `hypot` override: Base's neither overflows (Float32 is computed in Float64, Float64
# scaled). `Base.FastMath.hypot_fast` is the unscaled `sqrt(x*x + y*y)`.
