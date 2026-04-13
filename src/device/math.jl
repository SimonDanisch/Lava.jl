# Device-side math function overrides for Lava.jl
#
# Strategy: PREFER LLVM intrinsics / GLSL.std.450 wherever possible.
#
# Three tiers:
#   1. LLVM intrinsics → SPIR-V emitter maps to GLSL.std.450
#   2. Custom _lava_glsl_* externals → SPIR-V emitter maps to GLSL.std.450 (for ops LLVM lacks)
#   3. Float64 transcendentals: f64→f32 downcast, compute via f32 GLSL.std.450, upcast back
#
# GLSL.std.450 Float64 support:
#   - SUPPORTED (f64): Sqrt, FAbs, Floor, Ceil, Round, Trunc, FMin, FMax, Fma
#   - NOT SUPPORTED (f32/f16 only): Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh,
#     Exp, Exp2, Log, Log2, Pow, Atan2
#   For unsupported f64 transcendentals, we cast to f32 and back (GPU f64 transcendentals
#   are rare; if high precision is needed, add polynomial implementations later).

# ── Helpers: create llvmcall for LLVM intrinsics ──

macro _lava_unary_intrinsic(jl_func, llvm_base, T, llvm_ty)
    suffix = T == Float32 ? "f32" : "f64"
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
    suffix = T == Float32 ? "f32" : "f64"
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
    suffix = T == Float32 ? "f32" : "f64"
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
        @_lava_unary_intrinsic  Base.round "llvm.round" $T $llvm_ty
        @_lava_unary_intrinsic  Base.trunc "llvm.trunc" $T $llvm_ty
        @_lava_binary_intrinsic Base.min   "llvm.minnum" $T $llvm_ty
        @_lava_binary_intrinsic Base.max   "llvm.maxnum" $T $llvm_ty
        @_lava_ternary_intrinsic Base.fma    "llvm.fma"     $T $llvm_ty
        @_lava_ternary_intrinsic Base.muladd "llvm.fma"     $T $llvm_ty
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
@_lava_binary_intrinsic Base.:(^)  "llvm.pow"   Float32 "float"

# ══════════════════════════════════════════════════════════════════════
# Tier 2: GLSL.std.450 ops via custom _lava_glsl_* function names
# LLVM has no intrinsics for these, so we use custom function names
# that the SPIR-V emitter recognizes and maps to GLSL.std.450.
# Float32 only (GLSL.std.450 transcendentals are 16/32-bit).
# ══════════════════════════════════════════════════════════════════════

@_lava_glsl_unary  Base.tan  "_lava_glsl_tan_f32"  Float32 "float"
@_lava_glsl_unary  Base.asin "_lava_glsl_asin_f32" Float32 "float"
@_lava_glsl_unary  Base.acos "_lava_glsl_acos_f32" Float32 "float"
@_lava_glsl_unary  Base.atan "_lava_glsl_atan_f32" Float32 "float"
@_lava_glsl_unary  Base.sinh "_lava_glsl_sinh_f32" Float32 "float"
@_lava_glsl_unary  Base.cosh "_lava_glsl_cosh_f32" Float32 "float"
@_lava_glsl_unary  Base.tanh "_lava_glsl_tanh_f32" Float32 "float"

# atan(y, x) — atan2, GLSL.std.450 Atan2 (op 25), Float32 only
@_lava_glsl_binary Base.atan "_lava_glsl_atan2_f32" Float32 "float"

# ══════════════════════════════════════════════════════════════════════
# Tier 3: Float64 transcendentals — downcast to Float32, compute, upcast
# GLSL.std.450 does not support f64 for these operations.
# This provides ~7 decimal digits of precision (Float32 mantissa).
# For higher precision, add polynomial implementations later.
# ══════════════════════════════════════════════════════════════════════

@lava_device_override @inline Base.sin(x::Float64)  = Float64(sin(Float32(x)))
@lava_device_override @inline Base.cos(x::Float64)  = Float64(cos(Float32(x)))
@lava_device_override @inline Base.tan(x::Float64)  = Float64(tan(Float32(x)))
@lava_device_override @inline Base.exp(x::Float64)  = Float64(exp(Float32(x)))
@lava_device_override @inline Base.exp2(x::Float64) = Float64(exp2(Float32(x)))
@lava_device_override @inline Base.log(x::Float64)  = Float64(log(Float32(x)))
@lava_device_override @inline Base.log2(x::Float64) = Float64(log2(Float32(x)))
@lava_device_override @inline Base.asin(x::Float64) = Float64(asin(Float32(x)))
@lava_device_override @inline Base.acos(x::Float64) = Float64(acos(Float32(x)))
@lava_device_override @inline Base.atan(x::Float64) = Float64(atan(Float32(x)))
@lava_device_override @inline Base.sinh(x::Float64) = Float64(sinh(Float32(x)))
@lava_device_override @inline Base.cosh(x::Float64) = Float64(cosh(Float32(x)))
@lava_device_override @inline Base.tanh(x::Float64) = Float64(tanh(Float32(x)))
@lava_device_override @inline Base.atan(y::Float64, x::Float64) = Float64(atan(Float32(y), Float32(x)))
@lava_device_override @inline Base.:(^)(x::Float64, y::Float64) = Float64(Float32(x) ^ Float32(y))

# sincos — composition
@lava_device_override @inline Base.sincos(x::Float32) = (sin(x), cos(x))
@lava_device_override @inline Base.sincos(x::Float64) = (sin(x), cos(x))

# log10 — composition from log2
@lava_device_override @inline Base.log10(x::Float32) = log2(x) * Float32(0.30102999566398114)
@lava_device_override @inline Base.log10(x::Float64) = Float64(log10(Float32(x)))

# clamp — composition from min/max (works for all float types)
@lava_device_override @inline Base.clamp(x::Float32, lo::Float32, hi::Float32) = min(max(x, lo), hi)
@lava_device_override @inline Base.clamp(x::Float64, lo::Float64, hi::Float64) = min(max(x, lo), hi)

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

# ── Power with integer exponent ──

@lava_device_override @inline function Base.:^(x::Float32, n::Int64)
    Base.llvmcall(("""
        declare float @llvm.pow.f32(float, float)
        define float @entry(float %x, i64 %n) #0 {
            %nf = sitofp i64 %n to float
            %r = call float @llvm.pow.f32(float %x, float %nf)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32, Int64}, x, n)
end

@lava_device_override @inline function Base.:^(x::Float64, n::Int64)
    Base.llvmcall(("""
        declare double @llvm.pow.f64(double, double)
        define double @entry(double %x, i64 %n) #0 {
            %nf = sitofp i64 %n to double
            %r = call double @llvm.pow.f64(double %x, double %nf)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64, Int64}, x, n)
end

for T in (Float32, Float64)
    @eval begin
        @lava_device_override @inline Base.literal_pow(::typeof(^), x::$T, ::Val{2}) = x * x
        @lava_device_override @inline Base.literal_pow(::typeof(^), x::$T, ::Val{3}) = x * x * x
    end
end

# ══════════════════════════════════════════════════════════════════════
# Complex abs (simplified, branchless)
# ══════════════════════════════════════════════════════════════════════

@lava_device_override @inline function Base.abs(z::ComplexF32)
    re = real(z)
    im = imag(z)
    sqrt(muladd(re, re, im * im))
end

@lava_device_override @inline function Base.abs(z::ComplexF64)
    re = abs(real(z))
    im = abs(imag(z))
    mx = max(re, im)
    mn = min(re, im)
    r = ifelse(mx > 0.0, mn / mx, 0.0)
    mx * sqrt(muladd(r, r, 1.0))
end

@lava_device_override @inline function Base.abs(z::Complex{Float16})
    re = Float32(real(z))
    im = Float32(imag(z))
    Float16(sqrt(muladd(re, re, im * im)))
end

# ── Functions that ccall libm and must be overridden to avoid GPU crashes ──

@lava_device_override @inline Base.Math.log1p(x::Float32) = log(1.0f0 + x)
@lava_device_override @inline Base.Math.log1p(x::Float64) = Float64(log(1.0f0 + Float32(x)))

@lava_device_override @inline function Base.Math.cbrt(x::Float32)
    s = sign(x)
    s * abs(x)^(1.0f0 / 3.0f0)
end
@lava_device_override @inline function Base.Math.cbrt(x::Float64)
    s = sign(x)
    Float64(Float32(s) * Float32(abs(x))^(1.0f0 / 3.0f0))
end

@lava_device_override @inline function Base.Math.hypot(x::Float32, y::Float32)
    sqrt(muladd(x, x, y * y))
end
@lava_device_override @inline function Base.Math.hypot(x::Float64, y::Float64)
    Float64(hypot(Float32(x), Float32(y)))
end
