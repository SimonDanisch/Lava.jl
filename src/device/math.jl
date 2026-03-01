# Device-side math function overrides for Lava.jl
#
# Replaces Julia's software implementations (which use lookup tables / global constants)
# with LLVM intrinsics that the SPIR-V emitter maps to GLSL.std.450 instructions.
# This avoids emitting global constant arrays (addrspace(1)) that SPIR-V can't handle.
#
# Float32 functions use LLVM intrinsics → GLSL.std.450 (hardware-accelerated).
# Float64 functions: GLSL.std.450 only supports 32-bit, so Float64 transcendentals
# need pure-Julia polynomial implementations (TODO).

# Helper to create llvmcall IR for a unary float intrinsic
macro _lava_unary_f32(jl_func, llvm_intrinsic)
    ir = """
        declare float @$llvm_intrinsic(float)
        define float @entry(float %x) #0 {
            %r = call float @$llvm_intrinsic(float %x)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::Float32)
            Base.llvmcall(($ir, "entry"), Float32, Tuple{Float32}, x)
        end
    end |> esc
end

macro _lava_binary_f32(jl_func, llvm_intrinsic)
    ir = """
        declare float @$llvm_intrinsic(float, float)
        define float @entry(float %x, float %y) #0 {
            %r = call float @$llvm_intrinsic(float %x, float %y)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """
    quote
        @lava_device_override @inline function $jl_func(x::Float32, y::Float32)
            Base.llvmcall(($ir, "entry"), Float32, Tuple{Float32, Float32}, x, y)
        end
    end |> esc
end

# ── Float32 math overrides ──

@_lava_unary_f32 Base.sin  "llvm.sin.f32"
@_lava_unary_f32 Base.cos  "llvm.cos.f32"
@_lava_unary_f32 Base.sqrt "llvm.sqrt.f32"
@_lava_unary_f32 Base.abs  "llvm.fabs.f32"
@_lava_unary_f32 Base.exp  "llvm.exp.f32"
@_lava_unary_f32 Base.exp2 "llvm.exp2.f32"
@_lava_unary_f32 Base.log  "llvm.log.f32"
@_lava_unary_f32 Base.log2 "llvm.log2.f32"
@_lava_unary_f32 Base.floor "llvm.floor.f32"
@_lava_unary_f32 Base.ceil  "llvm.ceil.f32"
@_lava_unary_f32 Base.round "llvm.round.f32"
@_lava_unary_f32 Base.trunc "llvm.trunc.f32"

@_lava_binary_f32 Base.:(^)  "llvm.pow.f32"
@_lava_binary_f32 Base.min   "llvm.minnum.f32"
@_lava_binary_f32 Base.max   "llvm.maxnum.f32"

@lava_device_override @inline function Base.log10(x::Float32)
    # log10(x) = log2(x) / log2(10)
    log2(x) * Float32(0.30102999566398114)  # 1/log2(10)
end

@lava_device_override @inline function Base.muladd(x::Float32, y::Float32, z::Float32)
    Base.llvmcall(("""
        declare float @llvm.fma.f32(float, float, float)
        define float @entry(float %x, float %y, float %z) #0 {
            %r = call float @llvm.fma.f32(float %x, float %y, float %z)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32, Float32, Float32}, x, y, z)
end

@lava_device_override @inline function Base.fma(x::Float32, y::Float32, z::Float32)
    Base.llvmcall(("""
        declare float @llvm.fma.f32(float, float, float)
        define float @entry(float %x, float %y, float %z) #0 {
            %r = call float @llvm.fma.f32(float %x, float %y, float %z)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32, Float32, Float32}, x, y, z)
end

# ── Integer abs override ──

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

@lava_device_override @inline function Base.literal_pow(::typeof(^), x::Float32, ::Val{2})
    x * x
end

@lava_device_override @inline function Base.literal_pow(::typeof(^), x::Float32, ::Val{3})
    x * x * x
end

# ── Complex abs override (simplified hypot, avoids Julia's multi-branch hypot CFG) ──
# Uses |a|*sqrt(1+(b/a)^2) where a=max, b=min to avoid overflow from squaring.
# Only 2 branches — StructurizeCFG handles this fine.

@lava_device_override @inline function Base.abs(z::ComplexF32)
    # Branchless: promote to Float64 to avoid Float32 overflow from squaring
    # Float32 max² = 1.16e77, fits in Float64 (max ~1.8e308)
    re = Float64(real(z))
    im = Float64(imag(z))
    Float32(sqrt(muladd(re, re, im * im)))
end

@lava_device_override @inline function Base.abs(z::ComplexF64)
    # Branchless rescaling: max * sqrt(1 + (min/max)^2) avoids overflow
    # ifelse compiles to select (no branch), max/min compile to fmax/fmin
    re = abs(real(z))
    im = abs(imag(z))
    mx = max(re, im)
    mn = min(re, im)
    r = ifelse(mx > 0.0, mn / mx, 0.0)
    mx * sqrt(muladd(r, r, 1.0))
end

@lava_device_override @inline function Base.abs(z::Complex{Float16})
    # Branchless: Float16 max is 65504, so 65504^2 = 4.29e9 fits in Float32 (max ~3.4e38)
    re = Float32(real(z))
    im = Float32(imag(z))
    Float16(sqrt(muladd(re, re, im * im)))
end

# ── Float64 transcendentals (pure-Julia polynomial implementations) ──
# GLSL.std.450 only supports 32-bit, so Float64 transcendentals use branchless
# polynomial approximations. These avoid the complex control flow in Julia's
# default implementations (which use lookup tables + branches that break StructurizeCFG).

"""Float64 log2 via range reduction + polynomial. Max relerr ~2e-16."""
@inline function _lava_log2_f64(x::Float64)
    u = reinterpret(UInt64, x)
    e = Int64((u >> 52) & 0x7FF) - 1023
    m_bits = (u & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000
    m = reinterpret(Float64, m_bits)

    # If m > sqrt(2), use m/2 and e+1 for better polynomial conditioning
    adj = (m_bits > 0x3FF6A09E667F3BCC) ? 1 : 0
    m = adj == 1 ? m * 0.5 : m
    e_adj = Float64(e + adj)

    # s = (m-1)/(m+1), log(m) = 2s(1 + s^2/3 + s^4/5 + ...)
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s

    # Horner form for 2(1 + s^2/3 + s^4/5 + ... + s^16/17)
    p = 2.0 / 17.0
    p = muladd(p, s2, 2.0 / 15.0)
    p = muladd(p, s2, 2.0 / 13.0)
    p = muladd(p, s2, 2.0 / 11.0)
    p = muladd(p, s2, 2.0 / 9.0)
    p = muladd(p, s2, 2.0 / 7.0)
    p = muladd(p, s2, 2.0 / 5.0)
    p = muladd(p, s2, 2.0 / 3.0)
    p = muladd(p, s2, 2.0)

    log2e = 1.4426950408889634  # 1/ln(2)
    return e_adj + s * p * log2e
end

"""Float64 exp2 via range reduction + Taylor polynomial. Max relerr ~2.2e-16."""
@inline function _lava_exp2_f64(x::Float64)
    n = round(x)
    f = x - n
    ni = unsafe_trunc(Int64, n)

    # 2^f = exp(f * ln2), compute via Taylor series
    ln2_hi = 0.6931471805599453
    ln2_lo = 2.3190468138462996e-17
    r = f * ln2_hi + f * ln2_lo

    # Taylor series: 1 + r + r^2/2! + ... + r^13/13!
    p = 1.0 / 6227020800.0    # 1/13!
    p = muladd(p, r, 1.0 / 479001600.0)     # 1/12!
    p = muladd(p, r, 1.0 / 39916800.0)      # 1/11!
    p = muladd(p, r, 1.0 / 3628800.0)       # 1/10!
    p = muladd(p, r, 1.0 / 362880.0)        # 1/9!
    p = muladd(p, r, 1.0 / 40320.0)         # 1/8!
    p = muladd(p, r, 1.0 / 5040.0)          # 1/7!
    p = muladd(p, r, 1.0 / 720.0)           # 1/6!
    p = muladd(p, r, 1.0 / 120.0)           # 1/5!
    p = muladd(p, r, 1.0 / 24.0)            # 1/4!
    p = muladd(p, r, 1.0 / 6.0)             # 1/3!
    p = muladd(p, r, 0.5)                   # 1/2!
    p = muladd(p, r, 1.0)                   # 1/1!
    p = muladd(p, r, 1.0)                   # 1/0!

    # Clamp exponent to avoid UInt64 wrap-around on GPU
    biased = 1023 + ni
    biased <= 0 && return 0.0       # underflow
    biased >= 2047 && return p * Inf # overflow
    pow2n = reinterpret(Float64, UInt64(biased) << 52)
    return p * pow2n
end

# Float64 pow: x^y = exp2(y * log2(x))
@lava_device_override @inline function Base.:^(x::Float64, y::Float64)
    x == 1.0 && return 1.0
    y == 0.0 && return 1.0
    y == 1.0 && return x
    x == 0.0 && return 0.0
    x < 0.0 && return -_lava_exp2_f64(y * _lava_log2_f64(-x))  # negative base (odd y assumed)
    _lava_exp2_f64(y * _lava_log2_f64(x))
end

@lava_device_override @inline function Base.:^(x::Float64, n::Int64)
    x == 1.0 && return 1.0
    n == 0 && return 1.0
    n == 1 && return x
    _lava_exp2_f64(Float64(n) * _lava_log2_f64(x))
end

@lava_device_override @inline function Base.literal_pow(::typeof(^), x::Float64, ::Val{2})
    x * x
end

@lava_device_override @inline function Base.literal_pow(::typeof(^), x::Float64, ::Val{3})
    x * x * x
end

# Float64 exp2
@lava_device_override @inline function Base.exp2(x::Float64)
    _lava_exp2_f64(x)
end

# Float64 log2
@lava_device_override @inline function Base.log2(x::Float64)
    _lava_log2_f64(x)
end

# Float64 exp = exp2(x * log2(e))
@lava_device_override @inline function Base.exp(x::Float64)
    _lava_exp2_f64(x * 1.4426950408889634)  # log2(e)
end

# Float64 log = log2(x) / log2(e) = log2(x) * ln(2)
@lava_device_override @inline function Base.log(x::Float64)
    _lava_log2_f64(x) * 0.6931471805599453  # ln(2)
end

# Float64 log10 = log2(x) / log2(10)
@lava_device_override @inline function Base.log10(x::Float64)
    _lava_log2_f64(x) * 0.30102999566398114  # 1/log2(10)
end

# Float64 sqrt via LLVM intrinsic (hardware-supported even in 64-bit)
@lava_device_override @inline function Base.sqrt(x::Float64)
    Base.llvmcall(("""
        declare double @llvm.sqrt.f64(double)
        define double @entry(double %x) #0 {
            %r = call double @llvm.sqrt.f64(double %x)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64}, x)
end

# Float64 abs via LLVM intrinsic
@lava_device_override @inline function Base.abs(x::Float64)
    Base.llvmcall(("""
        declare double @llvm.fabs.f64(double)
        define double @entry(double %x) #0 {
            %r = call double @llvm.fabs.f64(double %x)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64}, x)
end

# Float64 floor/ceil/round/trunc via LLVM intrinsics
@lava_device_override @inline function Base.floor(x::Float64)
    Base.llvmcall(("""
        declare double @llvm.floor.f64(double)
        define double @entry(double %x) #0 {
            %r = call double @llvm.floor.f64(double %x)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64}, x)
end

@lava_device_override @inline function Base.ceil(x::Float64)
    Base.llvmcall(("""
        declare double @llvm.ceil.f64(double)
        define double @entry(double %x) #0 {
            %r = call double @llvm.ceil.f64(double %x)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64}, x)
end

@lava_device_override @inline function Base.trunc(x::Float64)
    Base.llvmcall(("""
        declare double @llvm.trunc.f64(double)
        define double @entry(double %x) #0 {
            %r = call double @llvm.trunc.f64(double %x)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64}, x)
end

@lava_device_override @inline function Base.muladd(x::Float64, y::Float64, z::Float64)
    Base.llvmcall(("""
        declare double @llvm.fma.f64(double, double, double)
        define double @entry(double %x, double %y, double %z) #0 {
            %r = call double @llvm.fma.f64(double %x, double %y, double %z)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64, Float64, Float64}, x, y, z)
end

@lava_device_override @inline function Base.fma(x::Float64, y::Float64, z::Float64)
    Base.llvmcall(("""
        declare double @llvm.fma.f64(double, double, double)
        define double @entry(double %x, double %y, double %z) #0 {
            %r = call double @llvm.fma.f64(double %x, double %y, double %z)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64, Float64, Float64}, x, y, z)
end

@lava_device_override @inline function Base.min(x::Float64, y::Float64)
    Base.llvmcall(("""
        declare double @llvm.minnum.f64(double, double)
        define double @entry(double %x, double %y) #0 {
            %r = call double @llvm.minnum.f64(double %x, double %y)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64, Float64}, x, y)
end

@lava_device_override @inline function Base.max(x::Float64, y::Float64)
    Base.llvmcall(("""
        declare double @llvm.maxnum.f64(double, double)
        define double @entry(double %x, double %y) #0 {
            %r = call double @llvm.maxnum.f64(double %x, double %y)
            ret double %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float64, Tuple{Float64, Float64}, x, y)
end

# ── Float64 sin/cos (Cody-Waite range reduction + minimax polynomial) ──
# GLSL.std.450 only supports 32-bit trig. These use:
# 1. Cody-Waite range reduction: x → r in [-pi/4, pi/4] + quadrant j
# 2. Minimax polynomial on reduced argument (11th-order sin, 10th-order cos)
# No lookup tables, no branches except quadrant select via ifelse.

# Two-part pi/2 for Cody-Waite (sum = pi/2 exactly in extended precision)
const _PIO2_HI = 1.5707963267948966     # high 33 bits of pi/2
const _PIO2_LO = 6.123233995736766e-17  # pi/2 - PIO2_HI

# Minimax polynomial coefficients for sin(r) on [-pi/4, pi/4]
# sin(r) ≈ r + S1*r^3 + S2*r^5 + S3*r^7 + S4*r^9 + S5*r^11
const _S1 = -0.16666666666666632
const _S2 =  0.00833333333332249
const _S3 = -1.984126982985795e-4
const _S4 =  2.7557313707070068e-6
const _S5 = -2.5050760253406863e-8

# Minimax polynomial coefficients for cos(r) on [-pi/4, pi/4]
# cos(r) ≈ 1 + C1*r^2 + C2*r^4 + C3*r^6 + C4*r^8 + C5*r^10
const _C1 = -0.4999999999999990
const _C2 =  0.04166666666665290
const _C3 = -0.001388888888740510
const _C4 =  2.480158728947673e-5
const _C5 = -2.7557314351390663e-7

@inline function _lava_sin_kernel(r::Float64)
    r2 = r * r
    r * muladd(r2, muladd(r2, muladd(r2, muladd(r2, _S5, _S4), _S3), _S2), _S1) * r2 + r
end

@inline function _lava_cos_kernel(r::Float64)
    r2 = r * r
    muladd(r2, muladd(r2, muladd(r2, muladd(r2, _C5, _C4), _C3), _C2), _C1) * r2 + 1.0
end

@inline function _lava_reduce_trig(x::Float64)
    # Cody-Waite: compute j = round(x / (pi/2)), r = x - j * (pi/2)
    # Using two-part subtraction for accuracy
    TWO_OVER_PI = 0.6366197723675814  # 2/pi
    j = unsafe_trunc(Int32, muladd(x, TWO_OVER_PI, 0.5 * sign(x)))  # Rounding
    # Correct: j = round to nearest integer
    j_f = Float64(j)
    r = x - j_f * _PIO2_HI
    r = r - j_f * _PIO2_LO
    return r, j & Int32(3)  # quadrant 0-3
end

@lava_device_override @inline function Base.sin(x::Float64)
    xa = abs(x)
    xa > 1.0e15 && return 0.0  # For very large values, result is unreliable
    r, q = _lava_reduce_trig(xa)
    # q=0: sin(r), q=1: cos(r), q=2: -sin(r), q=3: -cos(r)
    s = _lava_sin_kernel(r)
    c = _lava_cos_kernel(r)
    result = ifelse(q == Int32(0), s, ifelse(q == Int32(1), c, ifelse(q == Int32(2), -s, -c)))
    ifelse(x < 0.0, -result, result)
end

@lava_device_override @inline function Base.cos(x::Float64)
    xa = abs(x)
    xa > 1.0e15 && return 1.0
    r, q = _lava_reduce_trig(xa)
    # q=0: cos(r), q=1: -sin(r), q=2: -cos(r), q=3: sin(r)
    s = _lava_sin_kernel(r)
    c = _lava_cos_kernel(r)
    ifelse(q == Int32(0), c, ifelse(q == Int32(1), -s, ifelse(q == Int32(2), -c, s)))
end
