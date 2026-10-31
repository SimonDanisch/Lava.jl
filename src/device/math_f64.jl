# Float64 transcendentals in Float64.
#
# GLSL.std.450 has no 64-bit Exp, Log, Asin, Acos, Atan or Atan2, and the tier-3
# fallback in `math.jl` used to answer them through Float32: seven digits. That
# is wrong for any kernel that computes in Float64 for a reason. An IPC contact
# barrier is `-(d - d̂)² log(d / d̂)`, and a line search compares energies whose
# relative change is 1e-11; through Float32 every comparison is noise.
#
# These are the table-free algorithms from fdlibm (Sun Microsystems, 1993) and
# their ports in Julia's Base: argument reduction by bit manipulation, then a
# polynomial or rational approximation, accurate to within an ulp. Base's own
# `log` and `exp` read lookup tables (`t_log_Float64`, `J_TABLE`), which become
# Private-storage arrays here (see tier 3b in `math.jl`), so fdlibm's table-free
# versions are used instead. Domain errors return NaN: a kernel cannot throw.

@inline _f64_hi(x::Float64) = (reinterpret(UInt64, x) >> 32) % UInt32
@inline _f64_lo(x::Float64) = reinterpret(UInt64, x) % UInt32
@inline _f64_sethi(x::Float64, hi::UInt32) =
    reinterpret(Float64, (UInt64(hi) << 32) | (reinterpret(UInt64, x) & 0x00000000ffffffff))

# ── log: fdlibm e_log.c ──────────────────────────────────────────────────────
@lava_device_override @inline function Base.log(x::Float64)
    ln2_hi = 6.93147180369123816490e-01
    ln2_lo = 1.90821492927058770002e-10
    Lg1 = 6.666666666666735130e-01
    Lg2 = 3.999999999940941908e-01
    Lg3 = 2.857142874366239149e-01
    Lg4 = 2.222219843214978396e-01
    Lg5 = 1.818357216161805012e-01
    Lg6 = 1.531383769920937332e-01
    Lg7 = 1.479819860511658591e-01
    hx = reinterpret(Int32, _f64_hi(x))
    lx = _f64_lo(x)
    k = Int32(0)
    if hx < Int32(0x00100000)                       # x < 2^-1022
        ((hx & Int32(0x7fffffff)) | reinterpret(Int32, lx)) == Int32(0) && return -Inf
        hx < Int32(0) && return NaN                 # log(-x) = NaN
        k -= Int32(54)
        x *= 1.80143985094819840000e+16             # subnormal: scale up by 2^54
        hx = reinterpret(Int32, _f64_hi(x))
    end
    hx >= Int32(0x7ff00000) && return x + x         # Inf or NaN
    k += (hx >> 20) - Int32(1023)
    hx &= Int32(0x000fffff)
    i = (hx + Int32(0x95f64)) & Int32(0x100000)
    x = _f64_sethi(x, reinterpret(UInt32, hx | (i ⊻ Int32(0x3ff00000))))   # x or x/2 into [√2/2, √2)
    k += i >> 20
    f = x - 1.0
    dk = Float64(k)
    if (Int32(0x000fffff) & (Int32(2) + hx)) < Int32(3)   # |f| < 2^-20
        if f == 0.0
            return k == Int32(0) ? 0.0 : dk * ln2_hi + dk * ln2_lo
        end
        R = f * f * (0.5 - 0.33333333333333333 * f)
        return k == Int32(0) ? f - R : dk * ln2_hi - ((R - dk * ln2_lo) - f)
    end
    s = f / (2.0 + f)
    z = s * s
    i = hx - Int32(0x6147a)
    w = z * z
    j = Int32(0x6b851) - hx
    t1 = w * (Lg2 + w * (Lg4 + w * Lg6))
    t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)))
    i |= j
    R = t2 + t1
    if i > Int32(0)
        hfsq = 0.5 * f * f
        return k == Int32(0) ? f - (hfsq - s * (hfsq + R)) :
               dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f)
    else
        return k == Int32(0) ? f - s * (f - R) : dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f)
    end
end

@lava_device_override @inline Base.log2(x::Float64) = log(x) * 1.4426950408889634
@lava_device_override @inline Base.log10(x::Float64) = log(x) * 0.4342944819032518

# ── exp: fdlibm e_exp.c ──────────────────────────────────────────────────────
@lava_device_override @inline function Base.exp(x::Float64)
    ln2_hi = 6.93147180369123816490e-01
    ln2_lo = 1.90821492927058770002e-10
    P1 = 1.66666666666666019037e-01
    P2 = -2.77777777770155933842e-03
    P3 = 6.61375632143793436117e-05
    P4 = -1.65339022054652515390e-06
    P5 = 4.13813679705723846039e-08
    hx = _f64_hi(x)
    negative = (hx >> 31) != UInt32(0)
    hx &= UInt32(0x7fffffff)
    if hx >= UInt32(0x40862E42)                     # |x| >= 709.78
        if hx >= UInt32(0x7ff00000)
            ((hx & UInt32(0xfffff)) | _f64_lo(x)) != UInt32(0) && return x + x   # NaN
            return negative ? 0.0 : x                                          # exp(±Inf)
        end
        x > 7.09782712893383973096e+02 && return Inf
        x < -7.45133219101941108420e+02 && return 0.0
    end
    k = Int32(0)
    hi = 0.0
    lo = 0.0
    if hx > UInt32(0x3fd62e42)                      # |x| > ln2 / 2
        if hx < UInt32(0x3FF0A2B2)                  # |x| < 3 ln2 / 2
            hi = negative ? x + ln2_hi : x - ln2_hi
            lo = negative ? -ln2_lo : ln2_lo
            k = negative ? Int32(-1) : Int32(1)
        else
            k = unsafe_trunc(Int32, 1.44269504088896338700e+00 * x + (negative ? -0.5 : 0.5))
            t = Float64(k)
            hi = x - t * ln2_hi
            lo = t * ln2_lo
        end
        x = hi - lo
    elseif hx < UInt32(0x3e300000)                  # |x| < 2^-28
        return 1.0 + x
    end
    t = x * x
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    k == Int32(0) && return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= Int32(-1021)
        return _f64_sethi(y, _f64_hi(y) + (reinterpret(UInt32, k) << 20))
    end
    return _f64_sethi(y, _f64_hi(y) + (reinterpret(UInt32, k + Int32(1000)) << 20)) *
           9.33263618503218878990e-302      # 2^-1000
end

# 2^x = 2^k · e^(f ln 2) with k = round(x): the power of two is exact, and the
# exponential runs on |f ln 2| <= ln 2 / 2, where it is accurate to an ulp.
@lava_device_override @inline function Base.exp2(x::Float64)
    isnan(x) && return x
    x >= 1024.0 && return Inf
    x <= -1075.0 && return 0.0
    k = unsafe_trunc(Int32, x + copysign(0.5, x))
    y = exp((x - Float64(k)) * 0.6931471805599453)
    k >= Int32(-1021) && return _f64_sethi(y, _f64_hi(y) + (reinterpret(UInt32, k) << 20))
    return _f64_sethi(y, _f64_hi(y) + (reinterpret(UInt32, k + Int32(1000)) << 20)) * 9.33263618503218878990e-302
end

# ── asin, acos, atan, atan(y, x): Base's ports of fdlibm (base/special/trig.jl) ──
@inline function _f64_arc_tRt(t::Float64)
    p = t * (1.66666666666666657415e-01 + t * (-3.25565818622400915405e-01 + t * (2.01212532134862925881e-01 +
        t * (-4.00555345006794114027e-02 + t * (7.91534994289814532176e-04 + t * 3.47933107596021167570e-05)))))
    q = 1.0 + t * (-2.40339491173441421878e+00 + t * (2.02094576023350569471e+00 +
        t * (-6.88283971605453293030e-01 + t * 7.70381505559019352791e-02)))
    return p / q
end

@lava_device_override @inline function Base.asin(x::Float64)
    absx = abs(x)
    if absx >= 1.0
        return absx == 1.0 ? copysign(1.5707963267948966, x) : NaN
    elseif absx < 0.5
        absx < 1.4901161193847656e-8 && return x
        return muladd(x, _f64_arc_tRt(x * x), x)
    end
    t = (1.0 - absx) / 2
    pio2_lo = 6.12323399573676603587e-17
    s = sqrt(t)
    tRt = _f64_arc_tRt(t)
    if absx >= 0.975
        return copysign(1.5707963267948966 - (2.0 * (s + s * tRt) - pio2_lo), x)
    end
    s0 = reinterpret(Float64, (reinterpret(UInt64, s) >> 32) << 32)
    c = (t - s0 * s0) / (s + s0)
    p = 2.0 * s * tRt - (pio2_lo - 2.0 * c)
    q = 0.7853981633974483 - 2.0 * s0
    return copysign(0.7853981633974483 - (p - q), x)
end

@lava_device_override @inline function Base.acos(x::Float64)
    pio2_hi = 1.57079632679489655800e+00
    pio2_lo = 6.12323399573676603587e-17
    absx = abs(x)
    if absx >= 1.0
        absx == 1.0 && return x > 0.0 ? 0.0 : 3.141592653589793
        return NaN
    elseif absx < 0.5
        absx < 6.938893903907228e-18 && return 1.5707963267948966
        return pio2_hi - (x - (pio2_lo - x * _f64_arc_tRt(x * x)))
    end
    z = (1.0 - absx) * 0.5
    zRz = _f64_arc_tRt(z)
    s = sqrt(z)
    if x < 0.0
        return 3.14159265358979311600e+00 - 2.0 * (s + (zRz * s - pio2_lo))
    end
    df = reinterpret(Float64, (reinterpret(UInt64, s) >> 32) << 32)
    c = (z - df * df) / (s + df)
    return 2.0 * (df + (zRz * s + c))
end

@inline function _f64_atan_pq(x::Float64)
    z = x * x
    w = z * z
    p = z * (3.33333333333329318027e-01 + w * (1.42857142725034663711e-01 + w * (9.09088713343650656196e-02 +
        w * (6.66107313738753120669e-02 + w * (4.97687799461593236017e-02 + w * 1.62858201153657823623e-02)))))
    q = w * (-1.99999999998764832476e-01 + w * (-1.11111104054623557880e-01 + w * (-7.69187620504482999495e-02 +
        w * (-5.83357013379057348645e-02 + w * -3.65315727442169155270e-02))))
    return p, q
end

@lava_device_override @inline function Base.atan(x::Float64)
    absx = abs(x)
    absx >= 7.378697629483821e19 && return copysign(1.5707963267948966, x)   # |x| >= 2^66
    if absx < 0.4375
        absx < 7.450580596923828e-9 && return x                               # |x| < 2^-27
        p, q = _f64_atan_pq(x)
        return x - x * (p + q)
    end
    hi = 0.0
    lo = 0.0
    t = 0.0
    if absx < 1.1875
        if absx < 0.6875
            hi = 4.63647609000806093515e-01     # atan(1/2)
            lo = 2.26987774529616870924e-17
            t = (2.0 * absx - 1.0) / (2.0 + absx)
        else
            hi = 7.85398163397448278999e-01     # atan(1)
            lo = 3.06161699786838301793e-17
            t = (absx - 1.0) / (absx + 1.0)
        end
    elseif absx < 2.4375
        hi = 9.82793723247329054082e-01         # atan(3/2)
        lo = 1.39033110312309984516e-17
        t = (absx - 1.5) / (1.0 + 1.5 * absx)
    else
        hi = 1.57079632679489655800e+00         # atan(Inf)
        lo = 6.12323399573676603587e-17
        t = -1.0 / absx
    end
    p, q = _f64_atan_pq(t)
    return copysign(hi - ((t * (p + q) - lo) - t), x)
end

@lava_device_override @inline function Base.atan(y::Float64, x::Float64)
    pi_lo = 1.2246467991473531772e-16
    (isnan(x) | isnan(y)) && return x + y
    x == 1.0 && return atan(y)
    m = 2 * signbit(x) + signbit(y)
    if y == 0.0
        m <= 1 && return y                                  # atan(±0, +x) = ±0
        return m == 2 ? 3.141592653589793 : -3.141592653589793
    elseif x == 0.0
        return copysign(1.5707963267948966, y)
    end
    if isinf(x)
        if isinf(y)
            m == 0 && return 0.7853981633974483
            m == 1 && return -0.7853981633974483
            m == 2 && return 2.356194490192345
            return -2.356194490192345
        end
        m == 0 && return 0.0
        m == 1 && return -0.0
        m == 2 && return 3.141592653589793
        return -3.141592653589793
    end
    isinf(y) && return copysign(1.5707963267948966, y)
    k = reinterpret(Int32, (_f64_hi(y) & UInt32(0x7fffffff)) - (_f64_hi(x) & UInt32(0x7fffffff))) >> 20
    z = 0.0
    if k > Int32(60)                                        # |y / x| > 2^60
        z = 1.5707963267948966 + 0.5 * pi_lo
        m &= 1
    elseif x < 0.0 && k < Int32(-60)                        # |y| / x < -2^60
        z = 0.0
    else
        z = atan(abs(y / x))
    end
    m == 0 && return z
    m == 1 && return -z
    m == 2 && return 3.141592653589793 - (z - pi_lo)
    return (z - pi_lo) - 3.141592653589793
end
