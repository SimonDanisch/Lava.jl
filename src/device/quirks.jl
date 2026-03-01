# Device overrides for functions that throw exceptions or use features
# unavailable on GPU. These replace host-side throwing implementations
# with GPU-safe no-ops or trap instructions.
#
# Based on CUDA.jl and AMDGPU.jl quirks patterns.

# ── Exception throwing overrides ──
# These all become no-ops or traps on GPU to avoid generating
# dynamic function calls that GPUCompiler rejects.

@lava_device_override function Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where T
    return nothing  # silently ignore on GPU
end

@lava_device_override function Base.throw_boundserror(A, I)
    return nothing
end

@lava_device_override function Base.Math.throw_complex_domainerror(f::Symbol, x)
    return nothing
end

@lava_device_override function Base.Math.throw_exp_domainerror(x)
    return nothing
end

@lava_device_override function Base.throw_domerr_powbysq(::Any, p)
    return nothing
end

@lava_device_override function Base.throw_domerr_powbysq(::Integer, p)
    return nothing
end

@lava_device_override function Base.Checked.throw_overflowerr_binaryop(op, x, y)
    return nothing
end

@lava_device_override function Base.Checked.throw_overflowerr_negation(x)
    return nothing
end

@lava_device_override function Base.__throw_gcd_overflow(a, b)
    return nothing
end

# ── Checked arithmetic overrides ──
# checked_abs uses throw_overflowerr_negation which we suppress above,
# but it's simpler to override the whole function

@lava_device_override function Base.Checked.checked_abs(x::Base.Checked.SignedInt)
    abs(x)
end

# ── Reshape dimension mismatch ──
@lava_device_override function Base._throw_dmrs(n, str, dims)
    return nothing
end

# ── CartesianIndices indexing ──
# CartesianIndices{N}[Int] for N>1 calls _ind2sub which has throw paths.
# Override to avoid the dynamic dispatch.
@lava_device_override function Base.getindex(iter::CartesianIndices{N,R}, I::Vararg{Int, N}) where {N,R}
    CartesianIndex(I...)
end

# Linear indexing: CartesianIndices{N}[linear_idx::Int] → CartesianIndex{N}
# Computes _ind2sub without throws. Required for KA's expand() on 2D+ dispatch.
@lava_device_override function Base.getindex(iter::CartesianIndices{N,R}, i::Int) where {N,R}
    @inbounds _gpu_ind2sub(iter, i)
end

# Convert 1-based linear index to CartesianIndex over CartesianIndices ranges
@inline function _gpu_ind2sub(iter::CartesianIndices{0}, i::Int)
    CartesianIndex()
end

@inline function _gpu_ind2sub(iter::CartesianIndices{1}, i::Int)
    f = first(iter)
    @inbounds CartesianIndex(f[1] + i - 1)
end

@inline function _gpu_ind2sub(iter::CartesianIndices{2}, i::Int)
    f = first(iter)
    s = size(iter)
    i0 = i - 1
    d1 = s[1]
    j1 = i0 % d1
    j2 = i0 ÷ d1
    @inbounds CartesianIndex(f[1] + j1, f[2] + j2)
end

@inline function _gpu_ind2sub(iter::CartesianIndices{3}, i::Int)
    f = first(iter)
    s = size(iter)
    i0 = i - 1
    d1 = s[1]
    d12 = d1 * s[2]
    j1 = i0 % d1
    j2 = (i0 % d12) ÷ d1
    j3 = i0 ÷ d12
    @inbounds CartesianIndex(f[1] + j1, f[2] + j2, f[3] + j3)
end

# 4D
@inline function _gpu_ind2sub(iter::CartesianIndices{4}, i::Int)
    f = first(iter)
    s = size(iter)
    i0 = i - 1
    d1 = s[1]
    d12 = d1 * s[2]
    d123 = d12 * s[3]
    j1 = i0 % d1
    j2 = (i0 % d12) ÷ d1
    j3 = (i0 % d123) ÷ d12
    j4 = i0 ÷ d123
    @inbounds CartesianIndex(f[1] + j1, f[2] + j2, f[3] + j3, f[4] + j4)
end

# Generic N-dimensional (for large N like 18D permutedims)
@inline function _gpu_ind2sub(iter::CartesianIndices{N}, i::Int) where N
    f = first(iter)
    s = size(iter)
    i0 = i - 1
    # Compute indices via repeated div/mod
    inds = ntuple(Val(N)) do d
        stride = 1
        for k in 1:(d-1)
            stride *= s[k]
        end
        f[d] + (i0 ÷ stride) % s[d]
    end
    @inbounds CartesianIndex(inds)
end

# Type conversion throw overrides are handled by throw_inexacterror above.
# Int32(x::Int64) calls checked_trunc_sint → throw_inexacterror which we suppress.

# ── Multiplicative inverse without i128 ──
# _mul_high(Int64, Int64) calls widen(Int64) → Int128 which SPIR-V doesn't support.
# Implement via 32-bit decomposition instead.
@lava_device_override function Base.MultiplicativeInverses._mul_high(a::UInt64, b::UInt64)
    shift = UInt64(32)
    mask = UInt64(0xFFFFFFFF)
    a1, a2 = a >>> shift, a & mask
    b1, b2 = b >>> shift, b & mask
    a1b1 = a1 * b1
    a1b2 = a1 * b2
    a2b1 = a2 * b1
    a2b2 = a2 * b2
    carry = ((a1b2 & mask) + (a2b1 & mask) + (a2b2 >>> shift)) >>> shift
    a1b1 + (a1b2 >>> shift) + (a2b1 >>> shift) + carry
end

@lava_device_override function Base.MultiplicativeInverses._mul_high(a::Int64, b::Int64)
    # Signed mul_high from unsigned mul_high (same pattern as Julia's Int128 version)
    t1 = (a >> 63) & (b % UInt64)
    t2 = (b >> 63) & (a % UInt64)
    (Base.MultiplicativeInverses._mul_high(a % UInt64, b % UInt64) - t1 - t2) % Int64
end

# ── sincos domain error ──
@lava_device_override function Base.Math.sincos_domain_error(x)
    return nothing
end

# ── String / IO suppression ──
# Any code path that tries to print or construct strings on GPU is a no-op

@lava_device_override function Base.print(io::IO, args...)
    return nothing
end

@lava_device_override function Base.println(io::IO, args...)
    return nothing
end

@lava_device_override function Base.show(io::IO, x)
    return nothing
end
