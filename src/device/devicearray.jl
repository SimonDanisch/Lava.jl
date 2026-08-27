# The array a KERNEL receives. Nothing here is Vulkan's.
#
# Split out of `array/lavaarray.jl`, which holds the host-side `LavaArray` — the
# buffer, its lifetime, the pool it came from and the transfers. Those are the
# runtime's and are moving to Mantle. This is not: a `(pointer, dims)` pair is
# what the compiler emits loads and stores against, `device/atomics.jl` builds
# `Atomix` references over it, and neither needs a device to exist.
#
# The two are still a pair — `LavaDeviceArray(a::LavaArray)` is beside the host
# type, because it is that type that knows its own buffer address.

"""
    LavaDeviceArray{T,N}

Device-side isbits array representation for GPU kernels.

The pointer is a Vulkan buffer device address, which is why this is a plain
`Ptr{T}`: BDA is a 64-bit number in the shader's address space, so a kernel
indexes it exactly as it would host memory and the emitter needs no separate
descriptor path.
"""
struct LavaDeviceArray{T,N} <: GPUArrays.AbstractDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end

Base.size(a::LavaDeviceArray) = a.dims
Base.pointer(a::LavaDeviceArray{T}) where {T} = a.ptr
Base.pointer(a::LavaDeviceArray{T}, i::Integer) where {T} = a.ptr + (i - 1) * sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, a::LavaDeviceArray{T}) where {T} = a.ptr
