# High-level kernel launch API for Lava.jl
#
# Handles: compile → pipeline cache → arg packing → dispatch → sync.

# ── Typed GPU buffer ──

"""
    LavaBuffer{T} — A typed GPU buffer wrapping VkManagedBuffer.

Carries element type `T` so the compiler knows the pointer type.
"""
struct LavaBuffer{T}
    buf::VkManagedBuffer
    length::Int
end

function LavaBuffer{T}(n::Integer) where T
    buf = vk_alloc(n * sizeof(T))
    LavaBuffer{T}(buf, Int(n))
end

Base.eltype(::LavaBuffer{T}) where T = T
Base.length(b::LavaBuffer) = b.length
Base.sizeof(b::LavaBuffer{T}) where T = b.length * sizeof(T)

function upload!(dst::LavaBuffer{T}, data::AbstractVector{T}) where T
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(data))))
    upload!(dst.buf, bytes)
end

function download(src::LavaBuffer{T}) where T
    result = Vector{T}(undef, src.length)
    download_typed!(result, src.buf)
    return result
end

# Cache compiled GPU kernels by (function, type tuple, workgroup_size)
const _kernel_cache = Dict{UInt64, LavaGPUKernel}()

"""
    lava_launch!(f, args...; ndrange, workgroup_size=(64,1,1))

Compile and dispatch a Julia function as a Vulkan compute kernel.

Each argument must be either:
- A `LavaBuffer{T}` (passed as `Ptr{T}` to the kernel)
- A `VkManagedBuffer` with a type annotation via `Pair`: `buf => Float32`
- A scalar (Int32, Float32, UInt64, etc.)

`ndrange` is the total number of work items (can be Int or NTuple{1-3,Int}).
`workgroup_size` is threads per workgroup.

Example:
    a = LavaBuffer{Float32}(n)
    lava_launch!(my_kernel, a, b, Int32(n); ndrange=n, workgroup_size=(256,1,1))
"""
function lava_launch!(@nospecialize(f), args...;
                       ndrange::Union{Integer, NTuple{3,<:Integer}},
                       workgroup_size::NTuple{3,Int} = (64, 1, 1))
    # Normalize ndrange to 3D
    if ndrange isa Integer
        ndrange_3d = (Int(ndrange), 1, 1)
    else
        ndrange_3d = (Int(ndrange[1]), Int(ndrange[2]), Int(ndrange[3]))
    end

    # Compute number of workgroups
    groups = (
        cld(ndrange_3d[1], workgroup_size[1]),
        cld(ndrange_3d[2], workgroup_size[2]),
        cld(ndrange_3d[3], workgroup_size[3]),
    )

    # Build the type tuple from arguments
    tt = Tuple{map(_arg_llvm_type, args)...}

    # Compile (cached)
    compiled = _get_compiled_kernel(f, tt, workgroup_size)

    # Create pipeline (cached in pipeline.jl)
    pipeline = get_compute_pipeline(compiled.spirv_bytes, compiled.entry_name;
                                     push_constant_size=compiled.push_info.push_size)

    # Pack arguments into BDA buffer
    # Include f as first arg — GPUCompiler includes typeof(f) as the first LLVM parameter,
    # and wrap_entry_for_vulkan! creates a BDA slot for it (unless ghost-elided).
    bda_args = _args_to_bda_filtered((f, args...))

    # Compute total size: base layout + inline struct data
    inline_extra = sum(arg isa InlineStructArg ? ((length(arg.bytes) + 7) & ~7) : 0
                       for arg in bda_args; init=0)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # Get host-visible mapped arg buffer (zero-cost write via memcpy)
    arg_buf = _get_arg_buffer(total_size)

    # Pack with self-referencing BDAs for inline structs
    arg_data = pack_kernel_args_inline(bda_args, compiled.push_info.arg_layout,
                                        compiled.push_info.arg_buffer_size,
                                        arg_buf.address)
    # Write directly to mapped memory — no staging copy needed
    unsafe_copyto!(arg_buf.mapped_ptr, pointer(arg_data), length(arg_data))

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)

    # Dispatch (batched — call vk_flush!() to submit)
    vk_dispatch!(pipeline, push_data, groups)

    return nothing
end

# ── Inline struct args ──

"""
    InlineStructArg

Represents a struct argument whose bytes should be inlined directly
into the argument buffer. The entry wrapper reads a BDA from the arg slot
that points into the same arg buffer (self-referencing BDA), eliminating
the need for a separate GPU buffer and double BDA indirection.
"""
struct InlineStructArg
    bytes::Vector{UInt8}
end

"""
    pack_kernel_args_inline(args, layout, base_size, arg_buf_bda) -> Vector{UInt8}

Pack kernel arguments, inlining struct data directly into the arg buffer.
Struct args (InlineStructArg) are appended after the normal arg slots,
and their pointer slots contain self-referencing BDA pointers.
"""
function pack_kernel_args_inline(args::Tuple, layout::Vector{Pair{Int,Int}},
                                  base_size::Int, arg_buf_bda::UInt64)
    # First pass: compute total size including inline structs
    inline_offset = base_size
    struct_offsets = Int[]
    for (i, arg) in enumerate(args)
        if arg isa InlineStructArg
            inline_offset = (inline_offset + 7) & ~7  # Align to 8 bytes
            push!(struct_offsets, inline_offset)
            inline_offset += length(arg.bytes)
        else
            push!(struct_offsets, -1)
        end
    end
    total_size = inline_offset

    buf = zeros(UInt8, total_size)
    si = 0  # struct_offsets index
    for (i, arg) in enumerate(args)
        offset = layout[i].first
        si += 1
        if arg isa InlineStructArg
            so = struct_offsets[si]
            copyto!(buf, so + 1, arg.bytes, 1, length(arg.bytes))
            # Self-referencing BDA: points into same arg buffer
            unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 1)), arg_buf_bda + UInt64(so))
        elseif arg isa UInt64
            unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 1)), arg)
        elseif arg isa Ptr
            unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 1)), UInt64(arg))
        else
            ptr = Ptr{typeof(arg)}(pointer(buf, offset + 1))
            unsafe_store!(ptr, arg)
        end
    end
    return buf
end

# ── Argument type mapping ──

# Map Julia arg types to LLVM-level types for compilation
_arg_llvm_type(::LavaBuffer{T}) where T = Ptr{T}
_arg_llvm_type(::LavaArray{T}) where T = Ptr{T}
_arg_llvm_type(x::T) where T = T  # Scalars pass through

# Convert arguments to BDA-compatible values for pack_kernel_args
_arg_to_bda(buf::LavaBuffer) = buf.buf.address
_arg_to_bda(a::LavaArray) = bda_address(a)
function _arg_to_bda(x)
    # Julia passes isbits structs by pointer at the LLVM level.
    # Inline struct bytes into the arg buffer to avoid double BDA indirection.
    T = typeof(x)
    if isbitstype(T) && !isprimitivetype(T)
        data = Vector{UInt8}(undef, sizeof(T))
        unsafe_store!(Ptr{T}(pointer(data)), x)
        return InlineStructArg(data)
    end
    return x
end

"""
    _args_to_bda_filtered(args) -> Tuple

Convert arguments to BDA-compatible values, filtering ghost types (zero-sized singletons)
that GPUCompiler elides from LLVM IR.
"""
function _args_to_bda_filtered(args::Tuple)
    result = Any[]
    for x in args
        T = typeof(x)
        if GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T)
            continue
        end
        push!(result, _arg_to_bda(x))
    end
    return tuple(result...)
end

function _get_compiled_kernel(@nospecialize(f), @nospecialize(tt), workgroup_size)
    key = hash((f, tt, workgroup_size))
    cached = get(_kernel_cache, key, nothing)
    if cached !== nothing
        return cached
    end
    compiled = lava_compile_gpu(f, tt; workgroup_size)
    _kernel_cache[key] = compiled
    return compiled
end

# Reusable arg buffer pool — host-visible mapped memory for zero-cost upload.
# Each in-flight dispatch needs its own arg buffer (GPU reads it asynchronously).
# Pool grows dynamically when batched dispatches exceed current pool size.
# Reset to start of pool on vk_flush!() (all dispatches complete).
const _arg_buffers = VkMappedBuffer[]
const _arg_buffer_idx = Ref(0)

function _get_arg_buffer(nbytes::Integer)
    alloc_size = max(256, nextpow(2, nbytes))

    _arg_buffer_idx[] += 1
    idx = _arg_buffer_idx[]

    # Grow pool if needed (more in-flight dispatches than current pool size)
    while length(_arg_buffers) < idx
        push!(_arg_buffers, vk_alloc_mapped(alloc_size))
    end

    buf = _arg_buffers[idx]

    # Grow individual buffer if too small
    if buf.size < nbytes
        buf = vk_alloc_mapped(alloc_size)
        _arg_buffers[idx] = buf
    end

    return buf
end

"""Reset arg buffer pool index after flush (all in-flight dispatches completed)."""
function _reset_arg_buffer_pool!()
    _arg_buffer_idx[] = 0
end
