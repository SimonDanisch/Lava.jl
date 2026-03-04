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
# Cache pipelines alongside compiled kernels (avoids re-hashing SPIR-V bytes)
const _pipeline_by_kernel = Dict{UInt64, LavaComputePipeline}()
# Cache arg layout offsets as Vector{Int} (Vector{Int} indexing is zero-alloc,
# unlike Vector{Pair{Int,Int}} which boxes Pair on access)
const _arg_offsets_cache = Dict{UInt64, Vector{Int}}()

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

    # Compile + pipeline (cached, single lookup — avoids re-hashing SPIR-V)
    compiled, pipeline, offsets = _get_compiled_kernel_and_pipeline(f, tt, workgroup_size)

    # Include f as first arg — GPUCompiler includes typeof(f) as the first LLVM parameter,
    # and wrap_entry_for_vulkan! creates a BDA slot for it (unless ghost-elided).
    all_args = (f, args...)

    # Compute total size: base layout + inline struct data (compile-time constant)
    inline_extra = _compute_inline_extra(typeof(all_args))
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # Get host-visible mapped arg buffer
    arg_buf = get_arg_buffer(total_size)

    # Pack args directly to mapped memory (zero intermediate allocations)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, all_args)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)

    # Keep data buffer references alive until vk_flush!() — BDA addresses in the
    # arg buffer are raw pointers with no GC reference to the backing VkManagedBuffer.
    keep_data_alive!(args)

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

# ── Zero-allocation arg packing (replaces _args_to_bda + pack_kernel_args_inline) ──

"""
    _is_bda_buffer(::Type{T})

Check at compile time whether a type is a GPU buffer that should be passed as a BDA address.
"""
_is_bda_buffer(::Type{<:LavaBuffer}) = true
_is_bda_buffer(::Type{<:LavaArray}) = true
_is_bda_buffer(::Type{VkManagedBuffer}) = true
_is_bda_buffer(::Type) = false

"""
    _compute_inline_extra(::Type{T}) where T <: Tuple

Compute at compile time the total bytes needed for inline struct data appended
after the base arg layout. Returns a constant. Buffer types (LavaBuffer, LavaArray,
VkManagedBuffer) are passed as UInt64 BDA addresses, not inlined.
"""
@generated function _compute_inline_extra(::Type{T}) where T <: Tuple
    types = T.parameters
    extra = 0
    for Ti in types
        sizeof(Ti) == 0 && continue
        _is_bda_buffer(Ti) && continue  # buffers → UInt64 BDA, no inline data
        if isbitstype(Ti) && !isprimitivetype(Ti)
            extra = (extra + 7) & ~7  # align to 8
            extra += sizeof(Ti)
        end
    end
    return :($extra)
end

"""
    _pack_args_direct!(mapped_ptr, arg_buf_bda, offsets, base_size, all_args)

Write kernel arguments directly to mapped GPU memory, inlining struct data.
This is a `@generated` function that statically filters ghost types and avoids
intermediate `Any[]` boxing and `Vector{UInt8}` allocations — zero allocations
for the arg packing itself.

Handles all argument types:
- Ghost types (sizeof==0): skipped
- LavaBuffer/LavaArray/VkManagedBuffer: written as UInt64 BDA address
- isbits structs: inlined after base layout, BDA pointer at arg slot
- UInt64/Ptr: written directly
- Other primitives: written directly
"""
@generated function _pack_args_direct!(mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                                        offsets::Vector{Int}, base_size::Int,
                                        all_args::T) where {T <: Tuple}
    types = T.parameters
    non_ghost = Int[]
    for (i, Ti) in enumerate(types)
        sizeof(Ti) == 0 && continue
        push!(non_ghost, i)
    end

    exprs = Expr[]
    for (layout_i, arg_i) in enumerate(non_ghost)
        Ti = types[arg_i]
        if _is_bda_buffer(Ti)
            # GPU buffer → write BDA address as UInt64
            if Ti <: LavaBuffer
                push!(exprs, :(unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])),
                                             all_args[$arg_i].buf.address)))
            elseif Ti <: LavaArray
                push!(exprs, :(unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])),
                                             bda_address(all_args[$arg_i]))))
            else  # VkManagedBuffer
                push!(exprs, :(unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])),
                                             all_args[$arg_i].address)))
            end
        elseif isbitstype(Ti) && !isprimitivetype(Ti)
            push!(exprs, quote
                let x = all_args[$arg_i]
                    inline_offset = (inline_offset + 7) & ~7
                    unsafe_store!(Ptr{$Ti}(mapped_ptr + inline_offset), x)
                    unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])),
                                  arg_buf_bda + UInt64(inline_offset))
                    inline_offset += $(sizeof(Ti))
                end
            end)
        elseif Ti === UInt64
            push!(exprs, :(unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])), all_args[$arg_i])))
        elseif Ti <: Ptr
            push!(exprs, :(unsafe_store!(Ptr{UInt64}(mapped_ptr + @inbounds(offsets[$layout_i])), UInt64(all_args[$arg_i]))))
        else
            push!(exprs, :(unsafe_store!(Ptr{$Ti}(mapped_ptr + @inbounds(offsets[$layout_i])), all_args[$arg_i])))
        end
    end

    quote
        inline_offset = base_size
        $(exprs...)
        return nothing
    end
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

# Fast ghost type check — sizeof(T)==0 is equivalent to GPUCompiler.isghosttype for isbits types.
# Avoids creating an LLVM Context on every call (~4μs → ~0.01μs per type).
_is_ghost(@nospecialize(T::Type)) = sizeof(T) == 0

"""
    _args_to_bda_filtered(args) -> Tuple

Convert arguments to BDA-compatible values, filtering ghost types (zero-sized singletons)
that GPUCompiler elides from LLVM IR.
"""
function _args_to_bda_filtered(args::Tuple)
    result = Any[]
    for x in args
        T = typeof(x)
        _is_ghost(T) && continue
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

function _get_compiled_kernel_and_pipeline(@nospecialize(f), @nospecialize(tt), workgroup_size)
    key = hash((f, tt, workgroup_size))

    compiled = get(_kernel_cache, key, nothing)
    if compiled === nothing
        compiled = lava_compile_gpu(f, tt; workgroup_size)
        _kernel_cache[key] = compiled
    end

    pipeline = get(_pipeline_by_kernel, key, nothing)
    if pipeline === nothing
        pipeline = get_compute_pipeline(compiled.spirv_bytes, compiled.entry_name;
                                        push_constant_size=compiled.push_info.push_size)
        _pipeline_by_kernel[key] = pipeline
    end

    offsets = get(_arg_offsets_cache, key, nothing)
    if offsets === nothing
        offsets = Int[p.first for p in compiled.push_info.arg_layout]
        _arg_offsets_cache[key] = offsets
    end

    return compiled, pipeline, offsets
end

# Reusable arg buffer pool — host-visible mapped memory for zero-cost upload.
# Each in-flight dispatch needs its own arg buffer (GPU reads it asynchronously).
# Pool grows dynamically when batched dispatches exceed current pool size.
# Reset to start of pool on vk_flush!() (all dispatches complete).
const _arg_buffers = VkMappedBuffer[]
const _arg_buffer_idx = Ref(0)

function get_arg_buffer(nbytes::Integer)
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
function reset_arg_buffer_pool!()
    _arg_buffer_idx[] = 0
end
