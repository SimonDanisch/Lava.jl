# BDA Entry Wrapper for Vulkan compute kernels
#
# Transforms: kernel(arg1::T1, arg2::T2, ...) → void wrapper()
#
# The wrapper loads all kernel arguments from a device-memory buffer
# via PhysicalStorageBuffer (BDA). A single i64 push constant holds
# the BDA of the argument buffer.
#
# Push constant layout: { i64 bda_address }
# Argument buffer layout: [ arg1_bytes | arg2_bytes | ... ] (natural alignment)
#
# For pointer arguments (Ptr{T} → i64 in LLVM): load i64 BDA from arg buffer,
# then inttoptr to ptr addrspace(0).
#
# For scalar arguments (Int32, Float32, etc.): load directly from arg buffer.
#
# For byval struct arguments: alloca on stack, load flattened fields from
# arg buffer, store into alloca, pass pointer.

"""
    PushConstantInfo

Describes the push constant and argument buffer layout for a wrapped kernel.
"""
struct PushConstantInfo
    wrapper_name::String
    push_size::Int              # Always 8 (single i64 BDA)
    arg_buffer_size::Int        # Total size of argument data
    arg_layout::Vector{Pair{Int,Int}}  # (offset, size) per argument
    byval_llvm_sizes::Vector{Int}  # LLVM alloc size per arg (>0 only for byval struct args)
end

"""
    wrap_entry_for_vulkan!(mod, entry; workgroup_size) -> PushConstantInfo

Transform the LLVM module so the entry point is a void() function that
loads kernel arguments from a BDA argument buffer.

The original entry function is marked internal+alwaysinline and will be
inlined into the wrapper by the AlwaysInliner pass.
"""
function wrap_entry_for_vulkan!(mod::LLVM.Module, entry::LLVM.Function;
                                 workgroup_size::NTuple{3,Int}=(64,1,1))
    entry_name = LLVM.name(entry)
    ft = LLVM.function_type(entry)
    param_types = collect(LLVM.parameters(ft))

    # No parameters → no wrapping needed
    if isempty(param_types)
        return PushConstantInfo(entry_name, 0, 0, Pair{Int,Int}[], Int[])
    end

    # Mark original entry as internal + alwaysinline
    LLVM.linkage!(entry, LLVM.API.LLVMInternalLinkage)
    attrs = LLVM.function_attributes(entry)
    delete!(attrs, LLVM.EnumAttribute("noinline"))
    push!(attrs, LLVM.EnumAttribute("alwaysinline"))

    # Compute argument buffer layout
    arg_layout = Pair{Int,Int}[]
    offset = 0
    for pt in param_types
        sz = _llvm_sizeof(pt)
        align = max(4, sz)  # Natural alignment, minimum 4
        offset = (offset + align - 1) & ~(align - 1)
        push!(arg_layout, offset => sz)
        offset += sz
    end
    arg_buffer_size = offset

    # Extract byval type sizes using LLVM DataLayout for accurate struct sizes.
    # _llvm_sizeof sums field sizes WITHOUT alignment padding, undercounting for
    # structs with mixed-size fields (e.g., WorkQueue{T} has {DevArr, DevArr, i32}
    # → _llvm_sizeof=36 but ABI size=40 due to trailing padding).
    # Multiple byval args with padding gaps cause inline data overlap in the arg buffer.
    dl = LLVM.datalayout(mod)
    byval_kind_id = LLVM.API.LLVMGetEnumAttributeKindForName("byval", 5)
    byval_llvm_sizes = zeros(Int, length(param_types))
    for (i, pt) in enumerate(param_types)
        pt isa LLVM.PointerType || continue
        for attr in collect(LLVM.parameter_attributes(entry, i))
            if attr isa LLVM.TypeAttribute && LLVM.kind(attr) == byval_kind_id
                byval_type = LLVM.value(attr)
                byval_llvm_sizes[i] = Int(LLVM.API.LLVMABISizeOfType(dl, byval_type))
                break
            end
        end
    end

    # Create push constant global: { i64 } in addrspace(2) → PushConstant storage class
    T_i64 = LLVM.Int64Type()
    T_push = LLVM.StructType([T_i64])
    gv = LLVM.GlobalVariable(mod, T_push, "__push_constants", 2)
    LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)

    # Create wrapper function: void()
    T_void = LLVM.VoidType()
    wrapper_ft = LLVM.FunctionType(T_void)
    wrapper_name = "main"
    wrapper = LLVM.Function(mod, wrapper_name, wrapper_ft)

    # Build wrapper body
    bb = LLVM.BasicBlock(wrapper, "entry")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, bb)

        # Load BDA from push constant
        push_val = LLVM.load!(builder, T_push, gv, "push_load")
        bda_int = LLVM.extract_value!(builder, push_val, 0, "bda")

        # Load each argument from the BDA buffer
        T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)
        args = LLVM.Value[]

        for (i, pt) in enumerate(param_types)
            field_offset = arg_layout[i].first

            if pt isa LLVM.PointerType
                # Pointer arg: load i64 BDA from buffer, inttoptr to ptr
                addr = LLVM.add!(builder, bda_int,
                                 LLVM.ConstantInt(T_i64, field_offset),
                                 "arg$(i)_addr")
                field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1, "arg$(i)_ptr")
                bda_val = LLVM.load!(builder, T_i64, field_ptr, "arg$(i)_bda")
                LLVM.alignment!(bda_val, 8)
                ptr_val = LLVM.inttoptr!(builder, bda_val, pt, "arg$(i)")
                push!(args, ptr_val)
            else
                # Scalar arg: load directly from BDA buffer
                addr = LLVM.add!(builder, bda_int,
                                 LLVM.ConstantInt(T_i64, field_offset),
                                 "arg$(i)_addr")
                field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1, "arg$(i)_ptr")
                val = LLVM.load!(builder, pt, field_ptr, "arg$(i)")
                align = max(4, _llvm_sizeof(pt))
                LLVM.alignment!(val, align)
                push!(args, val)
            end
        end

        # Call original entry function
        LLVM.call!(builder, ft, entry, args)
        LLVM.ret!(builder)
    end

    return PushConstantInfo(wrapper_name, 8, arg_buffer_size, arg_layout, byval_llvm_sizes)
end

"""
    pack_kernel_args(args::Tuple, layout::Vector{Pair{Int,Int}}) -> Vector{UInt8}

Pack kernel arguments into a byte buffer matching the BDA argument buffer layout.
Pointers are stored as UInt64 BDA addresses; scalars as their natural representation.
"""
function pack_kernel_args(args::Tuple, layout::Vector{Pair{Int,Int}}, total_size::Int)
    buf = zeros(UInt8, total_size)
    for (i, arg) in enumerate(args)
        offset = layout[i].first
        sz = layout[i].second
        if arg isa UInt64
            # BDA address
            unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 1)), arg)
        elseif arg isa Ptr
            # Julia pointer → UInt64
            unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 1)), UInt64(arg))
        else
            # Scalar — copy bytes directly
            ptr = Ptr{typeof(arg)}(pointer(buf, offset + 1))
            unsafe_store!(ptr, arg)
        end
    end
    return buf
end

"""Size of an LLVM type in bytes."""
function _llvm_sizeof(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return max(1, LLVM.width(t) ÷ 8)
    elseif t isa LLVM.LLVMFloat
        return 4
    elseif t isa LLVM.LLVMDouble
        return 8
    elseif t isa LLVM.LLVMHalf
        return 2
    elseif t isa LLVM.PointerType
        return 8  # 64-bit pointers → stored as i64 BDA
    elseif t isa LLVM.ArrayType
        return length(t) * _llvm_sizeof(eltype(t))
    elseif t isa LLVM.StructType
        total = 0
        for m in LLVM.elements(t)
            total += _llvm_sizeof(m)
        end
        return total
    else
        return 8  # Fallback
    end
end
