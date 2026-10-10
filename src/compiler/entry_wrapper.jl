# BDA Entry Wrapper for Vulkan compute kernels
#
# Transforms: kernel(arg1::T1, arg2::T2, ...) → void wrapper()
#
# The wrapper loads all kernel arguments from a device-memory buffer
# via PhysicalStorageBuffer (BDA). A single i64 push constant holds
# the BDA of the argument buffer.
#
# Push constant layout: { i64 bda_address }, or { i64 bda_address, i64 flag }
# for a compute kernel that can throw (see `LavaRuntime.signal_exception`).
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
    push_size::Int              # 8 (the argument BDA), 16 with the exception flag after it
    arg_buffer_size::Int        # Total size of argument data
    arg_layout::Vector{Pair{Int,Int}}  # (offset, size) per argument
    byval_llvm_sizes::Vector{Int}  # LLVM alloc size per arg (>0 only for byval struct args)
    # Just the offsets from `arg_layout`, because that is what packing wants and
    # rebuilding it there allocated a vector per draw per frame for a value fixed
    # at compile time.
    arg_offsets::Vector{Int}
end

PushConstantInfo(name, push_size, arg_buffer_size, arg_layout, byval_sizes) =
    PushConstantInfo(name, push_size, arg_buffer_size, arg_layout, byval_sizes,
                     [first(p) for p in arg_layout])

"""
    wrap_entry_for_vulkan!(mod, entry; workgroup_size, exceptions = false) -> PushConstantInfo

Transform the LLVM module so the entry point is a void() function that
loads kernel arguments from a BDA argument buffer.

The original entry function is marked internal+alwaysinline and will be
inlined into the wrapper by the AlwaysInliner pass.

`exceptions = true` for a compute kernel: if it can throw, the push constants
grow a second word, the address of the device's exception flag, and every throw
(`LavaRuntime.signal_exception`) becomes a store of 1 through it. Everywhere else
— a graphics or ray-tracing stage, a kernel with no arguments to push — the throw
stores nothing and only stops the invocation.
"""
function wrap_entry_for_vulkan!(mod::LLVM.Module, entry::LLVM.Function;
                                 workgroup_size::NTuple{3,Int}=(64,1,1),
                                 exceptions::Bool=false)
    entry_name = entry.name
    ft = entry.function_type
    param_types = collect(ft.parameters)
    signals = exceptionsignals(mod)

    # No parameters → no wrapping needed
    if isempty(param_types)
        lowersignals!(signals, nothing)
        return PushConstantInfo(entry_name, 0, 0, Pair{Int,Int}[], Int[])
    end

    # Mark original entry as internal + alwaysinline
    entry.linkage = LLVM.API.LLVMInternalLinkage
    attrs = entry.function_attributes
    delete!(attrs, LLVM.EnumAttribute("noinline"))
    push!(attrs, LLVM.EnumAttribute("alwaysinline"))

    # Compute argument buffer layout
    arg_layout = Pair{Int,Int}[]
    offset = 0
    for pt in param_types
        sz = llvm_sizeof(pt)
        align = max(4, sz)  # Natural alignment, minimum 4
        offset = (offset + align - 1) & ~(align - 1)
        push!(arg_layout, offset => sz)
        offset += sz
    end
    arg_buffer_size = offset

    # Extract byval type sizes using LLVM DataLayout for accurate struct sizes.
    # llvm_sizeof sums field sizes WITHOUT alignment padding, undercounting for
    # structs with mixed-size fields (e.g., WorkQueue{T} has {DevArr, DevArr, i32}
    # → llvm_sizeof=36 but ABI size=40 due to trailing padding).
    # Multiple byval args with padding gaps cause inline data overlap in the arg buffer.
    dl = mod.datalayout
    byval_llvm_sizes = zeros(Int, length(param_types))
    for (i, pt) in enumerate(param_types)
        pt isa LLVM.PointerType || continue
        for attr in collect(entry.parameter_attributes[i])
            if attr isa LLVM.TypeAttribute && attr.kind === :byval
                byval_type = attr.value
                byval_llvm_sizes[i] = Int(LLVM.API.LLVMABISizeOfType(dl, byval_type))
                break
            end
        end
    end

    # Create push constant global: { i64 } in addrspace(2) → PushConstant storage
    # class, and a second i64 when the kernel can raise the exception flag.
    T_i64 = LLVM.Int64Type()
    flagged = exceptions && !isempty(signals)
    # Named when it holds the flag: a literal `{ i64, i64 }` is the same LLVM type
    # as every two-word struct in the kernel (a `UnitRange{Int64}`), and the
    # emitter keys SPIR-V types on the LLVM type, so the push block's `Block` and
    # member offsets would land on that one too.
    T_push = if flagged
        t = LLVM.StructType("lava.push.flagged")
        LLVM.elements!(t, [T_i64, T_i64])
        t
    else
        LLVM.StructType([T_i64])
    end
    gv = LLVM.GlobalVariable(mod, T_push, "__push_constants", 2)
    gv.linkage = LLVM.API.LLVMExternalLinkage
    lowersignals!(signals, flagged ? (T_push, gv) : nothing)

    # Create wrapper function: void()
    T_void = LLVM.VoidType()
    wrapper_ft = LLVM.FunctionType(T_void)
    wrapper_name = "main"
    wrapper = LLVM.Function(mod, wrapper_name, wrapper_ft)

    # Build wrapper body
    bb = LLVM.BasicBlock(wrapper, "entry")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, insertion_point(bb))

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
                bda_val.alignment = 8
                ptr_val = LLVM.inttoptr!(builder, bda_val, pt, "arg$(i)")
                push!(args, ptr_val)
            else
                # Scalar arg: load directly from BDA buffer
                addr = LLVM.add!(builder, bda_int,
                                 LLVM.ConstantInt(T_i64, field_offset),
                                 "arg$(i)_addr")
                field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1, "arg$(i)_ptr")
                val = LLVM.load!(builder, pt, field_ptr, "arg$(i)")
                align = max(4, llvm_sizeof(pt))
                val.alignment = align
                push!(args, val)
            end
        end

        # Call original entry function
        LLVM.call!(builder, ft, entry, args)
        LLVM.ret!(builder)
    end

    return PushConstantInfo(wrapper_name, flagged ? 16 : 8, arg_buffer_size, arg_layout,
                            byval_llvm_sizes)
end

"""Every call of `_lava_signal_exception` in `mod`: the throws, wherever inlining
has put them."""
function exceptionsignals(mod::LLVM.Module)
    calls = LLVM.CallInst[]
    for fn in mod.functions, bb in fn.blocks, inst in bb.instructions
        inst isa LLVM.CallInst || continue
        callee = inst.called_operand
        callee isa LLVM.Function && callee.name == "_lava_signal_exception" &&
            push!(calls, inst)
    end
    return calls
end

"""
Lower each throw's flag raise: a store of 1 through the second push-constant word,
when `push` is the push struct and its global, or nothing at all. Straight-line
either way, which is what keeps a throwing block recognisable to
`fix_barrier_skipping_paths!`.
"""
function lowersignals!(calls::Vector{LLVM.CallInst}, push)
    for call in calls
        if push !== nothing
            T_push, gv = push
            LLVM.@dispose builder = LLVM.IRBuilder() begin
                LLVM.position!(builder, insertion_point(call))
                pc = LLVM.load!(builder, T_push, gv, "push_flag_load")
                addr = LLVM.extract_value!(builder, pc, 1, "exception_flag")
                ptr = LLVM.inttoptr!(builder, addr, LLVM.PointerType(LLVM.Int32Type(), 1),
                                     "exception_flag_ptr")
                st = LLVM.store!(builder, LLVM.ConstantInt(LLVM.Int32Type(), 1), ptr)
                st.alignment = 4
            end
        end
        LLVM.erase!(call)
    end
    return nothing
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
function llvm_sizeof(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return max(1, t.width ÷ 8)
    elseif t isa LLVM.FloatType
        return 4
    elseif t isa LLVM.DoubleType
        return 8
    elseif t isa LLVM.HalfType
        return 2
    elseif t isa LLVM.PointerType
        return 8  # 64-bit pointers → stored as i64 BDA
    elseif t isa LLVM.ArrayType
        return t.length * llvm_sizeof(t.element_type)
    elseif t isa LLVM.StructType
        total = 0
        for m in t.elements
            total += llvm_sizeof(m)
        end
        return total
    else
        return 8  # Fallback
    end
end
