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
        sz = llvm_type_size(pt)
        align = max(4, sz)  # Natural alignment, minimum 4
        offset = (offset + align - 1) & ~(align - 1)
        push!(arg_layout, offset => sz)
        offset += sz
    end
    arg_buffer_size = offset

    # Extract byval type sizes using LLVM DataLayout for accurate struct sizes.
    # llvm_type_size sums field sizes WITHOUT alignment padding, undercounting for
    # structs with mixed-size fields (e.g., WorkQueue{T} has {DevArr, DevArr, i32}
    # → llvm_type_size=36 but ABI size=40 due to trailing padding).
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
                align = max(4, llvm_type_size(pt))
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

"""
    wrap_entry_for_webgpu!(mod, entry) -> PushConstantInfo

The WebGPU counterpart of `wrap_entry_for_vulkan!`: the entry becomes a `void()`
function that reads its arguments from the argument block, a read-only storage
binding of 32-bit words (`_lava_wgpu_argload(word)`), as WGSL has no device
addresses to load them through.

Every argument takes a slot of whole words at a 4-byte aligned offset, in order; a
`byval` aggregate takes its ABI size inline, where the Vulkan layout holds its
address. The wrapper copies a slot's words into a private array and reads the
argument from it (or passes it, for `byval`); SROA dissolves the copy. A throw
raises the heap's exception word and carries on (`continuethrows!`).
"""
function wrap_entry_for_webgpu!(mod::LLVM.Module, entry::LLVM.Function)
    entry_name = entry.name
    ft = entry.function_type
    param_types = collect(ft.parameters)
    continuethrows!(mod)
    isempty(param_types) && return PushConstantInfo(entry_name, 0, 0, Pair{Int,Int}[], Int[])

    entry.linkage = LLVM.API.LLVMInternalLinkage
    attrs = entry.function_attributes
    delete!(attrs, LLVM.EnumAttribute("noinline"))
    push!(attrs, LLVM.EnumAttribute("alwaysinline"))

    dl = mod.datalayout
    byval_types = Vector{Union{Nothing,LLVM.LLVMType}}(nothing, length(param_types))
    byval_sizes = zeros(Int, length(param_types))
    for (i, pt) in enumerate(param_types)
        pt isa LLVM.PointerType || continue
        for attr in collect(entry.parameter_attributes[i])
            if attr isa LLVM.TypeAttribute && attr.kind === :byval
                byval_types[i] = attr.value
                byval_sizes[i] = Int(LLVM.API.LLVMABISizeOfType(dl, attr.value))
                break
            end
        end
        byval_types[i] === nothing &&
            error("$(entry_name): argument $i is a pointer; a WebGPU kernel takes no device addresses")
    end

    arg_layout = Pair{Int,Int}[]
    offset = 0
    for (i, pt) in enumerate(param_types)
        sz = byval_types[i] === nothing ? Int(LLVM.API.LLVMABISizeOfType(dl, pt)) : byval_sizes[i]
        push!(arg_layout, offset => sz)
        offset += cld(sz, 4) * 4
    end

    T_i32 = LLVM.Int32Type()
    argload_ty = LLVM.FunctionType(T_i32, [T_i32])
    argload = get(LLVM.functions(mod), "_lava_wgpu_argload", nothing)
    argload === nothing && (argload = LLVM.Function(mod, "_lava_wgpu_argload", argload_ty))

    wrapper = LLVM.Function(mod, "main", LLVM.FunctionType(LLVM.VoidType()))
    bb = LLVM.BasicBlock(wrapper, "entry")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, insertion_point(bb))
        args = LLVM.Value[]
        for (i, pt) in enumerate(param_types)
            off, sz = arg_layout[i]
            nwords = cld(sz, 4)
            T_words = LLVM.ArrayType(T_i32, nwords)
            slot = LLVM.alloca!(builder, T_words, "arg$(i)_words")
            slot.alignment = 16
            for w in 0:nwords - 1
                word = LLVM.call!(builder, argload_ty, argload,
                                  LLVM.Value[LLVM.ConstantInt(T_i32, off ÷ 4 + w)], "arg$(i)_w$(w)")
                p = LLVM.gep!(builder, T_words, slot,
                              LLVM.Value[LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, w)])
                st = LLVM.store!(builder, word, p)
                st.alignment = 4
            end
            if byval_types[i] === nothing
                val = LLVM.load!(builder, pt, slot, "arg$(i)")
                val.alignment = 4
                push!(args, val)
            else
                push!(args, slot)
            end
        end
        LLVM.call!(builder, ft, entry, args)
        LLVM.ret!(builder)
    end
    return PushConstantInfo("main", 0, offset, arg_layout, byval_sizes)
end

"""The word of heap 0 a WebGPU kernel raises when it throws; the device reserves it."""
const WEBGPU_EXCEPTION_WORD = 0
"""The word of heap 0 the stores and atomics of an invocation that threw go to instead; the device reserves it."""
const WEBGPU_SINK_WORD = 1

"""
    continuethrows!(mod)

WebGPU's lowering of a throw: raise the exception word in the heap and carry on as
if the check had passed, instead of stopping the invocation. WGSL requires every
barrier to be in control flow its compiler proves uniform; an invocation that
leaves early makes everything after the check divergent, and the check is
usually in front of the barrier (a bounds check in a reduction loop). Carrying on
keeps the control flow as uniform as the source. It is safe on WebGPU because
every memory access is bounds-clamped by the implementation, and the throw is
still reported: the device reads the word when it next synchronizes. What the
invocation writes to the heap after the throw goes to a sink word instead
([`gatethrown!`](@ref)), so it stops where it threw as far as memory shows.

Before inlining, `_lava_signal_exception` is called from GPUCompiler's runtime
(`gpu_signal_exception`), which a throwing helper such as `throw_boundserror`
calls in turn; the call becomes the store of the word. The throwing blocks are
those calling a function that raises it, directly or not. Each branch into one
from a two-way predecessor is taken over by a block that stores the word and goes
on to that predecessor's other successor. A throwing block reached some other way
still stops the invocation, after its call has raised the word.
"""
function continuethrows!(mod::LLVM.Module)
    signals = exceptionsignals(mod)
    # Functions that throw whenever called: the signalling one, and every function
    # each of whose returns follows a call of one of those (`throw_boundserror`).
    # GPUCompiler has turned their `unreachable` into `ret` by now, so the returns
    # are what is left to tell them by.
    throwing_fns = Set{LLVM.Function}(LLVM.parent(LLVM.parent(c)) for c in signals)
    raises(bb) = any(i isa LLVM.CallInst && LLVM.called_operand(i) in throwing_fns
                     for i in LLVM.instructions(bb))
    grown = true
    while grown
        grown = false
        for fn in LLVM.functions(mod)
            (fn in throwing_fns || isempty(LLVM.blocks(fn))) && continue
            rets = [bb for bb in LLVM.blocks(fn) if LLVM.terminator(bb) isa LLVM.RetInst]
            if !isempty(rets) && all(raises, rets)
                push!(throwing_fns, fn)
                grown = true
            end
        end
    end
    T_i32 = LLVM.Int32Type()
    heapstore_ty = LLVM.FunctionType(LLVM.VoidType(), [T_i32, T_i32, T_i32])
    heapstore = get(LLVM.functions(mod), "_lava_wgpu_heapstore", nothing)
    heapstore === nothing && (heapstore = LLVM.Function(mod, "_lava_wgpu_heapstore", heapstore_ty))
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        for call in signals
            LLVM.position!(builder, insertion_point(call))
            LLVM.call!(builder, heapstore_ty, heapstore,
                       LLVM.Value[LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, WEBGPU_EXCEPTION_WORD), LLVM.ConstantInt(T_i32, 1)])
            LLVM.erase!(call)
        end
        # A throwing block calls a function that always throws; a call of one that
        # only might (it returns normally too) does not make its block throw.
        throwing = Tuple{LLVM.BasicBlock, LLVM.CallInst}[]
        for fn in LLVM.functions(mod)
            fn in throwing_fns && continue
            for bb in LLVM.blocks(fn)
                i = findfirst(inst -> inst isa LLVM.CallInst && LLVM.called_operand(inst) in throwing_fns,
                              collect(LLVM.instructions(bb)))
                i === nothing || push!(throwing, (bb, collect(LLVM.instructions(bb))[i]))
            end
        end
        thrown_ty = LLVM.FunctionType(LLVM.VoidType())
        thrown = get(LLVM.functions(mod), "_lava_wgpu_thrown", nothing)
        thrown === nothing && (thrown = LLVM.Function(mod, "_lava_wgpu_thrown", thrown_ty))
        function raise!()
            LLVM.call!(builder, heapstore_ty, heapstore,
                LLVM.Value[LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, WEBGPU_EXCEPTION_WORD), LLVM.ConstantInt(T_i32, 1)])
            # Where the invocation threw, for `gatethrown!` once everything is inlined.
            LLVM.call!(builder, thrown_ty, thrown, LLVM.Value[])
        end
        for (bb, _) in throwing
            continuethrow!(builder, bb, raise!)
        end
    end
    return nothing
end

"""
    gatethrown!(fn)

The second half of [`continuethrows!`](@ref), after inlining: an invocation that
threw writes nothing more to the heap. `fn` gets a flag, set where a throw carried
on (the `_lava_wgpu_thrown` calls), and every heap store and atomic takes the sink
word of heap 0 ([`WEBGPU_SINK_WORD`](@ref)) instead of its own once the flag is set. A
select, not a branch, so the control flow stays as uniform as the source. The
raise of the exception word itself is left alone. Workgroup memory is not gated:
an invocation that threw can still write shared memory its workgroup reads.
"""
function gatethrown!(fn::LLVM.Function)
    marks = [i for bb in LLVM.blocks(fn) for i in LLVM.instructions(bb)
             if i isa LLVM.CallInst && LLVM.name(LLVM.called_operand(i)) == "_lava_wgpu_thrown"]
    isempty(marks) && return nothing
    T_i1 = LLVM.Int1Type()
    T_i32 = LLVM.Int32Type()
    # The raise: heap 0, the exception word, both constant.
    raise(i) = all(k -> LLVM.operands(i)[k] isa LLVM.ConstantInt, 1:2) &&
        convert(Int, LLVM.operands(i)[1]) == 0 && convert(Int, LLVM.operands(i)[2]) == WEBGPU_EXCEPTION_WORD
    heapwrite(i) = i isa LLVM.CallInst &&
        (n = LLVM.name(LLVM.called_operand(i)); n == "_lava_wgpu_heapstore" || startswith(n, "_lava_wgpu_atomic_")) &&
        !raise(i)
    writes = [i for bb in LLVM.blocks(fn) for i in LLVM.instructions(bb) if heapwrite(i)]
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        LLVM.position!(builder, insertion_point(first(LLVM.instructions(first(LLVM.blocks(fn))))))
        flag = LLVM.alloca!(builder, T_i1, "thrown")
        LLVM.store!(builder, LLVM.ConstantInt(T_i1, 0), flag)
        for m in marks
            LLVM.position!(builder, insertion_point(m))
            LLVM.store!(builder, LLVM.ConstantInt(T_i1, 1), flag)
            LLVM.erase!(m)
        end
        for w in writes
            LLVM.position!(builder, insertion_point(w))
            t = LLVM.load!(builder, T_i1, flag, "thrown")
            heap = LLVM.select!(builder, t, LLVM.ConstantInt(T_i32, 0), LLVM.operands(w)[1])
            word = LLVM.select!(builder, t, LLVM.ConstantInt(T_i32, WEBGPU_SINK_WORD), LLVM.operands(w)[2])
            LLVM.operands(w)[1] = heap
            LLVM.operands(w)[2] = word
        end
    end
    return nothing
end

"""
Take every branch into the throwing block `bb` from a block with another successor
over by a block that raises the word (`raise!()`) and continues at that other
successor: a two-way branch's other side, a switch's default (a `DivideError`
check is `switch d [0 => throw, -1 => ...]`).
"""
function continuethrow!(builder::LLVM.IRBuilder, bb::LLVM.BasicBlock, raise!)
    fn = LLVM.parent(bb)
    preds = unique(collect(LLVM.predecessors(bb)))
    isempty(preds) && return nothing
    for pred in preds
        term = LLVM.terminator(pred)
        succs = collect(LLVM.successors(term))
        k = findfirst(!=(bb), succs)
        k === nothing && continue
        other = succs[k]
        resumed = LLVM.BasicBlock(fn, "throw.resume")
        LLVM.API.LLVMMoveBasicBlockAfter(resumed, pred)
        LLVM.position!(builder, insertion_point(resumed))
        raise!()
        LLVM.br!(builder, other)
        for k in eachindex(succs)
            succs[k] == bb && (LLVM.successors(term)[k] = resumed)
        end
        for phi in [i for i in LLVM.instructions(other) if i isa LLVM.PHIInst]
            incoming = LLVM.incoming(phi)
            push!(incoming, (first(v for (v, b) in incoming if b == pred), resumed))
        end
        for phi in [i for i in LLVM.instructions(bb) if i isa LLVM.PHIInst]
            drop_incoming!(phi, pred)
        end
    end
    isempty(LLVM.predecessors(bb)) && LLVM.erase!(bb)
    return nothing
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

