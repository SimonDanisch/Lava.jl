# The bindings of a WebGPU module and the intrinsics that read and write them
# (`device/heap.jl`, `wrap_entry_for_webgpu!`).
#
#   set 0, binding 0  args   { uint data[]; }  StorageBuffer, NonWritable   the arguments
#   set 0, binding 1  heaps  { uint data[]; }[WEBGPU_MAX_HEAPS]  StorageBuffer  device memory
#
# A heap access indexes the array of heaps by the pointer's heap, a value: the
# module does not depend on how many heaps a device has. WGSL has no arrays of
# bindings, so the printer declares the device's heaps as bindings 1, 2, ... and
# reaches them through functions that switch on the heap.
#
# Atomics act on the heaps themselves. WGSL atomics need `atomic<u32>` storage, and
# WebGPU forbids a second binding overlapping a writable one ("Encoder bind groups
# alias a writable resource"), so there is no atomic view beside the plain one: the
# WGSL printer declares the heaps of a kernel that uses atomics `array<atomic<u32>>`,
# and that kernel's every heap access is atomic.
#
# Each block is its own struct type: `NonWritable` is a member decoration, and the
# type cache would give both variables one type.

"""The most heaps a module's array of heaps has; a device binds at most this many."""
const WEBGPU_MAX_HEAPS = 16
const WEBGPU_FRAGMENT_ARGS_BINDING = WEBGPU_MAX_HEAPS + 1

"""A `Block` struct of one runtime array of 32-bit words, not shared with any other variable."""
function word_block_type!(mod::SPIRVModule, uint::UInt32; readonly::Bool)
    words = emit_runtime_array_type!(mod, uint; stride = UInt32(4))
    block = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeStruct, block, words)
    emit_decorate!(mod, block, Dec.Block)
    emit_member_decorate!(mod, block, UInt32(0), Dec.Offset, UInt32(0))
    readonly && emit_member_decorate!(mod, block, UInt32(0), Dec.NonWritable)
    return block
end

"""A StorageBuffer variable of `type` at `binding` of set 0."""
function word_block_variable!(mod::SPIRVModule, type::UInt32, binding::Integer, name::String)
    ptr = emit_type_pointer!(mod, SC.StorageBuffer, type)
    var = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr, var, SC.StorageBuffer)
    emit_decorate!(mod, var, Dec.DescriptorSet, UInt32(0))
    emit_decorate!(mod, var, Dec.Binding, UInt32(binding))
    emit_name!(mod, var, name)
    return var
end

"""
    emit_webgpu_bindings!(state) -> Vector{UInt32}

Declare the argument binding and the array of heaps, record them in `state.webgpu`
and return their ids for the entry point's interface.
"""
function emit_webgpu_bindings!(state::SPIRVEmitterState; argbinding::Int = 0)
    mod = state.mod
    uint = emit_type_int!(mod, UInt32(32), UInt32(0))
    args = word_block_variable!(mod, word_block_type!(mod, uint; readonly = true), argbinding, "args")
    nheaps = emit_constant_int!(mod, uint, 32, UInt64(WEBGPU_MAX_HEAPS))
    heaptype = emit_type_array!(mod, word_block_type!(mod, uint; readonly = false), nheaps)
    heaps = word_block_variable!(mod, heaptype, 1, "heaps")
    require_capability!(mod, Cap.StorageBufferArrayDynamicIndexing)
    state.webgpu = WebGPUBindings(heaps, args, uint, emit_type_pointer!(mod, SC.StorageBuffer, uint))
    return UInt32[args, heaps]
end

"""Pointer to word `index` (an LLVM i32) of the argument block."""
function arg_word_pointer!(state::SPIRVEmitterState, index::LLVM.Value)
    b = state.webgpu
    zero = emit_constant_int!(state.mod, b.uint, 32, UInt64(0))
    ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpAccessChain, b.heap_word_ptr, ptr, b.args, zero,
                        get_value_id!(state, index))
    return ptr
end

"""Pointer to word `index` of heap `heap` (LLVM i32s): `heaps[heap].data[index]`."""
function heap_word_pointer!(state::SPIRVEmitterState, heap::LLVM.Value, index::LLVM.Value)
    b = state.webgpu
    zero = emit_constant_int!(state.mod, b.uint, 32, UInt64(0))
    ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpAccessChain, b.heap_word_ptr, ptr, b.heaps,
                        get_value_id!(state, heap), zero, get_value_id!(state, index))
    return ptr
end

function emit_lava_wgpu!(state::SPIRVEmitterState, inst::LLVM.CallInst, name::String)
    state.webgpu === nothing &&
        error("$name in a module that is not compiled for WebGPU (TargetFeatures(webgpu = true))")
    b = state.webgpu
    ops = inst.operands
    if name == "_lava_wgpu_heapload" || name == "_lava_wgpu_argload"
        ptr = name == "_lava_wgpu_argload" ? arg_word_pointer!(state, ops[1]) :
                                             heap_word_pointer!(state, ops[1], ops[2])
        result = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpLoad, b.uint, result, ptr)
        state.value_map[inst] = result
    elseif name == "_lava_wgpu_heapstore"
        ptr = heap_word_pointer!(state, ops[1], ops[2])
        encode_instruction!(state.mod.functions, Op.OpStore, ptr, get_value_id!(state, ops[3]))
    elseif startswith(name, "_lava_wgpu_atomic_")
        emit_heap_atomic!(state, inst, name[length("_lava_wgpu_atomic_") + 1:end])
    else
        error("unknown WebGPU intrinsic $name")
    end
    return nothing
end

# Opcodes of the heap atomics, by the intrinsic's suffix.
const WGPU_ATOMIC_RMW = Dict("add" => Op.OpAtomicIAdd, "sub" => Op.OpAtomicISub,
    "and" => Op.OpAtomicAnd, "or" => Op.OpAtomicOr, "xor" => Op.OpAtomicXor,
    "xchg" => Op.OpAtomicExchange, "umin" => Op.OpAtomicUMin, "umax" => Op.OpAtomicUMax)

"""
An atomic on word `ops[2]` of heap `ops[1]`, device scope, relaxed: WGSL atomics
have no ordering beyond relaxed, and the barriers order memory.
"""
function emit_heap_atomic!(state::SPIRVEmitterState, inst::LLVM.CallInst, op::String)
    b = state.webgpu
    mod = state.mod
    require_capability!(mod, Cap.VulkanMemoryModelDeviceScope)
    ops = inst.operands
    ptr = heap_word_pointer!(state, ops[1], ops[2])
    scope = emit_constant_int!(mod, b.uint, 32, UInt64(1))       # Device
    relaxed = emit_constant_int!(mod, b.uint, 32, UInt64(0))
    if op == "store"
        encode_instruction!(mod.functions, Op.OpAtomicStore, ptr, scope, relaxed, get_value_id!(state, ops[3]))
        return nothing
    elseif op == "storepart"
        encode_instruction!(mod.functions, Op.OpAtomicAnd, b.uint, fresh_id!(mod), ptr, scope, relaxed,
                            get_value_id!(state, ops[3]))
        encode_instruction!(mod.functions, Op.OpAtomicOr, b.uint, fresh_id!(mod), ptr, scope, relaxed,
                            get_value_id!(state, ops[4]))
        return nothing
    end
    result = fresh_id!(mod)
    if op == "load"
        encode_instruction!(mod.functions, Op.OpAtomicLoad, b.uint, result, ptr, scope, relaxed)
    elseif op == "cas"
        encode_instruction!(mod.functions, Op.OpAtomicCompareExchange, b.uint, result, ptr, scope,
                            relaxed, relaxed, get_value_id!(state, ops[4]), get_value_id!(state, ops[3]))
    else
        encode_instruction!(mod.functions, WGPU_ATOMIC_RMW[op], b.uint, result, ptr, scope, relaxed,
                            get_value_id!(state, ops[3]))
    end
    state.value_map[inst] = result
    return nothing
end
