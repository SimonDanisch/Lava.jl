# SPIR-V Type Mapper: LLVM → SPIR-V type conversion with opaque pointer recovery.
#
# LLVM 18 uses opaque pointers (`ptr`) — SPIR-V requires typed pointers.
# Type recovery sources:
#   1. GEP source element type (LLVMGetGEPSourceElementType)
#   2. Load/store value types
#   3. Alloca allocated types
#   4. Global variable value types
#   5. Function parameter/return types
#
# The PointeeTypeMap maps each LLVM pointer value → its pointee type in SPIR-V.

# ================================================================
# Pointee Type Map — opaque pointer → SPIR-V typed pointer recovery
# ================================================================

"""
    PointeeTypeMap

Maps LLVM pointer values to their pointee LLVM types.
Built by scanning GEPs, loads, stores, allocas, globals before emission.

Uses priority levels so loads (which define the "real" type) win over stores
(which may use a different type due to bitwise optimization, e.g. `store i32 0`
instead of `store float 0.0`).
"""
struct PointeeTypeMap
    # LLVM Value (pointer) → (LLVM Type, priority)
    # Higher priority wins. Priorities: alloca/gep=3, load=2, store=1
    map::Dict{LLVM.Value, Tuple{LLVM.LLVMType, Int}}
end

PointeeTypeMap() = PointeeTypeMap(Dict{LLVM.Value, Tuple{LLVM.LLVMType, Int}}())

function set_pointee_type!(ptm::PointeeTypeMap, ptr::LLVM.Value, ty::LLVM.LLVMType; priority::Int=1)
    existing = get(ptm.map, ptr, nothing)
    if existing === nothing || priority >= existing[2]
        ptm.map[ptr] = (ty, priority)
    end
end

function get_pointee_type(ptm::PointeeTypeMap, ptr::LLVM.Value)
    entry = get(ptm.map, ptr, nothing)
    entry === nothing ? nothing : entry[1]
end

"""
    build_pointee_type_map(mod::LLVM.Module) -> PointeeTypeMap

Scan the entire LLVM module to recover pointee types for all pointer values.
"""
function build_pointee_type_map(mod::LLVM.Module)
    ptm = PointeeTypeMap()

    # 1. Global variables: their value type IS the pointee type
    for gv in LLVM.globals(mod)
        set_pointee_type!(ptm, gv, LLVM.global_value_type(gv); priority=3)
    end

    # 2. Scan all instructions in all functions
    for fn in LLVM.functions(mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                collect_pointee_types!(ptm, inst)
            end
        end
    end

    return ptm
end

function collect_pointee_types!(ptm::PointeeTypeMap, inst::LLVM.Instruction)
    # Dispatch on instruction type
    if inst isa LLVM.AllocaInst
        collect_alloca!(ptm, inst)
    elseif inst isa LLVM.LoadInst
        collect_load!(ptm, inst)
    elseif inst isa LLVM.StoreInst
        collect_store!(ptm, inst)
    elseif inst isa LLVM.GetElementPtrInst
        collect_gep!(ptm, inst)
    elseif inst isa LLVM.CallInst
        collect_call!(ptm, inst)
    elseif inst isa LLVM.BitCastInst
        collect_bitcast!(ptm, inst)
    elseif inst isa LLVM.IntToPtrInst
        # inttoptr — infer pointee type from users.
        # If used directly in load/store, those will set the type.
        # If used only via byte-offset GEPs, trace through to find the real type.
        collect_inttoptr!(ptm, inst)
    elseif inst isa LLVM.AtomicRMWInst
        collect_atomicrmw!(ptm, inst)
    elseif inst isa LLVM.AtomicCmpXchgInst
        collect_cmpxchg!(ptm, inst)
    end
end

function collect_alloca!(ptm::PointeeTypeMap, inst::LLVM.AllocaInst)
    # Alloca: the allocated type IS the definitive pointee type.
    # Priority 5 (highest) because the alloca's type is authoritative and must not
    # be overwritten by GEPs that use a sub-type as source element type (LLVM
    # allows GEPs with source type != alloca type when the first field is at offset 0).
    alloc_type = LLVM.LLVMType(API.LLVMGetAllocatedType(inst))
    set_pointee_type!(ptm, inst, alloc_type; priority=5)
end

function collect_load!(ptm::PointeeTypeMap, inst::LLVM.LoadInst)
    # Load: the loaded type is the pointee type of the pointer operand
    # Priority 2: loads reveal the "true" type the code uses
    ptr_operand = LLVM.operands(inst)[1]
    loaded_type = LLVM.value_type(inst)
    set_pointee_type!(ptm, ptr_operand, loaded_type; priority=2)

    # When the loaded value IS a pointer (loading a pointer from a struct/buffer),
    # trace through the loaded pointer's users to determine ITS pointee type.
    # Same logic as inttoptr: the loaded pointer may only be used via byte-offset GEPs.
    if loaded_type isa LLVM.PointerType
        infer_loaded_ptr_pointee!(ptm, inst)
    end
end

"""Infer the pointee type of a loaded pointer value by tracing its users."""
function infer_loaded_ptr_pointee!(ptm::PointeeTypeMap, load_inst::LLVM.LoadInst)
    # First try direct users (GEP, load, store-as-target, atomicrmw, cmpxchg)
    pointee = infer_pointee_from_users(load_inst)
    if pointee !== nothing
        set_pointee_type!(ptm, load_inst, pointee; priority=2)
        return
    end

    # If the loaded ptr is stored as VALUE into another pointer (e.g., alloca),
    # trace through: store val→alloca → load from alloca → check that load's users.
    # Pattern from SROA of struct containing pointer fields.
    for use in LLVM.uses(load_inst)
        user = LLVM.user(use)
        if user isa LLVM.StoreInst && LLVM.operands(user)[1] === load_inst
            store_target = LLVM.operands(user)[2]
            pointee = infer_type_through_alloca(store_target, load_inst)
            if pointee !== nothing
                set_pointee_type!(ptm, load_inst, pointee; priority=1)
                return
            end
        end
    end
end

"""Trace a pointer stored into an alloca (or GEP of alloca) to find its pointee type.
Follow: store ptr→target → load from target → check that load's users.
Also handles the case where store target is a GEP and the corresponding
load comes from a different GEP to the same alloca at the same byte offset."""
function infer_type_through_alloca(store_target::LLVM.Value, original_ptr::LLVM.Value)
    # Case 1: Direct loads from the same store target
    for use in LLVM.uses(store_target)
        user = LLVM.user(use)
        if user isa LLVM.LoadInst && LLVM.value_type(user) isa LLVM.PointerType
            # This load reads back the pointer. Check its users for type info.
            result = infer_pointee_from_users(user)
            result !== nothing && return result
        end
    end

    # Case 2: Store target is a GEP into an alloca.
    # The corresponding load may come from a DIFFERENT GEP path to the same alloca.
    # Walk GEP chain to find the base alloca, compute byte offset, then search all
    # ptr-type loads from the alloca at the same offset.
    if store_target isa LLVM.GetElementPtrInst
        base, offset = walk_gep_to_base(store_target)
        if base !== nothing && offset !== nothing
            result = find_ptr_load_at_offset(base, offset)
            result !== nothing && return result
        end
    end

    return nothing
end

"""Walk a chain of GEPs to find the base alloca and compute the byte offset."""
function walk_gep_to_base(gep::LLVM.GetElementPtrInst)
    total_offset = 0
    current = gep

    # Follow GEP chains (max 10 levels to avoid infinite loops)
    for _ in 1:10
        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(current))
        indices = LLVM.operands(current)[2:end]  # skip base pointer

        offset = compute_gep_byte_offset(src_ty, indices)
        offset === nothing && return (nothing, nothing)
        total_offset += offset

        base_ptr = LLVM.operands(current)[1]
        if base_ptr isa LLVM.AllocaInst
            return (base_ptr, total_offset)
        elseif base_ptr isa LLVM.GetElementPtrInst
            current = base_ptr
            continue
        else
            return (nothing, nothing)  # Not backed by an alloca
        end
    end
    return (nothing, nothing)
end

"""Compute the byte offset of a GEP given its source type and constant indices."""
function compute_gep_byte_offset(src_ty::LLVM.LLVMType, indices::Vector{<:LLVM.Value})
    offset = 0
    current_ty = src_ty
    dl = nothing  # We'll compute sizes manually

    for (i, idx_val) in enumerate(indices)
        # Index must be a constant integer
        if !(idx_val isa LLVM.ConstantInt)
            return nothing  # Non-constant index, can't compute offset
        end
        idx = convert(Int, idx_val)

        if i == 1
            # First index: scales by sizeof(src_ty)
            sz = approx_sizeof(current_ty)
            sz === nothing && return nothing
            offset += idx * sz
        else
            # Subsequent indices index into the current type
            if current_ty isa LLVM.StructType
                # Struct: sum field sizes up to idx
                for f in 0:(idx-1)
                    ft = LLVM.elements(current_ty)[f+1]
                    sz = approx_sizeof(ft)
                    sz === nothing && return nothing
                    offset += sz
                end
                current_ty = LLVM.elements(current_ty)[idx+1]
            elseif current_ty isa LLVM.ArrayType
                elem_ty = LLVM.eltype(current_ty)
                sz = approx_sizeof(elem_ty)
                sz === nothing && return nothing
                offset += idx * sz
                current_ty = elem_ty
            else
                return nothing
            end
        end
    end
    return offset
end

"""Approximate sizeof for LLVM types (no padding/alignment — good enough for offset matching)."""
function approx_sizeof(ty::LLVM.LLVMType)
    if ty isa LLVM.IntegerType
        return div(LLVM.width(ty) + 7, 8)
    elseif ty isa LLVM.LLVMHalf
        return 2
    elseif ty isa LLVM.LLVMFloat
        return 4
    elseif ty isa LLVM.LLVMDouble
        return 8
    elseif ty isa LLVM.PointerType
        return 8  # 64-bit pointers
    elseif ty isa LLVM.StructType
        total = 0
        for elem in LLVM.elements(ty)
            sz = approx_sizeof(elem)
            sz === nothing && return nothing
            total += sz
        end
        return total
    elseif ty isa LLVM.ArrayType
        n = LLVM.length(ty)
        elem_sz = approx_sizeof(LLVM.eltype(ty))
        elem_sz === nothing && return nothing
        return n * elem_sz
    elseif ty isa LLVM.VectorType
        n = LLVM.size(ty)
        elem_sz = approx_sizeof(LLVM.eltype(ty))
        elem_sz === nothing && return nothing
        return n * elem_sz
    else
        return nothing
    end
end

"""Find all ptr-type loads from an alloca at a specific byte offset,
and check if any reveal their pointee type from downstream usage."""
function find_ptr_load_at_offset(alloca::LLVM.AllocaInst, target_offset::Int)
    # Collect all GEPs and direct accesses from this alloca
    for use in LLVM.uses(alloca)
        user = LLVM.user(use)
        result = check_gep_or_load_at_offset(user, alloca, target_offset, 0)
        result !== nothing && return result
    end
    return nothing
end

"""Recursively check if a user of an alloca (or GEP chain) accesses a ptr at the target offset."""
function check_gep_or_load_at_offset(user::LLVM.Value, base::LLVM.Value, target_offset::Int, current_offset::Int)
    if user isa LLVM.LoadInst
        # Direct load from base at current_offset
        if current_offset == target_offset && LLVM.value_type(user) isa LLVM.PointerType
            result = infer_pointee_from_users(user)
            result !== nothing && return result
        end
    elseif user isa LLVM.GetElementPtrInst && LLVM.operands(user)[1] === base
        # GEP from base — compute offset and recurse into users
        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
        indices = LLVM.operands(user)[2:end]
        gep_offset = compute_gep_byte_offset(src_ty, indices)
        gep_offset === nothing && return nothing
        new_offset = current_offset + gep_offset

        for use2 in LLVM.uses(user)
            user2 = LLVM.user(use2)
            result = check_gep_or_load_at_offset(user2, user, target_offset, new_offset)
            result !== nothing && return result
        end
    end
    return nothing
end

"""Infer a pointer's pointee type from how it's used (GEP, load, store, atomic, etc.)."""
function infer_pointee_from_users(ptr_value::LLVM.Value)
    infer_pointee_from_users(ptr_value, Set{LLVM.Value}())
end

function infer_pointee_from_users(ptr_value::LLVM.Value, visited::Set{LLVM.Value})
    ptr_value in visited && return nothing
    push!(visited, ptr_value)
    # Two passes: first prefer struct GEPs (non-byte-offset) which give the real base type,
    # then fall back to byte-offset GEPs, loads, stores, atomics, PHIs.
    # This prevents byte-offset GEPs (accessing individual fields) from overriding the
    # struct source type when both exist on the same pointer.
    for use in LLVM.uses(ptr_value)
        user = LLVM.user(use)
        if user isa LLVM.GetElementPtrInst
            src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(user))
            if !(src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8)
                return src_ty
            end
        end
    end
    # Second pass: byte-offset GEPs, loads, stores, atomics, PHIs
    for use in LLVM.uses(ptr_value)
        user = LLVM.user(use)
        if user isa LLVM.GetElementPtrInst
            src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(user))
            if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8
                result = infer_type_from_gep_users(user)
                result !== nothing && return result
            end
        elseif user isa LLVM.LoadInst
            return LLVM.value_type(user)
        elseif user isa LLVM.StoreInst
            if LLVM.operands(user)[2] === ptr_value
                return LLVM.value_type(LLVM.operands(user)[1])
            end
        elseif user isa LLVM.AtomicRMWInst
            return LLVM.value_type(LLVM.operands(user)[2])
        elseif user isa LLVM.AtomicCmpXchgInst
            return LLVM.value_type(LLVM.operands(user)[2])
        elseif user isa LLVM.PHIInst
            result = infer_pointee_from_users(user, visited)
            result !== nothing && return result
        end
    end
    return nothing
end

function collect_store!(ptm::PointeeTypeMap, inst::LLVM.StoreInst)
    # Store: the stored value type is the pointee type of the pointer operand
    # Priority 1 (lowest): LLVM may optimize e.g. `store float 0.0` → `store i32 0`
    ops = LLVM.operands(inst)
    value = ops[1]
    ptr = ops[2]
    stored_type = LLVM.value_type(value)
    set_pointee_type!(ptm, ptr, stored_type; priority=1)
end

function collect_gep!(ptm::PointeeTypeMap, inst::LLVM.GetElementPtrInst)
    # GEP source element type tells us what the base pointer points to (highest priority)
    source_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(inst))
    base_ptr = LLVM.operands(inst)[1]

    # Byte-offset GEPs (source type = i8) carry no real type information.
    # LLVM uses `getelementptr i8, ptr %p, i64 <byte_offset>` for pointer arithmetic.
    # Don't let this overwrite the base pointer's actual type, and don't assign i8
    # as the result type. Downstream loads/stores will provide the real type.
    if source_ty isa LLVM.IntegerType && LLVM.width(source_ty) == 8
        # Don't set base_ptr type (would overwrite the real type)
        # Don't set result type (i8 is not the real pointee type)
        return
    end

    set_pointee_type!(ptm, base_ptr, source_ty; priority=3)

    # The GEP result itself is a pointer — compute its pointee type
    result_pointee = compute_gep_result_type(source_ty, inst)
    if result_pointee !== nothing
        set_pointee_type!(ptm, inst, result_pointee; priority=3)
    end
end

function collect_inttoptr!(ptm::PointeeTypeMap, inst::LLVM.IntToPtrInst)
    # inttoptr produces a pointer — infer type from users.
    # Direct loads/stores will set it via their own _collect functions.
    # But if the only user is a byte-offset GEP (source=i8), the type won't
    # be propagated back. Trace through byte-offset GEP chains to find
    # the eventual load/store type and assign it to the inttoptr result.
    for use in LLVM.uses(inst)
        user = LLVM.user(use)
        if user isa LLVM.GetElementPtrInst
            src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(user))
            if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8
                # Byte-offset GEP — trace to find actual type from load/store
                pointee = infer_type_from_gep_users(user)
                if pointee !== nothing
                    set_pointee_type!(ptm, inst, pointee; priority=1)
                    return
                end
            end
        end
    end
end

"""Infer the pointee type by looking at loads/stores that use a GEP result.
Follows chained byte-offset GEPs (i8 source type) to reach the final load/store."""
function infer_type_from_gep_users(gep::LLVM.GetElementPtrInst)
    for use in LLVM.uses(gep)
        user = LLVM.user(use)
        if user isa LLVM.LoadInst
            return LLVM.value_type(user)
        elseif user isa LLVM.StoreInst
            # Check if gep is the pointer (operand 2), not the value
            if LLVM.operands(user)[2] === gep
                return LLVM.value_type(LLVM.operands(user)[1])
            end
        elseif user isa LLVM.GetElementPtrInst
            # Follow chained byte-offset GEPs (common pattern: two i8 GEPs in sequence)
            sub_src = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(user))
            if sub_src isa LLVM.IntegerType && LLVM.width(sub_src) == 8
                result = infer_type_from_gep_users(user)
                result !== nothing && return result
            end
        end
    end
    return nothing
end

function collect_call!(ptm::PointeeTypeMap, inst::LLVM.CallInst)
    # Handle llvm.memcpy: propagate pointee type between dst and src
    # memcpy(dst, src, len, isvolatile) — if dst is an alloca with known type,
    # src gets the same type (and vice versa).
    called = LLVM.called_operand(inst)
    if called isa LLVM.Function
        fname = LLVM.name(called)
        if startswith(fname, "llvm.memcpy")
            ops = LLVM.operands(inst)
            dst = ops[1]
            src = ops[2]
            # If dst has a known type (e.g. alloca), propagate to src
            dst_ty = get_pointee_type(ptm, dst)
            if dst_ty !== nothing
                set_pointee_type!(ptm, src, dst_ty; priority=2)
            end
            # If src has a known type, propagate to dst
            src_ty = get_pointee_type(ptm, src)
            if src_ty !== nothing
                set_pointee_type!(ptm, dst, src_ty; priority=2)
            end
        end
    end
end

function collect_bitcast!(ptm::PointeeTypeMap, inst::LLVM.BitCastInst)
    # Bitcast: if source has a known pointee type, propagate to result
    src = LLVM.operands(inst)[1]
    src_ty = get_pointee_type(PointeeTypeMap(), src)  # will be filled in later passes
end

function collect_atomicrmw!(ptm::PointeeTypeMap, inst::LLVM.AtomicRMWInst)
    # AtomicRMW: pointer operand's pointee type = value type
    ops = LLVM.operands(inst)
    ptr = ops[1]
    val = ops[2]
    set_pointee_type!(ptm, ptr, LLVM.value_type(val); priority=2)
end

function collect_cmpxchg!(ptm::PointeeTypeMap, inst::LLVM.AtomicCmpXchgInst)
    # CmpXchg: pointer operand's pointee type = compare value type
    ops = LLVM.operands(inst)
    ptr = ops[1]
    cmp_val = ops[2]
    set_pointee_type!(ptm, ptr, LLVM.value_type(cmp_val); priority=2)
end

"""
    compute_gep_result_type(source_ty, gep_inst) -> Union{LLVMType, Nothing}

Compute the result pointee type of a GEP by walking its indices through the source type.

For `getelementptr T, ptr %base, i64 %idx1, i32 %idx2, ...`:
- First index: indexes into array of T, result is still T
- Subsequent indices: drill into nested types (struct fields, array elements)
"""
function compute_gep_result_type(source_ty::LLVM.LLVMType, inst::LLVM.GetElementPtrInst)
    ops = LLVM.operands(inst)
    compute_gep_result_type_from_ops(source_ty, ops)
end

function compute_gep_result_type(source_ty::LLVM.LLVMType, val::LLVM.ConstantExpr)
    ops = LLVM.operands(val)
    compute_gep_result_type_from_ops(source_ty, ops)
end

function compute_gep_result_type_from_ops(source_ty::LLVM.LLVMType, ops)
    # ops[1] = base pointer, ops[2..end] = indices
    n_indices = length(ops) - 1
    if n_indices == 0
        return source_ty
    end

    # First index just offsets the base pointer — result type is still source_ty
    if n_indices == 1
        return source_ty
    end

    # Walk subsequent indices through nested types
    current_ty = source_ty
    for i in 3:length(ops)  # skip base ptr and first index
        idx = ops[i]
        current_ty = index_into_type(current_ty, idx)
        if current_ty === nothing
            return nothing
        end
    end
    return current_ty
end

"""
    index_into_type(ty, idx) -> Union{LLVMType, Nothing}

Index into a composite type with a GEP index.
- Struct: index selects a field (must be constant)
- Array: index selects an element (can be variable)
"""
function index_into_type(ty::LLVM.LLVMType, idx::LLVM.Value)
    if ty isa LLVM.StructType
        # Struct field access — index must be a constant integer
        if idx isa LLVM.ConstantInt
            field_idx = convert(Int, idx) + 1  # Julia is 1-indexed
            elems = LLVM.elements(ty)
            if 1 <= field_idx <= length(elems)
                return elems[field_idx]
            end
        end
        return nothing
    elseif ty isa LLVM.ArrayType
        # Array element access — result is the element type
        return eltype(ty)
    elseif ty isa LLVM.VectorType
        return eltype(ty)
    else
        return nothing
    end
end

# ================================================================
# LLVM Type → SPIR-V Type Mapping
# ================================================================

"""
    SPIRVTypeContext

Holds type mapping state during LLVM → SPIR-V emission.
Maps LLVM types to SPIR-V type IDs, handling deduplication.
"""
struct SPIRVTypeContext
    # LLVM type → SPIR-V type ID
    llvm_to_spirv::Dict{LLVM.LLVMType, UInt32}
    # Pointer type dedup: (storage_class, pointee_spirv_id) → pointer type ID
    pointer_types::Dict{Tuple{UInt32, UInt32}, UInt32}
    # The SPIR-V module being built
    mod::SPIRVModule
    # Pointee type map for opaque pointer recovery
    ptm::PointeeTypeMap
    # Struct pointer member types: (struct_llvm_type, member_idx_0based) → (pointee_llvm_type, addr_space)
    # Built by scanning GEPs into structs to find how pointer members are used.
    struct_ptr_members::Dict{Tuple{LLVM.LLVMType, Int}, Tuple{LLVM.LLVMType, Int}}
    # Workgroup (shared memory) type map: LLVM type → fresh SPIR-V type ID.
    # Types used in Workgroup storage class must NOT have explicit layout decorations
    # (ArrayStride, MemberOffset, Block). When a type is used in BOTH PSB and Workgroup,
    # we create separate SPIR-V type IDs: the main cache gets decorations, this map doesn't.
    workgroup_type_map::Dict{LLVM.LLVMType, UInt32}
end

function SPIRVTypeContext(mod::SPIRVModule, ptm::PointeeTypeMap)
    SPIRVTypeContext(
        Dict{LLVM.LLVMType, UInt32}(),
        Dict{Tuple{UInt32, UInt32}, UInt32}(),
        mod,
        ptm,
        Dict{Tuple{LLVM.LLVMType, Int}, Tuple{LLVM.LLVMType, Int}}(),
        Dict{LLVM.LLVMType, UInt32}(),
    )
end

"""
    map_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType) -> UInt32

Map an LLVM type to a SPIR-V type ID, creating it if necessary.
"""
function map_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    # Check cache
    cached = get(ctx.llvm_to_spirv, ty, nothing)
    if cached !== nothing
        return cached
    end

    spirv_id = emit_llvm_type!(ctx, ty)
    ctx.llvm_to_spirv[ty] = spirv_id
    return spirv_id
end

"""
    map_workgroup_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType) -> UInt32

Create a fresh SPIR-V type ID for Workgroup (shared memory) usage, separate from the
main type cache. This is needed because the same LLVM type (e.g., `[3 x float]`) may be
used in both PSB (needs ArrayStride/MemberOffset) and Workgroup (must NOT have layout
decorations). By using separate SPIR-V IDs, PSB types get decorated while Workgroup
types remain undecorated.

For scalar/primitive types that don't need decorations, reuses the main cache.
Only creates fresh IDs for array and struct types that could receive layout decorations.
"""
function map_workgroup_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    # Check workgroup cache first
    cached = get(ctx.workgroup_type_map, ty, nothing)
    cached !== nothing && return cached

    # For types that never get layout decorations, reuse main cache
    if !(ty isa LLVM.ArrayType || ty isa LLVM.StructType)
        return map_type!(ctx, ty)
    end

    # Create fresh type ID (not stored in main cache)
    spirv_id = emit_workgroup_type!(ctx, ty)
    ctx.workgroup_type_map[ty] = spirv_id
    return spirv_id
end

function emit_workgroup_type!(ctx::SPIRVTypeContext, ty::LLVM.ArrayType)
    # [N x T] → OpTypeArray, with element mapped via workgroup path.
    # MUST bypass emit_type_array!'s type_cache to get a fresh ID —
    # the cache deduplicates on (element_id, length_id), which would return
    # the same type ID that PSB decorations are applied to.
    elem_spirv = map_workgroup_type!(ctx, eltype(ty))
    n = length(ty)
    len_id = emit_constant_u32!(ctx.mod, UInt32(n))
    id = fresh_id!(ctx.mod)
    encode_instruction!(ctx.mod.types_constants, Op.OpTypeArray, id, elem_spirv, len_id)

    # ArrayStride decoration is required by VK_KHR_workgroup_memory_explicit_layout
    # for ALL arrays inside Block-decorated structs, not just arrays of structs.
    elem_llvm = eltype(ty)
    stride = UInt32(wg_compute_type_size(elem_llvm))
    emit_decorate!(ctx.mod, id, Dec.ArrayStride, stride)

    return id
end

function emit_workgroup_type!(ctx::SPIRVTypeContext, ty::LLVM.StructType)
    # { T1, T2, ... } → OpTypeStruct, with members mapped via workgroup path.
    # MUST bypass emit_type_struct!'s type_cache for the same reason as arrays.
    member_types = LLVM.elements(ty)
    member_spirv_ids = UInt32[]
    for mt in member_types
        push!(member_spirv_ids, map_workgroup_type!(ctx, mt))
    end
    id = fresh_id!(ctx.mod)
    word_count = UInt32(2 + length(member_spirv_ids))
    push!(ctx.mod.types_constants, (word_count << 16) | UInt32(Op.OpTypeStruct))
    push!(ctx.mod.types_constants, id)
    append!(ctx.mod.types_constants, member_spirv_ids)

    # Add MemberOffset decorations for explicit layout (VK_KHR_workgroup_memory_explicit_layout).
    # Compute offsets matching LLVM's struct layout (with alignment padding).
    running_offset = UInt32(0)
    for (i, mt) in enumerate(member_types)
        member_align = UInt32(wg_compute_type_alignment(mt))
        running_offset = (running_offset + member_align - UInt32(1)) & ~(member_align - UInt32(1))
        emit_member_decorate!(ctx.mod, id, UInt32(i - 1), Dec.Offset, UInt32(running_offset))
        running_offset += wg_compute_type_size(mt)
    end

    return id
end

# Size/alignment helpers for workgroup explicit layout decorations.
# Mirror compute_type_size/compute_type_alignment from emit.jl but available in types.jl.
function wg_compute_type_size(ty::LLVM.LLVMType)
    if ty isa LLVM.LLVMFloat
        return UInt32(4)
    elseif ty isa LLVM.LLVMDouble
        return UInt32(8)
    elseif ty isa LLVM.LLVMHalf
        return UInt32(2)
    elseif ty isa LLVM.IntegerType
        return UInt32(max(1, LLVM.width(ty) ÷ 8))
    elseif ty isa LLVM.StructType
        total = UInt32(0)
        struct_align = UInt32(1)
        for elem in LLVM.elements(ty)
            elem_align = UInt32(wg_compute_type_alignment(elem))
            struct_align = max(struct_align, elem_align)
            total = (total + elem_align - 1) & ~(elem_align - 1)
            total += wg_compute_type_size(elem)
        end
        total = (total + struct_align - 1) & ~(struct_align - 1)
        return total
    elseif ty isa LLVM.ArrayType
        return UInt32(length(ty)) * wg_compute_type_size(eltype(ty))
    elseif ty isa LLVM.PointerType
        return UInt32(8)
    else
        return UInt32(4)
    end
end

function wg_compute_type_alignment(ty::LLVM.LLVMType)
    if ty isa LLVM.LLVMFloat
        return 4
    elseif ty isa LLVM.LLVMDouble
        return 8
    elseif ty isa LLVM.LLVMHalf
        return 2
    elseif ty isa LLVM.IntegerType
        return max(1, LLVM.width(ty) ÷ 8)
    elseif ty isa LLVM.StructType
        max_align = 1
        for elem in LLVM.elements(ty)
            max_align = max(max_align, wg_compute_type_alignment(elem))
        end
        return max_align
    elseif ty isa LLVM.ArrayType
        return wg_compute_type_alignment(eltype(ty))
    elseif ty isa LLVM.PointerType
        return 8
    else
        return 4
    end
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.VoidType)
    return emit_type_void!(ctx.mod)
end

"""
    spirv_int_width(llvm_width) -> UInt32

Round an LLVM integer bit-width to the nearest valid SPIR-V integer width.
LLVM allows arbitrary widths (i2, i3, i7, ...) but SPIR-V only supports 8, 16, 32, 64.
Width 1 (i1) is NOT handled here -- callers must map it to OpTypeBool separately.
"""
function spirv_int_width(w::Integer)::UInt32
    w <= 8  && return UInt32(8)
    w <= 16 && return UInt32(16)
    w <= 32 && return UInt32(32)
    return UInt32(64)
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.IntegerType)
    w = LLVM.width(ty)
    if w == 1
        # i1 → OpTypeBool
        return emit_type_bool!(ctx.mod)
    else
        # SPIR-V only supports 8/16/32/64-bit integers; round up non-standard widths
        return emit_type_int!(ctx.mod, spirv_int_width(w), UInt32(0))
    end
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMHalf)
    return emit_type_float!(ctx.mod, UInt32(16))
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMFloat)
    return emit_type_float!(ctx.mod, UInt32(32))
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMDouble)
    return emit_type_float!(ctx.mod, UInt32(64))
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.FloatingPointType)
    error("Unsupported floating point type: $(typeof(ty))")
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.ArrayType)
    # [N x T] → OpTypeArray
    elem_spirv = map_type!(ctx, eltype(ty))
    n = length(ty)
    # OpTypeArray needs the length as an OpConstant
    len_id = emit_constant_u32!(ctx.mod, UInt32(n))
    return emit_type_array!(ctx.mod, elem_spirv, len_id)
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.StructType)
    # { T1, T2, ... } → OpTypeStruct
    # Pointer members require special handling: LLVM uses opaque pointers (ptr)
    # but SPIR-V requires typed pointers. Use struct_ptr_members map to resolve.
    member_types = LLVM.elements(ty)
    member_spirv_ids = UInt32[]
    for (i, mt) in enumerate(member_types)
        if mt isa LLVM.PointerType
            push!(member_spirv_ids, map_struct_ptr_member!(ctx, ty, i - 1, mt))
        else
            push!(member_spirv_ids, map_type!(ctx, mt))
        end
    end
    return emit_type_struct!(ctx.mod, member_spirv_ids)
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.VectorType)
    # <N x T> → OpTypeVector
    elem_spirv = map_type!(ctx, eltype(ty))
    n = length(ty)
    return emit_type_vector!(ctx.mod, elem_spirv, UInt32(n))
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.PointerType)
    # Opaque pointer — need storage class from address space
    # Pointee type comes from PointeeTypeMap during emission
    # For now, return a placeholder that will be resolved per-use
    error("Cannot map opaque pointer type without context. Use map_pointer_type! instead.")
end

"""
    map_struct_ptr_member!(ctx, struct_ty, member_idx, ptr_ty) -> UInt32

Map a pointer member inside a struct type to a SPIR-V pointer type.
Uses the struct_ptr_members map (built by scanning GEPs) to recover the
pointee type. Falls back to PhysicalStorageBuffer pointer to i8 if unknown.
"""
function map_struct_ptr_member!(ctx::SPIRVTypeContext, struct_ty::LLVM.StructType,
                                  member_idx::Int, ptr_ty::LLVM.PointerType)
    as = LLVM.addrspace(ptr_ty)
    sc = llvm_addrspace_to_storage_class(as)
    # Non-alloca addrspace 0 pointers are PSB in our convention
    if sc == SC.Function
        sc = SC.PhysicalStorageBuffer
    end

    # Look up in the pre-built struct member map
    info = get(ctx.struct_ptr_members, (struct_ty, member_idx), nothing)
    if info !== nothing
        pointee_ty, _as = info
        pointee_spirv = map_type!(ctx, pointee_ty)
        return map_pointer_type!(ctx, pointee_spirv, sc)
    end

    # Fallback: PhysicalStorageBuffer pointer to i8 (generic byte pointer)
    i8_spirv = map_type!(ctx, LLVM.IntType(8))
    return map_pointer_type!(ctx, i8_spirv, sc)
end

"""
    build_struct_ptr_member_types!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)

Scan the module for GEPs into struct types that access pointer members.
For each such GEP, trace through loads to find how the extracted pointer is
used, then record the pointee type in ctx.struct_ptr_members.

This enables SPIR-V struct emission to use correct typed pointers for
pointer members (LLVM's opaque `ptr` → SPIR-V typed pointer).
"""
function build_struct_ptr_member_types!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)
    for fn in LLVM.functions(llvm_mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                if inst isa LLVM.GetElementPtrInst
                    scan_gep_for_struct_ptr_member!(ctx, inst)
                elseif inst isa LLVM.LoadInst
                    # Handle "load ptr, ptr %alloca" where alloca is a struct:
                    # the load extracts the first pointer member at offset 0.
                    scan_load_from_struct_base!(ctx, inst)
                end
            end
        end
    end

    # Second pass: resolve struct ptr members that couldn't be traced through
    # static GEP patterns (e.g., Tuple elements accessed via dynamic byte-GEPs).
    # Uses PTM entries for alloca-reloaded pointers to infer struct member types.
    resolve_unresolved_struct_ptr_members!(ctx, llvm_mod)
end

"""
Resolve struct pointer members that weren't found by the GEP/load scanning passes.

When structs containing pointers are stored into allocas and later accessed via
dynamic byte-offset GEPs (common for Tuple element indexing), the standard scanning
can't trace the pointer's type back to the struct member. But PTM *does* know the
type of the reloaded pointer (from its eventual `load float, ptr %reloaded` usage).

This pass finds such loads, walks the GEP chain to the alloca, identifies which
struct types with unresolved ptr members exist in the alloca's type, and resolves them.
"""
function resolve_unresolved_struct_ptr_members!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)
    for fn in LLVM.functions(llvm_mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.LoadInst || continue
                LLVM.value_type(inst) isa LLVM.PointerType || continue

                # Does PTM know this loaded pointer's pointee type?
                pointee = get_pointee_type(ctx.ptm, inst)
                pointee === nothing && continue
                # Skip i8 (the default fallback — not a real resolved type)
                pointee isa LLVM.IntegerType && LLVM.width(pointee) == 8 && continue

                # Walk GEP chain to find the base alloca
                alloca = find_alloca_base(LLVM.operands(inst)[1])
                alloca === nothing && continue

                alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                resolve_array_struct_ptr_members!(ctx, alloca_ty, pointee)
            end
        end
    end
end

"""Walk a GEP chain (including byte-offset GEPs) to find the base AllocaInst."""
function find_alloca_base(ptr::LLVM.Value)
    current = ptr
    for _ in 1:20
        if current isa LLVM.AllocaInst
            return current
        elseif current isa LLVM.GetElementPtrInst
            current = LLVM.operands(current)[1]
        else
            return nothing
        end
    end
    return nothing
end

"""
Resolve unresolved struct ptr members within array contexts of a type hierarchy.

Only resolves ptr members in structs that appear inside arrays (e.g., `[3 x {ptr, ...}]`),
because dynamic byte-GEPs are generated specifically for array element access. Structs
accessed via static GEPs should already be resolved by the main scan.
"""
function resolve_array_struct_ptr_members!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType, pointee::LLVM.LLVMType;
                                             in_array::Bool=false)
    if ty isa LLVM.StructType
        for (i, field_ty) in enumerate(LLVM.elements(ty))
            if field_ty isa LLVM.PointerType && in_array
                key = (ty, i - 1)
                if !haskey(ctx.struct_ptr_members, key)
                    ctx.struct_ptr_members[key] = (pointee, 0)
                end
            elseif field_ty isa LLVM.StructType || field_ty isa LLVM.ArrayType
                resolve_array_struct_ptr_members!(ctx, field_ty, pointee; in_array)
            end
        end
    elseif ty isa LLVM.ArrayType
        resolve_array_struct_ptr_members!(ctx, LLVM.eltype(ty), pointee; in_array=true)
    end
end

"""
Handle the case where a pointer is loaded directly from a struct base address
(no GEP). LLVM generates `load ptr, ptr %alloca` which reads the first 8 bytes —
the pointer field of the first nested struct member at offset 0.

Walk the struct layout following member 0 to find the first pointer member,
then trace the loaded pointer's users to determine its pointee type.
"""
function scan_load_from_struct_base!(ctx::SPIRVTypeContext, load_inst::LLVM.LoadInst)
    load_ty = LLVM.value_type(load_inst)
    load_ty isa LLVM.PointerType || return

    # Get the source pointer and find its pointee type
    src_ptr = LLVM.operands(load_inst)[1]
    src_pointee = get_pointee_type(ctx.ptm, src_ptr)
    src_pointee === nothing && return
    (src_pointee isa LLVM.StructType || src_pointee isa LLVM.ArrayType) || return

    # Walk struct layout following member 0 until we find a pointer field
    struct_ty, member_idx = find_offset0_ptr_member(src_pointee)
    struct_ty === nothing && return

    key = (struct_ty, member_idx)
    haskey(ctx.struct_ptr_members, key) && return  # Already resolved

    # Trace the loaded pointer's users to determine pointee type
    for use in LLVM.uses(load_inst)
        user = LLVM.user(use)
        if user isa LLVM.GetElementPtrInst
            sub_src = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(user))
            if sub_src isa LLVM.IntegerType && LLVM.width(sub_src) == 8
                # Byte-offset GEP — trace through to find real type
                pointee = infer_type_from_gep_users(user)
                if pointee !== nothing
                    ctx.struct_ptr_members[key] = (pointee, 0)
                    return
                end
            else
                # Typed GEP (including non-byte integer types like i32)
                # The source element type IS the pointee type
                ctx.struct_ptr_members[key] = (sub_src, 0)
                return
            end
        elseif user isa LLVM.StoreInst
            if LLVM.operands(user)[2] === load_inst
                val_ty = LLVM.value_type(LLVM.operands(user)[1])
                ctx.struct_ptr_members[key] = (val_ty, 0)
                return
            end
        elseif user isa LLVM.AtomicRMWInst || user isa LLVM.AtomicCmpXchgInst
            # atomicrmw add ptr %loaded_ptr, i32 1 → pointee is i32
            val_ty = LLVM.value_type(LLVM.operands(user)[2])
            ctx.struct_ptr_members[key] = (val_ty, 0)
            return
        elseif user isa LLVM.LoadInst
            # load i32, ptr %loaded_ptr → pointee is i32
            ld_ty = LLVM.value_type(user)
            if !(ld_ty isa LLVM.PointerType)
                ctx.struct_ptr_members[key] = (ld_ty, 0)
                return
            end
        end
    end
end

"""
Walk a struct layout following member 0 at each level until finding a pointer field.
Returns (containing_struct_type, member_index) or (nothing, 0).
"""
function find_offset0_ptr_member(ty::LLVM.LLVMType)
    current = ty
    while true
        if current isa LLVM.StructType
            members = LLVM.elements(current)
            isempty(members) && return (nothing, 0)
            first_member = members[1]
            if first_member isa LLVM.PointerType
                return (current, 0)  # Found it: member 0 of this struct is a pointer
            end
            current = first_member
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return (nothing, 0)
            current = LLVM.eltype(current)
        else
            return (nothing, 0)  # Hit a scalar, no pointer at offset 0
        end
    end
end

function scan_gep_for_struct_ptr_member!(ctx::SPIRVTypeContext, gep::LLVM.GetElementPtrInst)
    src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(gep))

    ops = LLVM.operands(gep)
    # Need at least: base_ptr, first_idx, and one more index
    length(ops) >= 3 || return

    # Walk the GEP indices to find the FINAL struct + member index.
    # Multi-level GEPs like (0, 0, 0, 1, 0) drill through struct→struct→array→struct→ptr.
    current_ty = src_ty
    final_struct_ty = nothing
    final_member_idx = -1

    # ops[1] = base_ptr, ops[2] = first index (array deref), ops[3..end] = type indices
    n_ops = length(ops)
    for i in 3:n_ops  # Skip base_ptr (ops[1]) and first index (ops[2])
        idx_val = ops[i]
        idx_val isa LLVM.ConstantInt || return
        idx = convert(Int, idx_val)

        if current_ty isa LLVM.StructType
            members = LLVM.elements(current_ty)
            (idx + 1) <= length(members) || return
            next_ty = members[idx + 1]
            # Record struct + member index at every level
            final_struct_ty = current_ty
            final_member_idx = idx
            current_ty = next_ty
        elseif current_ty isa LLVM.ArrayType
            current_ty = LLVM.eltype(current_ty)
        else
            return  # Can't index further
        end
    end

    # The final accessed type must be a pointer
    current_ty isa LLVM.PointerType || return
    final_struct_ty === nothing && return

    as = LLVM.addrspace(current_ty)

    key = (final_struct_ty, final_member_idx)
    haskey(ctx.struct_ptr_members, key) && return  # Already resolved

    trace_gep_ptr_users!(ctx, gep, key, as)
end

"""
Trace a GEP's users (loads → downstream usage) to determine the pointee type
of a pointer member accessed by the GEP.
"""
function trace_gep_ptr_users!(ctx::SPIRVTypeContext, gep::LLVM.Value,
                                key::Tuple{LLVM.StructType, Int}, as::Int)
    for use in LLVM.uses(gep)
        user = LLVM.user(use)
        if user isa LLVM.LoadInst
            # The load gives us the pointer value — check PTM for its pointee type
            pointee = get_pointee_type(ctx.ptm, user)
            if pointee !== nothing
                ctx.struct_ptr_members[key] = (pointee, as)
                return
            end
            # Also check: how is the loaded pointer used? (byte-offset GEP → load/store)
            for load_use in LLVM.uses(user)
                load_user = LLVM.user(load_use)
                if load_user isa LLVM.GetElementPtrInst
                    sub_src = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(load_user))
                    if sub_src isa LLVM.IntegerType && LLVM.width(sub_src) == 8
                        # Byte-offset GEP — trace through to find real type
                        pointee = infer_type_from_gep_users(load_user)
                        if pointee !== nothing
                            ctx.struct_ptr_members[key] = (pointee, as)
                            return
                        end
                    else
                        # Typed GEP (including non-byte integers like i32)
                        ctx.struct_ptr_members[key] = (sub_src, as)
                        return
                    end
                elseif load_user isa LLVM.StoreInst
                    # Pointer used directly in store (as the pointer operand)
                    if LLVM.operands(load_user)[2] === user
                        val_ty = LLVM.value_type(LLVM.operands(load_user)[1])
                        ctx.struct_ptr_members[key] = (val_ty, as)
                        return
                    end
                elseif load_user isa LLVM.AtomicRMWInst || load_user isa LLVM.AtomicCmpXchgInst
                    # atomicrmw/cmpxchg on loaded pointer → value type is pointee type
                    val_ty = LLVM.value_type(LLVM.operands(load_user)[2])
                    ctx.struct_ptr_members[key] = (val_ty, as)
                    return
                elseif load_user isa LLVM.LoadInst
                    # load from loaded pointer → load type is pointee type
                    ld_ty = LLVM.value_type(load_user)
                    if !(ld_ty isa LLVM.PointerType)
                        ctx.struct_ptr_members[key] = (ld_ty, as)
                        return
                    end
                end
            end
        end
    end
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.FunctionType)
    # Function types: return type + param types
    ret_ty = map_type!(ctx, LLVM.return_type(ty))
    param_types = UInt32[]
    for pt in LLVM.parameters(ty)
        push!(param_types, map_type!(ctx, pt))
    end
    return emit_type_function!(ctx.mod, ret_ty, param_types)
end

function emit_llvm_type!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    error("Unsupported LLVM type: $(typeof(ty))")
end

# ================================================================
# Pointer Type Mapping (address space → storage class)
# ================================================================

"""
    llvm_addrspace_to_storage_class(addrspace::Int) -> UInt32

Map LLVM address space to SPIR-V storage class.
"""
function llvm_addrspace_to_storage_class(addrspace::Int)
    if addrspace == 0
        return SC.Function
    elseif addrspace == 1
        return SC.PhysicalStorageBuffer  # BDA pointers
    elseif addrspace == 2
        return SC.PushConstant
    elseif addrspace == 3
        return SC.Workgroup  # Shared memory
    elseif addrspace == 7
        return SC.Input  # Built-in variables
    else
        error("Unknown LLVM address space: $addrspace")
    end
end

"""
    map_pointer_type!(ctx::SPIRVTypeContext, pointee_spirv_id::UInt32, storage_class::UInt32) -> UInt32

Get or create a SPIR-V pointer type for the given pointee type and storage class.
"""
function map_pointer_type!(ctx::SPIRVTypeContext, pointee_spirv_id::UInt32, storage_class::UInt32)
    key = (storage_class, pointee_spirv_id)
    get!(ctx.pointer_types, key) do
        emit_type_pointer!(ctx.mod, storage_class, pointee_spirv_id)
    end
end

"""
Infer a loaded pointer's pointee type by examining the GEP source type hierarchy.

When a `load ptr` comes from a GEP into a struct/array, the GEP's resolved
pointee type tells us what type of struct the ptr field belongs to. We walk down
the type hierarchy to find the first struct with a ptr member and look it up
in struct_ptr_members.

Handles SROA memcpy patterns: `load ptr, ptr (gep {outer_struct}, %bda_arg, 0, N)`
where the GEP result type is `[K x {ptr, ...}]` or `{ptr, ...}` and the loaded
ptr is a struct member with a declared pointee type.
"""
function infer_ptr_type_from_source_gep(ctx::SPIRVTypeContext, ptr_value::LLVM.Value)
    ptr_value isa LLVM.LoadInst || return nothing
    loaded_ty = LLVM.value_type(ptr_value)
    loaded_ty isa LLVM.PointerType || return nothing

    src_ptr = LLVM.operands(ptr_value)[1]
    src_pointee = get_pointee_type(ctx.ptm, src_ptr)
    src_pointee === nothing && return nothing

    # Walk down to find the innermost struct with a ptr member
    return find_ptr_member_type_in_hierarchy(ctx, src_pointee)
end

"""Walk type hierarchy (arrays, nested structs) to find a struct with a ptr member
whose pointee type is declared in struct_ptr_members.
Falls back to i8 if the struct has a ptr member but no declared type (e.g., ptr
is passed through but never dereferenced)."""
function find_ptr_member_type_in_hierarchy(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    if ty isa LLVM.ArrayType
        return find_ptr_member_type_in_hierarchy(ctx, LLVM.eltype(ty))
    elseif ty isa LLVM.StructType
        for (i, ft) in enumerate(LLVM.elements(ty))
            if ft isa LLVM.PointerType
                info = get(ctx.struct_ptr_members, (ty, i - 1), nothing)
                if info !== nothing
                    return info[1]  # declared pointee type
                end
                # Struct has a ptr member but no declared type — default to i8
                return LLVM.Int8Type()
            end
        end
        # Recurse into first non-ptr member in case of nested structs
        for ft in LLVM.elements(ty)
            if ft isa LLVM.StructType || ft isa LLVM.ArrayType
                result = find_ptr_member_type_in_hierarchy(ctx, ft)
                result !== nothing && return result
            end
        end
    end
    return nothing
end

"""
    map_pointer_type_for_value!(ctx::SPIRVTypeContext, ptr_value::LLVM.Value) -> UInt32

Map a pointer value to its SPIR-V pointer type, using the PointeeTypeMap for type recovery.
"""
function map_pointer_type_for_value!(ctx::SPIRVTypeContext, ptr_value::LLVM.Value)
    # Get pointee type from the map
    pointee_llvm = get_pointee_type(ctx.ptm, ptr_value)
    if pointee_llvm === nothing
        # Fallback: if this is a loaded ptr, trace through GEP source types
        # and struct_ptr_members to find the pointee type.
        # Handles SROA memcpy patterns where ptr is loaded from a BDA struct
        # but only stored into an alloca (no downstream GEP/load to infer from).
        pointee_llvm = infer_ptr_type_from_source_gep(ctx, ptr_value)
    end
    if pointee_llvm === nothing
        # Last resort fallbacks:
        if ptr_value isa LLVM.LoadInst && LLVM.value_type(ptr_value) isa LLVM.PointerType
            # Loaded pointers that are only stored (never dereferenced) → i8
            pointee_llvm = LLVM.Int8Type()
        elseif ptr_value isa LLVM.BitCastInst
            # Pointer bitcast (typepun): try to get pointee from source operand
            src_op = LLVM.operands(ptr_value)[1]
            if LLVM.value_type(src_op) isa LLVM.PointerType
                src_pointee = get_pointee_type(ctx.ptm, src_op)
                if src_pointee !== nothing
                    pointee_llvm = src_pointee
                else
                    pointee_llvm = LLVM.Int8Type()
                end
            else
                pointee_llvm = LLVM.Int8Type()
            end
        else
            error("Could not recover pointee type for pointer value: $(LLVM.name(ptr_value))")
        end
    end

    # Get storage class from address space
    ptr_ty = LLVM.value_type(ptr_value)
    if !(ptr_ty isa LLVM.PointerType)
        error("Expected pointer type, got: $(typeof(ptr_ty))")
    end
    as = LLVM.addrspace(ptr_ty)
    sc = llvm_addrspace_to_storage_class(as)

    # Override: In Vulkan SPIR-V, function parameters and GEP results in addrspace 0
    # should be PhysicalStorageBuffer (device memory), not Function.
    # Only allocas genuinely produce Function storage class pointers.
    if sc == SC.Function && !(ptr_value isa LLVM.AllocaInst)
        sc = SC.PhysicalStorageBuffer
    end

    # Map pointee type and create pointer type
    pointee_spirv = map_type!(ctx, pointee_llvm)
    return map_pointer_type!(ctx, pointee_spirv, sc)
end

# ================================================================
# SPIR-V Constant Mapping
# ================================================================

"""
    map_constant!(ctx::SPIRVTypeContext, val::LLVM.Constant) -> UInt32

Map an LLVM constant to a SPIR-V constant ID.
"""
function map_constant!(ctx::SPIRVTypeContext, val::LLVM.Constant)
    ty = LLVM.value_type(val)

    if val isa LLVM.ConstantInt
        return map_constant_int!(ctx, val, ty)
    elseif val isa LLVM.ConstantFP
        return map_constant_fp!(ctx, val, ty)
    elseif val isa LLVM.UndefValue || val isa LLVM.PoisonValue
        return map_undef!(ctx, ty)
    elseif val isa LLVM.ConstantAggregateZero
        return map_null_constant!(ctx, ty)
    elseif val isa LLVM.ConstantArray || val isa LLVM.ConstantDataArray
        return map_constant_array!(ctx, val, ty)
    else
        error("Unsupported constant type: $(typeof(val))")
    end
end

function map_constant_int!(ctx::SPIRVTypeContext, val::LLVM.ConstantInt, ty::LLVM.IntegerType)
    w = LLVM.width(ty)
    type_id = map_type!(ctx, ty)
    int_val = convert(Int64, val)

    if w == 1
        # Boolean constant
        # OpConstantTrue (opcode 41) / OpConstantFalse (opcode 42)
        id = fresh_id!(ctx.mod)
        if int_val != 0
            encode_instruction!(ctx.mod.types_constants, UInt16(41), type_id, id)  # OpConstantTrue
        else
            encode_instruction!(ctx.mod.types_constants, UInt16(42), type_id, id)  # OpConstantFalse
        end
        return id
    elseif w <= 32
        # Create constant with the CORRECT type (not always i32).
        # SPIR-V requires type-matched operands in comparisons.
        # Mask to actual bit width — SPIR-V requires high bits to be 0 for unsigned
        # integer types (Signedness=0), which is what we emit.
        bits = UInt32(int_val & ((UInt64(1) << w) - 1))
        key = (:const, type_id, bits)
        return get!(ctx.mod.constant_cache, key) do
            id = fresh_id!(ctx.mod)
            encode_instruction!(ctx.mod.types_constants, Op.OpConstant, type_id, id, bits)
            id
        end
    elseif w == 64
        # 64-bit constant: two UInt32 words (little-endian)
        bits = reinterpret(UInt64, Int64(int_val))
        lo = UInt32(bits & 0xFFFFFFFF)
        hi = UInt32((bits >> 32) & 0xFFFFFFFF)
        key = (:const, type_id, bits)
        get!(ctx.mod.constant_cache, key) do
            id = fresh_id!(ctx.mod)
            encode_instruction!(ctx.mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
            id
        end
    else
        error("Unsupported integer width: $w")
    end
end

function map_constant_fp!(ctx::SPIRVTypeContext, val::LLVM.ConstantFP, ty::LLVM.LLVMFloat)
    fval = convert(Float32, val)
    return emit_constant_f32!(ctx.mod, fval)
end

function map_constant_fp!(ctx::SPIRVTypeContext, val::LLVM.ConstantFP, ty::LLVM.LLVMDouble)
    type_id = map_type!(ctx, ty)
    fval = convert(Float64, val)
    bits = reinterpret(UInt64, fval)
    lo = UInt32(bits & 0xFFFFFFFF)
    hi = UInt32((bits >> 32) & 0xFFFFFFFF)
    key = (:const, type_id, bits)
    get!(ctx.mod.constant_cache, key) do
        id = fresh_id!(ctx.mod)
        encode_instruction!(ctx.mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
        id
    end
end

function map_constant_fp!(ctx::SPIRVTypeContext, val::LLVM.ConstantFP, ty::LLVM.LLVMHalf)
    type_id = map_type!(ctx, ty)
    fval = convert(Float16, val)
    bits = UInt32(reinterpret(UInt16, fval))
    key = (:const, type_id, bits)
    get!(ctx.mod.constant_cache, key) do
        id = fresh_id!(ctx.mod)
        encode_instruction!(ctx.mod.types_constants, Op.OpConstant, type_id, id, bits)
        id
    end
end

function map_constant_fp!(ctx::SPIRVTypeContext, val::LLVM.ConstantFP, ty::LLVM.FloatingPointType)
    error("Unsupported float constant type: $(typeof(ty))")
end

function map_undef!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    type_id = map_type!(ctx, ty)
    # OpUndef (opcode 1)
    id = fresh_id!(ctx.mod)
    encode_instruction!(ctx.mod.types_constants, UInt16(1), type_id, id)
    return id
end

function map_null_constant!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    type_id = map_type!(ctx, ty)
    # OpConstantNull (opcode 46)
    key = (:null, type_id)
    get!(ctx.mod.constant_cache, key) do
        id = fresh_id!(ctx.mod)
        encode_instruction!(ctx.mod.types_constants, UInt16(46), type_id, id)
        id
    end
end

function map_constant_array!(ctx::SPIRVTypeContext, val::Union{LLVM.ConstantArray, LLVM.ConstantDataArray}, ty::LLVM.LLVMType)
    type_id = map_type!(ctx, ty)
    elem_ty = LLVM.eltype(ty)
    n = length(ty)
    elem_ids = UInt32[]

    if val isa LLVM.ConstantDataArray
        # ConstantDataArray stores data as packed scalars — extract element-by-element
        for i in 0:(n-1)
            # Use LLVM API to get element at index
            elem_val = LLVM.API.LLVMGetElementAsConstant(val, Cuint(i))
            elem = LLVM.Value(elem_val)
            elem_id = map_constant!(ctx, elem)
            push!(elem_ids, elem_id)
        end
    else
        # ConstantArray stores elements as operands
        for i in 0:(n-1)
            elem = LLVM.operands(val)[i + 1]
            elem_id = map_constant!(ctx, elem)
            push!(elem_ids, elem_id)
        end
    end

    id = fresh_id!(ctx.mod)
    encode_instruction!(ctx.mod.types_constants, Op.OpConstantComposite, type_id, id, elem_ids...)
    return id
end

# ================================================================
# Runtime Array Type (SPIR-V-specific, no LLVM equivalent)
# ================================================================

"""
    emit_runtime_array_type!(mod::SPIRVModule, element_type_id::UInt32; stride::UInt32) -> UInt32

Emit OpTypeRuntimeArray with ArrayStride decoration.
Used for StorageBuffer data arrays.
"""
function emit_runtime_array_type!(mod::SPIRVModule, element_type_id::UInt32; stride::UInt32)
    key = (:runtime_array, element_type_id)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        # OpTypeRuntimeArray (opcode 29)
        encode_instruction!(mod.types_constants, UInt16(29), id, element_type_id)
        emit_decorate!(mod, id, Dec.ArrayStride, stride)
        id
    end
end

# ================================================================
# Struct Layout Decorations
# ================================================================

"""
    decorate_struct_layout!(ctx::SPIRVTypeContext, struct_spirv_id::UInt32, struct_llvm_ty::LLVM.StructType, dl::LLVM.DataLayout)

Add MemberOffset decorations to a SPIR-V struct type based on LLVM DataLayout.
"""
function decorate_struct_layout!(ctx::SPIRVTypeContext, struct_spirv_id::UInt32, struct_llvm_ty::LLVM.StructType, dl::LLVM.DataLayout)
    n_members = length(LLVM.elements(struct_llvm_ty))
    for i in 0:(n_members - 1)
        offset = API.LLVMOffsetOfElement(dl, struct_llvm_ty, UInt32(i))
        emit_member_decorate!(ctx.mod, struct_spirv_id, UInt32(i), Dec.Offset, UInt32(offset))
    end
end

"""
    decorate_psb_struct_layouts!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)

Add Block + MemberOffset decorations to ALL struct types (and nested struct types)
that are accessed via PhysicalStorageBuffer pointers. Required by Vulkan SPIR-V
scalar block layout rules.

Scans the PTM for pointer values in addrspace 1 (PSB) or addrspace 0 non-allocas
(which map to PSB), collects their struct pointee types, and recursively decorates
them and all nested structs.
"""
function decorate_psb_struct_layouts!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)
    dl = LLVM.datalayout(llvm_mod)
    decorated_structs = Set{UInt32}()
    decorated_arrays = Set{UInt32}()

    # Collect DIRECT PSB pointee types (for Block decoration)
    # and ALL nested types (for MemberOffset/ArrayStride decoration)
    direct_psb_struct_types = Set{LLVM.StructType}()  # Only top-level → get Block
    all_psb_struct_types = Set{LLVM.StructType}()       # All nested → get MemberOffset
    all_psb_array_types = Set{LLVM.ArrayType}()          # All nested → get ArrayStride

    for (ptr_val, (pointee_ty, _priority)) in ctx.ptm.map
        is_psb_value(ptr_val) || continue
        # Top-level pointee struct gets Block
        if pointee_ty isa LLVM.StructType
            push!(direct_psb_struct_types, pointee_ty)
        end
        # Collect ALL nested types for layout decorations
        collect_nested_types_for_psb!(all_psb_struct_types, all_psb_array_types, pointee_ty)
    end

    # Collect structs that are nested inside other structs (directly or via arrays).
    # These must NOT get Block decoration, because a Block cannot be nested inside another Block.
    nested_structs = Set{LLVM.StructType}()
    for llvm_sty in all_psb_struct_types
        collect_nested_member_structs!(nested_structs, llvm_sty)
    end
    # Also mark structs that are elements of PSB arrays as nested —
    # SPIR-V forbids ArrayStride on arrays whose element has Block decoration.
    for llvm_aty in all_psb_array_types
        elem_ty = eltype(llvm_aty)
        if elem_ty isa LLVM.StructType
            push!(nested_structs, elem_ty)
        end
    end

    # Decorate ALL nested struct types with MemberOffset
    # Only DIRECT PSB pointees that are NOT nested inside any other struct get Block
    for llvm_sty in all_psb_struct_types
        spirv_id = get(ctx.llvm_to_spirv, llvm_sty, nothing)
        spirv_id === nothing && continue
        spirv_id in decorated_structs && continue
        push!(decorated_structs, spirv_id)

        # Block only for top-level PSB pointees that aren't nested in another struct
        if llvm_sty in direct_psb_struct_types && !(llvm_sty in nested_structs)
            emit_decorate!(ctx.mod, spirv_id, Dec.Block)
        end
        decorate_struct_layout!(ctx, spirv_id, llvm_sty, dl)
    end

    # Decorate each array type with ArrayStride
    for llvm_aty in all_psb_array_types
        spirv_id = get(ctx.llvm_to_spirv, llvm_aty, nothing)
        spirv_id === nothing && continue
        spirv_id in decorated_arrays && continue
        push!(decorated_arrays, spirv_id)

        elem_ty = eltype(llvm_aty)
        # Use ABI size for stride — includes alignment padding
        stride = UInt32(API.LLVMABISizeOfType(dl, elem_ty))
        emit_decorate!(ctx.mod, spirv_id, Dec.ArrayStride, stride)
    end
end

"""Check if an LLVM value is a PSB pointer (addrspace 1, or addrspace 0 non-alloca)."""
function is_psb_value(val::LLVM.Value)
    ty = LLVM.value_type(val)
    ty isa LLVM.PointerType || return false
    as = LLVM.addrspace(ty)
    as == 1 && return true
    # addrspace 0 non-allocas map to PSB in our convention
    as == 0 && !(val isa LLVM.AllocaInst) && return true
    return false
end

"""Collect all struct types that are members of `sty` (directly or via arrays)."""
function collect_nested_member_structs!(nested::Set{LLVM.StructType}, sty::LLVM.StructType)
    for elem in LLVM.elements(sty)
        collect_nested_member_structs_inner!(nested, elem)
    end
end

function collect_nested_member_structs_inner!(nested::Set{LLVM.StructType}, ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        ty in nested && return
        push!(nested, ty)
        for elem in LLVM.elements(ty)
            collect_nested_member_structs_inner!(nested, elem)
        end
    elseif ty isa LLVM.ArrayType
        collect_nested_member_structs_inner!(nested, eltype(ty))
    end
end

"""Recursively collect all struct and array types within a type (for PSB decoration)."""
function collect_nested_types_for_psb!(structs::Set{LLVM.StructType},
                                         arrays::Set{LLVM.ArrayType},
                                         ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        ty in structs && return
        push!(structs, ty)
        for elem in LLVM.elements(ty)
            collect_nested_types_for_psb!(structs, arrays, elem)
        end
    elseif ty isa LLVM.ArrayType
        if !(ty in arrays)
            push!(arrays, ty)
            collect_nested_types_for_psb!(structs, arrays, eltype(ty))
        end
    end
end

"""Compute byte size of an LLVM type using DataLayout for accurate struct sizes."""
function compute_type_size_with_dl(ty::LLVM.LLVMType, dl::LLVM.DataLayout)
    if ty isa LLVM.StructType
        # Use DataLayout for accurate struct size (includes padding)
        n = length(LLVM.elements(ty))
        n == 0 && return UInt32(0)
        last_offset = API.LLVMOffsetOfElement(dl, ty, UInt32(n - 1))
        last_elem = collect(LLVM.elements(ty))[n]
        return UInt32(last_offset + compute_type_size_with_dl(last_elem, dl))
    elseif ty isa LLVM.ArrayType
        return UInt32(length(ty)) * compute_type_size_with_dl(eltype(ty), dl)
    elseif ty isa LLVM.IntegerType
        return UInt32(max(1, LLVM.width(ty) ÷ 8))
    elseif ty isa LLVM.LLVMFloat
        return UInt32(4)
    elseif ty isa LLVM.LLVMDouble
        return UInt32(8)
    elseif ty isa LLVM.LLVMHalf
        return UInt32(2)
    elseif ty isa LLVM.PointerType
        return UInt32(8)
    else
        return UInt32(4)
    end
end

# ================================================================
# Collect all types from an LLVM module
# ================================================================

"""
    collect_module_types!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)

Pre-collect and map all types used in the LLVM module to SPIR-V types.
This ensures type IDs are allocated before instruction emission begins.
"""
function collect_module_types!(ctx::SPIRVTypeContext, llvm_mod::LLVM.Module)
    # Collect types from global variables
    for gv in LLVM.globals(llvm_mod)
        collect_value_type!(ctx, gv)
    end

    # Collect types from all functions
    for fn in LLVM.functions(llvm_mod)
        isempty(LLVM.blocks(fn)) && continue  # Skip declarations (intrinsics, etc.)

        # Function return type
        fn_ty = LLVM.function_type(fn)
        ret_ty = LLVM.return_type(fn_ty)
        if !(ret_ty isa LLVM.PointerType)
            map_type!(ctx, ret_ty)
        end

        # Parameter types (skip pointer types — handled per-use)
        for param in LLVM.parameters(fn)
            collect_value_type!(ctx, param)
        end

        # Instruction result types
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                collect_value_type!(ctx, inst)
            end
        end
    end
end

function collect_value_type!(ctx::SPIRVTypeContext, val::LLVM.Value)
    ty = LLVM.value_type(val)
    if ty isa LLVM.PointerType
        # Pointer type — don't map directly, handled per-use via PointeeTypeMap
        return
    end
    collect_type_recursive!(ctx, ty)
end

function collect_type_recursive!(ctx::SPIRVTypeContext, ty::LLVM.LLVMType)
    if ty isa LLVM.PointerType
        return  # Pointer types are mapped per-use
    end
    map_type!(ctx, ty)
end
