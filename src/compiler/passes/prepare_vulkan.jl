# LLVM pass: Prepare module for Vulkan SPIR-V emission.
#
# Ported from Abacus compilation.jl (_prepare_module_for_vulkan! and sub-passes).
# This is the final LLVM-level pass before the custom SPIR-V emitter runs.
#
# Sub-passes:
# 1. Strip lifetime intrinsics + set internal linkage + fix PSB alignment
# 2. _fix_shared_geps! -- fix GEPs on addrspace(3) shared memory globals
# 3. _flatten_bda_array_geps! -- replace composite-typed GEPs in addrspace(1) with ptr arithmetic
# 4. _lower_psb_memops! -- lower memset/memcpy on PSB pointers to explicit stores/loads
# 5. _decompose_composite_psb_accesses! -- decompose struct load/store on PSB to scalar ops
# 6. _warn_constant_globals! -- warn about invalid constant globals in addrspace(1)
#
# NOTE: We do NOT include _hoist_push_constant_loads! from Abacus -- that was
# an llc workaround for ConstantExpr GEP mishandling. The custom emitter handles
# push constant access directly.

# ============================================================================
# Utility functions
# ============================================================================

"""Get the size in bytes of an LLVM type (packed, no padding between fields)."""
function _llvm_type_size(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return div(LLVM.width(t) + 7, 8)
    elseif t == LLVM.FloatType()
        return 4
    elseif t == LLVM.DoubleType()
        return 8
    elseif t == LLVM.HalfType()
        return 2
    elseif t isa LLVM.ArrayType
        return length(t) * _llvm_type_size(LLVM.eltype(t))
    elseif t isa LLVM.StructType
        total = 0
        for m in LLVM.elements(t)
            total += _llvm_type_size(m)
        end
        return total
    else
        # Fallback: assume 8 bytes (pointer-sized)
        return 8
    end
end

"""Return the size in bytes of the largest scalar component in a type.

Used to compute minimum alignment for Vulkan PhysicalStorageBuffer accesses
(VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314).
"""
function _scalar_size(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return div(LLVM.width(t) + 7, 8)
    elseif t == LLVM.FloatType()
        return 4
    elseif t == LLVM.DoubleType()
        return 8
    elseif t == LLVM.HalfType()
        return 2
    elseif t isa LLVM.StructType
        sz = 0
        for m in LLVM.elements(t)
            sz = max(sz, _scalar_size(m))
        end
        return sz
    elseif t isa LLVM.ArrayType || t isa LLVM.VectorType
        return _scalar_size(LLVM.eltype(t))
    elseif t isa LLVM.PointerType
        return 8
    else
        return 4
    end
end

# ============================================================================
# Composite PSB load/store helpers
# ============================================================================

"""
    _load_composite_from_psb(builder, base_int, type, byte_offset, dl)

Recursively load a composite type from PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: inttoptr(base + offset) -> load.
For struct/array types: recursively load members and reconstruct via insertvalue.
"""
function _load_composite_from_psb(builder, base_int, type::LLVM.LLVMType,
                                   byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        result = LLVM.UndefValue(type)
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = _load_composite_from_psb(builder, base_int, member_type,
                                                   byte_offset + member_offset, dl)
            result = LLVM.insert_value!(builder, result, member_val, i)
        end
        return result
    elseif type isa LLVM.ArrayType
        result = LLVM.UndefValue(type)
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = _load_composite_from_psb(builder, base_int, elem_type,
                                                 byte_offset + i * elem_size, dl)
            result = LLVM.insert_value!(builder, result, elem_val, i)
        end
        return result
    else
        # Leaf scalar: inttoptr(base + offset) -> load
        if byte_offset == 0
            field_addr = base_int
        else
            field_addr = LLVM.add!(builder, base_int,
                LLVM.ConstantInt(T_i64, byte_offset))
        end
        field_ptr = LLVM.inttoptr!(builder, field_addr, T_ptr_as1)
        val = LLVM.load!(builder, type, field_ptr)
        LLVM.alignment!(val, max(4, _llvm_type_size(type)))
        return val
    end
end

"""
    _store_composite_to_psb(builder, val, base_int, type, byte_offset, dl)

Recursively store a composite type to PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: extractvalue -> inttoptr(base + offset) -> store.
For struct/array types: recursively extract and store members.
"""
function _store_composite_to_psb(builder, val, base_int, type::LLVM.LLVMType,
                                  byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_psb(builder, member_val, base_int, member_type,
                                     byte_offset + member_offset, dl)
        end
    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_psb(builder, elem_val, base_int, elem_type,
                                     byte_offset + i * elem_size, dl)
        end
    else
        # Leaf scalar: inttoptr(base + offset) -> store
        if byte_offset == 0
            field_addr = base_int
        else
            field_addr = LLVM.add!(builder, base_int,
                LLVM.ConstantInt(T_i64, byte_offset))
        end
        field_ptr = LLVM.inttoptr!(builder, field_addr, T_ptr_as1)
        st = LLVM.store!(builder, val, field_ptr)
        LLVM.alignment!(st, max(4, _llvm_type_size(type)))
    end
end

# ============================================================================
# Sub-pass: Lower PSB memops
# ============================================================================

"""
    _lower_psb_memops!(mod::LLVM.Module)

Lower `llvm.memset` and `llvm.memcpy` intrinsics on `ptr addrspace(1)` to
explicit store/load loops. SPIR-V doesn't support these intrinsics on
PhysicalStorageBuffer pointers.

Memset is lowered to i32 stores (4-byte aligned) + i8 tail stores.
Memcpy is lowered to i32 load/store pairs + i8 tail.
Only handles constant-length operations (common for struct initialization).
"""
function _lower_psb_memops!(mod::LLVM.Module)
    T_i8 = LLVM.Int8Type()
    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(T_i8, 1)

    for f in LLVM.functions(mod)
        for bb in LLVM.blocks(f)
            to_erase = LLVM.Instruction[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                callee = LLVM.called_operand(inst)
                callee isa LLVM.Function || continue
                cname = LLVM.name(callee)

                if startswith(cname, "llvm.memset.p1")
                    # llvm.memset.p1.iN(ptr as(1) dst, i8 val, iN len, i1 volatile)
                    ops = LLVM.operands(inst)
                    dst_ptr = ops[1]
                    fill_val = ops[2]  # i8
                    len_val = ops[3]

                    # Only handle constant-length memsets
                    len_val isa LLVM.ConstantInt || continue
                    nbytes = convert(Int, len_val)

                    LLVM.IRBuilder() do builder
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, dst_ptr, T_i64, "memset.base")

                        # Build fill word: replicate i8 val to i32
                        val8 = fill_val
                        val32 = LLVM.zext!(builder, val8, T_i32)
                        v1 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 8))
                        val32 = LLVM.or!(builder, val32, v1)
                        v2 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 16))
                        val32 = LLVM.or!(builder, val32, v2)

                        # Store i32s for the bulk
                        n_words = nbytes ÷ 4
                        for i in 0:(n_words-1)
                            off = i * 4
                            addr = if off == 0
                                base_int
                            else
                                LLVM.add!(builder, base_int, LLVM.ConstantInt(T_i64, off))
                            end
                            ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                            st = LLVM.store!(builder, val32, ptr)
                            LLVM.alignment!(st, 4)
                        end

                        # Handle tail bytes
                        for i in (n_words*4):(nbytes-1)
                            addr = LLVM.add!(builder, base_int, LLVM.ConstantInt(T_i64, i))
                            ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                            st = LLVM.store!(builder, val8, ptr)
                            LLVM.alignment!(st, 1)
                        end
                    end
                    push!(to_erase, inst)

                elseif startswith(cname, "llvm.memcpy.p1") ||
                       startswith(cname, "llvm.memcpy.p0.p1") ||
                       startswith(cname, "llvm.memcpy.p1.p0")
                    # Lower memcpy involving addrspace(1)
                    ops = LLVM.operands(inst)
                    dst_ptr = ops[1]
                    src_ptr = ops[2]
                    len_val = ops[3]

                    len_val isa LLVM.ConstantInt || continue
                    nbytes = convert(Int, len_val)

                    LLVM.IRBuilder() do builder
                        LLVM.position!(builder, inst)
                        dst_int = LLVM.ptrtoint!(builder, dst_ptr, T_i64, "memcpy.dst")
                        src_int = LLVM.ptrtoint!(builder, src_ptr, T_i64, "memcpy.src")

                        dst_as = LLVM.addrspace(LLVM.value_type(dst_ptr))
                        src_as = LLVM.addrspace(LLVM.value_type(src_ptr))
                        T_dst_ptr = LLVM.PointerType(T_i8, dst_as)
                        T_src_ptr = LLVM.PointerType(T_i8, src_as)

                        # Copy i32 words
                        n_words = nbytes ÷ 4
                        for i in 0:(n_words-1)
                            off = i * 4
                            s_addr = off == 0 ? src_int : LLVM.add!(builder, src_int, LLVM.ConstantInt(T_i64, off))
                            s_ptr = LLVM.inttoptr!(builder, s_addr, T_src_ptr)
                            val = LLVM.load!(builder, T_i32, s_ptr)
                            LLVM.alignment!(val, 4)

                            d_addr = off == 0 ? dst_int : LLVM.add!(builder, dst_int, LLVM.ConstantInt(T_i64, off))
                            d_ptr = LLVM.inttoptr!(builder, d_addr, T_dst_ptr)
                            st = LLVM.store!(builder, val, d_ptr)
                            LLVM.alignment!(st, 4)
                        end

                        # Copy tail bytes
                        for i in (n_words*4):(nbytes-1)
                            s_addr = LLVM.add!(builder, src_int, LLVM.ConstantInt(T_i64, i))
                            s_ptr = LLVM.inttoptr!(builder, s_addr, T_src_ptr)
                            val = LLVM.load!(builder, T_i8, s_ptr)
                            LLVM.alignment!(val, 1)

                            d_addr = LLVM.add!(builder, dst_int, LLVM.ConstantInt(T_i64, i))
                            d_ptr = LLVM.inttoptr!(builder, d_addr, T_dst_ptr)
                            st = LLVM.store!(builder, val, d_ptr)
                            LLVM.alignment!(st, 1)
                        end
                    end
                    push!(to_erase, inst)
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end

    # Clean up now-unused memset/memcpy declarations
    for f in collect(LLVM.functions(mod))
        fname = LLVM.name(f)
        if (startswith(fname, "llvm.memset.p1") || startswith(fname, "llvm.memcpy.p1") ||
            startswith(fname, "llvm.memcpy.p0.p1") || startswith(fname, "llvm.memcpy.p1.p0"))
            if isempty(LLVM.blocks(f)) && isempty(LLVM.uses(f))
                LLVM.erase!(f)
            end
        end
    end
end

# ============================================================================
# Sub-pass: Decompose composite PSB accesses
# ============================================================================

"""
    _decompose_composite_psb_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)

Decompose composite (struct/array) loads/stores on `ptr addrspace(1)` into
individual scalar loads/stores with `inttoptr(base + offset)`.

The SPIR-V emitter needs scalar-level access for PhysicalStorageBuffer pointers.
Composite loads like `load %large_struct, ptr addrspace(1)` must be decomposed
into individual scalar loads, reconstructed via insertvalue. Similarly for stores.
"""
function _decompose_composite_psb_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                # Handle composite loads from addrspace(1)
                if inst isa LLVM.LoadInst
                    loaded_type = LLVM.value_type(inst)
                    (loaded_type isa LLVM.StructType || loaded_type isa LLVM.ArrayType) || continue

                    ptr = LLVM.operands(inst)[1]
                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 1 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, ptr, T_i64, "psb_base")
                        result = _load_composite_from_psb(builder, base_int,
                                                           loaded_type, 0, dl)
                        LLVM.replace_uses!(inst, result)
                    end
                    push!(to_erase, inst)

                # Handle composite stores to addrspace(1)
                elseif inst isa LLVM.StoreInst
                    ops = LLVM.operands(inst)
                    val = ops[1]
                    ptr = ops[2]
                    stored_type = LLVM.value_type(val)
                    (stored_type isa LLVM.StructType || stored_type isa LLVM.ArrayType) || continue

                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 1 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, ptr, T_i64, "psb_st_base")
                        _store_composite_to_psb(builder, val, base_int,
                                                 stored_type, 0, dl)
                    end
                    push!(to_erase, inst)
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

# ============================================================================
# Sub-pass: Decompose composite Workgroup (shared memory) accesses
# ============================================================================

"""
    _decompose_composite_workgroup_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)

Decompose composite (struct/array) loads and stores on addrspace(3) (Workgroup/shared
memory) into per-field scalar operations. SPIR-V shared memory is flattened to scalar
arrays, so composite accesses must be broken down.

Also handles type-punned loads from allocas: `load i64, ptr %alloca_of_struct`
where LLVM's memcpy optimization reads raw bytes from a struct alloca.
These are replaced by loading the first scalar field and zero-extending.
"""
function _decompose_composite_workgroup_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                # Handle composite loads from addrspace(3)
                if inst isa LLVM.LoadInst
                    loaded_type = LLVM.value_type(inst)
                    (loaded_type isa LLVM.StructType || loaded_type isa LLVM.ArrayType) || continue

                    ptr = LLVM.operands(inst)[1]
                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 3 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        result = _load_composite_from_wg(builder, ptr, loaded_type, 0, dl)
                        LLVM.replace_uses!(inst, result)
                    end
                    push!(to_erase, inst)

                # Handle composite stores to addrspace(3)
                elseif inst isa LLVM.StoreInst
                    ops = LLVM.operands(inst)
                    val = ops[1]
                    ptr = ops[2]
                    stored_type = LLVM.value_type(val)
                    (stored_type isa LLVM.StructType || stored_type isa LLVM.ArrayType) || continue

                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 3 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        _store_composite_to_wg(builder, val, ptr, stored_type, 0, dl)
                    end
                    push!(to_erase, inst)
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

# Use byte-offset GEPs (getelementptr i8) to avoid ConstantExpr issues with globals
function _load_composite_from_wg(builder, base_ptr, type::LLVM.LLVMType,
                                  byte_offset::Int, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    if type isa LLVM.StructType
        result = LLVM.UndefValue(type)
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = _load_composite_from_wg(builder, base_ptr, member_type,
                                                  byte_offset + member_offset, dl)
            result = LLVM.insert_value!(builder, result, member_val, i)
        end
        return result
    elseif type isa LLVM.ArrayType
        result = LLVM.UndefValue(type)
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = _load_composite_from_wg(builder, base_ptr, elem_type,
                                                byte_offset + i * elem_size, dl)
            result = LLVM.insert_value!(builder, result, elem_val, i)
        end
        return result
    else
        # Leaf scalar: GEP i8 to byte offset, then load
        field_ptr = if byte_offset == 0
            base_ptr
        else
            LLVM.gep!(builder, T_i8, base_ptr,
                       [LLVM.ConstantInt(T_i64, byte_offset)], "wg_field_ptr")
        end
        val = LLVM.load!(builder, type, field_ptr)
        LLVM.alignment!(val, max(1, _llvm_type_size(type)))
        return val
    end
end

function _store_composite_to_wg(builder, val, base_ptr, type::LLVM.LLVMType,
                                 byte_offset::Int, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_wg(builder, member_val, base_ptr, member_type,
                                    byte_offset + member_offset, dl)
        end
    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_wg(builder, elem_val, base_ptr, elem_type,
                                    byte_offset + i * elem_size, dl)
        end
    else
        # Leaf scalar: GEP i8 to byte offset, then store
        field_ptr = if byte_offset == 0
            base_ptr
        else
            LLVM.gep!(builder, T_i8, base_ptr,
                       [LLVM.ConstantInt(T_i64, byte_offset)], "wg_field_ptr")
        end
        st = LLVM.store!(builder, val, field_ptr)
        LLVM.alignment!(st, max(1, _llvm_type_size(type)))
    end
end

# ============================================================================
# Sub-pass: Decompose workgroup typepun copies
# ============================================================================

# Resolve the type that a pointer in addrspace(3) actually points to,
# walking through ConstantExpr GEPs, GEP instructions, and global variables.
# In LLVM GEP, the first index is pointer arithmetic (doesn't change type),
# and only subsequent indices drill into the type hierarchy.
function _resolve_wg_ptr_type(ptr::LLVM.Value)
    if ptr isa LLVM.GlobalVariable
        return LLVM.global_value_type(ptr)
    elseif ptr isa LLVM.ConstantExpr
        opcode = LLVM.API.LLVMGetConstOpcode(ptr)
        if opcode == LLVM.API.LLVMGetElementPtr
            source_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
            n_ops = Int(LLVM.API.LLVMGetNumOperands(ptr))
            current_ty = source_ty
            # Skip operand 0 (base pointer) and operand 1 (first index = pointer arithmetic)
            for i in 2:(n_ops - 1)
                idx_val = LLVM.Value(LLVM.API.LLVMGetOperand(ptr, i))
                if current_ty isa LLVM.ArrayType
                    current_ty = LLVM.eltype(current_ty)
                elseif current_ty isa LLVM.StructType
                    if idx_val isa LLVM.ConstantInt
                        idx = convert(Int, idx_val)
                        members = LLVM.elements(current_ty)
                        current_ty = members[idx + 1]
                    else
                        return nothing
                    end
                else
                    return nothing
                end
            end
            return current_ty
        end
    elseif ptr isa LLVM.GetElementPtrInst
        source_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
        ops = LLVM.operands(ptr)
        current_ty = source_ty
        # Skip operand 1 (base pointer) and operand 2 (first index = pointer arithmetic)
        for i in 3:length(ops)
            idx_val = ops[i]
            if current_ty isa LLVM.ArrayType
                current_ty = LLVM.eltype(current_ty)
            elseif current_ty isa LLVM.StructType
                if idx_val isa LLVM.ConstantInt
                    idx = convert(Int, idx_val)
                    members = LLVM.elements(current_ty)
                    current_ty = members[idx + 1]
                else
                    return nothing
                end
            else
                return nothing
            end
        end
        return current_ty
    end
    return nothing
end

# Get the struct type from a type, unwrapping arrays.
function _unwrap_to_struct(ty::LLVM.LLVMType)
    current = ty
    while current isa LLVM.ArrayType
        current = LLVM.eltype(current)
    end
    return current isa LLVM.StructType ? current : nothing
end

# Find struct fields fully contained in a byte range [start_byte, end_byte).
function _fields_in_byte_range(struct_ty::LLVM.StructType, dl::LLVM.DataLayout,
                                start_byte::Int, end_byte::Int)
    fields = Tuple{Int, LLVM.LLVMType, Int}[]  # (0-based index, type, byte offset)
    n = Int(LLVM.API.LLVMCountStructElementTypes(struct_ty))
    for i in 0:(n - 1)
        field_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, struct_ty, i))
        field_ty = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(struct_ty, i))
        field_size = Int(LLVM.storage_size(dl, field_ty))
        if field_offset >= start_byte && (field_offset + field_size) <= end_byte
            push!(fields, (i, field_ty, field_offset))
        end
    end
    return fields
end

"""
    _lift_byte_geps_on_workgroup_globals!(mod::LLVM.Module, dl::LLVM.DataLayout)

Convert byte-offset ConstantExpr GEPs on workgroup globals to typed struct-member GEPs.

LLVM may generate `getelementptr i8, ptr addrspace(3) @shared, i64 <offset>` to access
struct fields in shared memory. The SPIR-V emitter doesn't handle byte-offset ConstantExpr GEPs
on workgroup variables — it falls back to OpBitcast, losing the offset.

This pass replaces such ConstantExpr operands with typed GEP instructions that access the
correct struct member, so the emitter can generate proper OpAccessChain.
"""
function _lift_byte_geps_on_workgroup_globals!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i32 = LLVM.Int32Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                # Handle loads and stores
                if inst isa LLVM.LoadInst
                    ptr = LLVM.operands(inst)[1]
                    ptr isa LLVM.ConstantExpr || continue
                    new_ptr = _lift_wg_byte_gep(ptr, dl, inst, T_i32)
                    new_ptr === nothing && continue
                    LLVM.operands(inst)[1] = new_ptr
                elseif inst isa LLVM.StoreInst
                    ptr = LLVM.operands(inst)[2]
                    ptr isa LLVM.ConstantExpr || continue
                    new_ptr = _lift_wg_byte_gep(ptr, dl, inst, T_i32)
                    new_ptr === nothing && continue
                    LLVM.operands(inst)[2] = new_ptr
                end
            end
        end
    end
end

"""
    _lift_wg_byte_gep(cexpr, dl, insert_before, T_i32) -> LLVM.Value or nothing

If `cexpr` is a byte-offset ConstantExpr GEP (source type i8) on an addrspace(3) global,
create a typed GEP instruction that accesses the correct struct field and return it.
"""
function _lift_wg_byte_gep(cexpr::LLVM.ConstantExpr, dl::LLVM.DataLayout,
                            insert_before::LLVM.Instruction, T_i32::LLVM.IntegerType)
    opcode = LLVM.API.LLVMGetConstOpcode(cexpr)
    opcode == LLVM.API.LLVMGetElementPtr || return nothing

    # Check source element type is i8 (byte GEP)
    source_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(cexpr))
    source_ty isa LLVM.IntegerType || return nothing
    LLVM.width(source_ty) == 8 || return nothing

    # Get base pointer
    n_ops = Int(LLVM.API.LLVMGetNumOperands(cexpr))
    n_ops == 2 || return nothing  # i8 GEP: base_ptr + one index
    base_ptr = LLVM.Value(LLVM.API.LLVMGetOperand(cexpr, 0))

    # Check addrspace(3)
    ptr_ty = LLVM.value_type(base_ptr)
    ptr_ty isa LLVM.PointerType || return nothing
    LLVM.addrspace(ptr_ty) == 3 || return nothing

    # Get constant byte offset
    idx_val = LLVM.Value(LLVM.API.LLVMGetOperand(cexpr, 1))
    idx_val isa LLVM.ConstantInt || return nothing
    byte_offset = convert(Int, idx_val)
    byte_offset > 0 || return nothing  # offset 0 is just the base

    # Resolve the type of the base pointer (global variable)
    base_type = _resolve_wg_ptr_type(base_ptr)
    base_type === nothing && return nothing

    # Walk through array/struct types to find the field at this byte offset
    indices = Int32[]
    current_type = base_type
    remaining_offset = byte_offset

    while remaining_offset > 0 || current_type isa LLVM.ArrayType || current_type isa LLVM.StructType
        if current_type isa LLVM.ArrayType
            elem_ty = LLVM.eltype(current_type)
            elem_size = Int(LLVM.abi_size(dl, elem_ty))
            elem_size > 0 || break
            elem_idx = div(remaining_offset, elem_size)
            push!(indices, Int32(elem_idx))
            remaining_offset -= elem_idx * elem_size
            current_type = elem_ty
        elseif current_type isa LLVM.StructType
            n_fields = Int(LLVM.API.LLVMCountStructElementTypes(current_type))
            found = false
            for fi in 0:(n_fields - 1)
                field_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, current_type, fi))
                field_ty = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(current_type, fi))
                field_size = Int(LLVM.abi_size(dl, field_ty))
                if field_offset <= remaining_offset < field_offset + field_size
                    push!(indices, Int32(fi))
                    remaining_offset -= field_offset
                    current_type = field_ty
                    found = true
                    break
                end
            end
            found || break
        else
            break
        end
        remaining_offset == 0 && break
    end

    remaining_offset == 0 || return nothing
    isempty(indices) && return nothing

    # Build typed GEP instruction
    # LLVM GEP first index is always pointer arithmetic (doesn't drill into the type),
    # so prepend a 0 index before the type-walk indices.
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, insert_before)
        gep_indices = LLVM.Value[LLVM.ConstantInt(T_i32, 0)]  # pointer arithmetic
        append!(gep_indices, [LLVM.ConstantInt(T_i32, idx) for idx in indices])
        return LLVM.gep!(builder, base_type, base_ptr, gep_indices, "wg_field_gep")
    end
end

"""
    _decompose_workgroup_typepun_copies!(mod::LLVM.Module, dl::LLVM.DataLayout)

LLVM may optimize struct copies in shared memory (e.g., `shared[1] = shared[2]`)
into raw integer block copies:
  %val = load i64, ptr addrspace(3) <struct_ptr>
  store i64 %val, ptr addrspace(3) <struct_ptr>

SPIR-V requires typed access — can't load i64 from a struct pointer.
This pass detects such workgroup-to-workgroup typepun copy pairs and replaces
them with per-field typed copies.
"""
function _decompose_workgroup_typepun_copies!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.LoadInst || continue
                load_ty = LLVM.value_type(inst)
                load_ty isa LLVM.IntegerType || continue

                ptr = LLVM.operands(inst)[1]
                ptr_ty = LLVM.value_type(ptr)
                ptr_ty isa LLVM.PointerType || continue
                LLVM.addrspace(ptr_ty) == 3 || continue

                # Resolve what the pointer actually points to
                pointed_ty = _resolve_wg_ptr_type(ptr)
                pointed_ty === nothing && continue

                # Get the struct type (unwrap arrays)
                struct_ty = _unwrap_to_struct(pointed_ty)
                struct_ty === nothing && continue

                # Check it's actually a typepun (loaded type != pointed type)
                load_ty == struct_ty && continue

                load_size = div(LLVM.width(load_ty), 8)
                fields = _fields_in_byte_range(struct_ty, dl, 0, load_size)
                isempty(fields) && continue

                # Find all store uses in addrspace(3)
                wg_stores = LLVM.StoreInst[]
                all_uses_are_wg_stores = true
                for use in LLVM.uses(inst)
                    user = LLVM.user(use)
                    if user isa LLVM.StoreInst && LLVM.operands(user)[1] == inst
                        store_ptr = LLVM.operands(user)[2]
                        store_ptr_ty = LLVM.value_type(store_ptr)
                        if store_ptr_ty isa LLVM.PointerType && LLVM.addrspace(store_ptr_ty) == 3
                            push!(wg_stores, user)
                        else
                            all_uses_are_wg_stores = false
                        end
                    else
                        all_uses_are_wg_stores = false
                    end
                end

                # Only handle when all uses are WG stores
                all_uses_are_wg_stores || continue
                isempty(wg_stores) && continue

                # Replace each load+store pair with per-field copies
                # We need the base pointer that points to the ARRAY (before element indexing).
                # For ConstantExpr GEPs like `gep [64 x {i8,i64}], ptr @shared, 0, <idx>`,
                # `ptr` points to the element (struct). We need to generate typed GEPs
                # using the struct type so the emitter sees proper struct member access.
                T_i32 = LLVM.Int32Type()

                for store in wg_stores
                    dst_ptr = LLVM.operands(store)[2]

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, store)

                        for (field_idx, field_ty, field_offset) in fields
                            # Source: typed struct-member GEP into the struct
                            src_field_ptr = LLVM.gep!(builder, struct_ty, ptr,
                                [LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, field_idx)],
                                "wg_copy_src")
                            src_val = LLVM.load!(builder, field_ty, src_field_ptr, "wg_copy_val")
                            LLVM.alignment!(src_val, max(1, _llvm_type_size(field_ty)))

                            # Destination: typed struct-member GEP into the struct
                            dst_field_ptr = LLVM.gep!(builder, struct_ty, dst_ptr,
                                [LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, field_idx)],
                                "wg_copy_dst")
                            st = LLVM.store!(builder, src_val, dst_field_ptr)
                            LLVM.alignment!(st, max(1, _llvm_type_size(field_ty)))
                        end
                    end
                    push!(to_erase, store)
                end
                push!(to_erase, inst)  # erase the original load
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

# ============================================================================
# Sub-pass: Decompose type-punned alloca loads
# ============================================================================

"""
    _decompose_typepun_alloca_loads!(mod::LLVM.Module, dl::LLVM.DataLayout)

LLVM's memcpy optimization may generate `load i64, ptr %alloca_of_struct` to copy
raw bytes from a padded struct alloca. SPIR-V requires strict type matching.
Replace with: load first field → zero-extend to target integer type.
"""
function _decompose_typepun_alloca_loads!(mod::LLVM.Module, dl::LLVM.DataLayout)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.LoadInst || continue
                load_ty = LLVM.value_type(inst)
                load_ty isa LLVM.IntegerType || continue

                ptr = LLVM.operands(inst)[1]
                # Only handle direct alloca loads (not GEP-derived pointers)
                ptr isa LLVM.AllocaInst || continue
                alloca = ptr

                alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                alloca_ty == load_ty && continue
                !(alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType) && continue

                # Check: load size < alloca size (partial/prefix read)
                alloca_size = Int(LLVM.storage_size(dl, alloca_ty))
                load_size = div(LLVM.width(load_ty), 8)
                load_size >= alloca_size && continue  # handled by _fix_alloca_type_mismatched_loads!

                # Check if this typepun load feeds into a workgroup store.
                # If so, decompose the entire copy pair into per-field struct copies
                # instead of the single-byte decomposition.
                wg_store = _find_wg_store_user(inst)
                if wg_store !== nothing
                    _decompose_memcpy_to_wg_fields!(alloca, alloca_ty, wg_store, to_erase, dl)
                    continue
                end

                # Non-workgroup case: load all scalar fields within load_size
                # and combine with shift+or (handles multi-component types like Complex)
                fields = _flatten_type_to_scalars(alloca_ty)
                isempty(fields) && continue

                # Filter to fields that fit within load_size bytes
                all_ok = true
                for (_, fty) in fields
                    fsz = _llvm_type_size(fty)
                    if !(fsz in (1, 2, 4, 8))
                        all_ok = false
                        break
                    end
                end
                all_ok || continue

                LLVM.@dispose builder=LLVM.IRBuilder() begin
                    LLVM.position!(builder, inst)
                    combined = nothing
                    bit_offset = 0

                    for (gep_indices, field_ty) in fields
                        field_size = _llvm_type_size(field_ty)
                        field_bits = field_size * 8

                        # Stop when we've covered all bits in the load
                        bit_offset + field_bits > load_size * 8 && break

                        # GEP to field
                        idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                        for idx in gep_indices
                            push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                        end
                        field_ptr = LLVM.gep!(builder, alloca_ty, ptr, idx_values, "typepun_gep")
                        field_val = LLVM.load!(builder, field_ty, field_ptr, "typepun_load")

                        # Bitcast to integer if needed
                        int_val = if field_ty isa LLVM.IntegerType
                            field_val
                        else
                            LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                        end

                        # Zero-extend to target width
                        wide_val = if field_bits == LLVM.width(load_ty)
                            int_val
                        else
                            LLVM.zext!(builder, int_val, load_ty, "typepun_zext")
                        end

                        # Shift left by bit_offset
                        if bit_offset > 0
                            shift = LLVM.ConstantInt(load_ty, bit_offset)
                            wide_val = LLVM.shl!(builder, wide_val, shift, "typepun_shl")
                        end

                        # OR into combined value
                        combined = if combined === nothing
                            wide_val
                        else
                            LLVM.or!(builder, combined, wide_val, "typepun_or")
                        end

                        bit_offset += field_bits
                    end

                    if combined !== nothing
                        LLVM.replace_uses!(inst, combined)
                        push!(to_erase, inst)
                    end
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

"""Find a workgroup store that is the sole user of a loaded value."""
function _find_wg_store_user(load_inst::LLVM.LoadInst)
    for use in LLVM.uses(load_inst)
        user = LLVM.user(use)
        if user isa LLVM.StoreInst
            store_ptr = LLVM.operands(user)[2]
            ptr_ty = LLVM.value_type(store_ptr)
            if ptr_ty isa LLVM.PointerType && LLVM.addrspace(ptr_ty) == 3
                return user
            end
        end
    end
    return nothing
end

"""
Decompose a memcpy-style struct copy (load i64 from alloca, store i64 to workgroup)
into per-field typed copies. This handles the LLVM memcpy optimization that converts
struct copies into raw integer block copies.

Given:
  %raw = load i64, ptr %alloca_of_{i8, i64}
  store i64 %raw, ptr addrspace(3) %wg_struct_ptr
  %gep = gep i8, ptr %alloca, i64 8
  %raw2 = load i64, ptr %gep
  %wg_gep = gep i8, ptr addrspace(3) %wg_struct_ptr, i64 8
  store i64 %raw2, ptr addrspace(3) %wg_gep

Produces per-field copies:
  %f0_ptr = gep {i8, i64}, ptr %alloca, 0, 0, 0  (unwrap nested struct)
  %f0 = load i8, ptr %f0_ptr
  store i8 %f0, ptr addrspace(3) %wg_struct_ptr  (emitter resolves to member 0)
  %f1_ptr = gep {i8, i64}, ptr %alloca, 0, 0, 1  (unwrap nested struct)
  %f1 = load i64, ptr %f1_ptr
  ... (composite workgroup decomposition pass handles the i64 store)
"""
function _decompose_memcpy_to_wg_fields!(alloca::LLVM.AllocaInst, alloca_ty::LLVM.LLVMType,
                                          wg_store::LLVM.StoreInst,
                                          to_erase::Vector{LLVM.Instruction},
                                          dl::LLVM.DataLayout)
    load_inst = LLVM.operands(wg_store)[1]
    wg_ptr = LLVM.operands(wg_store)[2]

    # Get the inner struct type (unwrap one level of nesting if needed)
    inner_ty = alloca_ty
    if inner_ty isa LLVM.StructType
        elems = LLVM.elements(inner_ty)
        if length(elems) == 1 && first(elems) isa LLVM.StructType
            inner_ty = first(elems)
        end
    end

    inner_ty isa LLVM.StructType || return
    member_types = LLVM.elements(inner_ty)

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, wg_store)

        # Build a struct value by loading each field from the alloca
        gep_prefix = alloca_ty == inner_ty ?
            LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)] :
            LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)]

        struct_val = LLVM.UndefValue(inner_ty)
        for (i, mt) in enumerate(member_types)
            idx = LLVM.ConstantInt(LLVM.IntType(32), i - 1)
            field_gep = LLVM.gep!(builder, alloca_ty, alloca, vcat(gep_prefix, [idx]), "field_src_$i")
            field_val = LLVM.load!(builder, mt, field_gep, "field_val_$i")
            struct_val = LLVM.insert_value!(builder, struct_val, field_val, i - 1)
        end

        # Store the struct to workgroup memory (let _decompose_composite_workgroup_accesses! handle it)
        LLVM.store!(builder, struct_val, wg_ptr)
    end

    # Also find and decompose the SECOND half of the memcpy (the i64 field copy)
    # Pattern: gep i8, ptr addrspace(3) %wg_ptr, i64 8 → store i64 %field1
    # These are handled by _decompose_composite_workgroup_accesses! after we
    # replace the first store with a struct store above.
    # Find the second store that copies the remaining bytes
    load_bb = LLVM.parent(load_inst)
    for inst in LLVM.instructions(load_bb)
        inst isa LLVM.StoreInst || continue
        inst == wg_store && continue
        store_val = LLVM.operands(inst)[1]
        store_ptr = LLVM.operands(inst)[2]
        ptr_ty = LLVM.value_type(store_ptr)
        ptr_ty isa LLVM.PointerType || continue
        LLVM.addrspace(ptr_ty) == 3 || continue

        # Check if this is a store to the SAME shared memory struct (via byte GEP)
        if store_ptr isa LLVM.GetElementPtrInst
            gep_base = LLVM.operands(store_ptr)[1]
            if gep_base == wg_ptr
                # This is the second-half store — remove it since we already
                # included all fields in the struct store above
                push!(to_erase, inst)
                # Also remove the load feeding this store if it only has this one user
                if store_val isa LLVM.LoadInst && _has_single_use(store_val)
                    push!(to_erase, store_val)
                end
                # Remove the GEP if single-use
                if _has_single_use(store_ptr)
                    push!(to_erase, store_ptr)
                end
            end
        end
    end

    # Remove the original typepun load and store
    push!(to_erase, wg_store)
    if _has_single_use(load_inst) || _count_uses(load_inst) == 0
        push!(to_erase, load_inst)
    end
end

function _has_single_use(val::LLVM.Value)
    count = 0
    for _ in LLVM.uses(val)
        count += 1
        count > 1 && return false
    end
    return count == 1
end

function _count_uses(val::LLVM.Value)
    count = 0
    for _ in LLVM.uses(val)
        count += 1
    end
    return count
end

"""Get the first scalar field type from a struct/array type, recursing into nested types."""
function _get_first_scalar_field(ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        elems = LLVM.elements(ty)
        isempty(elems) && return nothing
        return _get_first_scalar_field(first(elems))
    elseif ty isa LLVM.ArrayType
        LLVM.length(ty) == 0 && return nothing
        return _get_first_scalar_field(LLVM.eltype(ty))
    else
        return ty  # scalar
    end
end

"""Get GEP index path to first scalar field (e.g., { { i8, i64 } } → [0, 0])."""
function _gep_path_to_first_scalar(ty::LLVM.LLVMType)
    path = Int[]
    current = ty
    while current isa LLVM.StructType || current isa LLVM.ArrayType
        push!(path, 0)
        if current isa LLVM.StructType
            current = first(LLVM.elements(current))
        else
            current = LLVM.eltype(current)
        end
    end
    return path
end

# ============================================================================
# Sub-pass: Fix shared memory GEPs
# ============================================================================

"""
    _fix_shared_geps!(mod::LLVM.Module)

Fix GEPs on addrspace(3) array globals for SPIR-V compatibility.

Three categories of fixups:

**Part 0**: Fix negative-index GEPs. Julia's 1-based `unsafe_load(ptr, i)` generates
negative array indices that SPIR-V's OpAccessChain cannot represent.
  - Pattern A: ConstantExpr GEP base with negative array index
  - Pattern B: Flat GEP chains with constant -1 offset

**Part 1**: Flatten nested array GEPs to use flat scalar array type with computed index.
SROA decomposes nested struct accesses into GEPs with nested array source types
(e.g., `[2 x [1 x [3 x float]]]`). The global is a flat `[N x scalar]`, so we
collapse all chains into a single flat GEP.

**Parts 2+3**: Fix constant-index shared memory accesses (bare global loads/stores,
ConstantExpr GEPs). Uses non-constant zero trick (`sub %val, %val`) to prevent
IRBuilder's constant folder from collapsing GEPs back to bare globals.
"""
function _fix_shared_geps!(mod::LLVM.Module)
    # Collect addrspace(3) globals with array type
    shared_globals = Dict{LLVM.Value, LLVM.LLVMType}()
    for gv in LLVM.globals(mod)
        pointee_ty = LLVM.global_value_type(gv)
        ptr_ty = LLVM.value_type(gv)
        if ptr_ty isa LLVM.PointerType && LLVM.addrspace(ptr_ty) == 3 &&
           pointee_ty isa LLVM.ArrayType
            shared_globals[gv] = pointee_ty
        end
    end

    isempty(shared_globals) && return

    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Lazily created non-constant i64 zero (one per function).
        # Uses `sub %val, %val` on an existing i32 instruction to produce 0
        # without being recognized as a constant by the IRBuilder's folder.
        nonconst_zero = Ref{Union{Nothing, LLVM.Value}}(nothing)

        function _get_nonconst_zero!()
            nonconst_zero[] !== nothing && return nonconst_zero[]::LLVM.Value
            entry_bb = first(LLVM.blocks(f))
            # Find an existing i32 instruction in the entry block to derive zero from
            donor = nothing
            for inst in LLVM.instructions(entry_bb)
                if LLVM.value_type(inst) == T_i32 && inst != LLVM.terminator(entry_bb)
                    donor = inst
                    break
                end
            end
            if donor === nothing
                # Fallback: no i32 instruction found, use first i32 param
                for p in LLVM.parameters(f)
                    if LLVM.value_type(p) == T_i32
                        donor = p
                        break
                    end
                end
            end
            @assert donor !== nothing "No i32 value found for non-constant zero"

            # Insert sub+sext right after the donor (or at entry start for params)
            ncz = LLVM.IRBuilder() do b
                insert_pt = if donor isa LLVM.Instruction
                    next = LLVM.API.LLVMGetNextInstruction(donor)
                    next == C_NULL ? LLVM.terminator(entry_bb) : LLVM.Instruction(next)
                else
                    first(LLVM.instructions(entry_bb))
                end
                LLVM.position!(b, insert_pt)
                z32 = LLVM.sub!(b, donor, donor, "shared_zero32")
                return LLVM.sext!(b, z32, T_i64, "shared_idx_zero")
            end
            nonconst_zero[] = ncz
            return ncz
        end

        for bb in LLVM.blocks(f)
            # ----------------------------------------------------------------
            # Part 0: Fix negative-index GEPs on shared memory (addrspace 3).
            # ----------------------------------------------------------------

            # Pattern A: ConstantExpr GEP bases with negative array index
            constexpr_fixes = Tuple{LLVM.GetElementPtrInst, LLVM.Value, LLVM.LLVMType, Int}[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue
                ops = LLVM.operands(inst)
                length(ops) == 2 || continue
                ptr_op = ops[1]
                ptr_op isa LLVM.ConstantExpr || continue
                LLVM.API.LLVMGetConstOpcode(ptr_op) == LLVM.API.LLVMGetElementPtr || continue
                ce_ops = LLVM.operands(ptr_op)
                length(ce_ops) >= 3 || continue
                gv = ce_ops[1]
                haskey(shared_globals, gv) || continue
                arr_idx = ce_ops[2]
                elem_idx = ce_ops[3]
                arr_idx isa LLVM.ConstantInt || continue
                elem_idx isa LLVM.ConstantInt || continue
                a = convert(Int, arr_idx)
                b = convert(Int, elem_idx)
                arr_ty = shared_globals[gv]
                N = length(arr_ty)
                offset = a * N + b
                push!(constexpr_fixes, (inst, gv, arr_ty, offset))
            end

            if !isempty(constexpr_fixes)
                ncz = _get_nonconst_zero!()
                zero_const = LLVM.ConstantInt(T_i64, 0)
                LLVM.IRBuilder() do builder
                    for (gep, gv, arr_ty, offset) in constexpr_fixes
                        LLVM.position!(builder, gep)
                        dyn_idx = LLVM.operands(gep)[2]
                        adj_idx = if offset == 0
                            dyn_idx
                        else
                            LLVM.add!(builder, dyn_idx,
                                     LLVM.ConstantInt(LLVM.value_type(dyn_idx), offset),
                                     "shmem_adj_idx")
                        end
                        if LLVM.value_type(adj_idx) != T_i64
                            adj_idx = LLVM.sext!(builder, adj_idx, T_i64, "shmem_idx64")
                        end
                        new_gep = LLVM.gep!(builder, arr_ty, gv,
                                           LLVM.Value[zero_const, adj_idx],
                                           LLVM.name(gep) == "" ? "shmem_gep.fix" : LLVM.name(gep) * ".fix")
                        LLVM.replace_uses!(gep, new_gep)
                        LLVM.erase!(gep)
                    end
                end
            end

            # Pattern B: Flat GEP chains on addrspace(3) GEP results.
            # Merge flat GEPs into the preceding structured GEP by adding the
            # offset to the last index. Runs iteratively until fixpoint.
            changed = true
            while changed
                changed = false
                chain_fixes = Tuple{LLVM.GetElementPtrInst, LLVM.GetElementPtrInst, LLVM.Value}[]
                # First pass: collect all candidates
                candidates = Set{LLVM.GetElementPtrInst}()
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) == 2 || continue
                    ptr_op = ops[1]
                    ptr_op isa LLVM.GetElementPtrInst || continue
                    ptr_ty = LLVM.value_type(ptr_op)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 3 || continue
                    push!(candidates, inst)
                end
                # Second pass: only fix GEPs whose base is NOT also a candidate
                # (process leaves first to avoid erasing bases)
                for inst in candidates
                    base = LLVM.operands(inst)[1]::LLVM.GetElementPtrInst
                    base in candidates && continue
                    push!(chain_fixes, (inst, base, LLVM.operands(inst)[2]))
                end

                if !isempty(chain_fixes)
                    changed = true
                    LLVM.IRBuilder() do builder
                        for (gep, base_gep, offset_val) in chain_fixes
                            LLVM.position!(builder, gep)
                            base_idx = LLVM.operands(base_gep)[end]
                            idx_ty = LLVM.value_type(base_idx)
                            off_ty = LLVM.value_type(offset_val)
                            if off_ty != idx_ty
                                if idx_ty == T_i64
                                    offset_val = LLVM.sext!(builder, offset_val, T_i64, "shmem_off64")
                                else
                                    base_idx = LLVM.sext!(builder, base_idx, T_i64, "shmem_base64")
                                    idx_ty = T_i64
                                end
                            end
                            adj_idx = LLVM.add!(builder, base_idx, offset_val,
                                               "shmem_chain_adj")
                            src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base_gep))
                            base_ptr = LLVM.operands(base_gep)[1]
                            base_ops = LLVM.operands(base_gep)
                            new_indices = LLVM.Value[base_ops[i] for i in 2:length(base_ops)-1]
                            push!(new_indices, adj_idx)
                            new_gep = LLVM.gep!(builder, src_ty, base_ptr, new_indices,
                                               LLVM.name(gep) == "" ? "shmem_chain.fix" : LLVM.name(gep) * ".fix")
                            LLVM.replace_uses!(gep, new_gep)
                            LLVM.erase!(gep)
                        end
                    end
                end
            end

            # ----------------------------------------------------------------
            # Part 1: Flatten all GEPs on shared memory globals to use the flat
            # scalar array type with a single computed index.
            # ----------------------------------------------------------------

            # Helper: compute total leaf scalar elements in a (nested) array type.
            function _flat_count(ty::LLVM.LLVMType, leaf_ty::LLVM.LLVMType)
                ty == leaf_ty && return 1
                ty isa LLVM.ArrayType || return nothing
                inner = _flat_count(eltype(ty), leaf_ty)
                inner === nothing && return nothing
                return length(ty) * inner
            end

            # Helper: compute flat index from nested indices on a (nested) array type.
            function _compute_flat_index!(builder, src_ty, elem_ty, indices, T_i64)
                flat_idx = nothing
                cur_ty = src_ty
                for (k, idx) in enumerate(indices)
                    stride_count = _flat_count(cur_ty, elem_ty)
                    stride = stride_count === nothing ? 1 : stride_count

                    idx_val = idx
                    if LLVM.value_type(idx_val) != T_i64
                        idx_val = LLVM.sext!(builder, idx_val, T_i64, "shmem_idx64")
                    end

                    term = stride == 1 ? idx_val :
                        LLVM.mul!(builder, idx_val,
                                 LLVM.ConstantInt(T_i64, stride), "shmem_stride")

                    flat_idx = flat_idx === nothing ? term :
                        LLVM.add!(builder, flat_idx, term, "shmem_flat")

                    # Descend into element type for next index
                    cur_ty = cur_ty isa LLVM.ArrayType ? eltype(cur_ty) : elem_ty
                end
                return flat_idx
            end

            # Run iteratively: fix direct-to-global GEPs, then fix GEPs chained
            # on already-fixed GEPs, until no more changes.
            fixed_geps = Dict{LLVM.Value, LLVM.Value}()
            part1_changed = true
            while part1_changed
                part1_changed = false

                to_fix = Tuple{LLVM.GetElementPtrInst, LLVM.Value, LLVM.LLVMType}[]
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) >= 2 || continue
                    ptr_op = ops[1]
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))

                    if haskey(shared_globals, ptr_op)
                        arr_ty = shared_globals[ptr_op]
                        src_ty == arr_ty && continue  # already correct
                        push!(to_fix, (inst, ptr_op, arr_ty))
                    elseif haskey(fixed_geps, ptr_op)
                        base_gep = ptr_op::LLVM.GetElementPtrInst
                        base_ptr = LLVM.operands(base_gep)[1]
                        haskey(shared_globals, base_ptr) || continue
                        arr_ty = shared_globals[base_ptr]
                        push!(to_fix, (inst, base_ptr, arr_ty))
                    end
                end

                isempty(to_fix) && break

                LLVM.IRBuilder() do builder
                    for (gep, global_ptr, arr_ty) in to_fix
                        ops = LLVM.operands(gep)
                        ptr_op = ops[1]
                        LLVM.position!(builder, gep)
                        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
                        elem_ty = eltype(arr_ty)

                        total = _flat_count(src_ty, elem_ty)
                        total === nothing && continue

                        indices = LLVM.Value[ops[i] for i in 2:length(ops)]
                        flat_idx = _compute_flat_index!(builder, src_ty, elem_ty, indices, T_i64)
                        flat_idx === nothing && continue

                        # If chained on a fixed GEP, add the base's flat index
                        if haskey(fixed_geps, ptr_op)
                            base_idx = fixed_geps[ptr_op]
                            flat_idx = LLVM.add!(builder, base_idx, flat_idx, "shmem_chain_flat")
                        end

                        zero = LLVM.ConstantInt(T_i64, 0)
                        new_gep = LLVM.gep!(builder, arr_ty, global_ptr,
                                            LLVM.Value[zero, flat_idx],
                                            LLVM.name(gep) * ".fix")
                        fixed_geps[new_gep] = flat_idx
                        LLVM.replace_uses!(gep, new_gep)
                        LLVM.erase!(gep)
                        part1_changed = true
                    end
                end
            end

            # ----------------------------------------------------------------
            # Parts 2+3: Fix constant-index shared memory accesses.
            # Bare global loads/stores and ConstantExpr GEP accesses.
            # ----------------------------------------------------------------
            const_fixes = Tuple{LLVM.Instruction, Int, LLVM.Value, LLVM.LLVMType, Int}[]
            for inst in LLVM.instructions(bb)
                if inst isa LLVM.LoadInst
                    ptr_op = LLVM.operands(inst)[1]
                    op_idx = 0
                elseif inst isa LLVM.StoreInst
                    ptr_op = LLVM.operands(inst)[2]
                    op_idx = 1
                else
                    continue
                end

                # Case A: Bare global (constant-folded GEP @arr, 0, 0 -> @arr)
                if haskey(shared_globals, ptr_op)
                    push!(const_fixes, (inst, op_idx, ptr_op, shared_globals[ptr_op], 0))
                    continue
                end

                # Case B: ConstantExpr GEP (e.g. GEP @arr, 0, 1)
                if ptr_op isa LLVM.ConstantExpr
                    LLVM.API.LLVMGetConstOpcode(ptr_op) == LLVM.API.LLVMGetElementPtr || continue
                    ce_ops = LLVM.operands(ptr_op)
                    gv = ce_ops[1]
                    haskey(shared_globals, gv) || continue
                    length(ce_ops) >= 3 || continue
                    idx_val = ce_ops[3]
                    idx_val isa LLVM.ConstantInt || continue
                    elem_idx = convert(Int, idx_val)
                    push!(const_fixes, (inst, op_idx, gv, shared_globals[gv], elem_idx))
                end
            end

            isempty(const_fixes) && continue

            ncz = _get_nonconst_zero!()
            zero_const = LLVM.ConstantInt(T_i64, 0)

            LLVM.IRBuilder() do builder
                for (inst, op_idx, gv, arr_ty, elem_idx) in const_fixes
                    LLVM.position!(builder, inst)
                    idx = if elem_idx == 0
                        ncz
                    else
                        LLVM.add!(builder, ncz, LLVM.ConstantInt(T_i64, elem_idx),
                                  "shared_idx.fix")
                    end
                    # First index MUST be constant 0 (array-of-arrays offset).
                    new_gep = LLVM.gep!(builder, arr_ty, gv,
                                       LLVM.Value[zero_const, idx], "shared_gep.fix")
                    LLVM.API.LLVMSetOperand(inst, op_idx, new_gep)
                end
            end
        end
    end
end

# ============================================================================
# Sub-pass: Flatten BDA array GEPs
# ============================================================================

"""
    _gep_byte_offset(dl, src_ty, indices) -> Int or nothing

Compute the total byte offset for a sequence of constant GEP indices starting
from `src_ty`. Returns `nothing` if any index is non-constant.
"""
function _gep_byte_offset(dl::LLVM.DataLayout, src_ty::LLVM.LLVMType,
                          indices::AbstractVector)
    offset = 0
    cur_ty = src_ty
    for idx_val in indices
        idx_val isa LLVM.ConstantInt || return nothing
        idx = convert(Int, idx_val)
        if cur_ty isa LLVM.ArrayType
            elem_ty = LLVM.eltype(cur_ty)
            elem_size = Int(LLVM.storage_size(dl, elem_ty))
            offset += idx * elem_size
            cur_ty = elem_ty
        elseif cur_ty isa LLVM.StructType
            offset += Int(LLVM.API.LLVMOffsetOfElement(dl, cur_ty, idx))
            cur_ty = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(cur_ty, idx))
        else
            return nothing
        end
    end
    return offset
end

"""
    _flatten_bda_array_geps!(mod::LLVM.Module)

Replace composite-typed GEPs in addrspace(1) with explicit pointer arithmetic
(ptrtoint -> add -> inttoptr).

Julia represents tuples as LLVM array types (e.g. `[3 x float]` for NTuple{3,Float32}).
The SPIR-V emitter would need ArrayStride decorations for PhysicalStorageBuffer arrays,
and multi-level GEPs crash the bitcast legalization. By replacing composite-typed GEPs
with explicit byte arithmetic, we avoid these composite types in SPIR-V entirely.

Only handles GEPs with all-constant indices starting with 0.
"""
function _flatten_bda_array_geps!(mod::LLVM.Module)
    i64 = LLVM.Int64Type()
    dl = LLVM.datalayout(mod)
    for f in LLVM.functions(mod)
        for bb in LLVM.blocks(f)
            to_fix = LLVM.GetElementPtrInst[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue

                # Check pointer operand is addrspace(1) (BDA / PhysicalStorageBuffer)
                ptr_op = LLVM.operands(inst)[1]
                ptr_ty = LLVM.value_type(ptr_op)
                ptr_ty isa LLVM.PointerType || continue
                LLVM.addrspace(ptr_ty) == 1 || continue

                # Source element type must be composite (array or struct)
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                (src_ty isa LLVM.ArrayType || src_ty isa LLVM.StructType) || continue

                ops = LLVM.operands(inst)
                length(ops) >= 3 || continue  # ptr, idx0, idx1, ...

                # First index must be constant 0 (no base pointer scaling)
                idx0 = ops[2]
                idx0 isa LLVM.ConstantInt || continue
                convert(Int, idx0) == 0 || continue

                # All remaining indices must be constant for compile-time offset
                remaining = @view ops[3:end]
                byte_off = _gep_byte_offset(dl, src_ty, remaining)
                byte_off === nothing && continue

                push!(to_fix, inst)
            end

            isempty(to_fix) && continue

            LLVM.IRBuilder() do builder
                for gep in to_fix
                    ops = LLVM.operands(gep)
                    ptr_op = ops[1]
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
                    remaining = @view ops[3:end]
                    byte_off = _gep_byte_offset(dl, src_ty, remaining)

                    LLVM.position!(builder, gep)
                    # ptrtoint -> add byte offset -> inttoptr
                    base_int = LLVM.ptrtoint!(builder, ptr_op, i64, "bda.base")
                    if byte_off == 0
                        new_addr = base_int
                    else
                        off_const = LLVM.ConstantInt(i64, byte_off)
                        new_addr = LLVM.add!(builder, base_int, off_const, "bda.addr")
                    end
                    new_ptr = LLVM.inttoptr!(builder, new_addr, LLVM.value_type(ptr_op), LLVM.name(gep))

                    LLVM.replace_uses!(gep, new_ptr)
                    LLVM.erase!(gep)
                end
            end
        end
    end
end

# ============================================================================
# Sub-pass: Warn about constant globals
# ============================================================================

"""
    _warn_constant_globals!(mod::LLVM.Module)

Check for constant global variables in addrspace(1) that would be invalid in
Vulkan SPIR-V. These indicate a missing device function override -- the Julia
stdlib function is using lookup tables that produce global constant arrays.

PhysicalStorageBuffer storage class cannot have OpVariable declarations.
The fix is to override the functions that create these globals (e.g. ^(Float64,Float64))
with intrinsic-based or polynomial implementations.
"""
function _warn_constant_globals!(mod::LLVM.Module)
    for gv in LLVM.globals(mod)
        as = LLVM.API.LLVMGetPointerAddressSpace(LLVM.API.LLVMTypeOf(gv))
        as == 1 || continue
        LLVM.API.LLVMGetInitializer(gv) == C_NULL && continue
        name = LLVM.name(gv)
        @warn "Lava: constant global '$name' in addrspace(1) -- needs device function override" maxlog=1
    end
end

# ============================================================================
# Main pass: prepare_module_for_vulkan!
# ============================================================================

"""
    _prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String)

Final LLVM-level preparation pass before the custom SPIR-V emitter runs.

Applies 6 targeted sub-passes:
1. Strip lifetime intrinsics + set internal linkage + fix PSB alignment
2. Fix shared memory GEPs (addrspace 3)
3. Flatten BDA array GEPs (addrspace 1) to pointer arithmetic
4. Lower memset/memcpy on PSB pointers to explicit stores/loads
5. Decompose composite loads/stores on PSB to scalar operations
6. Warn about invalid constant globals in addrspace(1)

NOTE: Unlike Abacus, we do NOT:
- Call _hoist_push_constant_loads! (that was an llc workaround)
- Strip debug info (we KEEP it for source mapping in the custom emitter)
"""
function _prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String;
                                     dl::LLVM.DataLayout=LLVM.datalayout(mod))
    # 1. Strip llvm.lifetime.start/end intrinsics, set internal linkage,
    #    and fix alignment for PhysicalStorageBuffer accesses.
    for f in LLVM.functions(mod)
        fname = LLVM.name(f)

        # Set non-entry function definitions to internal linkage.
        # This prevents the SPIR-V emitter from needing Linkage capability (illegal in Vulkan).
        # Skip declarations (no basic blocks) -- they must keep external linkage.
        is_declaration = LLVM.API.LLVMIsDeclaration(f) != 0
        if fname != entry_name && !startswith(fname, "llvm.") && !is_declaration
            LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
        end

        # Walk instructions: erase lifetime intrinsics + fix alignment
        for bb in LLVM.blocks(f)
            to_erase = LLVM.Instruction[]
            for inst in LLVM.instructions(bb)
                if inst isa LLVM.CallInst
                    callee = LLVM.called_operand(inst)
                    if callee isa LLVM.Function
                        cname = LLVM.name(callee)
                        if startswith(cname, "llvm.lifetime.")
                            push!(to_erase, inst)
                        end
                    end
                elseif inst isa LLVM.LoadInst || inst isa LLVM.StoreInst
                    # Fix alignment: Vulkan PhysicalStorageBuffer requires
                    # alignment >= natural size of the largest scalar in the type
                    # (VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314).
                    align = LLVM.alignment(inst)
                    if align > 0
                        ty = if inst isa LLVM.LoadInst
                            LLVM.value_type(inst)
                        else
                            LLVM.value_type(LLVM.operands(inst)[1])
                        end
                        min_align = max(4, _scalar_size(ty))
                        if align < min_align
                            LLVM.alignment!(inst, min_align)
                        end
                    end
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end

    # 2. Remove declarations of lifetime intrinsics (now unused)
    for f in collect(LLVM.functions(mod))
        fname = LLVM.name(f)
        if startswith(fname, "llvm.lifetime.") && isempty(LLVM.blocks(f))
            LLVM.erase!(f)
        end
    end

    # 3. Fix flat GEPs on addrspace(3) array globals
    _fix_shared_geps!(mod)

    # 4. Flatten array-typed GEPs in addrspace(1) (BDA / PhysicalStorageBuffer)
    _flatten_bda_array_geps!(mod)

    # 5. Lower llvm.memset/memcpy on addrspace(1) to explicit stores/loads
    _lower_psb_memops!(mod)

    # 6. Decompose composite loads/stores on addrspace(1)
    _decompose_composite_psb_accesses!(mod, dl)

    # 6b. Decompose composite loads/stores on addrspace(3) (shared memory)
    _decompose_composite_workgroup_accesses!(mod, dl)

    # 6c. Decompose type-punned alloca loads (partial reads from struct allocas)
    _decompose_typepun_alloca_loads!(mod, dl)

    # 7. Warn about invalid constant globals in addrspace(1)
    _warn_constant_globals!(mod)

    # 8. Fix type-mismatched loads from allocas
    # LLVM may optimize loads where the loaded scalar type differs from the alloca's
    # type (e.g., `load i16, ptr %alloca_of_{[2 x i8]}`). SPIR-V requires strict
    # type matching for loads, so we rewrite these to load the alloca type instead.
    _fix_alloca_type_mismatched_loads!(mod)
end

"""
Fix loads where the loaded type differs from the alloca's element type.

LLVM's SROA packs small structs into integers:
    %alloca = alloca { [2 x float] }     ; 8-byte struct (ComplexF32)
    %val = load i64, ptr %alloca          ; load as i64 (type-pun)

SPIR-V requires strict type matching — can't load i64 from a struct pointer.
This pass decomposes such loads into properly-typed field loads + bitcast + combine.

The decomposition for `load iN, ptr %alloca_of_struct`:
  1. Flatten the struct to its scalar fields
  2. Load each scalar with a typed GEP
  3. Bitcast each scalar to iM (M = scalar bit width)
  4. Zero-extend + shift + OR to build the combined iN value
"""
function _fix_alloca_type_mismatched_loads!(mod::LLVM.Module)
    dl = LLVM.datalayout(mod)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.LoadInst || continue
                    load_ty = LLVM.value_type(inst)
                    load_ty isa LLVM.IntegerType || continue

                    ptr = LLVM.operands(inst)[1]
                    alloca = _trace_to_alloca(ptr)
                    alloca === nothing && continue

                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                    # Only fix when types differ and sizes match
                    alloca_ty == load_ty && continue
                    # Use DataLayout for proper struct size (includes padding)
                    alloca_size = Int(LLVM.storage_size(dl, alloca_ty))
                    load_size = div(LLVM.width(load_ty), 8)
                    alloca_size == load_size || continue

                    # Skip loads we already created (prevents infinite loop when decomposition
                    # produces a load of the same size, e.g. alloca { [1 x i64] } → load i64)
                    nm = LLVM.name(inst)
                    if startswith(nm, "typepun_")
                        continue
                    end

                    # Flatten the alloca type into scalar fields with GEP index paths
                    fields = _flatten_type_to_scalars(alloca_ty)
                    isempty(fields) && continue

                    # Verify all fields are scalar types we can bitcast to integer
                    all_ok = true
                    for (_, fty) in fields
                        fsz = _llvm_type_size(fty)
                        if !(fsz in (1, 2, 4, 8))
                            all_ok = false
                            break
                        end
                    end
                    all_ok || continue

                    # Build the replacement
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        combined = nothing
                        bit_offset = 0

                        for (gep_indices, field_ty) in fields
                            field_bits = _llvm_type_size(field_ty) * 8

                            # GEP to field
                            idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                            for idx in gep_indices
                                push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                            end
                            field_ptr = LLVM.gep!(builder, alloca_ty, ptr, idx_values, "typepun_gep")

                            # Load field
                            field_val = LLVM.load!(builder, field_ty, field_ptr, "typepun_load")

                            # Bitcast to integer if needed (float → iN)
                            field_int_ty = LLVM.IntType(field_bits)
                            int_val = if field_ty isa LLVM.IntegerType
                                field_val
                            else
                                LLVM.bitcast!(builder, field_val, field_int_ty, "typepun_cast")
                            end

                            # Zero-extend to target width
                            wide_val = if field_bits == LLVM.width(load_ty)
                                int_val
                            else
                                LLVM.zext!(builder, int_val, load_ty, "typepun_zext")
                            end

                            # Shift left by bit_offset
                            if bit_offset > 0
                                shift = LLVM.ConstantInt(load_ty, bit_offset)
                                wide_val = LLVM.shl!(builder, wide_val, shift, "typepun_shl")
                            end

                            # OR into combined value
                            combined = if combined === nothing
                                wide_val
                            else
                                LLVM.or!(builder, combined, wide_val, "typepun_or")
                            end

                            bit_offset += field_bits
                        end

                        if combined !== nothing
                            LLVM.replace_uses!(inst, combined)
                            LLVM.erase!(inst)
                            changed = true
                            break
                        end
                    end
                    changed && break
                end
                changed && break
            end
        end
    end
end

"""Trace a pointer through GEPs back to its alloca, if any."""
function _trace_to_alloca(ptr::LLVM.Value)
    ptr isa LLVM.AllocaInst && return ptr
    if ptr isa LLVM.GetElementPtrInst
        return _trace_to_alloca(LLVM.operands(ptr)[1])
    end
    return nothing
end

"""
Flatten a composite LLVM type into a list of (index_path, scalar_type) pairs.
Each index_path is an array of integer indices for GEP.
"""
function _flatten_type_to_scalars(ty::LLVM.LLVMType, prefix::Vector{Int}=Int[])
    result = Tuple{Vector{Int}, LLVM.LLVMType}[]

    if ty isa LLVM.StructType
        for (i, field_ty) in enumerate(LLVM.elements(ty))
            new_prefix = copy(prefix)
            push!(new_prefix, i - 1)
            append!(result, _flatten_type_to_scalars(field_ty, new_prefix))
        end
    elseif ty isa LLVM.ArrayType
        elem_ty = LLVM.eltype(ty)
        for i in 0:(LLVM.length(ty) - 1)
            new_prefix = copy(prefix)
            push!(new_prefix, i)
            append!(result, _flatten_type_to_scalars(elem_ty, new_prefix))
        end
    else
        # Scalar type (int, float, half, double)
        push!(result, (copy(prefix), ty))
    end

    return result
end
