# LLVM pass: Prepare module for Vulkan SPIR-V emission.
#
# Ported from Abacus compilation.jl (prepare_module_for_vulkan! and sub-passes).
# This is the final LLVM-level pass before the custom SPIR-V emitter runs.
#
# Sub-passes:
# 1. Strip lifetime intrinsics + set internal linkage + fix PSB alignment
# 2. fix_shared_geps! -- fix GEPs on addrspace(3) shared memory globals
# 3. flatten_bda_array_geps! -- replace composite-typed GEPs in addrspace(1) with ptr arithmetic
# 4. lower_psb_memops! -- lower memset/memcpy on PSB pointers to explicit stores/loads
# 5. decompose_composite_psb_accesses! -- decompose struct load/store on PSB to scalar ops
# 6. warn_constant_globals! -- warn about invalid constant globals in addrspace(1)
#
# NOTE: We do NOT include _hoist_push_constant_loads! from Abacus -- that was
# an llc workaround for ConstantExpr GEP mishandling. The custom emitter handles
# push constant access directly.

# ============================================================================
# Utility functions
# ============================================================================

"""Get the size in bytes of an LLVM type (packed, no padding between fields)."""
function llvm_type_size(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return div(LLVM.width(t) + 7, 8)
    elseif t == LLVM.FloatType()
        return 4
    elseif t == LLVM.DoubleType()
        return 8
    elseif t == LLVM.HalfType()
        return 2
    elseif t isa LLVM.ArrayType
        return length(t) * llvm_type_size(LLVM.eltype(t))
    elseif t isa LLVM.StructType
        total = 0
        for m in LLVM.elements(t)
            total += llvm_type_size(m)
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
function scalar_size(t::LLVM.LLVMType)
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
            sz = max(sz, scalar_size(m))
        end
        return sz
    elseif t isa LLVM.ArrayType || t isa LLVM.VectorType
        return scalar_size(LLVM.eltype(t))
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
    load_composite_from_psb(builder, base_int, type, byte_offset, dl)

Recursively load a composite type from PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: inttoptr(base + offset) -> load.
For struct/array types: recursively load members and reconstruct via insertvalue.
"""
function load_composite_from_psb(builder, base_int, type::LLVM.LLVMType,
                                   byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        result = LLVM.UndefValue(type)
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = load_composite_from_psb(builder, base_int, member_type,
                                                   byte_offset + member_offset, dl)
            result = LLVM.insert_value!(builder, result, member_val, i)
        end
        return result
    elseif type isa LLVM.ArrayType
        result = LLVM.UndefValue(type)
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = load_composite_from_psb(builder, base_int, elem_type,
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
        type_align = llvm_type_size(type)
        offset_align = byte_offset == 0 ? type_align : (1 << trailing_zeros(byte_offset))
        LLVM.alignment!(val, min(type_align, offset_align))
        return val
    end
end

"""
    store_composite_to_psb(builder, val, base_int, type, byte_offset, dl)

Recursively store a composite type to PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: extractvalue -> inttoptr(base + offset) -> store.
For struct/array types: recursively extract and store members.
"""
function store_composite_to_psb(builder, val, base_int, type::LLVM.LLVMType,
                                  byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = LLVM.extract_value!(builder, val, i)
            store_composite_to_psb(builder, member_val, base_int, member_type,
                                     byte_offset + member_offset, dl)
        end
    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = LLVM.extract_value!(builder, val, i)
            store_composite_to_psb(builder, elem_val, base_int, elem_type,
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
        # Alignment = minimum of type's natural alignment and what the byte offset guarantees.
        # For a Bool (i8) at offset 153, natural align is 1, offset guarantees 1.
        # For a Float32 at offset 8, natural align is 4, offset guarantees 8 -> use 4.
        type_align = llvm_type_size(type)  # natural alignment = size for scalar types
        offset_align = byte_offset == 0 ? type_align : (1 << trailing_zeros(byte_offset))
        LLVM.alignment!(st, min(type_align, offset_align))
    end
end

# ============================================================================
# Sub-pass: Lower PSB memops
# ============================================================================

"""
    lower_psb_memops!(mod::LLVM.Module)

Lower `llvm.memset` and `llvm.memcpy` intrinsics on `ptr addrspace(1)` to
explicit store/load loops. SPIR-V doesn't support these intrinsics on
PhysicalStorageBuffer pointers.

Memset is lowered to i32 stores (4-byte aligned) + i8 tail stores.
Memcpy is lowered to i32 load/store pairs + i8 tail.
Only handles constant-length operations (common for struct initialization).
"""
function lower_psb_memops!(mod::LLVM.Module)
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

                if startswith(cname, "llvm.memset.p0")
                    # llvm.memset.p0.iN(ptr dst, i8 val, iN len, i1 volatile)
                    # Lower to explicit GEP + store (no ptrtoint on addrspace 0)
                    ops = LLVM.operands(inst)
                    dst_ptr = ops[1]
                    fill_val = ops[2]  # i8
                    len_val = ops[3]

                    len_val isa LLVM.ConstantInt || continue
                    nbytes = convert(Int, len_val)

                    LLVM.IRBuilder() do builder
                        LLVM.position!(builder, inst)

                        # Build fill word: replicate i8 val to i32
                        val8 = fill_val
                        val32 = LLVM.zext!(builder, val8, T_i32)
                        v1 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 8))
                        val32 = LLVM.or!(builder, val32, v1)
                        v2 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 16))
                        val32 = LLVM.or!(builder, val32, v2)

                        # Use GEP i8 + bitcast approach for addrspace 0
                        n_words = nbytes ÷ 4
                        for i in 0:(n_words-1)
                            off = i * 4
                            ptr = if off == 0
                                dst_ptr
                            else
                                LLVM.gep!(builder, T_i8, dst_ptr, [LLVM.ConstantInt(T_i64, off)])
                            end
                            st = LLVM.store!(builder, val32, ptr)
                            LLVM.alignment!(st, 4)
                        end

                        # Handle tail bytes
                        for i in (n_words*4):(nbytes-1)
                            ptr = LLVM.gep!(builder, T_i8, dst_ptr, [LLVM.ConstantInt(T_i64, i)])
                            st = LLVM.store!(builder, val8, ptr)
                            LLVM.alignment!(st, 1)
                        end
                    end
                    push!(to_erase, inst)

                elseif startswith(cname, "llvm.memset.p1")
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
        if (startswith(fname, "llvm.memset.p0") || startswith(fname, "llvm.memset.p1") ||
            startswith(fname, "llvm.memcpy.p1") ||
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
    decompose_composite_psb_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)

Decompose composite (struct/array) loads/stores on `ptr addrspace(1)` into
individual scalar loads/stores with `inttoptr(base + offset)`.

The SPIR-V emitter needs scalar-level access for PhysicalStorageBuffer pointers.
Composite loads like `load %large_struct, ptr addrspace(1)` must be decomposed
into individual scalar loads, reconstructed via insertvalue. Similarly for stores.
"""
function decompose_composite_psb_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)
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
                        result = load_composite_from_psb(builder, base_int,
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
                        store_composite_to_psb(builder, val, base_int,
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
    decompose_composite_workgroup_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)

Decompose composite (struct/array) loads and stores on addrspace(3) (Workgroup/shared
memory) into per-field scalar operations. SPIR-V shared memory is flattened to scalar
arrays, so composite accesses must be broken down.

Also handles type-punned loads from allocas: `load i64, ptr %alloca_of_struct`
where LLVM's memcpy optimization reads raw bytes from a struct alloca.
These are replaced by loading the first scalar field and zero-extending.
"""
function decompose_composite_workgroup_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)
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
                        result = load_composite_from_wg(builder, ptr, loaded_type, 0, dl)
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
                        store_composite_to_wg(builder, val, ptr, stored_type, 0, dl)
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
function load_composite_from_wg(builder, base_ptr, type::LLVM.LLVMType,
                                  byte_offset::Int, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    if type isa LLVM.StructType
        result = LLVM.UndefValue(type)
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = load_composite_from_wg(builder, base_ptr, member_type,
                                                  byte_offset + member_offset, dl)
            result = LLVM.insert_value!(builder, result, member_val, i)
        end
        return result
    elseif type isa LLVM.ArrayType
        result = LLVM.UndefValue(type)
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = load_composite_from_wg(builder, base_ptr, elem_type,
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
        LLVM.alignment!(val, max(1, llvm_type_size(type)))
        return val
    end
end

function store_composite_to_wg(builder, val, base_ptr, type::LLVM.LLVMType,
                                 byte_offset::Int, dl::LLVM.DataLayout)
    T_i8 = LLVM.Int8Type()
    T_i64 = LLVM.Int64Type()

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = LLVM.extract_value!(builder, val, i)
            store_composite_to_wg(builder, member_val, base_ptr, member_type,
                                    byte_offset + member_offset, dl)
        end
    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = LLVM.extract_value!(builder, val, i)
            store_composite_to_wg(builder, elem_val, base_ptr, elem_type,
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
        LLVM.alignment!(st, max(1, llvm_type_size(type)))
    end
end

# ============================================================================
# Sub-pass: Decompose workgroup typepun copies
# ============================================================================

# Resolve the type that a pointer in addrspace(3) actually points to,
# walking through ConstantExpr GEPs, GEP instructions, and global variables.
# In LLVM GEP, the first index is pointer arithmetic (doesn't change type),
# and only subsequent indices drill into the type hierarchy.
function resolve_wg_ptr_type(ptr::LLVM.Value)
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
function unwrap_to_struct(ty::LLVM.LLVMType)
    current = ty
    while current isa LLVM.ArrayType
        current = LLVM.eltype(current)
    end
    return current isa LLVM.StructType ? current : nothing
end

# Find struct fields fully contained in a byte range [start_byte, end_byte).
function fields_in_byte_range(struct_ty::LLVM.StructType, dl::LLVM.DataLayout,
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
    lift_byte_geps_on_workgroup_globals!(mod::LLVM.Module, dl::LLVM.DataLayout)

Convert byte-offset ConstantExpr GEPs on workgroup globals to typed struct-member GEPs.

LLVM may generate `getelementptr i8, ptr addrspace(3) @shared, i64 <offset>` to access
struct fields in shared memory. The SPIR-V emitter doesn't handle byte-offset ConstantExpr GEPs
on workgroup variables — it falls back to OpBitcast, losing the offset.

This pass replaces such ConstantExpr operands with typed GEP instructions that access the
correct struct member, so the emitter can generate proper OpAccessChain.
"""
function lift_byte_geps_on_workgroup_globals!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i32 = LLVM.Int32Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                # Handle loads and stores
                if inst isa LLVM.LoadInst
                    ptr = LLVM.operands(inst)[1]
                    ptr isa LLVM.ConstantExpr || continue
                    new_ptr = lift_wg_byte_gep(ptr, dl, inst, T_i32)
                    new_ptr === nothing && continue
                    LLVM.operands(inst)[1] = new_ptr
                elseif inst isa LLVM.StoreInst
                    ptr = LLVM.operands(inst)[2]
                    ptr isa LLVM.ConstantExpr || continue
                    new_ptr = lift_wg_byte_gep(ptr, dl, inst, T_i32)
                    new_ptr === nothing && continue
                    LLVM.operands(inst)[2] = new_ptr
                end
            end
        end
    end
end

"""
    lift_wg_byte_gep(cexpr, dl, insert_before, T_i32) -> LLVM.Value or nothing

If `cexpr` is a byte-offset ConstantExpr GEP (source type i8) on an addrspace(3) global,
create a typed GEP instruction that accesses the correct struct field and return it.
"""
function lift_wg_byte_gep(cexpr::LLVM.ConstantExpr, dl::LLVM.DataLayout,
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
    base_type = resolve_wg_ptr_type(base_ptr)
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
    decompose_workgroup_typepun_copies!(mod::LLVM.Module, dl::LLVM.DataLayout)

LLVM may optimize struct copies in shared memory (e.g., `shared[1] = shared[2]`)
into raw integer block copies:
  %val = load i64, ptr addrspace(3) <struct_ptr>
  store i64 %val, ptr addrspace(3) <struct_ptr>

SPIR-V requires typed access — can't load i64 from a struct pointer.
This pass detects such workgroup-to-workgroup typepun copy pairs and replaces
them with per-field typed copies.
"""
function decompose_workgroup_typepun_copies!(mod::LLVM.Module, dl::LLVM.DataLayout)
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
                pointed_ty = resolve_wg_ptr_type(ptr)
                pointed_ty === nothing && continue

                # Get the struct type (unwrap arrays)
                struct_ty = unwrap_to_struct(pointed_ty)
                struct_ty === nothing && continue

                # Check it's actually a typepun (loaded type != pointed type)
                load_ty == struct_ty && continue

                load_size = div(LLVM.width(load_ty), 8)
                fields = fields_in_byte_range(struct_ty, dl, 0, load_size)
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
                            LLVM.alignment!(src_val, max(1, llvm_type_size(field_ty)))

                            # Destination: typed struct-member GEP into the struct
                            dst_field_ptr = LLVM.gep!(builder, struct_ty, dst_ptr,
                                [LLVM.ConstantInt(T_i32, 0), LLVM.ConstantInt(T_i32, field_idx)],
                                "wg_copy_dst")
                            st = LLVM.store!(builder, src_val, dst_field_ptr)
                            LLVM.alignment!(st, max(1, llvm_type_size(field_ty)))
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
    decompose_typepun_alloca_loads!(mod::LLVM.Module, dl::LLVM.DataLayout)

LLVM's memcpy optimization may generate `load i64, ptr %alloca_of_struct` to copy
raw bytes from a padded struct alloca. SPIR-V requires strict type matching.
Replace with: load first field → zero-extend to target integer type.
"""
function decompose_typepun_alloca_loads!(mod::LLVM.Module, dl::LLVM.DataLayout)
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
                load_size >= alloca_size && continue  # handled by fix_alloca_type_mismatched_loads!

                # Check if this typepun load feeds into a workgroup store.
                # If so, decompose the entire copy pair into per-field struct copies
                # instead of the single-byte decomposition.
                wg_store = find_wg_store_user(inst)
                if wg_store !== nothing
                    decompose_memcpy_to_wg_fields!(alloca, alloca_ty, wg_store, to_erase, dl)
                    continue
                end

                # Non-workgroup case: load all scalar fields within load_size
                # and combine with shift+or (handles multi-component types like Complex)
                fields = flatten_type_to_scalars(alloca_ty)
                isempty(fields) && continue

                # Filter to fields that fit within load_size bytes
                all_ok = true
                for (_, fty) in fields
                    fsz = llvm_type_size(fty)
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
                        field_size = llvm_type_size(field_ty)
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
function find_wg_store_user(load_inst::LLVM.LoadInst)
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
function decompose_memcpy_to_wg_fields!(alloca::LLVM.AllocaInst, alloca_ty::LLVM.LLVMType,
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

        # Store the struct to workgroup memory (let decompose_composite_workgroup_accesses! handle it)
        LLVM.store!(builder, struct_val, wg_ptr)
    end

    # Also find and decompose the SECOND half of the memcpy (the i64 field copy)
    # Pattern: gep i8, ptr addrspace(3) %wg_ptr, i64 8 → store i64 %field1
    # These are handled by decompose_composite_workgroup_accesses! after we
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
                if store_val isa LLVM.LoadInst && has_single_use(store_val)
                    push!(to_erase, store_val)
                end
                # Remove the GEP if single-use
                if has_single_use(store_ptr)
                    push!(to_erase, store_ptr)
                end
            end
        end
    end

    # Remove the original typepun load and store
    push!(to_erase, wg_store)
    if has_single_use(load_inst) || count_uses(load_inst) == 0
        push!(to_erase, load_inst)
    end
end

function has_single_use(val::LLVM.Value)
    count = 0
    for _ in LLVM.uses(val)
        count += 1
        count > 1 && return false
    end
    return count == 1
end

function count_uses(val::LLVM.Value)
    count = 0
    for _ in LLVM.uses(val)
        count += 1
    end
    return count
end

"""Get the first scalar field type from a struct/array type, recursing into nested types."""
function get_first_scalar_field(ty::LLVM.LLVMType)
    if ty isa LLVM.StructType
        elems = LLVM.elements(ty)
        isempty(elems) && return nothing
        return get_first_scalar_field(first(elems))
    elseif ty isa LLVM.ArrayType
        LLVM.length(ty) == 0 && return nothing
        return get_first_scalar_field(LLVM.eltype(ty))
    else
        return ty  # scalar
    end
end

"""Get GEP index path to first scalar field (e.g., { { i8, i64 } } → [0, 0])."""
function gep_path_to_first_scalar(ty::LLVM.LLVMType)
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
# Sub-pass: Flatten nested workgroup array globals
# ============================================================================

"""
    flatten_nested_workgroup_arrays!(mod::LLVM.Module)

Replace addrspace(3) globals with nested array types (e.g. `[32 x [2 x float]]`)
by flat scalar arrays (`[64 x float]`). SPIR-V Workgroup variables without explicit
layout (VK_KHR_workgroup_memory_explicit_layout) cannot have nested array types —
NVIDIA drivers miscompute the stride, causing only part of the array to be accessible.

Creates a new flat global and rewrites GEP instructions to use flat indices.
ConstantExpr GEPs (from reduce_group!-style constant-index accesses) are handled
by replacing the global pointer and letting fix_shared_geps! clean them up.
"""
# Fix ConstantExpr GEPs on addrspace(3) globals with negative inner indices.
# Julia's 1-based indexing creates `gep [N x T], ptr @shared, 0, -1` as a base pointer
# adjustment. After flattening nested arrays, this becomes `gep [M x scalar], ptr @flat, 0, -K`.
# SPIR-V OpAccessChain treats indices as unsigned, so -K wraps to ~4 billion -- invalid.
# This pass finds such CEs, computes their byte offset, and replaces each instruction user
# with a flat GEP that folds the negative offset into the dynamic index.
function fixup_negative_wg_constexprs!(mod::LLVM.Module)
    T_i64 = LLVM.Int64Type()

    for gv in collect(LLVM.globals(mod))
        ptr_ty = LLVM.value_type(gv)
        ptr_ty isa LLVM.PointerType || continue
        LLVM.addrspace(ptr_ty) == 3 || continue
        pointee_ty = LLVM.global_value_type(gv)
        pointee_ty isa LLVM.ArrayType || continue

        # Only fix flat scalar arrays (from flatten_nested_workgroup_arrays!).
        # Struct element arrays handle negative CEs correctly via PtrAccessChain
        # with proper ArrayStride on the struct pointer type.
        elem_ty = LLVM.eltype(pointee_ty)
        (elem_ty isa LLVM.StructType || elem_ty isa LLVM.ArrayType) && continue

        # Collect ConstantExpr users with negative indices
        for use in collect(LLVM.uses(gv))
            user = LLVM.user(use)
            user isa LLVM.ConstantExpr || continue
            LLVM.API.LLVMGetConstOpcode(user) == LLVM.API.LLVMGetElementPtr || continue

            # Check if any index is negative
            ce_ops = LLVM.operands(user)
            has_negative = false
            for i in 2:length(ce_ops)
                if ce_ops[i] isa LLVM.ConstantInt && convert(Int64, ce_ops[i]) < 0
                    has_negative = true
                    break
                end
            end
            has_negative || continue

            # Compute the byte offset of this CE
            ce_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
            dl = LLVM.datalayout(mod)
            byte_offset = 0
            cur_ty = ce_src_ty
            for i in 2:length(ce_ops)
                idx = ce_ops[i]
                idx isa LLVM.ConstantInt || continue
                val = convert(Int64, idx)
                elem_size = Int64(LLVM.storage_size(dl, cur_ty))
                byte_offset += val * elem_size
                if cur_ty isa LLVM.ArrayType
                    cur_ty = LLVM.eltype(cur_ty)
                end
            end

            # Replace each instruction user
            for inst_use in collect(LLVM.uses(user))
                inst = LLVM.user(inst_use)
                inst isa LLVM.Instruction || continue

                if inst isa LLVM.GetElementPtrInst
                    # GEP on CE: fold CE byte offset into the GEP's indexing
                    ops = LLVM.operands(inst)
                    gep_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    gep_elem_size = Int64(LLVM.storage_size(dl, gep_src_ty))

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        zero = LLVM.ConstantInt(T_i64, 0)
                        scalar_ty = LLVM.eltype(pointee_ty)
                        scalar_size = Int64(LLVM.storage_size(dl, scalar_ty))

                        # Compute total byte offset: CE offset + GEP dynamic offset
                        # Then convert to flat scalar index
                        if length(ops) == 2 && gep_elem_size > 0
                            # Simple: gep T, ptr %ce, i64 %idx
                            # Total bytes = byte_offset + idx * gep_elem_size
                            # Flat index = total_bytes / scalar_size
                            idx = ops[2]
                            if LLVM.value_type(idx) != T_i64
                                idx = LLVM.sext!(builder, idx, T_i64, "wg_negce_idx64")
                            end
                            if gep_elem_size != scalar_size
                                scale = LLVM.ConstantInt(T_i64, gep_elem_size ÷ scalar_size)
                                idx = LLVM.mul!(builder, idx, scale, "wg_negce_scale")
                            end
                            ce_flat = LLVM.ConstantInt(T_i64, byte_offset ÷ scalar_size)
                            flat_idx = LLVM.add!(builder, idx, ce_flat, "wg_negce_adj")
                        elseif length(ops) >= 3
                            # gep T, ptr %ce, i64 0, i64 %idx (or more)
                            # Compute dynamic flat index from all GEP indices
                            flat_idx = LLVM.ConstantInt(T_i64, byte_offset ÷ scalar_size)
                            gep_cur = gep_src_ty
                            for i in 2:length(ops)
                                idx = ops[i]
                                if LLVM.value_type(idx) != T_i64
                                    idx = LLVM.sext!(builder, idx, T_i64, "wg_negce_idx64")
                                end
                                stride = total_scalar_count(gep_cur)
                                if stride != 1
                                    idx = LLVM.mul!(builder, idx, LLVM.ConstantInt(T_i64, stride), "wg_negce_s")
                                end
                                flat_idx = LLVM.add!(builder, flat_idx, idx, "wg_negce_acc")
                                if gep_cur isa LLVM.ArrayType
                                    gep_cur = LLVM.eltype(gep_cur)
                                end
                            end
                        else
                            continue
                        end

                        new_gep = LLVM.gep!(builder, pointee_ty, gv,
                                           LLVM.Value[zero, flat_idx], "wg_negce_gep")
                        LLVM.replace_uses!(inst, new_gep)
                    end
                    LLVM.erase!(inst)

                elseif inst isa LLVM.LoadInst
                    # Load from negative CE: compute flat index and load from there
                    scalar_ty = LLVM.eltype(pointee_ty)
                    scalar_size = Int64(LLVM.storage_size(dl, scalar_ty))
                    flat_idx_val = byte_offset ÷ scalar_size

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        zero = LLVM.ConstantInt(T_i64, 0)
                        off = LLVM.ConstantInt(T_i64, flat_idx_val)
                        ptr = LLVM.gep!(builder, pointee_ty, gv,
                                       LLVM.Value[zero, off], "wg_negce_ptr")
                        new_load = LLVM.load!(builder, LLVM.value_type(inst), ptr, "wg_negce_load")
                        LLVM.alignment!(new_load, max(1, Int(LLVM.alignment(inst))))
                        LLVM.replace_uses!(inst, new_load)
                    end
                    LLVM.erase!(inst)

                elseif inst isa LLVM.StoreInst
                    scalar_ty = LLVM.eltype(pointee_ty)
                    scalar_size = Int64(LLVM.storage_size(dl, scalar_ty))
                    flat_idx_val = byte_offset ÷ scalar_size

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        zero = LLVM.ConstantInt(T_i64, 0)
                        off = LLVM.ConstantInt(T_i64, flat_idx_val)
                        ptr = LLVM.gep!(builder, pointee_ty, gv,
                                       LLVM.Value[zero, off], "wg_negce_ptr")
                        val = LLVM.operands(inst)[1]
                        new_store = LLVM.store!(builder, val, ptr)
                        LLVM.alignment!(new_store, max(1, Int(LLVM.alignment(inst))))
                    end
                    LLVM.erase!(inst)
                end
            end
        end
    end
end

function flatten_nested_workgroup_arrays!(mod::LLVM.Module)
    T_i64 = LLVM.Int64Type()

    for gv in collect(LLVM.globals(mod))
        pointee_ty = LLVM.global_value_type(gv)
        ptr_ty = LLVM.value_type(gv)
        ptr_ty isa LLVM.PointerType || continue
        LLVM.addrspace(ptr_ty) == 3 || continue
        pointee_ty isa LLVM.ArrayType || continue

        # Check if this is a nested array (element is also an array)
        leaf_ty = LLVM.eltype(pointee_ty)
        leaf_ty isa LLVM.ArrayType || continue  # only fix nested arrays -- structs handled elsewhere

        # Skip flattening for simple 2-level nesting (e.g. [N x [2 x i32]] from Julia structs).
        # The flattening creates negative-index ConstantExprs from Julia's 1-based pointer
        # adjustment that are invalid in SPIR-V. Only flatten deeper nesting (3+ levels)
        # which doesn't occur with Julia struct types.
        inner_leaf = LLVM.eltype(leaf_ty)
        if !(inner_leaf isa LLVM.ArrayType)
            continue
        end

        # Compute leaf scalar type and total count
        total_count = LLVM.length(pointee_ty)
        cur = leaf_ty
        while cur isa LLVM.ArrayType
            total_count *= LLVM.length(cur)
            cur = LLVM.eltype(cur)
        end
        scalar_ty = cur  # e.g. float
        inner_count = total_count ÷ LLVM.length(pointee_ty)

        # Create new flat global: [total_count x scalar_ty]
        flat_arr_ty = LLVM.ArrayType(scalar_ty, total_count)
        old_name = LLVM.name(gv)
        new_gv = LLVM.GlobalVariable(mod, flat_arr_ty, old_name * "_flat", 3)
        LLVM.linkage!(new_gv, LLVM.API.LLVMExternalLinkage)
        LLVM.unnamed_addr!(new_gv, true)

        # Rewrite all uses of the old global to use the flat global
        rewrite_all_wg_uses_to_flat!(gv, new_gv, flat_arr_ty, inner_count, T_i64)

        # Handle remaining uses: ConstantExprs with negative indices (Julia 1-based
        # adjustment) that survived rewrite_all_wg_uses_to_flat!.
        # RAUW would blindly replace @shared with @flat, creating invalid negative-index
        # ConstantExprs (SPIR-V can't handle negative OpAccessChain indices).
        # Instead, explicitly handle each remaining CE by inlining the flat offset.
        remaining_ces = LLVM.Value[]
        for use in LLVM.uses(gv)
            user = LLVM.user(use)
            user isa LLVM.ConstantExpr && push!(remaining_ces, user)
        end
        for ce in remaining_ces
            eliminate_remaining_wg_constexpr!(ce, gv, new_gv, flat_arr_ty, T_i64)
        end
        # Only RAUW if there are still remaining uses (shouldn't happen after above)
        if !isempty(collect(LLVM.uses(gv)))
            LLVM.replace_uses!(gv, new_gv)
        end
        LLVM.erase!(gv)
    end
end

"""Rewrite all users of a workgroup global to use flat array indexing."""
function rewrite_all_wg_uses_to_flat!(old_gv, new_gv, flat_arr_ty, inner_count, T_i64)
    # Iteratively process uses until none remain.
    # ConstantExpr users persist as LLVM constants even after their instruction-level
    # users are removed — track them to avoid infinite loops.
    processed_ces = Set{UInt}()

    for _iter in 1:100  # safety bound
        uses = Tuple{LLVM.Value, Symbol}[]
        for use in LLVM.uses(old_gv)
            user = LLVM.user(use)
            if user isa LLVM.GetElementPtrInst
                push!(uses, (user, :gep))
            elseif user isa LLVM.ConstantExpr
                ce_id = UInt(user.ref)
                ce_id in processed_ces && continue
                push!(uses, (user, :constexpr))
            elseif user isa LLVM.LoadInst
                push!(uses, (user, :load))
            elseif user isa LLVM.StoreInst
                push!(uses, (user, :store))
            end
        end
        isempty(uses) && break

        for (user, kind) in uses
            if kind == :gep
                rewrite_one_wg_gep!(user, old_gv, new_gv, flat_arr_ty, inner_count, T_i64)
            elseif kind == :constexpr
                eliminate_wg_constexpr!(user, old_gv, new_gv, flat_arr_ty, inner_count, T_i64)
                push!(processed_ces, UInt(user.ref))
            elseif kind == :load
                # Bare load from global: load from [0]
                LLVM.@dispose builder=LLVM.IRBuilder() begin
                    LLVM.position!(builder, user)
                    zero = LLVM.ConstantInt(T_i64, 0)
                    ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                   LLVM.Value[zero, zero], "wg_flat_ptr")
                    loaded_ty = LLVM.value_type(user)
                    new_load = LLVM.load!(builder, loaded_ty, ptr, "wg_flat_load")
                    LLVM.alignment!(new_load, max(1, Int(LLVM.alignment(user))))
                    LLVM.replace_uses!(user, new_load)
                end
                LLVM.erase!(user)
            elseif kind == :store
                # Bare store to global: store to [0]
                LLVM.@dispose builder=LLVM.IRBuilder() begin
                    LLVM.position!(builder, user)
                    zero = LLVM.ConstantInt(T_i64, 0)
                    ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                   LLVM.Value[zero, zero], "wg_flat_ptr")
                    val = LLVM.operands(user)[1]
                    new_store = LLVM.store!(builder, val, ptr)
                    LLVM.alignment!(new_store, max(1, Int(LLVM.alignment(user))))
                end
                LLVM.erase!(user)
            end
        end
    end
end

"""Total number of scalar elements in an LLVM type (recursing through arrays)."""
function total_scalar_count(ty::LLVM.LLVMType)
    ty isa LLVM.ArrayType || return 1
    return LLVM.length(ty) * total_scalar_count(LLVM.eltype(ty))
end

"""Rewrite one GetElementPtrInst on a workgroup global to use flat indexing.

Uses `LLVMGetGEPSourceElementType` to determine the stride for each index level:
- First GEP index does pointer arithmetic (stride = total scalars in source type)
- Each subsequent index descends one array level (stride = total scalars in element type)

If the GEP doesn't fully resolve to scalar level (e.g., only indexes the outer dimension
of `[N x [6 x float]]`), chained GEP users are recursively folded into a single flat index.
"""
function rewrite_one_wg_gep!(gep::LLVM.GetElementPtrInst, old_gv, new_gv,
                                flat_arr_ty, inner_count, T_i64)
    ops = LLVM.operands(gep)
    length(ops) >= 2 || return  # need at least base + 1 index

    # Get the GEP source element type
    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, gep)
        zero = LLVM.ConstantInt(T_i64, 0)
        flat_idx = zero  # accumulate flat scalar index

        cur_ty = src_ty  # type at current level
        for i in 2:length(ops)
            idx = ops[i]
            if LLVM.value_type(idx) != T_i64
                idx = LLVM.sext!(builder, idx, T_i64, "wg_flat_idx64")
            end

            # Stride = total scalar elements in cur_ty
            stride = total_scalar_count(cur_ty)

            if stride != 1
                scaled = LLVM.mul!(builder, idx, LLVM.ConstantInt(T_i64, stride), "wg_flat_scale")
            else
                scaled = idx
            end
            flat_idx = LLVM.add!(builder, flat_idx, scaled, "wg_flat_acc")

            # Descend into the element type for the next index
            if cur_ty isa LLVM.ArrayType
                cur_ty = LLVM.eltype(cur_ty)
            end
        end

        if !(cur_ty isa LLVM.ArrayType)
            # Fully resolved to scalar — simple replacement
            new_gep = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, flat_idx], "wg_flat_gep")
            LLVM.replace_uses!(gep, new_gep)
            LLVM.erase!(gep)
            # After replacing uses, new_gep may have chained GEP users with stale
            # non-scalar source types (e.g., `gep [2 x i32], ptr %new_gep, i64 N`).
            # This happens when Julia splits an index expression into two GEPs and the
            # flattening only rewrote the first one. Rewrite such users to flat indexing.
            fixup_chained_flat_gep_users!(new_gep, new_gv, flat_arr_ty, flat_idx, T_i64)
        else
            # GEP didn't reach scalar level — fold chained users into flat index
            rewrite_partial_wg_gep_users!(gep, new_gv, flat_arr_ty, flat_idx, cur_ty, T_i64)
        end
    end
end

"""Rewrite users of a partially-resolved workgroup GEP by folding their indices into the flat index.

Called when a GEP on a flattened workgroup global doesn't descend to scalar level
(e.g., `gep [N x [6 x float]], @shared, 0, %outer` reaches `[6 x float]`, not `float`).
Chained GEP users have their indices folded into the base flat_idx to produce fully-resolved
scalar GEPs on the new flat global.
"""
function rewrite_partial_wg_gep_users!(gep, new_gv, flat_arr_ty, base_flat_idx,
                                          remaining_ty, T_i64)
    zero = LLVM.ConstantInt(T_i64, 0)

    # Collect users before mutation
    users = LLVM.Value[]
    for use in LLVM.uses(gep)
        push!(users, LLVM.user(use))
    end

    for user in users
        if user isa LLVM.GetElementPtrInst
            # Fold this chained GEP's indices into base_flat_idx
            chained_ops = LLVM.operands(user)
            chained_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))

            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, user)
                flat_idx = base_flat_idx
                cur_ty = chained_src_ty

                for i in 2:length(chained_ops)
                    idx = chained_ops[i]
                    if LLVM.value_type(idx) != T_i64
                        idx = LLVM.sext!(builder, idx, T_i64, "wg_chain_idx64")
                    end
                    stride = total_scalar_count(cur_ty)
                    if stride != 1
                        scaled = LLVM.mul!(builder, idx, LLVM.ConstantInt(T_i64, stride), "wg_chain_scale")
                    else
                        scaled = idx
                    end
                    flat_idx = LLVM.add!(builder, flat_idx, scaled, "wg_chain_acc")
                    if cur_ty isa LLVM.ArrayType
                        cur_ty = LLVM.eltype(cur_ty)
                    end
                end

                if !(cur_ty isa LLVM.ArrayType)
                    # Fully resolved — create scalar GEP
                    new_gep_val = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                       LLVM.Value[zero, flat_idx], "wg_flat_chain_gep")
                    LLVM.replace_uses!(user, new_gep_val)
                    LLVM.erase!(user)
                else
                    # Still not fully resolved — recurse
                    rewrite_partial_wg_gep_users!(user, new_gv, flat_arr_ty,
                                                     flat_idx, cur_ty, T_i64)
                end
            end
        elseif user isa LLVM.LoadInst
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, user)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, base_flat_idx], "wg_flat_ptr")
                new_load = LLVM.load!(builder, LLVM.value_type(user), ptr, "wg_flat_load")
                LLVM.alignment!(new_load, max(1, Int(LLVM.alignment(user))))
                LLVM.replace_uses!(user, new_load)
            end
            LLVM.erase!(user)
        elseif user isa LLVM.StoreInst
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, user)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, base_flat_idx], "wg_flat_ptr")
                val = LLVM.operands(user)[1]
                new_store = LLVM.store!(builder, val, ptr)
                LLVM.alignment!(new_store, max(1, Int(LLVM.alignment(user))))
            end
            LLVM.erase!(user)
        end
    end

    # Erase old GEP if no remaining uses
    if isempty(collect(LLVM.uses(gep)))
        LLVM.erase!(gep)
    end
end

"""Fix chained GEPs on a fully-resolved flat workgroup pointer.

After flattening `[N x [K x T]]` to `[N*K x T]` and resolving a GEP to scalar level,
the result pointer may still be used by GEPs with stale `[K x T]` source types.
E.g., `gep [2 x i32], ptr %flat_scalar, i64 64` should become `flat_idx + 64 * 2`.
Rewrites such users to proper flat index arithmetic.
"""
function fixup_chained_flat_gep_users!(base_gep, new_gv, flat_arr_ty, base_flat_idx, T_i64)
    zero = LLVM.ConstantInt(T_i64, 0)

    for use in collect(LLVM.uses(base_gep))
        user = LLVM.user(use)
        user isa LLVM.GetElementPtrInst || continue
        chained_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
        # Only fix GEPs with non-scalar source types (the stale inner array type)
        (chained_src_ty isa LLVM.ArrayType || chained_src_ty isa LLVM.StructType) || continue

        chained_ops = LLVM.operands(user)
        local new_gep_val
        local final_flat_idx
        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, user)
            flat_idx = base_flat_idx
            cur_ty = chained_src_ty

            for i in 2:length(chained_ops)
                idx = chained_ops[i]
                if LLVM.value_type(idx) != T_i64
                    idx = LLVM.sext!(builder, idx, T_i64, "wg_flat_fix_idx64")
                end
                stride = total_scalar_count(cur_ty)
                if stride != 1
                    scaled = LLVM.mul!(builder, idx, LLVM.ConstantInt(T_i64, stride), "wg_flat_fix_scale")
                else
                    scaled = idx
                end
                flat_idx = LLVM.add!(builder, flat_idx, scaled, "wg_flat_fix_acc")
                if cur_ty isa LLVM.ArrayType
                    cur_ty = LLVM.eltype(cur_ty)
                end
            end

            new_gep_val = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                   LLVM.Value[zero, flat_idx], "wg_flat_fix_gep")
            final_flat_idx = flat_idx
            LLVM.replace_uses!(user, new_gep_val)
        end
        LLVM.erase!(user)
        # Recursively fix further chained GEPs on the new result
        fixup_chained_flat_gep_users!(new_gep_val, new_gv, flat_arr_ty, final_flat_idx, T_i64)
    end
end

"""Handle a ConstantExpr GEP on the OLD workgroup global that survived rewrite_all_wg_uses_to_flat!.

Computes the flat scalar offset from the CE's constant indices (which use the old nested-array
type), then replaces all instruction users with flat GEPs on the new global that incorporate
the offset into the dynamic index. This avoids creating ConstantExprs with negative indices
(from Julia's 1-based pointer adjustment) that would be invalid in SPIR-V.
"""
function eliminate_remaining_wg_constexpr!(ce::LLVM.ConstantExpr, old_gv, new_gv,
                                              flat_arr_ty, T_i64)
    opcode = LLVM.API.LLVMGetConstOpcode(ce)
    opcode == LLVM.API.LLVMGetElementPtr || return

    # Compute flat offset from the CE's indices using the OLD type hierarchy
    ce_ops = LLVM.operands(ce)
    pointee_ty = LLVM.global_value_type(old_gv)
    flat_offset = 0
    cur_ty = pointee_ty
    for i in 2:length(ce_ops)
        idx = ce_ops[i]
        idx isa LLVM.ConstantInt || continue
        val = convert(Int, idx)
        stride = total_scalar_count(cur_ty)
        flat_offset += val * stride
        if cur_ty isa LLVM.ArrayType
            cur_ty = LLVM.eltype(cur_ty)
        end
    end

    # Replace each instruction user with a flat GEP that incorporates the offset
    for use in collect(LLVM.uses(ce))
        user = LLVM.user(use)
        if user isa LLVM.LoadInst || user isa LLVM.StoreInst || user isa LLVM.GetElementPtrInst
            # Delegate to the existing eliminate_wg_constexpr! logic
        elseif user isa LLVM.ConstantExpr
            # Nested CE: recursively handle
            continue
        else
            continue
        end

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, user)
            zero = LLVM.ConstantInt(T_i64, 0)

            if user isa LLVM.LoadInst
                # load from CE: create flat GEP + load
                # For negative offsets this is always the CE being used as a base ptr by
                # a dynamic GEP chain -- but some loads may have been constant-folded.
                # Create instruction-level GEP so the emitter can handle it properly.
                off = LLVM.ConstantInt(T_i64, flat_offset)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, off], "wg_flat_rem_ptr")
                new_load = LLVM.load!(builder, LLVM.value_type(user), ptr, "wg_flat_rem_load")
                LLVM.alignment!(new_load, max(1, Int(LLVM.alignment(user))))
                LLVM.replace_uses!(user, new_load)
                LLVM.erase!(user)
            elseif user isa LLVM.StoreInst
                off = LLVM.ConstantInt(T_i64, flat_offset)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, off], "wg_flat_rem_ptr")
                val = LLVM.operands(user)[1]
                new_store = LLVM.store!(builder, val, ptr)
                LLVM.alignment!(new_store, max(1, Int(LLVM.alignment(user))))
                LLVM.erase!(user)
            elseif user isa LLVM.GetElementPtrInst
                # GEP on CE: fold CE offset into the GEP's dynamic index
                ops = LLVM.operands(user)
                gep_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                stride = total_scalar_count(gep_src_ty)
                dynamic_inner = zero
                # Compute the dynamic part from GEP indices
                gep_cur_ty = gep_src_ty
                for i in 2:length(ops)
                    idx = ops[i]
                    if LLVM.value_type(idx) != T_i64
                        idx = LLVM.sext!(builder, idx, T_i64, "wg_flat_rem_idx64")
                    end
                    s = total_scalar_count(gep_cur_ty)
                    if s != 1
                        idx = LLVM.mul!(builder, idx, LLVM.ConstantInt(T_i64, s), "wg_flat_rem_scale")
                    end
                    dynamic_inner = LLVM.add!(builder, dynamic_inner, idx, "wg_flat_rem_acc")
                    if gep_cur_ty isa LLVM.ArrayType
                        gep_cur_ty = LLVM.eltype(gep_cur_ty)
                    end
                end
                # Add CE flat offset to dynamic index
                adj_idx = if flat_offset != 0
                    LLVM.add!(builder, dynamic_inner, LLVM.ConstantInt(T_i64, flat_offset), "wg_flat_rem_adj")
                else
                    dynamic_inner
                end
                new_gep = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                   LLVM.Value[zero, adj_idx], "wg_flat_rem_gep")
                LLVM.replace_uses!(user, new_gep)
                LLVM.erase!(user)
            end
        end
    end
end

"""Replace ConstantExpr GEP users with instruction-level GEPs on the flat global."""
function eliminate_wg_constexpr!(ce::LLVM.ConstantExpr, old_gv, new_gv,
                                    flat_arr_ty, inner_count, T_i64)
    # Compute the flat offset from the ConstantExpr's constant indices.
    # Walk the global's pointee type structure to compute strides at each level.
    # CEs on workgroup globals always use the outer array type as source.
    ce_ops = LLVM.operands(ce)
    pointee_ty = LLVM.global_value_type(old_gv)
    flat_offset = 0
    cur_ty = pointee_ty
    for i in 2:length(ce_ops)
        idx = ce_ops[i]
        idx isa LLVM.ConstantInt || continue
        val = convert(Int, idx)
        stride = total_scalar_count(cur_ty)
        flat_offset += val * stride
        if cur_ty isa LLVM.ArrayType
            cur_ty = LLVM.eltype(cur_ty)
        end
    end

    # Replace each instruction user of this ConstantExpr with a flat GEP
    ce_inst_users = LLVM.Value[]
    for use in LLVM.uses(ce)
        push!(ce_inst_users, LLVM.user(use))
    end

    for inst in ce_inst_users
        if inst isa LLVM.LoadInst
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, inst)
                zero = LLVM.ConstantInt(T_i64, 0)
                off = LLVM.ConstantInt(T_i64, flat_offset)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, off], "wg_flat_ce_ptr")
                loaded_ty = LLVM.value_type(inst)
                new_load = LLVM.load!(builder, loaded_ty, ptr, "wg_flat_ce_load")
                LLVM.alignment!(new_load, max(1, Int(LLVM.alignment(inst))))
                LLVM.replace_uses!(inst, new_load)
            end
            LLVM.erase!(inst)
        elseif inst isa LLVM.StoreInst
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, inst)
                zero = LLVM.ConstantInt(T_i64, 0)
                off = LLVM.ConstantInt(T_i64, flat_offset)
                ptr = LLVM.gep!(builder, flat_arr_ty, new_gv,
                               LLVM.Value[zero, off], "wg_flat_ce_ptr")
                val = LLVM.operands(inst)[1]
                new_store = LLVM.store!(builder, val, ptr)
                LLVM.alignment!(new_store, max(1, Int(LLVM.alignment(inst))))
            end
            LLVM.erase!(inst)
        elseif inst isa LLVM.GetElementPtrInst
            # GEP on ConstantExpr result: CE provides base offset, GEP adds dynamic index.
            # Common pattern: gep [M x T], %ce, i64 0, i64 %inner_idx (3 ops)
            # or:             gep [M x T], %ce, i64 %ptr_idx           (2 ops)
            ops = LLVM.operands(inst)
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, inst)
                zero = LLVM.ConstantInt(T_i64, 0)
                dynamic_inner = zero  # default: no dynamic offset
                if length(ops) >= 4
                    # ops = [base, 0, inner_idx, ...]
                    idx = ops[4]
                    if LLVM.value_type(idx) != T_i64
                        idx = LLVM.sext!(builder, idx, T_i64, "wg_flat_ce_idx64")
                    end
                    dynamic_inner = idx
                elseif length(ops) >= 3
                    # ops = [base, 0, inner_idx]
                    idx = ops[3]
                    if LLVM.value_type(idx) != T_i64
                        idx = LLVM.sext!(builder, idx, T_i64, "wg_flat_ce_idx64")
                    end
                    dynamic_inner = idx
                end
                adj_idx = if flat_offset != 0
                    LLVM.add!(builder, dynamic_inner, LLVM.ConstantInt(T_i64, flat_offset), "wg_flat_ce_adj")
                else
                    dynamic_inner
                end
                new_gep = LLVM.gep!(builder, flat_arr_ty, new_gv,
                                   LLVM.Value[zero, adj_idx], "wg_flat_ce_gep")
                LLVM.replace_uses!(inst, new_gep)
            end
            LLVM.erase!(inst)
        end
    end
    # Note: the ConstantExpr itself will be cleaned up when it has no more uses
end

# ============================================================================
# Sub-pass: Fix shared memory GEPs
# ============================================================================

"""
    fix_shared_geps!(mod::LLVM.Module)

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
function fix_shared_geps!(mod::LLVM.Module)
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
    gep_byte_offset(dl, src_ty, indices) -> Int or nothing

Compute the total byte offset for a sequence of constant GEP indices starting
from `src_ty`. Returns `nothing` if any index is non-constant.
"""
function gep_byte_offset(dl::LLVM.DataLayout, src_ty::LLVM.LLVMType,
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
    flatten_bda_array_geps!(mod::LLVM.Module)

Replace composite-typed GEPs in addrspace(1) with explicit pointer arithmetic
(ptrtoint -> add -> inttoptr).

Julia represents tuples as LLVM array types (e.g. `[3 x float]` for NTuple{3,Float32}).
The SPIR-V emitter would need ArrayStride decorations for PhysicalStorageBuffer arrays,
and multi-level GEPs crash the bitcast legalization. By replacing composite-typed GEPs
with explicit byte arithmetic, we avoid these composite types in SPIR-V entirely.

Only handles GEPs with all-constant indices starting with 0.
"""
function flatten_bda_array_geps!(mod::LLVM.Module)
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
                byte_off = gep_byte_offset(dl, src_ty, remaining)
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
                    byte_off = gep_byte_offset(dl, src_ty, remaining)

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
    warn_constant_globals!(mod::LLVM.Module)

Check for constant global variables in addrspace(1) that would be invalid in
Vulkan SPIR-V. These indicate a missing device function override -- the Julia
stdlib function is using lookup tables that produce global constant arrays.

PhysicalStorageBuffer storage class cannot have OpVariable declarations.
The fix is to override the functions that create these globals (e.g. ^(Float64,Float64))
with intrinsic-based or polynomial implementations.
"""
function warn_constant_globals!(mod::LLVM.Module)
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
    prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String)

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
function prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String;
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
                        min_align = max(4, scalar_size(ty))
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

    # 3b. Fix flat GEPs on addrspace(3) array globals
    fix_shared_geps!(mod)

    # 4. Flatten array-typed GEPs in addrspace(1) (BDA / PhysicalStorageBuffer)
    flatten_bda_array_geps!(mod)

    # 5. Lower llvm.memset/memcpy on addrspace(1) to explicit stores/loads
    lower_psb_memops!(mod)

    # 6. Decompose composite loads/stores on addrspace(1)
    decompose_composite_psb_accesses!(mod, dl)

    # 6b. Decompose composite loads/stores on addrspace(3) (shared memory)
    decompose_composite_workgroup_accesses!(mod, dl)

    # 6c. Decompose type-punned alloca loads (partial reads from struct allocas)
    decompose_typepun_alloca_loads!(mod, dl)

    # 7. Warn about invalid constant globals in addrspace(1)
    warn_constant_globals!(mod)

    # 8. Fix type-mismatched loads from allocas
    # LLVM may optimize loads where the loaded scalar type differs from the alloca's
    # type (e.g., `load i16, ptr %alloca_of_{[2 x i8]}`). SPIR-V requires strict
    # type matching for loads, so we rewrite these to load the alloca type instead.
    fix_alloca_type_mismatched_loads!(mod)

    # 9. Fix type-mismatched stores to allocas
    # LLVM SROA/memcpy lowering creates `store i32, ptr %alloca_of_[16 x i64]` etc.
    # SPIR-V requires the stored value type to match the pointer's pointee type.
    # Rewrite these stores to drill into the alloca type via GEP.
    fix_alloca_type_mismatched_stores!(mod, dl)

    # 10. Lower chained mismatched-type GEPs on allocas.
    # Julia's MArray/StaticArray patterns create chains like:
    #   %base = getelementptr i32, ptr %alloca_[16 x i64], i64 -1
    #   %elem = getelementptr i32, ptr %base, i64 %var
    #   store i32 %val, ptr %elem
    # The i32-typed GEP on an i64-element alloca can't be represented in SPIR-V.
    # Lower these to proper element-level access with runtime index computation.
    lower_chained_mismatched_geps!(mod)
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
function fix_alloca_type_mismatched_loads!(mod::LLVM.Module)
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
                    alloca = trace_to_alloca(ptr)
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
                    fields = flatten_type_to_scalars(alloca_ty)
                    isempty(fields) && continue

                    # Verify all fields are scalar types we can bitcast to integer
                    all_ok = true
                    for (_, fty) in fields
                        fsz = llvm_type_size(fty)
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
                            field_bits = llvm_type_size(field_ty) * 8

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

"""
Fix stores where the stored type differs from the alloca's type.

LLVM SROA/memcpy lowering creates patterns like:
    %alloca = alloca [16 x i64]
    store i32 0, ptr %alloca             ; type mismatch!
    store i32 0, ptr (gep i8, %alloca, 4) ; byte-offset store

Also handles struct unpacking:
    %alloca = alloca { [1 x [3 x float]] }
    store float %val, ptr %alloca        ; first field store

This pass rewrites type-mismatched stores to use typed GEPs that SPIR-V can handle.
For partial element writes (i32 into i64), we use read-modify-write.
"""

"""
    fold_typepun_scalar_alloca_constants!(mod, dl)

Fold type-punned scalar allocas where ALL stores are constants and there's a load
of the alloca's type. SROA decomposes e.g. `zero(Float64)` into partial stores:

    %a = alloca double
    store float 0.0, ptr %a              ; bytes [0..3]
    %p4 = gep i8, ptr %a, i64 4
    store i32 0, ptr %p4                  ; bytes [4..7]
    %v = load double, ptr %a             ; full 8-byte load

Replace the load with a constant computed from the stored bytes, then remove
the dead stores and alloca.
"""
function fold_typepun_scalar_alloca_constants!(mod::LLVM.Module, dl)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        entry = first(LLVM.blocks(fn))
        allocas_to_process = LLVM.AllocaInst[]
        for inst in LLVM.instructions(entry)
            inst isa LLVM.AllocaInst || continue
            alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
            (alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType) && continue
            push!(allocas_to_process, inst)
        end
        for inst in allocas_to_process
            alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
            (alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType) && continue
            alloca_size = Int(LLVM.storage_size(dl, alloca_ty))
            alloca_size == 0 && continue

            bytes = fill(UInt8(0xff), alloca_size)
            all_const = true
            stores = LLVM.Instruction[]
            loads = LLVM.Instruction[]
            geps = LLVM.Instruction[]
            other_uses = false

            for use in LLVM.uses(inst)
                user_inst = LLVM.user(use)
                if user_inst isa LLVM.StoreInst
                    ops = LLVM.operands(user_inst)
                    ops[2] == inst || (other_uses = true; break)
                    val = ops[1]
                    val isa LLVM.ConstantFP || val isa LLVM.ConstantInt || (all_const = false; break)
                    vsize = Int(LLVM.storage_size(dl, LLVM.value_type(val)))
                    raw = constant_to_bytes(val, vsize)
                    raw === nothing && (all_const = false; break)
                    for (j, b) in enumerate(raw)
                        j <= alloca_size || break
                        bytes[j] = b
                    end
                    push!(stores, user_inst)
                elseif user_inst isa LLVM.LoadInst
                    push!(loads, user_inst)
                elseif user_inst isa LLVM.GetElementPtrInst
                    push!(geps, user_inst)
                    for gep_use in LLVM.uses(user_inst)
                        gep_user = LLVM.user(gep_use)
                        if gep_user isa LLVM.StoreInst
                            gep_ops = LLVM.operands(gep_user)
                            gep_ops[2] == user_inst || (other_uses = true; break)
                            val = gep_ops[1]
                            val isa LLVM.ConstantFP || val isa LLVM.ConstantInt || (all_const = false; break)
                            gep_operands = LLVM.operands(user_inst)
                            length(gep_operands) == 2 || (all_const = false; break)
                            gep_operands[2] isa LLVM.ConstantInt || (all_const = false; break)
                            offset = convert(Int, gep_operands[2])
                            vsize = Int(LLVM.storage_size(dl, LLVM.value_type(val)))
                            raw = constant_to_bytes(val, vsize)
                            raw === nothing && (all_const = false; break)
                            for (j, b) in enumerate(raw)
                                idx = offset + j
                                1 <= idx <= alloca_size || (all_const = false; break)
                                bytes[idx] = b
                            end
                            push!(stores, gep_user)
                        else
                            other_uses = true; break
                        end
                    end
                else
                    other_uses = true
                end
                (!all_const || other_uses) && break
            end

            !all_const && continue
            other_uses && continue
            isempty(loads) && continue
            any(==(0xff), bytes) && continue

            for load_inst in loads
                load_ty = LLVM.value_type(load_inst)
                const_val = bytes_to_constant(bytes, load_ty)
                const_val === nothing && continue
                LLVM.replace_uses!(load_inst, const_val)
                LLVM.erase!(load_inst)
            end
            for s in stores; LLVM.erase!(s); end
            for g in geps; isempty(LLVM.uses(g)) && LLVM.erase!(g); end
            isempty(LLVM.uses(inst)) && LLVM.erase!(inst)
        end
    end
end

function constant_to_bytes(val::LLVM.ConstantInt, size::Int)
    v = convert(UInt64, val)
    return [UInt8((v >> (8*(i-1))) & 0xff) for i in 1:size]
end
function constant_to_bytes(val::LLVM.ConstantFP, size::Int)
    ty = LLVM.value_type(val)
    if ty == LLVM.FloatType() && size == 4
        return collect(reinterpret(UInt8, [convert(Float32, val)]))
    elseif ty == LLVM.DoubleType() && size == 8
        return collect(reinterpret(UInt8, [convert(Float64, val)]))
    end
    return nothing
end
constant_to_bytes(::LLVM.Value, ::Int) = nothing

function bytes_to_constant(bytes::Vector{UInt8}, ty::LLVM.FloatingPointType)
    if ty == LLVM.FloatType() && length(bytes) >= 4
        return LLVM.ConstantFP(ty, Float64(reinterpret(Float32, bytes[1:4])[1]))
    elseif ty == LLVM.DoubleType() && length(bytes) >= 8
        return LLVM.ConstantFP(ty, reinterpret(Float64, bytes[1:8])[1])
    end
    return nothing
end
function bytes_to_constant(bytes::Vector{UInt8}, ty::LLVM.IntegerType)
    nbytes = (LLVM.width(ty) + 7) ÷ 8
    length(bytes) >= nbytes || return nothing
    v = UInt64(0)
    for i in 1:nbytes; v |= UInt64(bytes[i]) << (8*(i-1)); end
    return LLVM.ConstantInt(ty, v)
end
bytes_to_constant(::Vector{UInt8}, ::LLVM.LLVMType) = nothing

function fix_alloca_type_mismatched_stores!(mod::LLVM.Module, dl)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            to_erase = LLVM.Instruction[]
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.StoreInst || continue

                    ops = LLVM.operands(inst)
                    value = ops[1]
                    ptr = ops[2]
                    store_ty = LLVM.value_type(value)

                    # Handle stores to allocas or byte-offset GEPs into allocas
                    alloca = nothing
                    byte_offset_from_alloca = 0
                    if ptr isa LLVM.AllocaInst
                        alloca = ptr
                    elseif ptr isa LLVM.GetElementPtrInst
                        # Check if this is a byte-offset GEP into an alloca
                        gep_ops = LLVM.operands(ptr)
                        base = gep_ops[1]
                        if base isa LLVM.AllocaInst
                            src_ety = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
                            if src_ety isa LLVM.IntegerType && LLVM.width(src_ety) == 8
                                # Byte-offset GEP: gep i8, alloca, offset
                                if length(gep_ops) == 2 && gep_ops[2] isa LLVM.ConstantInt
                                    byte_offset_from_alloca = convert(Int, gep_ops[2])
                                    alloca = base
                                end
                            end
                        end
                    end
                    alloca === nothing && continue

                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))

                    # For direct stores to alloca (offset=0), skip if types match
                    if byte_offset_from_alloca == 0
                        store_ty == alloca_ty && continue
                    end

                    # Only fix composite alloca types (struct/array)
                    !(alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType) && continue

                    store_size = llvm_type_size(store_ty)
                    alloca_size = Int(LLVM.storage_size(dl, alloca_ty))
                    byte_offset_from_alloca == 0 && store_size >= alloca_size && continue  # full-size store

                    # Find the element at the given byte offset
                    elem_ty, elem_gep_indices, inner_byte_offset = find_element_at_offset(alloca_ty, byte_offset_from_alloca)
                    elem_ty === nothing && continue

                    elem_size = llvm_type_size(elem_ty)

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)

                        if inner_byte_offset == 0 && store_size == elem_size
                            # Perfect match: store value directly into the element
                            idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                            for idx in elem_gep_indices
                                push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                            end
                            field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "store_fix_gep")

                            if store_ty == elem_ty
                                LLVM.store!(builder, value, field_ptr)
                            else
                                cast_val = LLVM.bitcast!(builder, value, elem_ty, "store_fix_cast")
                                LLVM.store!(builder, cast_val, field_ptr)
                            end
                            push!(to_erase, inst)
                            changed = true

                        elseif elem_ty isa LLVM.IntegerType && store_ty isa LLVM.IntegerType && store_size < elem_size
                            # Partial write into element: RMW with bit shifting
                            idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                            for idx in elem_gep_indices
                                push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                            end
                            field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "store_rmw_gep")

                            old_val = LLVM.load!(builder, elem_ty, field_ptr, "store_rmw_load")
                            elem_bits = LLVM.width(elem_ty)
                            store_bits = LLVM.width(store_ty)
                            shift_bits = inner_byte_offset * 8

                            # Zero-extend stored value to element width, then shift into position
                            ext_val = LLVM.zext!(builder, value, elem_ty, "store_rmw_zext")
                            if shift_bits > 0
                                shift = LLVM.ConstantInt(elem_ty, shift_bits)
                                ext_val = LLVM.shl!(builder, ext_val, shift, "store_rmw_shl")
                            end

                            # Mask out the target bits of old value
                            clear_mask = ~(((UInt64(1) << store_bits) - 1) << shift_bits)
                            mask = LLVM.ConstantInt(elem_ty, clear_mask)
                            masked = LLVM.and!(builder, old_val, mask, "store_rmw_mask")

                            # OR in the new value
                            merged = LLVM.or!(builder, masked, ext_val, "store_rmw_merge")
                            LLVM.store!(builder, merged, field_ptr)
                            push!(to_erase, inst)
                            changed = true
                        end
                    end
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end
end

"""
    lower_chained_mismatched_geps!(mod::LLVM.Module)

Lower stores/loads through chained GEPs where the GEP source element type doesn't match
the alloca's element type.

Pattern:
    %alloca = alloca [N x i64]
    %base = getelementptr i32, ptr %alloca, i64 <const>   ; mismatched source type!
    %elem = getelementptr i32, ptr %base, i64 %var
    store i32 %val, ptr %elem
    ; or: %loaded = load i32, ptr %elem

The i32 GEP on an i64 alloca is a type reinterpretation (viewing i64 as pairs of i32).
We lower this to:
    %adj = add i64 %var, <const>                 ; combined i32-index from alloca
    %i64_idx = lshr i64 %adj, 1                  ; which i64 element (adj / 2)
    %is_high = and i64 %adj, 1                    ; low half (0) or high half (1)
    %gep = getelementptr [N x i64], ptr %alloca, i64 0, i64 %i64_idx
    ; For stores: read-modify-write with shift/mask based on is_high
    ; For loads: load i64, shift, truncate
"""
function lower_chained_mismatched_geps!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            to_erase = LLVM.Instruction[]
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    # Find the "base" GEP: gep <small_type>, ptr %alloca, i64 <const>
                    # where small_type doesn't match the alloca element type
                    inst isa LLVM.GetElementPtrInst || continue
                    gep_ops = LLVM.operands(inst)
                    base = gep_ops[1]
                    base isa LLVM.AllocaInst || continue
                    length(gep_ops) == 2 || continue  # single-index GEP
                    gep_ops[2] isa LLVM.ConstantInt || continue  # constant index

                    alloca = base
                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                    alloca_ty isa LLVM.ArrayType || continue

                    src_ety = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    alloca_elem_ty = LLVM.eltype(alloca_ty)

                    # Only handle when GEP source type is smaller than alloca element type
                    src_size = llvm_type_size(src_ety)
                    elem_size = llvm_type_size(alloca_elem_ty)
                    (src_size > 0 && elem_size > 0 && src_size < elem_size) || continue
                    # And both are integer types (or at least the alloca element is)
                    alloca_elem_ty isa LLVM.IntegerType || continue

                    const_idx = convert(Int64, gep_ops[2])
                    ratio = elem_size ÷ src_size  # how many small elements fit in one large element

                    # Process all users of this base GEP
                    for use in collect(LLVM.uses(inst))
                        user = LLVM.user(use)

                        if user isa LLVM.GetElementPtrInst
                            # Chained GEP: gep <type>, ptr %base, i64 %var
                            user_ops = LLVM.operands(user)
                            length(user_ops) == 2 || continue
                            user_src_ety = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                            user_src_size = llvm_type_size(user_src_ety)

                            if user_src_size == src_size
                                # Same source type: combined index is const_idx + var_idx
                                var_idx = user_ops[2]
                                lower_chained_gep_users!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                          elem_size, src_size, ratio,
                                                          const_idx, var_idx, user, to_erase)
                                changed = true
                            elseif user_src_size == 1
                                # Byte-offset GEP from the base: gep i8, ptr %base, i64 %byte_off
                                # Combined byte offset from alloca = const_idx * src_size + byte_off
                                byte_off_var = user_ops[2]
                                lower_byte_offset_from_base_gep!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                                  elem_size, const_idx, src_size,
                                                                  byte_off_var, user, to_erase)
                                changed = true
                            else
                                continue
                            end

                        elseif user isa LLVM.StoreInst
                            # Direct store through base GEP (constant index only)
                            store_ops = LLVM.operands(user)
                            store_ops[2] == inst || continue  # inst must be the pointer operand
                            lower_direct_mismatched_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                             elem_size, src_size, ratio,
                                                             const_idx, user, :store, to_erase)
                            changed = true

                        elseif user isa LLVM.LoadInst
                            lower_direct_mismatched_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                             elem_size, src_size, ratio,
                                                             const_idx, user, :load, to_erase)
                            changed = true
                        end
                    end

                    # If all uses are handled, erase the base GEP too
                    if isempty(collect(LLVM.uses(inst)))
                        push!(to_erase, inst)
                    end
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end
end

"""Lower users of a chained GEP (base_gep → user_gep → store/load)."""
function lower_chained_gep_users!(fn, alloca, alloca_ty, alloca_elem_ty,
                                   elem_size, src_size, ratio,
                                   const_idx, var_idx, user_gep, to_erase)
    for use in collect(LLVM.uses(user_gep))
        user = LLVM.user(use)
        if user isa LLVM.StoreInst
            store_ops = LLVM.operands(user)
            store_ops[2] == user_gep || continue
            lower_variable_idx_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                        elem_size, src_size, ratio,
                                        const_idx, var_idx,
                                        store_ops[1], user, :store, to_erase)
        elseif user isa LLVM.LoadInst
            lower_variable_idx_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                        elem_size, src_size, ratio,
                                        const_idx, var_idx,
                                        nothing, user, :load, to_erase)
        elseif user isa LLVM.GetElementPtrInst
            # Triple-chained GEP (e.g., gep i8, ptr %user_gep, %byte_off)
            # The user_gep already has an i32-typed variable index, and this adds a byte offset.
            # Compute combined byte offset: (const_idx + var_idx) * src_size + this_gep_byte_offset
            user2_ops = LLVM.operands(user)
            length(user2_ops) == 2 || continue
            user2_src_ety = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
            user2_src_size = llvm_type_size(user2_src_ety)
            # This must be a byte-offset GEP (i8 source)
            user2_src_size == 1 || continue

            # The byte offset from this third GEP: user2_ops[2] (may be variable)
            # Combined byte offset = (const_idx + var_idx) * src_size + byte_offset_3
            # This is too complex for general handling; handle users of THIS gep recursively
            byte_off_3 = user2_ops[2]
            for use3 in collect(LLVM.uses(user))
                user3 = LLVM.user(use3)
                if user3 isa LLVM.StoreInst
                    store3_ops = LLVM.operands(user3)
                    store3_ops[2] == user || continue
                    lower_triple_chain_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                elem_size, src_size, ratio,
                                                const_idx, var_idx, byte_off_3,
                                                store3_ops[1], user3, :store, to_erase)
                elseif user3 isa LLVM.LoadInst
                    lower_triple_chain_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                                elem_size, src_size, ratio,
                                                const_idx, var_idx, byte_off_3,
                                                nothing, user3, :load, to_erase)
                end
            end
            if isempty(collect(LLVM.uses(user)))
                push!(to_erase, user)
            end
        end
    end
    # Erase user_gep if all its uses are handled
    if isempty(collect(LLVM.uses(user_gep)))
        push!(to_erase, user_gep)
    end
end

"""
Lower a variable-index access through a chained GEP to proper element-level access.
Combined index from alloca: adj = const_idx + var_idx
Byte offset: adj * src_size
Element index: byte_offset / elem_size
Inner byte offset: byte_offset % elem_size
"""
function lower_variable_idx_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                     elem_size, src_size, ratio,
                                     const_idx, var_idx,
                                     store_value, inst, mode, to_erase)
    i64 = LLVM.Int64Type()
    i32 = LLVM.Int32Type()

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        # Compute adjusted index: adj = var_idx + const_idx
        var_i64 = if LLVM.value_type(var_idx) != i64
            LLVM.sext!(builder, var_idx, i64, "chain_sext")
        else
            var_idx
        end
        adj = if const_idx == 0
            var_i64
        else
            LLVM.add!(builder, var_i64, LLVM.ConstantInt(i64, const_idx), "chain_adj")
        end

        # Compute byte offset: byte_off = adj * src_size
        byte_off = if src_size == 1
            adj
        else
            LLVM.mul!(builder, adj, LLVM.ConstantInt(i64, src_size), "chain_byte")
        end

        emit_element_rmw_access!(builder, alloca, alloca_ty, alloca_elem_ty,
                                  elem_size, byte_off, store_value, inst, mode, to_erase)
    end
end

"""
Lower a triple-chained GEP access.
Combined byte offset: (const_idx + var_idx) * src_size + byte_off_3
"""
function lower_triple_chain_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                     elem_size, src_size, ratio,
                                     const_idx, var_idx, byte_off_3,
                                     store_value, inst, mode, to_erase)
    i64 = LLVM.Int64Type()

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        # adj = var_idx + const_idx
        var_i64 = if LLVM.value_type(var_idx) != i64
            LLVM.sext!(builder, var_idx, i64, "tc_sext")
        else
            var_idx
        end
        adj = if const_idx == 0
            var_i64
        else
            LLVM.add!(builder, var_i64, LLVM.ConstantInt(i64, const_idx), "tc_adj")
        end

        # byte_off_base = adj * src_size
        byte_off_base = if src_size == 1
            adj
        else
            LLVM.mul!(builder, adj, LLVM.ConstantInt(i64, src_size), "tc_base_byte")
        end

        # total_byte_off = byte_off_base + byte_off_3
        byte_off_3_i64 = if LLVM.value_type(byte_off_3) != i64
            LLVM.sext!(builder, byte_off_3, i64, "tc_bo3_sext")
        else
            byte_off_3
        end
        total_byte_off = LLVM.add!(builder, byte_off_base, byte_off_3_i64, "tc_total_byte")

        emit_element_rmw_access!(builder, alloca, alloca_ty, alloca_elem_ty,
                                  elem_size, total_byte_off, store_value, inst, mode, to_erase)
    end
end

"""
Lower accesses through a base mismatched GEP followed by a byte-offset GEP.
Pattern: base = gep i32, ptr alloca, i64 const; ptr = gep i8, ptr base, i64 byte_off_var
Combined byte offset from alloca: const * src_size + byte_off_var
"""
function lower_byte_offset_from_base_gep!(fn, alloca, alloca_ty, alloca_elem_ty,
                                           elem_size, const_idx, src_size,
                                           byte_off_var, user_gep, to_erase)
    i64 = LLVM.Int64Type()
    base_byte_offset = const_idx * src_size  # constant part

    for use in collect(LLVM.uses(user_gep))
        user = LLVM.user(use)
        if user isa LLVM.StoreInst
            store_ops = LLVM.operands(user)
            store_ops[2] == user_gep || continue
            lower_byte_offset_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                       elem_size, base_byte_offset, byte_off_var,
                                       store_ops[1], user, :store, to_erase)
        elseif user isa LLVM.LoadInst
            lower_byte_offset_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                       elem_size, base_byte_offset, byte_off_var,
                                       nothing, user, :load, to_erase)
        end
    end
    if isempty(collect(LLVM.uses(user_gep)))
        push!(to_erase, user_gep)
    end
end

function lower_byte_offset_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                    elem_size, base_byte_offset, byte_off_var,
                                    store_value, inst, mode, to_erase)
    i64 = LLVM.Int64Type()

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        # total_byte_off = base_byte_offset + byte_off_var
        bo_var_i64 = if LLVM.value_type(byte_off_var) != i64
            LLVM.sext!(builder, byte_off_var, i64, "bo_sext")
        else
            byte_off_var
        end
        total_byte_off = if base_byte_offset == 0
            bo_var_i64
        else
            LLVM.add!(builder, bo_var_i64, LLVM.ConstantInt(i64, base_byte_offset), "bo_total")
        end

        emit_element_rmw_access!(builder, alloca, alloca_ty, alloca_elem_ty,
                                  elem_size, total_byte_off, store_value, inst, mode, to_erase)
    end
end

"""
Lower a direct access (constant index only) through the base mismatched GEP.
"""
function lower_direct_mismatched_access!(fn, alloca, alloca_ty, alloca_elem_ty,
                                          elem_size, src_size, ratio,
                                          const_idx, inst, mode, to_erase)
    byte_offset = const_idx * src_size
    elem_idx = byte_offset ÷ elem_size
    inner = byte_offset % elem_size

    if elem_idx < 0
        # Negative offset — this is just the base adjustment, skip
        return
    end

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        gep = LLVM.gep!(builder, alloca_ty, alloca,
                         [LLVM.ConstantInt(LLVM.Int64Type(), 0),
                          LLVM.ConstantInt(LLVM.Int64Type(), elem_idx)],
                         "dm_gep")

        if mode == :store
            value = LLVM.operands(inst)[1]
            store_ty = LLVM.value_type(value)
            store_bits = llvm_type_size(store_ty) * 8
            elem_bits = elem_size * 8
            shift_bits = inner * 8

            old_val = LLVM.load!(builder, alloca_elem_ty, gep, "dm_load")
            ext_val = LLVM.zext!(builder, value, alloca_elem_ty, "dm_zext")
            if shift_bits > 0
                ext_val = LLVM.shl!(builder, ext_val, LLVM.ConstantInt(alloca_elem_ty, shift_bits), "dm_shl")
            end
            clear_mask = ~(((UInt64(1) << store_bits) - 1) << shift_bits)
            masked = LLVM.and!(builder, old_val, LLVM.ConstantInt(alloca_elem_ty, clear_mask), "dm_mask")
            merged = LLVM.or!(builder, masked, ext_val, "dm_merge")
            LLVM.store!(builder, merged, gep)
            push!(to_erase, inst)
        else  # :load
            load_ty = LLVM.value_type(inst)
            load_bits = llvm_type_size(load_ty) * 8
            shift_bits = inner * 8

            full = LLVM.load!(builder, alloca_elem_ty, gep, "dm_full")
            if shift_bits > 0
                full = LLVM.lshr!(builder, full, LLVM.ConstantInt(alloca_elem_ty, shift_bits), "dm_shr")
            end
            result = LLVM.trunc!(builder, full, load_ty, "dm_trunc")
            LLVM.replace_uses!(inst, result)
            push!(to_erase, inst)
        end
    end
end

"""
Emit element-level RMW access given a byte offset expression (may be variable).
Computes: elem_idx = byte_off / elem_size, inner = byte_off % elem_size
Then does proper load/store with shift/mask for partial element access.
"""
function emit_element_rmw_access!(builder, alloca, alloca_ty, alloca_elem_ty,
                                   elem_size, byte_off, store_value, inst, mode, to_erase)
    i64 = LLVM.Int64Type()
    elem_bits = elem_size * 8

    # elem_idx = byte_off / elem_size (arithmetic shift right for power-of-2)
    elem_shift = Int(log2(elem_size))
    elem_idx = LLVM.ashr!(builder, byte_off, LLVM.ConstantInt(i64, elem_shift), "chain_eidx")

    # inner_byte = byte_off % elem_size (mask for power-of-2)
    inner_byte = LLVM.and!(builder, byte_off, LLVM.ConstantInt(i64, elem_size - 1), "chain_inner")

    # shift_bits = inner_byte * 8
    shift_bits = LLVM.shl!(builder, inner_byte, LLVM.ConstantInt(i64, 3), "chain_shift")

    # GEP to the element
    gep = LLVM.gep!(builder, alloca_ty, alloca,
                     [LLVM.ConstantInt(i64, 0), elem_idx], "chain_gep")

    if mode == :store
        store_ty = LLVM.value_type(store_value)
        store_bits = llvm_type_size(store_ty) * 8

        old_val = LLVM.load!(builder, alloca_elem_ty, gep, "chain_old")

        # Zero-extend stored value to element width
        ext_val = if store_ty == alloca_elem_ty
            store_value
        elseif llvm_type_size(store_ty) < elem_size
            LLVM.zext!(builder, store_value, alloca_elem_ty, "chain_zext")
        else
            store_value
        end

        # Convert shift from i64 to alloca_elem_ty
        shift_in_elem_ty = LLVM.trunc!(builder, shift_bits, alloca_elem_ty, "chain_shift_t")

        # Shift value into position
        shifted_val = LLVM.shl!(builder, ext_val, shift_in_elem_ty, "chain_shl")

        # Create mask: ~(((1 << store_bits) - 1) << shift)
        mask_base = LLVM.ConstantInt(alloca_elem_ty, (UInt64(1) << store_bits) - 1)
        mask_shifted = LLVM.shl!(builder, mask_base, shift_in_elem_ty, "chain_mask_s")
        mask_inv = LLVM.not!(builder, mask_shifted, "chain_mask_inv")
        masked_old = LLVM.and!(builder, old_val, mask_inv, "chain_masked")

        # OR in new value
        merged = LLVM.or!(builder, masked_old, shifted_val, "chain_merged")
        LLVM.store!(builder, merged, gep)
        push!(to_erase, inst)

    else  # :load
        load_ty = LLVM.value_type(inst)
        load_bits = llvm_type_size(load_ty) * 8

        full = LLVM.load!(builder, alloca_elem_ty, gep, "chain_full")

        # Convert shift from i64 to alloca_elem_ty
        shift_in_elem_ty = LLVM.trunc!(builder, shift_bits, alloca_elem_ty, "chain_shift_t")

        # Shift right
        shifted = LLVM.lshr!(builder, full, shift_in_elem_ty, "chain_shr")

        # Truncate
        result = LLVM.trunc!(builder, shifted, load_ty, "chain_trunc")
        LLVM.replace_uses!(inst, result)
        push!(to_erase, inst)
    end
end

"""
    find_element_at_offset(ty, byte_offset) → (element_type, gep_indices, inner_byte_offset)

Find the scalar element at a given byte offset within a composite type.
Returns the element type, the GEP index path to reach it, and any remaining
byte offset within that element (for partial writes).
"""
function find_element_at_offset(ty::LLVM.LLVMType, byte_offset::Int)
    indices = Int[]
    current = ty
    remaining = byte_offset
    while true
        if current isa LLVM.ArrayType
            elem_ty = LLVM.eltype(current)
            elem_size = llvm_type_size(elem_ty)
            elem_size == 0 && return nothing, Int[], 0
            idx = remaining ÷ elem_size
            remaining = remaining % elem_size
            push!(indices, idx)
            if remaining == 0 || !(elem_ty isa LLVM.StructType || elem_ty isa LLVM.ArrayType)
                return elem_ty, indices, remaining
            end
            current = elem_ty
        elseif current isa LLVM.StructType
            elems = LLVM.elements(current)
            offset_acc = 0
            found = false
            for (i, member_ty) in enumerate(elems)
                member_size = llvm_type_size(member_ty)
                if offset_acc + member_size > remaining
                    push!(indices, i - 1)
                    remaining -= offset_acc
                    if remaining == 0 || !(member_ty isa LLVM.StructType || member_ty isa LLVM.ArrayType)
                        return member_ty, indices, remaining
                    end
                    current = member_ty
                    found = true
                    break
                end
                offset_acc += member_size
            end
            !found && return nothing, Int[], 0
        else
            # Scalar type at this level
            return current, indices, remaining
        end
    end
end

"""Get the first scalar field type and its GEP index path for a composite type."""
function first_scalar_field(ty::LLVM.LLVMType)
    indices = Int[]
    current = ty
    while true
        if current isa LLVM.StructType
            elems = LLVM.elements(current)
            isempty(elems) && return nothing, Int[]
            push!(indices, 0)
            current = elems[1]
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return nothing, Int[]
            push!(indices, 0)
            current = LLVM.eltype(current)
        else
            # Reached a scalar
            return current, indices
        end
    end
end

"""Trace a pointer through GEPs back to its alloca, if any."""
# Check if a GEP chain from ptr to its alloca contains any GEP with dynamic indices.
function gep_chain_has_dynamic_indices(ptr::LLVM.Value)
    current = ptr
    while current isa LLVM.GetElementPtrInst
        ops = LLVM.operands(current)
        for i in 2:length(ops)
            if resolve_const_gep_index(ops[i]) === nothing
                return true
            end
        end
        current = ops[1]
    end
    return false
end

function trace_to_alloca(ptr::LLVM.Value)
    ptr isa LLVM.AllocaInst && return ptr
    if ptr isa LLVM.GetElementPtrInst
        return trace_to_alloca(LLVM.operands(ptr)[1])
    elseif ptr isa LLVM.BitCastInst || ptr isa LLVM.AddrSpaceCastInst
        return trace_to_alloca(LLVM.operands(ptr)[1])
    end
    return nothing
end

"""
Try to resolve an LLVM value used as a GEP index to a compile-time integer.
Handles literal constants and simple constant-expression instruction chains.
"""
function resolve_const_gep_index(v::LLVM.Value)
    if v isa LLVM.ConstantInt
        return convert(Int, v)
    elseif v isa LLVM.ZExtInst || v isa LLVM.SExtInst || v isa LLVM.TruncInst
        return resolve_const_gep_index(LLVM.operands(v)[1])
    elseif v isa LLVM.AddInst
        a = resolve_const_gep_index(LLVM.operands(v)[1])
        b = resolve_const_gep_index(LLVM.operands(v)[2])
        return (a === nothing || b === nothing) ? nothing : (a + b)
    elseif v isa LLVM.SubInst
        a = resolve_const_gep_index(LLVM.operands(v)[1])
        b = resolve_const_gep_index(LLVM.operands(v)[2])
        return (a === nothing || b === nothing) ? nothing : (a - b)
    end
    return nothing
end

"""
    gep_element_type_and_offset(src_ty, indices) → (element_type, byte_offset)

Walk GEP indices through a type hierarchy to find the pointed-to element type
and its byte offset from the base. The first index is a pointer-level array
offset; subsequent indices drill into struct members / array elements.
"""
function gep_element_type_and_offset(src_ty::LLVM.LLVMType, indices::Vector{Int})
    offset = 0
    current_ty = src_ty
    for (j, idx) in enumerate(indices)
        if j == 1
            # First index: array offset from pointer base
            offset += idx * llvm_type_size(src_ty)
        elseif current_ty isa LLVM.StructType
            elems = LLVM.elements(current_ty)
            for i in 0:(idx-1)
                offset += llvm_type_size(elems[i+1])
            end
            current_ty = elems[idx+1]
        elseif current_ty isa LLVM.ArrayType
            elem_ty = LLVM.eltype(current_ty)
            offset += idx * llvm_type_size(elem_ty)
            current_ty = elem_ty
        else
            return nothing, 0
        end
    end
    return current_ty, offset
end

"""
    flatten_type_with_offsets(ty; dl=nothing) → [(index_path, scalar_type, byte_offset), ...]

Like `flatten_type_to_scalars` but also computes byte offsets.
When `dl` is provided, uses LLVM's DataLayout for correct struct padding.
Without `dl`, offsets are computed by summing field sizes (no padding -- only correct for packed types).
"""
function flatten_type_with_offsets(ty::LLVM.LLVMType; dl::Union{Nothing,LLVM.DataLayout}=nothing)
    result = Tuple{Vector{Int}, LLVM.LLVMType, Int}[]
    if dl !== nothing
        flatten_with_dl!(result, Int[], ty, 0, dl)
    else
        # Legacy path: sum sizes without padding (only correct for packed types)
        scalars = flatten_type_to_scalars(ty)
        offset = 0
        for (path, sty) in scalars
            push!(result, (path, sty, offset))
            offset += llvm_type_size(sty)
        end
    end
    return result
end

function flatten_with_dl!(result, path, ty::LLVM.StructType, base_offset, dl::LLVM.DataLayout)
    for i in 0:(length(LLVM.elements(ty)) - 1)
        member_ty = LLVM.elements(ty)[i + 1]
        member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, ty, UInt32(i)))
        flatten_with_dl!(result, vcat(path, [i]), member_ty, base_offset + member_offset, dl)
    end
end

function flatten_with_dl!(result, path, ty::LLVM.ArrayType, base_offset, dl::LLVM.DataLayout)
    elem_ty = LLVM.eltype(ty)
    elem_size = Int(LLVM.storage_size(dl, elem_ty))
    for i in 0:(LLVM.length(ty) - 1)
        flatten_with_dl!(result, vcat(path, [i]), elem_ty, base_offset + i * elem_size, dl)
    end
end

function flatten_with_dl!(result, path, ty::LLVM.LLVMType, base_offset, ::LLVM.DataLayout)
    push!(result, (path, ty, base_offset))
end

"""
    decompose_typepun_gep_loads!(mod, dl)

Decompose type-punning loads through GEPs into struct allocas.

LLVM's memcpy optimization generates patterns like:
    %gep = getelementptr { [3 x float], ... }, ptr %alloca, 0, 0, 0, 2
    %val = load i64, ptr %gep   ; reads 8 bytes starting from a float field

SPIR-V requires strict type matching — can't load i64 from a float*.
This pass decomposes such loads into properly-typed field loads + pack:
    %f1 = load float, ptr %gep_to_field_at_offset_8
    %f2 = load float, ptr %gep_to_field_at_offset_12
    %i1 = bitcast float %f1 to i32
    %i2 = bitcast float %f2 to i32
    %w1 = zext i32 %i1 to i64
    %w2 = zext i32 %i2 to i64
    %w2s = shl i64 %w2, 32
    %val = or i64 %w1, %w2s

For narrower loads (e.g., load i8 from i32*), loads the full type and truncates.
"""
function decompose_typepun_gep_loads!(mod::LLVM.Module, dl::LLVM.DataLayout)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.LoadInst || continue
                    load_ty = LLVM.value_type(inst)
                    is_int_load = load_ty isa LLVM.IntegerType
                    is_float_load = (load_ty == LLVM.FloatType() ||
                                     load_ty == LLVM.DoubleType() ||
                                     load_ty == LLVM.HalfType())
                    (is_int_load || is_float_load) || continue
                    load_size = llvm_type_size(load_ty)
                    load_bits = load_size * 8
                    load_int_ty = is_int_load ? load_ty : LLVM.IntType(load_bits)

                    ptr = LLVM.operands(inst)[1]
                    ptr isa LLVM.GetElementPtrInst || continue

                    # Skip already-decomposed loads
                    startswith(LLVM.name(inst), "typepun_") && continue

                    alloca = trace_to_alloca(ptr)
                    alloca === nothing && continue

                    # Skip if the GEP chain to the alloca passes through any GEP with
                    # dynamic (non-constant) indices. We only decompose fully-constant
                    # offset chains. Dynamic offsets must be handled by the SPIR-V emitter.
                    gep_chain_has_dynamic_indices(ptr) && continue

                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))

                    # Get GEP source element type, indices, and compute destination type + offset
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
                    ops = LLVM.operands(ptr)
                    indices = Int[]
                    for i in 2:length(ops)
                        op = ops[i]
                        idx = resolve_const_gep_index(op)
                        idx === nothing && @goto next_inst
                        push!(indices, idx)
                    end

                    elem_ty, gep_byte_off = gep_element_type_and_offset(src_ty, indices)
                    elem_ty === nothing && continue

                    elem_size = llvm_type_size(elem_ty)

                    # Handle byte-GEPs into struct/array allocas (gep i8, ptr %alloca, <offset>)
                    # These have src_ty=i8, elem_ty=i8, but the alloca is a struct/array.
                    # Resolve byte offset to the containing field, load it, and extract
                    # the relevant bits via bitcast+shift+trunc.
                    if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8 &&
                       (alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType)
                        all_fields = flatten_type_with_offsets(alloca_ty; dl)
                        # Find the scalar field that contains this byte offset
                        containing = nothing
                        for (path, fty, foff) in all_fields
                            fsz = llvm_type_size(fty)
                            if gep_byte_off >= foff && gep_byte_off < foff + fsz
                                containing = (path, fty, foff)
                                break
                            end
                        end
                        containing === nothing && continue
                        cpath, cfty, cfoff = containing
                        byte_within_field = gep_byte_off - cfoff

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            # GEP to the containing field
                            idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                            for idx in cpath
                                push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                            end
                            field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")
                            field_val = LLVM.load!(builder, cfty, field_ptr, "typepun_load")

                            # Convert to integer
                            field_bits = llvm_type_size(cfty) * 8
                            int_val = if cfty isa LLVM.IntegerType
                                field_val
                            else
                                LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                            end

                            # Shift right to extract the byte
                            result = if byte_within_field > 0
                                shift = LLVM.ConstantInt(LLVM.value_type(int_val), byte_within_field * 8)
                                shifted = LLVM.lshr!(builder, int_val, shift, "typepun_shr")
                                LLVM.trunc!(builder, shifted, load_int_ty, "typepun_trunc")
                            elseif field_bits > load_bits
                                LLVM.trunc!(builder, int_val, load_int_ty, "typepun_trunc")
                            else
                                int_val
                            end

                            typed_result = is_int_load ? result :
                                LLVM.bitcast!(builder, result, load_ty, "typepun_bcast")

                            LLVM.replace_uses!(inst, typed_result)
                            LLVM.erase!(inst)
                            changed = true
                            break
                        end
                        changed && break  # restart inner loop
                        continue
                    end

                    # Only handle type-punning (load size != element size)
                    elem_size == load_size && continue

                    if load_size > elem_size
                        # Wider load: select scalar fields within [gep_byte_off, gep_byte_off + load_size)
                        all_fields = flatten_type_with_offsets(alloca_ty; dl)
                        selected = filter(all_fields) do (path, fty, foff)
                            fsz = llvm_type_size(fty)
                            foff >= gep_byte_off && foff + fsz <= gep_byte_off + load_size
                        end

                        isempty(selected) && continue

                        # Verify all fields are standard sizes
                        all(f -> llvm_type_size(f[2]) in (1, 2, 4, 8), selected) || continue

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            combined = nothing

                            for (gep_indices, field_ty, field_offset) in selected
                                field_bits = llvm_type_size(field_ty) * 8
                                bit_offset = (field_offset - gep_byte_off) * 8

                                # GEP from alloca base to this field
                                idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                                for idx in gep_indices
                                    push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                                end
                                field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")
                                field_val = LLVM.load!(builder, field_ty, field_ptr, "typepun_load")

                                # Bitcast to integer if needed
                                int_val = if field_ty isa LLVM.IntegerType
                                    field_val
                                else
                                    LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                                end

                                # Zero-extend to load width
                                wide_val = if field_bits == load_bits
                                    int_val
                                else
                                    LLVM.zext!(builder, int_val, load_int_ty, "typepun_zext")
                                end

                                # Shift left
                                if bit_offset > 0
                                    shift = LLVM.ConstantInt(load_int_ty, bit_offset)
                                    wide_val = LLVM.shl!(builder, wide_val, shift, "typepun_shl")
                                end

                                combined = combined === nothing ? wide_val :
                                    LLVM.or!(builder, combined, wide_val, "typepun_or")
                            end

                            if combined !== nothing
                                typed_result = is_int_load ? combined :
                                    LLVM.bitcast!(builder, combined, load_ty, "typepun_bcast")
                                LLVM.replace_uses!(inst, typed_result)
                                LLVM.erase!(inst)
                                changed = true
                                break
                            end
                        end

                    elseif load_size < elem_size
                        # Narrower load: load the full field and truncate
                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            replaced = false

                            # Avoid illegal i128/i256 bitcasts (unsupported in Vulkan SPIR-V):
                            # for wide composite pointees, assemble only the needed low bytes
                            # from scalar subfields.
                            if (elem_ty isa LLVM.StructType || elem_ty isa LLVM.ArrayType) &&
                               (elem_size * 8 > 64)
                                subfields = flatten_type_with_offsets(elem_ty; dl)
                                selected = filter(subfields) do (_, fty, foff)
                                    fsz = llvm_type_size(fty)
                                    foff >= 0 && foff + fsz <= load_size
                                end
                                if !isempty(selected) &&
                                   all(f -> llvm_type_size(f[2]) in (1, 2, 4, 8), selected)
                                    combined = nothing
                                    for (gep_indices, field_ty, field_offset) in selected
                                        field_bits = llvm_type_size(field_ty) * 8
                                        bit_offset = field_offset * 8

                                        idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                                        for idx in gep_indices
                                            push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                                        end
                                        field_ptr = LLVM.gep!(builder, elem_ty, ptr, idx_values, "typepun_gep")
                                        field_val = LLVM.load!(builder, field_ty, field_ptr, "typepun_load")

                                        int_val = if field_ty isa LLVM.IntegerType
                                            field_val
                                        else
                                            LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                                        end
                                        wide_val = field_bits == load_bits ? int_val :
                                            LLVM.zext!(builder, int_val, load_int_ty, "typepun_zext")
                                        if bit_offset > 0
                                            shift = LLVM.ConstantInt(load_int_ty, bit_offset)
                                            wide_val = LLVM.shl!(builder, wide_val, shift, "typepun_shl")
                                        end
                                        combined = combined === nothing ? wide_val :
                                            LLVM.or!(builder, combined, wide_val, "typepun_or")
                                    end

                                    if combined !== nothing
                                        result = is_int_load ? combined :
                                            LLVM.bitcast!(builder, combined, load_ty, "typepun_bcast")
                                        LLVM.replace_uses!(inst, result)
                                        LLVM.erase!(inst)
                                        changed = true
                                        replaced = true
                                    end
                                end
                            end

                            if !replaced
                                field_bits = elem_size * 8
                                if field_bits <= 64
                                    field_val = LLVM.load!(builder, elem_ty, ptr, "typepun_load")
                                    int_val = if elem_ty isa LLVM.IntegerType
                                        field_val
                                    else
                                        LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                                    end

                                    result_i = LLVM.trunc!(builder, int_val, load_int_ty, "typepun_trunc")
                                    result = is_int_load ? result_i :
                                        LLVM.bitcast!(builder, result_i, load_ty, "typepun_bcast")

                                    LLVM.replace_uses!(inst, result)
                                    LLVM.erase!(inst)
                                    changed = true
                                    replaced = true
                                end
                            end

                            replaced && break
                        end
                    end

                    @label next_inst
                    changed && break
                end
                changed && break
            end
        end
    end

    # Second pass: decompose type-punning STORES through GEPs into struct allocas
    # Pattern: store i64 %packed, ptr %gep_to_float_field
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.StoreInst || continue

                    store_val = LLVM.operands(inst)[1]
                    store_ty = LLVM.value_type(store_val)
                    store_ty isa LLVM.IntegerType || continue

                    ptr = LLVM.operands(inst)[2]
                    ptr isa LLVM.GetElementPtrInst || continue

                    alloca = trace_to_alloca(ptr)
                    alloca === nothing && continue

                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))

                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
                    ops = LLVM.operands(ptr)
                    indices = Int[]
                    for i in 2:length(ops)
                        op = ops[i]
                        idx = resolve_const_gep_index(op)
                        idx === nothing && @goto next_store
                        push!(indices, idx)
                    end

                    elem_ty, gep_byte_off = gep_element_type_and_offset(src_ty, indices)
                    elem_ty === nothing && continue

                    elem_size = llvm_type_size(elem_ty)
                    store_size = div(LLVM.width(store_ty), 8)

                    # Handle byte-GEP stores (gep i8, ptr %struct_alloca, <offset>)
                    if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8 &&
                       (alloca_ty isa LLVM.StructType || alloca_ty isa LLVM.ArrayType) &&
                       elem_size == store_size
                        all_fields = flatten_type_with_offsets(alloca_ty; dl)
                        containing = nothing
                        for (path, fty, foff) in all_fields
                            fsz = llvm_type_size(fty)
                            if gep_byte_off >= foff && gep_byte_off < foff + fsz
                                containing = (path, fty, foff)
                                break
                            end
                        end
                        containing === nothing && continue
                        cpath, cfty, cfoff = containing
                        byte_within_field = gep_byte_off - cfoff

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                            for idx in cpath
                                push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                            end
                            field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")

                            # Read-modify-write: load field, replace byte, store back
                            field_val = LLVM.load!(builder, cfty, field_ptr, "typepun_load")
                            field_bits = llvm_type_size(cfty) * 8
                            int_val = if cfty isa LLVM.IntegerType
                                field_val
                            else
                                LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                            end

                            # Create mask and insert byte
                            byte_val = if LLVM.width(store_ty) < field_bits
                                LLVM.zext!(builder, store_val, LLVM.IntType(field_bits), "typepun_zext")
                            else
                                store_val
                            end
                            if byte_within_field > 0
                                shift = LLVM.ConstantInt(LLVM.IntType(field_bits), byte_within_field * 8)
                                byte_val = LLVM.shl!(builder, byte_val, shift, "typepun_shl")
                            end
                            mask_val = ~(UInt64(0xFF) << (byte_within_field * 8))
                            mask = LLVM.ConstantInt(LLVM.IntType(field_bits), mask_val % UInt64)
                            masked = LLVM.and!(builder, int_val, mask, "typepun_mask")
                            merged = LLVM.or!(builder, masked, byte_val, "typepun_merge")

                            result = if cfty isa LLVM.IntegerType
                                merged
                            else
                                LLVM.bitcast!(builder, merged, cfty, "typepun_bcast")
                            end
                            LLVM.store!(builder, result, field_ptr)
                        end
                        LLVM.erase!(inst)
                        changed = true
                        break
                    end

                    elem_size == store_size && continue

                    if store_size > elem_size
                        # Wider store: decompose into per-field stores
                        all_fields = flatten_type_with_offsets(alloca_ty; dl)
                        selected = filter(all_fields) do (path, fty, foff)
                            fsz = llvm_type_size(fty)
                            foff >= gep_byte_off && foff + fsz <= gep_byte_off + store_size
                        end

                        isempty(selected) && continue
                        all(f -> llvm_type_size(f[2]) in (1, 2, 4, 8), selected) || continue

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)

                            for (gep_indices, field_ty, field_offset) in selected
                                field_bits = llvm_type_size(field_ty) * 8
                                bit_offset = (field_offset - gep_byte_off) * 8

                                # Extract bits for this field
                                extracted = store_val
                                if bit_offset > 0
                                    shift = LLVM.ConstantInt(store_ty, bit_offset)
                                    extracted = LLVM.lshr!(builder, extracted, shift, "typepun_shr")
                                end
                                if field_bits < LLVM.width(store_ty)
                                    extracted = LLVM.trunc!(builder, extracted, LLVM.IntType(field_bits), "typepun_trunc")
                                end

                                # Bitcast to field type if needed
                                field_val = if field_ty isa LLVM.IntegerType
                                    extracted
                                else
                                    LLVM.bitcast!(builder, extracted, field_ty, "typepun_bcast")
                                end

                                # GEP and store
                                idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                                for idx in gep_indices
                                    push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                                end
                                field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")
                                LLVM.store!(builder, field_val, field_ptr)
                            end
                        end
                        LLVM.erase!(inst)
                        changed = true
                        break

                    elseif store_size < elem_size
                        # Narrower store to wider field: find the containing scalar and
                        # do read-modify-write at the scalar level
                        all_fields = flatten_type_with_offsets(alloca_ty; dl)

                        # First check: exact size match (store directly)
                        target_field = nothing
                        for (path, fty, foff) in all_fields
                            fsz = llvm_type_size(fty)
                            if foff == gep_byte_off && fsz == store_size
                                target_field = (path, fty, foff)
                                break
                            end
                        end

                        if target_field !== nothing
                            fpath, fty, _ = target_field
                            LLVM.@dispose builder=LLVM.IRBuilder() begin
                                LLVM.position!(builder, inst)
                                idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                                for idx in fpath
                                    push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                                end
                                field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")
                                val_to_store = if fty isa LLVM.IntegerType || fty == store_ty
                                    store_val
                                else
                                    LLVM.bitcast!(builder, store_val, fty, "typepun_bcast")
                                end
                                LLVM.store!(builder, val_to_store, field_ptr)
                            end
                            LLVM.erase!(inst)
                            changed = true
                            break
                        end

                        # Second check: containing scalar field → read-modify-write
                        containing = nothing
                        for (path, fty, foff) in all_fields
                            fsz = llvm_type_size(fty)
                            if gep_byte_off >= foff && gep_byte_off + store_size <= foff + fsz
                                containing = (path, fty, foff)
                                break
                            end
                        end

                        if containing !== nothing
                            cpath, cfty, cfoff = containing
                            byte_within = gep_byte_off - cfoff
                            field_bits = llvm_type_size(cfty) * 8

                            LLVM.@dispose builder=LLVM.IRBuilder() begin
                                LLVM.position!(builder, inst)
                                # GEP to containing scalar field
                                idx_values = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(32), 0)]
                                for idx in cpath
                                    push!(idx_values, LLVM.ConstantInt(LLVM.IntType(32), idx))
                                end
                                field_ptr = LLVM.gep!(builder, alloca_ty, alloca, idx_values, "typepun_gep")

                                # Load, modify byte, store back
                                field_val = LLVM.load!(builder, cfty, field_ptr, "typepun_load")
                                int_val = if cfty isa LLVM.IntegerType
                                    field_val
                                else
                                    LLVM.bitcast!(builder, field_val, LLVM.IntType(field_bits), "typepun_cast")
                                end

                                # Zext store value and shift into position
                                byte_val = LLVM.zext!(builder, store_val, LLVM.IntType(field_bits), "typepun_zext")
                                if byte_within > 0
                                    shift = LLVM.ConstantInt(LLVM.IntType(field_bits), byte_within * 8)
                                    byte_val = LLVM.shl!(builder, byte_val, shift, "typepun_shl")
                                end

                                # Mask: clear the byte position
                                mask_val = ~(UInt64(0xFF) << (byte_within * 8))
                                mask = LLVM.ConstantInt(LLVM.IntType(field_bits), mask_val % UInt64)
                                masked = LLVM.and!(builder, int_val, mask, "typepun_mask")
                                merged = LLVM.or!(builder, masked, byte_val, "typepun_merge")

                                result = if cfty isa LLVM.IntegerType
                                    merged
                                else
                                    LLVM.bitcast!(builder, merged, cfty, "typepun_bcast")
                                end
                                LLVM.store!(builder, result, field_ptr)
                            end
                            LLVM.erase!(inst)
                            changed = true
                            break
                        end
                    end

                    @label next_store
                    changed && break
                end
                changed && break
            end
        end
    end
end

"""
Flatten a composite LLVM type into a list of (index_path, scalar_type) pairs.
Each index_path is an array of integer indices for GEP.
"""
function flatten_type_to_scalars(ty::LLVM.LLVMType, prefix::Vector{Int}=Int[])
    result = Tuple{Vector{Int}, LLVM.LLVMType}[]

    if ty isa LLVM.StructType
        for (i, field_ty) in enumerate(LLVM.elements(ty))
            new_prefix = copy(prefix)
            push!(new_prefix, i - 1)
            append!(result, flatten_type_to_scalars(field_ty, new_prefix))
        end
    elseif ty isa LLVM.ArrayType
        elem_ty = LLVM.eltype(ty)
        for i in 0:(LLVM.length(ty) - 1)
            new_prefix = copy(prefix)
            push!(new_prefix, i)
            append!(result, flatten_type_to_scalars(elem_ty, new_prefix))
        end
    else
        # Scalar type (int, float, half, double)
        push!(result, (copy(prefix), ty))
    end

    return result
end

"""
    fix_inttoptr_addrspace!(mod::LLVM.Module)

After SROA eliminates allocas, `inttoptr i64 %bda_val to ptr` instructions
(addrspace 0) appear where BDA pointer fields are loaded directly. These should
be `inttoptr i64 %bda_val to ptr addrspace(1)` for PhysicalStorageBuffer.

This pass finds `inttoptr to ptr` (addrspace 0) that feed into loads/stores/GEPs
and converts them to addrspace 1. We also need to update any GEPs and loads/stores
that use the converted pointer to use the correct addrspace(1) pointer type.
"""
function fix_inttoptr_addrspace!(mod::LLVM.Module)
    T_i8 = LLVM.Int8Type()
    T_ptr_as1 = LLVM.PointerType(T_i8, 1)

    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue

        to_fix = LLVM.Instruction[]
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            inst isa LLVM.IntToPtrInst || continue
            result_ty = LLVM.value_type(inst)
            result_ty isa LLVM.PointerType || continue
            # Only fix addrspace 0 → addrspace 1
            LLVM.addrspace(result_ty) == 0 || continue
            # Only fix if the source is an i64 (BDA value)
            src_ty = LLVM.value_type(LLVM.operands(inst)[1])
            src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 64 || continue
            # Verify it has memory-access uses (loads, stores, GEPs) — not just control flow
            has_mem_use = false
            for use in LLVM.uses(inst)
                usr = LLVM.user(use)
                if usr isa LLVM.LoadInst || usr isa LLVM.StoreInst || usr isa LLVM.GetElementPtrInst
                    has_mem_use = true
                    break
                end
            end
            has_mem_use || continue
            push!(to_fix, inst)
        end

        for old_inst in to_fix
            src_val = LLVM.operands(old_inst)[1]
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, old_inst)
                new_inst = LLVM.inttoptr!(builder, src_val, T_ptr_as1, "bda_ptr")
                # Update all uses: replace addrspace(0) pointer with addrspace(1)
                # For GEPs, loads, stores — they work with opaque pointers,
                # the address space is determined by the pointer operand.
                LLVM.replace_uses!(old_inst, new_inst)
                LLVM.erase!(old_inst)
            end
        end
    end
end

"""
    fix_gep_alloca_type_mismatches!(mod::LLVM.Module)

Fix GEPs on allocas whose source element type differs from the alloca's allocated type.

After SROA + inlining of the BDA entry wrapper, some GEPs still reference the original
full tuple type through a pointer to a smaller alloca (valid with opaque pointers, invalid
in SPIR-V where types must match). This pass converts such GEPs to byte-offset GEPs
(`gep i8, ptr %alloca, i64 <offset>`) which the `lift_byte_geps_on_allocas!` pass
will then convert to proper typed GEPs using the alloca's actual type.
"""
function fix_gep_alloca_type_mismatches!(mod::LLVM.Module)
    dl = LLVM.datalayout(mod)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        to_replace = Pair{LLVM.GetElementPtrInst, Int64}[]

        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            inst isa LLVM.GetElementPtrInst || continue

            # Only fix GEPs that directly use an alloca as the pointer base
            ptr_op = LLVM.operands(inst)[1]
            ptr_op isa LLVM.AllocaInst || continue

            alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(ptr_op))
            gep_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))

            # Skip if types match
            alloca_ty == gep_src_ty && continue

            # Skip if GEP source type is i8 (already a byte-offset GEP)
            gep_src_ty isa LLVM.IntegerType && LLVM.width(gep_src_ty) == 8 && continue

            # Compute byte offset of this GEP using the GEP's source element type
            ops = LLVM.operands(inst)
            n_indices = length(ops) - 1  # operands = [ptr, idx0, idx1, ...]
            n_indices == 0 && continue

            # Only handle all-constant indices (the common case for struct accesses)
            all_const = true
            for i in 2:length(ops)
                if !(ops[i] isa LLVM.ConstantInt)
                    all_const = false
                    break
                end
            end
            all_const || continue

            # Compute byte offset through the GEP's source element type
            offset = compute_gep_byte_offset(gep_src_ty, ops, dl)
            offset === nothing && continue

            push!(to_replace, inst => offset)
        end

        # Replace each GEP with a byte-offset GEP
        for (gep_inst, offset) in to_replace
            ptr_op = LLVM.operands(gep_inst)[1]
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, gep_inst)
                i8_ty = LLVM.Int8Type()
                offset_val = LLVM.ConstantInt(LLVM.Int64Type(), offset)
                new_gep = LLVM.inbounds_gep!(builder, i8_ty, ptr_op, [offset_val])
                LLVM.replace_uses!(gep_inst, new_gep)
                LLVM.erase!(gep_inst)
            end
        end
    end
end

"""
    convert_typepunned_geps_to_byte_geps!(mod::LLVM.Module)

Convert typed GEPs on array allocas where the GEP source element type differs from
the alloca element type into equivalent byte-offset GEPs.

Pattern (MVector{16,UInt32} stored as [8 x i64], accessed via i32-typed GEPs):
    %alloca = alloca [8 x i64]
    %10 = getelementptr i32, ptr %alloca, i64 %dynamic
    %gep = getelementptr i32, ptr %10, i64 -1
    %val = load i32, ptr %gep

After conversion:
    %10_byte = getelementptr i8, ptr %alloca, i64 (%dynamic * 4)
    %gep_byte = getelementptr i8, ptr %10_byte, i64 -4
    %val = load i32, ptr %gep_byte

The resulting byte-offset GEPs are then handled by `lower_byte_gep_chain_on_allocas!`
which applies the correct shift/mask extraction for sub-element access.
"""
function convert_typepunned_geps_to_byte_geps!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue

                    # Only single-index GEPs
                    ops = LLVM.operands(inst)
                    length(ops) == 2 || continue

                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    # Skip if already a byte GEP
                    src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8 && continue

                    src_size = llvm_type_size(src_ty)
                    src_size > 0 || continue

                    # Trace back through GEP chain to find the alloca
                    base = ops[1]
                    while base isa LLVM.GetElementPtrInst
                        base = LLVM.operands(base)[1]
                    end
                    base isa LLVM.AllocaInst || continue

                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(base))

                    # Only process composite alloca types where the GEP source type
                    # differs from the alloca's element type (type-punned access)
                    if alloca_ty isa LLVM.ArrayType
                        alloca_elem_ty = LLVM.eltype(alloca_ty)
                        elem_size = llvm_type_size(alloca_elem_ty)
                        elem_size > 0 || continue
                        # Only convert when accessing with smaller type than alloca element
                        src_size < elem_size || continue
                    elseif alloca_ty isa LLVM.StructType
                        # Struct alloca: GEP treats struct as flat array of src_ty.
                        # Convert to byte GEP so downstream passes can decompose properly.
                        alloca_size = llvm_type_size(alloca_ty)
                        alloca_size > 0 || continue
                        src_ty != alloca_ty || continue
                    else
                        continue
                    end

                    # Convert: gep T, ptr %base, i64 %idx → gep i8, ptr %base, i64 (%idx * sizeof(T))
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        idx = ops[2]
                        i64 = LLVM.Int64Type()
                        i8 = LLVM.Int8Type()
                        if LLVM.value_type(idx) != i64
                            idx = LLVM.sext!(builder, idx, i64, "tpgep_sext")
                        end
                        byte_off = if idx isa LLVM.ConstantInt
                            LLVM.ConstantInt(i64, convert(Int64, idx) * src_size)
                        else
                            scale = LLVM.ConstantInt(i64, src_size)
                            LLVM.mul!(builder, idx, scale, "tpgep_mul")
                        end
                        new_gep = LLVM.gep!(builder, i8, ops[1], [byte_off], "tpgep_byte")
                        LLVM.replace_uses!(inst, new_gep)
                        LLVM.erase!(inst)
                    end
                    changed = true
                    break  # restart iteration
                end
                changed && break
            end
        end
    end
end

"""
    lower_byte_gep_chain_on_allocas!(mod::LLVM.Module)

Lower chained byte-offset GEPs on array allocas where the access type (from store/load)
doesn't match the alloca element type.

Pattern (from MArray 1-based indexing, after InstCombine splits the offset):
    %alloca = alloca [16 x i64]
    %gep1 = getelementptr i8, ptr %alloca, i64 %dynamic   ; e.g., idx * 4
    %gep2 = getelementptr i8, ptr %gep1, i64 -4           ; 1-based adjustment
    store i32 %val, ptr %gep2                              ; i32 into i64 alloca!

Also handles the single GEP case:
    %gep = getelementptr i8, ptr %alloca, i64 %dynamic
    store/load i32, ptr %gep

These are lowered to proper element-level access with shift/mask via
`emit_element_rmw_access!`:
    total_byte_off = gep1_offset + gep2_offset
    elem_idx = total_byte_off >> log2(sizeof(alloca_elem))
    inner = total_byte_off & (sizeof(alloca_elem) - 1)
    → GEP [N x i64], ptr %alloca, 0, elem_idx → load/shift/mask/store
"""
function lower_byte_gep_chain_on_allocas!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            to_erase = LLVM.Instruction[]
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    # Look for store/load instructions
                    if inst isa LLVM.StoreInst
                        ptr = LLVM.operands(inst)[2]
                        store_val = LLVM.operands(inst)[1]
                        access_ty = LLVM.value_type(store_val)
                        mode = :store
                    elseif inst isa LLVM.LoadInst
                        ptr = LLVM.operands(inst)[1]
                        access_ty = LLVM.value_type(inst)
                        mode = :load
                        store_val = nothing
                    else
                        continue
                    end

                    # Check if ptr is a byte-offset GEP chain from an alloca
                    ptr isa LLVM.GetElementPtrInst || continue
                    ptr_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(ptr))
                    ptr_src_ty isa LLVM.IntegerType && LLVM.width(ptr_src_ty) == 8 || continue

                    # Collect byte offsets from GEP chain
                    gep_chain = LLVM.GetElementPtrInst[ptr]
                    current = LLVM.operands(ptr)[1]
                    while current isa LLVM.GetElementPtrInst
                        cur_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(current))
                        cur_src isa LLVM.IntegerType && LLVM.width(cur_src) == 8 || break
                        length(LLVM.operands(current)) == 2 || break
                        pushfirst!(gep_chain, current)
                        current = LLVM.operands(current)[1]
                    end

                    # current must be an alloca with an array type
                    current isa LLVM.AllocaInst || continue
                    alloca = current
                    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                    alloca_ty isa LLVM.ArrayType || continue
                    alloca_elem_ty = LLVM.eltype(alloca_ty)
                    alloca_elem_ty isa LLVM.IntegerType || continue

                    # Access type must be smaller than alloca element type
                    access_size = llvm_type_size(access_ty)
                    elem_size = llvm_type_size(alloca_elem_ty)
                    (access_size > 0 && elem_size > 0 && access_size < elem_size) || continue
                    # elem_size must be power of 2 for shift/mask
                    (elem_size & (elem_size - 1)) == 0 || continue

                    # Sum all byte offsets in the chain
                    # Must have at least one dynamic (non-constant) offset to trigger this pass
                    # (constant-only chains are handled by lift_byte_geps_on_allocas!)
                    has_dynamic = false
                    for gep in gep_chain
                        gep_ops = LLVM.operands(gep)
                        if !(gep_ops[2] isa LLVM.ConstantInt)
                            has_dynamic = true
                            break
                        end
                    end
                    has_dynamic || continue

                    # Build total byte offset expression
                    i64 = LLVM.Int64Type()
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)

                        total_off = nothing
                        for gep in gep_chain
                            gep_ops = LLVM.operands(gep)
                            off = gep_ops[2]
                            # Ensure i64
                            if LLVM.value_type(off) != i64
                                off = LLVM.sext!(builder, off, i64, "bgc_sext")
                            end
                            if total_off === nothing
                                total_off = off
                            else
                                total_off = LLVM.add!(builder, total_off, off, "bgc_add")
                            end
                        end

                        emit_element_rmw_access!(builder, alloca, alloca_ty, alloca_elem_ty,
                                                  elem_size, total_off, store_val, inst, mode, to_erase)
                    end
                    changed = true
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
            # Clean up dead GEPs
            if changed
                for bb in LLVM.blocks(fn)
                    dead = LLVM.Instruction[]
                    for inst in LLVM.instructions(bb)
                        inst isa LLVM.GetElementPtrInst || continue
                        isempty(LLVM.uses(inst)) || continue
                        push!(dead, inst)
                    end
                    for inst in dead
                        LLVM.erase!(inst)
                    end
                end
            end
        end
    end
end

"""
    lower_phi_typepunned_loads!(mod::LLVM.Module)

Lower loads of smaller types through PHI chains that originate from byte-offset GEPs
on array allocas with larger element types.

Pattern (MVector{32,UInt32} stored as [16 x i64], accessed via byte-offset GEPs):

    %gep1 = getelementptr i8, ptr %alloca_[16 x i64], i64 %dynamic
    %gep2 = getelementptr i8, ptr %gep1, i64 -4
    ; ... flows through PHI chain (from StructurizeCFG Flow blocks) ...
    %phi = phi ptr [ %gep2, %bb1 ], [ undef, %bb2 ]
    %val = load i32, ptr %phi

The emitter can't handle this: it divides each GEP's offset by elem_size independently
((-4)/8 = 0, losing the offset), and OpBitcast always reads the lower 32 bits.

Fix: "lift" the load to each leaf GEP site using shift/mask extraction, create parallel
i32 PHI chains, and replace the original load with the final i32 PHI value.
"""
function lower_phi_typepunned_loads!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.LoadInst || continue
                    ptr = LLVM.operands(inst)[1]
                    ptr isa LLVM.PHIInst || continue

                    access_ty = LLVM.value_type(inst)
                    access_size = llvm_type_size(access_ty)
                    access_size > 0 || continue

                    # Trace PHI chain to verify all leaves are byte-GEPs on same alloca
                    alloca_info = trace_phi_chain_to_alloca(ptr)
                    alloca_info === nothing && continue
                    alloca, alloca_ty, elem_ty, elem_size = alloca_info

                    access_size < elem_size || continue
                    (elem_size & (elem_size - 1)) == 0 || continue  # power of 2

                    # Create parallel value PHI chain
                    phi_map = Dict{LLVM.PHIInst, LLVM.Value}()
                    new_val = create_value_phi_chain!(
                        ptr, alloca, alloca_ty, elem_ty, elem_size, access_ty, phi_map)
                    new_val === nothing && continue

                    LLVM.replace_uses!(inst, new_val)
                    LLVM.erase!(inst)
                    changed = true
                    break
                end
                changed && break
            end
        end
    end
end

"""
Trace a PHI chain to find whether all non-undef leaf values are byte-offset GEP chains
on the same [N x integer] array alloca. Returns (alloca, alloca_ty, elem_ty, elem_size)
or nothing if not eligible.
"""
function trace_phi_chain_to_alloca(phi::LLVM.PHIInst,
                                     visited::Set{LLVM.Value}=Set{LLVM.Value}())
    phi in visited && return nothing  # cycle
    push!(visited, phi)

    alloca = nothing
    for (val, _) in LLVM.incoming(phi)
        if val isa LLVM.UndefValue
            continue
        elseif val isa LLVM.PHIInst
            sub = trace_phi_chain_to_alloca(val, visited)
            sub === nothing && return nothing
            if alloca === nothing
                alloca = sub[1]
            elseif alloca !== sub[1]
                return nothing  # different allocas
            end
        elseif val isa LLVM.GetElementPtrInst
            gep_alloca = trace_gep_chain_to_array_alloca(val)
            gep_alloca === nothing && return nothing
            if alloca === nothing
                alloca = gep_alloca
            elseif alloca !== gep_alloca
                return nothing
            end
        else
            return nothing  # unknown value type
        end
    end

    alloca === nothing && return nothing

    alloca_ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
    alloca_ty isa LLVM.ArrayType || return nothing
    elem_ty = LLVM.eltype(alloca_ty)
    elem_ty isa LLVM.IntegerType || return nothing
    elem_size = llvm_type_size(elem_ty)
    elem_size > 0 || return nothing

    return (alloca, alloca_ty, elem_ty, elem_size)
end

"""
Trace a byte-offset GEP chain (all i8 source type) back to an alloca.
Returns the alloca or nothing.
"""
function trace_gep_chain_to_array_alloca(val::LLVM.Value)
    current = val
    while current isa LLVM.GetElementPtrInst
        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(current))
        src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8 || return nothing
        length(LLVM.operands(current)) == 2 || return nothing
        current = LLVM.operands(current)[1]
    end
    current isa LLVM.AllocaInst || return nothing
    return current
end

"""
Recursively create a parallel value-typed PHI chain for a pointer PHI chain.
At each leaf GEP, inserts shift/mask extraction code. Returns the new value
(a PHI or extracted i32) or nothing on failure.
"""
function create_value_phi_chain!(phi::LLVM.PHIInst, alloca, alloca_ty,
                                   elem_ty, elem_size, access_ty,
                                   phi_map::Dict{LLVM.PHIInst, LLVM.Value})
    haskey(phi_map, phi) && return phi_map[phi]

    phi_bb = LLVM.parent(phi)
    i64 = LLVM.Int64Type()

    # Create new PHI of access_ty positioned after existing PHIs
    first_non_phi = nothing
    for inst in LLVM.instructions(phi_bb)
        if !(inst isa LLVM.PHIInst)
            first_non_phi = inst
            break
        end
    end

    new_phi = LLVM.@dispose builder=LLVM.IRBuilder() begin
        if first_non_phi !== nothing
            LLVM.position!(builder, first_non_phi)
        else
            LLVM.position!(builder, phi_bb)
        end
        LLVM.phi!(builder, access_ty, "phi_extract")
    end
    phi_map[phi] = new_phi  # register before recursion (handles cycles)

    # Process incoming values
    for (val, bb) in LLVM.incoming(phi)
        if val isa LLVM.UndefValue
            push!(LLVM.incoming(new_phi), (LLVM.UndefValue(access_ty), bb))
        elseif val isa LLVM.PHIInst
            sub_val = create_value_phi_chain!(val, alloca, alloca_ty,
                                                elem_ty, elem_size, access_ty, phi_map)
            sub_val === nothing && return nothing
            push!(LLVM.incoming(new_phi), (sub_val, bb))
        elseif val isa LLVM.GetElementPtrInst
            extracted = emit_gep_chain_extract!(val, alloca, alloca_ty,
                                                  elem_ty, elem_size, access_ty)
            extracted === nothing && return nothing
            push!(LLVM.incoming(new_phi), (extracted, bb))
        else
            return nothing
        end
    end

    return new_phi
end

"""
At a leaf byte-offset GEP site, emit shift/mask extraction code to read a smaller
type from a larger array element. Inserts code before the block's terminator.
Returns the extracted value (e.g., i32 from i64 element).
"""
function emit_gep_chain_extract!(gep_end::LLVM.GetElementPtrInst, alloca, alloca_ty,
                                    elem_ty, elem_size, access_ty)
    i64 = LLVM.Int64Type()

    # Collect byte offsets from GEP chain
    gep_chain = LLVM.GetElementPtrInst[gep_end]
    current = LLVM.operands(gep_end)[1]
    while current isa LLVM.GetElementPtrInst
        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(current))
        src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8 || break
        length(LLVM.operands(current)) == 2 || break
        pushfirst!(gep_chain, current)
        current = LLVM.operands(current)[1]
    end
    current === alloca || return nothing

    # Insert extraction before the terminator of the GEP's block
    gep_bb = LLVM.parent(gep_end)
    term = LLVM.terminator(gep_bb)

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, term)

        # Sum byte offsets from the GEP chain
        total_off = nothing
        for gep in gep_chain
            gep_ops = LLVM.operands(gep)
            off = gep_ops[2]
            if LLVM.value_type(off) != i64
                off = LLVM.sext!(builder, off, i64, "phi_bgc_sext")
            end
            if total_off === nothing
                total_off = off
            else
                total_off = LLVM.add!(builder, total_off, off, "phi_bgc_add")
            end
        end

        # elem_idx = total_off >> log2(elem_size)
        elem_shift = Int(log2(elem_size))
        elem_idx = LLVM.ashr!(builder, total_off, LLVM.ConstantInt(i64, elem_shift), "phi_eidx")

        # inner = total_off & (elem_size - 1)
        inner = LLVM.and!(builder, total_off, LLVM.ConstantInt(i64, elem_size - 1), "phi_inner")

        # shift_bits = inner * 8
        shift_bits = LLVM.shl!(builder, inner, LLVM.ConstantInt(i64, 3), "phi_shift")

        # GEP to the array element
        typed_gep = LLVM.gep!(builder, alloca_ty, alloca,
                              [LLVM.ConstantInt(i64, 0), elem_idx], "phi_gep")

        # Load full element
        full = LLVM.load!(builder, elem_ty, typed_gep, "phi_full")

        # Shift right by sub-element bit offset
        shift_in_elem = LLVM.trunc!(builder, shift_bits, elem_ty, "phi_shift_t")
        shifted = LLVM.lshr!(builder, full, shift_in_elem, "phi_shr")

        # Truncate to target type
        result = LLVM.trunc!(builder, shifted, access_ty, "phi_trunc")

        return result
    end
end

"""
    flatten_chained_geps_on_allocas!(mod::LLVM.Module)

Flatten chained GEPs where a typed GEP uses a byte-offset GEP from an alloca as its base.

Pattern:
    %base = gep i8, ptr %alloca, i64 <const_offset>
    %result = gep <T>, ptr %base, i64 %idx    ; T is NOT i8
    → replace with:
    %result = gep i8, ptr %alloca, i64 (<const_offset> + %idx * sizeof(T))

This handles Julia's 1-based MArray indexing pattern where the base is shifted by
-sizeof(element) to make index 1 point to element 0. After this pass, the
`lift_byte_geps_on_allocas!` pass can properly convert the byte-offset GEPs.
"""
function flatten_chained_geps_on_allocas!(mod::LLVM.Module)
    dl = LLVM.datalayout(mod)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            to_replace = []

            for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue

                ops = LLVM.operands(inst)
                base_ptr = ops[1]
                base_ptr isa LLVM.GetElementPtrInst || continue

                gep_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                # Skip if this GEP is already a byte-offset GEP
                gep_src_ty isa LLVM.IntegerType && LLVM.width(gep_src_ty) == 8 && continue

                # Only handle single-index GEPs (the common case: gep T, ptr, idx)
                length(ops) - 1 == 1 || continue

                # Check if base is a byte-offset GEP with constant offset from an alloca
                base_ops = LLVM.operands(base_ptr)
                base_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base_ptr))
                base_src_ty isa LLVM.IntegerType && LLVM.width(base_src_ty) == 8 || continue
                length(base_ops) - 1 == 1 || continue
                base_ops[2] isa LLVM.ConstantInt || continue

                # Verify base traces to an alloca
                alloca = base_ops[1]
                alloca isa LLVM.AllocaInst || continue

                base_offset = convert(Int64, base_ops[2])
                elem_size = Int64(LLVM.API.LLVMABISizeOfType(dl, gep_src_ty))
                idx = ops[2]

                push!(to_replace, (inst, alloca, base_offset, elem_size, idx))
            end

            for (gep_inst, alloca, base_offset, elem_size, idx) in to_replace
                LLVM.@dispose builder=LLVM.IRBuilder() begin
                    LLVM.position!(builder, gep_inst)
                    i8_ty = LLVM.Int8Type()
                    i64_ty = LLVM.Int64Type()

                    # Compute: base_offset + idx * elem_size
                    # Ensure idx is i64
                    idx_val = if LLVM.value_type(idx) != i64_ty
                        LLVM.sext!(builder, idx, i64_ty)
                    else
                        idx
                    end

                    if elem_size == 1
                        byte_idx = idx_val
                    else
                        size_val = LLVM.ConstantInt(i64_ty, elem_size)
                        byte_idx = LLVM.mul!(builder, idx_val, size_val)
                    end

                    if base_offset != 0
                        base_val = LLVM.ConstantInt(i64_ty, base_offset)
                        final_offset = LLVM.add!(builder, base_val, byte_idx)
                    else
                        final_offset = byte_idx
                    end

                    new_gep = LLVM.inbounds_gep!(builder, i8_ty, alloca, [final_offset])
                    LLVM.replace_uses!(gep_inst, new_gep)
                    LLVM.erase!(gep_inst)
                    changed = true
                end
            end
        end

        # Clean up dead byte-offset GEPs that are no longer used
        for bb in LLVM.blocks(fn)
            to_erase = LLVM.Instruction[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue
                isempty(LLVM.uses(inst)) || continue
                push!(to_erase, inst)
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end
end

"""
Compute the byte offset of a constant-index GEP given its source element type.
Returns the offset as Int64, or nothing if computation fails.
"""
function compute_gep_byte_offset(src_ty::LLVM.LLVMType, operands, dl::LLVM.DataLayout)
    offset = Int64(0)
    n_indices = length(operands) - 1
    n_indices == 0 && return offset

    # First index: scales by sizeof(src_ty)
    idx0 = convert(Int64, operands[2])
    type_size = Int64(LLVM.API.LLVMABISizeOfType(dl, src_ty))
    offset += idx0 * type_size

    # Remaining indices: drill into the type
    current_ty = src_ty
    for i in 3:length(operands)
        idx = convert(Int64, operands[i])
        if current_ty isa LLVM.StructType
            # Struct: use LLVM's struct layout for correct padding/alignment
            offset += Int64(LLVM.API.LLVMOffsetOfElement(dl, current_ty, UInt32(idx)))
            current_ty = LLVM.elements(current_ty)[idx+1]
        elseif current_ty isa LLVM.ArrayType
            # Array: offset = idx * element_size
            elem_ty = eltype(current_ty)
            elem_size = Int64(LLVM.API.LLVMABISizeOfType(dl, elem_ty))
            offset += idx * elem_size
            current_ty = elem_ty
        else
            # Scalar or unknown type — can't drill further
            return nothing
        end
    end
    return offset
end
