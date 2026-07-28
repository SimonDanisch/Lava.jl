# LLVM pass: Lift byte-offset GEPs on struct/array allocas to typed GEPs.
#
# Ported from Abacus compilation.jl (lift_byte_geps_on_allocas!, byte_offset_to_gep_indices!).
# This is the most complex LLVM pass in the pipeline (~300+ lines).
#
# Problem:
#   After SROA partially decomposes struct allocas, it rewrites accesses as:
#       %p = getelementptr i8, ptr %alloca, i64 <byte_offset>
#   The SPIR-V emitter (and llc) needs typed GEPs to recover struct member info
#   for OpAccessChain generation. Byte-offset GEPs have no type information.
#
# Solution:
#   Convert to typed GEPs using the alloca's known type:
#       %p = getelementptr %struct_type, ptr %alloca, 0, <member_path...>
#
# Four sub-patterns are handled:
#   1. Byte-offset GEPs on struct allocas (main pattern)
#   2. Element-typed flat GEPs on array allocas (gep float, ptr [3xfloat], idx)
#   3. Direct store/load to array alloca (missing gep [3xfloat], ptr, 0, 0)
#   4. Flat GEP chains on alloca-derived typed GEPs (merge chain into single GEP)
#
# This pass runs TWICE in the pipeline:
#   - After initial SROA+InstCombine (to recover types before structurization)
#   - After StructurizeCFG+InstCombine (to fix any new byte-GEPs from reg2mem)

"""
    compute_zero_index_path(ty::LLVM.LLVMType) -> Vector{Int} or nothing

Navigate through nested struct/array types following index 0 at each level until
reaching a scalar (non-composite) type. Returns the index path (excluding the
leading 0 for the base pointer).

Example: for `{ { [1 x [1 x [1 x i64]]], ... } }` returns [0, 0, 0, 0, 0]
"""
function compute_zero_index_path(ty::LLVM.LLVMType)
    path = Int[]
    current = ty
    while current isa LLVM.StructType || current isa LLVM.ArrayType
        push!(path, 0)
        if current isa LLVM.StructType
            elems = LLVM.elements(current)
            isempty(elems) && return nothing
            current = first(elems)
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return nothing
            current = LLVM.eltype(current)
        end
    end
    return path
end

"""
    byte_offset_to_gep_indices(type, offset, dl, stop_at_aggregate) -> Vector{Int} or nothing

Map a byte offset within a struct/array type to a sequence of GEP indices.

If `stop_at_aggregate` is true, stop navigating when the remaining offset is 0
and the current type is an aggregate (struct/array). This is needed when the GEP
result is used as the base of another typed GEP (e.g., variable-indexed array access).

If `stop_at_aggregate` is false, navigate all the way to the scalar leaf field.
"""
function byte_offset_to_gep_indices(type::LLVM.LLVMType, offset::Int,
                                      dl::LLVM.DataLayout, stop_at_aggregate::Bool)
    indices = Int[]
    resolve_offset!(indices, type, offset, dl, stop_at_aggregate) && return indices
    return nothing
end

function resolve_offset!(indices::Vector{Int}, type::LLVM.LLVMType, offset::Int,
                           dl::LLVM.DataLayout, stop_at_aggregate::Bool)
    # At aggregate boundary with offset=0: stop if we want aggregate-level navigation
    if offset == 0 && stop_at_aggregate
        return true
    end
    # At scalar leaf with offset=0: always stop
    if offset == 0 && !(type isa LLVM.StructType) && !(type isa LLVM.ArrayType)
        return true
    end

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_size = Int(LLVM.storage_size(dl, member_type))
            if offset >= member_offset && offset < member_offset + member_size
                push!(indices, i)
                return resolve_offset!(indices, member_type,
                                        offset - member_offset, dl, stop_at_aggregate)
            end
        end
        return false  # offset lands in padding

    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        elem_size == 0 && return false
        n = LLVM.length(type)
        idx = div(offset, elem_size)
        idx < n || return false
        push!(indices, idx)
        return resolve_offset!(indices, elem_type, offset - idx * elem_size, dl,
                                stop_at_aggregate)

    else
        return offset == 0
    end
end

"""Position IRBuilder right after `inst`, handling the case where inst is in
the entry block (allocas). This ensures GEPs placed here dominate all uses."""
function position_after!(builder::LLVM.IRBuilder, inst::LLVM.Instruction)
    next = LLVM.API.LLVMGetNextInstruction(inst)
    if next != C_NULL
        LLVM.position!(builder, LLVM.Instruction(next))
    else
        # inst is last in block — position at end
        bb = LLVM.parent(inst)
        LLVM.position!(builder, bb)
    end
end

"""
    lift_byte_geps_on_allocas!(mod::LLVM.Module)

Convert byte-offset GEPs on struct/array allocas to properly typed struct GEPs.

After SROA partially decomposes struct allocas, it rewrites accesses as:
    %p = getelementptr i8, ptr %alloca, <byte_offset>
This pass converts them to:
    %p = getelementptr %alloca_type, ptr %alloca, 0, <member_path...>
giving the SPIR-V emitter the type information it needs for OpAccessChain.

Handles four sub-patterns:
1. Byte-offset GEPs on struct allocas (the main pattern)
2. Element-typed flat GEPs on array allocas
3. Direct store/load of element type to array alloca
4. Flat GEP chains on alloca-derived typed GEPs (merged iteratively)
"""
function lift_byte_geps_on_allocas!(mod::LLVM.Module)
    dl = LLVM.datalayout(mod)

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Collect all allocas with aggregate types (struct or array)
        allocas = Pair{LLVM.Instruction, LLVM.LLVMType}[]
        for inst in LLVM.instructions(first(LLVM.blocks(f)))
            inst isa LLVM.AllocaInst || continue
            at = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
            (at isa LLVM.StructType || at isa LLVM.ArrayType) || continue
            push!(allocas, inst => at)
        end
        isempty(allocas) && continue

        to_erase = LLVM.Instruction[]

        for (alloca_inst, alloca_type) in allocas
            # ------------------------------------------------------------------
            # Pattern 1: Byte-offset GEPs that use this alloca as base.
            #   %p = getelementptr [inbounds] i8, ptr %alloca, i64 <const>
            # ------------------------------------------------------------------
            for use in LLVM.uses(alloca_inst)
                user = LLVM.user(use)
                user isa LLVM.GetElementPtrInst || continue

                # Check if this is a byte-offset GEP (source element type = i8)
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                src_ty == LLVM.Int8Type() || continue

                # Get the byte offset (must be a constant)
                ops = LLVM.operands(user)
                length(ops) == 2 || continue  # expecting: ptr, offset
                offset_val = ops[2]
                offset_val isa LLVM.ConstantInt || continue
                byte_offset = convert(Int, offset_val)

                # Classify users: typed-GEP users need aggregate depth,
                # load/store users need leaf depth.
                has_typed_gep_users = false
                has_leaf_users = false
                for u in LLVM.uses(user)
                    usr = LLVM.user(u)
                    if usr isa LLVM.GetElementPtrInst
                        u_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(usr))
                        if u_src != LLVM.Int8Type()
                            has_typed_gep_users = true
                        else
                            has_leaf_users = true
                        end
                    else
                        has_leaf_users = true
                    end
                end

                is_inbounds = LLVM.API.LLVMIsInBounds(user)

                # Helper to build a typed GEP from index path
                function _build_typed_gep(builder, indices, name)
                    idx_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                    for idx in indices
                        push!(idx_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                    end
                    gep = LLVM.gep!(builder, alloca_type, alloca_inst, idx_vals, name)
                    LLVM.API.LLVMSetIsInBounds(gep, is_inbounds)
                    return gep
                end

                if has_typed_gep_users && has_leaf_users
                    # DUAL-USE: the same byte-offset GEP is used by both:
                    # 1. Typed GEPs (e.g., variable-indexed array access)
                    # 2. Loads/stores (need scalar leaf pointer)
                    #
                    # For typed-GEP users: MERGE the aggregate path + user's
                    # indices into a single GEP from the alloca. This avoids
                    # an intermediate aggregate pointer that the SPIR-V emitter
                    # can't type-match with the downstream GEP's source type.
                    agg_indices = byte_offset_to_gep_indices(alloca_type, byte_offset, dl, true)
                    leaf_indices = byte_offset_to_gep_indices(alloca_type, byte_offset, dl, false)
                    (agg_indices === nothing || leaf_indices === nothing) && continue

                    # Collect typed-GEP users FIRST -- modifying operands
                    # invalidates the use iterator.
                    typed_gep_users = LLVM.Instruction[]
                    for u in LLVM.uses(user)
                        usr = LLVM.user(u)
                        if usr isa LLVM.GetElementPtrInst
                            u_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(usr))
                            if u_src != LLVM.Int8Type()
                                push!(typed_gep_users, usr)
                            end
                        end
                    end

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        # For each typed-GEP user, create a merged GEP that
                        # goes directly from the alloca through the aggregate
                        # path and then the user's own indices.
                        for usr in typed_gep_users
                            LLVM.position!(builder, usr)
                            usr_ops = LLVM.operands(usr)
                            # Build merged index path: [0, agg_path..., user_indices...]
                            merged_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                            for idx in agg_indices
                                push!(merged_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                            end
                            # Append the user GEP's indices (skip operand 0 = base pointer)
                            for oi in 2:length(usr_ops)
                                push!(merged_vals, usr_ops[oi])
                            end
                            merged_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                                    merged_vals, "typed_gep_merged")
                            LLVM.API.LLVMSetIsInBounds(merged_gep, is_inbounds)
                            LLVM.replace_uses!(usr, merged_gep)
                            push!(to_erase, usr)
                        end

                        # Redirect all remaining uses (loads/stores) to leaf GEP
                        LLVM.position!(builder, user)
                        leaf_gep = _build_typed_gep(builder, leaf_indices, "typed_gep_leaf")
                        LLVM.replace_uses!(user, leaf_gep)
                    end
                else
                    # Single-use case: pick the appropriate depth
                    stop_at_agg = has_typed_gep_users
                    indices = byte_offset_to_gep_indices(alloca_type, byte_offset, dl,
                                                           stop_at_agg)
                    indices === nothing && continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, user)
                        new_gep = _build_typed_gep(builder, indices, "typed_gep")
                        LLVM.replace_uses!(user, new_gep)
                    end
                end
                push!(to_erase, user)
            end

            # ------------------------------------------------------------------
            # Pattern 2: Element-typed flat GEPs on array allocas.
            #   LLVM generates: gep float, ptr %[3xfloat]_alloca, i64 %idx
            #   SPIR-V needs:   gep [3 x float], ptr %alloca, i64 0, %idx
            # ------------------------------------------------------------------
            if alloca_type isa LLVM.ArrayType
                elem_type = LLVM.eltype(alloca_type)
                for use in LLVM.uses(alloca_inst)
                    user = LLVM.user(use)
                    user isa LLVM.GetElementPtrInst || continue
                    user in to_erase && continue

                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                    # Match: source type is the array's element type (not i8, not the array type)
                    src_ty == elem_type || continue
                    src_ty == alloca_type && continue  # already correct

                    ops = LLVM.operands(user)
                    length(ops) == 2 || continue  # gep elem, ptr, idx

                    is_inbounds = LLVM.API.LLVMIsInBounds(user)
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, user)
                        idx_vals = LLVM.Value[
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                            ops[2]  # the original index (may be variable)
                        ]
                        new_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                            idx_vals, "typed_arr_gep")
                        LLVM.API.LLVMSetIsInBounds(new_gep, is_inbounds)
                        LLVM.replace_uses!(user, new_gep)
                    end
                    push!(to_erase, user)
                end

                # ------------------------------------------------------------------
                # Pattern 3: Direct store/load of element type to array alloca.
                #   LLVM optimizes `gep i8, ptr %alloca, 0` away, leaving:
                #       store float %val, ptr %byval_arg  (where alloca is [3 x float])
                #   Fix: insert gep [3 x float], ptr %alloca, 0, 0 before each such use.
                # ------------------------------------------------------------------
                direct_users = LLVM.Instruction[]
                for use in LLVM.uses(alloca_inst)
                    usr = LLVM.user(use)
                    usr in to_erase && continue
                    if usr isa LLVM.StoreInst || usr isa LLVM.LoadInst
                        push!(direct_users, usr)
                    end
                end
                if !isempty(direct_users)
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        # Insert GEP right after alloca to dominate all uses
                        position_after!(builder, alloca_inst)
                        idx_vals = LLVM.Value[
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                        ]
                        elem0_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                               idx_vals, "arr_elem0")
                        # Redirect all direct load/store uses from alloca to elem0_gep
                        for usr in direct_users
                            if usr isa LLVM.StoreInst
                                # store val, ptr %alloca -> store val, ptr %elem0_gep
                                # The pointer is operand 2 (value=op1, ptr=op2)
                                LLVM.API.LLVMSetOperand(usr, 1, elem0_gep)
                            elseif usr isa LLVM.LoadInst
                                LLVM.API.LLVMSetOperand(usr, 0, elem0_gep)
                            end
                        end
                    end
                end
            end

            # ------------------------------------------------------------------
            # Pattern 3b: Direct load/store of scalar type to composite alloca
            #   (generalized — handles struct/array allocas, not just arrays).
            #   LLVM collapses GEPs for offset-0 accesses, leaving:
            #       load i64, ptr %alloca  (where alloca is { { [1x[1x[1xi64]]], ... } })
            #   Fix: insert gep chain navigating to the first scalar field.
            # ------------------------------------------------------------------
            if alloca_type isa LLVM.StructType || alloca_type isa LLVM.ArrayType
                mismatched_users = LLVM.Instruction[]
                for use in LLVM.uses(alloca_inst)
                    usr = LLVM.user(use)
                    usr in to_erase && continue
                    if usr isa LLVM.LoadInst
                        loaded_ty = LLVM.value_type(usr)
                        if loaded_ty != alloca_type
                            push!(mismatched_users, usr)
                        end
                    elseif usr isa LLVM.StoreInst
                        stored_val = LLVM.operands(usr)[1]
                        stored_ty = LLVM.value_type(stored_val)
                        # Only fix if store is TO the alloca (ptr is operand 2)
                        if stored_ty != alloca_type && LLVM.operands(usr)[2] === alloca_inst
                            push!(mismatched_users, usr)
                        end
                    end
                end

                if !isempty(mismatched_users)
                    # Compute zero-index path from alloca_type to scalar leaf
                    leaf_path = compute_zero_index_path(alloca_type)
                    if leaf_path !== nothing
                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            # Place GEP right after the alloca so it dominates ALL uses
                            # (users may be in different basic blocks)
                            position_after!(builder, alloca_inst)
                            idx_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                            for idx in leaf_path
                                push!(idx_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                            end
                            scalar_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                                     idx_vals, "scalar_field")
                            for usr in mismatched_users
                                if usr isa LLVM.StoreInst
                                    LLVM.API.LLVMSetOperand(usr, 1, scalar_gep)
                                elseif usr isa LLVM.LoadInst
                                    LLVM.API.LLVMSetOperand(usr, 0, scalar_gep)
                                end
                            end
                        end
                    end
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end

        # ------------------------------------------------------------------
        # Pattern 5: Type-mismatched loads/stores on GEP results.
        #   A typed GEP may land on a composite sub-type, but the load/store
        #   uses a scalar type. Insert additional zero-index GEPs to drill down.
        #   Example:
        #     %p = gep %struct, ptr %alloca, 0, 1  -> points to { [1x[1xi64]] }
        #     load i64, ptr %p                      -> needs gep to i64 leaf
        # ------------------------------------------------------------------
        for (alloca_inst, alloca_type) in allocas
            for use in LLVM.uses(alloca_inst)
                gep_user = LLVM.user(use)
                gep_user isa LLVM.GetElementPtrInst || continue

                # Compute what type this GEP result points to
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep_user))
                result_ty = compute_gep_result_type(src_ty, gep_user)
                result_ty === nothing && continue
                (result_ty isa LLVM.StructType || result_ty isa LLVM.ArrayType) || continue

                # Check for mismatched load/store users
                mismatched = LLVM.Instruction[]
                for u in LLVM.uses(gep_user)
                    usr = LLVM.user(u)
                    if usr isa LLVM.LoadInst
                        loaded_ty = LLVM.value_type(usr)
                        loaded_ty != result_ty && push!(mismatched, usr)
                    elseif usr isa LLVM.StoreInst
                        stored_val = LLVM.operands(usr)[1]
                        stored_ty = LLVM.value_type(stored_val)
                        if stored_ty != result_ty && LLVM.operands(usr)[2] === gep_user
                            push!(mismatched, usr)
                        end
                    end
                end
                isempty(mismatched) && continue

                leaf_path = compute_zero_index_path(result_ty)
                leaf_path === nothing && continue

                LLVM.@dispose builder=LLVM.IRBuilder() begin
                    # Place GEP right after the base GEP to dominate all users
                    position_after!(builder, gep_user)
                    idx_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                    for idx in leaf_path
                        push!(idx_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                    end
                    scalar_gep = LLVM.gep!(builder, result_ty, gep_user,
                                             idx_vals, "gep_scalar_field")
                    for usr in mismatched
                        if usr isa LLVM.StoreInst
                            LLVM.API.LLVMSetOperand(usr, 1, scalar_gep)
                        elseif usr isa LLVM.LoadInst
                            LLVM.API.LLVMSetOperand(usr, 0, scalar_gep)
                        end
                    end
                end
            end
        end

        # ------------------------------------------------------------------
        # Pattern 4: Flat GEP chains on alloca-derived typed GEPs.
        #   %a = gep [N x T], ptr %alloca, <indices...>, %array_idx
        #   %b = gep T, ptr %a, %offset, <member_indices...>
        # Merge into:
        #   gep [N x T], ptr %alloca, <indices...>, (%array_idx + %offset), <member_indices...>
        #
        # The last index of the base GEP must index into an ArrayType.
        # Runs iteratively until fixpoint (fixing one chain may expose another).
        # ------------------------------------------------------------------
        changed = true
        while changed
            changed = false
            alloca_set = Set{LLVM.Instruction}(first(p) for p in allocas)
            for bb in LLVM.blocks(f)
                chain_to_fix = Tuple{LLVM.GetElementPtrInst, LLVM.GetElementPtrInst}[]
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) >= 3 || continue  # need: ptr, first_idx, member_idx...
                    base = ops[1]
                    base isa LLVM.GetElementPtrInst || continue
                    # Check that the base GEP targets one of our allocas
                    base_ops = LLVM.operands(base)
                    base_ptr = base_ops[1]
                    base_ptr in alloca_set || continue
                    # Walk base GEP indices through the type hierarchy to find
                    # what type the LAST index indexes into. Must be an ArrayType.
                    base_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base))
                    n_base_idx = length(base_ops) - 1  # number of index operands
                    n_base_idx >= 2 || continue  # need at least ptr-level + one real index
                    # Walk indices (skip first = ptr-level index)
                    cur_type = base_src
                    last_indexed_type = nothing
                    valid = true
                    for idx_i in 2:n_base_idx
                        last_indexed_type = cur_type
                        if cur_type isa LLVM.StructType
                            idx_op = base_ops[idx_i + 1]  # +1 because ops[1] is ptr
                            idx_op isa LLVM.ConstantInt || (valid = false; break)
                            member = convert(Int, idx_op)
                            cur_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(cur_type, member))
                        elseif cur_type isa LLVM.ArrayType
                            cur_type = LLVM.eltype(cur_type)
                        else
                            valid = false; break
                        end
                    end
                    valid || continue
                    # The last level indexed must be an array
                    last_indexed_type isa LLVM.ArrayType || continue
                    # The chain GEP's source type should match the array element type
                    chain_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    chain_src == cur_type || continue
                    push!(chain_to_fix, (inst, base))
                end

                if !isempty(chain_to_fix)
                    changed = true
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        for (gep, base_gep) in chain_to_fix
                            LLVM.position!(builder, gep)
                            gep_ops = LLVM.operands(gep)
                            base_ops = LLVM.operands(base_gep)
                            # base_gep indices: [..., %last_idx]
                            # gep indices: [%offset, member_indices...]
                            # merged: [..., %last_idx + %offset, member_indices...]
                            base_last_idx = base_ops[length(base_ops)]
                            chain_first_idx = gep_ops[2]  # the pointer arithmetic offset
                            # Type-match for add
                            idx_ty = LLVM.value_type(base_last_idx)
                            off_ty = LLVM.value_type(chain_first_idx)
                            if off_ty != idx_ty
                                chain_first_idx = LLVM.sext!(builder, chain_first_idx, idx_ty, "chain_off_ext")
                            end
                            adj_idx = LLVM.add!(builder, base_last_idx, chain_first_idx, "arr_chain_adj")
                            # Build new GEP from alloca with merged indices
                            src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base_gep))
                            new_indices = LLVM.Value[]
                            for i in 2:(length(base_ops)-1)
                                push!(new_indices, base_ops[i])
                            end
                            push!(new_indices, adj_idx)
                            for i in 3:length(gep_ops)  # skip ptr and first_idx
                                push!(new_indices, gep_ops[i])
                            end
                            new_gep = LLVM.gep!(builder, src_ty, base_ops[1], new_indices,
                                               "alloca_chain_fix")
                            is_ib = LLVM.API.LLVMIsInBounds(gep) | LLVM.API.LLVMIsInBounds(base_gep)
                            LLVM.API.LLVMSetIsInBounds(new_gep, is_ib)
                            LLVM.replace_uses!(gep, new_gep)
                            LLVM.erase!(gep)
                        end
                    end
                end
            end
        end
    end
end

# ── Combine chained single-index GEPs ──
#
# Pattern: gep T, (gep T, p, i), j → gep T, p, add(i, j)
#
# LLVM's optimizer sometimes leaves two consecutive GEPs with the same source
# type and single index. In SPIR-V, each becomes an OpPtrAccessChain. Some
# drivers (AMD RADV) produce wrong results with chained OpPtrAccessChain on
# PhysicalStorageBuffer struct pointers. Combining them into a single GEP
# produces a single OpPtrAccessChain that works correctly.

function combine_chained_geps!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        changed = true
        while changed
            changed = false
            for bb in LLVM.blocks(fn)
                # Snapshot the block: a combine erases the current GEP and its
                # `base` (a dominating def — always already visited), so iterating
                # a snapshot stays valid while we fold EVERY combinable GEP in one
                # sweep. The old code `break`d after the first fold per block and
                # let `while changed` rescan the whole function, which is O(n²) on
                # a large shader (folds one GEP per full-function pass).
                for inst in collect(LLVM.instructions(bb))
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) != 2 && continue  # Single-index outer GEP only

                    base = ops[1]
                    base isa LLVM.GetElementPtrInst || continue
                    base_ops = LLVM.operands(base)

                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    base_src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base))
                    idx_outer = ops[2]

                    # The base GEP must only be used by this GEP (safe to eliminate)
                    n_uses = 0
                    for _ in LLVM.uses(base)
                        n_uses += 1
                    end
                    n_uses == 1 || continue

                    if length(base_ops) == 2
                        # Case 1: both are single-index GEPs with same source type
                        src_ty == base_src_ty || continue
                        idx_inner = base_ops[2]
                        LLVM.value_type(idx_outer) == LLVM.value_type(idx_inner) || continue

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            combined_idx = LLVM.add!(builder, idx_inner, idx_outer, "gep_combined_idx")
                            new_gep = LLVM.gep!(builder, src_ty, base_ops[1], [combined_idx], "gep_combined")
                            LLVM.replace_uses!(inst, new_gep)
                            LLVM.erase!(inst)
                            LLVM.erase!(base)
                        end
                        changed = true
                    else
                        # Case 2: outer is single-index, base is multi-index.
                        # Pattern: gep T, (gep Struct, ptr, 0, ..., i64 %arr_idx), i64 %offset
                        # The base GEP's last index accesses an array element of type T.
                        # Fold by adding offset to the last index.
                        last_idx = base_ops[end]
                        # Last index must be dynamic (not a constant struct member index)
                        # and same integer type as the outer index
                        LLVM.value_type(idx_outer) == LLVM.value_type(last_idx) || continue
                        # Verify the base GEP's last index accesses an array of src_ty
                        base_last_index_is_array_of(base, src_ty) || continue

                        LLVM.@dispose builder=LLVM.IRBuilder() begin
                            LLVM.position!(builder, inst)
                            combined_idx = LLVM.add!(builder, last_idx, idx_outer, "gep_combined_idx")
                            # Rebuild base GEP with modified last index
                            indices = LLVM.Value[base_ops[i] for i in 2:length(base_ops)-1]
                            push!(indices, combined_idx)
                            new_gep = LLVM.gep!(builder, base_src_ty, base_ops[1], indices, "gep_combined")
                            LLVM.API.LLVMSetIsInBounds(new_gep, LLVM.API.LLVMIsInBounds(base))
                            LLVM.replace_uses!(inst, new_gep)
                            LLVM.erase!(inst)
                            LLVM.erase!(base)
                        end
                        changed = true
                    end
                end
            end
        end
    end
end

"""
Check if a multi-index GEP's last index accesses an array whose element type matches `elem_ty`.
E.g., `gep { { ptr, [2 x i64] } }, ptr %p, i64 0, i32 0, i32 1, i64 %idx`
The last index `%idx` indexes into `[2 x i64]`, so this returns true if `elem_ty == i64`.
"""
function base_last_index_is_array_of(gep::LLVM.GetElementPtrInst, elem_ty::LLVM.LLVMType)
    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
    ops = LLVM.operands(gep)
    n_indices = length(ops) - 1
    n_indices < 2 && return false

    # Walk through the type hierarchy following the indices (except first and last).
    # ops[2] is the first GEP index (pointer array offset) — doesn't change the type.
    # ops[3..end-1] are struct/array drilling indices.
    # ops[end] is the last index (the one we want to fold).
    current_ty = src_ty
    for i in 3:(length(ops) - 1)  # Skip base ptr (ops[1]), first idx (ops[2]), and last idx
        if current_ty isa LLVM.StructType
            idx = ops[i]
            idx isa LLVM.ConstantInt || return false
            member_idx = convert(Int, idx)
            elems = collect(LLVM.elements(current_ty))
            member_idx < length(elems) || return false
            current_ty = elems[member_idx + 1]
        elseif current_ty isa LLVM.ArrayType
            current_ty = LLVM.eltype(current_ty)
        else
            return false
        end
    end

    # current_ty should now be an array type whose element matches elem_ty
    return current_ty isa LLVM.ArrayType && LLVM.eltype(current_ty) == elem_ty
end
