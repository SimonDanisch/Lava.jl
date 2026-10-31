# Keep numeric private arguments in their caller's layout across OpFunctionCall.
# Byte views are normalized on local copies; Function pointers cannot be cast.

function private_numeric_layout(ty::LLVM.LLVMType)
    ty isa Union{LLVM.IntegerType, LLVM.FloatingPointType} && return true
    ty isa LLVM.ArrayType && return private_numeric_layout(ty.element_type)
    ty isa LLVM.StructType && return all(private_numeric_layout, ty.elements)
    return false
end

function private_numeric_helper(fn::LLVM.Function)
    for (i, param) in enumerate(fn.parameters)
        param.value_type isa LLVM.PointerType || continue
        attrs = collect(fn.parameter_attributes[i])
        LLVM.EnumAttribute("nocapture") in attrs || return false
        readonly = LLVM.EnumAttribute("readonly") in attrs
        output = any(a -> a isa LLVM.TypeAttribute && a.kind === :sret, attrs) &&
            LLVM.EnumAttribute("noalias") in attrs
        readonly || output || return false
    end
    return all(
        inst ->
        !(inst isa Union{LLVM.IntToPtrInst, LLVM.PtrToIntInst}) &&
            !(inst isa LLVM.LoadInst && inst.value_type isa LLVM.PointerType) &&
            !(inst isa LLVM.StoreInst && inst.operands[1].value_type isa LLVM.PointerType),
        (inst for bb in fn.blocks for inst in bb.instructions)
    )
end

function private_argument_layout(fn, i, mod)
    layout = nothing
    for caller in mod.functions, bb in caller.blocks, inst in bb.instructions
        inst isa LLVM.CallInst && inst.called_operand === fn || continue
        arg = inst.operands[i]
        arg isa LLVM.AllocaInst || return nothing
        ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(arg))
        layout === nothing || layout == ty || return nothing
        layout = ty
    end
    layout isa Union{LLVM.ArrayType, LLVM.StructType} || return nothing
    return private_numeric_layout(layout) ? layout : nothing
end

"""
Normalize numeric byte views on readonly arguments and owned sret results.

All call sites must pass whole private objects with the same layout. Inputs
cannot be captured or written; outputs cannot alias any input. Local copies
let the existing alloca normalizations handle wide loads and subword stores
without changing the pointer type at the function boundary. Run after LLVM's
optimizations so SROA cannot fold the copies back into the parameters.
"""
function materialize_private_numeric_arguments!(mod::LLVM.Module)
    processed = Set{LLVM.Value}()
    outputs = Set{LLVM.Value}()
    changed = true
    while changed
        changed = false
        for fn in mod.functions
            isempty(fn.blocks) && continue
            private_numeric_helper(fn) || continue
            for (i, param) in enumerate(fn.parameters)
                param in processed && continue
                param.value_type isa LLVM.PointerType || continue
                ty = private_argument_layout(fn, i, mod)
                ty === nothing && continue
                attrs = collect(fn.parameter_attributes[i])
                output = any(a -> a isa LLVM.TypeAttribute && a.kind === :sret, attrs)
                LLVM.@dispose builder = LLVM.IRBuilder() begin
                    LLVM.position!(builder, LLVM.before(first(first(fn.blocks).instructions)))
                    local_array = LLVM.alloca!(builder, ty, "private_argument")
                    output && push!(outputs, local_array)
                    LLVM.replace_uses!(param, local_array)
                    # A byte GEP at zero lets the field-join pass recognize a
                    # direct load spanning several fields. Calls keep the whole
                    # object, so their argument type does not become i8*.
                    for use in collect(LLVM.uses(local_array))
                        inst = use.user
                        load = inst isa LLVM.LoadInst && inst.operands[1] == local_array && inst.value_type != ty
                        store = inst isa LLVM.StoreInst && inst.operands[2] == local_array && inst.operands[1].value_type != ty
                        load || store || continue
                        LLVM.position!(builder, LLVM.before(inst))
                        bytes = LLVM.gep!(
                            builder, LLVM.Int8Type(), local_array,
                            LLVM.Value[LLVM.ConstantInt(LLVM.Int64Type(), 0)], "private_argument_bytes"
                        )
                        LLVM.API.LLVMSetOperand(inst, load ? 0 : 1, bytes)
                    end
                    LLVM.position!(builder, LLVM.after(local_array))
                    if output
                        for bb in fn.blocks, inst in bb.instructions
                            inst isa LLVM.RetInst || continue
                            LLVM.position!(builder, LLVM.before(inst))
                            value = LLVM.load!(builder, ty, local_array, "private_result")
                            LLVM.store!(builder, value, param)
                        end
                    else
                        value = LLVM.load!(builder, ty, param, "private_input")
                        LLVM.store!(builder, value, local_array)
                    end
                end
                push!(processed, param)
                changed = true
            end
        end
    end
    isempty(processed) && return nothing
    # No CFG-changing optimization follows these copies.
    dl = mod.datalayout
    fix_gep_alloca_type_mismatches!(mod)
    decompose_typepun_gep_loads!(mod, dl)
    fix_alloca_type_mismatched_loads!(mod)
    fix_alloca_type_mismatched_stores!(mod, dl)
    lower_constant_subelement_access!(mod)
    index_private_array_choices!(mod, dl)
    drop_private_result_padding_stores!(mod, outputs)
    return nothing
end

function drop_private_result_padding_stores!(mod, outputs)
    for fn in mod.functions, bb in fn.blocks, inst in collect(bb.instructions)
        inst isa LLVM.StoreInst && inst.operands[1].value_type == LLVM.Int8Type() || continue
        ptr, offset, fixed = inst.operands[2], 0, true
        while ptr isa LLVM.GetElementPtrInst
            bytes, dynamic = analyze_gep_offset(ptr, mod.datalayout)
            if bytes === nothing || !isempty(dynamic)
                fixed = false
                break
            end
            offset += bytes
            ptr = ptr.operands[1]
        end
        fixed && ptr in outputs || continue
        ty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(ptr))
        ty isa LLVM.StructType || continue
        fields = flatten_type_with_offsets(ty; dl = mod.datalayout)
        any(f -> f[3] <= offset < f[3] + llvm_type_size(f[2]), fields) && continue
        # LLVM sret padding is unspecified. It has no field/address in logical
        # SPIR-V. Copy the actual result fields, not byte stores into those gaps.
        LLVM.erase!(inst)
    end
    return nothing
end
