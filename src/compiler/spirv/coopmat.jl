# SPV_KHR_cooperative_matrix emission.
#
# The front end (`device/coopmat_intrinsics.jl`) emits calls named
# `_lava_coopmat_<op>_<dtype>_<M>x<N>_<use>` returning an `i32` handle. A
# cooperative matrix is an SSA value in SPIR-V just as it is in LLVM, so the two
# graphs line up: the handle's LLVM value is simply mapped to the
# `OpCooperativeMatrix*` result id via `state.value_map`, and no
# `OpVariable`, slot table or store/reload is needed anywhere.
#
# The shape travels in the function name because a cooperative matrix type is
# built from literal constants -- component type, rows, columns, use -- and none
# of those can be a runtime operand.

"""`Use` operand of OpTypeCooperativeMatrixKHR."""
module CoopMatUse
const MatrixA           = UInt32(0)
const MatrixB           = UInt32(1)
const MatrixAccumulator = UInt32(2)
end

"""`MemoryLayout` operand of OpCooperativeMatrix{Load,Store}KHR."""
module CoopMatLayout
const RowMajor    = UInt32(0)
const ColumnMajor = UInt32(1)
end

const COOPMAT_SCOPE_SUBGROUP = UInt32(3)

"""
    parse_coopmat_name(fn_name) -> (op, T, M, N, use) | nothing

Invert `coopmat_intrinsic_name`. Returns `nothing` for names that are not
cooperative-matrix intrinsics.
"""
function parse_coopmat_name(fn_name::AbstractString)
    startswith(fn_name, "_lava_coopmat_") || return nothing
    parts = split(fn_name, '_')
    # ["", "lava", "coopmat", op, dtype, "MxN", use] with an optional trailing
    # "row" on a load or store — see `coopmat_intrinsic_name`.
    length(parts) >= 7 || return nothing
    op = parts[4]
    dtype = parts[5]
    dims = split(parts[6], 'x')
    length(dims) == 2 || return nothing
    M = parse(Int, dims[1])
    N = parse(Int, dims[2])
    use = parts[7] == "a" ? CoopMatUse.MatrixA :
          parts[7] == "b" ? CoopMatUse.MatrixB : CoopMatUse.MatrixAccumulator
    rowmajor = length(parts) >= 8 && parts[8] == "row"
    (op, dtype, M, N, use, rowmajor)
end

"""Component type id for the dtype suffix used in the intrinsic name."""
function coopmat_component_type!(state::SPIRVEmitterState, dtype::AbstractString)
    mod = state.mod
    dtype == "f16" && return emit_type_float!(mod, UInt32(16))
    dtype == "f32" && return emit_type_float!(mod, UInt32(32))
    dtype == "f64" && return emit_type_float!(mod, UInt32(64))
    dtype == "i8" && return emit_type_int!(mod, UInt32(8), UInt32(1))
    dtype == "u8" && return emit_type_int!(mod, UInt32(8), UInt32(0))
    dtype == "i32" && return emit_type_int!(mod, UInt32(32), UInt32(1))
    dtype == "u32" && return emit_type_int!(mod, UInt32(32), UInt32(0))
    error("unsupported cooperative-matrix component type: $dtype")
end

"""
    emit_coopmat_type!(state, dtype, M, N, use) -> UInt32

`OpTypeCooperativeMatrixKHR`, deduplicated per module. Declares the capability
and extension on first use. Scope is always Subgroup — the only scope Vulkan
exposes, and the one every driver-reported shape carries.
"""
function emit_coopmat_type!(state::SPIRVEmitterState, dtype::AbstractString,
                            M::Integer, N::Integer, use::UInt32)
    key = (dtype, Int(M), Int(N), use)
    cached = get(state.coopmat_type_ids, key, nothing)
    cached === nothing || return cached

    mod = state.mod
    require_capability!(mod, Cap.CooperativeMatrixKHR)
    require_extension!(mod, "SPV_KHR_cooperative_matrix")

    comp_ty = coopmat_component_type!(state, dtype)
    # Scope/Rows/Columns/Use are <id>s of constants, not literals.
    scope_id = emit_constant_u32!(mod, COOPMAT_SCOPE_SUBGROUP)
    rows_id = emit_constant_u32!(mod, UInt32(M))
    cols_id = emit_constant_u32!(mod, UInt32(N))
    use_id = emit_constant_u32!(mod, use)

    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeCooperativeMatrixKHR, id,
                        comp_ty, scope_id, rows_id, cols_id, use_id)
    state.coopmat_type_ids[key] = id
    return id
end

"""
Pointer to the tile's first element, in the PhysicalStorageBuffer class the
device arrays already use. The front end hands us the byte address as an i64.
"""
function coopmat_base_pointer!(state::SPIRVEmitterState, addr_val::LLVM.Value,
                               comp_ty::UInt32)
    mod = state.mod
    addr_id = get_value_id!(state, addr_val)
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                        SC.PhysicalStorageBuffer, comp_ty)
    ptr_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpConvertUToPtr, ptr_ty, ptr_id, addr_id)
    return ptr_id
end

"""
Pointer to the tile's first element in `Workgroup` memory — the `loadw`/`storew`
half of the intrinsics.

Nothing is converted here, and in particular nothing is **bitcast**. Vulkan's
Logical addressing model permits `OpBitcast` between pointer types only for
`PhysicalStorageBuffer`; a `Workgroup` pointer must be arrived at by
`OpAccessChain`. Emitting one anyway is not caught by `spirv-val` — the module
validates — and NVIDIA's shader compiler **segfaults on it**, inside
`vkCreateComputePipelines`, which is how this was found. GLSL's `coopMatLoad`
on a `shared` array produces an access chain for the same reason.

`ptr_val` is always the `@localmem` array itself and `idx_val` the element index
— the intrinsic passes the index rather than folding it into a `getelementptr`,
precisely so that only one shape ever reaches here. See `coopmat_intrinsics.jl`.
"""
function coopmat_workgroup_pointer!(state::SPIRVEmitterState, ptr_val::LLVM.Value,
                                    idx_val::LLVM.Value, comp_ty::UInt32)
    mod = state.mod
    src_id = get_value_id!(state, ptr_val)
    idx_id = get_value_id!(state, idx_val)
    ptr_ty = map_pointer_type!(state.type_ctx, comp_ty, SC.Workgroup)
    id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, ptr_ty, id, src_id, idx_id)
    return id
end

"""
    emit_coopmat_call!(state, inst, fn_name) -> nothing

Lower one `_lava_coopmat_*` call. Dispatched from `emit_call!`.
"""
function emit_coopmat_call!(state::SPIRVEmitterState, inst::LLVM.CallInst,
                            fn_name::AbstractString)
    parsed = parse_coopmat_name(fn_name)
    parsed === nothing && error("not a cooperative-matrix intrinsic: $fn_name")
    op, dtype, M, N, use, rowmajor = parsed

    mod = state.mod
    mat_ty = emit_coopmat_type!(state, dtype, M, N, use)
    args = LLVM.operands(inst)
    # Both layouts, because the operands want different ones from the same block:
    # `mul_mm.comp` stages A and B identically and reads A `RowMajor`, B
    # `ColumnMajor`. Hardcoding one forces a transposing staging pass for the
    # other, or a second shared copy.
    layout = rowmajor ? CoopMatLayout.RowMajor : CoopMatLayout.ColumnMajor

    # `load`/`store` address global memory through a device address;
    # `loadw`/`storew` address `@localmem` through a Workgroup pointer. The only
    # difference is how the pointer is produced — the instruction, the layout and
    # the stride are identical — so the two share everything below.
    if op == "load" || op == "loadw"
        comp_ty = coopmat_component_type!(state, dtype)
        # `loadw` carries an extra element-index argument; see below.
        ptr_id, stride_arg = op == "loadw" ?
            (coopmat_workgroup_pointer!(state, args[1], args[2], comp_ty), args[3]) :
            (coopmat_base_pointer!(state, args[1], comp_ty), args[2])
        stride_id = get_value_id!(state, stride_arg)
        layout_id = emit_constant_u32!(mod, layout)
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixLoadKHR,
                            mat_ty, id, ptr_id, layout_id, stride_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "store" || op == "storew"
        comp_ty = coopmat_component_type!(state, dtype)
        ptr_id, stride_arg, obj_arg = op == "storew" ?
            (coopmat_workgroup_pointer!(state, args[1], args[2], comp_ty), args[3], args[4]) :
            (coopmat_base_pointer!(state, args[1], comp_ty), args[2], args[3])
        stride_id = get_value_id!(state, stride_arg)
        obj_id = get_value_id!(state, obj_arg)
        layout_id = emit_constant_u32!(mod, layout)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixStoreKHR,
                            ptr_id, obj_id, layout_id, stride_id)

    elseif op == "zero"
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstantNull, mat_ty, id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "muladd"
        a_id = get_value_id!(state, args[1])
        b_id = get_value_id!(state, args[2])
        c_id = get_value_id!(state, args[3])
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixMulAddKHR,
                            mat_ty, id, a_id, b_id, c_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "convert"
        # `OpFConvert` between two cooperative matrix types of the same scope,
        # rows, columns and use — the one instruction that lets an fp32
        # accumulator be stored as fp16 without a round trip through memory.
        # `mat_ty` is the *result* type, from the intrinsic's name; the source
        # type comes with the operand.
        src_id = get_value_id!(state, args[1])
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpFConvert, mat_ty, id, src_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    else
        error("unknown cooperative-matrix op: $op")
    end
    return nothing
end
