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

# `Scope` operand of OpTypeCooperativeMatrixKHR. Both values are read off
# glslang's output for `gl_ScopeSubgroup` / `gl_ScopeWorkgroup` — see
# `test/glsl/workgroup_scope_coopmat.comp`, whose `TypeCooperativeMatrixKHR`
# names a constant 2. Nothing else about a workgroup-scope module differs: the
# capability and extension list is identical, and so is every instruction that
# consumes the type.
const COOPMAT_SCOPE_WORKGROUP = UInt32(2)
const COOPMAT_SCOPE_SUBGROUP  = UInt32(3)

# `OpTypeTensorLayoutNV`'s clamp mode. Values observed, not remembered: a shader
# declaring one layout per `gl_CooperativeMatrixClampMode*NV` emits
# `TypeTensorLayoutNV %dim %0`, `%1` and `%2` in that order.
#
# `TENSOR_CLAMP_CONSTANT` is the one that matters for the GEMM. Under it the
# driver bounds-checks the load itself and substitutes a constant out of range,
# which is what makes an unpadded extent legal — i.e. what would retire
# `gemm_padn`, the `GEMM_BLOCK` pad on M and `padtile`/`crsextent` on K.
const TENSOR_CLAMP_UNDEFINED    = UInt32(0)
const TENSOR_CLAMP_CONSTANT     = UInt32(1)
const TENSOR_CLAMP_TO_EDGE      = UInt32(2)

"""
    parse_coopmat_name(fn_name) -> (op, T, M, N, use, rowmajor, scope) | nothing

Invert `coopmat_intrinsic_name`. Returns `nothing` for names that are not
cooperative-matrix intrinsics.
"""
function parse_coopmat_name(fn_name::AbstractString)
    startswith(fn_name, "_lava_coopmat_") || return nothing
    parts = split(fn_name, '_')
    # ["", "lava", "coopmat", op, dtype, "MxN", use] with optional trailing flags
    # "wg" (workgroup scope) and "row" (row-major memory layout) — see
    # `coopmat_intrinsic_name`. Absence of "wg" is subgroup scope, which is what
    # keeps every pre-existing name byte-identical.
    length(parts) >= 7 || return nothing
    op = parts[4]
    dtype = parts[5]
    dims = split(parts[6], 'x')
    length(dims) == 2 || return nothing
    M = parse(Int, dims[1])
    N = parse(Int, dims[2])
    use = parts[7] == "a" ? CoopMatUse.MatrixA :
          parts[7] == "b" ? CoopMatUse.MatrixB : CoopMatUse.MatrixAccumulator
    flags = @view parts[8:end]
    rowmajor = "row" in flags
    scope = "wg" in flags ? COOPMAT_SCOPE_WORKGROUP : COOPMAT_SCOPE_SUBGROUP
    (op, dtype, M, N, use, rowmajor, scope)
end

"""Size in bytes of the dtype suffix used in the intrinsic name — the alignment
a `PhysicalStorageBuffer` access has to declare."""
function coopmat_component_bytes(dtype::AbstractString)
    dtype == "f16" && return 2
    dtype == "f32" && return 4
    dtype == "f64" && return 8
    (dtype == "i8" || dtype == "u8") && return 1
    (dtype == "i32" || dtype == "u32") && return 4
    error("unsupported cooperative-matrix component type: $dtype")
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
    coopmat_type_use(state, ty_id) -> UInt32 | nothing

The `Use` of an already-emitted cooperative-matrix type, by inverting
`state.coopmat_type_ids` (keyed `(dtype, M, N, use, scope)`).

Needed because the conversion instruction depends on it: `OpFConvert` is legal
only when source and destination agree on use, and a use change is
`OpCooperativeMatrixConvertNV`. Returns `nothing` for an id that is not a
cooperative-matrix type.
"""
function coopmat_type_use(state::SPIRVEmitterState, ty_id::UInt32)
    k = coopmat_type_key(state, ty_id)
    return k === nothing ? nothing : k[4]
end

"""
    coopmat_type_key(state, ty_id) -> (dtype, M, N, use, scope) | nothing

The full `coopmat_type_ids` key behind an emitted type id. `coopmat_type_use`
wants only the use; a conversion that changes BOTH component type and use needs
the rest as well, to build the intermediate type it goes through.
"""
function coopmat_type_key(state::SPIRVEmitterState, ty_id::UInt32)
    for (k, v) in state.coopmat_type_ids
        v == ty_id && return k
    end
    return nothing
end

"""
    emit_coopmat_type!(state, dtype, M, N, use, scope) -> UInt32

`OpTypeCooperativeMatrixKHR`, deduplicated per module. Declares the capability
and extension on first use.

`scope` defaults to Subgroup, which is the portable one: every KHR-reported
shape carries it and it is what AMD's WMMA path has. Workgroup scope needs
`VK_NV_cooperative_matrix2`'s `cooperativeMatrixWorkgroupScope` — enabled at
device creation and reported as `ctx.coopmat2.workgroup_scope` — and the shape
must then match the granularity that device reports **for the workgroup size the
kernel launches with**. Nothing else in the module changes; see
`test/glsl/workgroup_scope_coopmat.comp`.
"""
function emit_coopmat_type!(state::SPIRVEmitterState, dtype::AbstractString,
                            M::Integer, N::Integer, use::UInt32,
                            scope::UInt32 = COOPMAT_SCOPE_SUBGROUP)
    key = (dtype, Int(M), Int(N), use, scope)
    cached = get(state.coopmat_type_ids, key, nothing)
    cached === nothing || return cached

    mod = state.mod
    require_capability!(mod, Cap.CooperativeMatrixKHR)
    require_extension!(mod, "SPV_KHR_cooperative_matrix")

    comp_ty = coopmat_component_type!(state, dtype)
    # Scope/Rows/Columns/Use are <id>s of constants, not literals.
    scope_id = emit_constant_u32!(mod, scope)
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
    emit_tensor_layout_type!(state, dim, clampmode) -> UInt32

`OpTypeTensorLayoutNV`, deduplicated per module.

`dim` and `clampmode` are **constant <id>s, not literals** — the same shape as
`OpTypeCooperativeMatrixKHR`'s rows/columns, and not what the operand list
suggests. Read off a glslang binary; see `test/glsl/tensor_addressing_opcodes.comp`.

The capability is `TensorAddressingNV` from `SPV_NV_tensor_addressing`, a
DIFFERENT extension from the coopmat2 one that supplies the load. Declaring only
`CooperativeMatrixTensorAddressingNV` yields a module the driver rejects.
"""
function emit_tensor_layout_type!(state::SPIRVEmitterState, dim::Integer,
                                  clampmode::UInt32 = TENSOR_CLAMP_UNDEFINED)
    key = (Int(dim), clampmode)
    cached = get(state.tensor_layout_type_ids, key, nothing)
    cached === nothing || return cached

    mod = state.mod
    require_capability!(mod, Cap.TensorAddressingNV)
    require_extension!(mod, "SPV_NV_tensor_addressing")

    dim_id = emit_constant_u32!(mod, UInt32(dim))
    clamp_id = emit_constant_u32!(mod, clampmode)
    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeTensorLayoutNV, id, dim_id, clamp_id)
    state.tensor_layout_type_ids[key] = id
    return id
end

"""
    emit_tensor_view_type!(state, dim, hasdims, perm) -> UInt32

`OpTypeTensorViewNV`, deduplicated per module. `perm` is the dimension
permutation — `(1, 0)` is the transpose that `mul_mm_cm2.comp` uses for its B
operand, and it is what replaces staging a transposed copy.

Operands are `%Dim %HasDimensions %p0 %p1 …`, all constant <id>s; `HasDimensions`
is a Bool constant, so it is `OpConstantTrue`/`OpConstantFalse` rather than an
integer.
"""
function emit_tensor_view_type!(state::SPIRVEmitterState, dim::Integer,
                                hasdims::Bool, perm::AbstractVector{<:Integer})
    length(perm) == dim ||
        error("tensor view permutation has $(length(perm)) entries for dim $dim")
    permu = UInt32[UInt32(p) for p in perm]
    key = (Int(dim), hasdims, permu)
    cached = get(state.tensor_view_type_ids, key, nothing)
    cached === nothing || return cached

    mod = state.mod
    require_capability!(mod, Cap.TensorAddressingNV)
    require_extension!(mod, "SPV_NV_tensor_addressing")

    dim_id = emit_constant_u32!(mod, UInt32(dim))
    has_id = emit_constant_bool!(mod, hasdims)
    perm_ids = UInt32[emit_constant_u32!(mod, p) for p in permu]
    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeTensorViewNV, id,
                        dim_id, has_id, perm_ids...)
    state.tensor_view_type_ids[key] = id
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
Get or create the `Function`-storage `OpVariable` that backs component access for
one cooperative-matrix type.

SPIR-V has no "extract component N" instruction for a cooperative matrix. What it
has is a rule that a matrix in a `Function` variable may be indexed by
`OpAccessChain`, one index, yielding a pointer to a component — which is what
GLSL's `mat[i]` compiles to. So component access needs somewhere to put the
matrix, and that somewhere has to be a variable in the entry block's preamble,
not a value.

One variable per *type*, cached, and reused by every get and set: a matrix is an
SSA value, so a variable holding one is scratch, not storage, and two different
matrices of the same type never need to be in it at once — each access writes
before it reads.
"""
function coopmat_component_var!(state::SPIRVEmitterState, mat_ty::UInt32)
    cached = get(state.coopmat_component_vars, mat_ty, nothing)
    cached === nothing || return cached
    mod = state.mod
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                        SC.Function, mat_ty)
    var_id = fresh_id!(mod)
    encode_instruction!(state.entry_function_locals, Op.OpVariable, ptr_ty, var_id,
                        SC.Function)
    state.coopmat_component_vars[mat_ty] = var_id
    return var_id
end

"""
Allocate the component-access variable for every matrix type this function reads
or writes components of, **before** any block is emitted.

The same hazard `prescan_function_for_rayquery!` exists for: the `OpVariable`
words are buffered in `state.entry_function_locals` and injected into the entry
block's preamble, so allocating one lazily from inside a loop body emits an
`OpStore` to an id that is not defined yet. `spirv-val` catches it — "ID '96' has
not been defined" — which is how this was found, and it is the reason this
pre-scan is not optional.
"""
function prescan_function_for_coopmat_components!(state::SPIRVEmitterState,
                                                  fn::LLVM.Function)
    isempty(LLVM.blocks(fn)) && return nothing
    for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        callee isa LLVM.Function || continue
        parsed = parse_coopmat_name(LLVM.name(callee))
        parsed === nothing && continue
        op, dtype, M, N, use, _, scope = parsed
        (op == "getcomp" || op == "setcomp") || continue
        coopmat_component_var!(state, emit_coopmat_type!(state, dtype, M, N, use, scope))
    end
    return nothing
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
    coopmat_spill_once!(state, var_id, m_id) -> nothing

Put matrix `m_id` into the component variable `var_id`, unless it is already
there.

A cooperative matrix can only be indexed through a `Function`-storage variable,
so both `getcomp` and `setcomp` have to spill the whole tile before they can
reach one component. Done literally that is quadratic in the obvious usage: an
epilogue or a rescale that touches all `n` components spills and reloads the
entire matrix `n` times to change `n` values.

**This is worth doing and it bought no measured time.** It was built to explain
why holding SAM 2's attention `O` in accumulators loses, and it does not: the
flash kernel measured 4.955 ms before and 4.950 after, the GEMM's gelu epilogue
is unchanged at 255 registers, and the real cause turned out to be occupancy
(see `DNNKernels`' `FLASHCM_HELD`). NVIDIA's shader compiler evidently coalesces
these stores itself. Kept because emitting `n` stores of a tile to change `n` of
its components is indefensible on its own terms and another driver need not be as
forgiving — but do not expect it to move a number, and do not let its existence
imply it once did.

The variable is scratch and only these two ops write it, so within a basic block
its contents are known exactly. `setcomp` records the value it loaded back, and
the next access on that value finds it already resident and emits nothing. What
remains is one store, `n` access-chain pairs, and one load — the loads in between
have no consumer left and the driver drops them.

Sound only inside one block: `emit_block!` clears the tracking at every
`OpLabel`, because a branch can arrive with the variable holding anything.
"""
function coopmat_spill_once!(state::SPIRVEmitterState, var_id::UInt32, m_id::UInt32)
    get(state.coopmat_var_contents, var_id, nothing) === m_id && return nothing
    encode_instruction!(state.mod.functions, Op.OpStore, var_id, m_id)
    state.coopmat_var_contents[var_id] = m_id
    return nothing
end

"""
    parse_tensor_name(fn_name) -> (op, dim, clamp, rest) | nothing

Invert the tensor-addressing intrinsic names, which are
`_lava_tensor_<op>_<dim>_<clamp>[_<rest>]`. `dim` and `clamp` ride in the name
for the same reason the matrix shape does: they are operands of the TYPE, built
from constants, and no constant can be a runtime value.
"""
function parse_tensor_name(fn_name::AbstractString)
    startswith(fn_name, "_lava_tensor_") || return nothing
    parts = split(fn_name, '_')
    # ["", "lava", "tensor", op, dim, clamp, rest...]
    length(parts) >= 6 || return nothing
    op = parts[4]
    dim = tryparse(Int, parts[5]);   dim === nothing && return nothing
    cl  = tryparse(UInt32, parts[6]); cl === nothing && return nothing
    (op, dim, cl, parts[7:end])
end

"""
    emit_tensor_call!(state, inst, fn_name) -> nothing

Lower one `_lava_tensor_*` call — the `SPV_NV_tensor_addressing` half of the
coopmat2 staging path. Dispatched from `emit_call!`.

A layout is an SSA value exactly as a cooperative matrix is, so the handle's
LLVM value maps straight to the SPIR-V result id and needs no variable or slot.
Every operand other than the type parameters is a runtime `<id>`: the dimensions
and the slice offsets are ordinary i32 values, which is the entire point — the
block a workgroup owns is computed, not baked.
"""
function emit_tensor_call!(state::SPIRVEmitterState, inst::LLVM.CallInst,
                           fn_name::AbstractString)
    parsed = parse_tensor_name(fn_name)
    parsed === nothing && error("not a tensor-addressing intrinsic: $fn_name")
    op, dim, clamp, rest = parsed

    mod = state.mod
    args = LLVM.operands(inst)          # trailing operand is the callee
    nargs = length(args) - 1
    layout_ty = emit_tensor_layout_type!(state, dim, clamp)
    require_capability!(mod, Cap.TensorAddressingNV)
    require_extension!(mod, "SPV_NV_tensor_addressing")

    if op == "create"
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCreateTensorLayoutNV, layout_ty, id)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = layout_ty

    elseif op == "view"
        # `_lava_tensor_view_<dim>_0_<p0>x<p1>...` — the clamp field is unused
        # (a view has no clamp mode) and carries 0 so the shared name parser
        # still applies. `perm` is the dimension permutation: (1, 0) is the
        # transpose that lets K be read in place instead of staged as a copy.
        length(rest) == 1 || error("malformed tensor view name: $fn_name")
        perm = [parse(Int, x) for x in split(rest[1], 'x')]
        view_ty = emit_tensor_view_type!(state, dim, false, perm)
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCreateTensorViewNV, view_ty, id)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = view_ty

    elseif op == "setdim"
        # %layout followed by one <id> per dimension.
        nargs == dim + 1 ||
            error("tensor setdim for dim $dim takes $(dim + 1) arguments, got $nargs")
        ids = UInt32[get_value_id!(state, args[k]) for k in 1:nargs]
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpTensorLayoutSetDimensionNV,
                            layout_ty, id, ids...)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = layout_ty

    elseif op == "setstride"
        # Same operand shape as `setdim`: %layout followed by one <id> per
        # dimension. What it carries is the distance in ELEMENTS between
        # successive indices of that dimension, which is how a layout describes a
        # slab of a larger array rather than a packed one.
        nargs == dim + 1 ||
            error("tensor setstride for dim $dim takes $(dim + 1) arguments, got $nargs")
        ids = UInt32[get_value_id!(state, args[k]) for k in 1:nargs]
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpTensorLayoutSetStrideNV,
                            layout_ty, id, ids...)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = layout_ty

    elseif op == "setclampvalue"
        # %layout %value — ONE operand whatever the rank, unlike setdim and
        # setstride, because the fill is a single element and not a per-axis
        # quantity.
        nargs == 2 ||
            error("tensor setclampvalue takes 2 arguments, got $nargs")
        ids = UInt32[get_value_id!(state, args[k]) for k in 1:nargs]
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpTensorLayoutSetClampValueNV,
                            layout_ty, id, ids...)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = layout_ty

    elseif op == "slice"
        # %layout then OFFSET/SIZE PAIRS, one pair per dimension — not all the
        # offsets followed by all the sizes, which is the natural mis-reading.
        nargs == 2dim + 1 ||
            error("tensor slice for dim $dim takes $(2dim + 1) arguments, got $nargs")
        ids = UInt32[get_value_id!(state, args[k]) for k in 1:nargs]
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpTensorLayoutSliceNV,
                            layout_ty, id, ids...)
        state.value_map[inst] = id
        state.tensor_value_types[inst] = layout_ty

    elseif op == "load" || op == "loadv"
        # `_lava_tensor_load_<dim>_<clamp>_<dtype>_<M>x<N>_<use>`, and `loadv`
        # for the same thing through a TensorView — a separate name because the
        # arity differs and one LLVM symbol cannot carry both signatures.
        #   (i64 address, i32 object-handle, i32 layout-handle) -> i32 matrix
        3 <= length(rest) <= 4 || error("malformed tensor load name: $fn_name")
        dtype = rest[1]
        dims = split(rest[2], 'x'); length(dims) == 2 || error("bad shape in $fn_name")
        M = parse(Int, dims[1]); N = parse(Int, dims[2])
        # The use rides as the same "a"/"b"/"acc" suffix `parse_coopmat_name`
        # decodes, not as a number — and so does the optional "wg" scope flag
        # after it.
        use = rest[3] == "a" ? CoopMatUse.MatrixA :
              rest[3] == "b" ? CoopMatUse.MatrixB : CoopMatUse.MatrixAccumulator
        scope = (length(rest) == 4 && rest[4] == "wg") ? COOPMAT_SCOPE_WORKGROUP :
                                                         COOPMAT_SCOPE_SUBGROUP
        wantargs = op == "loadv" ? 4 : 3
        nargs == wantargs || error("tensor $op takes $wantargs arguments, got $nargs")

        mat_ty = emit_coopmat_type!(state, dtype, M, N, use, scope)
        comp_ty = coopmat_component_type!(state, dtype)
        ptr_id = coopmat_base_pointer!(state, args[1], comp_ty)
        obj_id = get_value_id!(state, args[2])
        layout_id = get_value_id!(state, args[3])

        # The load is the coopmat2 half, so it needs THAT capability as well as
        # the tensor-addressing one declared above — two extensions, one
        # instruction sequence.
        require_capability!(mod, Cap.CooperativeMatrixTensorAddressingNV)
        require_extension!(mod, "SPV_NV_cooperative_matrix2")

        id = fresh_id!(mod)
        # MemoryOperands must be `Aligned` with a literal alignment, NOT `None`.
        # glslang's reference emits `None None` — but its shader loads from a
        # descriptor binding, and ours is a `PhysicalStorageBuffer` address, where
        # VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708 requires the
        # alignment. spirv-val says so precisely; copying the reference verbatim
        # would have produced a module that only fails on the buffer-device-address
        # path this actually uses.
        # And `TensorAddressingOperands` is then NOT optional: with the alignment
        # present the decoder expects the trailing word, and omitting it aborts
        # with "expected more operands after 8 words" rather than anything about
        # tensors. Order is MemoryOperands, its literal, then the tensor mask.
        align = coopmat_component_bytes(dtype)
        if op == "loadv"
            # With a view, the trailing mask names it and the view's <id> follows
            # the mask. `TENSOR_ADDR_TENSORVIEW` is read from the SPIR-V headers,
            # not inferred — the wrong bit loads the wrong elements silently.
            view_id = get_value_id!(state, args[4])
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixLoadTensorNV,
                                mat_ty, id, ptr_id, obj_id, layout_id,
                                MemOp.Aligned, UInt32(align),
                                Op.TENSOR_ADDR_TENSORVIEW, view_id)
        else
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixLoadTensorNV,
                                mat_ty, id, ptr_id, obj_id, layout_id,
                                MemOp.Aligned, UInt32(align), Op.TENSOR_ADDR_NONE)
        end
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "store" || op == "storev"
        # `_lava_tensor_store_<dim>_<clamp>_<dtype>_<M>x<N>_<use>`, and `storev`
        # for the same thing through a TensorView — separate names because the
        # arity differs, exactly as for the load.
        #   (i64 address, i32 matrix-handle, i32 layout-handle[, i32 view]) -> nothing
        #
        # The mirror of the load, and the half that makes a ragged OUTPUT legal:
        # under a clamping layout the store writes only the elements the layout's
        # dimensions admit, so an edge tile stops at the extent instead of running
        # past it. That is what lets a GEMM skip `gemm_padn`/`GEMM_BLOCK` on the
        # DESTINATION as well as the operands.
        3 <= length(rest) <= 4 || error("malformed tensor store name: $fn_name")
        dtype = rest[1]
        dims = split(rest[2], 'x'); length(dims) == 2 || error("bad shape in $fn_name")
        M = parse(Int, dims[1]); N = parse(Int, dims[2])
        use = rest[3] == "a" ? CoopMatUse.MatrixA :
              rest[3] == "b" ? CoopMatUse.MatrixB : CoopMatUse.MatrixAccumulator
        scope = (length(rest) == 4 && rest[4] == "wg") ? COOPMAT_SCOPE_WORKGROUP :
                                                         COOPMAT_SCOPE_SUBGROUP
        wantargs = op == "storev" ? 4 : 3
        nargs == wantargs || error("tensor $op takes $wantargs arguments, got $nargs")

        emit_coopmat_type!(state, dtype, M, N, use, scope)   # declares caps/extension
        comp_ty = coopmat_component_type!(state, dtype)
        ptr_id = coopmat_base_pointer!(state, args[1], comp_ty)
        mat_id = get_value_id!(state, args[2])
        layout_id = get_value_id!(state, args[3])

        require_capability!(mod, Cap.CooperativeMatrixTensorAddressingNV)
        require_extension!(mod, "SPV_NV_cooperative_matrix2")

        # Same two operand rules as the load, for the same reasons: `Aligned` with
        # a literal rather than `None`, because ours is a `PhysicalStorageBuffer`
        # address (VUID-…-04708) where glslang's reference uses a descriptor
        # binding and emits `None`; and `TensorAddressingOperands` is then
        # mandatory, not optional. No result type, no result id.
        align = coopmat_component_bytes(dtype)
        if op == "storev"
            # The view names itself in the trailing mask and its <id> follows —
            # the same encoding as the load's, which is why the bit is read from
            # the SPIR-V headers once and shared.
            view_id = get_value_id!(state, args[4])
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixStoreTensorNV,
                                ptr_id, mat_id, layout_id,
                                MemOp.Aligned, UInt32(align),
                                Op.TENSOR_ADDR_TENSORVIEW, view_id)
        else
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixStoreTensorNV,
                                ptr_id, mat_id, layout_id,
                                MemOp.Aligned, UInt32(align), Op.TENSOR_ADDR_NONE)
        end
        # A store produces no value, so nothing is recorded in `value_map`.

    else
        error("unsupported tensor-addressing op `$op` in $fn_name")
    end
    return nothing
end

"""
    emit_coopmat_call!(state, inst, fn_name) -> nothing

Lower one `_lava_coopmat_*` call. Dispatched from `emit_call!`.
"""
function emit_coopmat_call!(state::SPIRVEmitterState, inst::LLVM.CallInst,
                            fn_name::AbstractString)
    parsed = parse_coopmat_name(fn_name)
    parsed === nothing && error("not a cooperative-matrix intrinsic: $fn_name")
    op, dtype, M, N, use, rowmajor, scope = parsed

    mod = state.mod
    mat_ty = emit_coopmat_type!(state, dtype, M, N, use, scope)
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
    # `loadw`/`storew` address `@localmem`, which every invocation in the group
    # can see, so the access must say so — under the Vulkan memory model an
    # untagged access is private and a workgroup barrier grants it nothing. This
    # is the whole correctness of a staged GEMM: without it the tiles a
    # cooperative-matrix load reads are whatever the driver felt like leaving in
    # shared memory. See `nonprivate` and the barrier in `emit.jl`.
    if op == "load" || op == "loadw" || startswith(op, "loadw")
        comp_ty = coopmat_component_type!(state, dtype)
        # `loadw<W>` addresses a `@localmem` of W-wide vectors, so the access
        # chain must yield a pointer to the *vector*: `OpCooperativeMatrixLoadKHR`
        # accepts a vector pointee whose component type matches the matrix, and
        # counts `Stride` in those vectors. That is how `mul_mm.comp` gets 32-bit
        # shared accesses out of an fp16 tile. See `coopmat_vecwidth`.
        #
        # The width is read from the name rather than fixed at 2, so 4-wide
        # (64-bit) shared accesses work by the same path. SPIR-V vectors are 2, 3
        # or 4 components, which is the ceiling here; `coopmat_sharedok` rejects
        # anything else before a name is ever built.
        vw = (op == "load" || op == "loadw") ? 0 : parse(Int, op[6:end])
        pointee = vw == 0 ? comp_ty : emit_type_vector!(mod, comp_ty, UInt32(vw))
        # `loadw`/`loadw2` carry an extra element-index argument; see below.
        ptr_id, stride_arg = op != "load" ?
            (coopmat_workgroup_pointer!(state, args[1], args[2], pointee), args[3]) :
            (coopmat_base_pointer!(state, args[1], comp_ty), args[2])
        stride_id = get_value_id!(state, stride_arg)
        layout_id = emit_constant_u32!(mod, layout)
        id = fresh_id!(mod)
        if op != "load"
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixLoadKHR,
                                mat_ty, id, ptr_id, layout_id, stride_id,
                                MemOp.NonPrivatePointer)
        else
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixLoadKHR,
                                mat_ty, id, ptr_id, layout_id, stride_id)
        end
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
        if op == "storew"
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixStoreKHR,
                                ptr_id, obj_id, layout_id, stride_id,
                                MemOp.NonPrivatePointer)
        else
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixStoreKHR,
                                ptr_id, obj_id, layout_id, stride_id)
        end

    elseif op == "zero"
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstantNull, mat_ty, id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "undef"
        # An UNINITIALISED matrix, for a destination whose prior contents cannot
        # be observed — which is what `tensor_load` under a CONSTANT clamp mode
        # has: out-of-range elements come from the layout's clamp value, not from
        # the destination. GLSL gets this for free by declaring `coopmat mat_a;`
        # and never assigning it; we were passing `coopmat_zero`, which is a real
        # value the allocator has to hold until the load overwrites it.
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpUndef, mat_ty, id)
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
        src_ty = get(state.coopmat_value_types, args[1], UInt32(0))
        id = fresh_id!(mod)
        # `OpFConvert` requires the two matrix types to agree on USE as well as
        # shape. A conversion that also changes use — fp32 Accumulator to fp16
        # MatrixA, which is what feeds the second product of a flash kernel — is
        # `OpCooperativeMatrixConvertNV`. Choosing on the emitted type ids keeps
        # the same intrinsic serving both, so callers do not have to know which
        # instruction their conversion turns into.
        sk = src_ty == UInt32(0) ? nothing : coopmat_type_key(state, src_ty)
        dk = coopmat_type_key(state, mat_ty)
        if sk !== nothing && dk !== nothing && sk[4] != dk[4]
            # A use change is `OpCooperativeMatrixConvertNV`, and that instruction
            # requires the two component types to be IDENTICAL — spirv-val:
            # "Result Type and Matrix component types mismatch". So a conversion
            # that changes BOTH (fp32 Accumulator -> fp16 MatrixA, which is what
            # feeds a flash kernel's second product) is two instructions: convert
            # the component type first, at the SOURCE use, then the use. The
            # reference sidesteps this by keeping `P` in fp16 throughout, so only
            # the use ever changes there.
            require_capability!(mod, Cap.CooperativeMatrixConversionsNV)
            require_extension!(mod, "SPV_NV_cooperative_matrix2")
            from = src_id
            if sk[1] != dk[1]
                mid_ty = emit_coopmat_type!(state, dk[1], sk[2], sk[3], sk[4], sk[5])
                mid_id = fresh_id!(mod)
                encode_instruction!(mod.functions, Op.OpFConvert, mid_ty, mid_id, src_id)
                from = mid_id
            end
            encode_instruction!(mod.functions, Op.OpCooperativeMatrixConvertNV,
                                mat_ty, id, from)
        else
            encode_instruction!(mod.functions, Op.OpFConvert, mat_ty, id, src_id)
        end
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "length"
        # Takes the matrix TYPE, not a value — the component count is a property
        # of how the implementation splits the type across the subgroup.
        u32 = emit_type_int!(mod, UInt32(32), UInt32(0))
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixLengthKHR,
                            u32, id, mat_ty)
        state.value_map[inst] = id

    elseif op == "getcomp"
        comp_ty = coopmat_component_type!(state, dtype)
        var_id = coopmat_component_var!(state, mat_ty)
        m_id = get_value_id!(state, args[1])
        i_id = get_value_id!(state, args[2])
        coopmat_spill_once!(state, var_id, m_id)
        ptr_ty = map_pointer_type!(state.type_ctx, comp_ty, SC.Function)
        ptr_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpAccessChain, ptr_ty, ptr_id,
                            var_id, i_id)
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, comp_ty, id, ptr_id)
        state.value_map[inst] = id

    elseif op == "setcomp"
        comp_ty = coopmat_component_type!(state, dtype)
        var_id = coopmat_component_var!(state, mat_ty)
        m_id = get_value_id!(state, args[1])
        i_id = get_value_id!(state, args[2])
        v_id = get_value_id!(state, args[3])
        coopmat_spill_once!(state, var_id, m_id)
        ptr_ty = map_pointer_type!(state.type_ctx, comp_ty, SC.Function)
        ptr_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpAccessChain, ptr_ty, ptr_id,
                            var_id, i_id)
        encode_instruction!(mod.functions, Op.OpStore, ptr_id, v_id)
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, mat_ty, id, var_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty
        # The variable now holds what this load produced, so a `getcomp` or a
        # further `setcomp` on the result needs no store of its own. This is
        # what turns an N-component update from N spills into one.
        state.coopmat_var_contents[var_id] = id

    elseif op == "mul" || op == "add"
        # `OpFMul` and `OpFAdd` on two cooperative matrices of the same type are
        # defined component-wise. Nothing is materialised: this is the one way to
        # combine two matrices without reaching for a component at all.
        #
        # `add` is what lets a GEMM start from a bias AND a residual — two
        # accumulator loads, one at stride 0 for the bias broadcast and one at
        # the residual's real leading dimension, summed before the k-loop. See
        # `accinit`.
        a_id = get_value_id!(state, args[1])
        b_id = get_value_id!(state, args[2])
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, op == "mul" ? Op.OpFMul : Op.OpFAdd,
                            mat_ty, id, a_id, b_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif op == "perelem" || op == "perelemm"
        # `OpCooperativeMatrixPerElementOpNV` — apply a function to every element
        # of the matrix, with the element's (row, column) handed to it.
        #
        # `perelemm` is the same instruction with a MATRIX as its extra operand,
        # which the spec allows and for which the callback receives that
        # matrix's corresponding ELEMENT. Its operand cannot come from the marker
        # call the way a scalar extra does: the marker fixes the callback's
        # signature and so must carry an element, while the instruction needs the
        # matrix. The matrix therefore rides on the `llvmcall` as a handle, and
        # is resolved here.
        #
        # The whole point is that the driver reaches the elements itself. Written
        # with `getcomp`/`setcomp` the same rescale costs a `Function`-storage
        # variable holding the entire tile, which on SAM 2's attention took the
        # kernel from 123 registers to 192 and so from two resident workgroups per
        # SM to one — see `FLASHCM_HELD` in DNNKernels.
        #
        # The callback travels as the SECOND argument, and it is not a value: it
        # is a *call* to the callback, placed there by `coopmat_perelement` purely
        # so the callee appears in the LLVM call graph. `emit_call!` drops that
        # call (see `coopmat_perelement_marker`); here we resolve it back to the
        # callee's SPIR-V function id, which `spirv_from_llvm!` pre-allocated for
        # every function with a body.
        require_capability!(mod, Cap.CooperativeMatrixPerElementOperationsNV)
        require_extension!(mod, "SPV_NV_cooperative_matrix2")
        m_id = get_value_id!(state, args[1])
        marker = args[2]
        marker isa LLVM.CallInst ||
            error("cooperative-matrix per-element callback did not survive as a call; " *
                  "the callback must be a top-level `@noinline` function of " *
                  "(::UInt32, ::UInt32, ::$dtype), not a closure")
        callee = LLVM.called_operand(marker)
        callee isa LLVM.Function ||
            error("cooperative-matrix per-element callback is an indirect call")
        func_id = get(state.value_map, callee, nothing)
        func_id === nothing &&
            error("cooperative-matrix per-element callback $(LLVM.name(callee)) has no " *
                  "SPIR-V function; it was probably inlined away")
        # Everything the callback needs beyond (row, col, element) rides along as
        # a trailing operand. They are read off the marker call, where they are
        # the real values — only the first three arguments there are dummies.
        # A CallInst's operand list ends with the callee, hence `end - 1`.
        margs = LLVM.operands(marker)
        extras = if op == "perelemm"
            # args = (matrix, marker, other-matrix); the marker's own trailing
            # argument is the dummy element that gave `f` its signature.
            # `nargs` belongs to the tensor emitter; here the count is the
            # operand list minus its trailing callee.
            length(args) - 1 == 3 ||
                error("perelemm takes 3 arguments, got $(length(args) - 1)")
            UInt32[get_value_id!(state, args[3])]
        else
            UInt32[get_value_id!(state, margs[k]) for k in 4:(length(margs) - 1)]
        end
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixPerElementOpNV,
                            mat_ty, id, m_id, func_id, extras...)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    elseif startswith(op, "reduce")
        # `OpCooperativeMatrixReduceNV` — combine elements along an axis with a
        # binary function, then SMEAR the result across the destination.
        #
        # Semantics taken from `flash_attn_cm2.comp`, which is the only reference
        # implementation to hand and uses all of them:
        #
        #     coopMatReduceNV(rowmax, S, gl_CooperativeMatrixReduceRowNV, maxReduce)
        #     ACC_TYPE maxReduce(const in ACC_TYPE x, const in ACC_TYPE y) { return max(x, y); }
        #
        # Two things that are not obvious and cost a wrong guess each:
        #
        #  * the callback is a BINARY COMBINER `(x, y) -> z`, not the per-element
        #    `(row, col, elem)` of `OpCooperativeMatrixPerElementOpNV`;
        #  * the result is not a vector. It has the destination matrix's shape,
        #    with the reduced value repeated along the reduced axis — which is why
        #    the reference has a `smearReduce(x, y) = x` that reduces nothing and
        #    exists purely to broadcast. Its `eM` is `Br x Bc` and its `eMdiag` is
        #    `Br x HSV_pad`, so the destination may also have a DIFFERENT extent
        #    along that axis; that half needs flexible dimensions.
        #
        # The mask is a literal operand, so it rides in the intrinsic name
        # (`reduce1`, `reduce3`, …) rather than as an argument — the same reason
        # the component type and extents do.
        require_capability!(mod, Cap.CooperativeMatrixReductionsNV)
        require_extension!(mod, "SPV_NV_cooperative_matrix2")
        mask = parse(UInt32, op[length("reduce")+1:end])
        m_id = get_value_id!(state, args[1])
        marker = args[2]
        marker isa LLVM.CallInst ||
            error("cooperative-matrix reduce callback did not survive as a call; " *
                  "it must be a top-level function of (::$dtype, ::$dtype), not a closure")
        callee = LLVM.called_operand(marker)
        callee isa LLVM.Function ||
            error("cooperative-matrix reduce callback is an indirect call")
        func_id = get(state.value_map, callee, nothing)
        func_id === nothing &&
            error("cooperative-matrix reduce callback $(LLVM.name(callee)) has no " *
                  "SPIR-V function; it was probably inlined away")
        id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCooperativeMatrixReduceNV,
                            mat_ty, id, m_id, mask, func_id)
        state.value_map[inst] = id
        state.coopmat_value_types[inst] = mat_ty

    else
        error("unknown cooperative-matrix op: $op")
    end
    return nothing
end

"""
    coopmat_perelement_marker(inst) -> Bool

True when `inst` exists only to name the callback of an
`OpCooperativeMatrixPerElementOpNV`, and must therefore not be emitted as a call.

There is no way to hand an LLVM function to `Base.llvmcall` as an operand, so
`coopmat_perelement` calls the callback once, with `(0, 0, zero(T))`, and passes
the result to the intrinsic. That call is a *marker*: it puts the callee in the
call graph so the emitter gives it an `OpFunction`, and its value is never used
for anything else. Emitting it as well would run the callback an extra time per
matrix, inside the loop the whole exercise is meant to make cheaper.

Deliberately narrow: the call is dropped only when **every** use of it is a
per-element intrinsic. A callback whose result is genuinely used elsewhere keeps
its call, and the pattern degrades to a redundant evaluation rather than to a
missing one.
"""
function coopmat_perelement_marker(inst::LLVM.CallInst)
    callee = LLVM.called_operand(inst)
    callee isa LLVM.Function || return false
    startswith(LLVM.name(callee), "_lava_") && return false
    n = 0
    for use in LLVM.uses(inst)
        user = LLVM.user(use)
        user isa LLVM.CallInst || return false
        target = LLVM.called_operand(user)
        target isa LLVM.Function || return false
        parsed = parse_coopmat_name(LLVM.name(target))
        # `perelemm` — the MATRIX-operand form — is a per-element op too. Matching
        # `perelem` exactly excluded it here and in `collect_inline_callbacks!`
        # below, so its callback was never marked `Inline` and the driver emitted
        # a real function call per element. That is the 8.5x documented on
        # `coopmat_perelement`, and it made the fused softmax measure +30% and
        # read as "fewer passes do not pay".
        (parsed !== nothing && startswith(parsed[1], "perelem")) || return false
        n += 1
    end
    return n > 0
end

"""
    collect_inline_callbacks!(state, llvm_mod) -> nothing

Record every function that an `OpCooperativeMatrixPerElementOpNV` will name.

Runs over the whole module before any function is emitted, because a callback is
emitted *ahead of* the entry function that names it and `emit_function!` has to
already know not to mark it `DontInline`. See `inline_callbacks`.
"""
function collect_inline_callbacks!(state::SPIRVEmitterState, llvm_mod::LLVM.Module)
    for fn in LLVM.functions(llvm_mod)
        isempty(LLVM.blocks(fn)) && continue
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = LLVM.called_operand(inst)
            callee isa LLVM.Function || continue
            parsed = parse_coopmat_name(LLVM.name(callee))
            parsed === nothing && continue
            # `reduce<mask>` too. `OpCooperativeMatrixReduceNV` names a combiner
            # the driver is meant to inline into its reduction loop for exactly
            # the same reason, and `coopmat_reduce`'s docstring already said so
            # — but nothing put it in this set, so every combiner was emitted
            # `DontInline` and each combine step was a real function call.
            (startswith(parsed[1], "perelem") || startswith(parsed[1], "reduce")) || continue
            # Operand 2 is the marker in all three forms: the scalar llvmcall is
            # `(i32 %m, T %p)`, the matrix one `(i32 %m, T %p, i32 %o)`, and the
            # reduce one `(i32 %m, T %p)`.
            marker = LLVM.operands(inst)[2]
            marker isa LLVM.CallInst || continue
            cb = LLVM.called_operand(marker)
            cb isa LLVM.Function && push!(state.inline_callbacks, cb)
        end
    end
    return nothing
end
