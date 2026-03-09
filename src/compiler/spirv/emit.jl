# SPIR-V Instruction Emitter: LLVM IR instruction → SPIR-V instruction translation.
#
# Translates ~33 LLVM opcodes to their SPIR-V equivalents.
# Operates on an LLVM Module via LLVM.jl API, emitting into an SPIRVModule.
#
# Key challenges handled here:
# 1. GEP → OpAccessChain/OpPtrAccessChain translation
# 2. PHI node deferred emission (must be at block start)
# 3. fcmp ord/uno → OpIsNan lowering (Vulkan doesn't support OpOrdered/OpUnordered)
# 4. LLVM intrinsics → GLSL.std.450 extended instructions

# ================================================================
# Emitter State
# ================================================================

"""
    SPIRVEmitterState

Tracks state during LLVM → SPIR-V emission.
"""
# Pre-computed loop info: maps header blocks → (merge block, continue block)
struct LoopInfo
    # header block → (merge block, continue block)
    loops::Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}
end

LoopInfo() = LoopInfo(Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}())

mutable struct SPIRVEmitterState
    # The SPIR-V module being built
    mod::SPIRVModule
    # Type context for LLVM → SPIR-V type mapping
    type_ctx::SPIRVTypeContext
    # LLVM Value → SPIR-V result ID
    value_map::Dict{LLVM.Value, UInt32}
    # LLVM BasicBlock → SPIR-V label ID
    block_map::Dict{LLVM.BasicBlock, UInt32}
    # Deferred PHI nodes: (spirv_id, type_id, [(value, block)], owning_block_label_id)
    deferred_phis::Vector{Tuple{UInt32, UInt32, Vector{Tuple{LLVM.Value, LLVM.BasicBlock}}, UInt32}}
    # Pre-computed loop info for current function
    loop_info::LoopInfo
    # Blocks already claimed as merge targets (a block can be merge for at most one header)
    used_merge_blocks::Set{UInt32}
    # Trampolines: maps (from_block_id, original_target_id) → trampoline_label_id
    trampolines::Dict{Tuple{UInt32, UInt32}, UInt32}
    # Immediate post-dominator tree: block → ipdom block (for merge block finding)
    ipdom::Dict{LLVM.BasicBlock, LLVM.BasicBlock}
    # cmpxchg compare value IDs: cmpxchg_inst → compare_value_spirv_id
    # Used by extractvalue to compute success flag (old == expected)
    cmpxchg_cmp_vals::Dict{LLVM.Value, UInt32}
    # Block redirects for PHI resolution: (original_block_id, target_block_id) → new_block_id
    # When a loop header is split into header + selection header, PHIs referencing
    # the original header as predecessor must reference the selection header instead.
    phi_block_redirects::Dict{Tuple{UInt32, UInt32}, UInt32}
    # Array element origin: when a value was produced by OpAccessChain indexing into an array,
    # maps LLVM inst → (alloca_base_spirv_id, static_path_indices, dyn_index_spirv_id, array_llvm_type)
    # The static_path contains the OpConstant IDs for struct field indices leading to the array.
    # Used to fold chained GEPs like array[a][b] → array[a+b] in Function storage class.
    array_element_origin::Dict{LLVM.Value, Tuple{UInt32, Vector{UInt32}, UInt32, LLVM.ArrayType}}
    # SPIR-V element type of pointer values: LLVM Value (pointer) → SPIR-V type ID of element.
    # Used to detect when a load's expected result type differs from the pointer's declared
    # element type (e.g., same LLVM struct type used for ptr<i64> and ptr<i16> members).
    spirv_ptr_element_type::Dict{LLVM.Value, UInt32}
    # ── RT shader state (set by _emit_spirv_from_llvm_rt, unused for compute) ──
    rt_payload_var_id::Union{Nothing, UInt32}
    rt_tlas_var_id::Union{Nothing, UInt32}
    rt_accel_type_id::Union{Nothing, UInt32}
    rt_payload_type::Symbol  # :f32, :struct, etc.
    rt_hit_attrib_var_id::Union{Nothing, UInt32}  # HitAttributeKHR variable (vec2 barycentrics)
    # Set true when OpIgnoreIntersectionKHR/OpTerminateRayKHR is emitted (block terminators).
    # Suppresses the redundant OpReturn from the trailing `ret void`.
    rt_block_terminated::Bool
    # ── Graphics shader state (set by _emit_spirv_from_llvm_gfx, unused for compute/RT) ──
    gfx_io::Any  # GfxIOState or nothing
    # ── PSB conversion cache ──
    # Block-local cache: (base_ptr_spirv_id, block_label_id) → u64_spirv_id
    # Avoids emitting redundant OpConvertPtrToU for repeated accesses to the same
    # PSB base pointer within a block. Keyed by block to prevent dominance errors.
    psb_ptr_to_u64::Dict{Tuple{UInt32, UInt32}, UInt32}
    # Current block label ID (updated when emitting each block)
    current_block_label::UInt32
    # PSB pointer byte offsets: maps LLVM pointer values to their known constant byte offset
    # from the base PSB address. Used to compute correct alignment for loads/stores when
    # SROA generates i64 accesses at non-8-aligned offsets (e.g., byte 12 from integer
    # arithmetic GEP path). Without this, we'd declare Aligned 8 for i64 at byte 12,
    # and NVIDIA GPUs round down to byte 8, corrupting data.
    psb_known_byte_offsets::Dict{LLVM.Value, Int64}
    # PSB pointer guaranteed alignment: maps LLVM pointer values to their minimum
    # guaranteed alignment (in bytes). Populated when runtime-indexed byte GEPs on
    # composite types (structs/arrays) have a stride that's not a multiple of 8
    # (e.g., 60-byte struct → 4-byte alignment). Propagated through constant-offset
    # GEP chains. Used by _psb_needs_decomposition() to detect i64 stores that need
    # decomposition into two i32 stores.
    psb_ptr_alignment::Dict{LLVM.Value, UInt32}
end

function SPIRVEmitterState(mod::SPIRVModule, type_ctx::SPIRVTypeContext)
    SPIRVEmitterState(
        mod, type_ctx,
        Dict{LLVM.Value, UInt32}(),
        Dict{LLVM.BasicBlock, UInt32}(),
        Tuple{UInt32, UInt32, Vector{Tuple{LLVM.Value, LLVM.BasicBlock}}, UInt32}[],
        LoopInfo(),
        Set{UInt32}(),
        Dict{Tuple{UInt32, UInt32}, UInt32}(),
        Dict{LLVM.BasicBlock, LLVM.BasicBlock}(),
        Dict{LLVM.Value, UInt32}(),
        Dict{Tuple{UInt32, UInt32}, UInt32}(),
        Dict{LLVM.Value, Tuple{UInt32, Vector{UInt32}, UInt32, LLVM.ArrayType}}(),
        Dict{LLVM.Value, UInt32}(),
        nothing, nothing, nothing, :none, nothing, false,
        nothing,  # gfx_io
        Dict{Tuple{UInt32, UInt32}, UInt32}(), UInt32(0),
        Dict{LLVM.Value, Int64}(),
        Dict{LLVM.Value, UInt32}(),
    )
end

"""
    _emit_psb_ptr_reinterpret!(state, target_ptr_ty_id, source_id) -> UInt32

Reinterpret a PhysicalStorageBuffer pointer to a different SPIR-V pointer type.

NVIDIA's RT shader compiler crashes on `OpBitcast` between PSB pointer types
(driver bug in libnvidia-glvkspirv.so). This function uses the equivalent
`OpConvertPtrToU` + `OpConvertUToPtr` roundtrip instead, which works on all vendors.

Returns `source_id` unchanged if the target type already matches (no-op).
"""
function _emit_psb_ptr_reinterpret!(state::SPIRVEmitterState, target_ptr_ty_id::UInt32,
                                     source_id::UInt32)
    u64_ty = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    tmp_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_ty, tmp_id, source_id)
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, target_ptr_ty_id, result_id, tmp_id)
    return result_id
end

"""
    get_value_id!(state, val) -> UInt32

Get the SPIR-V ID for an LLVM Value. Creates constants on-the-fly for ConstantInt/ConstantFP.
"""
function get_value_id!(state::SPIRVEmitterState, val::LLVM.Value)
    cached = get(state.value_map, val, nothing)
    cached !== nothing && return cached

    # Auto-create constants
    if val isa LLVM.ConstantInt || val isa LLVM.ConstantFP ||
       val isa LLVM.UndefValue || val isa LLVM.PoisonValue ||
       val isa LLVM.ConstantAggregateZero
        id = map_constant!(state.type_ctx, val)
        state.value_map[val] = id
        return id
    end

    # ConstantExpr — evaluate and map
    # NOTE: We do NOT cache ConstantExpr GEP results in value_map because the same
    # ConstantExpr object is reused across different LLVM blocks. If we cached the
    # SPIR-V AccessChain ID from block A and reused it in block B, we'd get a
    # domination error (the ID is defined in A but used in B without dominance).
    # Re-emitting is cheap (just OpAccessChain with constant indices).
    if val isa LLVM.ConstantExpr
        return _emit_constant_expr!(state, val)
    end

    # Julia runtime function declarations used as global variables (type tags):
    # e.g., @jl_int64_type is declared as a function but used as `load i64, ptr @jl_int64_type`.
    # These appear in error/boxing paths that should never execute on GPU.
    # Emit a zero constant of pointer-width as a safe fallback.
    if val isa LLVM.Function && LLVM.isintrinsic(val) == false && isempty(LLVM.blocks(val))
        # This is a declaration (no body) used as a value — Julia runtime type tag
        u64_ty = emit_type_int!(state.mod, UInt32(64), UInt32(0))
        zero_id = _emit_u64_constant!(state.mod, UInt64(0))
        state.value_map[val] = zero_id
        return zero_id
    end

    error("LLVM value not in value map and not a constant: $(typeof(val)) = $val")
end

"""
    _ensure_index_i32!(state, llvm_val) -> UInt32

Get the SPIR-V ID for an LLVM index value, ensuring it's 32-bit.
Vulkan SPIR-V requires OpAccessChain indices to be 32-bit integers.
If the source value is i64, inserts an OpUConvert to truncate to i32.
"""
function _ensure_index_i32!(state::SPIRVEmitterState, val::LLVM.Value)
    id = get_value_id!(state, val)

    # Check if the LLVM type is i64 (or wider) — if so, truncate to i32
    ty = LLVM.value_type(val)
    if ty isa LLVM.IntegerType && LLVM.width(ty) > 32
        # Insert OpUConvert: i64 → i32
        u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
        conv_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, conv_id, id)
        return conv_id
    end

    return id
end

function get_block_id!(state::SPIRVEmitterState, bb::LLVM.BasicBlock)
    get!(state.block_map, bb) do
        fresh_id!(state.mod)
    end
end

# ================================================================
# Block Ordering
# ================================================================

"""
    _reverse_postorder(fn::LLVM.Function) -> Vector{LLVM.BasicBlock}

Compute reverse post-order traversal of the function's CFG.
This ensures dominators are emitted before dominated blocks in SPIR-V,
preventing forward references in non-PHI instructions.
"""
function _reverse_postorder(fn::LLVM.Function)
    blocks = collect(LLVM.blocks(fn))
    isempty(blocks) && return blocks

    visited = Set{LLVM.BasicBlock}()
    postorder = LLVM.BasicBlock[]

    function dfs(bb)
        bb in visited && return
        push!(visited, bb)
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            dfs(succ)
        end
        push!(postorder, bb)
    end

    dfs(first(blocks))  # Start from entry block

    # Add any unreachable blocks not visited by DFS
    for bb in blocks
        if !(bb in visited)
            push!(postorder, bb)
        end
    end

    return reverse(postorder)
end

# ================================================================
# Immediate Post-Dominator Tree
# ================================================================

"""
    _compute_ipostdom(fn::LLVM.Function) -> Dict{LLVM.BasicBlock, LLVM.BasicBlock}

Compute the immediate post-dominator for each basic block using the iterative
algorithm (Cooper, Harvey, Kennedy). The ipdom of a selection header is the
correct merge block for SPIR-V's OpSelectionMerge.
"""
function _compute_ipostdom(fn::LLVM.Function)
    blocks = collect(LLVM.blocks(fn))
    isempty(blocks) && return Dict{LLVM.BasicBlock, LLVM.BasicBlock}()

    rpo = _reverse_postorder(fn)

    # For post-dominators, we need indices where the EXIT has the LOWEST number.
    # Use post-order of forward graph (= reverse of RPO) as the numbering.
    # This way, "walking up the tree" goes toward lower numbers (toward root/exit).
    po = reverse(rpo)
    po_idx = Dict{LLVM.BasicBlock, Int}()
    for (i, bb) in enumerate(po)
        po_idx[bb] = i
    end

    # Find exit blocks (ret/unreachable/no successors)
    exits = LLVM.BasicBlock[]
    for bb in blocks
        term = LLVM.terminator(bb)
        if term isa LLVM.RetInst || term isa LLVM.UnreachableInst
            push!(exits, bb)
        else
            succs = collect(LLVM.successors(term))
            isempty(succs) && push!(exits, bb)
        end
    end

    # If no exit found, use the last block in RPO (= first in PO) as exit
    isempty(exits) && push!(exits, po[1])

    # Initialize: exits post-dominate themselves
    ipdom = Dict{LLVM.BasicBlock, LLVM.BasicBlock}()
    for bb in exits
        ipdom[bb] = bb
    end

    # Cooper-Harvey-Kennedy intersect using post-order indices.
    # Walking "up" the tree means going toward LOWER indices (toward exit root).
    function intersect_doms(b1, b2)
        finger1 = b1
        finger2 = b2
        while finger1 !== finger2
            while po_idx[finger1] > po_idx[finger2]
                next = ipdom[finger1]
                next === finger1 && @goto done  # at root
                finger1 = next
            end
            while po_idx[finger2] > po_idx[finger1]
                next = ipdom[finger2]
                next === finger2 && @goto done  # at root
                finger2 = next
            end
        end
        @label done
        return finger1
    end

    changed = true
    while changed
        changed = false
        # Process in post-order of forward graph (exit first, entry last).
        # This is the correct order for post-dominator convergence.
        for bb in po
            any(bb === e for e in exits) && continue

            # Collect successors that already have an ipdom
            term = LLVM.terminator(bb)
            succs = collect(LLVM.successors(term))
            processed = [s for s in succs if haskey(ipdom, s)]
            isempty(processed) && continue

            new_ipdom = processed[1]
            for s in processed[2:end]
                new_ipdom = intersect_doms(new_ipdom, s)
            end

            if !haskey(ipdom, bb) || ipdom[bb] !== new_ipdom
                ipdom[bb] = new_ipdom
                changed = true
            end
        end
    end

    return ipdom
end

# ================================================================
# Main Emission Entry Point
# ================================================================

"""
    emit_function!(state::SPIRVEmitterState, fn::LLVM.Function; is_entry::Bool=false)

Emit a single LLVM function as a SPIR-V function.
"""
function emit_function!(state::SPIRVEmitterState, fn::LLVM.Function; is_entry::Bool=false)
    fn_ty = LLVM.function_type(fn)
    ret_ty = LLVM.return_type(fn_ty)
    ret_spirv = map_type!(state.type_ctx, ret_ty)

    # Map parameter types — use actual parameter values to resolve pointer types
    param_spirv = UInt32[]
    for param in LLVM.parameters(fn)
        param_ty = LLVM.value_type(param)
        if param_ty isa LLVM.PointerType
            push!(param_spirv, map_pointer_type_for_value!(state.type_ctx, param))
        else
            push!(param_spirv, map_type!(state.type_ctx, param_ty))
        end
    end
    func_type_id = emit_type_function!(state.mod, ret_spirv, param_spirv)

    # Function ID
    func_id = get!(state.value_map, fn) do
        fresh_id!(state.mod)
    end

    # Function control
    fc = is_entry ? FuncControl.None : FuncControl.None
    # TODO: Use DontInline for @noinline functions

    # OpFunction
    encode_instruction!(state.mod.functions, Op.OpFunction, ret_spirv, func_id, fc, func_type_id)

    # OpFunctionParameter for each parameter
    for param in LLVM.parameters(fn)
        param_id = fresh_id!(state.mod)
        state.value_map[param] = param_id
        param_ty = LLVM.value_type(param)
        if param_ty isa LLVM.PointerType
            # For pointer params, use the pointer type from PointeeTypeMap
            spirv_ty = map_pointer_type_for_value!(state.type_ctx, param)
        else
            spirv_ty = map_type!(state.type_ctx, param_ty)
        end
        encode_instruction!(state.mod.functions, Op.OpFunctionParameter, spirv_ty, param_id)
    end

    # Pre-allocate block labels and PHI result IDs
    for bb in LLVM.blocks(fn)
        get_block_id!(state, bb)
        for inst in LLVM.instructions(bb)
            if inst isa LLVM.PHIInst
                phi_id = fresh_id!(state.mod)
                state.value_map[inst] = phi_id
            end
        end
    end

    # Pre-analyze loops to identify headers and their merge/continue targets
    state.loop_info = _analyze_loops(fn)

    # Compute immediate post-dominator tree for merge block finding
    state.ipdom = _compute_ipostdom(fn)

    # Pre-register loop merge blocks as claimed (loop merges are emitted first)
    empty!(state.used_merge_blocks)
    empty!(state.trampolines)
    for (header, (merge_bb, continue_bb)) in state.loop_info.loops
        merge_id = get_block_id!(state, merge_bb)
        push!(state.used_merge_blocks, merge_id)
    end

    # SPIR-V requires all Function-scope OpVariable at the start of the entry block.
    # Pre-emit all allocas into a temporary buffer, register them in value_map,
    # then inject the preamble right after the entry block's OpLabel during emission.
    all_allocas = LLVM.AllocaInst[]
    for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        inst isa LLVM.AllocaInst && push!(all_allocas, inst)
    end
    # Emit allocas to a temporary buffer (not state.mod.functions)
    preamble_words = UInt32[]
    if !isempty(all_allocas)
        # Temporarily swap functions buffer to capture preamble
        real_buf = state.mod.functions
        state.mod.functions = preamble_words
        for alloca_inst in all_allocas
            _emit_alloca!(state, alloca_inst)
        end
        state.mod.functions = real_buf
    end

    # Emit basic blocks in reverse post-order (dominators before dominated blocks).
    # After StructurizeCFG, LLVM block order may not respect dominance, causing
    # forward references in non-PHI instructions.
    rpo_blocks = _reverse_postorder(fn)
    entry_label_emitted = false
    entry_label_pos = 0
    for bb in rpo_blocks
        if !entry_label_emitted
            entry_label_pos = length(state.mod.functions) + 1  # OpLabel will be emitted here
        end
        emit_block!(state, bb)
        # After the entry block is emitted (first block), inject OpVariable preamble
        # right after the OpLabel (which is the first instruction emitted by emit_block!).
        if !entry_label_emitted
            entry_label_emitted = true
            if !isempty(preamble_words)
                # The entry block's OpLabel was emitted at `entry_label_pos`.
                # OpLabel is 2 words. Insert preamble at entry_label_pos + 2.
                insert_pos = entry_label_pos + 2
                # Use a tail copy to avoid overlapping copyto! issues
                tail_words = state.mod.functions[insert_pos:end]
                n_preamble = length(preamble_words)
                resize!(state.mod.functions, insert_pos - 1 + n_preamble + length(tail_words))
                copyto!(state.mod.functions, insert_pos, preamble_words, 1, n_preamble)
                copyto!(state.mod.functions, insert_pos + n_preamble, tail_words, 1, length(tail_words))
            end
        end
        # Emit any trampolines targeting the NEXT block in RPO order.
        # Trampolines must appear before the blocks that reference them.
        _emit_pending_trampolines!(state, bb, rpo_blocks)
    end

    # Emit any remaining trampolines not yet emitted
    _emit_remaining_trampolines!(state)

    # Now resolve deferred PHIs — insert at correct block positions
    _resolve_deferred_phis!(state)

    # OpFunctionEnd
    encode_instruction!(state.mod.functions, Op.OpFunctionEnd)

    return func_id
end

"""
    emit_block!(state::SPIRVEmitterState, bb::LLVM.BasicBlock)

Emit a basic block as OpLabel + instructions.
"""
function emit_block!(state::SPIRVEmitterState, bb::LLVM.BasicBlock)
    label_id = get_block_id!(state, bb)

    if get(ENV, "LAVA_DEBUG_PHI", "") == "1"
        println("  EMIT BLOCK: $(String(LLVM.name(bb))) → SPIR-V %$label_id")
    end

    encode_instruction!(state.mod.functions, Op.OpLabel, label_id)

    # Track current block for PSB conversion cache
    state.current_block_label = label_id

    # Reset RT block termination flag for this block
    state.rt_block_terminated = false

    # PHI nodes: defer to after all blocks are emitted (operands may be forward references)
    # Their result IDs are already pre-allocated in value_map.
    for inst in LLVM.instructions(bb)
        if inst isa LLVM.PHIInst
            _defer_phi!(state, inst, label_id)
        end
    end

    # Emit non-PHI, non-terminator instructions
    insts = collect(LLVM.instructions(bb))
    for inst in insts
        inst isa LLVM.PHIInst && continue
        # Terminator (branch/ret) is handled specially for loop headers
        if LLVM.isterminator(inst)
            # If this is a loop header, emit OpLoopMerge right before the terminator
            loop_entry = get(state.loop_info.loops, bb, nothing)
            if loop_entry !== nothing
                merge_bb, continue_bb = loop_entry
                merge_id = get_block_id!(state, merge_bb)
                continue_id = get_block_id!(state, continue_bb)
                encode_instruction!(state.mod.functions, Op.OpLoopMerge, merge_id, continue_id, UInt32(0))
            end
        end
        emit_instruction!(state, inst)
    end
end

# ================================================================
# Instruction Dispatch
# ================================================================

function emit_instruction!(state::SPIRVEmitterState, inst::LLVM.Instruction)
    # Dispatch on instruction type
    if inst isa LLVM.RetInst
        _emit_ret!(state, inst)
    elseif inst isa LLVM.BrInst
        _emit_br!(state, inst)
    # Arithmetic — Float
    elseif inst isa LLVM.FAddInst
        _emit_binary_op!(state, inst, Op.OpFAdd)
    elseif inst isa LLVM.FSubInst
        _emit_binary_op!(state, inst, Op.OpFSub)
    elseif inst isa LLVM.FMulInst
        _emit_binary_op!(state, inst, Op.OpFMul)
    elseif inst isa LLVM.FDivInst
        _emit_binary_op!(state, inst, Op.OpFDiv)
    elseif inst isa LLVM.FRemInst
        _emit_binary_op!(state, inst, Op.OpFRem)
    elseif inst isa LLVM.FNegInst
        _emit_unary_op!(state, inst, Op.OpFNegate)
    # Arithmetic — Integer
    elseif inst isa LLVM.AddInst
        _emit_binary_op!(state, inst, Op.OpIAdd)
    elseif inst isa LLVM.SubInst
        _emit_binary_op!(state, inst, Op.OpISub)
    elseif inst isa LLVM.MulInst
        _emit_binary_op!(state, inst, Op.OpIMul)
    elseif inst isa LLVM.SDivInst
        _emit_binary_op!(state, inst, Op.OpSDiv)
    elseif inst isa LLVM.UDivInst
        _emit_binary_op!(state, inst, Op.OpUDiv)
    elseif inst isa LLVM.SRemInst
        _emit_binary_op!(state, inst, Op.OpSRem)
    elseif inst isa LLVM.URemInst
        _emit_binary_op!(state, inst, Op.OpUMod)
    # Bitwise / Logical (i1 operands use OpLogical*, int uses OpBitwise*)
    elseif inst isa LLVM.AndInst
        _emit_bitwise_or_logical!(state, inst, Op.OpBitwiseAnd, Op.OpLogicalAnd)
    elseif inst isa LLVM.OrInst
        _emit_bitwise_or_logical!(state, inst, Op.OpBitwiseOr, Op.OpLogicalOr)
    elseif inst isa LLVM.XorInst
        _emit_xor!(state, inst)
    elseif inst isa LLVM.ShlInst
        _emit_binary_op!(state, inst, Op.OpShiftLeftLogical)
    elseif inst isa LLVM.LShrInst
        _emit_binary_op!(state, inst, Op.OpShiftRightLogical)
    elseif inst isa LLVM.AShrInst
        _emit_binary_op!(state, inst, Op.OpShiftRightArithmetic)
    # Comparisons
    elseif inst isa LLVM.ICmpInst
        _emit_icmp!(state, inst)
    elseif inst isa LLVM.FCmpInst
        _emit_fcmp!(state, inst)
    # Memory
    elseif inst isa LLVM.LoadInst
        _emit_load!(state, inst)
    elseif inst isa LLVM.StoreInst
        _emit_store!(state, inst)
    elseif inst isa LLVM.GetElementPtrInst
        _emit_gep!(state, inst)
    elseif inst isa LLVM.AllocaInst
        # Allocas are emitted as OpVariable at the start of the entry block
        # (see emit_function!). Skip them during normal block emission.
        if !haskey(state.value_map, inst)
            _emit_alloca!(state, inst)
        end
    # Conversions
    elseif inst isa LLVM.SExtInst
        _emit_sext!(state, inst)
    elseif inst isa LLVM.ZExtInst
        _emit_zext!(state, inst)
    elseif inst isa LLVM.TruncInst
        _emit_trunc!(state, inst)
    elseif inst isa LLVM.FPExtInst
        _emit_conversion!(state, inst, Op.OpFConvert)
    elseif inst isa LLVM.FPTruncInst
        _emit_conversion!(state, inst, Op.OpFConvert)
    elseif inst isa LLVM.SIToFPInst
        _emit_conversion!(state, inst, Op.OpConvertSToF)
    elseif inst isa LLVM.UIToFPInst
        _emit_conversion!(state, inst, Op.OpConvertUToF)
    elseif inst isa LLVM.FPToSIInst
        _emit_conversion!(state, inst, Op.OpConvertFToS)
    elseif inst isa LLVM.FPToUIInst
        _emit_conversion!(state, inst, Op.OpConvertFToU)
    elseif inst isa LLVM.BitCastInst
        _emit_conversion!(state, inst, Op.OpBitcast)
    elseif inst isa LLVM.PtrToIntInst
        _emit_conversion!(state, inst, Op.OpConvertPtrToU)
    elseif inst isa LLVM.IntToPtrInst
        _emit_inttoptr!(state, inst)
    elseif inst isa LLVM.AddrSpaceCastInst
        _emit_conversion!(state, inst, Op.OpBitcast)
    # Control flow
    elseif inst isa LLVM.SelectInst
        _emit_select!(state, inst)
    # Function calls
    elseif inst isa LLVM.CallInst
        _emit_call!(state, inst)
    # Aggregate operations
    elseif inst isa LLVM.ExtractValueInst
        _emit_extractvalue!(state, inst)
    elseif inst isa LLVM.InsertValueInst
        _emit_insertvalue!(state, inst)
    # Vector operations
    elseif inst isa LLVM.ExtractElementInst
        _emit_extractelement!(state, inst)
    elseif inst isa LLVM.InsertElementInst
        _emit_insertelement!(state, inst)
    # Atomics
    elseif inst isa LLVM.AtomicRMWInst
        _emit_atomicrmw!(state, inst)
    elseif inst isa LLVM.AtomicCmpXchgInst
        _emit_cmpxchg!(state, inst)
    # Freeze — treat as no-op (pass through operand)
    elseif inst isa LLVM.FreezeInst
        ops = LLVM.operands(inst)
        state.value_map[inst] = get_value_id!(state, ops[1])
    # Unreachable — should have been removed by passes
    elseif inst isa LLVM.UnreachableInst
        encode_instruction!(state.mod.functions, Op.OpReturn)
    else
        error("Unsupported LLVM instruction: $(typeof(inst)): $inst")
    end
end

# ================================================================
# Binary / Unary Operations
# ================================================================

function _emit_binary_op!(state::SPIRVEmitterState, inst::LLVM.Instruction, opcode::UInt16)
    ops = LLVM.operands(inst)
    lhs = get_value_id!(state, ops[1])
    rhs = get_value_id!(state, ops[2])
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, opcode, result_ty, result_id, lhs, rhs)
    state.value_map[inst] = result_id
end

function _emit_unary_op!(state::SPIRVEmitterState, inst::LLVM.Instruction, opcode::UInt16)
    ops = LLVM.operands(inst)
    operand = get_value_id!(state, ops[1])
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, opcode, result_ty, result_id, operand)
    state.value_map[inst] = result_id
end

function _is_bool_type(ty::LLVM.LLVMType)
    ty isa LLVM.IntegerType && LLVM.width(ty) == 1
end

function _emit_bitwise_or_logical!(state::SPIRVEmitterState, inst::LLVM.Instruction,
                                    bitwise_op::UInt16, logical_op::UInt16)
    result_ty = LLVM.value_type(inst)
    opcode = _is_bool_type(result_ty) ? logical_op : bitwise_op
    _emit_binary_op!(state, inst, opcode)
end

function _emit_xor!(state::SPIRVEmitterState, inst::LLVM.Instruction)
    result_ty = LLVM.value_type(inst)
    if _is_bool_type(result_ty)
        # xor i1 %a, true → OpLogicalNot; xor i1 %a, %b → OpLogicalNotEqual
        ops = LLVM.operands(inst)
        rhs = ops[2]
        if rhs isa LLVM.ConstantInt && convert(Int64, rhs) != 0
            # xor i1 %a, true = logical not
            src = get_value_id!(state, ops[1])
            spirv_ty = map_type!(state.type_ctx, result_ty)
            result_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpLogicalNot, spirv_ty, result_id, src)
            state.value_map[inst] = result_id
        else
            _emit_binary_op!(state, inst, Op.OpLogicalNotEqual)
        end
    else
        _emit_binary_op!(state, inst, Op.OpBitwiseXor)
    end
end

# ================================================================
# Comparisons
# ================================================================

# ICmp predicate → SPIR-V opcode mapping
const ICMP_OPCODE_MAP = Dict{LLVM.API.LLVMIntPredicate, UInt16}(
    LLVM.API.LLVMIntEQ  => Op.OpIEqual,
    LLVM.API.LLVMIntNE  => Op.OpINotEqual,
    LLVM.API.LLVMIntSGT => Op.OpSGreaterThan,
    LLVM.API.LLVMIntSGE => Op.OpSGreaterThanEqual,
    LLVM.API.LLVMIntSLT => Op.OpSLessThan,
    LLVM.API.LLVMIntSLE => Op.OpSLessThanEqual,
    LLVM.API.LLVMIntUGT => Op.OpUGreaterThan,
    LLVM.API.LLVMIntUGE => Op.OpUGreaterThanEqual,
    LLVM.API.LLVMIntULT => Op.OpULessThan,
    LLVM.API.LLVMIntULE => Op.OpULessThanEqual,
)

function _emit_icmp!(state::SPIRVEmitterState, inst::LLVM.ICmpInst)
    pred = LLVM.predicate(inst)
    opcode = get(ICMP_OPCODE_MAP, pred, nothing)
    if opcode === nothing
        error("Unsupported ICmp predicate: $pred")
    end

    ops = LLVM.operands(inst)
    lhs_val = ops[1]
    rhs_val = ops[2]
    lhs = get_value_id!(state, lhs_val)
    rhs = get_value_id!(state, rhs_val)

    # SPIR-V requires both operands to have the same type.
    # LLVM constants may be i32 when the other operand is i8/i64/etc.
    # Insert OpUConvert or OpSConvert if widths don't match.
    lhs_ty = LLVM.value_type(lhs_val)
    rhs_ty = LLVM.value_type(rhs_val)
    if lhs_ty isa LLVM.IntegerType && rhs_ty isa LLVM.IntegerType
        lw = LLVM.width(lhs_ty)
        rw = LLVM.width(rhs_ty)
        if lw != rw
            # Widen the narrower operand to match the wider one
            target_w = max(lw, rw)
            target_ty = map_type!(state.type_ctx, LLVM.IntType(target_w))
            if lw < rw
                conv_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpUConvert, target_ty, conv_id, lhs)
                lhs = conv_id
            else
                conv_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpUConvert, target_ty, conv_id, rhs)
                rhs = conv_id
            end
        end
    end

    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))  # i1 → OpTypeBool
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, opcode, result_ty, result_id, lhs, rhs)
    state.value_map[inst] = result_id
end

# FCmp predicate → SPIR-V opcode mapping
# Vulkan Shader model does NOT support OpOrdered/OpUnordered — these need special handling
const FCMP_OPCODE_MAP = Dict{LLVM.API.LLVMRealPredicate, UInt16}(
    LLVM.API.LLVMRealOEQ => Op.OpFOrdEqual,
    LLVM.API.LLVMRealOGT => Op.OpFOrdGreaterThan,
    LLVM.API.LLVMRealOGE => Op.OpFOrdGreaterThanEqual,
    LLVM.API.LLVMRealOLT => Op.OpFOrdLessThan,
    LLVM.API.LLVMRealOLE => Op.OpFOrdLessThanEqual,
    LLVM.API.LLVMRealONE => Op.OpFOrdNotEqual,
    LLVM.API.LLVMRealUEQ => Op.OpFUnordEqual,
    LLVM.API.LLVMRealUGT => Op.OpFUnordGreaterThan,
    LLVM.API.LLVMRealUGE => Op.OpFUnordGreaterThanEqual,
    LLVM.API.LLVMRealULT => Op.OpFUnordLessThan,
    LLVM.API.LLVMRealULE => Op.OpFUnordLessThanEqual,
    LLVM.API.LLVMRealUNE => Op.OpFUnordNotEqual,
)

function _emit_fcmp!(state::SPIRVEmitterState, inst::LLVM.FCmpInst)
    pred = LLVM.predicate(inst)
    ops = LLVM.operands(inst)
    lhs = get_value_id!(state, ops[1])
    rhs = get_value_id!(state, ops[2])
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))

    if pred == LLVM.API.LLVMRealORD
        # fcmp ord x, y → NOT(IsNan(x) OR IsNan(y))
        _emit_fcmp_ord!(state, inst, lhs, rhs, result_ty)
        return
    elseif pred == LLVM.API.LLVMRealUNO
        # fcmp uno x, y → IsNan(x) OR IsNan(y)
        _emit_fcmp_uno!(state, inst, lhs, rhs, result_ty)
        return
    end

    opcode = get(FCMP_OPCODE_MAP, pred, nothing)
    if opcode === nothing
        error("Unsupported FCmp predicate: $pred")
    end

    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, opcode, result_ty, result_id, lhs, rhs)
    state.value_map[inst] = result_id
end

function _emit_fcmp_ord!(state::SPIRVEmitterState, inst, lhs, rhs, bool_ty)
    # ord x, y → NOT(IsNan(x) OR IsNan(y))
    nan_x = fresh_id!(state.mod)
    nan_y = fresh_id!(state.mod)
    nan_or = fresh_id!(state.mod)
    result_id = fresh_id!(state.mod)
    # OpIsNan (opcode 156)
    encode_instruction!(state.mod.functions, UInt16(156), bool_ty, nan_x, lhs)
    encode_instruction!(state.mod.functions, UInt16(156), bool_ty, nan_y, rhs)
    encode_instruction!(state.mod.functions, Op.OpLogicalOr, bool_ty, nan_or, nan_x, nan_y)
    encode_instruction!(state.mod.functions, Op.OpLogicalNot, bool_ty, result_id, nan_or)
    state.value_map[inst] = result_id
end

function _emit_fcmp_uno!(state::SPIRVEmitterState, inst, lhs, rhs, bool_ty)
    # uno x, y → IsNan(x) OR IsNan(y)
    nan_x = fresh_id!(state.mod)
    nan_y = fresh_id!(state.mod)
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, UInt16(156), bool_ty, nan_x, lhs)
    encode_instruction!(state.mod.functions, UInt16(156), bool_ty, nan_y, rhs)
    encode_instruction!(state.mod.functions, Op.OpLogicalOr, bool_ty, result_id, nan_x, nan_y)
    state.value_map[inst] = result_id
end

# ================================================================
# Memory Operations
# ================================================================

function _emit_load!(state::SPIRVEmitterState, inst::LLVM.LoadInst)
    ptr = LLVM.operands(inst)[1]
    load_ty = LLVM.value_type(inst)

    ptr_id = get_value_id!(state, ptr)
    # When loading a pointer value (e.g., extracting a Ptr{T} from a struct),
    # the result is a typed pointer in SPIR-V. Use the PTM to resolve.
    result_ty = if load_ty isa LLVM.PointerType
        map_pointer_type_for_value!(state.type_ctx, inst)
    else
        map_type!(state.type_ctx, load_ty)
    end

    # Handle struct-pointer mismatch: SROA may optimize
    #   load {i64}, ptr @global  +  extractvalue 0  →  load i64, ptr @global
    # When the pointer's declared pointee is a struct but the load type is a field,
    # insert an OpAccessChain(index=0) to get a pointer to the first field.
    # Returns 3-tuple: (ptr_id, actual_load_ty, ptr_load_override)
    # - actual_load_ty: non-pointer type mismatch (load this type, bitcast to load_ty)
    # - ptr_load_override: SPIR-V type ID for pointer loads where struct member type
    #   differs from usage-inferred type (load this type, bitcast to result_ty)
    ptr_id, actual_load_ty, ptr_load_override = _resolve_struct_field_load!(state, ptr, ptr_id, load_ty, result_ty)

    # Determine if we need to load a different type and bitcast
    needs_bitcast = actual_load_ty !== nothing && actual_load_ty != load_ty
    needs_ptr_bitcast = ptr_load_override !== nothing && ptr_load_override != result_ty
    spirv_load_ty = if needs_ptr_bitcast
        ptr_load_override
    elseif needs_bitcast
        map_type!(state.type_ctx, actual_load_ty)
    else
        result_ty
    end

    load_id = fresh_id!(state.mod)

    # PhysicalStorageBuffer loads MUST have Aligned memory operand
    if _is_psb_pointer(ptr)
        actual_load = needs_bitcast ? actual_load_ty : load_ty
        pointee_ty_ld = get_pointee_type(state.type_ctx.ptm, ptr)

        # Check if this wide load needs decomposition due to misaligned PSB address.
        # SPIR-V requires Aligned ≥ scalar_size for PSB (VUID 06314), so we can't
        # just lower alignment. Instead, decompose i64/double loads at non-8-aligned
        # addresses into two i32 loads with Aligned 4.
        llvm_align = UInt32(LLVM.alignment(inst))
        if _psb_needs_decomposition(state, ptr, actual_load; llvm_align)
            load_id = _emit_psb_decomposed_load!(state, ptr_id, actual_load, spirv_load_ty)
            # Decomposed load returns i64 integer. If original load was a pointer,
            # convert the integer to a typed PSB pointer.
            if actual_load isa LLVM.PointerType
                int_id = load_id
                load_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ty, load_id, int_id)
            end
        else
            align_ty = needs_bitcast ? actual_load_ty : load_ty
            align = _get_alignment_for_type(align_ty)

            # Special case: loading ptr from PSB where pointee was mapped to i64
            # (from _emit_psb_byte_offset_with_user_type!). Load i64, then ConvertUToPtr.
            psb_ptr_as_i64 = false
            if actual_load isa LLVM.PointerType && pointee_ty_ld !== nothing &&
               pointee_ty_ld isa LLVM.IntegerType && LLVM.width(pointee_ty_ld) == 64
                psb_ptr_as_i64 = true
                align = UInt32(8)
            elseif pointee_ty_ld !== nothing && actual_load != pointee_ty_ld &&
               !(actual_load isa LLVM.PointerType) && !(pointee_ty_ld isa LLVM.PointerType)
                ld_spirv_ty = map_type!(state.type_ctx, actual_load)
                ld_ptr_ty = map_pointer_type!(state.type_ctx, ld_spirv_ty, SC.PhysicalStorageBuffer)
                ptr_id = _emit_psb_ptr_reinterpret!(state, ld_ptr_ty, ptr_id)
            end

            if psb_ptr_as_i64
                # Load as i64, then convert to typed pointer
                i64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
                raw_load_id = fresh_id!(state.mod)
                push!(state.mod.functions, (UInt32(6) << 16) | UInt32(Op.OpLoad))
                push!(state.mod.functions, i64_spirv)
                push!(state.mod.functions, raw_load_id)
                push!(state.mod.functions, ptr_id)
                push!(state.mod.functions, UInt32(0x02))  # Aligned
                push!(state.mod.functions, align)
                # Convert i64 → typed PSB pointer
                encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ty, load_id, raw_load_id)
            else
                # OpLoad: result_type, result_id, ptr, memory_operand, alignment
                push!(state.mod.functions, (UInt32(6) << 16) | UInt32(Op.OpLoad))
                push!(state.mod.functions, spirv_load_ty)
                push!(state.mod.functions, load_id)
                push!(state.mod.functions, ptr_id)
                push!(state.mod.functions, UInt32(0x02))  # Aligned
                push!(state.mod.functions, align)
            end
        end
    else
        sc = _get_pointer_storage_class(ptr)
        if sc == SC.Workgroup || sc == SC.Function
            # For Workgroup/Function loads: if load type doesn't match pointer's pointee type,
            # bitcast the POINTER to match the load type. This happens when:
            # - byte-offset GEPs produce i8* but the actual field may be i64
            # - GEP chains access sub-elements (e.g. i32 within [16 x i64])
            pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
            if pointee_ty !== nothing
                eff_load_ty = needs_bitcast ? actual_load_ty : load_ty
                if eff_load_ty != pointee_ty && eff_load_ty isa LLVM.IntegerType && pointee_ty isa LLVM.IntegerType
                    val_spirv_ty = map_type!(state.type_ctx, eff_load_ty)
                    new_ptr_ty = map_pointer_type!(state.type_ctx, val_spirv_ty, sc)
                    cast_id = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpBitcast, new_ptr_ty, cast_id, ptr_id)
                    ptr_id = cast_id
                end
            end
        end
        # For pointer loads: check if the pointer's declared SPIR-V element type differs
        # from what we want to load. This happens when the same LLVM struct type is used
        # for pointer members with different element types (e.g., ptr<i64> for indices and
        # ptr<i16> for values share the same `{ ptr, [1 x i64] }` struct).
        if load_ty isa LLVM.PointerType
            declared_elem_ty = get(state.spirv_ptr_element_type, ptr, UInt32(0))
            if declared_elem_ty != UInt32(0) && declared_elem_ty != spirv_load_ty
                # Load as the declared type, then bitcast the result
                spirv_load_ty = declared_elem_ty
                needs_ptr_bitcast = true
            end
        end
        encode_instruction!(state.mod.functions, Op.OpLoad, spirv_load_ty, load_id, ptr_id)
    end

    if needs_ptr_bitcast
        # Pointer type mismatch: struct member is Ptr{Float32} but usage needs Ptr{i32}
        # OpBitcast between pointer types of same storage class
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, result_ty, result_id, load_id)
        state.value_map[inst] = result_id
    elseif needs_bitcast
        # Bitcast from actual_load_ty to load_ty (e.g., float → i32)
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, result_ty, result_id, load_id)
        state.value_map[inst] = result_id
    else
        state.value_map[inst] = load_id
    end
end

"""
When the load type doesn't match the pointer's declared pointee type in SPIR-V,
drill through nested structs/arrays using index 0 at each level until we reach
the matching type. Emits OpAccessChain with the computed index path.

This handles SROA optimizations where LLVM collapses GEPs for offset-0 accesses:
  `load {T}, ptr %p` + `extractvalue 0` → `load T, ptr %p`

Returns `(ptr_id, actual_load_ty)` where `actual_load_ty` is the type at the
resolved pointer (may differ from `load_ty` if a bitcast is needed, e.g.,
float vs i32). Returns `(ptr_id, nothing)` if no type resolution was done.
"""
function _resolve_struct_field_load!(state::SPIRVEmitterState, ptr::LLVM.Value,
                                      ptr_id::UInt32, load_ty::LLVM.LLVMType,
                                      result_spirv_ty::UInt32=UInt32(0))
    pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
    pointee_ty === nothing && return (ptr_id, nothing, nothing)

    # For pointer loads from struct bases: "load ptr, ptr %struct_alloca"
    # Drill through nested structs/arrays to reach the first pointer member.
    if load_ty isa LLVM.PointerType
        pointee_ty isa LLVM.PointerType && return (ptr_id, nothing, nothing)  # Already a pointer-to-pointer
        # Find the path to the first pointer member at offset 0
        index_path = _find_zero_index_path_to_pointer(pointee_ty)
        if index_path !== nothing && !isempty(index_path)
            sc = _get_pointer_storage_class(ptr)
            # Use the struct's DECLARED member pointer type for the AccessChain,
            # NOT the usage-inferred type. Both counter (i32 atomics) and A (float loads)
            # may share the same struct type, so the AccessChain must match the struct def.
            struct_member_spirv_ty = _get_struct_member_ptr_spirv_type(state, pointee_ty)
            element_spirv_ty = struct_member_spirv_ty !== nothing ? struct_member_spirv_ty : result_spirv_ty
            field_ptr_ty = map_pointer_type!(state.type_ctx, element_spirv_ty, sc)
            zero_id = emit_constant_u32!(state.mod, UInt32(0))
            ac_id = fresh_id!(state.mod)
            word_count = UInt32(4 + length(index_path))
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
            push!(state.mod.functions, field_ptr_ty)
            push!(state.mod.functions, ac_id)
            push!(state.mod.functions, ptr_id)
            for _ in index_path
                push!(state.mod.functions, zero_id)
            end
            # If struct member type differs from usage-inferred type, caller must bitcast
            ptr_load_override = (struct_member_spirv_ty !== nothing && struct_member_spirv_ty != result_spirv_ty) ? element_spirv_ty : nothing
            return (ac_id, nothing, ptr_load_override)
        end
        return (ptr_id, nothing, nothing)
    end

    pointee_ty == load_ty && return (ptr_id, nothing, nothing)  # Types match, no fixup needed

    # Compute zero-index path from pointee type to load type
    index_path = _find_zero_index_path(pointee_ty, load_ty)
    if index_path !== nothing
        # Exact type match found — emit OpAccessChain
        sc = _get_pointer_storage_class(ptr)
        field_spirv_ty = map_type!(state.type_ctx, load_ty)
        field_ptr_ty = map_pointer_type!(state.type_ctx, field_spirv_ty, sc)
        zero_id = emit_constant_u32!(state.mod, UInt32(0))
        ac_id = fresh_id!(state.mod)
        word_count = UInt32(4 + length(index_path))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
        push!(state.mod.functions, field_ptr_ty)
        push!(state.mod.functions, ac_id)
        push!(state.mod.functions, ptr_id)
        for _ in index_path
            push!(state.mod.functions, zero_id)
        end
        return (ac_id, nothing, nothing)
    end

    # No exact match. Try to find a same-sized leaf type (e.g., i32 vs float).
    # Navigate to the scalar leaf and check if sizes match.
    leaf_path, leaf_ty = _find_zero_index_path_to_leaf(pointee_ty)
    if leaf_path !== nothing && leaf_ty !== nothing && _types_same_size(leaf_ty, load_ty)
        # Navigate to the leaf, then the caller will bitcast
        sc = _get_pointer_storage_class(ptr)
        leaf_spirv_ty = map_type!(state.type_ctx, leaf_ty)
        leaf_ptr_ty = map_pointer_type!(state.type_ctx, leaf_spirv_ty, sc)
        zero_id = emit_constant_u32!(state.mod, UInt32(0))
        ac_id = fresh_id!(state.mod)
        word_count = UInt32(4 + length(leaf_path))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
        push!(state.mod.functions, leaf_ptr_ty)
        push!(state.mod.functions, ac_id)
        push!(state.mod.functions, ptr_id)
        for _ in leaf_path
            push!(state.mod.functions, zero_id)
        end
        return (ac_id, leaf_ty, nothing)
    end

    # Direct scalar type pun: pointee is a scalar with same size as load type
    # (e.g., pointee_ty = double, load_ty = i64). Load the pointee type, caller will bitcast.
    if _types_same_size(pointee_ty, load_ty)
        return (ptr_id, pointee_ty, nothing)
    end

    return (ptr_id, nothing, nothing)
end

"""
Look up the SPIR-V pointer type that the struct definition declares for its
first pointer member at offset 0. Returns the SPIR-V type ID for the pointer
(e.g., `_ptr_PhysicalStorageBuffer_float`), or nothing if not found.
Used to ensure AccessChain result types match the struct definition.
"""
function _get_struct_member_ptr_spirv_type(state::SPIRVEmitterState, pointee_ty::LLVM.LLVMType)
    struct_ty, member_idx = _find_offset0_ptr_member(pointee_ty)
    struct_ty === nothing && return nothing

    info = get(state.type_ctx.struct_ptr_members, (struct_ty, member_idx), nothing)
    if info !== nothing
        declared_pointee, _as = info
        declared_pointee_spirv = map_type!(state.type_ctx, declared_pointee)
        return map_pointer_type!(state.type_ctx, declared_pointee_spirv, SC.PhysicalStorageBuffer)
    end

    # Fallback: struct_ptr_members has no entry, but the struct type was already mapped
    # with an i8 fallback for ptr members (see _find_ptr_member_type_in_hierarchy).
    # Match that fallback so the AccessChain result type agrees with the struct definition.
    fallback_pointee = _find_ptr_member_type_in_hierarchy(state.type_ctx, struct_ty)
    if fallback_pointee !== nothing
        fallback_spirv = map_type!(state.type_ctx, fallback_pointee)
        return map_pointer_type!(state.type_ctx, fallback_spirv, SC.PhysicalStorageBuffer)
    end

    return nothing
end

"""
Recursively find the member index path to reach a given byte offset in a (possibly nested) struct.
Returns `(index_path, leaf_type)` where index_path is a Vector{Int} of 0-based member indices,
or `(nothing, nothing)` if the offset falls in padding or can't be decomposed.

Example: `{ { float, i64 } }` at offset 8 → ([0, 1], i64)
  - outer member 0: { float, i64 } at offset 0 → recurse with remaining offset 8
  - inner member 1: i64 at offset 8 → match!
"""
function _find_struct_member_path_by_offset(struct_ty::LLVM.StructType, target_offset::Int)
    path = Int[]
    _find_struct_member_path_recursive!(path, struct_ty, target_offset) || return (nothing, nothing)
    leaf_ty = struct_ty
    for idx in path
        if leaf_ty isa LLVM.StructType
            leaf_ty = LLVM.elements(leaf_ty)[idx + 1]
        else
            return (nothing, nothing)
        end
    end
    return (path, leaf_ty)
end

function _find_struct_member_path_recursive!(path::Vector{Int}, struct_ty::LLVM.StructType, target_offset::Int)
    member_types = LLVM.elements(struct_ty)
    running_offset = 0
    for (i, mt) in enumerate(member_types)
        member_align = _compute_type_alignment(mt)
        running_offset = (running_offset + member_align - 1) & ~(member_align - 1)
        member_size = _compute_type_size(mt)
        if running_offset == target_offset
            # Exact match at this level
            push!(path, i - 1)  # 0-based
            return true
        elseif running_offset < target_offset && target_offset < running_offset + member_size
            # Offset falls WITHIN this member — recurse if it's a struct
            if mt isa LLVM.StructType
                push!(path, i - 1)  # 0-based
                remaining = target_offset - running_offset
                if _find_struct_member_path_recursive!(path, mt, remaining)
                    return true
                end
                pop!(path)  # recursion failed, backtrack
            end
            # Not a struct or recursion failed → offset is in padding/middle of scalar
            return false
        end
        running_offset += member_size
    end
    return false
end

"""
Find array fields within a (possibly nested) struct that match a given element stride.
Returns `(path, byte_offset, array_type)` or `nothing`.

When `elem_stride > 0`, selects the array whose element size matches the stride.
When `elem_stride == 0`, returns the first array found (fallback).

Used for dynamic byte-offset GEPs on Function-SC struct allocas, where the dynamic index
targets an array field within the struct (e.g., accessing dims in a LavaDeviceArray).
"""
function _find_array_field_in_struct(struct_ty::LLVM.StructType, elem_stride::Int=0)
    candidates = Vector{Tuple{Vector{Int}, Int, LLVM.ArrayType}}()
    _collect_array_fields!(candidates, Int[], struct_ty, 0)
    isempty(candidates) && return nothing
    # If stride hint given, find matching array
    if elem_stride > 0
        for (path, offset, arr_ty) in candidates
            if _compute_type_size(LLVM.eltype(arr_ty)) == elem_stride
                return (path, offset, arr_ty)
            end
        end
    end
    # Fallback: return first
    path, offset, arr_ty = candidates[1]
    return (path, offset, arr_ty)
end

function _collect_array_fields!(results::Vector{Tuple{Vector{Int}, Int, LLVM.ArrayType}},
                                 path::Vector{Int}, struct_ty::LLVM.StructType, base_offset::Int)
    member_types = LLVM.elements(struct_ty)
    running_offset = 0
    for (i, mt) in enumerate(member_types)
        member_align = _compute_type_alignment(mt)
        running_offset = (running_offset + member_align - 1) & ~(member_align - 1)
        if mt isa LLVM.ArrayType
            push!(results, (vcat(path, [i - 1]), base_offset + running_offset, mt))
        elseif mt isa LLVM.StructType
            _collect_array_fields!(results, vcat(path, [i - 1]), mt, base_offset + running_offset)
        end
        running_offset += _compute_type_size(mt)
    end
end

"""
Decompose a byte offset into a multi-level index path through nested arrays/structs.
E.g., `[1 x [2 x i64]]` at offset 8 → [0, 1] (first outer element, second inner element),
leaf_type = i64.
"""
function _find_array_nested_path(ty::LLVM.LLVMType, target_offset::Int)
    path = Int[]
    leaf = _find_array_nested_path_recursive!(path, ty, target_offset)
    leaf === nothing && return (nothing, nothing)
    return (path, leaf)
end

function _find_array_nested_path_recursive!(path::Vector{Int}, ty::LLVM.ArrayType, target_offset::Int)
    elem_ty = LLVM.eltype(ty)
    elem_size = _compute_type_size(elem_ty)
    elem_size == 0 && return nothing
    outer_idx = target_offset ÷ elem_size
    remainder = target_offset % elem_size
    outer_idx >= LLVM.length(ty) && return nothing
    push!(path, outer_idx)
    if remainder == 0
        # Exact element boundary
        return elem_ty
    else
        # Sub-element access: recurse into the element
        result = _find_array_nested_path_recursive!(path, elem_ty, remainder)
        if result !== nothing
            return result
        end
        pop!(path)
        return nothing
    end
end

function _find_array_nested_path_recursive!(path::Vector{Int}, ty::LLVM.StructType, target_offset::Int)
    member_types = LLVM.elements(ty)
    running_offset = 0
    for (i, mt) in enumerate(member_types)
        member_align = _compute_type_alignment(mt)
        running_offset = (running_offset + member_align - 1) & ~(member_align - 1)
        member_size = _compute_type_size(mt)
        if running_offset == target_offset
            push!(path, i - 1)
            return mt
        elseif running_offset < target_offset && target_offset < running_offset + member_size
            push!(path, i - 1)
            remaining = target_offset - running_offset
            result = _find_array_nested_path_recursive!(path, mt, remaining)
            if result !== nothing
                return result
            end
            pop!(path)
            return nothing
        end
        running_offset += member_size
    end
    return nothing
end

# Fallback: scalar types can't be decomposed further
function _find_array_nested_path_recursive!(path::Vector{Int}, ty::LLVM.LLVMType, target_offset::Int)
    return nothing
end

"""
Find a zero-index path from `from_ty` to `target_ty` by following the first
element at each level of nested struct/array types.
Returns the index path (list of 0s) or nothing if target_ty can't be reached.
"""
function _find_zero_index_path(from_ty::LLVM.LLVMType, target_ty::LLVM.LLVMType)
    path = Int[]
    current = from_ty
    while current != target_ty
        if current isa LLVM.StructType
            elems = LLVM.elements(current)
            isempty(elems) && return nothing
            push!(path, 0)
            current = first(elems)
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return nothing
            push!(path, 0)
            current = LLVM.eltype(current)
        else
            return nothing  # Can't drill further, type not found
        end
    end
    isempty(path) && return nothing  # No drilling needed (shouldn't happen, checked above)
    return path
end

"""
Navigate to the scalar leaf (first non-composite element) following index 0.
Returns (path, leaf_type) or (nothing, nothing).
"""
function _find_zero_index_path_to_leaf(from_ty::LLVM.LLVMType)
    path = Int[]
    current = from_ty
    while current isa LLVM.StructType || current isa LLVM.ArrayType
        if current isa LLVM.StructType
            elems = LLVM.elements(current)
            isempty(elems) && return (nothing, nothing)
            push!(path, 0)
            current = first(elems)
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return (nothing, nothing)
            push!(path, 0)
            current = LLVM.eltype(current)
        end
    end
    isempty(path) && return (nothing, nothing)
    return (path, current)
end

"""
Find a zero-index path from `from_ty` to the first pointer member.
Returns the index path (list of 0s) or nothing if no pointer is reachable.
Used for "load ptr, ptr %struct_alloca" where LLVM loads the first pointer
field from a struct's base address.
"""
function _find_zero_index_path_to_pointer(from_ty::LLVM.LLVMType)
    path = Int[]
    current = from_ty
    while true
        if current isa LLVM.PointerType
            isempty(path) && return nothing  # Already a pointer, no drill needed
            return path
        elseif current isa LLVM.StructType
            elems = LLVM.elements(current)
            isempty(elems) && return nothing
            push!(path, 0)
            current = first(elems)
        elseif current isa LLVM.ArrayType
            LLVM.length(current) == 0 && return nothing
            push!(path, 0)
            current = LLVM.eltype(current)
        else
            return nothing  # Hit a scalar, no pointer at offset 0
        end
    end
end

"""Check if two LLVM types have the same bit width (for bitcast compatibility)."""
function _types_same_size(a::LLVM.LLVMType, b::LLVM.LLVMType)
    _llvm_type_bit_width(a) == _llvm_type_bit_width(b)
end

function _llvm_type_bit_width(ty::LLVM.LLVMType)
    if ty isa LLVM.IntegerType
        return LLVM.width(ty)
    elseif ty isa LLVM.LLVMFloat
        return 32
    elseif ty isa LLVM.LLVMDouble
        return 64
    elseif ty isa LLVM.LLVMHalf
        return 16
    else
        return -1  # Unknown
    end
end

"""
When the stored type doesn't match the pointer's declared pointee type in SPIR-V,
drill through nested structs/arrays using index 0 at each level until we reach
the matching type. Same logic as _resolve_struct_field_load! but for stores.
"""

"""
Re-derive a pointer at a parent struct level when a GEP drilled too deep.

When LLVM's `alloca_chain_fix` merges chained GEPs, it may drill all the way to
a scalar member (e.g., `ptr`), but the actual store writes a parent struct that
*contains* that member as its first field (at the same address due to zero offset).

This function examines the GEP instruction's indices and re-emits an OpAccessChain
with the last `length(reverse_path)` indices removed, producing a pointer at the
correct struct level for the store.
"""
function _rederive_store_ptr_at_parent_level!(state::SPIRVEmitterState,
                                               gep_inst::LLVM.GetElementPtrInst,
                                               gep_id::UInt32,
                                               store_ty::LLVM.LLVMType,
                                               reverse_path::Vector{Int})
    ops = LLVM.operands(gep_inst)
    n_indices = length(ops) - 1  # total GEP indices (including first)
    n_drop = length(reverse_path)  # how many trailing indices to remove

    if n_drop >= n_indices
        # Would remove all indices — can't back up further than the base pointer
        return (gep_id, nothing)
    end

    # Get the base pointer's SPIR-V ID
    base_ptr = ops[1]
    base_id = get_value_id!(state, base_ptr)

    sc = _get_pointer_storage_class(base_ptr)
    store_spirv_ty = if sc == SC.Workgroup
        map_workgroup_type!(state.type_ctx, store_ty)
    else
        map_type!(state.type_ctx, store_ty)
    end
    result_ptr_ty = map_pointer_type!(state.type_ctx, store_spirv_ty, sc)

    # Check if first index is 0 (the common case for alloca GEPs)
    first_idx = ops[2]
    is_first_zero = first_idx isa LLVM.ConstantInt && convert(Int, first_idx) == 0

    if is_first_zero
        # OpAccessChain: skip first index (always 0), use remaining minus the dropped ones
        n_keep = n_indices - 1 - n_drop  # indices after first, minus dropped

        if n_keep <= 0
            # All member indices would be dropped — just use base pointer
            return (base_id, nothing)
        end

        # Build AccessChain with kept indices
        # Also need to check if ptm_pointee differs from source_ty (same as in _emit_gep!)
        source_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep_inst))
        ptm_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)

        index_ids = UInt32[]
        if ptm_pointee !== nothing && ptm_pointee != source_ty
            prefix_path = _find_zero_index_path(ptm_pointee, source_ty)
            if prefix_path !== nothing
                zero_id = emit_constant_u32!(state.mod, UInt32(0))
                for _ in prefix_path
                    push!(index_ids, zero_id)
                end
            end
        end

        # Add GEP indices 3 through (3 + n_keep - 1), i.e., skip first index, keep n_keep
        for i in 3:(2 + n_keep)
            push!(index_ids, _ensure_index_i32!(state, ops[i]))
        end

        ac_id = fresh_id!(state.mod)
        word_count = UInt32(4 + length(index_ids))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
        push!(state.mod.functions, result_ptr_ty)
        push!(state.mod.functions, ac_id)
        push!(state.mod.functions, base_id)
        append!(state.mod.functions, index_ids)

        return (ac_id, nothing)
    else
        # Non-zero first index — more complex, fall back to no fix
        return (gep_id, nothing)
    end
end

function _resolve_struct_field_store!(state::SPIRVEmitterState, ptr::LLVM.Value,
                                       ptr_id::UInt32, store_ty::LLVM.LLVMType,
                                       store_value::LLVM.Value)
    pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
    pointee_ty === nothing && return (ptr_id, nothing)
    pointee_ty == store_ty && return (ptr_id, nothing)

    # Check forward path: pointee is a struct containing store_ty as first member
    index_path = _find_zero_index_path(pointee_ty, store_ty)

    if index_path === nothing
        # Check reverse path: store_ty is a struct containing pointee_ty as first member.
        # This means the GEP drilled too deep — it reached a scalar member but the store
        # writes the parent struct. Both have the same address (first member at offset 0).
        # Re-derive the pointer at the correct level from the GEP's base.
        reverse_path = _find_zero_index_path(store_ty, pointee_ty)
        if reverse_path !== nothing && ptr isa LLVM.GetElementPtrInst
            return _rederive_store_ptr_at_parent_level!(state, ptr, ptr_id, store_ty, reverse_path)
        end
        return (ptr_id, nothing)
    end

    sc = _get_pointer_storage_class(ptr)

    # For pointer store types, use the struct's declared member type for the AccessChain
    # (must match the struct definition), not the usage-inferred PTM type.
    store_val_bitcast_to = nothing
    if store_ty isa LLVM.PointerType
        struct_member_spirv_ty = _get_struct_member_ptr_spirv_type(state, pointee_ty)
        inferred_spirv_ty = map_pointer_type_for_value!(state.type_ctx, store_value)
        if struct_member_spirv_ty !== nothing
            field_spirv_ty = struct_member_spirv_ty
            # If the stored value's type differs from the struct's member type, bitcast
            if struct_member_spirv_ty != inferred_spirv_ty
                store_val_bitcast_to = struct_member_spirv_ty
            end
        else
            field_spirv_ty = inferred_spirv_ty
        end
    else
        field_spirv_ty = map_type!(state.type_ctx, store_ty)
    end
    field_ptr_ty = map_pointer_type!(state.type_ctx, field_spirv_ty, sc)

    zero_id = emit_constant_u32!(state.mod, UInt32(0))
    ac_id = fresh_id!(state.mod)
    word_count = UInt32(4 + length(index_path))
    push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
    push!(state.mod.functions, field_ptr_ty)
    push!(state.mod.functions, ac_id)
    push!(state.mod.functions, ptr_id)
    for _ in index_path
        push!(state.mod.functions, zero_id)
    end
    return (ac_id, store_val_bitcast_to)
end

function _emit_store!(state::SPIRVEmitterState, inst::LLVM.StoreInst)
    ops = LLVM.operands(inst)
    value = ops[1]
    ptr = ops[2]

    # Skip stores of poison/undef — storing undefined data is a no-op
    if value isa LLVM.PoisonValue || value isa LLVM.UndefValue
        return
    end

    # NOTE: Composite workgroup stores are decomposed in the LLVM pass
    # _decompose_composite_workgroup_accesses! before reaching the emitter.
    store_ty = LLVM.value_type(value)
    if store_ty isa LLVM.StructType && _get_pointer_storage_class(ptr) == SC.Workgroup
        error("Unexpected composite store to Workgroup memory — should have been decomposed by LLVM pass")
        return
    end

    val_id = get_value_id!(state, value)
    ptr_id = get_value_id!(state, ptr)

    # Handle struct-pointer mismatch for stores (same as loads):
    # If the stored type is a scalar but the pointer points to a composite, drill down.
    store_ty = LLVM.value_type(value)
    orig_ptr_id = ptr_id
    ptr_id, store_val_bitcast_to = _resolve_struct_field_store!(state, ptr, ptr_id, store_ty, value)

    # If storing a pointer with a type that differs from the struct's declared member type,
    # bitcast the value to match the struct definition.
    if store_val_bitcast_to !== nothing
        bitcast_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, store_val_bitcast_to, bitcast_id, val_id)
        val_id = bitcast_id
    end

    # Check for type mismatch (e.g., `store i32 0, ptr %p` when %p is a float*)
    # LLVM may optimize bit-identical constants to a different type.
    # In SPIR-V, we need to bitcast the value to match the pointer's declared type.
    # Skip if we already resolved the pointer to a scalar field above.
    # Also skip for PSB stores — those use _fix_psb_ptr_type_for_store! instead
    # (bitcasting the pointer to match the LLVM store type, not the other way around).
    is_psb = _is_psb_pointer(ptr)
    if !is_psb && ptr_id == orig_ptr_id && store_val_bitcast_to === nothing
        val_id = _bitcast_store_value_if_needed!(state, ptr, value, val_id)
    end

    # PhysicalStorageBuffer stores MUST have Aligned memory operand
    if is_psb
        store_ty = LLVM.value_type(value)
        # Check if this wide store needs decomposition due to misaligned PSB address.
        # SPIR-V requires Aligned ≥ scalar_size for PSB (VUID 06314), so we can't
        # just lower alignment. Instead, decompose i64/double stores at non-8-aligned
        # addresses into two i32 stores with Aligned 4.
        llvm_align = UInt32(LLVM.alignment(inst))
        if _psb_needs_decomposition(state, ptr, store_ty; llvm_align)
            _emit_psb_decomposed_store!(state, ptr_id, val_id, store_ty)
        else
            align = _get_alignment_for_type(store_ty)
            ptr_id = _fix_psb_ptr_type_for_store!(state, ptr, ptr_id, value)
            word_count = UInt32(5)  # opcode + ptr + val + mem_operand + alignment
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpStore))
            push!(state.mod.functions, ptr_id)
            push!(state.mod.functions, val_id)
            push!(state.mod.functions, UInt32(0x02))  # Aligned
            push!(state.mod.functions, align)
        end
    else
        # For Workgroup/Function stores: if value type doesn't match pointer's pointee type,
        # bitcast the POINTER to match the value type. This happens when:
        # - _decompose_composite_workgroup_accesses! creates byte-offset GEPs to struct fields
        # - GEP chains access sub-elements (e.g. i32 within [16 x i64]) through byte-offset
        #   GEPs that the lift pass couldn't fully resolve
        # We must NOT truncate the value — that loses data. Instead, OpBitcast the pointer.
        sc = _get_pointer_storage_class(ptr)
        if sc == SC.Workgroup || sc == SC.Function
            pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
            if pointee_ty !== nothing
                val_ty = LLVM.value_type(value)
                if val_ty != pointee_ty && val_ty isa LLVM.IntegerType && pointee_ty isa LLVM.IntegerType
                    # Bitcast pointer to match value type
                    val_spirv_ty = map_type!(state.type_ctx, val_ty)
                    new_ptr_ty = map_pointer_type!(state.type_ctx, val_spirv_ty, sc)
                    cast_id = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpBitcast, new_ptr_ty, cast_id, ptr_id)
                    ptr_id = cast_id
                end
            end
        end
        encode_instruction!(state.mod.functions, Op.OpStore, ptr_id, val_id)
    end
end

"""
Bitcast a store value if its LLVM type doesn't match the pointer's declared pointee type.
This happens when LLVM optimizes e.g. `store float 0.0` → `store i32 0` (same bit pattern).
Returns the (possibly new) value ID to use for the store.
"""
function _bitcast_store_value_if_needed!(state::SPIRVEmitterState, ptr::LLVM.Value,
                                          value::LLVM.Value, val_id::UInt32)
    val_ty = LLVM.value_type(value)
    pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
    pointee_ty === nothing && return val_id

    # For pointer-to-pointer stores: LLVM opaque ptrs are always equal at the LLVM level,
    # but the SPIR-V types may differ (e.g., _ptr_PSB_float vs _ptr_PSB__struct).
    # This happens when PTM infers a different pointee type for the loaded ptr vs what the
    # struct member declared. OpBitcast between pointer types of same storage class is valid.
    # Must check BEFORE the val_ty == pointee_ty early return since both are opaque `ptr`.
    if val_ty isa LLVM.PointerType && pointee_ty isa LLVM.PointerType
        expected_spirv_ty = get(state.spirv_ptr_element_type, ptr, UInt32(0))
        if expected_spirv_ty != UInt32(0)
            actual_spirv_ty = map_pointer_type_for_value!(state.type_ctx, value)
            if actual_spirv_ty != expected_spirv_ty
                cast_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpBitcast, expected_spirv_ty, cast_id, val_id)
                return cast_id
            end
        end
        return val_id
    end

    # If types already match, no bitcast needed
    val_ty == pointee_ty && return val_id

    if val_ty isa LLVM.PointerType || pointee_ty isa LLVM.PointerType
        return val_id
    end
    val_bw = _llvm_type_bit_width(val_ty)
    pointee_bw = _llvm_type_bit_width(pointee_ty)

    # If either type is non-scalar (struct, array, unknown), skip
    if val_bw < 0 || pointee_bw < 0
        return val_id
    end

    if val_bw == pointee_bw
        # Same width: OpBitcast (e.g. i32↔f32)
        target_spirv_ty = map_type!(state.type_ctx, pointee_ty)
        cast_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, target_spirv_ty, cast_id, val_id)
        return cast_id
    else
        # Different widths — bitcast the pointer to match the value type.
        # This happens when GEP chains access sub-elements (e.g. i32 within [16 x i64])
        # through byte-offset GEPs that the lift pass couldn't fully resolve.
        # Bitcast pointer: ptr<pointee_ty> → ptr<val_ty>, then store val_ty.
        return val_id  # Return original val_id — we'll handle this in _emit_store! instead
    end
end

"""
Fix PSB pointer type mismatch for stores. When storing a value whose type differs in
bit width from the pointer's PTM-inferred pointee type, bitcast the pointer to match
the value type. E.g. storing i8 (Bool) into ptr<i64,PSB> → bitcast ptr to ptr<i8,PSB>.
"""
function _fix_psb_ptr_type_for_store!(state::SPIRVEmitterState, ptr::LLVM.Value,
                                       ptr_id::UInt32, value::LLVM.Value)
    val_ty = LLVM.value_type(value)
    pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
    pointee_ty === nothing && return ptr_id
    val_ty == pointee_ty && return ptr_id
    # Skip pointer types — can't bitcast pointers
    val_ty isa LLVM.PointerType && return ptr_id
    pointee_ty isa LLVM.PointerType && return ptr_id
    # Bitcast the pointer to match the value type (handles type puns like
    # i32 stored to ptr<[2 x half]> for ComplexF16)
    val_spirv_ty = map_type!(state.type_ctx, val_ty)
    val_ptr_ty = map_pointer_type!(state.type_ctx, val_spirv_ty, SC.PhysicalStorageBuffer)
    return _emit_psb_ptr_reinterpret!(state, val_ptr_ty, ptr_id)
end

# Determine the SPIR-V storage class for a pointer value.
# Traces through GEPs to find whether the pointer originates from an alloca
# (Function storage) or from device memory (PhysicalStorageBuffer).
function _get_pointer_storage_class(ptr::LLVM.Value)
    ty = LLVM.value_type(ptr)
    if !(ty isa LLVM.PointerType)
        return SC.Function
    end
    as = LLVM.addrspace(ty)
    if as == 1
        # Addrspace 1: Julia constant globals (Private) or PSB pointers
        if _is_constant_global_ptr(ptr)
            return SC.Private
        end
        return SC.PhysicalStorageBuffer
    elseif as == 2
        return SC.PushConstant
    elseif as == 3
        return SC.Workgroup
    elseif as == 7
        return SC.Input
    elseif as == 0
        # Addrspace 0: Function if from alloca, PSB otherwise
        if _is_psb_pointer(ptr)
            return SC.PhysicalStorageBuffer
        else
            return SC.Function
        end
    else
        return SC.Function
    end
end

function _is_psb_pointer(ptr::LLVM.Value)
    ty = LLVM.value_type(ptr)
    ty isa LLVM.PointerType || return false
    as = LLVM.addrspace(ty)
    # Addrspace 1: PSB unless it's a constant global (Julia _j_const lookup tables → Private SC)
    as == 1 && return !_is_constant_global_ptr(ptr)
    # Addrspace 0: PSB unless it's an alloca (which is genuinely Function storage)
    as == 0 && !(ptr isa LLVM.AllocaInst) && return _trace_to_non_alloca(ptr)
    return false
end

"""
Check if an addrspace(1) pointer traces back to a constant global variable
(Julia's _j_const_N lookup tables). These map to Private SC, not PSB.
"""
function _is_constant_global_ptr(ptr::LLVM.Value)
    ptr isa LLVM.GlobalVariable && return true
    if ptr isa LLVM.GetElementPtrInst
        base = LLVM.operands(ptr)[1]
        return _is_constant_global_ptr(base)
    end
    if ptr isa LLVM.ConstantExpr
        # ConstantExpr GEP on a global
        opcode = LLVM.API.LLVMGetConstOpcode(ptr)
        if opcode == LLVM.API.LLVMGetElementPtr
            base = LLVM.operands(ptr)[1]
            return _is_constant_global_ptr(base)
        end
    end
    return false
end

"""
Trace a pointer value back to determine if it originates from an alloca.
GEPs on allocas produce Function pointers; GEPs on parameters produce PSB pointers.
"""
function _trace_to_non_alloca(ptr::LLVM.Value, visited::Set{LLVM.Value}=Set{LLVM.Value}())
    ptr in visited && return true  # Cycle → assume PSB (allocas don't form cycles, PSB loop ptrs do)
    push!(visited, ptr)
    ptr isa LLVM.AllocaInst && return false
    if ptr isa LLVM.GetElementPtrInst
        base = LLVM.operands(ptr)[1]
        return _trace_to_non_alloca(base, visited)
    end
    # PHI nodes: trace all incoming values. If ANY traces to an alloca,
    # the pointer may be Function-space — return false (not PSB).
    if ptr isa LLVM.PHIInst
        for (val, _) in LLVM.incoming(ptr)
            if !_trace_to_non_alloca(val, visited)
                return false
            end
        end
        # All incoming values are non-alloca → PSB
        return true
    end
    # Function parameter, inttoptr, etc. → PSB
    return true
end

"""Get alignment for a type (for PhysicalStorageBuffer Aligned operand)."""
function _get_alignment_for_type(ty::LLVM.LLVMType)
    if ty isa LLVM.LLVMFloat
        return UInt32(4)
    elseif ty isa LLVM.LLVMDouble
        return UInt32(8)
    elseif ty isa LLVM.LLVMHalf
        return UInt32(2)
    elseif ty isa LLVM.IntegerType
        w = LLVM.width(ty)
        return UInt32(max(1, w ÷ 8))
    elseif ty isa LLVM.PointerType
        return UInt32(8)  # Pointers are 8 bytes
    elseif ty isa LLVM.StructType
        # Alignment of a struct = max alignment of its members
        max_align = UInt32(1)
        for elem in LLVM.elements(ty)
            max_align = max(max_align, _get_alignment_for_type(elem))
        end
        return max_align
    elseif ty isa LLVM.ArrayType
        return _get_alignment_for_type(eltype(ty))
    else
        return UInt32(4)  # Default alignment
    end
end

"""
Check if a PSB pointer is potentially misaligned for a given access type.
Returns true when the pointer's declared pointee type has smaller alignment than
the type being loaded/stored. This happens when SROA packs pairs of i32 into i64
stores through a ptr<i32> that was obtained by GEP into an i32 array — the address
is only 4-aligned, but i64 requires `Aligned 8`.

SPIR-V validation requires PSB Aligned ≥ scalar type size (VUID 06314),
so we can't just lower alignment. Instead, callers must decompose wide accesses
into narrower ones (e.g., i64 → two i32).
"""
function _psb_needs_decomposition(state::SPIRVEmitterState, ptr::LLVM.Value, access_ty::LLVM.LLVMType;
                                   llvm_align::UInt32=UInt32(0))
    access_align = _get_alignment_for_type(access_ty)
    access_align <= 4 && return false  # i32/float/i16/i8 never need decomposition

    # Check 0 (universal): LLVM's own alignment on the load/store instruction.
    # LLVM SROA produces `load i64, ptr %gep, align 1` when the byte GEP offset may not
    # be naturally aligned. If LLVM says align < type's natural alignment, decompose.
    # Only applies to GEP-based pointers — arg buffer loads (IntToPtrInst) always have
    # proper alignment even when LLVM marks them `align 1` (conservative SROA).
    if llvm_align > 0 && llvm_align < access_align && ptr isa LLVM.GetElementPtrInst
        return true
    end

    # Check 1: pointer's PTM pointee type has smaller alignment
    pointee_ty = get_pointee_type(state.type_ctx.ptm, ptr)
    if pointee_ty !== nothing && !(pointee_ty isa LLVM.PointerType)
        pointee_align = _get_alignment_for_type(pointee_ty)
        if pointee_align < access_align
            return true
        end
    end

    # Check 2: known byte offset from integer-arithmetic GEP path
    offset = get(state.psb_known_byte_offsets, ptr, Int64(-1))
    if offset > 0
        addr_align = UInt32(1 << trailing_zeros(offset))
        if addr_align < access_align
            return true
        end
    end

    # Check 3: tracked pointer alignment from runtime-indexed GEP on composite types.
    # When a byte GEP has a non-constant offset with a stride not a multiple of 8
    # (e.g., 60-byte struct array), the result pointer is only stride-aligned.
    # This alignment is propagated through subsequent constant-offset GEP chains.
    ptr_align = get(state.psb_ptr_alignment, ptr, UInt32(0))
    if ptr_align > 0 && ptr_align < access_align
        return true
    end

    # Check 4: inttoptr(add(..., const)) — LLVM optimization passes convert
    # struct field GEPs to ptrtoint + add(constant_offset) + inttoptr chains.
    # E.g., accessing field at byte offset 140 in a 288-byte struct becomes:
    #   %addr = ptrtoint ptr %struct_ptr to i64
    #   %field = add i64 %addr, 140
    #   %ptr = inttoptr i64 %field to ptr
    #   %val = load i64, ptr %ptr  ; offset 140 is only 4-aligned, NOT 8!
    # The GEP-based alignment tracking (checks 2-3) misses this because there's no GEP.
    if ptr isa LLVM.IntToPtrInst
        src = LLVM.operands(ptr)[1]
        const_offset = _extract_constant_offset_from_adds(src)
        if const_offset != 0
            addr_align = UInt32(1 << trailing_zeros(abs(const_offset)))
            if addr_align < access_align
                return true
            end
        end
    end

    return false
end

"""
Extract the sum of all constant terms from a chain of `add` instructions.
Used to determine the constant byte offset in `ptrtoint + add(...) + inttoptr` patterns
where LLVM has converted GEPs to integer arithmetic.

E.g., `add(add(%base, 140), 8)` → returns 148.
E.g., `add(%base, add(mul(%idx, 288), 140))` → returns 140.
Non-add, non-constant operands contribute 0 (assumed dynamic/aligned).
"""
function _extract_constant_offset_from_adds(val::LLVM.Value)
    if val isa LLVM.ConstantInt
        return convert(Int64, val)
    end
    if !(val isa LLVM.Instruction) || LLVM.opcode(val) != LLVM.API.LLVMAdd
        return Int64(0)
    end
    ops = LLVM.operands(val)
    total = Int64(0)
    for op in (ops[1], ops[2])
        if op isa LLVM.ConstantInt
            total += convert(Int64, op)
        elseif op isa LLVM.Instruction && LLVM.opcode(op) == LLVM.API.LLVMAdd
            total += _extract_constant_offset_from_adds(op)
        end
    end
    return total
end

"""
Compute and record the minimum guaranteed alignment of a PSB pointer resulting from
a byte-offset GEP. Two sources of alignment info:

1. **Stride alignment**: when the base is a composite type (struct/array) and the byte
   offset is non-constant, the stride is the composite size. If stride % 8 != 0,
   the result pointer may not be 8-aligned for some indices.
   E.g., 60-byte struct → stride_align = 4, so odd-indexed elements are only 4-aligned.

2. **Base propagation**: if the base pointer already has tracked alignment (from a
   previous GEP in the chain), propagate it through constant offsets via gcd.
   E.g., base has align 4, constant offset 8 → gcd(4,8) = 4.

Only records alignment < 8, since 8+ is always sufficient for i64/double stores.
"""
function _track_psb_gep_alignment!(state::SPIRVEmitterState, inst::LLVM.Value,
                                    base_ptr::LLVM.Value, base_pointee::LLVM.LLVMType,
                                    byte_offset_val::LLVM.Value)
    # Source 1: stride-based alignment for non-constant offsets on composite types
    stride_align = UInt32(0)
    if !(byte_offset_val isa LLVM.ConstantInt) &&
       (base_pointee isa LLVM.StructType || base_pointee isa LLVM.ArrayType)
        composite_size = _compute_type_size(base_pointee)
        if composite_size > 0 && composite_size % 8 != 0
            stride_align = UInt32(1 << trailing_zeros(composite_size))
        end
    end

    # Source 2: base pointer's tracked alignment
    base_align = get(state.psb_ptr_alignment, base_ptr, UInt32(0))

    # Combine sources
    result_align = UInt32(0)
    if stride_align > 0 && base_align > 0
        result_align = min(stride_align, base_align)
    elseif stride_align > 0
        result_align = stride_align
    elseif base_align > 0
        if byte_offset_val isa LLVM.ConstantInt
            offset = convert(Int64, byte_offset_val)
            if offset != 0
                offset_align = UInt32(1 << trailing_zeros(abs(offset)))
                result_align = min(base_align, offset_align)
            else
                result_align = base_align
            end
        else
            # Non-constant offset on non-composite base: conservatively keep base alignment
            result_align = base_align
        end
    end

    if result_align > 0 && result_align < 8
        state.psb_ptr_alignment[inst] = result_align
    end
end

"""
Emit a decomposed PSB store: split a wide store (i64/double) into two i32 stores.
Used when the PSB pointer isn't aligned to the full type width.
Uses ConvertPtrToU/ConvertUToPtr for both halves (avoids OpBitcast between PSB
pointer types which can crash NVIDIA's SPIR-V compiler).
"""
function _emit_psb_decomposed_store!(state::SPIRVEmitterState, ptr_id::UInt32, val_id::UInt32,
                                      val_ty::LLVM.LLVMType)
    u32_spirv = emit_type_int!(state.mod, UInt32(32), UInt32(0))
    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    u32_ptr_ty = map_pointer_type!(state.type_ctx, u32_spirv, SC.PhysicalStorageBuffer)

    # Convert value to i64 if it's a double
    raw_val = val_id
    if val_ty isa LLVM.LLVMDouble
        raw_val = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, u64_spirv, raw_val, val_id)
    end

    # Extract low 32 bits: OpUConvert i64 → i32
    lo_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpUConvert, u32_spirv, lo_id, raw_val)

    # Extract high 32 bits: shift right by 32, then truncate
    const_32 = emit_constant_u64!(state.mod, UInt64(32))
    shifted = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpShiftRightLogical, u64_spirv, shifted, raw_val, const_32)
    hi_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpUConvert, u32_spirv, hi_id, shifted)

    # Get base address as integer
    base_u64 = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_spirv, base_u64, ptr_id)

    # Low store: ptr → ptr<u32,PSB>, store low 32 bits
    lo_ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, u32_ptr_ty, lo_ptr, base_u64)
    push!(state.mod.functions, (UInt32(5) << 16) | UInt32(Op.OpStore))
    push!(state.mod.functions, lo_ptr)
    push!(state.mod.functions, lo_id)
    push!(state.mod.functions, UInt32(0x02))  # Aligned
    push!(state.mod.functions, UInt32(4))

    # High store: (ptr + 4) → ptr<u32,PSB>, store high 32 bits
    const_4 = emit_constant_u64!(state.mod, UInt64(4))
    hi_addr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, hi_addr, base_u64, const_4)
    hi_ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, u32_ptr_ty, hi_ptr, hi_addr)
    push!(state.mod.functions, (UInt32(5) << 16) | UInt32(Op.OpStore))
    push!(state.mod.functions, hi_ptr)
    push!(state.mod.functions, hi_id)
    push!(state.mod.functions, UInt32(0x02))  # Aligned
    push!(state.mod.functions, UInt32(4))
end

"""
Emit a decomposed PSB load: split a wide load (i64/double) into two i32 loads.
Used when the PSB pointer isn't aligned to the full type width.
Uses ConvertPtrToU/ConvertUToPtr for both halves (avoids OpBitcast between PSB
pointer types which can crash NVIDIA's SPIR-V compiler).
Returns the SPIR-V result ID of the combined value.
"""
function _emit_psb_decomposed_load!(state::SPIRVEmitterState, ptr_id::UInt32,
                                     load_ty::LLVM.LLVMType, result_spirv_ty::UInt32)
    u32_spirv = emit_type_int!(state.mod, UInt32(32), UInt32(0))
    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    u32_ptr_ty = map_pointer_type!(state.type_ctx, u32_spirv, SC.PhysicalStorageBuffer)

    # Get base address as integer
    base_u64 = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_spirv, base_u64, ptr_id)

    # Low load: ptr → ptr<u32,PSB>, load low 32 bits
    lo_ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, u32_ptr_ty, lo_ptr, base_u64)
    lo_id = fresh_id!(state.mod)
    push!(state.mod.functions, (UInt32(6) << 16) | UInt32(Op.OpLoad))
    push!(state.mod.functions, u32_spirv)
    push!(state.mod.functions, lo_id)
    push!(state.mod.functions, lo_ptr)
    push!(state.mod.functions, UInt32(0x02))  # Aligned
    push!(state.mod.functions, UInt32(4))

    # High load: (ptr + 4) → ptr<u32,PSB>, load high 32 bits
    const_4 = emit_constant_u64!(state.mod, UInt64(4))
    hi_addr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, hi_addr, base_u64, const_4)
    hi_ptr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, u32_ptr_ty, hi_ptr, hi_addr)
    hi_id = fresh_id!(state.mod)
    push!(state.mod.functions, (UInt32(6) << 16) | UInt32(Op.OpLoad))
    push!(state.mod.functions, u32_spirv)
    push!(state.mod.functions, hi_id)
    push!(state.mod.functions, hi_ptr)
    push!(state.mod.functions, UInt32(0x02))  # Aligned
    push!(state.mod.functions, UInt32(4))

    # Combine: lo | (hi << 32)
    lo_u64 = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpUConvert, u64_spirv, lo_u64, lo_id)
    hi_u64 = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpUConvert, u64_spirv, hi_u64, hi_id)
    const_32 = emit_constant_u64!(state.mod, UInt64(32))
    hi_shifted = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpShiftLeftLogical, u64_spirv, hi_shifted, hi_u64, const_32)
    combined = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpBitwiseOr, u64_spirv, combined, lo_u64, hi_shifted)

    # If the result should be double, bitcast i64 → double
    if load_ty isa LLVM.LLVMDouble
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitcast, result_spirv_ty, result_id, combined)
        return result_id
    end
    return combined
end

function _emit_gep!(state::SPIRVEmitterState, inst::LLVM.GetElementPtrInst)
    ops = LLVM.operands(inst)
    base_ptr = ops[1]
    base_id = get_value_id!(state, base_ptr)

    source_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(inst))
    n_indices = length(ops) - 1

    if n_indices == 0
        state.value_map[inst] = base_id
        return
    end

    # Handle byte-offset GEPs: getelementptr i8, ptr %base, i64 %byte_offset
    # LLVM uses i8-sourced GEPs for pointer arithmetic with byte offsets.
    # In SPIR-V, we must use the base pointer's actual element type and convert
    # the byte offset to an element index (byte_offset / sizeof(element)).
    if source_ty isa LLVM.IntegerType && LLVM.width(source_ty) == 8 && n_indices == 1
        base_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)
        if base_pointee !== nothing && !(base_pointee isa LLVM.IntegerType && LLVM.width(base_pointee) == 8)
            if base_pointee isa LLVM.PointerType
                # Opaque pointer base (e.g. BDA inttoptr after SROA): infer actual access
                # type from users and use PSB byte arithmetic.
                sc = _get_pointer_storage_class(base_ptr)
                if sc == SC.PhysicalStorageBuffer
                    _emit_psb_byte_offset_with_user_type!(state, inst, base_id, ops)
                    return
                end
                # Non-PSB: fall through to regular GEP path
            else
                _emit_byte_offset_gep!(state, inst, base_ptr, base_id, base_pointee)
                return
            end
        end
    end

    # Determine the result pointee type
    result_pointee = _compute_gep_result_type(source_ty, inst)
    if result_pointee === nothing
        error("Could not compute GEP result type for: $inst")
    end

    # Get pointer storage class — use map_pointer_type_for_value! logic
    # GEP results inherit the base pointer's storage class
    sc = _get_pointer_storage_class(base_ptr)
    # Map the result pointee type. When it's a pointer (GEP into struct accessing a ptr member),
    # we need the typed pointer from the struct member map or PTM, not raw map_type!.
    # For Workgroup storage class, use fresh (undecorated) type IDs to avoid layout
    # decoration conflicts with PSB types.
    result_pointee_spirv = if result_pointee isa LLVM.PointerType
        # GEP result is a pointer-to-pointer. We need the inner pointer's SPIR-V type.
        # Look up via struct member map using the source struct type and field index.
        _map_gep_ptr_result!(state.type_ctx, source_ty, inst)
    elseif sc == SC.Workgroup
        map_workgroup_type!(state.type_ctx, result_pointee)
    else
        map_type!(state.type_ctx, result_pointee)
    end
    result_ptr_ty = map_pointer_type!(state.type_ctx, result_pointee_spirv, sc)

    if n_indices == 1
        # Single index: getelementptr T, ptr %base, i64 %idx → base + idx * sizeof(T)
        #
        # Check if the base pointer's actual pointee type is an array whose element type
        # matches source_ty. This happens with shared memory globals: the variable is
        # `ptr → [N x T]` but LLVM generates `gep T, ptr @global, i64 %idx`.
        # In SPIR-V we must use OpAccessChain into the array, not OpPtrAccessChain.
        base_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)
        if base_pointee isa LLVM.ArrayType && (LLVM.eltype(base_pointee) == source_ty ||
            _compute_type_size(LLVM.eltype(base_pointee)) == _compute_type_size(source_ty))
            idx_i32 = _ensure_index_i32!(state, ops[2])
            result_id = fresh_id!(state.mod)

            # When source_ty differs from array element type (type-punning via opaque ptrs,
            # e.g. gep float, ptr @alloca_of_i32_array, i64 %idx), use the array's actual
            # element type for OpAccessChain. Downstream load/store handles the bitcast.
            arr_elem_ty = LLVM.eltype(base_pointee)
            actual_result_ptr_ty = if arr_elem_ty != source_ty
                elem_spirv = sc == SC.Workgroup ? map_workgroup_type!(state.type_ctx, arr_elem_ty) :
                    map_type!(state.type_ctx, arr_elem_ty)
                map_pointer_type!(state.type_ctx, elem_spirv, sc)
            else
                result_ptr_ty
            end

            # Standard OpAccessChain to index into the array.
            # For struct arrays in Workgroup, this gives a pointer to the struct element.
            # Subsequent field accesses use byte-offset GEPs → struct member AccessChains.
            word_count = UInt32(5)  # result_type, result_id, base, index
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
            push!(state.mod.functions, actual_result_ptr_ty)
            push!(state.mod.functions, result_id)
            push!(state.mod.functions, base_id)
            push!(state.mod.functions, idx_i32)
            state.value_map[inst] = result_id
            # Set PTM to actual array element type so loads/stores can bitcast correctly
            ptm_ty = arr_elem_ty != source_ty ? arr_elem_ty : source_ty
            set_pointee_type!(state.type_ctx.ptm, inst, ptm_ty; priority=5)
            # Record array element origin for folding chained GEPs (e.g. array[a][b] → array[a+b])
            state.array_element_origin[inst] = (base_id, UInt32[], idx_i32, base_pointee)
        elseif (sc == SC.Function || sc == SC.Private) && haskey(state.array_element_origin, base_ptr)
            # Chained single-index GEP on a Function-SC array element:
            # base was from OpAccessChain into an array, fold by adding indices.
            # gep T, ptr (AccessChain array[a]), b  →  AccessChain array[a + b]
            arr_base_id, static_path, prev_idx_id, arr_type = state.array_element_origin[base_ptr]
            new_idx_i32 = _ensure_index_i32!(state, ops[2])
            # Emit IAdd to combine indices
            u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
            combined_idx = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIAdd, u32_ty, combined_idx, prev_idx_id, new_idx_i32)
            result_id = fresh_id!(state.mod)
            all_indices = vcat(static_path, [combined_idx])
            word_count = UInt32(4 + length(all_indices))
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
            push!(state.mod.functions, result_ptr_ty)
            push!(state.mod.functions, result_id)
            push!(state.mod.functions, arr_base_id)
            append!(state.mod.functions, all_indices)
            state.value_map[inst] = result_id
            set_pointee_type!(state.type_ctx.ptm, inst, source_ty; priority=5)
            # Propagate origin for further chaining
            state.array_element_origin[inst] = (arr_base_id, static_path, combined_idx, arr_type)
        else
            idx = get_value_id!(state, ops[2])
            if sc == SC.PhysicalStorageBuffer
                # PSB pointer arithmetic: use manual byte-offset instead of OpPtrAccessChain
                # (OpPtrAccessChain on PSB struct pointers is broken on AMD RADV)
                result_id = _emit_psb_ptr_arithmetic!(state, base_id, idx, result_ptr_ty, source_ty;
                    idx_llvm_ty=LLVM.value_type(ops[2]))
            elseif sc == SC.Function
                # Function storage class: OpPtrAccessChain is NOT valid.
                # Use base pointer directly; load/store handlers resolve type mismatches.
                result_id = base_id
            elseif sc == SC.Private
                # Private storage class: OpPtrAccessChain is NOT valid.
                # Use OpAccessChain with element index for Private array globals.
                result_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpAccessChain, result_ptr_ty, result_id, base_id, idx)
            else
                # Normal pointer arithmetic: OpPtrAccessChain (Workgroup, StorageBuffer)
                result_id = fresh_id!(state.mod)
                _ensure_array_stride_decoration!(state, result_ptr_ty, source_ty)
                encode_instruction!(state.mod.functions, UInt16(67), result_ptr_ty, result_id, base_id, idx)
            end
            state.value_map[inst] = result_id
            # Track pointee type for downstream store/load handlers
            set_pointee_type!(state.type_ctx.ptm, inst, result_pointee; priority=3)
        end
    else
        # Multiple indices: first index is array offset, rest are struct/array drilling
        first_idx = ops[2]
        first_idx_val = first_idx isa LLVM.ConstantInt ? convert(Int64, first_idx) : nothing

        if first_idx_val !== nothing && first_idx_val == 0
            # Degenerate GEP on a scalar base: LLVM may emit extra trailing zero
            # indices (e.g. gep <scalar>, ptr %p, 0, 0). In SPIR-V this cannot be
            # encoded as OpAccessChain on a non-composite pointer.
            # Treat all-zero trailing indices as a no-op pointer projection.
            base_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)
            if base_pointee !== nothing &&
               !(base_pointee isa LLVM.StructType) &&
               !(base_pointee isa LLVM.ArrayType) &&
               length(ops) > 3
                trailing_all_zero = true
                for i in 3:length(ops)
                    idx_op = ops[i]
                    if !(idx_op isa LLVM.ConstantInt) || convert(Int64, idx_op) != 0
                        trailing_all_zero = false
                        break
                    end
                end
                if trailing_all_zero
                    state.value_map[inst] = base_id
                    set_pointee_type!(state.type_ctx.ptm, inst, base_pointee; priority=5)
                    return
                end
            end

            # First index is 0 → use OpAccessChain with remaining indices
            index_ids = UInt32[]

            # Check if the GEP's source element type differs from the base pointer's
            # PTM pointee type. If so, we need to prepend zero indices to navigate
            # from the actual SPIR-V pointer type to the GEP's source type.
            # This happens when LLVM uses a sub-struct as GEP source type knowing the
            # first field is at offset 0 of the outer struct.
            ptm_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)
            use_ptr_arithmetic = false
            if ptm_pointee !== nothing && ptm_pointee != source_ty
                prefix_path = _find_zero_index_path(ptm_pointee, source_ty)
                if prefix_path !== nothing
                    zero_id = emit_constant_u32!(state.mod, UInt32(0))
                    for _ in prefix_path
                        push!(index_ids, zero_id)
                    end
                elseif sc == SC.PhysicalStorageBuffer
                    # Type-punned GEP on PSB pointer: GEP source type doesn't match the
                    # SPIR-V pointee type and there's no zero-index navigation path.
                    # This happens when LLVM SROA/memcpy expansion creates GEPs with a
                    # different type view (e.g., flat [36 x i32] array, or smaller struct
                    # reinterpretation of a larger struct). Compute byte offset from the
                    # GEP indices on the GEP's own source type and use PSB ptr arithmetic.
                    use_ptr_arithmetic = true
                end
            end

            if use_ptr_arithmetic
                # Compute total byte offset by walking GEP indices through the source type
                u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
                total_byte_offset = Int64(0)
                all_const = true

                current_walk_ty = source_ty
                for i in 3:length(ops)
                    idx_op = ops[i]
                    if !(idx_op isa LLVM.ConstantInt)
                        all_const = false
                        break
                    end
                    idx_val = convert(Int64, idx_op)
                    if current_walk_ty isa LLVM.StructType
                        elems = LLVM.elements(current_walk_ty)
                        for j in 0:(idx_val-1)
                            total_byte_offset += Int64(_compute_type_size(elems[j+1]))
                        end
                        current_walk_ty = elems[idx_val+1]
                    elseif current_walk_ty isa LLVM.ArrayType
                        elem_ty = LLVM.eltype(current_walk_ty)
                        total_byte_offset += idx_val * Int64(_compute_type_size(elem_ty))
                        current_walk_ty = elem_ty
                    else
                        total_byte_offset += idx_val * Int64(_compute_type_size(current_walk_ty))
                    end
                end

                if all_const
                    byte_offset_id = _emit_u64_constant!(state.mod, UInt64(total_byte_offset))
                else
                    # Fallback: compute byte offset dynamically (rare, only if non-const index)
                    # For now, just handle the simple [N x T] single-index case
                    if length(ops) == 3 && source_ty isa LLVM.ArrayType
                        elem_ty = LLVM.eltype(source_ty)
                        elem_size = _compute_type_size(elem_ty)
                        idx_op = ops[3]
                        idx_id = get_value_id!(state, idx_op)
                        idx_u64 = idx_id
                        idx_ty = LLVM.value_type(idx_op)
                        if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) < 64
                            idx_u64 = fresh_id!(state.mod)
                            encode_instruction!(state.mod.functions, Op.OpSConvert, u64_spirv, idx_u64, idx_id)
                        end
                        stride_const = _emit_u64_constant!(state.mod, UInt64(elem_size))
                        byte_offset_id = fresh_id!(state.mod)
                        encode_instruction!(state.mod.functions, Op.OpIMul, u64_spirv, byte_offset_id, idx_u64, stride_const)
                    else
                        error("Type-punned GEP on PSB with non-constant indices not fully supported: $inst")
                    end
                end

                # base_addr + byte_offset → result pointer
                base_u64 = _cached_psb_ptr_to_u64!(state, base_id)
                elem_addr_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, elem_addr_id, base_u64, byte_offset_id)
                result_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ptr_ty, result_id, elem_addr_id)
                state.value_map[inst] = result_id
                set_pointee_type!(state.type_ctx.ptm, inst, result_pointee; priority=3)
                # Record byte offset for alignment checking in store/load.
                # Without this, Check 2 in _psb_needs_decomposition can't detect
                # misaligned i64 loads through type-punned GEPs (e.g., VPMaterialEvalWorkItem
                # field at offset 140 in a 288-byte struct → 140 % 8 = 4, not 8-aligned).
                if all_const && total_byte_offset > 0
                    state.psb_known_byte_offsets[inst] = total_byte_offset
                end
            else
            for i in 3:length(ops)
                push!(index_ids, _ensure_index_i32!(state, ops[i]))
            end
            result_id = fresh_id!(state.mod)
            # OpAccessChain: result_type, result_id, base, indices...
            word_count = UInt32(4 + length(index_ids))
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
            push!(state.mod.functions, result_ptr_ty)
            push!(state.mod.functions, result_id)
            push!(state.mod.functions, base_id)
            append!(state.mod.functions, index_ids)
            state.value_map[inst] = result_id
            # Track pointee type for downstream GEPs (e.g. single-index array element access)
            set_pointee_type!(state.type_ctx.ptm, inst, result_pointee; priority=3)
            end
        else
            # First index is non-zero.
            # Check if the base pointer's pointee is an array whose element matches source_ty.
            # For workgroup/global array variables, LLVM generates:
            #   gep [3 x float], ptr addrspace(3) @global, i64 %idx, i64 1
            # The first index selects the array element, subsequent drill into it.
            # In SPIR-V this is OpAccessChain (not OpPtrAccessChain) because we're
            # indexing INTO the array, not doing pointer arithmetic.
            base_pointee = get_pointee_type(state.type_ctx.ptm, base_ptr)
            if base_pointee isa LLVM.ArrayType && LLVM.eltype(base_pointee) == source_ty
                # Use OpAccessChain with all indices
                index_ids = UInt32[]
                for i in 2:length(ops)
                    push!(index_ids, _ensure_index_i32!(state, ops[i]))
                end
                result_id = fresh_id!(state.mod)
                word_count = UInt32(4 + length(index_ids))
                push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                push!(state.mod.functions, result_ptr_ty)
                push!(state.mod.functions, result_id)
                push!(state.mod.functions, base_id)
                append!(state.mod.functions, index_ids)
                state.value_map[inst] = result_id
                set_pointee_type!(state.type_ctx.ptm, inst, result_pointee; priority=3)
            else
                if sc == SC.PhysicalStorageBuffer && length(ops) == 3
                    # PSB with element offset + one struct member index:
                    # First offset to element, then OpAccessChain for member
                    first_idx = get_value_id!(state, ops[2])
                    elem_ptr_id = _emit_psb_ptr_arithmetic!(state, base_id, first_idx,
                        map_pointer_type!(state.type_ctx, map_type!(state.type_ctx, source_ty), sc),
                        source_ty; idx_llvm_ty=LLVM.value_type(ops[2]))
                    # Step 2: OpAccessChain for struct member
                    member_idx = _ensure_index_i32!(state, ops[3])
                    result_id = fresh_id!(state.mod)
                    word_count = UInt32(5)
                    push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                    push!(state.mod.functions, result_ptr_ty)
                    push!(state.mod.functions, result_id)
                    push!(state.mod.functions, elem_ptr_id)
                    push!(state.mod.functions, member_idx)
                elseif sc == SC.PhysicalStorageBuffer
                    # PSB with element offset + multiple indices:
                    # First offset to element, then OpAccessChain for remaining
                    first_idx = get_value_id!(state, ops[2])
                    elem_ptr_id = _emit_psb_ptr_arithmetic!(state, base_id, first_idx,
                        map_pointer_type!(state.type_ctx, map_type!(state.type_ctx, source_ty), sc),
                        source_ty; idx_llvm_ty=LLVM.value_type(ops[2]))
                    remaining_ids = UInt32[]
                    for i in 3:length(ops)
                        push!(remaining_ids, _ensure_index_i32!(state, ops[i]))
                    end
                    result_id = fresh_id!(state.mod)
                    word_count = UInt32(4 + length(remaining_ids))
                    push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                    push!(state.mod.functions, result_ptr_ty)
                    push!(state.mod.functions, result_id)
                    push!(state.mod.functions, elem_ptr_id)
                    append!(state.mod.functions, remaining_ids)
                elseif sc == SC.Function
                    # Function storage class: OpPtrAccessChain is NOT valid.
                    # Use base pointer directly; load/store handlers resolve type mismatches.
                    result_id = base_id
                elseif sc == SC.Private
                    # Private storage class: OpPtrAccessChain is NOT valid.
                    # Use OpAccessChain with all indices for Private array globals.
                    index_ids = UInt32[]
                    push!(index_ids, get_value_id!(state, ops[2]))
                    for i in 3:length(ops)
                        push!(index_ids, _ensure_index_i32!(state, ops[i]))
                    end
                    result_id = fresh_id!(state.mod)
                    word_count = UInt32(4 + length(index_ids))
                    push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                    push!(state.mod.functions, result_ptr_ty)
                    push!(state.mod.functions, result_id)
                    push!(state.mod.functions, base_id)
                    append!(state.mod.functions, index_ids)
                else
                    # Normal case: OpPtrAccessChain with all indices (Workgroup, StorageBuffer)
                    _ensure_array_stride_decoration!(state, result_ptr_ty, source_ty)
                    index_ids = UInt32[]
                    push!(index_ids, get_value_id!(state, ops[2]))
                    for i in 3:length(ops)
                        push!(index_ids, _ensure_index_i32!(state, ops[i]))
                    end
                    result_id = fresh_id!(state.mod)
                    word_count = UInt32(4 + length(index_ids))
                    push!(state.mod.functions, (word_count << 16) | UInt32(67))
                    push!(state.mod.functions, result_ptr_ty)
                    push!(state.mod.functions, result_id)
                    push!(state.mod.functions, base_id)
                    append!(state.mod.functions, index_ids)
                end
                state.value_map[inst] = result_id
                set_pointee_type!(state.type_ctx.ptm, inst, result_pointee; priority=3)
            end
        end
    end
    # Track SPIR-V element type for pointer results (used by load handler to detect
    # pointer type mismatches when same LLVM struct is used with different ptr element types)
    if result_pointee isa LLVM.PointerType
        state.spirv_ptr_element_type[inst] = result_pointee_spirv
    end
end

"""
Map a GEP result that is a pointer type (GEP into a struct accessing a pointer member).
Returns the SPIR-V type ID for the pointer member.
"""
function _map_gep_ptr_result!(ctx::SPIRVTypeContext, source_ty::LLVM.LLVMType,
                               gep::LLVM.GetElementPtrInst)
    # Walk the GEP indices to find the FINAL struct + member index where the result is a pointer.
    # Multi-level GEPs like (0, 0, 0, 1, 0) drill through struct→struct→array→struct→ptr.
    ops = LLVM.operands(gep)
    n_ops = length(ops)
    if n_ops >= 3
        current_ty = source_ty
        final_struct_ty = nothing
        final_member_idx = -1

        for i in 3:n_ops  # Skip base_ptr (ops[1]) and first index (ops[2])
            idx_val = ops[i]
            idx_val isa LLVM.ConstantInt || break
            idx = convert(Int, idx_val)

            if current_ty isa LLVM.StructType
                members = LLVM.elements(current_ty)
                (idx + 1) <= length(members) || break
                final_struct_ty = current_ty
                final_member_idx = idx
                current_ty = members[idx + 1]
            elseif current_ty isa LLVM.ArrayType
                current_ty = LLVM.eltype(current_ty)
            else
                break
            end
        end

        if final_struct_ty !== nothing && current_ty isa LLVM.PointerType
            info = get(ctx.struct_ptr_members, (final_struct_ty, final_member_idx), nothing)
            if info !== nothing
                pointee_ty, _as = info
                pointee_spirv = map_type!(ctx, pointee_ty)
                return map_pointer_type!(ctx, pointee_spirv, SC.PhysicalStorageBuffer)
            end
        end
    end

    # Fallback: look up the GEP result in the PTM
    gep_ptm = get_pointee_type(ctx.ptm, gep)
    if gep_ptm !== nothing && !(gep_ptm isa LLVM.PointerType)
        ptm_spirv = map_type!(ctx, gep_ptm)
        return map_pointer_type!(ctx, ptm_spirv, SC.PhysicalStorageBuffer)
    end

    # Ultimate fallback: generic byte pointer
    i8_spirv = map_type!(ctx, LLVM.IntType(8))
    return map_pointer_type!(ctx, i8_spirv, SC.PhysicalStorageBuffer)
end

"""
Emit a byte-offset GEP as a typed element access.
Converts `getelementptr i8, ptr %base, i64 %byte_offset` to an OpPtrAccessChain
using the base pointer's actual element type: idx = byte_offset / sizeof(element).
Also updates the PTM so downstream stores/loads see the correct pointee type.
"""
function _emit_byte_offset_gep!(state::SPIRVEmitterState, inst::LLVM.GetElementPtrInst,
                                 base_ptr::LLVM.Value, base_id::UInt32,
                                 base_pointee::LLVM.LLVMType)
    ops = LLVM.operands(inst)
    byte_offset_id = get_value_id!(state, ops[2])

    # Get pointer storage class
    sc = _get_pointer_storage_class(base_ptr)

    idx_ty = LLVM.value_type(ops[2])
    idx_spirv_ty = map_type!(state.type_ctx, idx_ty)

    # When base_pointee is an array or struct, the byte offset may be accessing
    # an element WITHIN the composite (not advancing past it).
    # E.g., `getelementptr i8, ptr %struct_ptr, i64 4` where ptr→[3 x float]
    # means "access element at byte 4" = float[1], NOT advance by 4 * sizeof([3 x float]).
    if base_pointee isa LLVM.ArrayType
        # Array: divide by element size to get array index, use OpAccessChain
        elem_ty = LLVM.eltype(base_pointee)
        elem_size = _compute_type_size(elem_ty)

        # Check for constant byte offset that needs recursive decomposition.
        # E.g., gep i8, ptr to [1 x [2 x i64]], i64 8 → outer_idx=0, inner_idx=1.
        byte_offset_val = ops[2]
        if byte_offset_val isa LLVM.ConstantInt && elem_size > 0
            const_offset = convert(Int64, byte_offset_val)
            remainder = const_offset % elem_size
            if remainder != 0
                if sc == SC.PhysicalStorageBuffer
                    # PSB non-aligned byte offset: use direct integer arithmetic.
                    # OpAccessChain(byte_offset / elem_size) truncates sub-element offsets
                    # (e.g., byte 21 into [12 x i32] → 21/4=5 → byte 20, losing +1).
                    # Instead, use ConvertPtrToU + IAdd(byte_offset) + ConvertUToPtr.
                    result_pointee = _infer_type_from_gep_users(inst)
                    if result_pointee === nothing
                        result_pointee = LLVM.IntType(8)  # default to byte
                    end
                    effective_pointee = if result_pointee isa LLVM.PointerType
                        LLVM.IntType(64)
                    else
                        result_pointee
                    end
                    pointee_spirv = map_type!(state.type_ctx, effective_pointee)
                    result_ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, SC.PhysicalStorageBuffer)

                    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
                    base_u64 = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_spirv, base_u64, base_id)

                    bo_u64 = byte_offset_id
                    if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) < 64
                        bo_u64 = fresh_id!(state.mod)
                        encode_instruction!(state.mod.functions, Op.OpSConvert, u64_spirv, bo_u64, byte_offset_id)
                    end

                    elem_addr = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, elem_addr, base_u64, bo_u64)

                    result_id = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ptr_ty, result_id, elem_addr)

                    state.value_map[inst] = result_id
                    set_pointee_type!(state.type_ctx.ptm, inst, effective_pointee; priority=4)
                    # Record byte offset for alignment computation in store/load
                    state.psb_known_byte_offsets[inst] = const_offset
                    return
                elseif elem_ty isa LLVM.ArrayType || elem_ty isa LLVM.StructType
                    # Sub-element access: decompose recursively through nested types
                    index_path, leaf_ty = _find_array_nested_path(base_pointee, const_offset)
                    if index_path !== nothing
                        leaf_spirv = if sc == SC.Workgroup
                            map_workgroup_type!(state.type_ctx, leaf_ty)
                        else
                            map_type!(state.type_ctx, leaf_ty)
                        end
                        result_ptr_ty = map_pointer_type!(state.type_ctx, leaf_spirv, sc)
                        u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
                        idx_ids = UInt32[]
                        for idx_val in index_path
                            push!(idx_ids, _emit_int_constant!(state, LLVM.IntType(32), Int64(idx_val)))
                        end
                        result_id = fresh_id!(state.mod)
                        word_count = UInt32(4 + length(idx_ids))
                        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                        push!(state.mod.functions, result_ptr_ty)
                        push!(state.mod.functions, result_id)
                        push!(state.mod.functions, base_id)
                        append!(state.mod.functions, idx_ids)
                        state.value_map[inst] = result_id
                        set_pointee_type!(state.type_ctx.ptm, inst, leaf_ty; priority=4)
                        return
                    end
                end
            end
        end

        # Simple case: byte offset evenly divides by element size
        # Result is a pointer to the element type (not the array type)
        elem_spirv = if sc == SC.Workgroup
            map_workgroup_type!(state.type_ctx, elem_ty)
        else
            map_type!(state.type_ctx, elem_ty)
        end
        result_ptr_ty = map_pointer_type!(state.type_ctx, elem_spirv, sc)

        if elem_size == 1
            idx_id = byte_offset_id
        else
            size_id = _emit_int_constant!(state, idx_ty, Int64(elem_size))
            idx_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpSDiv, idx_spirv_ty, idx_id, byte_offset_id, size_id)
        end

        # Convert to i32 for OpAccessChain index (SPIR-V requires i32 indices)
        idx_i32 = if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) > 32
            u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
            conv_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, conv_id, idx_id)
            conv_id
        else
            idx_id
        end

        # OpAccessChain into the array (not OpPtrAccessChain past it)
        result_id = fresh_id!(state.mod)
        word_count = UInt32(5)
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
        push!(state.mod.functions, result_ptr_ty)
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, base_id)
        push!(state.mod.functions, idx_i32)
        state.value_map[inst] = result_id
        set_pointee_type!(state.type_ctx.ptm, inst, elem_ty; priority=4)
        # Register array element origin so chained byte-offset GEPs can fold offsets.
        # E.g., gep i8, (gep i8, ptr @alloca, i64 %a), i64 %b  → AccessChain @alloca[a/sz + b/sz]
        state.array_element_origin[inst] = (base_id, UInt32[], idx_i32, base_pointee)
        return
    end

    # StructType: decompose constant byte offset into struct member index.
    # E.g., `getelementptr i8, ptr %struct_ptr, i64 4` where ptr→{float,float,float,i32}
    # → member index 1 (byte offset 4 = second float field).
    # Recurse into nested structs: e.g., `{ { float, i64 } }` at offset 8
    # → outer member 0 + inner member 1 = OpAccessChain [0, 1].
    if base_pointee isa LLVM.StructType
        byte_offset_val = ops[2]
        if byte_offset_val isa LLVM.ConstantInt
            offset = convert(Int64, byte_offset_val)
            # Recursively find the member index path for this byte offset
            index_path, leaf_ty = _find_struct_member_path_by_offset(base_pointee, offset)

            if index_path !== nothing
                member_spirv = if sc == SC.Workgroup
                    map_workgroup_type!(state.type_ctx, leaf_ty)
                else
                    map_type!(state.type_ctx, leaf_ty)
                end
                result_ptr_ty = map_pointer_type!(state.type_ctx, member_spirv, sc)

                # OpAccessChain with index path (may be multi-level for nested structs)
                result_id = fresh_id!(state.mod)
                word_count = UInt32(4 + length(index_path))
                push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                push!(state.mod.functions, result_ptr_ty)
                push!(state.mod.functions, result_id)
                push!(state.mod.functions, base_id)
                for idx in index_path
                    push!(state.mod.functions, emit_constant_u32!(state.mod, UInt32(idx)))
                end
                state.value_map[inst] = result_id
                set_pointee_type!(state.type_ctx.ptm, inst, leaf_ty; priority=4)
                return
            end

            # Offset exceeds struct size or falls in padding.
            # If the base came from a dynamic array element access (array_element_origin),
            # the constant offset may span across multiple array elements.
            # E.g., gep i8, (AccessChain arr[%dyn]), 32 where arr element is 16 bytes
            # → the offset 32 means +2 elements from the dynamic index.
            if (sc == SC.Function || sc == SC.Private) && haskey(state.array_element_origin, base_ptr)
                arr_base_id, static_path, prev_idx_id, arr_type = state.array_element_origin[base_ptr]
                arr_elem_ty = LLVM.eltype(arr_type)
                arr_elem_size = _compute_type_size(arr_elem_ty)
                if arr_elem_size > 0
                    arr_idx_adjust = offset ÷ arr_elem_size
                    sub_offset = offset % arr_elem_size

                    # Adjust the dynamic array index
                    adj_id = emit_constant_u32!(state.mod, UInt32(arr_idx_adjust))
                    u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
                    combined_idx = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpIAdd, u32_ty, combined_idx, prev_idx_id, adj_id)

                    if sub_offset == 0
                        # Exact element boundary: result is pointer to array element
                        arr_elem_spirv = map_type!(state.type_ctx, arr_elem_ty)
                        result_ptr_ty = map_pointer_type!(state.type_ctx, arr_elem_spirv, sc)
                        all_indices = vcat(static_path, [combined_idx])
                        result_id = fresh_id!(state.mod)
                        word_count = UInt32(4 + length(all_indices))
                        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                        push!(state.mod.functions, result_ptr_ty)
                        push!(state.mod.functions, result_id)
                        push!(state.mod.functions, arr_base_id)
                        append!(state.mod.functions, all_indices)
                        state.value_map[inst] = result_id
                        set_pointee_type!(state.type_ctx.ptm, inst, arr_elem_ty; priority=4)
                        state.array_element_origin[inst] = (arr_base_id, static_path, combined_idx, arr_type)
                        return
                    elseif arr_elem_ty isa LLVM.StructType
                        # Sub-element offset: access array element then struct field
                        sub_path, sub_leaf = _find_struct_member_path_by_offset(arr_elem_ty, sub_offset)
                        if sub_path !== nothing
                            sub_leaf_spirv = map_type!(state.type_ctx, sub_leaf)
                            result_ptr_ty = map_pointer_type!(state.type_ctx, sub_leaf_spirv, sc)
                            all_indices = copy(static_path)
                            push!(all_indices, combined_idx)
                            for idx_val in sub_path
                                push!(all_indices, emit_constant_u32!(state.mod, UInt32(idx_val)))
                            end
                            result_id = fresh_id!(state.mod)
                            word_count = UInt32(4 + length(all_indices))
                            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                            push!(state.mod.functions, result_ptr_ty)
                            push!(state.mod.functions, result_id)
                            push!(state.mod.functions, arr_base_id)
                            append!(state.mod.functions, all_indices)
                            state.value_map[inst] = result_id
                            set_pointee_type!(state.type_ctx.ptm, inst, sub_leaf; priority=4)
                            return
                        end
                    end
                end
            end

            # Offset falls in struct padding (no member at this offset).
            # For Function storage class, we can't use OpPtrAccessChain.
            # Use the base pointer directly — the load handler (_resolve_struct_field_load!)
            # will drill into the struct to find a compatible type. Since padding bytes are
            # undefined, any loaded value is semantically acceptable.
            if sc == SC.Function || sc == SC.Private
                state.value_map[inst] = base_id
                set_pointee_type!(state.type_ctx.ptm, inst, base_pointee; priority=4)
                return
            end
        else
            # Dynamic byte offset on struct (Function/Private SC):
            # Find the array field within the struct and emit a dynamic AccessChain.
            # Pattern: gep i8, ptr %struct_alloca, i64 %dynamic_offset
            # where the struct contains an array field (e.g., { ptr, [N x i64] })
            # and the dynamic offset targets elements within that array.
            if sc == SC.Function || sc == SC.Private
                # Extract stride hint from the byte offset expression:
                # shl i64 %x, N → stride = 2^N
                # mul i64 %x, N → stride = N
                elem_stride = 0
                byte_offset_val = ops[2]
                if byte_offset_val isa LLVM.Instruction
                    bo_opcode = LLVM.opcode(byte_offset_val)
                    bo_ops = LLVM.operands(byte_offset_val)
                    if bo_opcode == LLVM.API.LLVMShl && length(bo_ops) >= 2 && bo_ops[2] isa LLVM.ConstantInt
                        shift = convert(Int64, bo_ops[2])
                        elem_stride = 1 << shift
                    elseif bo_opcode == LLVM.API.LLVMMul && length(bo_ops) >= 2
                        if bo_ops[1] isa LLVM.ConstantInt
                            elem_stride = convert(Int64, bo_ops[1])
                        elseif bo_ops[2] isa LLVM.ConstantInt
                            elem_stride = convert(Int64, bo_ops[2])
                        end
                    end
                end
                arr_info = _find_array_field_in_struct(base_pointee, elem_stride)
                if arr_info !== nothing
                    arr_path, arr_byte_offset, arr_ty = arr_info
                    arr_elem_ty = LLVM.eltype(arr_ty)
                    arr_elem_size = _compute_type_size(arr_elem_ty)
                    if arr_elem_size > 0
                        # Compute: element_idx = (byte_offset - arr_byte_offset) / arr_elem_size
                        arr_elem_spirv = map_type!(state.type_ctx, arr_elem_ty)
                        result_ptr_ty = map_pointer_type!(state.type_ctx, arr_elem_spirv, sc)

                        # Subtract the array field's byte offset
                        adjusted_id = byte_offset_id
                        if arr_byte_offset > 0
                            off_id = _emit_int_constant!(state, idx_ty, Int64(arr_byte_offset))
                            adjusted_id = fresh_id!(state.mod)
                            encode_instruction!(state.mod.functions, Op.OpISub, idx_spirv_ty, adjusted_id, byte_offset_id, off_id)
                        end

                        # Divide by element size
                        dyn_idx_id = adjusted_id
                        if arr_elem_size > 1
                            sz_id = _emit_int_constant!(state, idx_ty, Int64(arr_elem_size))
                            dyn_idx_id = fresh_id!(state.mod)
                            encode_instruction!(state.mod.functions, Op.OpSDiv, idx_spirv_ty, dyn_idx_id, adjusted_id, sz_id)
                        end

                        # Convert to i32 for OpAccessChain
                        dyn_idx_i32 = if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) > 32
                            u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
                            conv_id = fresh_id!(state.mod)
                            encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, conv_id, dyn_idx_id)
                            conv_id
                        else
                            dyn_idx_id
                        end

                        # Build AccessChain: base, <static path to array>, dynamic_index
                        all_indices = UInt32[]
                        for idx_val in arr_path
                            push!(all_indices, emit_constant_u32!(state.mod, UInt32(idx_val)))
                        end
                        push!(all_indices, dyn_idx_i32)

                        result_id = fresh_id!(state.mod)
                        word_count = UInt32(4 + length(all_indices))
                        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
                        push!(state.mod.functions, result_ptr_ty)
                        push!(state.mod.functions, result_id)
                        push!(state.mod.functions, base_id)
                        append!(state.mod.functions, all_indices)
                        state.value_map[inst] = result_id
                        set_pointee_type!(state.type_ctx.ptm, inst, arr_elem_ty; priority=4)
                        # Store static path (indices BEFORE the dynamic index) for chained GEPs
                        static_path = UInt32[emit_constant_u32!(state.mod, UInt32(idx_val)) for idx_val in arr_path]
                        state.array_element_origin[inst] = (base_id, static_path, dyn_idx_i32, arr_ty)
                        return
                    end
                end
            end
        end
    end

    # For non-composite types: divide byte offset by type size for OpPtrAccessChain
    # For PSB pointers from SROA-decomposed BDA struct accesses, the base pointee
    # (from the first load) may differ from the actual field type at this byte offset.
    # Infer the correct type from GEP users.
    effective_pointee = base_pointee
    if sc == SC.PhysicalStorageBuffer
        user_type = _infer_type_from_gep_users(inst)
        if user_type !== nothing
            if user_type isa LLVM.PointerType
                effective_pointee = LLVM.IntType(64)  # PSB pointers are 8-byte addresses
            else
                effective_pointee = user_type
            end
        end
    end

    elem_size = _compute_type_size(effective_pointee)

    pointee_spirv = map_type!(state.type_ctx, effective_pointee)
    result_ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, sc)

    if elem_size == 1
        idx_id = byte_offset_id
    else
        size_id = _emit_int_constant!(state, idx_ty, Int64(elem_size))
        idx_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpSDiv, idx_spirv_ty, idx_id, byte_offset_id, size_id)
    end

    # Emit element access
    if sc == SC.PhysicalStorageBuffer
        # PSB: use manual byte-offset arithmetic (OpPtrAccessChain broken on AMD RADV)
        # byte_offset_id is already the byte offset, just add it directly
        u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
        base_u64 = _cached_psb_ptr_to_u64!(state, base_id)
        # Widen byte offset to u64 if needed
        bo_u64 = byte_offset_id
        if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) < 64
            bo_u64 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpSConvert, u64_spirv, bo_u64, byte_offset_id)
        end
        elem_addr = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, elem_addr, base_u64, bo_u64)
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ptr_ty, result_id, elem_addr)
        # Track alignment for PSB misalignment detection (Fix #5).
        # When stride (composite size) isn't a multiple of 8, or base has tracked alignment,
        # record so _psb_needs_decomposition() can detect i64 stores needing decomposition.
        _track_psb_gep_alignment!(state, inst, base_ptr, base_pointee, ops[2])
        # Record constant byte offset for alignment checking in store/load (Fix #8b).
        byte_offset_val = ops[2]
        if byte_offset_val isa LLVM.ConstantInt
            const_bo = convert(Int64, byte_offset_val)
            if const_bo != 0
                state.psb_known_byte_offsets[inst] = const_bo
            end
        end
    elseif sc == SC.Function || sc == SC.Private
        # Function/Private storage class: OpPtrAccessChain is NOT valid (VUID-StandaloneSpirv-Base-07650).
        if sc == SC.Private
            result_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpAccessChain, result_ptr_ty, result_id, base_id, idx_id)
        elseif haskey(state.array_element_origin, base_ptr)
            # Chained byte-offset GEP on a Function-SC array element pointer:
            # base was from OpAccessChain into an array, fold by adding the element index offset.
            # E.g., gep i8, ptr (AccessChain array[a]), -4  →  AccessChain array[a + (-4/elemsize)]
            arr_base_id, static_path, prev_idx_id, arr_type = state.array_element_origin[base_ptr]
            # idx_id is already byte_offset / elem_size (computed above as SDiv)
            # Convert idx_id to i32 for OpAccessChain
            new_idx_i32 = if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) > 32
                u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
                conv_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, conv_id, idx_id)
                conv_id
            else
                idx_id
            end
            u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
            combined_idx = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIAdd, u32_ty, combined_idx, prev_idx_id, new_idx_i32)
            result_id = fresh_id!(state.mod)
            all_indices = vcat(static_path, [combined_idx])
            word_count = UInt32(4 + length(all_indices))
            push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
            push!(state.mod.functions, result_ptr_ty)
            push!(state.mod.functions, result_id)
            push!(state.mod.functions, arr_base_id)
            append!(state.mod.functions, all_indices)
            # Propagate origin for further chaining
            state.array_element_origin[inst] = (arr_base_id, static_path, combined_idx, arr_type)
        else
            # Function: use base pointer directly, let load/store handlers resolve.
            result_id = base_id
        end
    else
        _ensure_array_stride_decoration!(state, result_ptr_ty, base_pointee)
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, UInt16(67), result_ptr_ty, result_id, base_id, idx_id)
    end
    state.value_map[inst] = result_id

    # Update PTM so downstream stores/loads see the correct type
    set_pointee_type!(state.type_ctx.ptm, inst, effective_pointee; priority=4)
end

"""
Emit an integer constant of the given LLVM integer type.
"""
function _emit_int_constant!(state::SPIRVEmitterState, ty::LLVM.LLVMType, value::Int64)
    type_id = map_type!(state.type_ctx, ty)
    w = ty isa LLVM.IntegerType ? LLVM.width(ty) : 64
    if w <= 32
        return emit_constant_u32!(state.mod, UInt32(value & 0xFFFFFFFF))
    else
        bits = reinterpret(UInt64, value)
        lo = UInt32(bits & 0xFFFFFFFF)
        hi = UInt32((bits >> 32) & 0xFFFFFFFF)
        key = (:const, type_id, bits)
        get!(state.mod.constant_cache, key) do
            id = fresh_id!(state.mod)
            encode_instruction!(state.mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
            id
        end
    end
end

"""
    _cached_psb_ptr_to_u64!(state, base_id) -> UInt32

Emit OpConvertPtrToU for a PSB pointer, with block-local caching.
If the same `base_id` was already converted in the current block,
returns the cached u64 result ID instead of emitting a new instruction.
This eliminates redundant ptr→u64 conversions when the same PSB pointer
is accessed multiple times in one block (common in struct field accesses).
"""
function _cached_psb_ptr_to_u64!(state::SPIRVEmitterState, base_id::UInt32)
    key = (base_id, state.current_block_label)
    cached = get(state.psb_ptr_to_u64, key, nothing)
    cached !== nothing && return cached
    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    base_u64 = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_spirv, base_u64, base_id)
    state.psb_ptr_to_u64[key] = base_u64
    return base_u64
end

"""
Emit a PSB byte-offset GEP when the base pointee is an opaque PointerType.
This happens after SROA decomposes BDA argument struct loads into individual
field accesses via byte-offset GEPs. We infer the actual access type from
the GEP's users (loads/stores) and produce a correctly-typed PSB pointer.
"""
function _emit_psb_byte_offset_with_user_type!(state::SPIRVEmitterState,
                                                inst::LLVM.GetElementPtrInst,
                                                base_id::UInt32,
                                                ops)
    byte_offset_id = get_value_id!(state, ops[2])
    idx_ty = LLVM.value_type(ops[2])

    # Infer the result pointee type from users (what type will be loaded/stored)
    result_pointee = _infer_type_from_gep_users(inst)
    if result_pointee === nothing
        # Default to i64 (covers pointer loads — ptrs are 8 bytes)
        result_pointee = LLVM.IntType(64)
    end

    # Map the inferred pointee type to SPIR-V
    effective_pointee = if result_pointee isa LLVM.PointerType
        # Loading a pointer — PSB pointers are 8-byte addresses.
        # Use i64 as pointee; the load handler will convert i64 → typed ptr.
        LLVM.IntType(64)
    else
        result_pointee
    end
    pointee_spirv = map_type!(state.type_ctx, effective_pointee)
    result_ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, SC.PhysicalStorageBuffer)

    # PSB byte arithmetic: ptr→u64, add byte_offset, u64→ptr
    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    base_u64 = _cached_psb_ptr_to_u64!(state, base_id)

    bo_u64 = byte_offset_id
    if idx_ty isa LLVM.IntegerType && LLVM.width(idx_ty) < 64
        bo_u64 = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpSConvert, u64_spirv, bo_u64, byte_offset_id)
    end

    elem_addr = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, elem_addr, base_u64, bo_u64)

    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ptr_ty, result_id, elem_addr)

    state.value_map[inst] = result_id
    set_pointee_type!(state.type_ctx.ptm, inst, effective_pointee; priority=4)
end

# _infer_type_from_gep_users is defined in types.jl

"""
Infer what a loaded pointer value points to by examining its users.
When a `load ptr` produces a pointer, look at how that pointer is used
(GEPs with struct source type, loads, stores) to determine the pointee.
"""
function _infer_inner_ptr_pointee(gep_or_load::LLVM.Instruction)
    # Look at users of loads from this GEP/inttoptr
    for use in LLVM.uses(gep_or_load)
        user = LLVM.user(use)
        if user isa LLVM.LoadInst
            # The load produces a ptr — look at how that ptr is used.
            # Two passes: prefer struct GEPs (non-byte-offset) over byte-offset GEPs,
            # since byte-offset GEPs access individual fields while struct GEPs give
            # the real base type.
            for inner_use in LLVM.uses(user)
                inner_user = LLVM.user(inner_use)
                if inner_user isa LLVM.GetElementPtrInst
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inner_user))
                    if !(src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8)
                        return src_ty
                    end
                end
            end
            # Second pass: byte-offset GEPs, loads, stores
            for inner_use in LLVM.uses(user)
                inner_user = LLVM.user(inner_use)
                if inner_user isa LLVM.GetElementPtrInst
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inner_user))
                    if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 8
                        inner_result = _infer_type_from_gep_users(inner_user)
                        inner_result !== nothing && return inner_result
                    end
                elseif inner_user isa LLVM.LoadInst
                    return LLVM.value_type(inner_user)
                elseif inner_user isa LLVM.StoreInst
                    if LLVM.operands(inner_user)[2] === user
                        return LLVM.value_type(LLVM.operands(inner_user)[1])
                    end
                elseif inner_user isa LLVM.AtomicRMWInst
                    return LLVM.value_type(LLVM.operands(inner_user)[2])
                elseif inner_user isa LLVM.AtomicCmpXchgInst
                    return LLVM.value_type(LLVM.operands(inner_user)[2])
                end
            end
        end
    end
    return nothing
end

"""
Emit a PSB pointer element access using manual byte-offset arithmetic.
OpPtrAccessChain on PSB struct pointers is broken on AMD RADV — all threads
read element[0]. Instead, convert ptr→u64, add idx*stride, convert u64→ptr.
Returns the result SPIR-V ID.
"""
function _emit_psb_ptr_arithmetic!(state::SPIRVEmitterState, base_id::UInt32,
                                     idx_id::UInt32, result_ptr_ty::UInt32,
                                     element_ty::LLVM.LLVMType;
                                     idx_llvm_ty::Union{LLVM.LLVMType, Nothing}=nothing)
    stride = UInt64(_compute_type_size(element_ty))
    u64_spirv = emit_type_int!(state.mod, UInt32(64), UInt32(0))

    # OpConvertPtrToU: base_ptr → u64 (cached per block)
    base_u64 = _cached_psb_ptr_to_u64!(state, base_id)

    # Ensure index is u64 — widen i32 indices to i64
    idx_u64 = idx_id
    if idx_llvm_ty !== nothing && idx_llvm_ty isa LLVM.IntegerType && LLVM.width(idx_llvm_ty) < 64
        idx_u64 = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpSConvert, u64_spirv, idx_u64, idx_id)
    end

    # byte_offset = idx * stride
    # Emit stride as u64 constant directly via SPIR-V
    stride_const = _emit_u64_constant!(state.mod, stride)
    byte_offset_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpIMul, u64_spirv, byte_offset_id, idx_u64, stride_const)

    # elem_addr = base_addr + byte_offset
    elem_addr_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpIAdd, u64_spirv, elem_addr_id, base_u64, byte_offset_id)

    # OpConvertUToPtr: u64 → ptr
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, result_ptr_ty, result_id, elem_addr_id)

    return result_id
end

"""Emit a u64 constant directly via SPIR-V type system (no LLVM type needed)."""
function _emit_u64_constant!(mod::SPIRVModule, value::UInt64)
    type_id = emit_type_int!(mod, UInt32(64), UInt32(0))
    lo = UInt32(value & 0xFFFFFFFF)
    hi = UInt32((value >> 32) & 0xFFFFFFFF)
    key = (:const, type_id, value)
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
        id
    end
end

"""
    _emit_function_ptr_word!(state, llvm_val, word_idx, ptr_ty) -> UInt32

Get a SPIR-V pointer to word `word_idx` (0-based) within a Function-class alloca.
Walks the LLVM IR operand chain to find the base alloca and compute the total word
offset from GEPs, then uses OpAccessChain to index into the alloca's array type.

OpConvertPtrToU is invalid for Function storage class pointers (SPIR-V spec requires
Physical storage class). Using integer arithmetic on Function pointers produces
data corruption on NVIDIA (wrong values read/written in loops).
"""
function _emit_function_ptr_word!(state::SPIRVEmitterState, llvm_val::LLVM.Value,
                                   word_idx::Int, ptr_ty::UInt32)
    # Walk LLVM IR operand chain to find the alloca base and total byte offset
    current = llvm_val
    total_byte_offset = 0
    while true
        if current isa LLVM.AllocaInst
            break
        elseif current isa LLVM.GetElementPtrInst
            # Compute byte offset from this GEP's indices
            gep_ops = LLVM.operands(current)
            source_ty_gep = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(current))
            byte_offset = _compute_gep_constant_byte_offset(source_ty_gep, gep_ops)
            total_byte_offset += byte_offset
            current = gep_ops[1]  # walk up to base pointer
        elseif current isa LLVM.BitCastInst
            current = LLVM.operands(current)[1]
        else
            # Fallback: use array_element_origin if available
            if haskey(state.array_element_origin, llvm_val)
                arr_base_id, base_idx_id, _ = state.array_element_origin[llvm_val]
                u32_ty = emit_type_int!(state.mod, UInt32(32), UInt32(0))
                word_const = emit_constant_u32!(state.mod, UInt32(word_idx))
                if word_idx == 0
                    result_id = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpAccessChain, ptr_ty,
                                        result_id, arr_base_id, base_idx_id)
                    return result_id
                else
                    combined_idx = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpIAdd, u32_ty,
                                        combined_idx, base_idx_id, word_const)
                    result_id = fresh_id!(state.mod)
                    encode_instruction!(state.mod.functions, Op.OpAccessChain, ptr_ty,
                                        result_id, arr_base_id, combined_idx)
                    return result_id
                end
            end
            error("Cannot resolve Function-class pointer to alloca for memcpy/memset: $(typeof(current)) = $current")
        end
    end

    # current is now the AllocaInst; compute the total word index
    alloca_id = get_value_id!(state, current)
    total_word_idx = total_byte_offset ÷ 4 + word_idx
    word_const = emit_constant_u32!(state.mod, UInt32(total_word_idx))
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpAccessChain, ptr_ty,
                        result_id, alloca_id, word_const)
    return result_id
end

"""
Compute the constant byte offset of a GEP from its source type and operands.
Used by `_emit_function_ptr_word!` to resolve alloca offsets through GEP chains.
"""
function _compute_gep_constant_byte_offset(source_ty::LLVM.LLVMType, ops)
    total = 0
    # First index (ops[2]) is the base element offset: offset = idx * sizeof(source_ty)
    first_idx = ops[2]
    if first_idx isa LLVM.ConstantInt
        idx_val = convert(Int64, first_idx)
        if idx_val != 0
            total += idx_val * _compute_type_size(source_ty)
        end
    end

    # Remaining indices drill into the type hierarchy
    current_ty = source_ty
    for i in 3:length(ops)
        idx_op = ops[i]
        idx_val = idx_op isa LLVM.ConstantInt ? convert(Int64, idx_op) : 0
        if current_ty isa LLVM.StructType
            elems = LLVM.elements(current_ty)
            for j in 0:(idx_val - 1)
                total += _compute_type_size(elems[j + 1])
            end
            current_ty = elems[idx_val + 1]
        elseif current_ty isa LLVM.ArrayType
            elem_ty = LLVM.eltype(current_ty)
            total += idx_val * _compute_type_size(elem_ty)
            current_ty = elem_ty
        else
            total += idx_val * _compute_type_size(current_ty)
        end
    end
    return total
end

"""
Emit memset as scalar store pairs via PSB pointer arithmetic.
LLVM uses memset for zeroing structs (e.g., zero(ComplexF64) → memset 0, 16 bytes).
"""
function _emit_memset!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dest_ptr = ops[1]
    fill_val = ops[2]   # i8 value to fill with
    size_val = ops[3]

    # Size must be a constant
    if !(size_val isa LLVM.ConstantInt)
        @warn "memset with non-constant size, skipping" size_val
        return
    end
    nbytes = convert(Int, size_val)
    nbytes == 0 && return

    # For shared memory (addrspace 3), skip — GPU doesn't require zero-init
    dest_as = LLVM.value_type(dest_ptr) isa LLVM.PointerType ? LLVM.addrspace(LLVM.value_type(dest_ptr)) : 0
    if dest_as == 3
        return  # Skip shared memory memset
    end

    dest_id = get_value_id!(state, dest_ptr)
    dest_sc = _get_pointer_storage_class(dest_ptr)
    is_dest_psb = dest_sc == SC.PhysicalStorageBuffer
    is_dest_function = dest_sc == SC.Function

    u32_ty = emit_type_int!(state.mod, UInt32(32), UInt32(0))
    u64_ty = emit_type_int!(state.mod, UInt32(64), UInt32(0))
    dest_ptr_ty = map_pointer_type!(state.type_ctx, u32_ty, dest_sc)

    # Build the i32 fill word from the i8 fill value
    fill_word = if fill_val isa LLVM.ConstantInt
        v = convert(UInt8, fill_val) & 0xFF
        UInt32(v) | (UInt32(v) << 8) | (UInt32(v) << 16) | (UInt32(v) << 24)
    else
        UInt32(0)  # Non-constant fill value — assume 0
    end
    fill_const = map_constant!(state.type_ctx, LLVM.ConstantInt(LLVM.IntType(32), fill_word))

    nwords = nbytes ÷ 4

    if is_dest_function
        # Function storage class: use OpAccessChain instead of integer arithmetic
        # (OpConvertPtrToU is invalid for Function pointers)
        for i in 0:(nwords - 1)
            elem_id = _emit_function_ptr_word!(state, dest_ptr, i, dest_ptr_ty)
            encode_instruction!(state.mod.functions, Op.OpStore, elem_id, fill_const)
        end
    else
        # PSB or other storage class: use integer arithmetic
        dest_u64 = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_ty, dest_u64, dest_id)

        for i in 0:(nwords - 1)
            offset = UInt64(i * 4)
            addr_u64 = if offset == 0
                dest_u64
            else
                offset_const = _emit_u64_constant!(state.mod, offset)
                addr_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpIAdd, u64_ty, addr_id, dest_u64, offset_const)
                addr_id
            end
            elem_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, dest_ptr_ty, elem_id, addr_u64)
            if is_dest_psb
                encode_instruction!(state.mod.functions, Op.OpStore, elem_id, fill_const, UInt32(0x02), UInt32(4))
            else
                encode_instruction!(state.mod.functions, Op.OpStore, elem_id, fill_const)
            end
        end
    end
end

"""
Emit memcpy/memmove as scalar load/store pairs via PSB pointer arithmetic.
Handles struct copies like ComplexF64 (16 bytes) that LLVM optimizes to memcpy.
"""
function _emit_memcpy!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dest_ptr = ops[1]
    src_ptr = ops[2]
    size_val = ops[3]

    # Size must be a constant
    if !(size_val isa LLVM.ConstantInt)
        @warn "memcpy/memmove with non-constant size, skipping" size_val
        return
    end
    nbytes = convert(Int, size_val)
    if nbytes == 0
        return
    end

    dest_id = get_value_id!(state, dest_ptr)
    src_id = get_value_id!(state, src_ptr)

    # Determine storage classes
    dest_sc = _get_pointer_storage_class(dest_ptr)
    src_sc = _get_pointer_storage_class(src_ptr)

    # Use i32 for word-by-word copy
    u32_ty = emit_type_int!(state.mod, UInt32(32), UInt32(0))
    u64_ty = emit_type_int!(state.mod, UInt32(64), UInt32(0))

    # Create pointer types for i32 in appropriate storage classes
    src_ptr_ty = map_pointer_type!(state.type_ctx, u32_ty, src_sc)
    dest_ptr_ty = map_pointer_type!(state.type_ctx, u32_ty, dest_sc)

    # Memory operands for aligned access
    is_src_psb = src_sc == SC.PhysicalStorageBuffer
    is_dest_psb = dest_sc == SC.PhysicalStorageBuffer
    is_src_function = src_sc == SC.Function
    is_dest_function = dest_sc == SC.Function

    # Only compute u64 addresses for non-Function pointers
    # (OpConvertPtrToU is invalid for Function storage class)
    src_u64 = if !is_src_function
        id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_ty, id, src_id)
        id
    else
        UInt32(0)  # unused
    end
    dest_u64 = if !is_dest_function
        id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpConvertPtrToU, u64_ty, id, dest_id)
        id
    else
        UInt32(0)  # unused
    end

    # Copy 4 bytes at a time (i32)
    nwords = nbytes ÷ 4
    remainder = nbytes % 4

    for i in 0:(nwords - 1)
        offset = UInt64(i * 4)

        # Source pointer for this word
        src_elem_id = if is_src_function
            # Function storage class: use OpAccessChain from alloca base
            _emit_function_ptr_word!(state, src_ptr, i, src_ptr_ty)
        else
            # PSB or other: use integer arithmetic
            src_addr_u64 = if offset == 0
                src_u64
            else
                offset_const = _emit_u64_constant!(state.mod, offset)
                addr_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpIAdd, u64_ty, addr_id, src_u64, offset_const)
                addr_id
            end
            id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, src_ptr_ty, id, src_addr_u64)
            id
        end

        # Load i32 from source
        load_id = fresh_id!(state.mod)
        if is_src_psb
            encode_instruction!(state.mod.functions, Op.OpLoad, u32_ty, load_id, src_elem_id,
                                UInt32(0x02), UInt32(4))  # Aligned 4
        else
            encode_instruction!(state.mod.functions, Op.OpLoad, u32_ty, load_id, src_elem_id)
        end

        # Dest pointer for this word
        dest_elem_id = if is_dest_function
            # Function storage class: use OpAccessChain from alloca base
            _emit_function_ptr_word!(state, dest_ptr, i, dest_ptr_ty)
        else
            # PSB or other: use integer arithmetic
            dest_addr_u64 = if offset == 0
                dest_u64
            else
                offset_const = _emit_u64_constant!(state.mod, offset)
                addr_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpIAdd, u64_ty, addr_id, dest_u64, offset_const)
                addr_id
            end
            id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, dest_ptr_ty, id, dest_addr_u64)
            id
        end

        # Store i32 to dest
        if is_dest_psb
            encode_instruction!(state.mod.functions, Op.OpStore, dest_elem_id, load_id,
                                UInt32(0x02), UInt32(4))  # Aligned 4
        else
            encode_instruction!(state.mod.functions, Op.OpStore, dest_elem_id, load_id)
        end
    end

    # Handle remaining bytes (< 4) with i8 copies
    # NOTE: Function-class remainder is unlikely (LLVM aligns allocas to 4 bytes)
    # but handled for correctness
    if remainder > 0
        u8_ty = emit_type_int!(state.mod, UInt32(8), UInt32(0))
        src_u8_ptr_ty = map_pointer_type!(state.type_ctx, u8_ty, src_sc)
        dest_u8_ptr_ty = map_pointer_type!(state.type_ctx, u8_ty, dest_sc)

        for i in 0:(remainder - 1)
            offset = UInt64(nwords * 4 + i)
            offset_const = _emit_u64_constant!(state.mod, offset)

            # Source byte
            addr_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIAdd, u64_ty, addr_id, src_u64, offset_const)
            src_byte_ptr = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, src_u8_ptr_ty, src_byte_ptr, addr_id)

            load_id = fresh_id!(state.mod)
            if is_src_psb
                encode_instruction!(state.mod.functions, Op.OpLoad, u8_ty, load_id, src_byte_ptr,
                                    UInt32(0x02), UInt32(1))
            else
                encode_instruction!(state.mod.functions, Op.OpLoad, u8_ty, load_id, src_byte_ptr)
            end

            # Dest byte
            addr_id2 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIAdd, u64_ty, addr_id2, dest_u64, offset_const)
            dest_byte_ptr = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, dest_u8_ptr_ty, dest_byte_ptr, addr_id2)

            if is_dest_psb
                encode_instruction!(state.mod.functions, Op.OpStore, dest_byte_ptr, load_id,
                                    UInt32(0x02), UInt32(1))
            else
                encode_instruction!(state.mod.functions, Op.OpStore, dest_byte_ptr, load_id)
            end
        end
    end
end

"""
Ensure a pointer type used with OpPtrAccessChain has an ArrayStride decoration.
Required by SPIR-V validation for OpPtrAccessChain.
"""
function _ensure_array_stride_decoration!(state::SPIRVEmitterState, ptr_type_id::UInt32, element_ty::LLVM.LLVMType)
    # Track which pointer types already have ArrayStride decoration
    if !haskey(state.mod.constant_cache, (:array_stride, ptr_type_id))
        stride = UInt32(_compute_type_size(element_ty))
        emit_decorate!(state.mod, ptr_type_id, Dec.ArrayStride, stride)
        state.mod.constant_cache[(:array_stride, ptr_type_id)] = stride
    end
end

"""Compute the byte size of an LLVM type (for ArrayStride decoration)."""
# Compute the natural alignment of an LLVM type in bytes.
function _compute_type_alignment(ty::LLVM.LLVMType)
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
            max_align = max(max_align, _compute_type_alignment(elem))
        end
        return max_align
    elseif ty isa LLVM.ArrayType
        return _compute_type_alignment(eltype(ty))
    elseif ty isa LLVM.PointerType
        return 8
    else
        return 4
    end
end

function _compute_type_size(ty::LLVM.LLVMType)
    if ty isa LLVM.LLVMFloat
        return UInt32(4)
    elseif ty isa LLVM.LLVMDouble
        return UInt32(8)
    elseif ty isa LLVM.LLVMHalf
        return UInt32(2)
    elseif ty isa LLVM.IntegerType
        return UInt32(max(1, LLVM.width(ty) ÷ 8))
    elseif ty isa LLVM.StructType
        # Compute struct size with alignment padding (matching LLVM's DataLayout).
        total = UInt32(0)
        struct_align = UInt32(1)
        for elem in LLVM.elements(ty)
            elem_align = UInt32(_compute_type_alignment(elem))
            struct_align = max(struct_align, elem_align)
            # Align the offset
            total = (total + elem_align - 1) & ~(elem_align - 1)
            total += _compute_type_size(elem)
        end
        # Pad to struct alignment
        total = (total + struct_align - 1) & ~(struct_align - 1)
        return total
    elseif ty isa LLVM.ArrayType
        return UInt32(length(ty)) * _compute_type_size(eltype(ty))
    elseif ty isa LLVM.PointerType
        return UInt32(8)  # 64-bit pointers
    else
        return UInt32(4)  # Default
    end
end

function _emit_alloca!(state::SPIRVEmitterState, inst::LLVM.AllocaInst)
    alloc_ty = LLVM.LLVMType(API.LLVMGetAllocatedType(inst))
    pointee_spirv = map_type!(state.type_ctx, alloc_ty)
    ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, SC.Function)
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpVariable, ptr_ty, result_id, SC.Function)
    state.value_map[inst] = result_id
    # Register pointee type so byte-offset GEPs (gep i8, ptr %alloca, i64 %offset)
    # can resolve the actual element type and emit proper OpAccessChain.
    set_pointee_type!(state.type_ctx.ptm, inst, alloc_ty; priority=5)
end

# ================================================================
# Conversions
# ================================================================

function _emit_conversion!(state::SPIRVEmitterState, inst::LLVM.Instruction, opcode::UInt16)
    ops = LLVM.operands(inst)
    src = get_value_id!(state, ops[1])
    llvm_ty = LLVM.value_type(inst)
    result_ty = if llvm_ty isa LLVM.PointerType
        map_pointer_type_for_value!(state.type_ctx, inst)
    else
        map_type!(state.type_ctx, llvm_ty)
    end
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, opcode, result_ty, result_id, src)
    state.value_map[inst] = result_id
end

function _emit_trunc!(state::SPIRVEmitterState, inst::LLVM.TruncInst)
    # trunc can be i64→i32, i32→i16, etc.
    # For i-to-i: OpUConvert (or OpSConvert, but signedness is per-instruction in SPIR-V)
    src_ty = LLVM.value_type(LLVM.operands(inst)[1])
    dst_ty = LLVM.value_type(inst)
    if src_ty isa LLVM.IntegerType && dst_ty isa LLVM.IntegerType
        # Integer truncation
        if LLVM.width(dst_ty) == 1
            # trunc to i1: compare != 0
            src_id = get_value_id!(state, LLVM.operands(inst)[1])
            zero_id = _get_zero_constant!(state, src_ty)
            result_ty = map_type!(state.type_ctx, dst_ty)
            result_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpINotEqual, result_ty, result_id, src_id, zero_id)
            state.value_map[inst] = result_id
        else
            _emit_conversion!(state, inst, Op.OpUConvert)
        end
    else
        _emit_conversion!(state, inst, Op.OpUConvert)
    end
end

function _emit_sext!(state::SPIRVEmitterState, inst::LLVM.SExtInst)
    src_ty = LLVM.value_type(LLVM.operands(inst)[1])
    dst_ty = LLVM.value_type(inst)
    if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 1
        # sext i1 to iN: true → all-ones (-1), false → 0
        src_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty_id = map_type!(state.type_ctx, dst_ty)
        # sext true = all 1s = -1 in two's complement
        w = LLVM.width(dst_ty)
        all_ones = w >= 64 ? typemax(UInt64) : UInt64((UInt64(1) << w) - 1)
        one_id = _emit_int_constant!(state, dst_ty, all_ones)
        zero_id = _emit_int_constant!(state, dst_ty, UInt64(0))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpSelect, result_ty_id, result_id, src_id, one_id, zero_id)
        state.value_map[inst] = result_id
    else
        _emit_conversion!(state, inst, Op.OpSConvert)
    end
end

function _emit_zext!(state::SPIRVEmitterState, inst::LLVM.ZExtInst)
    src_ty = LLVM.value_type(LLVM.operands(inst)[1])
    dst_ty = LLVM.value_type(inst)
    if src_ty isa LLVM.IntegerType && LLVM.width(src_ty) == 1
        # zext i1 to iN: OpSelect %dst_type %bool_val %one %zero
        # SPIR-V OpBool can't be converted with OpUConvert
        src_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty_id = map_type!(state.type_ctx, dst_ty)
        one_id = _emit_int_constant!(state, dst_ty, UInt64(1))
        zero_id = _emit_int_constant!(state, dst_ty, UInt64(0))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpSelect, result_ty_id, result_id, src_id, one_id, zero_id)
        state.value_map[inst] = result_id
    else
        _emit_conversion!(state, inst, Op.OpUConvert)
    end
end

"""Create an integer constant with the given type and value."""
function _emit_int_constant!(state::SPIRVEmitterState, ty::LLVM.IntegerType, val::UInt64)
    type_id = map_type!(state.type_ctx, ty)
    w = LLVM.width(ty)
    if w <= 32
        bits = UInt32(val & 0xFFFFFFFF)
        key = (:const, type_id, bits)
        return get!(state.mod.constant_cache, key) do
            id = fresh_id!(state.mod)
            encode_instruction!(state.mod.types_constants, Op.OpConstant, type_id, id, bits)
            id
        end
    else
        key = (:const, type_id, val)
        return get!(state.mod.constant_cache, key) do
            id = fresh_id!(state.mod)
            lo = UInt32(val & 0xFFFFFFFF)
            hi = UInt32((val >> 32) & 0xFFFFFFFF)
            encode_instruction!(state.mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
            id
        end
    end
end

function _emit_inttoptr!(state::SPIRVEmitterState, inst::LLVM.IntToPtrInst)
    # inttoptr i64 %val to ptr addrspace(N)
    # In Vulkan SPIR-V, OpConvertUToPtr REQUIRES PhysicalStorageBuffer storage class.
    # inttoptr always produces a PSB pointer (it reconstructs a device address).
    src = get_value_id!(state, LLVM.operands(inst)[1])
    result_ptr_ty = LLVM.value_type(inst)
    if result_ptr_ty isa LLVM.PointerType
        # inttoptr always produces PhysicalStorageBuffer pointers
        # (OpConvertUToPtr requires this storage class)
        sc = SC.PhysicalStorageBuffer
        # Try to get pointee type from map, fall back to use-based inference
        pointee = get_pointee_type(state.type_ctx.ptm, inst)
        if pointee === nothing
            pointee = _infer_pointee_from_users(inst)
        end
        if pointee === nothing
            error("Cannot determine pointee type for inttoptr result: $inst")
        end
        # If pointee is itself a pointer type (e.g. loading a Ptr{T} from a struct),
        # we need to resolve it to a typed SPIR-V pointer.
        if pointee isa LLVM.PointerType
            # The inttoptr result points to a pointer. Find what THAT pointer points to
            # by looking at the users of the load that consumes this inttoptr.
            inner_pointee = _infer_inner_ptr_pointee(inst)
            if inner_pointee === nothing
                inner_pointee = LLVM.IntType(8)  # fallback: ptr to i8
            end
            inner_spirv = map_type!(state.type_ctx, inner_pointee)
            pointee_spirv = map_pointer_type!(state.type_ctx, inner_spirv, sc)
        else
            pointee_spirv = map_type!(state.type_ctx, pointee)
        end
        ptr_spirv = map_pointer_type!(state.type_ctx, pointee_spirv, sc)
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpConvertUToPtr, ptr_spirv, result_id, src)
        state.value_map[inst] = result_id
    else
        _emit_conversion!(state, inst, Op.OpConvertUToPtr)
    end
end

"""
Infer the pointee type of an inner pointer loaded through a PSB inttoptr.
The pattern is: inttoptr → load ptr → [store to alloca → load from alloca] → use as pointer.
We need to find what type the ultimate use stores/loads through that pointer.
"""
function _infer_inner_ptr_pointee(inttoptr_inst::LLVM.Value)
    # Find loads from this inttoptr (which load a ptr value)
    for use in LLVM.uses(inttoptr_inst)
        user = LLVM.user(use)
        if user isa LLVM.LoadInst
            # The loaded ptr — check direct uses first
            result = _infer_pointee_from_users(user)
            result !== nothing && return result

            # Follow store→alloca→load chain:
            # The loaded ptr might be stored to an alloca, then reloaded
            for use2 in LLVM.uses(user)
                user2 = LLVM.user(use2)
                if user2 isa LLVM.StoreInst
                    # user2: store ptr %loaded, ptr %alloca
                    alloca = LLVM.operands(user2)[2]
                    if alloca isa LLVM.AllocaInst
                        # Find loads from this alloca
                        for use3 in LLVM.uses(alloca)
                            user3 = LLVM.user(use3)
                            if user3 isa LLVM.LoadInst && user3 !== user
                                result = _infer_pointee_from_users(user3)
                                result !== nothing && return result
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _get_zero_constant!(state::SPIRVEmitterState, ty::LLVM.IntegerType)
    w = LLVM.width(ty)
    if w <= 32
        return emit_constant_u32!(state.mod, UInt32(0))
    else
        type_id = map_type!(state.type_ctx, ty)
        key = (:const, type_id, UInt64(0))
        get!(state.mod.constant_cache, key) do
            id = fresh_id!(state.mod)
            encode_instruction!(state.mod.types_constants, Op.OpConstant, type_id, id, UInt32(0), UInt32(0))
            id
        end
    end
end

# ================================================================
# Control Flow
# ================================================================

function _emit_ret!(state::SPIRVEmitterState, inst::LLVM.RetInst)
    # Skip if block was already terminated by OpIgnoreIntersectionKHR/OpTerminateRayKHR
    state.rt_block_terminated && return
    ops = LLVM.operands(inst)
    if isempty(ops)
        encode_instruction!(state.mod.functions, Op.OpReturn)
    else
        val_id = get_value_id!(state, ops[1])
        encode_instruction!(state.mod.functions, Op.OpReturnValue, val_id)
    end
end

function _emit_br!(state::SPIRVEmitterState, inst::LLVM.BrInst)
    current_bb = LLVM.parent(inst)

    if LLVM.isconditional(inst)
        cond = get_value_id!(state, LLVM.condition(inst))
        true_bb = LLVM.successors(inst)[1]
        false_bb = LLVM.successors(inst)[2]
        true_id = get_block_id!(state, true_bb)
        false_id = get_block_id!(state, false_bb)

        if haskey(state.loop_info.loops, current_bb)
            # Loop header with conditional branch — OpLoopMerge already emitted in emit_block!
            merge_bb, continue_bb = state.loop_info.loops[current_bb]
            merge_id_loop = get_block_id!(state, merge_bb)
            continue_id_loop = get_block_id!(state, continue_bb)
            targets_merge = (true_id == merge_id_loop || false_id == merge_id_loop)
            targets_continue = (true_id == continue_id_loop || false_id == continue_id_loop)
            if !targets_merge && !targets_continue
                # Neither branch target is the loop merge or continue block — both paths
                # stay inside the loop body. SPIR-V can't have OpLoopMerge + OpSelectionMerge
                # in one block, so we split: the loop header unconditionally branches to a new
                # selection header block that contains OpSelectionMerge + OpBranchConditional.
                sel_header_id = fresh_id!(state.mod)
                header_id = get_block_id!(state, current_bb)
                encode_instruction!(state.mod.functions, Op.OpBranch, sel_header_id)
                encode_instruction!(state.mod.functions, Op.OpLabel, sel_header_id)
                # Find which block the two paths reconverge at (inner selection merge)
                inner_merge_id = _find_inner_selection_merge(state, true_bb, false_bb, merge_bb)
                encode_instruction!(state.mod.functions, Op.OpSelectionMerge, inner_merge_id, UInt32(0))
                # Register redirect: PHIs referencing the original header as predecessor
                # must reference the new selection header instead.
                state.phi_block_redirects[(header_id, true_id)] = sel_header_id
                state.phi_block_redirects[(header_id, false_id)] = sel_header_id
            end
        elseif _is_continue_exit_branch(state, current_bb, true_bb, false_bb)
            # Continue block branching to loop header (back-edge) and loop merge (exit).
            # This is a structured loop exit from the continue construct.
            # The continue block IS the back-edge block, so the continue construct = {this block}.
            # No OpSelectionMerge needed — the OpLoopMerge at the header already covers this.
        else
            # Selection: find the merge block, avoiding conflicts
            merge_id = _find_merge_block(state, inst)
            current_id = get_block_id!(state, current_bb)

            # Check if merge target needs a trampoline:
            # 1. Already claimed as merge for another construct
            # 2. Is a loop continue target of an enclosing loop (SPIR-V forbids
            #    selection merge pointing to the continue construct of an outer loop)
            # 3. Is a loop merge target of an enclosing loop (same containment issue)
            needs_trampoline = merge_id in state.used_merge_blocks ||
                               _is_loop_continue_or_merge(state, merge_id, current_bb)
            if needs_trampoline
                # Conflict: create a trampoline block that branches to the real target.
                # The trampoline becomes the new merge target.
                trampoline_id = _get_or_create_trampoline!(state, current_id, merge_id)
                encode_instruction!(state.mod.functions, Op.OpSelectionMerge, trampoline_id, UInt32(0))
                push!(state.used_merge_blocks, trampoline_id)

                # Redirect whichever branch was going to the merge to go to the trampoline
                if true_id == merge_id
                    true_id = trampoline_id
                end
                if false_id == merge_id
                    false_id = trampoline_id
                end
            else
                encode_instruction!(state.mod.functions, Op.OpSelectionMerge, merge_id, UInt32(0))
                push!(state.used_merge_blocks, merge_id)
            end
        end
        encode_instruction!(state.mod.functions, Op.OpBranchConditional, cond, true_id, false_id)
    else
        # Unconditional branch — OpLoopMerge (if any) already emitted in emit_block!
        target = get_block_id!(state, LLVM.successors(inst)[1])
        encode_instruction!(state.mod.functions, Op.OpBranch, target)
    end
end

"""
    _analyze_loops(fn::LLVM.Function) -> LoopInfo

Pre-analyze the CFG to identify loop headers and their merge/continue targets.
A loop header is any block that is the target of a back-edge (edge from a block
that comes later in RPO order).
"""
function _analyze_loops(fn::LLVM.Function)
    loops = Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}()
    blocks = collect(LLVM.blocks(fn))
    isempty(blocks) && return LoopInfo(loops)

    # Compute RPO position for each block
    rpo = _reverse_postorder(fn)
    rpo_pos = Dict{LLVM.BasicBlock, Int}()
    for (i, bb) in enumerate(rpo)
        rpo_pos[bb] = i
    end

    # Find back-edges: edge A→B where B appears before A in RPO
    for bb in blocks
        term = LLVM.terminator(bb)
        bb_pos = get(rpo_pos, bb, 0)
        for succ in LLVM.successors(term)
            succ_pos = get(rpo_pos, succ, 0)
            if succ_pos > 0 && succ_pos <= bb_pos
                # Back-edge: bb → succ. succ is the loop header, bb is the latch/continue
                header = succ
                latch = bb
                haskey(loops, header) && continue  # Already found this loop

                merge_bb = _find_loop_merge(header, latch, rpo, rpo_pos)
                loops[header] = (merge_bb, latch)
            end
        end
    end

    return LoopInfo(loops)
end

"""Find the merge block for a loop: the first successor of a loop block that's outside the loop."""
function _find_loop_merge(header::LLVM.BasicBlock, latch::LLVM.BasicBlock,
                           rpo::Vector{LLVM.BasicBlock}, rpo_pos::Dict{LLVM.BasicBlock, Int})
    header_pos = rpo_pos[header]
    latch_pos = rpo_pos[latch]

    # Loop blocks are between header and latch in RPO order
    loop_blocks = Set{LLVM.BasicBlock}()
    for bb in rpo
        pos = rpo_pos[bb]
        if pos >= header_pos && pos <= latch_pos
            push!(loop_blocks, bb)
        end
    end

    # Find the first successor of any loop block that's outside the loop
    for bb in rpo
        bb in loop_blocks || continue
        for succ in LLVM.successors(LLVM.terminator(bb))
            if !(succ in loop_blocks)
                return succ
            end
        end
    end

    # Fallback: block after latch in RPO
    if latch_pos < length(rpo)
        return rpo[latch_pos + 1]
    end
    return latch
end

"""
Find the merge block for a conditional branch using the immediate post-dominator tree.
The ipdom of the branch's parent block is the correct merge block for SPIR-V's
OpSelectionMerge, since the merge is where all paths from the selection reconverge.
"""
function _find_merge_block(state::SPIRVEmitterState, inst::LLVM.BrInst)
    # After StructurizeCFG, one of the two branch targets is the merge (convergence) point.
    # The merge is the target that the OTHER target's path eventually reaches.
    # Two patterns:
    #   Pattern 1 (if-then): true does work, false is merge. false_bb reachable from true path.
    #   Pattern 2 (if-else-then): false does work, true is merge. true_bb reachable from false path.
    #
    # We check reachability: follow forward edges from one branch target to see if it
    # reaches the other (without going through loop back-edges or the header itself).
    current_bb = LLVM.parent(inst)
    true_bb = LLVM.successors(inst)[1]
    false_bb = LLVM.successors(inst)[2]

    # Check if true_bb is reachable from false_bb's path (pattern 2: true is merge)
    if _is_forward_reachable(false_bb, true_bb, current_bb, state)
        return get_block_id!(state, true_bb)
    end

    # Default: false branch is merge (pattern 1, most common after StructurizeCFG)
    return get_block_id!(state, false_bb)
end

"""
Check if `target` is reachable from `start` following forward edges only,
without going through `avoid` (the header block). Uses BFS.
Only blocks back-edges: edges from a loop's continue block back to its header.
"""
function _is_forward_reachable(start::LLVM.BasicBlock, target::LLVM.BasicBlock,
                                avoid::LLVM.BasicBlock, state::SPIRVEmitterState)
    visited = Set{LLVM.BasicBlock}()
    queue = LLVM.BasicBlock[start]
    push!(visited, start)
    push!(visited, avoid)  # Don't go through the header

    while !isempty(queue)
        bb = popfirst!(queue)
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            succ === target && return true
            if succ ∉ visited
                # Only block actual back-edges: from a loop's continue block to its header
                is_backedge = false
                if haskey(state.loop_info.loops, succ)
                    # succ is a loop header. This is a back-edge only if bb is the
                    # continue block for that loop.
                    _, continue_bb = state.loop_info.loops[succ]
                    is_backedge = (bb === continue_bb)
                end
                if !is_backedge
                    push!(visited, succ)
                    push!(queue, succ)
                end
            end
        end
    end
    return false
end

"""
Check if `merge_id` (SPIR-V block ID) corresponds to a loop continue target or
loop merge target of a loop that contains `current_bb`. A selection construct inside
a loop must not have its merge block be the continue or merge target of that loop,
because those blocks are in different constructs (continue construct / after the loop).
"""
function _is_loop_continue_or_merge(state::SPIRVEmitterState, merge_id::UInt32,
                                      current_bb::LLVM.BasicBlock)
    for (header, (loop_merge_bb, continue_bb)) in state.loop_info.loops
        continue_id = get_block_id!(state, continue_bb)
        loop_merge_id = get_block_id!(state, loop_merge_bb)
        if merge_id == continue_id || merge_id == loop_merge_id
            # Check if current_bb is inside this loop (i.e., it's not the header
            # or the merge/continue themselves)
            # A block is inside the loop if it's dominated by the header
            # and is not the merge block itself.
            # Simple check: current_bb != header (the header case is handled separately)
            # and current_bb != continue_bb and current_bb != loop_merge_bb
            if current_bb != header && current_bb != continue_bb && current_bb != loop_merge_bb
                return true
            end
        end
    end
    return false
end

# Check if a conditional branch from a continue block targets the loop header and merge.
# This is the "loop exit from continue" pattern — no OpSelectionMerge needed.
function _is_continue_exit_branch(state::SPIRVEmitterState,
                                   current_bb::LLVM.BasicBlock,
                                   true_bb::LLVM.BasicBlock,
                                   false_bb::LLVM.BasicBlock)
    for (header, (merge_bb, continue_bb)) in state.loop_info.loops
        if continue_bb == current_bb
            # This IS the continue block of a loop.
            # Check if branches go to {header, merge} in either order.
            targets = Set([true_bb, false_bb])
            if header in targets && merge_bb in targets
                return true
            end
            # Also match: one target is the header, the other leads to the merge
            if header in targets || merge_bb in targets
                return true
            end
        end
    end
    return false
end

"""
Find the merge block for a selection inside a loop header.
Both true_bb and false_bb are inside the loop (neither is the loop merge block).
The inner merge is where both paths reconverge.

After StructurizeCFG, two patterns occur:
1. true_bb does work, then joins at false_bb (false_bb is merge) — "if-then" pattern
2. false_bb does work, then joins at true_bb (true_bb is merge) — "if-else" pattern

Pattern 2 happens when the loop header's condition gates a skip: if true, go directly
to the flow block (convergence); if false, do the body then reach the same flow block.
In this case true_bb IS the merge and one branch target equals the merge block, which
is valid in SPIR-V.
"""
function _find_inner_selection_merge(state::SPIRVEmitterState,
                                      true_bb::LLVM.BasicBlock,
                                      false_bb::LLVM.BasicBlock,
                                      loop_merge_bb::LLVM.BasicBlock)
    # After StructurizeCFG, one branch does work and the other is the merge (skip).
    # Use _is_forward_reachable to determine which is which.
    # Must avoid back-edges AND the loop merge block.

    # Check pattern 1: true does work, false is merge
    # (true_bb eventually reaches false_bb without back-edges)
    if _is_forward_reachable_inner(true_bb, false_bb, loop_merge_bb, state)
        return get_block_id!(state, false_bb)
    end

    # Check pattern 2: false does work, true is merge
    # (false_bb eventually reaches true_bb without back-edges)
    if _is_forward_reachable_inner(false_bb, true_bb, loop_merge_bb, state)
        return get_block_id!(state, true_bb)
    end

    # Fallback: use false_bb as merge (most common after StructurizeCFG)
    return get_block_id!(state, false_bb)
end

"""
BFS reachability check that stops at `avoid` blocks and doesn't follow
back-edges to loop headers. Used by _find_inner_selection_merge.
"""
function _is_forward_reachable_inner(start::LLVM.BasicBlock, target::LLVM.BasicBlock,
                                      loop_merge_bb::LLVM.BasicBlock,
                                      state::SPIRVEmitterState)
    visited = Set{LLVM.BasicBlock}()
    queue = LLVM.BasicBlock[start]
    push!(visited, start)

    while !isempty(queue)
        bb = popfirst!(queue)
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            succ === target && return true
            succ === loop_merge_bb && continue
            if succ ∉ visited
                # Don't follow back-edges to loop headers
                is_backedge = false
                if haskey(state.loop_info.loops, succ)
                    _, continue_bb = state.loop_info.loops[succ]
                    is_backedge = (bb === continue_bb)
                end
                if !is_backedge
                    push!(visited, succ)
                    push!(queue, succ)
                end
            end
        end
    end
    return false
end

# Get or create a trampoline block that branches from current_block to target_id.
# Returns the trampoline's label ID. The trampoline contains just OpLabel + OpBranch.
function _get_or_create_trampoline!(state::SPIRVEmitterState, from_id::UInt32, target_id::UInt32)
    key = (from_id, target_id)
    existing = get(state.trampolines, key, nothing)
    if existing !== nothing
        return existing
    end
    trampoline_id = fresh_id!(state.mod)
    state.trampolines[key] = trampoline_id
    return trampoline_id
end

# Emit trampolines whose target is one of the upcoming blocks.
# Called after emitting each block to insert trampolines in the right position.
# A trampoline must appear before any block that it dominates.
function _emit_pending_trampolines!(state::SPIRVEmitterState, just_emitted::LLVM.BasicBlock,
                                      rpo_blocks::Vector{LLVM.BasicBlock})
    isempty(state.trampolines) && return
    just_emitted_id = get_block_id!(state, just_emitted)

    # Emit trampolines that branch FROM a block we've already emitted
    # and whose target is the just-emitted block or later.
    # This ensures they appear in the right position.
    emitted_keys = Tuple{UInt32, UInt32}[]
    for ((from_id, target_id), trampoline_id) in state.trampolines
        # Emit the trampoline right after the block that references it
        if from_id == just_emitted_id
            encode_instruction!(state.mod.functions, Op.OpLabel, trampoline_id)
            encode_instruction!(state.mod.functions, Op.OpBranch, target_id)
            push!(emitted_keys, (from_id, target_id))
        end
    end
    for key in emitted_keys
        # Preserve the redirect info for phi resolution — trampolines redirect
        # edges from the original block to the target through the trampoline.
        state.phi_block_redirects[key] = state.trampolines[key]
        delete!(state.trampolines, key)
    end
end

# Emit any remaining trampolines not emitted during block processing.
function _emit_remaining_trampolines!(state::SPIRVEmitterState)
    for ((from_id, target_id), trampoline_id) in state.trampolines
        encode_instruction!(state.mod.functions, Op.OpLabel, trampoline_id)
        encode_instruction!(state.mod.functions, Op.OpBranch, target_id)
        state.phi_block_redirects[(from_id, target_id)] = trampoline_id
    end
    empty!(state.trampolines)
end

function _emit_select!(state::SPIRVEmitterState, inst::LLVM.SelectInst)
    ops = LLVM.operands(inst)
    cond = get_value_id!(state, ops[1])
    true_val = get_value_id!(state, ops[2])
    false_val = get_value_id!(state, ops[3])
    llvm_ty = LLVM.value_type(inst)
    result_ty = if llvm_ty isa LLVM.PointerType
        # Opaque pointer — try to get type from true/false operands first
        pointee = get_pointee_type(state.type_ctx.ptm, ops[2])
        if pointee === nothing
            pointee = get_pointee_type(state.type_ctx.ptm, ops[3])
        end
        if pointee !== nothing
            set_pointee_type!(state.type_ctx.ptm, inst, pointee; priority=3)
        end
        map_pointer_type_for_value!(state.type_ctx, inst)
    else
        map_type!(state.type_ctx, llvm_ty)
    end
    # For pointer selects, ensure both operands have the same SPIR-V type as result.
    # LLVM opaque pointers are all `ptr`, but SPIR-V has distinct typed pointers.
    # Try to map each operand's pointer type; if it differs from result_ty, bitcast.
    if llvm_ty isa LLVM.PointerType
        for (i, op_llvm) in ((2, ops[2]), (3, ops[3]))
            op_pointee = get_pointee_type(state.type_ctx.ptm, op_llvm)
            res_pointee = get_pointee_type(state.type_ctx.ptm, inst)
            if op_pointee !== nothing && res_pointee !== nothing && op_pointee != res_pointee
                val_to_cast = i == 2 ? true_val : false_val
                cast_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpBitcast, result_ty, cast_id, val_to_cast)
                if i == 2
                    true_val = cast_id
                else
                    false_val = cast_id
                end
            end
        end
    end
    result_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpSelect, result_ty, result_id, cond, true_val, false_val)
    state.value_map[inst] = result_id
end

# ================================================================
# PHI Nodes (deferred emission)
# ================================================================

"""
Defer a PHI node for later insertion after all blocks are emitted.
Records the block label ID so the PHI can be inserted right after its OpLabel.
"""
function _defer_phi!(state::SPIRVEmitterState, inst::LLVM.PHIInst, block_label_id::UInt32)
    result_id = state.value_map[inst]  # Pre-allocated

    llvm_ty = LLVM.value_type(inst)
    result_ty = if llvm_ty isa LLVM.PointerType
        # Opaque pointer — infer pointee type from uses or incoming values
        pointee = _infer_pointee_from_users(inst)
        if pointee === nothing
            # Try incoming values
            n_inc = LLVM.API.LLVMCountIncoming(inst)
            for i in 0:(n_inc-1)
                val = LLVM.Value(LLVM.API.LLVMGetIncomingValue(inst, UInt32(i)))
                pointee = _infer_pointee_from_users(val)
                pointee !== nothing && break
                # Also check PTM
                ptm_info = get(state.type_ctx.ptm.map, val, nothing)
                if ptm_info !== nothing
                    pointee = ptm_info[1]
                    break
                end
            end
        end
        sc = _get_pointer_storage_class(inst)
        if pointee !== nothing
            pointee_spirv = map_type!(state.type_ctx, pointee)
            map_pointer_type!(state.type_ctx, pointee_spirv, sc)
        else
            # Fallback: byte pointer
            i8_spirv = map_type!(state.type_ctx, LLVM.IntType(8))
            map_pointer_type!(state.type_ctx, i8_spirv, sc)
        end
    else
        map_type!(state.type_ctx, llvm_ty)
    end

    # Collect (LLVM value, LLVM block) pairs — resolve IDs during resolution
    incoming = Tuple{LLVM.Value, LLVM.BasicBlock}[]
    n = LLVM.API.LLVMCountIncoming(inst)
    for i in 0:(n-1)
        val = LLVM.Value(LLVM.API.LLVMGetIncomingValue(inst, UInt32(i)))
        bb = LLVM.BasicBlock(LLVM.API.LLVMGetIncomingBlock(inst, UInt32(i)))
        push!(incoming, (val, bb))
    end

    push!(state.deferred_phis, (result_id, result_ty, incoming, block_label_id))
end

"""
Resolve all deferred PHI nodes by inserting them right after their block's OpLabel.
Must be called after all blocks are emitted so all operand values have IDs.
"""
function _resolve_deferred_phis!(state::SPIRVEmitterState)
    isempty(state.deferred_phis) && return

    # DEBUG: dump all deferred phis
    if get(ENV, "LAVA_DEBUG_PHI", "") == "1"
        println("=== DEFERRED PHIS: $(length(state.deferred_phis)) ===")
        for (result_id, type_id, incoming, block_label_id) in state.deferred_phis
            preds = [(get_block_id!(state, bb), String(LLVM.name(bb))) for (val, bb) in incoming]
            println("  PHI %$result_id (type %$type_id) in block %$block_label_id <- $preds")
        end
        println("=== PHI BLOCK REDIRECTS: $(length(state.phi_block_redirects)) ===")
        for ((from, to), redirect) in state.phi_block_redirects
            println("  ($from, $to) → $redirect")
        end
    end

    # Build: block_label_id → [phi_words...]
    phi_insertions = Dict{UInt32, Vector{UInt32}}()
    # Pre-terminator insertions: block_label_id → [bitcast_words...]
    # For pointer-type mismatches in PHI operands, we insert OpBitcast
    # in the predecessor block before its terminator instruction.
    pre_terminator_insertions = Dict{UInt32, Vector{UInt32}}()

    for (result_id, type_id, incoming, block_label_id) in state.deferred_phis
        operands = UInt32[]
        for (val, bb) in incoming
            val_id = if val isa LLVM.UndefValue && LLVM.value_type(val) isa LLVM.PointerType
                # UndefValue with opaque pointer type — can't map via map_type!.
                # Use the PHI's already-resolved typed pointer type_id to emit OpUndef.
                key = (:undef, type_id)
                get!(state.type_ctx.mod.constant_cache, key) do
                    uid = fresh_id!(state.type_ctx.mod)
                    encode_instruction!(state.type_ctx.mod.types_constants, UInt16(1), type_id, uid)
                    uid
                end
            else
                get_value_id!(state, val)
            end

            # Check for pointer type mismatch: LLVM opaque ptrs may map to different
            # SPIR-V pointer types. PHI requires all operands match result type.
            # Insert OpBitcast in predecessor block if types differ.
            # Skip PHI values themselves — their pointee types aren't in PTM.
            if LLVM.value_type(val) isa LLVM.PointerType && !(val isa LLVM.UndefValue || val isa LLVM.PoisonValue) && !(val isa LLVM.PHIInst)
                val_spirv_ty = try
                    map_pointer_type_for_value!(state.type_ctx, val)
                catch
                    type_id  # If we can't determine the type, assume it matches
                end
                if val_spirv_ty != type_id
                    # Check if this is a cross-storage-class mismatch by looking up
                    # the storage classes. Cross-SC bitcasts are INVALID in SPIR-V.
                    val_sc = nothing
                    target_sc = nothing
                    for ((sc, _pointee), ptr_ty) in state.type_ctx.pointer_types
                        if ptr_ty == val_spirv_ty
                            val_sc = sc
                        end
                        if ptr_ty == type_id
                            target_sc = sc
                        end
                    end

                    if val_sc !== nothing && target_sc !== nothing && val_sc != target_sc
                        # Cross-storage-class mismatch in PHI.
                        # OpConvertUToPtr only works for PhysicalStorageBuffer targets.
                        # If target is Function, we can't convert — this indicates a bug
                        # in _trace_to_non_alloca or type inference. Log warning and
                        # use OpBitcast as fallback (may fail validation but won't crash).
                        if target_sc != 5348 # StoragePhysicalStorageBuffer = 5348
                            @warn "Cross-SC PHI: target SC=$target_sc is not PSB, cannot ConvertUToPtr. Using OpBitcast fallback."
                            bitcast_id = fresh_id!(state.mod)
                            bitcast_words = UInt32[]
                            push!(bitcast_words, (UInt32(4) << 16) | UInt32(Op.OpBitcast))
                            push!(bitcast_words, type_id)
                            push!(bitcast_words, bitcast_id)
                            push!(bitcast_words, val_id)

                            from_id = get_block_id!(state, bb)
                            redirect = get(state.trampolines, (from_id, block_label_id), nothing)
                            if redirect === nothing
                                redirect = get(state.phi_block_redirects, (from_id, block_label_id), nothing)
                            end
                            pred_block_id = redirect !== nothing ? redirect : from_id
                            if !haskey(pre_terminator_insertions, pred_block_id)
                                pre_terminator_insertions[pred_block_id] = UInt32[]
                            end
                            append!(pre_terminator_insertions[pred_block_id], bitcast_words)
                            val_id = bitcast_id
                        else
                            # Target is PSB: safe to convert through integer (ConvertPtrToU → ConvertUToPtr)
                            ulong_ty = map_type!(state.type_ctx, LLVM.Int64Type())
                            ptr_to_u = fresh_id!(state.mod)
                            u_to_ptr = fresh_id!(state.mod)
                            conv_words = UInt32[]
                            # OpConvertPtrToU
                            push!(conv_words, (UInt32(4) << 16) | UInt32(Op.OpConvertPtrToU))
                            push!(conv_words, ulong_ty)
                            push!(conv_words, ptr_to_u)
                            push!(conv_words, val_id)
                            # OpConvertUToPtr
                            push!(conv_words, (UInt32(4) << 16) | UInt32(Op.OpConvertUToPtr))
                            push!(conv_words, type_id)
                            push!(conv_words, u_to_ptr)
                            push!(conv_words, ptr_to_u)

                            from_id = get_block_id!(state, bb)
                            redirect = get(state.trampolines, (from_id, block_label_id), nothing)
                            if redirect === nothing
                                redirect = get(state.phi_block_redirects, (from_id, block_label_id), nothing)
                            end
                            pred_block_id = redirect !== nothing ? redirect : from_id
                            if !haskey(pre_terminator_insertions, pred_block_id)
                                pre_terminator_insertions[pred_block_id] = UInt32[]
                            end
                            append!(pre_terminator_insertions[pred_block_id], conv_words)
                            val_id = u_to_ptr
                        end
                    else
                        # Same storage class: use OpBitcast (safe within same SC)
                        bitcast_id = fresh_id!(state.mod)
                        bitcast_words = UInt32[]
                        push!(bitcast_words, (UInt32(4) << 16) | UInt32(Op.OpBitcast))
                        push!(bitcast_words, type_id)
                        push!(bitcast_words, bitcast_id)
                        push!(bitcast_words, val_id)

                        from_id = get_block_id!(state, bb)
                        redirect = get(state.trampolines, (from_id, block_label_id), nothing)
                        if redirect === nothing
                            redirect = get(state.phi_block_redirects, (from_id, block_label_id), nothing)
                        end
                        pred_block_id = redirect !== nothing ? redirect : from_id
                        if !haskey(pre_terminator_insertions, pred_block_id)
                            pre_terminator_insertions[pred_block_id] = UInt32[]
                        end
                        append!(pre_terminator_insertions[pred_block_id], bitcast_words)
                        val_id = bitcast_id
                    end
                end
            end

            push!(operands, val_id)
            # Check if this incoming edge was redirected through a trampoline or block split
            from_id = get_block_id!(state, bb)
            redirect = get(state.trampolines, (from_id, block_label_id), nothing)
            if redirect === nothing
                redirect = get(state.phi_block_redirects, (from_id, block_label_id), nothing)
            end
            push!(operands, redirect !== nothing ? redirect : from_id)
        end

        word_count = UInt32(3 + length(operands))
        phi_words = UInt32[]
        push!(phi_words, (word_count << 16) | UInt32(Op.OpPhi))
        push!(phi_words, type_id)
        push!(phi_words, result_id)
        append!(phi_words, operands)

        if !haskey(phi_insertions, block_label_id)
            phi_insertions[block_label_id] = UInt32[]
        end
        append!(phi_insertions[block_label_id], phi_words)
    end

    # Scan the functions buffer and insert:
    # 1. PHIs right after each OpLabel
    # 2. Bitcasts before terminators in predecessor blocks (for PHI type mismatches)
    new_functions = UInt32[]
    i = 1
    buf = state.mod.functions

    # Terminator opcodes (instructions that end a block)
    TERMINATOR_OPCODES = Set{UInt32}([
        UInt32(Op.OpBranch), UInt32(Op.OpBranchConditional),
        UInt32(Op.OpReturn), UInt32(Op.OpReturnValue),
    ])

    # Track current block label for pre-terminator insertions
    current_block_label = UInt32(0)

    while i <= length(buf)
        word = buf[i]
        opcode = word & 0xFFFF
        wc = word >> 16

        # Before emitting a terminator, check for pre-terminator insertions
        if opcode in TERMINATOR_OPCODES && haskey(pre_terminator_insertions, current_block_label)
            append!(new_functions, pre_terminator_insertions[current_block_label])
            delete!(pre_terminator_insertions, current_block_label)
        end

        push!(new_functions, word)
        for j in 1:(wc-1)
            push!(new_functions, buf[i+j])
        end

        # If this was an OpLabel, track current block and check for PHI insertions
        if opcode == UInt32(Op.OpLabel) && wc >= 2
            current_block_label = buf[i+1]
            if haskey(phi_insertions, current_block_label)
                append!(new_functions, phi_insertions[current_block_label])
            end
        end

        i += wc
    end

    # Replace functions buffer
    empty!(state.mod.functions)
    append!(state.mod.functions, new_functions)
    empty!(state.deferred_phis)
end

# ================================================================
# Function Calls + LLVM Intrinsics → GLSL.std.450
# ================================================================

# LLVM intrinsic name → GLSL.std.450 instruction number
const GLSL_STD_450_MAP = Dict{String, UInt32}(
    "llvm.sqrt"     => UInt32(31),  # Sqrt
    "llvm.fabs"     => UInt32(4),   # FAbs
    "llvm.sin"      => UInt32(13),  # Sin
    "llvm.cos"      => UInt32(14),  # Cos
    "llvm.floor"    => UInt32(8),   # Floor
    "llvm.ceil"     => UInt32(9),   # Ceil
    "llvm.round"    => UInt32(1),   # Round
    "llvm.trunc"    => UInt32(3),   # Trunc
    "llvm.exp"      => UInt32(27),  # Exp
    "llvm.exp2"     => UInt32(29),  # Exp2
    "llvm.log"      => UInt32(28),  # Log
    "llvm.log2"     => UInt32(30),  # Log2
    "llvm.pow"      => UInt32(26),  # Pow
    "llvm.fma"      => UInt32(50),  # Fma
    "llvm.minnum"   => UInt32(37),  # FMin
    "llvm.maxnum"   => UInt32(40),  # FMax
    "llvm.minimum"  => UInt32(79),  # NMin (propagates NaN)
    "llvm.maximum"  => UInt32(80),  # NMax (propagates NaN)
    "llvm.smin"     => UInt32(39),  # SMin (signed integer min)
    "llvm.smax"     => UInt32(42),  # SMax (signed integer max)
    "llvm.umin"     => UInt32(38),  # UMin (unsigned integer min)
    "llvm.umax"     => UInt32(41),  # UMax (unsigned integer max)
    "llvm.fmuladd"  => UInt32(50),  # Fma (fmuladd ≈ fma for GPU)
    "llvm.rint"     => UInt32(1),   # RoundEven (rint = round to nearest even)
    "llvm.nearbyint"=> UInt32(1),   # RoundEven
    # Additional trig/hyperbolic (for future use via LLVM intrinsics)
    "llvm.asin"     => UInt32(16),  # Asin
    "llvm.acos"     => UInt32(17),  # Acos
    "llvm.atan"     => UInt32(18),  # Atan
    "llvm.sinh"     => UInt32(19),  # Sinh
    "llvm.cosh"     => UInt32(20),  # Cosh
    "llvm.tanh"     => UInt32(21),  # Tanh
    "llvm.atan2"    => UInt32(25),  # Atan2
    # copysign handled specially below (not a direct GLSL.std.450 op)
)

# Custom _lava_glsl_* function names → GLSL.std.450 instruction numbers.
# These are for GLSL.std.450 ops that LLVM has no intrinsic for (tan, asin, etc.).
# Name format: _lava_glsl_<op>_<suffix> where suffix is f32 or f64.
const LAVA_GLSL_MAP = Dict{String, UInt32}(
    "_lava_glsl_tan"   => UInt32(15),  # Tan
    "_lava_glsl_asin"  => UInt32(16),  # Asin
    "_lava_glsl_acos"  => UInt32(17),  # Acos
    "_lava_glsl_atan"  => UInt32(18),  # Atan
    "_lava_glsl_sinh"  => UInt32(19),  # Sinh
    "_lava_glsl_cosh"  => UInt32(20),  # Cosh
    "_lava_glsl_tanh"  => UInt32(21),  # Tanh
    "_lava_glsl_asinh" => UInt32(22),  # Asinh
    "_lava_glsl_acosh" => UInt32(23),  # Acosh
    "_lava_glsl_atanh" => UInt32(24),  # Atanh
    "_lava_glsl_atan2" => UInt32(25),  # Atan2
)

function _emit_call!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    called = LLVM.called_operand(inst)

    if called isa LLVM.Function
        fn_name = LLVM.name(called)

        # Check for LLVM intrinsics → GLSL.std.450
        if startswith(fn_name, "llvm.")
            return _emit_llvm_intrinsic!(state, inst, fn_name)
        end

        # Check for custom _lava_glsl_* → GLSL.std.450
        if startswith(fn_name, "_lava_glsl_")
            return _emit_lava_glsl!(state, inst, fn_name)
        end

        # Check for RT intrinsics → OpTraceRayKHR, payload load/store
        if fn_name == "_lava_rt_trace_ray"
            return _emit_rt_trace_ray!(state, inst)
        elseif fn_name == "_lava_rt_payload_store_f32"
            return _emit_rt_payload_store!(state, inst)
        elseif fn_name == "_lava_rt_payload_load_f32"
            return _emit_rt_payload_load!(state, inst)
        elseif fn_name == "_lava_rt_payload_store_f32_at"
            return _emit_rt_payload_store_at!(state, inst)
        elseif fn_name == "_lava_rt_payload_load_f32_at"
            return _emit_rt_payload_load_at!(state, inst)
        elseif fn_name == "_lava_rt_hit_attrib_load_f32_at"
            return _emit_rt_hit_attrib_load_at!(state, inst)
        elseif fn_name == "_lava_rt_ignore_intersection"
            return _emit_rt_ignore_intersection!(state, inst)
        elseif fn_name == "_lava_rt_terminate_ray"
            return _emit_rt_terminate_ray!(state, inst)
        end

        # Check for graphics intrinsics → I/O stores/loads, emit_vertex, etc.
        if fn_name == "_lava_gfx_set_position"
            return _emit_gfx_set_position!(state, inst)
        elseif fn_name == "_lava_gfx_set_point_size"
            return _emit_gfx_set_point_size!(state, inst)
        elseif fn_name == "_lava_gfx_output_vec4"
            return _emit_gfx_output_vec4!(state, inst)
        elseif fn_name == "_lava_gfx_output_vec3"
            return _emit_gfx_output_vec3!(state, inst)
        elseif fn_name == "_lava_gfx_output_vec2"
            return _emit_gfx_output_vec2!(state, inst)
        elseif fn_name == "_lava_gfx_output_f32"
            return _emit_gfx_output_f32!(state, inst)
        elseif fn_name == "_lava_gfx_input_vec4"
            return _emit_gfx_input!(state, inst, :vec4)
        elseif fn_name == "_lava_gfx_input_vec3"
            return _emit_gfx_input!(state, inst, :vec3)
        elseif fn_name == "_lava_gfx_input_vec2"
            return _emit_gfx_input!(state, inst, :vec2)
        elseif fn_name == "_lava_gfx_input_f32"
            return _emit_gfx_input!(state, inst, :f32)
        elseif fn_name == "_lava_gfx_emit_vertex"
            return _emit_gfx_emit_vertex!(state, inst)
        elseif fn_name == "_lava_gfx_end_primitive"
            return _emit_gfx_end_primitive!(state, inst)
        elseif fn_name == "_lava_gfx_set_tess_level_outer"
            return _emit_gfx_set_tess_level!(state, inst, true)
        elseif fn_name == "_lava_gfx_set_tess_level_inner"
            return _emit_gfx_set_tess_level!(state, inst, false)
        elseif fn_name == "_lava_gfx_sample_2d"
            return _emit_gfx_sample_2d!(state, inst)
        end

        # Regular function call
        _emit_direct_call!(state, inst, called)
    else
        error("Indirect calls not supported: $inst")
    end
end

function _emit_lava_glsl!(state::SPIRVEmitterState, inst::LLVM.CallInst, name::String)
    # Strip type suffix: "_lava_glsl_tan_f32" → "_lava_glsl_tan"
    base = replace(name, r"_f(16|32|64)$" => "")
    glsl_num = get(LAVA_GLSL_MAP, base, nothing)
    if glsl_num !== nothing
        return _emit_glsl_ext_inst!(state, inst, glsl_num)
    end
    error("Unknown _lava_glsl function: $name")
end

function _emit_llvm_intrinsic!(state::SPIRVEmitterState, inst::LLVM.CallInst, name::String)
    # Strip type suffix: "llvm.sqrt.f32" → "llvm.sqrt"
    base_name = _strip_intrinsic_suffix(name)

    glsl_num = get(GLSL_STD_450_MAP, base_name, nothing)
    if glsl_num !== nothing
        return _emit_glsl_ext_inst!(state, inst, glsl_num)
    end

    # copysign(x, y) = FAbs(x) * FSign(y) — not directly in GLSL.std.450
    if base_name == "llvm.copysign"
        glsl_id = setup_glsl_std_450!(state.mod)
        x_id = get_value_id!(state, LLVM.operands(inst)[1])
        y_id = get_value_id!(state, LLVM.operands(inst)[2])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        abs_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpExtInst, result_ty, abs_id, glsl_id, UInt32(4), x_id)  # FAbs=4
        sign_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpExtInst, result_ty, sign_id, glsl_id, UInt32(6), y_id)  # FSign=6
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpFMul, result_ty, result_id, abs_id, sign_id)
        state.value_map[inst] = result_id
        return
    end

    # Handle other intrinsics
    if startswith(name, "llvm.lifetime.")
        # lifetime.start/end — skip (should be stripped by passes)
        return
    elseif startswith(name, "llvm.dbg.")
        # Debug intrinsics — skip
        return
    elseif startswith(name, "llvm.assume")
        # assume — skip
        return
    elseif startswith(name, "llvm.memset.")
        _emit_memset!(state, inst)
        return
    elseif startswith(name, "llvm.memcpy.") || startswith(name, "llvm.memmove.")
        # memcpy/memmove → emit scalar load/store pairs
        # LLVM: llvm.memcpy(dest, src, size, isvolatile)
        # For struct copies (e.g., ComplexF64 = 16 bytes), size is constant.
        # Emit as i32 load/store pairs via PSB pointer arithmetic.
        _emit_memcpy!(state, inst)
        return
    elseif startswith(base_name, "llvm.abs") && !startswith(base_name, "llvm.abs.")
        # Integer abs → GLSL.std.450 SAbs (5)
        # llvm.abs.i32(val, is_int_min_poison) — ignore poison flag, take only val
        glsl_id = setup_glsl_std_450!(state.mod)
        val_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpExtInst, result_ty, result_id, glsl_id, UInt32(5), val_id)
        state.value_map[inst] = result_id
        return
    elseif startswith(base_name, "llvm.cttz")
        # Count trailing zeros → GLSL.std.450 FindILsb (73)
        # llvm.cttz.i32(val, is_zero_poison) — ignore is_zero_poison, take only val
        glsl_id = setup_glsl_std_450!(state.mod)
        val_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpExtInst, result_ty, result_id, glsl_id, UInt32(73), val_id)
        state.value_map[inst] = result_id
        return
    elseif startswith(base_name, "llvm.ctlz")
        # Count leading zeros: (bitwidth - 1) - FindUMsb(val)
        # GLSL.std.450 FindUMsb (75) returns position of MSB (0-based from LSB)
        # ctlz = bitwidth - 1 - FindUMsb for non-zero values
        glsl_id = setup_glsl_std_450!(state.mod)
        val_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_llvm_ty = LLVM.value_type(inst)
        result_ty = map_type!(state.type_ctx, result_llvm_ty)
        bw = LLVM.width(result_llvm_ty)

        if bw <= 32
            msb_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpExtInst, result_ty, msb_id, glsl_id, UInt32(75), val_id)
            # ctlz = (bitwidth - 1) - msb
            bw_minus_1 = map_constant!(state.type_ctx, LLVM.ConstantInt(result_llvm_ty, bw - 1))
            result_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpISub, result_ty, result_id, bw_minus_1, msb_id)
        else
            # 64-bit ctlz: FindUMsb only supports 32-bit, so split into hi/lo
            u32_llvm = LLVM.IntType(32)
            u32_ty = map_type!(state.type_ctx, u32_llvm)
            u64_ty = result_ty

            # Extract hi = val >> 32, lo = val & 0xFFFFFFFF
            c32_id = map_constant!(state.type_ctx, LLVM.ConstantInt(result_llvm_ty, 32))
            hi_64 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpShiftRightLogical, u64_ty, hi_64, val_id, c32_id)
            hi_32 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, hi_32, hi_64)
            lo_32 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpUConvert, u32_ty, lo_32, val_id)

            # FindUMsb on hi and lo (32-bit each)
            msb_hi = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpExtInst, u32_ty, msb_hi, glsl_id, UInt32(75), hi_32)
            msb_lo = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpExtInst, u32_ty, msb_lo, glsl_id, UInt32(75), lo_32)

            # hi_clz = 31 - msb_hi; lo_clz = 31 - msb_lo + 32
            c31_32 = map_constant!(state.type_ctx, LLVM.ConstantInt(u32_llvm, 31))
            c32_32 = map_constant!(state.type_ctx, LLVM.ConstantInt(u32_llvm, 32))
            c0_32 = map_constant!(state.type_ctx, LLVM.ConstantInt(u32_llvm, 0))

            hi_clz_32 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpISub, u32_ty, hi_clz_32, c31_32, msb_hi)

            lo_clz_32_base = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpISub, u32_ty, lo_clz_32_base, c31_32, msb_lo)
            lo_clz_32 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIAdd, u32_ty, lo_clz_32, lo_clz_32_base, c32_32)

            # Select based on hi == 0
            bool_ty = map_type!(state.type_ctx, LLVM.IntType(1))
            hi_is_zero = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpIEqual, bool_ty, hi_is_zero, hi_32, c0_32)

            selected_32 = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpSelect, u32_ty, selected_32, hi_is_zero, lo_clz_32, hi_clz_32)

            # Convert result to 64-bit
            result_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpUConvert, u64_ty, result_id, selected_32)
        end
        state.value_map[inst] = result_id
        return
    elseif startswith(base_name, "llvm.ctpop")
        # Population count → OpBitCount
        val_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitCount, result_ty, result_id, val_id)
        state.value_map[inst] = result_id
        return
    elseif startswith(base_name, "llvm.bitreverse")
        # Bit reverse → OpBitReverse
        val_id = get_value_id!(state, LLVM.operands(inst)[1])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitReverse, result_ty, result_id, val_id)
        state.value_map[inst] = result_id
        return
    elseif startswith(base_name, "llvm.bswap")
        # Byte swap — not available in SPIR-V, implement with shifts
        # For now, error with clear message
        error("llvm.bswap not yet implemented in SPIR-V emitter")
    elseif name == "llvm.spv.group.memory.barrier.with.group.sync"
        # Workgroup barrier: OpControlBarrier Workgroup Workgroup
        # Vulkan memory model requires MakeAvailableKHR + MakeVisibleKHR for writes
        # to be visible across invocations (unlike GLSL450 where barrier() implies this).
        # Semantics: AcquireRelease (0x8) | WorkgroupMemory (0x100)
        #          | MakeAvailableKHR (0x2000) | MakeVisibleKHR (0x4000) = 0x6108
        u32_ty = map_type!(state.type_ctx, LLVM.IntType(32))
        scope_wg = map_constant!(state.type_ctx, LLVM.ConstantInt(LLVM.IntType(32), 2))   # Workgroup = 2
        semantics = map_constant!(state.type_ctx, LLVM.ConstantInt(LLVM.IntType(32), 0x6108))  # AcquireRelease|WorkgroupMemory|MakeAvailable|MakeVisible
        encode_instruction!(state.mod.functions, Op.OpControlBarrier, scope_wg, scope_wg, semantics)
        return
    end

    # ── Funnel shifts ──
    # llvm.fshl(a, b, shift) = (a << shift) | (b >> (bitwidth - shift))
    # llvm.fshr(a, b, shift) = (a << (bitwidth - shift)) | (b >> shift)
    if base_name == "llvm.fshl" || base_name == "llvm.fshr"
        ops = LLVM.operands(inst)
        a_id = get_value_id!(state, ops[1])
        b_id = get_value_id!(state, ops[2])
        shift_id = get_value_id!(state, ops[3])
        result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
        bitwidth = LLVM.width(LLVM.value_type(ops[1]))
        bw_id = map_constant!(state.type_ctx, LLVM.ConstantInt(LLVM.value_type(ops[1]), bitwidth))
        # shift_mod = shift % bitwidth (SPIR-V shift is modulo, but be explicit)
        shift_mod = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpUMod, result_ty, shift_mod, shift_id, bw_id)
        # complement = bitwidth - shift_mod
        complement = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpISub, result_ty, complement, bw_id, shift_mod)
        if name == "llvm.fshl"
            hi = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpShiftLeftLogical, result_ty, hi, a_id, shift_mod)
            lo = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpShiftRightLogical, result_ty, lo, b_id, complement)
        else  # fshr
            hi = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpShiftLeftLogical, result_ty, hi, a_id, complement)
            lo = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpShiftRightLogical, result_ty, lo, b_id, shift_mod)
        end
        result_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.functions, Op.OpBitwiseOr, result_ty, result_id, hi, lo)
        state.value_map[inst] = result_id
        return
    end

    error("Unsupported LLVM intrinsic: $name")
end

function _strip_intrinsic_suffix(name::String)
    # "llvm.sqrt.f32" → "llvm.sqrt"
    # "llvm.fma.f32" → "llvm.fma"
    parts = split(name, '.')
    if length(parts) >= 3 && startswith(parts[end], "f") || startswith(parts[end], "i") || parts[end] == "v2f32"
        return join(parts[1:end-1], '.')
    end
    return name
end

function _emit_glsl_ext_inst!(state::SPIRVEmitterState, inst::LLVM.CallInst, glsl_num::UInt32)
    # Ensure GLSL.std.450 is imported
    glsl_id = setup_glsl_std_450!(state.mod)

    ops = LLVM.operands(inst)
    # Last operand of a CallInst is the called function itself — skip it
    n_args = length(ops) - 1

    arg_ids = UInt32[]
    for i in 1:n_args
        push!(arg_ids, get_value_id!(state, ops[i]))
    end

    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)

    # OpExtInst: result_type result_id set instruction [operands]
    word_count = UInt32(5 + length(arg_ids))
    push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpExtInst))
    push!(state.mod.functions, result_ty)
    push!(state.mod.functions, result_id)
    push!(state.mod.functions, glsl_id)
    push!(state.mod.functions, glsl_num)
    append!(state.mod.functions, arg_ids)

    state.value_map[inst] = result_id
end

function _emit_direct_call!(state::SPIRVEmitterState, inst::LLVM.CallInst, called::LLVM.Function)
    fn_id = get!(state.value_map, called) do
        fresh_id!(state.mod)
    end

    ops = LLVM.operands(inst)
    n_args = length(ops) - 1  # Last operand is the called function

    arg_ids = UInt32[]
    for i in 1:n_args
        push!(arg_ids, get_value_id!(state, ops[i]))
    end

    result_ty_llvm = LLVM.value_type(inst)
    if result_ty_llvm isa LLVM.VoidType
        # Void call — no result
        word_count = UInt32(4 + length(arg_ids))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpFunctionCall))
        push!(state.mod.functions, map_type!(state.type_ctx, result_ty_llvm))
        result_id = fresh_id!(state.mod)  # SPIR-V requires result even for void
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, fn_id)
        append!(state.mod.functions, arg_ids)
    else
        result_ty = map_type!(state.type_ctx, result_ty_llvm)
        result_id = fresh_id!(state.mod)
        word_count = UInt32(4 + length(arg_ids))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpFunctionCall))
        push!(state.mod.functions, result_ty)
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, fn_id)
        append!(state.mod.functions, arg_ids)
        state.value_map[inst] = result_id
    end
end

# ================================================================
# Aggregate Operations
# ================================================================

function _emit_extractvalue!(state::SPIRVEmitterState, inst::LLVM.ExtractValueInst)
    ops = LLVM.operands(inst)
    agg_val = ops[1]

    # Get indices from the extractvalue instruction
    n_indices = API.LLVMGetNumIndices(inst)
    indices_ptr = API.LLVMGetIndices(inst)
    indices = UInt32[unsafe_load(indices_ptr, i) for i in 1:n_indices]

    # Special handling for cmpxchg results: { T, i1 }
    # Our _emit_cmpxchg! maps the cmpxchg instruction to just the old value,
    # and stores the compare value for computing success.
    if agg_val isa LLVM.AtomicCmpXchgInst
        old_id = get_value_id!(state, agg_val)
        if length(indices) == 1 && indices[1] == 0
            # extractvalue { T, i1 } %result, 0 → old value (already have it)
            state.value_map[inst] = old_id
            return
        elseif length(indices) == 1 && indices[1] == 1
            # extractvalue { T, i1 } %result, 1 → success flag (old == expected)
            cmp_id = get(state.cmpxchg_cmp_vals, agg_val, nothing)
            if cmp_id !== nothing
                # Emit: success = (old == expected) via OpIEqual
                bool_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
                result_id = fresh_id!(state.mod)
                encode_instruction!(state.mod.functions, Op.OpIEqual,
                    bool_ty, result_id, old_id, cmp_id)
                state.value_map[inst] = result_id
                return
            end
        end
    end

    # Normal extractvalue path
    agg = get_value_id!(state, agg_val)
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)

    # OpCompositeExtract (opcode 81): result_type result_id composite [indices]
    word_count = UInt32(4 + length(indices))
    push!(state.mod.functions, (word_count << 16) | UInt32(81))
    push!(state.mod.functions, result_ty)
    push!(state.mod.functions, result_id)
    push!(state.mod.functions, agg)
    append!(state.mod.functions, indices)
    state.value_map[inst] = result_id
end

function _emit_insertvalue!(state::SPIRVEmitterState, inst::LLVM.InsertValueInst)
    ops = LLVM.operands(inst)
    agg = get_value_id!(state, ops[1])
    val = get_value_id!(state, ops[2])
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)

    n_indices = API.LLVMGetNumIndices(inst)
    indices_ptr = API.LLVMGetIndices(inst)
    indices = UInt32[unsafe_load(indices_ptr, i) for i in 1:n_indices]

    # OpCompositeInsert (opcode 82): result_type result_id object composite [indices]
    word_count = UInt32(5 + length(indices))
    push!(state.mod.functions, (word_count << 16) | UInt32(82))
    push!(state.mod.functions, result_ty)
    push!(state.mod.functions, result_id)
    push!(state.mod.functions, val)
    push!(state.mod.functions, agg)
    append!(state.mod.functions, indices)
    state.value_map[inst] = result_id
end

# ================================================================
# Vector Element Operations
# ================================================================

function _emit_extractelement!(state::SPIRVEmitterState, inst::LLVM.ExtractElementInst)
    ops = LLVM.operands(inst)
    vec = get_value_id!(state, ops[1])
    idx = ops[2]
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)

    if idx isa LLVM.ConstantInt
        # Constant index → OpCompositeExtract (literal index)
        idx_val = UInt32(convert(Int, idx))
        encode_instruction!(state.mod.functions, Op.OpCompositeExtract,
                            result_ty, result_id, vec, idx_val)
    else
        # Variable index → OpVectorExtractDynamic
        idx_id = get_value_id!(state, idx)
        encode_instruction!(state.mod.functions, Op.OpVectorExtractDynamic,
                            result_ty, result_id, vec, idx_id)
    end
    state.value_map[inst] = result_id
end

function _emit_insertelement!(state::SPIRVEmitterState, inst::LLVM.InsertElementInst)
    ops = LLVM.operands(inst)
    vec = get_value_id!(state, ops[1])
    val = get_value_id!(state, ops[2])
    idx = ops[3]
    result_ty = map_type!(state.type_ctx, LLVM.value_type(inst))
    result_id = fresh_id!(state.mod)

    if idx isa LLVM.ConstantInt
        # Constant index → OpCompositeInsert
        idx_val = UInt32(convert(Int, idx))
        word_count = UInt32(6)
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpCompositeInsert))
        push!(state.mod.functions, result_ty)
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, val)
        push!(state.mod.functions, vec)
        push!(state.mod.functions, idx_val)
    else
        # Variable index → OpVectorInsertDynamic (opcode 78)
        idx_id = get_value_id!(state, idx)
        word_count = UInt32(6)
        push!(state.mod.functions, (word_count << 16) | UInt32(78))  # OpVectorInsertDynamic
        push!(state.mod.functions, result_ty)
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, vec)
        push!(state.mod.functions, val)
        push!(state.mod.functions, idx_id)
    end
    state.value_map[inst] = result_id
end

# ================================================================
# Atomics
# ================================================================

"""
Map LLVM atomic ordering + pointer storage class to SPIR-V memory semantics.

With the Vulkan memory model (VulkanKHR), atomic operations need explicit
MakeAvailableKHR/MakeVisibleKHR flags for writes to be visible across
invocations. Without these, even seq_cst atomics provide only atomicity
(correct counter values) but NOT memory visibility for non-atomic stores
— which breaks cross-workgroup patterns like BVH refit where Thread A
writes node data, atomics on a flag, and Thread B reads via the flag.
"""
function _atomic_mem_semantics(inst::LLVM.Instruction, ptr::LLVM.Value)::UInt32
    ord = LLVM.ordering(inst)

    # Determine storage class bit for the pointer's memory
    sc = _get_pointer_storage_class(ptr)
    sc_bit = if sc == SC.Workgroup
        MemSem.WorkgroupMemory   # 0x100
    else
        # StorageBuffer, PhysicalStorageBuffer, Uniform all use UniformMemory
        MemSem.UniformMemory     # 0x40
    end

    if ord == LLVM.API.LLVMAtomicOrderingMonotonic ||
       ord == LLVM.API.LLVMAtomicOrderingUnordered
        # Relaxed — just atomicity, no ordering
        return MemSem.Relaxed
    elseif ord == LLVM.API.LLVMAtomicOrderingAcquire
        return MemSem.Acquire | sc_bit | MemSem.MakeVisibleKHR
    elseif ord == LLVM.API.LLVMAtomicOrderingRelease
        return MemSem.Release | sc_bit | MemSem.MakeAvailableKHR
    else
        # AcquireRelease or SequentiallyConsistent
        # Vulkan memory model doesn't support SequentiallyConsistent;
        # AcquireRelease + MakeAvailable + MakeVisible is the equivalent.
        return MemSem.AcquireRelease | sc_bit | MemSem.MakeAvailableKHR | MemSem.MakeVisibleKHR
    end
end

"""Acquire-only variant for cmpxchg failure path (no release, a failed CAS writes nothing)."""
function _atomic_mem_semantics_acquire_only(inst::LLVM.Instruction, ptr::LLVM.Value)::UInt32
    ord = LLVM.ordering(inst)
    if ord == LLVM.API.LLVMAtomicOrderingMonotonic ||
       ord == LLVM.API.LLVMAtomicOrderingUnordered
        return MemSem.Relaxed
    end
    sc = _get_pointer_storage_class(ptr)
    sc_bit = sc == SC.Workgroup ? MemSem.WorkgroupMemory : MemSem.UniformMemory
    return MemSem.Acquire | sc_bit | MemSem.MakeVisibleKHR
end

function _emit_atomicrmw!(state::SPIRVEmitterState, inst::LLVM.AtomicRMWInst)
    ops = LLVM.operands(inst)
    ptr = ops[1]
    val = ops[2]

    ptr_id = get_value_id!(state, ptr)
    val_id = get_value_id!(state, val)

    # Result type is the value type (same as the loaded value)
    result_llvm_ty = LLVM.value_type(val)
    result_ty = map_type!(state.type_ctx, result_llvm_ty)
    result_id = fresh_id!(state.mod)

    # Determine scope: QueueFamily is equivalent to Device for single-queue Vulkan
    # and doesn't require VulkanMemoryModelDeviceScopeKHR capability
    scope_id = emit_constant_u32!(state.mod, Scope.QueueFamily)

    # Memory semantics — derived from LLVM ordering + pointer storage class
    mem_sem_id = emit_constant_u32!(state.mod, _atomic_mem_semantics(inst, ptr))

    # Map LLVM atomicrmw operation to SPIR-V opcode
    binop = LLVM.API.LLVMGetAtomicRMWBinOp(inst)
    is_signed = _is_signed_integer_context(result_llvm_ty)

    opcode = if binop == LLVM.API.LLVMAtomicRMWBinOpAdd
        Op.OpAtomicIAdd
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpSub
        Op.OpAtomicISub
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpAnd
        Op.OpAtomicAnd
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpOr
        Op.OpAtomicOr
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpXor
        Op.OpAtomicXor
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpMin
        Op.OpAtomicSMin
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpMax
        Op.OpAtomicSMax
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpUMin
        Op.OpAtomicUMin
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpUMax
        Op.OpAtomicUMax
    elseif binop == LLVM.API.LLVMAtomicRMWBinOpXchg
        Op.OpAtomicExchange
    else
        error("Unsupported atomicrmw operation: $binop")
    end

    # SPIR-V atomic instructions require a pointer to the value type.
    # Byte-offset GEPs (gep i8) produce pointers-to-i8, but atomicrmw needs
    # a pointer to the actual value type (e.g., i32). Bitcast if needed.
    ptr_pointee = get_pointee_type(state.type_ctx.ptm, ptr)
    if ptr_pointee !== nothing && ptr_pointee != result_llvm_ty
        # Need to reinterpret pointer to correct type
        sc = _get_pointer_storage_class(ptr)
        correct_ptr_ty = map_pointer_type!(state.type_ctx, result_ty, sc)
        if sc == SC.PhysicalStorageBuffer
            ptr_id = _emit_psb_ptr_reinterpret!(state, correct_ptr_ty, ptr_id)
        else
            bitcast_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpBitcast,
                correct_ptr_ty, bitcast_id, ptr_id)
            ptr_id = bitcast_id
        end
    end

    # Format: OpAtomic* result_type result_id pointer scope mem_semantics value
    encode_instruction!(state.mod.functions, opcode,
        result_ty, result_id, ptr_id, scope_id, mem_sem_id, val_id)

    state.value_map[inst] = result_id
end

# Helper: check if an integer type is signed (for min/max dispatch)
function _is_signed_integer_context(ty::LLVM.LLVMType)
    # LLVM integers are signless; the operation determines signedness
    # atomicrmw min/max vs umin/umax already encodes this
    return false
end

function _emit_cmpxchg!(state::SPIRVEmitterState, inst::LLVM.AtomicCmpXchgInst)
    ops = LLVM.operands(inst)
    ptr = ops[1]
    cmp_val = ops[2]    # expected value
    new_val = ops[3]    # desired value

    ptr_id = get_value_id!(state, ptr)
    cmp_id = get_value_id!(state, cmp_val)
    new_id = get_value_id!(state, new_val)

    # The value type (what we're comparing/exchanging)
    val_llvm_ty = LLVM.value_type(cmp_val)
    val_ty = map_type!(state.type_ctx, val_llvm_ty)

    # Scope — QueueFamily (equivalent to Device for single-queue, no extra capability)
    scope_id = emit_constant_u32!(state.mod, Scope.QueueFamily)
    # Memory semantics — derived from LLVM ordering + pointer storage class
    mem_sem = _atomic_mem_semantics(inst, ptr)
    mem_sem_equal_id = emit_constant_u32!(state.mod, mem_sem)
    # Failure semantics: acquire-only (failed CAS doesn't release)
    mem_sem_unequal_id = emit_constant_u32!(state.mod, _atomic_mem_semantics_acquire_only(inst, ptr))

    # Reinterpret pointer if pointee type doesn't match value type (byte-offset GEPs)
    ptr_pointee = get_pointee_type(state.type_ctx.ptm, ptr)
    if ptr_pointee !== nothing && ptr_pointee != val_llvm_ty
        sc = _get_pointer_storage_class(ptr)
        correct_ptr_ty = map_pointer_type!(state.type_ctx, val_ty, sc)
        if sc == SC.PhysicalStorageBuffer
            ptr_id = _emit_psb_ptr_reinterpret!(state, correct_ptr_ty, ptr_id)
        else
            bitcast_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpBitcast,
                correct_ptr_ty, bitcast_id, ptr_id)
            ptr_id = bitcast_id
        end
    end

    # OpAtomicCompareExchange returns just the old value (not a struct like LLVM)
    # LLVM cmpxchg returns { T, i1 } where T is the old value and i1 is success
    old_id = fresh_id!(state.mod)
    encode_instruction!(state.mod.functions, Op.OpAtomicCompareExchange,
        val_ty, old_id, ptr_id, scope_id,
        mem_sem_equal_id, mem_sem_unequal_id,
        new_id, cmp_id)

    # LLVM cmpxchg result is { T, i1 }. Users extract with extractvalue.
    # We store the old value and compute success (old == expected) lazily in extractvalue.
    # Store a mapping: inst → old_id, and handle extractvalue specially.
    state.value_map[inst] = old_id

    # Store the compare value for extractvalue index 1 (success = old == expected)
    state.cmpxchg_cmp_vals[inst] = cmp_id
end

# ================================================================
# ConstantExpr handling
# ================================================================

function _emit_constant_expr!(state::SPIRVEmitterState, val::LLVM.ConstantExpr)
    # ConstantExpr is an LLVM value that represents a constant computation
    # Common cases: GEP on globals, bitcast, inttoptr
    op = LLVM.opcode(val)

    if op == LLVM.API.LLVMGetElementPtr
        # Constant GEP on a global variable (e.g., shared memory array)
        # Pattern: getelementptr [N x T], ptr addrspace(3) @global, i64 0, i64 0
        # → OpAccessChain into the global variable
        ops = LLVM.operands(val)
        base = ops[1]
        base_id = get_value_id!(state, base)

        # Determine source element type and result type
        source_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(val))
        result_pointee = _compute_gep_result_type(source_ty, val)
        if result_pointee === nothing
            error("Could not compute ConstantExpr GEP result type: $val")
        end

        # Determine storage class from the base pointer
        base_ty = LLVM.value_type(base)
        as = base_ty isa LLVM.PointerType ? LLVM.addrspace(base_ty) : 0
        sc = llvm_addrspace_to_storage_class(as)

        # Use fresh workgroup types for addrspace(3) to avoid layout decoration conflicts
        result_pointee_spirv = if sc == SC.Workgroup
            map_workgroup_type!(state.type_ctx, result_pointee)
        else
            map_type!(state.type_ctx, result_pointee)
        end
        result_ptr_ty = map_pointer_type!(state.type_ctx, result_pointee_spirv, sc)

        # Emit OpAccessChain with constant indices.
        # In LLVM GEP: first index offsets the base pointer (array-of-T semantics),
        # subsequent indices drill into the composite type.
        # In SPIR-V OpAccessChain: base is already a pointer to the composite,
        # so we skip the first GEP index (pointer offset) and emit only the rest.
        #
        # IMPORTANT: When the first index is non-zero (e.g., -1 for Julia's 1-based
        # indexing trick), we can't just skip it. We must compute a flat element
        # offset and use OpAccessChain[0] + OpPtrAccessChain.
        n_indices = length(ops) - 1
        first_idx = ops[2] isa LLVM.ConstantInt ? convert(Int64, ops[2]) : 0

        if first_idx != 0 && source_ty isa LLVM.ArrayType && length(ops) >= 3
            # Non-zero first index on array type: compute flat element offset.
            # For getelementptr [N x T], ptr @base, i64 A, i64 B:
            #   flat_offset = A * N + B
            # Then: AccessChain @base[0] → PtrAccessChain by flat_offset
            array_len = Int64(LLVM.length(source_ty))
            inner_idx = ops[3] isa LLVM.ConstantInt ? convert(Int64, ops[3]) : 0
            flat_offset = first_idx * array_len + inner_idx

            # Get pointer to element 0
            zero_id = emit_constant_u32!(state.mod, UInt32(0))

            # Map element type for Workgroup storage class
            elem_ty = LLVM.eltype(source_ty)
            elem_spirv = if sc == SC.Workgroup
                map_workgroup_type!(state.type_ctx, elem_ty)
            else
                map_type!(state.type_ctx, elem_ty)
            end
            elem_ptr_ty = map_pointer_type!(state.type_ctx, elem_spirv, sc)

            base_elem_id = fresh_id!(state.mod)
            encode_instruction!(state.mod.functions, Op.OpAccessChain,
                                elem_ptr_ty, base_elem_id, base_id, zero_id)

            # PtrAccessChain by flat_offset (signed integer offset)
            # Use i32 constant with signed value
            flat_offset_i32 = Int32(flat_offset)
            offset_id = emit_constant_u32!(state.mod, reinterpret(UInt32, flat_offset_i32))

            # Ensure ArrayStride decoration for PtrAccessChain
            _ensure_array_stride_decoration!(state, elem_ptr_ty, elem_ty)

            result_id = fresh_id!(state.mod)
            # OpPtrAccessChain = opcode 67
            encode_instruction!(state.mod.functions, UInt16(67),
                                elem_ptr_ty, result_id, base_elem_id, offset_id)

            # If there are additional indices beyond the second (ops[4:end]),
            # chain more AccessChains
            if length(ops) > 3
                for i in 4:length(ops)
                    idx_val = ops[i]
                    idx_i64 = idx_val isa LLVM.ConstantInt ? convert(Int64, idx_val) : 0
                    idx_id = emit_constant_u32!(state.mod, UInt32(idx_i64))
                    new_id = fresh_id!(state.mod)
                    # TODO: compute correct result type for deeper indexing
                    encode_instruction!(state.mod.functions, Op.OpAccessChain,
                                        result_ptr_ty, new_id, result_id, idx_id)
                    result_id = new_id
                end
            end

            set_pointee_type!(state.type_ctx.ptm, val, result_pointee; priority=4)
            return result_id
        end

        # Normal case: first index is 0, skip it and use remaining indices
        # All indices in a ConstantExpr are constants — emit them as u32 SPIR-V constants
        index_ids = UInt32[]
        for i in 3:length(ops)  # skip base (ops[1]) AND first index (ops[2] = pointer offset)
            idx_val = ops[i]
            if idx_val isa LLVM.ConstantInt
                idx_i64 = convert(Int64, idx_val)
                push!(index_ids, emit_constant_u32!(state.mod, UInt32(idx_i64)))
            else
                error("ConstantExpr GEP with non-constant index: $val")
            end
        end

        if isempty(index_ids)
            # No remaining indices after skipping pointer offset — identity
            state.value_map[val] = base_id
            set_pointee_type!(state.type_ctx.ptm, val, result_pointee; priority=4)
            return base_id
        end

        result_id = fresh_id!(state.mod)
        word_count = UInt32(4 + length(index_ids))
        push!(state.mod.functions, (word_count << 16) | UInt32(Op.OpAccessChain))
        push!(state.mod.functions, result_ptr_ty)
        push!(state.mod.functions, result_id)
        push!(state.mod.functions, base_id)
        append!(state.mod.functions, index_ids)
        # Do NOT cache: state.value_map[val] = result_id
        # ConstantExpr GEPs are reused across blocks; caching would cause
        # domination errors. Each use gets a fresh AccessChain.
        # Update PTM for downstream resolution
        set_pointee_type!(state.type_ctx.ptm, val, result_pointee; priority=4)
        return result_id
    elseif op == LLVM.API.LLVMBitCast || op == LLVM.API.LLVMAddrSpaceCast
        # Constant bitcast / addrspacecast — pass through
        ops = LLVM.operands(val)
        return get_value_id!(state, ops[1])
    elseif op == LLVM.API.LLVMIntToPtr
        # inttoptr(small_constant) — Julia runtime error paths (GC tag slots).
        # These stores should have been removed by _remove_julia_runtime_artifacts!
        # but if any survive, emit as a zero pointer constant (dead code path).
        # SPIR-V doesn't have inttoptr; use OpConvertUToPtr if available,
        # or just return a null/zero constant since this is dead error code.
        result_ty = LLVM.value_type(val)
        # Create a null pointer — these paths never execute on GPU
        spirv_ty = map_type!(state.type_ctx, result_ty)
        null_id = fresh_id!(state.mod)
        encode_instruction!(state.mod.types_constants, Op.OpConstantNull, spirv_ty, null_id)
        return null_id
    else
        error("Unsupported ConstantExpr opcode: $op in $val")
    end
end
