# LLVM pass: CFG cleanup and intrinsic lowering.
#
# Ported from Abacus compilation.jl (replace_unreachable!, replace_freeze!,
# strip_noreturn!, strip_assume!).
#
# These passes clean up LLVM IR patterns that the SPIR-V emitter cannot handle:
# - `unreachable` terminators from GPUCompiler's lower_throw! (create dead-end blocks)
# - `freeze` instructions (no SPIR-V equivalent, SPIR-V has no poison/undef semantics)
# - `noreturn` attributes (SimplifyCFG reintroduces unreachable after replace_unreachable!)
# - `llvm.assume` calls (can cause validation failures when misplaced near merge blocks)
#
# Pass pipeline order (assembled in compilation.jl):
#   GPUCompiler.rm_trap! -> replace_unreachable! -> replace_freeze!
#   -> strip_noreturn! -> strip_assume!
#   (then SimplifyCFG -> structurize_cfg pipeline)

"""
    replace_unreachable!(mod::LLVM.Module)

Replace `unreachable` terminators with branches to a unified return block.

Ported from Metal.jl's `replace_unreachable!` (GPUCompiler/src/metal.jl).
After GPUCompiler's `lower_throw!` and `rm_trap!`, error paths end with
`unreachable` terminators. These create dead-end blocks that break structured
control flow required by Vulkan SPIR-V. We replace them with branches to a
unified return block.

For void functions, the return block contains `ret void`.
For value-returning functions, a PHI node merges the return values from all
predecessors (with `undef` for formerly-unreachable paths).
"""
function replace_unreachable!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Find unreachable instructions and exit blocks
        unreachables = LLVM.Instruction[]
        exit_blocks = LLVM.BasicBlock[]
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            if inst isa LLVM.UnreachableInst
                push!(unreachables, inst)
            end
            if inst isa LLVM.RetInst
                push!(exit_blocks, bb)
            end
        end
        isempty(unreachables) && continue

        # If no exit block exists, create one with `ret void` (or `ret null`)
        if isempty(exit_blocks)
            ret_type = LLVM.return_type(LLVM.function_type(f))
            return_block = LLVM.BasicBlock(f, "ret")
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, return_block)
                if ret_type == LLVM.VoidType()
                    LLVM.ret!(builder)
                else
                    LLVM.ret!(builder, LLVM.null(ret_type))
                end
            end
            push!(exit_blocks, return_block)
        end

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            # Use the last exit block (mirrors Metal's heuristic)
            exit_block = last(exit_blocks)
            ret = LLVM.terminator(exit_block)

            # Create a dedicated return block with only the return instruction.
            # If the exit block already only contains the ret, reuse it.
            if first(LLVM.instructions(exit_block)) == ret
                return_block = exit_block
            else
                return_block = LLVM.BasicBlock(f, "ret")
                LLVM.API.LLVMMoveBasicBlockAfter(return_block, exit_block)

                LLVM.position!(builder, ret)
                LLVM.br!(builder, return_block)

                LLVM.API.LLVMInstructionRemoveFromParent(ret)
                LLVM.position!(builder, return_block)
                LLVM.API.LLVMInsertIntoBuilder(builder, ret)
            end

            # When returning a value, add a phi node to merge return values
            ret_ops = LLVM.operands(ret)
            if !isempty(ret_ops)
                LLVM.position!(builder, ret)
                val = ret_ops[1]
                phi = LLVM.phi!(builder, LLVM.value_type(val))
                for pred in LLVM.predecessors(return_block)
                    push!(LLVM.incoming(phi), (val, pred))
                end
                LLVM.operands(ret)[1] = phi
            end

            # Replace unreachable terminators with branches to return block
            for unreachable in unreachables
                bb = LLVM.parent(unreachable)

                LLVM.position!(builder, unreachable)
                LLVM.br!(builder, return_block)
                LLVM.erase!(unreachable)

                # Patch up phi nodes in the return block with undef values
                for inst in LLVM.instructions(return_block)
                    if inst isa LLVM.PHIInst
                        undef = LLVM.UndefValue(LLVM.value_type(inst))
                        push!(LLVM.incoming(inst), (undef, bb))
                    end
                end
            end
        end
    end
end

"""
    replace_freeze!(mod::LLVM.Module)

Replace `freeze` instructions with their operands.

The `freeze` instruction makes poison/undef values deterministic, but SPIR-V
has no poison/undef semantics, so we can safely replace `freeze %x` with `%x`.
Without this, the SPIR-V emitter would need to handle an opcode that has no
SPIR-V equivalent.
"""
function replace_freeze!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        to_erase = LLVM.Instruction[]
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            inst isa LLVM.FreezeInst || continue
            LLVM.replace_uses!(inst, LLVM.operands(inst)[1])
            push!(to_erase, inst)
        end
        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

"""
    strip_noreturn!(mod::LLVM.Module)

Strip `noreturn` attribute from all functions and call sites in the module.

After `replace_unreachable!` converts error paths to normal returns,
the functions are no longer truly noreturn. But LLVM's SimplifyCFG sees the
`noreturn` attribute and reintroduces `unreachable` terminators after calls
to these functions, undoing our fix. Stripping the attribute prevents this.
"""
function strip_noreturn!(mod::LLVM.Module)
    noreturn_kind = LLVM.kind(LLVM.EnumAttribute("noreturn"))
    for f in LLVM.functions(mod)
        # Strip from function definition attributes
        for attr in collect(LLVM.function_attributes(f))
            if LLVM.kind(attr) == noreturn_kind
                delete!(LLVM.function_attributes(f), attr)
            end
        end
        # Strip from call-site attributes on call instructions
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            if inst isa LLVM.CallInst
                for attr in collect(LLVM.function_attributes(inst))
                    if LLVM.kind(attr) == noreturn_kind
                        delete!(LLVM.function_attributes(inst), attr)
                    end
                end
            end
        end
    end
end

"""
    strip_assume!(mod::LLVM.Module)

Remove `llvm.assume` intrinsic calls from the module.

These are optimization hints that have no effect on correctness. The SPIR-V
emitter doesn't need them, and they can cause issues if misplaced relative
to OpSelectionMerge instructions during structured control flow emission.
"""
function strip_assume!(mod::LLVM.Module)
    if haskey(LLVM.functions(mod), "llvm.assume")
        assume_fn = LLVM.functions(mod)["llvm.assume"]
        for use in collect(LLVM.uses(assume_fn))
            val = LLVM.user(use)
            if val isa LLVM.CallInst
                LLVM.erase!(val)
            end
        end
        if isempty(LLVM.uses(assume_fn))
            LLVM.erase!(assume_fn)
        end
    end
end

"""
    fix_barrier_skipping_paths!(entry_fn::LLVM.Function)

After inlining and optimization, detect error paths (from `replace_unreachable!`)
that branch directly to the return block, skipping `OpControlBarrier` calls that
other invocations in the workgroup will reach.

Per the Vulkan spec (and OpenCL), ALL invocations in a workgroup must reach every
`OpControlBarrier`. If an error path causes one invocation to return early while
others continue to a barrier, the behavior is undefined (deadlock on CPU/software
implementations, undefined on GPU hardware).

**Algorithm**: For each block that branches directly to the return block, check if
its predecessor's alternative path leads to a barrier. If so, redirect the block
to the barrier-containing path. This makes dead invocations participate in all
remaining barriers before returning.
"""
function fix_barrier_skipping_paths!(entry_fn::LLVM.Function)
    isempty(LLVM.blocks(entry_fn)) && return false

    # 1. Find all blocks containing barrier calls
    barrier_fn_name = "llvm.spv.group.memory.barrier.with.group.sync"
    barrier_blocks = Set{LLVM.BasicBlock}()
    for bb in LLVM.blocks(entry_fn), inst in LLVM.instructions(bb)
        if inst isa LLVM.CallInst
            callee = LLVM.called_operand(inst)
            if callee isa LLVM.Function && LLVM.name(callee) == barrier_fn_name
                push!(barrier_blocks, bb)
            end
        end
    end
    isempty(barrier_blocks) && return false

    # 2. Find the return block(s)
    return_blocks = Set{LLVM.BasicBlock}()
    for bb in LLVM.blocks(entry_fn)
        term = LLVM.terminator(bb)
        if term isa LLVM.RetInst
            push!(return_blocks, bb)
        end
    end
    isempty(return_blocks) && return false

    # 3. Find blocks that skip barriers: branch directly to a return block,
    #    while their predecessor's other path leads to a barrier.
    changed = false
    for bb in collect(LLVM.blocks(entry_fn))
        bb in barrier_blocks && continue
        bb in return_blocks && continue

        term = LLVM.terminator(bb)
        term isa LLVM.BrInst || continue
        succs = collect(LLVM.successors(term))
        length(succs) == 1 || continue   # must be unconditional branch

        target = succs[1]
        target in return_blocks || continue   # must branch to return block

        # This block branches directly to return. Check predecessor.
        preds = collect(LLVM.predecessors(bb))
        length(preds) == 1 || continue

        pred = preds[1]
        pred_term = LLVM.terminator(pred)
        pred_term isa LLVM.BrInst || continue
        pred_succs = collect(LLVM.successors(pred_term))
        length(pred_succs) == 2 || continue  # must be conditional branch

        # Find the "other" target (the non-error path)
        other_target = pred_succs[1] == bb ? pred_succs[2] : pred_succs[1]

        # Check if the other path leads to a barrier (directly or transitively)
        if path_reaches_barrier(other_target, barrier_blocks, return_blocks)
            # Redirect this block to the barrier-containing path
            LLVM.@dispose builder = LLVM.IRBuilder() begin
                LLVM.position!(builder, term)
                LLVM.br!(builder, other_target)
            end
            LLVM.erase!(term)

            # Add incoming values for any PHI nodes in the target
            for inst in LLVM.instructions(other_target)
                inst isa LLVM.PHIInst || break  # PHIs must be at block start
                undef = LLVM.UndefValue(LLVM.value_type(inst))
                push!(LLVM.incoming(inst), (undef, bb))
            end

            changed = true
        end
    end

    return changed
end

"""
Check if any path from `start` reaches a barrier block before reaching a return block.
Uses BFS with visited set to handle cycles.
"""
function path_reaches_barrier(start::LLVM.BasicBlock,
                               barrier_blocks::Set{LLVM.BasicBlock},
                               return_blocks::Set{LLVM.BasicBlock})
    start in barrier_blocks && return true
    visited = Set{LLVM.BasicBlock}()
    queue = LLVM.BasicBlock[start]
    while !isempty(queue)
        bb = popfirst!(queue)
        bb in visited && continue
        push!(visited, bb)
        bb in barrier_blocks && return true
        bb in return_blocks && continue  # don't search past return
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            succ in visited || push!(queue, succ)
        end
    end
    return false
end

"""
    remove_julia_runtime_artifacts!(mod::LLVM.Module)

Remove Julia runtime artifacts that appear after force-inlining error paths.

After GPUCompiler's lower_throw! + rm_trap! + force inlining, the entry function
may contain code from inlined error helpers that references Julia runtime:
- `load i64, ptr @jl_int64_type` — loading type tags for boxing
- `store i64 %val, ptr inttoptr (i64 1 to ptr)` — stores to GC tag slots

These are dead error paths that should never execute on GPU. We replace:
1. Stores to ConstantExpr inttoptr with small constants → delete
2. Loads from external function declarations (jl_*_type) → replace with zero
3. Runs DCE to clean up dead code chains
"""
function remove_julia_runtime_artifacts!(mod::LLVM.Module)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                # Remove stores to inttoptr(small_constant) — Julia GC/error paths
                if inst isa LLVM.StoreInst
                    ops = LLVM.operands(inst)
                    ptr_op = ops[2]
                    if ptr_op isa LLVM.ConstantExpr && LLVM.opcode(ptr_op) == LLVM.API.LLVMIntToPtr
                        ce_ops = LLVM.operands(ptr_op)
                        if ce_ops[1] isa LLVM.ConstantInt
                            addr = convert(UInt64, ce_ops[1])
                            if addr < 4096  # Small addresses are error paths
                                push!(to_erase, inst)
                            end
                        end
                    end
                end

                # Replace loads from external function decls (jl_*_type) with zero
                if inst isa LLVM.LoadInst
                    ptr_op = LLVM.operands(inst)[1]
                    if ptr_op isa LLVM.Function && isempty(LLVM.blocks(ptr_op))
                        load_ty = LLVM.value_type(inst)
                        if load_ty isa LLVM.IntegerType
                            zero = LLVM.ConstantInt(load_ty, 0)
                            LLVM.replace_uses!(inst, zero)
                            push!(to_erase, inst)
                        end
                    end
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

"""
    run_cfg_cleanup!(mod::LLVM.Module)

Run the full CFG cleanup pipeline. Must be called BEFORE the structurize_cfg pipeline.

Order:
1. GPUCompiler.rm_trap!(mod) -- call this separately before this function
2. replace_unreachable! -- replace dead-end blocks with branches to exit
3. replace_freeze! -- remove freeze instructions (no SPIR-V equivalent)
4. strip_noreturn! -- prevent SimplifyCFG from reintroducing unreachable
5. strip_assume! -- remove optimization hints that cause merge placement issues
"""
function run_cfg_cleanup!(mod::LLVM.Module)
    replace_unreachable!(mod)
    replace_freeze!(mod)
    strip_noreturn!(mod)
    strip_assume!(mod)
end
