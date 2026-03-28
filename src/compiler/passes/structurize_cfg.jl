# LLVM pass: CFG structurization for Vulkan SPIR-V compliance.
#
# Ported from Abacus compilation.jl (_fixup_structured_cfg!, _isolate_shared_merge_targets!,
# _insert_cfg_trampoline!).
#
# Vulkan requires structured control flow (no irreducible loops, single-entry/single-exit
# regions). Complex Julia code (CartesianIndices, bounds checking, error paths) generates
# CFGs that violate this. The standard LLVM pass sequence handles most cases:
#   LowerSwitch -> UnifyFunctionExitNodes -> FixIrreducible -> StructurizeCFG
#
# However, StructurizeCFG needs pre-processing: when a block is the target of multiple
# conditional branches from different nesting levels, StructurizeCFG can produce wrong
# back-edge conditions (loop exits after 1 iteration). This pass inserts trampoline blocks
# to give each construct its own unique merge target, analogous to clspv's
# FixupStructuredCFGPass / isolateContinue().
#
# Pass pipeline order (assembled in compilation.jl):
#   SimplifyCFG -> _fixup_structured_cfg! -> LowerSwitch -> UnifyFunctionExitNodes
#   -> FixIrreducible -> StructurizeCFG -> InstCombine

"""
    _fixup_structured_cfg!(mod::LLVM.Module)

Pre-StructurizeCFG fixup pass, analogous to clspv's `FixupStructuredCFGPass`.

Detects blocks that are the successor of multiple conditional branches from different
nesting levels and inserts trampoline blocks so each construct gets its own unique
merge target. Without this, StructurizeCFG can produce incorrect back-edge conditions
causing loops to exit after a single iteration.
"""
function _fixup_structured_cfg!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        _isolate_shared_merge_targets!(f)
    end
end

"""
    _isolate_shared_merge_targets!(f::LLVM.Function)

Find blocks that are successors of multiple conditional branches and insert
trampolines for all but the first source. This ensures each structured construct
has a unique merge target for SPIR-V's OpSelectionMerge/OpLoopMerge.
"""
function _isolate_shared_merge_targets!(f::LLVM.Function)
    # Build map: target_block -> [source_blocks that conditionally branch to it]
    cond_sources = Dict{LLVM.BasicBlock, Vector{LLVM.BasicBlock}}()
    for bb in LLVM.blocks(f)
        term = LLVM.terminator(bb)
        term isa LLVM.BrInst || continue
        LLVM.isconditional(term) || continue
        for succ in LLVM.successors(term)
            # Skip self-loops (loop back-edges). These are handled correctly by
            # the SPIR-V backend as OpLoopMerge, not OpSelectionMerge. Including
            # them creates dead trampoline blocks that break PHI node invariants.
            succ == bb && continue
            sources = get!(cond_sources, succ, LLVM.BasicBlock[])
            # Avoid counting the same source twice (both branches to same target)
            bb in sources || push!(sources, bb)
        end
    end

    for (target, sources) in cond_sources
        length(sources) <= 1 && continue

        # Keep the first source unchanged, redirect all others through trampolines.
        # The first source (in block order) keeps the direct edge; later ones get
        # trampolines. This matches clspv's isolateContinue() strategy.
        for src in sources[2:end]
            _insert_cfg_trampoline!(f, src, target)
        end
    end
end

"""
    _insert_cfg_trampoline!(f, src, target)

Insert a trampoline block isolating `src`'s construct from `target`.

When `src` is a conditional branch header whose construct (all blocks reachable
from `src` without passing through `target`) also has blocks that branch to
`target`, ALL those branches must be redirected through the trampoline. Otherwise
the SPIR-V backend creates a construct where inner blocks exit "not via structured
exit". This mirrors clspv's `isolateContinue()`.

Steps:
1. Find all blocks "inside" src's construct (reachable from src without crossing target)
2. Create trampoline block (unconditional branch to target)
3. Redirect ALL branches from inside blocks to target through the trampoline
4. Update PHI nodes in target (partition incoming values into inside/outside)
"""
function _insert_cfg_trampoline!(f::LLVM.Function, src::LLVM.BasicBlock,
                                  target::LLVM.BasicBlock)
    # Step 1: Find all blocks "inside" src's construct.
    # = blocks reachable from src without passing through target.
    inside = Set{LLVM.BasicBlock}()
    worklist = LLVM.BasicBlock[src]
    while !isempty(worklist)
        bb = pop!(worklist)
        bb in inside && continue    # already visited
        bb == target && continue     # don't enter target
        push!(inside, bb)
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            push!(worklist, succ)
        end
    end

    # Step 2: Create trampoline block (just branches to target)
    tramp = LLVM.BasicBlock(f, "cfg_fixup")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, tramp)
        LLVM.br!(builder, target)
    end

    # Step 3: Redirect ALL branches from inside blocks to target through trampoline
    for bb in inside
        term = LLVM.terminator(bb)
        succs = LLVM.successors(term)
        for i in 1:length(succs)
            if succs[i] == target
                succs[i] = tramp
            end
        end
    end

    # Step 4: Update PHI nodes in target.
    # Multiple inside blocks may have been predecessors of target with different
    # values. They all now arrive via trampoline, so we need a PHI in trampoline
    # to merge their values, and a single entry in target's PHI from trampoline.
    for inst in collect(LLVM.instructions(target))
        inst isa LLVM.PHIInst || break  # PHIs are always at block start

        inc = LLVM.incoming(inst)
        # Partition incoming into inside vs outside
        inside_pairs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        outside_pairs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        for k in 1:length(inc)
            val, blk = inc[k]
            if blk in inside
                push!(inside_pairs, (val, blk))
            else
                push!(outside_pairs, (val, blk))
            end
        end

        isempty(inside_pairs) && continue

        # Determine the value arriving from the trampoline
        tramp_val = if length(inside_pairs) == 1
            # Single inside predecessor: use its value directly
            inside_pairs[1][1]
        elseif all(p -> p[1] == inside_pairs[1][1], inside_pairs)
            # All inside predecessors contribute the same value
            inside_pairs[1][1]
        else
            # Multiple different values: create a PHI in the trampoline
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                # Position before the terminator (branch) in trampoline
                LLVM.position!(builder, LLVM.terminator(tramp))
                phi = LLVM.phi!(builder, LLVM.value_type(inst), "tramp.phi")
                append!(LLVM.incoming(phi), inside_pairs)
                phi
            end
        end

        # Rebuild target's PHI with trampoline as single predecessor for inside blocks
        new_pairs = copy(outside_pairs)
        push!(new_pairs, (tramp_val, tramp))

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, inst)
            new_phi = LLVM.phi!(builder, LLVM.value_type(inst), LLVM.name(inst) * ".fix")
            append!(LLVM.incoming(new_phi), new_pairs)
            LLVM.replace_uses!(inst, new_phi)
            LLVM.erase!(inst)
        end
    end
end

"""
    run_structurize_cfg_pipeline!(mod::LLVM.Module)

Run the full CFG structurization pipeline required for Vulkan SPIR-V.

This is the standard LLVM pass sequence used by AMD's GPU compiler, with our
pre-StructurizeCFG fixup pass inserted:

1. SimplifyCFG - merges nested conditionals sharing exit blocks
2. _fixup_structured_cfg! - trampoline insertion for shared merge targets
3. LowerSwitch - converts switch to if-else chains (SPIR-V OpSwitch has limitations)
4. UnifyFunctionExitNodes - ensures single return point (required by StructurizeCFG)
5. FixIrreducible - converts irreducible loops to reducible ones
6. LoopSimplify - creates dedicated exit blocks for nested loops, preventing
   StructurizeCFG from producing constant `br i1 true` on inner loop back-edges
   when inner loops have multi-level exits (exits bypassing outer loops)
7. StructurizeCFG - the critical pass: converts arbitrary CFG to structured control flow
8. InstCombine - cleanup bitcasts from StructurizeCFG's reg2mem patterns
   NOTE: Do NOT run SimplifyCFG after StructurizeCFG -- it destroys structured flow!
9. _fixup_post_structurize! - insert trampolines for continue-target merge conflicts
"""
function run_structurize_cfg_pipeline!(mod::LLVM.Module)
    LLVM.run!(LLVM.SimplifyCFGPass(), mod)
    _fixup_structured_cfg!(mod)
    LLVM.run!(LLVM.LowerSwitchPass(), mod)
    LLVM.run!(LLVM.UnifyFunctionExitNodesPass(), mod)
    LLVM.run!(LLVM.FixIrreduciblePass(), mod)
    LLVM.run!(LLVM.LoopSimplifyPass(), mod)
    LLVM.run!(LLVM.StructurizeCFGPass(), mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    _fixup_post_structurize!(mod)
end

# Post-StructurizeCFG fixup: insert trampolines for SPIR-V continue-construct conflicts.
#
# SPIR-V separates loops into "loop construct" and "continue construct". A selection
# header in the loop construct cannot have its merge block in the continue construct.
# StructurizeCFG often produces patterns where a conditional branch inside a loop
# converges at the loop's continue target. This pass inserts trampoline blocks between
# such selections and the continue target.
function _fixup_post_structurize!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        _fixup_continue_merge_conflicts!(f)
    end
end

# Find loops and insert trampolines where selections merge at the continue target.
function _fixup_continue_merge_conflicts!(f::LLVM.Function)
    blocks = collect(LLVM.blocks(f))
    length(blocks) <= 1 && return

    # Compute RPO
    rpo = _compute_rpo(f)
    rpo_pos = Dict{LLVM.BasicBlock, Int}()
    for (i, bb) in enumerate(rpo)
        rpo_pos[bb] = i
    end

    # Find loops: back-edges A→B where B appears before A in RPO
    loops = Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}()
    for bb in blocks
        term = LLVM.terminator(bb)
        bb_pos = get(rpo_pos, bb, 0)
        for succ in LLVM.successors(term)
            succ_pos = get(rpo_pos, succ, 0)
            if succ_pos > 0 && succ_pos <= bb_pos
                header = succ
                latch = bb
                haskey(loops, header) && continue
                # Find merge: first successor of a loop block that's outside the loop
                merge_bb = _find_loop_merge_llvm(header, latch, rpo, rpo_pos)
                loops[header] = (merge_bb, latch)
            end
        end
    end

    isempty(loops) && return

    # Collect all continue targets
    continue_targets = Set{LLVM.BasicBlock}()
    for (_, (_, latch)) in loops
        push!(continue_targets, latch)
    end

    # Fix 1: For each conditional branch inside a loop, check if both branches
    # converge at a continue target. If so, insert a trampoline.
    for _iter1 in 1:100  # Safety limit to prevent infinite loops
        found = false
        for bb in collect(LLVM.blocks(f))
            term = LLVM.terminator(bb)
            term isa LLVM.BrInst || continue
            LLVM.isconditional(term) || continue

            # Skip loop headers (their branches are handled by OpLoopMerge)
            haskey(loops, bb) && continue

            succs = LLVM.successors(term)
            true_bb = succs[1]
            false_bb = succs[2]

            # Check if both branches converge at a continue target.
            merge_bb = _find_shallow_convergence(true_bb, false_bb)
            merge_bb === nothing && continue

            if merge_bb in continue_targets
                _insert_cfg_trampoline!(f, bb, merge_bb)
                found = true
                break  # Restart since CFG changed
            end
        end
        found || break
    end

    # Fix 2: For each inner loop whose merge block IS a continue target of an
    # outer loop, insert a trampoline between the inner loop exit and the
    # continue target. The inner loop's conditional exit branches to either the
    # merge (= continue target) or the loop body. We redirect the merge-bound
    # branch through a trampoline.
    # Recompute loops since CFG may have changed.
    rpo = _compute_rpo(f)
    rpo_pos = Dict{LLVM.BasicBlock, Int}()
    for (i, bb) in enumerate(rpo)
        rpo_pos[bb] = i
    end
    loops = Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}()
    for bb in collect(LLVM.blocks(f))
        term = LLVM.terminator(bb)
        bb_pos = get(rpo_pos, bb, 0)
        for succ in LLVM.successors(term)
            succ_pos = get(rpo_pos, succ, 0)
            if succ_pos > 0 && succ_pos <= bb_pos
                header = succ
                latch = bb
                haskey(loops, header) && continue
                merge_bb = _find_loop_merge_llvm(header, latch, rpo, rpo_pos)
                loops[header] = (merge_bb, latch)
            end
        end
    end

    continue_targets = Set{LLVM.BasicBlock}()
    for (_, (_, latch)) in loops
        push!(continue_targets, latch)
    end

    for _iter2 in 1:100  # Safety limit to prevent infinite loops
        found2 = false
        for (header, (merge_bb, latch)) in loops
            if merge_bb in continue_targets
                _insert_cfg_trampoline!(f, header, merge_bb)
                found2 = true
                break
            end
        end

        found2 || break

        # Recompute since CFG changed
        begin
            # Recompute RPO and loops
            rpo = _compute_rpo(f)
            rpo_pos = Dict{LLVM.BasicBlock, Int}()
            for (i, bb) in enumerate(rpo)
                rpo_pos[bb] = i
            end
            loops = Dict{LLVM.BasicBlock, Tuple{LLVM.BasicBlock, LLVM.BasicBlock}}()
            for bb in collect(LLVM.blocks(f))
                term = LLVM.terminator(bb)
                bb_pos = get(rpo_pos, bb, 0)
                for succ in LLVM.successors(term)
                    succ_pos = get(rpo_pos, succ, 0)
                    if succ_pos > 0 && succ_pos <= bb_pos
                        h = succ
                        l = bb
                        haskey(loops, h) && continue
                        m = _find_loop_merge_llvm(h, l, rpo, rpo_pos)
                        loops[h] = (m, l)
                    end
                end
            end
            continue_targets = Set{LLVM.BasicBlock}()
            for (_, (_, l)) in loops
                push!(continue_targets, l)
            end
        end
    end
end

# Compute reverse post-order of basic blocks in a function.
function _compute_rpo(f::LLVM.Function)
    blocks = collect(LLVM.blocks(f))
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

    dfs(first(blocks))
    # Also visit unreachable blocks
    for bb in blocks
        bb in visited || push!(postorder, bb)
    end

    return reverse(postorder)
end

# Find the merge block for a loop (first successor outside the loop body).
function _find_loop_merge_llvm(header::LLVM.BasicBlock, latch::LLVM.BasicBlock,
                                rpo::Vector{LLVM.BasicBlock},
                                rpo_pos::Dict{LLVM.BasicBlock, Int})
    header_pos = rpo_pos[header]
    latch_pos = rpo_pos[latch]

    loop_blocks = Set{LLVM.BasicBlock}()
    for bb in rpo
        pos = rpo_pos[bb]
        if pos >= header_pos && pos <= latch_pos
            push!(loop_blocks, bb)
        end
    end

    # Find first successor (in RPO order) that's outside the loop
    for bb in rpo
        bb in loop_blocks || continue
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            if !(succ in loop_blocks)
                return succ
            end
        end
    end

    return latch  # Fallback
end

# Find where two branches converge (shallow: checks direct + 1-hop successors).
# After StructurizeCFG, convergence is always at most 1-2 hops away.
function _find_shallow_convergence(a::LLVM.BasicBlock, b::LLVM.BasicBlock)
    # Case 1: a is a direct successor of b (if-else with inverted condition)
    b_succs = Set(LLVM.successors(LLVM.terminator(b)))
    if a in b_succs
        return a
    end

    # Case 2: b is a direct successor of a (if-then pattern)
    a_succs = Set(LLVM.successors(LLVM.terminator(a)))
    if b in a_succs
        return b
    end

    # Case 3: common direct successor
    common = intersect(a_succs, b_succs)
    if !isempty(common)
        return first(common)
    end

    # Case 4: a leads to X, b leads to X (1-hop from each)
    a_2hop = Set{LLVM.BasicBlock}()
    for s in a_succs
        for ss in LLVM.successors(LLVM.terminator(s))
            push!(a_2hop, ss)
        end
    end
    b_2hop = Set{LLVM.BasicBlock}()
    for s in b_succs
        for ss in LLVM.successors(LLVM.terminator(s))
            push!(b_2hop, ss)
        end
    end

    # Check if b (or b's successors) reach something in a's direct successors
    common2 = intersect(a_succs, b_2hop)
    if !isempty(common2)
        return first(common2)
    end
    common3 = intersect(a_2hop, b_succs)
    if !isempty(common3)
        return first(common3)
    end
    common4 = intersect(a_2hop, b_2hop)
    if !isempty(common4)
        return first(common4)
    end

    return nothing
end
