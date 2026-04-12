# Shared CFG primitives used by the structurize pipeline
# (isolate_shared_merge_targets!, fixup_continue_merge_conflicts!, and
# anyone else who needs to insert trampolines or reason about back-edges).
#
# These are low-level LLVM-IR helpers: no globals, no Lava-specific state, no
# compilation-level caches. Each function takes its `LLVM.Function`/`Module`
# inputs and returns the new objects it created (or mutates phis in place).
#
# Design rules:
#   * One primitive per CFG transform. No "also does X if Y" branching.
#   * Callers compose primitives; primitives don't know about their callers.
#   * Phi updates are always done as part of the edge-redirection primitive —
#     never deferred to the caller. Keeping them paired keeps SSA consistent
#     across every intermediate call-graph state.

"""
    compute_rpo(f::LLVM.Function) -> Vector{LLVM.BasicBlock}

Reverse post-order of `f`'s basic blocks. The DFS starts at the function's
first block (the entry) and appends unreachable blocks at the end so every
block in `f` appears exactly once.

RPO is the canonical ordering for identifying natural loops: block `A` has a
back-edge to block `B` iff `B` precedes `A` in RPO.
"""
function compute_rpo(f::LLVM.Function)
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
    for bb in blocks
        bb in visited || push!(postorder, bb)
    end
    return reverse(postorder)
end

"""
    insert_edge_trampoline!(f::LLVM.Function,
                            sources::AbstractVector{LLVM.BasicBlock},
                            target::LLVM.BasicBlock;
                            name::String = "tramp") ->
        LLVM.BasicBlock

Create a fresh block that sits on every `source → target` edge: rewrite each
source's terminator to replace occurrences of `target` with the trampoline,
then emit a single `br target` in the trampoline. Update every phi in
`target` so the N entries previously from `sources` collapse into a single
entry from the trampoline.

If all `sources` contribute the same LLVM value to a given target phi (SSA
identity), the trampoline carries that value directly. Otherwise a new phi
is inserted at the top of the trampoline to merge the per-source values and
the trampoline passes its own phi result downstream.

Returns the newly-created trampoline block.
"""
function insert_edge_trampoline!(f::LLVM.Function,
                                   sources::AbstractVector{LLVM.BasicBlock},
                                   target::LLVM.BasicBlock;
                                   name::String = "tramp")
    source_set = Set(sources)
    tramp = LLVM.BasicBlock(f, name)
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, tramp)
        LLVM.br!(builder, target)
    end

    # Redirect every source's terminator away from `target` and into `tramp`.
    for src in sources
        term = LLVM.terminator(src)
        succs = LLVM.successors(term)
        for i in 1:length(succs)
            if succs[i] == target
                succs[i] = tramp
            end
        end
    end

    # Rewrite every phi in `target`: partition its incoming pairs into those
    # coming from `sources` vs. those coming from elsewhere. Collapse the
    # `sources` side to a single operand (either a shared value or a fresh
    # phi in `tramp`).
    for inst in collect(LLVM.instructions(target))
        inst isa LLVM.PHIInst || break

        inside_pairs  = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        outside_pairs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        for (val, blk) in LLVM.incoming(inst)
            if blk in source_set
                push!(inside_pairs, (val, blk))
            else
                push!(outside_pairs, (val, blk))
            end
        end
        isempty(inside_pairs) && continue

        tramp_val = if length(inside_pairs) == 1 ||
                       all(p -> p[1].ref == inside_pairs[1][1].ref, inside_pairs)
            inside_pairs[1][1]
        else
            # Heterogeneous sources — materialize a phi in the trampoline.
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, LLVM.terminator(tramp))
                phi = LLVM.phi!(builder, LLVM.value_type(inst), "tramp.phi")
                append!(LLVM.incoming(phi), inside_pairs)
                phi
            end
        end

        # Rebuild target's phi: outside operands untouched, one new operand
        # from the trampoline.
        new_pairs = copy(outside_pairs)
        push!(new_pairs, (tramp_val, tramp))
        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, inst)
            new_phi = LLVM.phi!(builder, LLVM.value_type(inst),
                                LLVM.name(inst) * ".tramp")
            append!(LLVM.incoming(new_phi), new_pairs)
            LLVM.replace_uses!(inst, new_phi)
            LLVM.erase!(inst)
        end
    end

    return tramp
end
