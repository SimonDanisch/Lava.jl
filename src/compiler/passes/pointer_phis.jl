# WGSL has no pointer variables: a pointer is never selected, merged or stored, and
# SPIR-V without VariablePointers says the same. Code reaching a WebGPU module
# still merges pointers wherever two paths index one array differently and then
# share the load: AcceleratedKernels' merge sort reads `s_keys[...]` after a branch,
# and the structurizer's Flow blocks add phis of their own.
#
# When every pointer flowing into a phi or select is the same array at some index
# (`getelementptr T, base, i`), the merge is a merge of indices: the phi/select
# becomes one of the index, and the pointer one GEP after it. A merge of different
# arrays has no WGSL form and is left for the emitter to refuse.

"""One array element: `base` read as an array of `elt`, at `index` (an LLVM integer)."""
struct ArrayElement
    elt::LLVM.LLVMType
    base::LLVM.Value
    index::LLVM.Value
end

"""
A pointer as an `ArrayElement`, `missing` for an undefined pointer (any array fits
it), or `nothing` for a pointer that is no array element.
"""
function arrayelement(v::LLVM.Value)
    (v isa LLVM.UndefValue || v isa LLVM.PoisonValue) && return missing
    isgep = v isa LLVM.GetElementPtrInst ||
            (v isa LLVM.ConstantExpr && LLVM.opcode(v) == LLVM.API.LLVMGetElementPtr)
    isgep || return nothing
    ops = LLVM.operands(v)
    base = ops[1]
    base isa Union{LLVM.GlobalVariable, LLVM.AllocaInst} || return nothing
    src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(v))
    length(ops) == 2 && return ArrayElement(src, base, ops[2])
    if length(ops) == 3 && src isa LLVM.ArrayType && ops[2] isa LLVM.ConstantInt && convert(Int, ops[2]) == 0
        return ArrayElement(eltype(src), base, ops[3])
    end
    return nothing
end

mergesources(m::LLVM.PHIInst) = [v for (v, _) in LLVM.incoming(m)]
mergesources(m::LLVM.SelectInst) = LLVM.operands(m)[2:3]

"""
The array every pointer reaching each merge points into: `(elt, base)`, or
`nothing` where they disagree or one is no array element. Merges feeding merges
(around a loop) are settled together; a merge that only ever sees undefined
pointers is unresolved, and so is everything it feeds.
"""
function mergetargets(merges::Vector{LLVM.Instruction})
    ismerge = Set(merges)
    target = Dict{LLVM.Instruction, Union{Nothing, Tuple{LLVM.LLVMType, LLVM.Value}}}()
    source(v) = if v in ismerge
        get(target, v, missing)
    else
        e = arrayelement(v)
        e isa ArrayElement ? (e.elt, e.base) : e
    end
    function settle!()
        changed = true
        while changed
            changed = false
            for m in merges
                t = get(target, m, missing)
                t === nothing && continue
                for v in mergesources(m)
                    s = source(v)
                    s === missing && continue
                    if s === nothing || (t !== missing && (t[1] != s[1] || t[2] != s[2]))
                        target[m] = nothing
                        changed = true
                        break
                    elseif t === missing
                        target[m] = t = s
                        changed = true
                    end
                end
            end
        end
    end
    settle!()
    foreach(m -> haskey(target, m) || (target[m] = nothing), merges)
    settle!()
    return target
end

"""
    index_pointer_phis!(fn) -> Int

Replace the phis and selects of pointers into one array in `fn` by phis and selects
of the index; returns how many it replaced.
"""
function index_pointer_phis!(fn::LLVM.Function)
    merges = LLVM.Instruction[]
    for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        (inst isa LLVM.PHIInst || inst isa LLVM.SelectInst) || continue
        LLVM.value_type(inst) isa LLVM.PointerType && push!(merges, inst)
    end
    isempty(merges) && return 0
    target = mergetargets(merges)
    resolved = [m for m in merges if target[m] !== nothing]
    isempty(resolved) && return 0

    T = LLVM.Int64Type()
    index = Dict{LLVM.Instruction, LLVM.Instruction}()
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        # The index merges first, so that merges feeding merges can name each other.
        for m in resolved
            LLVM.position!(builder, insertion_point(m))
            index[m] = m isa LLVM.PHIInst ? LLVM.phi!(builder, T, "idx.merge") :
                LLVM.select!(builder, LLVM.operands(m)[1], LLVM.UndefValue(T), LLVM.UndefValue(T), "idx.select")
        end
        # The index a source pointer stands for, computed before `at`.
        function indexof(v, at)
            haskey(index, v) && return index[v]
            e = arrayelement(v)
            e === missing && return LLVM.UndefValue(T)
            idx = (e::ArrayElement).index
            LLVM.value_type(idx) == T && return idx
            idx isa LLVM.ConstantInt && return LLVM.ConstantInt(T, convert(Int, idx))
            LLVM.position!(builder, insertion_point(at))
            return LLVM.sext!(builder, idx, T)
        end
        for m in resolved
            if m isa LLVM.PHIInst
                append!(LLVM.incoming(index[m]),
                        [(indexof(v, LLVM.terminator(bb)), bb) for (v, bb) in LLVM.incoming(m)])
            else
                a, b = LLVM.operands(m)[2:3]
                LLVM.operands(index[m])[2] = indexof(a, index[m])
                LLVM.operands(index[m])[3] = indexof(b, index[m])
            end
        end
        for m in resolved
            elt, base = target[m]
            m isa LLVM.PHIInst ? LLVM.position!(builder, LLVM.after_phis(LLVM.parent(m))) :
                                 LLVM.position!(builder, insertion_point(m))
            LLVM.replace_uses!(m, LLVM.gep!(builder, elt, base, LLVM.Value[index[m]], "ptr.merge"))
        end
        foreach(LLVM.erase!, resolved)
    end
    return length(resolved)
end
