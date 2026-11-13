# Loads from memory that no thread writes while the kernel runs.
#
# Mantle's access walk says which arguments a kernel reads and does not write, and whose
# bytes no written argument reaches; the compile job carries them as
# `LavaCompilerParams.readonly`, a bit per argument. A load from memory such an argument
# reaches, at an address every invocation of a subgroup shares, gets `!invariant.load`,
# which the emitter turns into a read through a NonWritable member, the only way SPIR-V
# says that memory behind a PSB pointer is read-only: RADV reads a uniform address
# through its scalar cache only then. The flags replaced wrapping those arguments in
# `Lava.Const`, which made the compiled signature another than the walked one, so Julia
# inferred every kernel a second time.
#
# Only uniform addresses, because NonWritable on a per-invocation load buys nothing a
# driver showed and frees it to move the load: marking every read-only load took a
# Hikari shading kernel from 168 to 255 registers on an RTX 4000 Ada (frame 2.5%
# slower), while the uniform loads alone left every kernel's registers as they were and
# still moved RADV's uniform reads to scalar loads.
#
# The pass runs once `force_inline_all!` has inlined the kernel into the entry wrapper.
# The wrapper loads each argument from the argument buffer, whose address is the first
# push constant. The slot of an argument passed by reference holds the address of its
# bytes, which the kernel loads the argument's arrays from; any other slot holds the
# argument itself. A pointer loaded from a read-only argument's bytes, or held by its slot,
# addresses memory nothing writes, and so does every address computed from it. Values
# loaded from that memory are data, never followed: a read-only array of arrays makes the
# outer array read-only, as `Lava.Const` did, not the arrays it holds.

"""
One argument the compile job marks read-only: its slot in the argument buffer, the slot's
size, and whether the slot holds the address of the argument's bytes (an argument passed
by reference) or the argument itself.
"""
struct ReadonlySlot
    offset::Int
    size::Int
    byreference::Bool
end

"""
    readonly_slots(job, entry, push_info) -> Vector{ReadonlySlot}

The argument-buffer slots of the arguments `job` marks read-only, through the parameters
GPUCompiler made of them for `entry` (an argument that occupies none has no slot).
"""
function readonly_slots(job::GPUCompiler.CompilerJob, entry::LLVM.Function, push_info::PushConstantInfo)
    mask = job.config.params.readonly
    mask == 0 && return ReadonlySlot[]
    args = GPUCompiler.classify_arguments(job, entry.function_type)
    # `args[1]` is the kernel function: bit `k - 1` is about argument `k + 1`.
    64 - leading_zeros(mask) < length(args) || throw(ArgumentError(
        "readonly bits up to argument $(64 - leading_zeros(mask)), but the kernel takes $(length(args) - 1)"))
    slots = ReadonlySlot[]
    for k in 1:length(args) - 1
        arg = args[k + 1]
        (isodd(mask >> (k - 1)) && arg.cc != GPUCompiler.GHOST) || continue
        offset, size = push_info.arg_layout[arg.idx]
        push!(slots, ReadonlySlot(offset, size, arg.cc == GPUCompiler.BITS_REF))
    end
    return slots
end

# Where an SSA value of the kernel points, as far as this pass follows it.
abstract type Origin end
"""The argument buffer's address plus `offset` bytes."""
struct ArgumentBuffer <: Origin
    offset::Int
end
"""The push-constant block as one value, whose first field is the argument buffer's address."""
struct PushConstants <: Origin end
"""An address in the bytes of a read-only argument passed by reference."""
struct ArgumentBytes <: Origin end
"""An address in device memory that nothing writes while the kernel runs."""
struct ReadOnlyMemory <: Origin end

const Origins = Dict{LLVM.Value,Origin}

# The origin of `inst`'s result from its operands' origins, or `nothing` where the pass
# stops following. Addresses inside an argument or an array stay inside it whatever is
# added to them.
origin(::LLVM.Instruction, o::Origins, slots, dl) = nothing

origin(inst::Union{LLVM.BitCastInst,LLVM.AddrSpaceCastInst,LLVM.IntToPtrInst,LLVM.PtrToIntInst,
                   LLVM.FreezeInst}, o::Origins, slots, dl) =
    get(o, inst.operands[1], nothing)

function origin(inst::LLVM.GetElementPtrInst, o::Origins, slots, dl)
    base = get(o, inst.operands[1], nothing)
    base isa ArgumentBuffer || return base
    offset = constant_gep_offset(inst, dl)
    return offset === nothing ? nothing : ArgumentBuffer(base.offset + offset)
end

function origin(inst::Union{LLVM.AddInst,LLVM.SubInst}, o::Origins, slots, dl)
    a, b = inst.operands
    oa, ob = get(o, a, nothing), get(o, b, nothing)
    sign = inst isa LLVM.SubInst ? -1 : 1
    oa isa ArgumentBuffer && b isa LLVM.ConstantInt && return ArgumentBuffer(oa.offset + sign * convert(Int, b))
    inst isa LLVM.AddInst && ob isa ArgumentBuffer && a isa LLVM.ConstantInt &&
        return ArgumentBuffer(ob.offset + convert(Int, a))
    oa isa Union{ArgumentBytes,ReadOnlyMemory} && ob === nothing && return oa
    inst isa LLVM.AddInst && ob isa Union{ArgumentBytes,ReadOnlyMemory} && oa === nothing && return ob
    return nothing
end

function origin(inst::LLVM.LoadInst, o::Origins, slots, dl)
    from = get(o, inst.operands[1], nothing)
    from isa ArgumentBuffer && return slotorigin(slots, from.offset)
    from isa ArgumentBytes && return ReadOnlyMemory()
    return nothing
end

function origin(inst::LLVM.ExtractValueInst, o::Origins, slots, dl)
    from = get(o, inst.operands[1], nothing)
    # The argument buffer's address is the push block's first field.
    from isa PushConstants && return unsafe_load(LLVM.API.LLVMGetIndices(inst)) == 0 ? ArgumentBuffer(0) : nothing
    return from isa ReadOnlyMemory ? from : nothing
end

origin(inst::Union{LLVM.PHIInst,LLVM.SelectInst}, o::Origins, slots, dl) =
    commonorigin(o, mergedvalues(inst))

mergedvalues(inst::LLVM.PHIInst) = (v for (v, _) in LLVM.incoming(inst))
mergedvalues(inst::LLVM.SelectInst) = (inst.operands[2], inst.operands[3])

ismissingvalue(v) = v isa Union{LLVM.UndefValue,LLVM.PoisonValue}

# The origin every merged value whose origin is known has, or `nothing` if two differ.
# Optimistic, for loops: `confirmed` checks it once the origins have settled.
function commonorigin(o::Origins, values)
    common = nothing
    for v in values
        ismissingvalue(v) && continue
        ov = get(o, v, nothing)
        ov === nothing && continue
        common === nothing ? (common = ov) : (common == ov || return nothing)
    end
    return common
end

# Whether every merged value has the origin the merge was given.
confirmed(o::Origins, inst, origin) =
    all(v -> ismissingvalue(v) || get(o, v, nothing) == origin, mergedvalues(inst))

"""What the slot at `offset` holds, if it belongs to a read-only argument."""
function slotorigin(slots::Vector{ReadonlySlot}, offset::Int)
    for s in slots
        if s.byreference
            offset == s.offset && return ArgumentBytes()
        elseif s.offset <= offset < s.offset + s.size
            return ReadOnlyMemory()
        end
    end
    return nothing
end

"""The origins of `kernel`'s values, from the loads of `push`, the push-constant block."""
function origins(kernel::LLVM.Function, push::LLVM.GlobalVariable, slots, dl)
    blocked = Set{LLVM.Value}()
    while true
        o = Origins()
        todo = LLVM.Value[]
        for bb in LLVM.blocks(kernel), inst in LLVM.instructions(bb)
            inst isa LLVM.LoadInst && inst.operands[1] == push || continue
            o[inst] = inst.value_type isa LLVM.StructType ? PushConstants() : ArgumentBuffer(0)
            push!(todo, inst)
        end
        while !isempty(todo)
            v = pop!(todo)
            for use in LLVM.uses(v)
                user = LLVM.user(use)
                user isa LLVM.Instruction && LLVM.parent(LLVM.parent(user)) == kernel || continue
                user in blocked && continue
                new = origin(user, o, slots, dl)
                new === nothing && continue
                get(o, user, nothing) == new && continue
                o[user] = new
                push!(todo, user)
            end
        end
        # A merge given an origin before all of its values had one: drop it and settle again.
        wrong = [inst for (inst, origin) in o if inst isa Union{LLVM.PHIInst,LLVM.SelectInst} &&
                                                 !confirmed(o, inst, origin)]
        isempty(wrong) && return o
        union!(blocked, wrong)
    end
end

# Whether the builtin variable `name` holds one value in every invocation of a subgroup.
uniformbuiltin(name::AbstractString) =
    name in ("__spirv_BuiltInWorkgroupId", "__spirv_BuiltInNumWorkgroups", "__spirv_BuiltInWorkgroupSize",
             "__spirv_BuiltInSubgroupSize", "__spirv_BuiltInSubgroupId", "__spirv_BuiltInNumSubgroups")

# Whether `inst` may give each invocation its own value whatever its operands are.
divergencesource(::LLVM.Instruction) = false
divergencesource(::Union{LLVM.AtomicRMWInst,LLVM.AtomicCmpXchgInst}) = true
function divergencesource(inst::LLVM.LoadInst)
    from = inst.operands[1]
    from isa LLVM.GlobalVariable && startswith(LLVM.name(from), "__spirv_BuiltIn") &&
        return !uniformbuiltin(LLVM.name(from))
    # Private memory holds whatever each invocation stored.
    return from isa LLVM.AllocaInst ||
           (from isa LLVM.GetElementPtrInst && from.operands[1] isa LLVM.AllocaInst)
end
function divergencesource(inst::LLVM.CallInst)
    callee = LLVM.called_operand(inst)
    return !(callee isa LLVM.Function && startswith(LLVM.name(callee), "llvm."))
end

"""
    divergentvalues(kernel) -> Set{LLVM.Value}

The values of `kernel` that may differ between the invocations of a subgroup, by data
dependence: a value is divergent if one of its operands is or it comes from a source of
divergence (an invocation id, private memory, an atomic, a call). Control dependence is
not followed, so a merge of uniform values under a divergent branch counts as uniform;
that only decides which read-only loads are marked, never whether marking one is safe.
"""
function divergentvalues(kernel::LLVM.Function)
    divergent = Set{LLVM.Value}()
    todo = LLVM.Value[]
    for bb in LLVM.blocks(kernel), inst in LLVM.instructions(bb)
        divergencesource(inst) || continue
        push!(divergent, inst)
        push!(todo, inst)
    end
    while !isempty(todo)
        v = pop!(todo)
        for use in LLVM.uses(v)
            user = LLVM.user(use)
            user isa LLVM.Instruction && !(user in divergent) || continue
            push!(divergent, user)
            push!(todo, user)
        end
    end
    return divergent
end

writtenaddress(inst::LLVM.StoreInst) = inst.operands[2]
writtenaddress(inst::Union{LLVM.AtomicRMWInst,LLVM.AtomicCmpXchgInst}) = inst.operands[1]
function writtenaddress(inst::LLVM.CallInst)
    callee = LLVM.called_operand(inst)
    callee isa LLVM.Function || return nothing
    name = LLVM.name(callee)
    any(prefix -> startswith(name, prefix), ("llvm.memcpy", "llvm.memmove", "llvm.memset")) || return nothing
    return LLVM.arguments(inst)[1]
end
writtenaddress(::LLVM.Instruction) = nothing

"""
    mark_readonly_loads!(mod, kernel, slots) -> Int

`!invariant.load` on every load in `kernel`, the entry wrapper with the kernel inlined,
from memory that a read-only argument in `slots` reaches, at an address that is not
divergent (`divergentvalues`). Throws when the kernel writes
such memory: the walk that called the argument read-only missed a store. Returns how many
loads it marked.
"""
function mark_readonly_loads!(mod::LLVM.Module, kernel::LLVM.Function, slots::Vector{ReadonlySlot})
    isempty(slots) && return 0
    push = get(LLVM.globals(mod), "__push_constants", nothing)
    push === nothing && return 0
    o = origins(kernel, push, slots, LLVM.datalayout(mod))
    divergent = divergentvalues(kernel)
    marked = 0
    for bb in LLVM.blocks(kernel), inst in LLVM.instructions(bb)
        address = writtenaddress(inst)
        if address !== nothing && get(o, address, nothing) isa ReadOnlyMemory
            error("$(LLVM.name(kernel)) writes memory of an argument its compile job marks read-only: $(inst)")
        end
        inst isa LLVM.LoadInst && get(o, inst.operands[1], nothing) isa ReadOnlyMemory || continue
        inst.operands[1] in divergent && continue
        LLVM.metadata(inst)[LLVM.MD_invariant_load] = LLVM.MDNode(LLVM.Metadata[])
        marked += 1
    end
    return marked
end
