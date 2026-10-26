# LLVM pass: no integer of an odd width reaches the emitter.
#
# SPIR-V has 8, 16, 32 and 64-bit integers. LLVM has any width, and the datalayout's
# native widths (`native!` in target.jl) keep InstCombine to the native ones, but
# SimplifyCFG's switch-to-lookup ignores them: a switch over seven cases that only
# decides a flag becomes a bitmap of seven bits,
#
#     %cast = trunc i8 %x to i7
#     %down = lshr i7 -50, %cast
#     %bit  = and i7 %down, 1
#
# (Hikari's shader materials, a closure lobe tested against four of seven). The
# emitter mapped `i7` to its 8-bit container and every conversion to `OpUConvert`,
# so the `trunc` became an `OpUConvert` from an 8-bit integer to an 8-bit integer,
# which spirv-val refuses; and arithmetic on the container would have kept bits
# above the seventh.
#
# So every odd-width value is carried in the next native width with the bits above
# its own width zero: an operation that can set them (`trunc`, `add`, `sub`, `mul`,
# `shl`) masks its result, and one that reads the value as signed (`sext`, `ashr`,
# signed division and compares, `sitofp`) sign-extends it in the container first.
# An instruction this pass has no widening for throws, naming it.

"Whether `T` is an integer, or a vector of them, of a width SPIR-V has no type for."
isoddint(T::LLVM.IntegerType) = !(LLVM.width(T) in (1, 8, 16, 32, 64))
isoddint(T::LLVM.VectorType) = isoddint(LLVM.element_type(T))
isoddint(::LLVM.LLVMType) = false

"The native width `T` is carried in: what `spirv_int_width` maps it to."
widened(T::LLVM.IntegerType) = LLVM.IntType(Int(spirv_int_width(LLVM.width(T))))
widened(T::LLVM.VectorType) = LLVM.VectorType(widened(LLVM.element_type(T)), lanes(T))

"The integer `v` of type `T`, splatted for a vector type."
intconst(T::LLVM.IntegerType, v::UInt64) = LLVM.ConstantInt(T, v)
intconst(T::LLVM.VectorType, v::UInt64) = LLVM.ConstantVector([intconst(LLVM.element_type(T), v) for _ in 1:lanes(T)])
lanes(T::LLVM.VectorType) = Int(LLVM.API.LLVMGetVectorSize(T))

oddwidth(T::LLVM.IntegerType) = LLVM.width(T)
oddwidth(T::LLVM.VectorType) = oddwidth(LLVM.element_type(T))
lowbits(width::Integer) = (UInt64(1) << width) - UInt64(1)

"`x`, carried for the odd type `T`, with the bits above `T`'s width cleared."
masked!(builder, x, T) = LLVM.and!(builder, x, intconst(LLVM.value_type(x), lowbits(oddwidth(T))))

"`x`, carried for the odd type `T`, sign-extended from `T`'s width to its container's."
function signextended!(builder, x, T)
    shift = intconst(LLVM.value_type(x), UInt64(LLVM.width(scalar(LLVM.value_type(x))) - oddwidth(T)))
    return LLVM.ashr!(builder, LLVM.shl!(builder, x, shift), shift)
end
scalar(T::LLVM.VectorType) = LLVM.element_type(T)
scalar(T::LLVM.LLVMType) = T

"`x` converted to the integer type `T` by `ext!` or `trunc!`, or `x` itself when it is of `T`."
function resized!(builder, x, T, ext!)
    from, to = LLVM.width(scalar(LLVM.value_type(x))), LLVM.width(scalar(T))
    from == to && return x
    return from < to ? ext!(builder, x, T) : LLVM.trunc!(builder, x, T)
end

"""
    widen_odd_integers!(mod::LLVM.Module) -> Bool

Carry every integer of a width SPIR-V has no type for in the next native width, as
described above. Last in the pipeline: nothing after it narrows again. Returns
whether anything changed.
"""
function widen_odd_integers!(mod::LLVM.Module)
    changed = false
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        changed |= widen_odd_integers!(f)
    end
    return changed
end

function widen_odd_integers!(f::LLVM.Function)
    touches(inst) = isoddint(LLVM.value_type(inst)) || any(op -> isoddint(LLVM.value_type(op)), LLVM.operands(inst))
    any(touches, (inst for bb in LLVM.blocks(f) for inst in LLVM.instructions(bb))) || return false
    wide = Dict{LLVM.Value, LLVM.Value}()
    replaced = LLVM.Instruction[]
    phis = Tuple{LLVM.PHIInst, LLVM.PHIInst}[]
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        # The phis first, so a value can flow around a loop into one.
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            (inst isa LLVM.PHIInst && isoddint(LLVM.value_type(inst))) || continue
            LLVM.position!(builder, LLVM.before(inst))
            p = LLVM.phi!(builder, widened(LLVM.value_type(inst)))
            wide[inst] = p
            push!(phis, (inst, p))
            push!(replaced, inst)
        end
        # Then in reverse postorder, where every operand but a phi's is defined before its use.
        for bb in reverse_postorder(f), inst in collect(LLVM.instructions(bb))
            inst isa LLVM.PHIInst && continue
            touches(inst) || continue
            LLVM.position!(builder, LLVM.before(inst))
            new = widened!(builder, inst, wide)
            new === inst && continue
            if isoddint(LLVM.value_type(inst))
                wide[inst] = new
            else
                LLVM.replace_uses!(inst, new)
            end
            push!(replaced, inst)
        end
        for (old, p) in phis
            append!(LLVM.incoming(p), [(widevalue(wide, v), bb) for (v, bb) in LLVM.incoming(old)])
        end
    end
    for inst in replaced
        LLVM.replace_uses!(inst, LLVM.UndefValue(LLVM.value_type(inst)))
    end
    foreach(LLVM.erase!, replaced)
    return true
end

"`v`, an odd-width value or constant, in its container."
function widevalue(wide, v::LLVM.Value)
    haskey(wide, v) && return wide[v]
    T = LLVM.value_type(v)
    v isa LLVM.ConstantInt && return intconst(widened(T), UInt64(LLVM.API.LLVMConstIntGetZExtValue(v)))
    v isa LLVM.PoisonValue && return LLVM.PoisonValue(widened(T))
    v isa LLVM.UndefValue && return LLVM.UndefValue(widened(T))
    error("widen_odd_integers!: the $(T) value $(v) is neither an instruction this pass widened nor a constant")
end

"`v` as the widened instruction reads it: in its container when it is of an odd width, else itself."
operand(wide, v::LLVM.Value) = isoddint(LLVM.value_type(v)) ? widevalue(wide, v) : v

# The widened form of `inst`; `inst` itself when it was changed in place.
function widened!(builder, inst::LLVM.TruncInst, wide)
    x, T = operand(wide, only(LLVM.operands(inst))), LLVM.value_type(inst)
    isoddint(T) || return resized!(builder, x, T, LLVM.zext!)
    return masked!(builder, resized!(builder, x, widened(T), LLVM.zext!), T)
end

function widened!(builder, inst::LLVM.ZExtInst, wide)
    x, T = operand(wide, only(LLVM.operands(inst))), LLVM.value_type(inst)
    return resized!(builder, x, isoddint(T) ? widened(T) : T, LLVM.zext!)
end

function widened!(builder, inst::LLVM.SExtInst, wide)
    src = only(LLVM.operands(inst))
    x, T = operand(wide, src), LLVM.value_type(inst)
    isoddint(LLVM.value_type(src)) && (x = signextended!(builder, x, LLVM.value_type(src)))
    isoddint(T) || return resized!(builder, x, T, LLVM.sext!)
    return masked!(builder, resized!(builder, x, widened(T), LLVM.sext!), T)
end

"Operations that can set the bits above the width: the result is masked."
const MaskingOp = Union{LLVM.AddInst, LLVM.SubInst, LLVM.MulInst, LLVM.ShlInst}
"Operations whose result keeps those bits zero when the operands do."
const ClosedOp = Union{LLVM.AndInst, LLVM.OrInst, LLVM.XorInst, LLVM.LShrInst, LLVM.UDivInst, LLVM.URemInst}

widenedbinop!(builder, inst, wide) = LLVM.binop!(builder, LLVM.opcode(inst), (operand(wide, v) for v in LLVM.operands(inst))...)

widened!(builder, inst::MaskingOp, wide) = masked!(builder, widenedbinop!(builder, inst, wide), LLVM.value_type(inst))
widened!(builder, inst::ClosedOp, wide) = widenedbinop!(builder, inst, wide)

function widened!(builder, inst::Union{LLVM.SDivInst, LLVM.SRemInst}, wide)
    T = LLVM.value_type(inst)
    a, b = (signextended!(builder, operand(wide, v), T) for v in LLVM.operands(inst))
    return masked!(builder, LLVM.binop!(builder, LLVM.opcode(inst), a, b), T)
end

# The shift amount is a count, not a signed value.
function widened!(builder, inst::LLVM.AShrInst, wide)
    T = LLVM.value_type(inst)
    a, b = (operand(wide, v) for v in LLVM.operands(inst))
    return masked!(builder, LLVM.ashr!(builder, signextended!(builder, a, T), b), T)
end

const SIGNED_PREDICATES = (LLVM.API.LLVMIntSGT, LLVM.API.LLVMIntSGE, LLVM.API.LLVMIntSLT, LLVM.API.LLVMIntSLE)

function widened!(builder, inst::LLVM.ICmpInst, wide)
    p = LLVM.predicate(inst)
    T = LLVM.value_type(first(LLVM.operands(inst)))
    a, b = (operand(wide, v) for v in LLVM.operands(inst))
    if p in SIGNED_PREDICATES
        a, b = signextended!(builder, a, T), signextended!(builder, b, T)
    end
    return LLVM.icmp!(builder, p, a, b)
end

function widened!(builder, inst::LLVM.SelectInst, wide)
    c, a, b = LLVM.operands(inst)
    return LLVM.select!(builder, c, operand(wide, a), operand(wide, b))
end

widened!(builder, inst::LLVM.FreezeInst, wide) = LLVM.freeze!(builder, operand(wide, only(LLVM.operands(inst))))

widened!(builder, inst::LLVM.UIToFPInst, wide) =
    LLVM.uitofp!(builder, operand(wide, only(LLVM.operands(inst))), LLVM.value_type(inst))

function widened!(builder, inst::LLVM.SIToFPInst, wide)
    src = only(LLVM.operands(inst))
    return LLVM.sitofp!(builder, signextended!(builder, widevalue(wide, src), LLVM.value_type(src)), LLVM.value_type(inst))
end

# A float out of the integer's range is poison, so the container's conversion, masked, is one.
widened!(builder, inst::LLVM.FPToUIInst, wide) =
    masked!(builder, LLVM.fptoui!(builder, only(LLVM.operands(inst)), widened(LLVM.value_type(inst))), LLVM.value_type(inst))
widened!(builder, inst::LLVM.FPToSIInst, wide) =
    masked!(builder, LLVM.fptosi!(builder, only(LLVM.operands(inst)), widened(LLVM.value_type(inst))), LLVM.value_type(inst))

# The condition and the case values, in place: a switch's cases are constants of its condition's type.
function widened!(builder, inst::LLVM.SwitchInst, wide)
    for (i, v) in enumerate(LLVM.operands(inst))
        isoddint(LLVM.value_type(v)) || continue
        LLVM.API.LLVMSetOperand(inst, i - 1, widevalue(wide, v))
    end
    return inst
end

widened!(builder, inst::LLVM.Instruction, wide) =
    error("widen_odd_integers!: no widening for an odd-width integer in $(inst)")
