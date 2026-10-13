# The constant RADV folds out of a PhysicalStorageBuffer address, and how the
# emitter keeps a negative one away from a load that ACO splits.
#
# RADV/ACO (Mesa 26.2.3 and main, reproduced outside Lava with glslang) loses
# 2^32 of an address in this sequence:
#
#  1. NIR's vectorizer merges adjacent sub-dword loads, `(B[g], B[g+1])` in
#     fp16, into one `16x2` load. RADV allows it at any element alignment.
#  2. `ac_nir_lower_global_access` sums every constant it reaches through
#     `iadd`, `imul`/`ishl` by a constant and no-wrap `u2u64`
#     (`try_extract_additions`) and folds the sum into the load's `base`. A
#     negative sum is stored as its low 32 bits, `0xfffffffe` for -2, with
#     the address pre-biased by -2^32.
#  3. ACO's `emit_load` splits the 2-byte-aligned `16x2` into two 16-bit
#     loads and advances the offset in a 32-bit `unsigned`: `0xfffffffe + 2`
#     wraps to 0, and the second half reads 4 GiB below the buffer. It reads
#     zeros, or the device is lost.
#
# Julia's 1-based indexing makes step 2 routine: `B[i]` is `p + 2(i - 1)`, and
# the `-1` survives whenever the `+1` of the index is out of NIR's reach: in a
# loop counter's phi, behind an Int32 index's sign extension, in an index read
# from memory. The wide staged GEMM lost every odd k-term this way.
#
# Stores are safe: ACO adds each piece's offset to the address in 64 bits.
# Loads that are not split are safe: their single piece takes the folded base
# correctly. So the emitter only acts on a load ACO may split whose address
# folds to a negative constant: it computes that address so no constant can be
# folded from it. Nothing here depends on the driver; on every other device it
# is the same address, at the cost of a few integer ops.

"""
    disjointor(v) -> Bool

An `or disjoint`: LLVM's `add` of operands with no bits in common, which is what
InstCombine turns `2i + 1` into. Emitted as `OpIAdd`, the same value by
definition at the same cost, because as `OpBitwiseOr` it hides the `+1` of a
1-based `B[2i + 1]` from RADV's constant folding while the `-1` of the indexing
stays visible (see the top of this file).
"""
disjointor(v::LLVM.Value) = v isa LLVM.OrInst && LLVM.API.LLVMGetIsDisjoint(v) != 0

"""
    foldedoffset(dl, v) -> Int64

The constant byte offset RADV folds out of the PSB address `v`: every constant
reachable through adds, constant multiplies and shifts and GEP indices, as
`ac_nir_lower_global_access` collects them. It stops where RADV stops (sign
extensions, phis, selects, loads); a zero extension counts as nothing, since a
negative constant under one never passes RADV's no-wrap check.

Where the two could disagree this errs negative: a subtraction contributes its
subtrahend's constant whenever that would lower the sum.
"""
function foldedoffset(dl::LLVM.DataLayout, v::LLVM.Value)
    if v isa LLVM.ConstantInt
        return convert(Int64, v)
    elseif v isa LLVM.GetElementPtrInst
        return foldedgep(dl, v)
    elseif v isa LLVM.BitCastInst || v isa LLVM.AddrSpaceCastInst ||
           v isa LLVM.IntToPtrInst || v isa LLVM.PtrToIntInst
        return foldedoffset(dl, LLVM.operands(v)[1])
    elseif v isa LLVM.AddInst || disjointor(v)
        a, b = LLVM.operands(v)
        return foldedoffset(dl, a) + foldedoffset(dl, b)
    elseif v isa LLVM.SubInst
        a, b = LLVM.operands(v)
        return foldedoffset(dl, a) - max(foldedoffset(dl, b), 0)
    elseif v isa LLVM.ShlInst
        a, b = LLVM.operands(v)
        return b isa LLVM.ConstantInt ? foldedoffset(dl, a) << convert(Int64, b) : 0
    elseif v isa LLVM.MulInst
        a, b = LLVM.operands(v)
        b isa LLVM.ConstantInt && return foldedoffset(dl, a) * convert(Int64, b)
        a isa LLVM.ConstantInt && return foldedoffset(dl, b) * convert(Int64, a)
        return 0
    else
        return 0
    end
end

"The constant part of a GEP's address: its base's, plus each index's times its stride."
function foldedgep(dl::LLVM.DataLayout, gep::LLVM.GetElementPtrInst)
    ops = LLVM.operands(gep)
    ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
    total = foldedoffset(dl, ops[1]) + Int64(compute_type_size(ty, dl)) * foldedoffset(dl, ops[2])
    for idx in ops[3:end]
        if ty isa LLVM.StructType
            field = convert(Int, idx::LLVM.ConstantInt)
            total += compute_struct_field_offset(ty, field; dl)
            ty = ty.elements[field + 1]
        else
            ty = ty.element_type
            total += Int64(compute_type_size(ty, dl)) * foldedoffset(dl, idx)
        end
    end
    return total
end

"""
    splitrisk(dl, ty, align, decomposed) -> Bool

Whether ACO may split a PSB load of `ty` declared `Aligned align`: below 4 bytes
of alignment, the load or the vector NIR merges it into is read in shorts or
bytes; above 16 bytes, in 16-byte pieces. `decomposed` loads are emitted as
byte or word pieces by the emitter itself.
"""
splitrisk(dl::LLVM.DataLayout, ty::LLVM.LLVMType, align::Integer, decomposed::Bool) =
    decomposed || align < 4 || compute_type_size(ty, dl) > 16

"""
    unfoldedaddress!(state, ptr, ptr_id, pointee) -> UInt32

`ptr_id`, the address of `ptr`, as a PSB pointer to `pointee` that RADV cannot
fold a negative constant out of. Returns `ptr_id` itself when the folded
offset is not negative.

The address is rebuilt as `opaque(root + c) + Σ terms` (see `AddressSum`): every
constant of the address goes into its root pointer through `opaqueadd!`, which
RADV does not fold through, and the terms are the address's own variable parts.
For a uniform root that is scalar work, and the terms are the products a
neighbouring load already computes. The rebuilt load can merge with no load but
one at the identical address, so ACO never splits it across zero.

Where the address does not come apart, the whole of it goes through
`opaqueadd!` instead: `opaque((ptr - c) + c)`, a few more vector ops.
"""
function unfoldedaddress!(state::SPIRVEmitterState, ptr::LLVM.Value, ptr_id::UInt32,
                          pointee::UInt32)
    dl = state.data_layout::LLVM.DataLayout
    c = foldedoffset(dl, ptr)
    c < 0 || return ptr_id
    ptrty = map_pointer_type!(state.type_ctx, pointee, SC.PhysicalStorageBuffer)
    sum = addresssum(dl, ptr)
    if sum !== nothing && rebuildable(state, sum)
        return rebuiltaddress!(state, sum, ptrty)
    end
    mod = state.mod
    u64 = emit_type_int!(mod, UInt32(64), UInt32(0))
    addr, stripped, result = fresh_id!(mod), fresh_id!(mod), fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpConvertPtrToU, u64, addr, ptr_id)
    encode_instruction!(mod.functions, Op.OpIAdd, u64, stripped, addr, emit_constant_u64!(mod, UInt64(-c)))
    encode_instruction!(mod.functions, Op.OpConvertUToPtr, ptrty, result, opaqueadd!(state, stripped, c))
    return result
end

"""
    opaqueadd!(state, x, c) -> UInt32

`x + c` for a 64-bit `x`, computed in two 32-bit halves with an explicit carry
and packed back: the same value, which RADV cannot fold `c` out of again.
"""
function opaqueadd!(state::SPIRVEmitterState, x::UInt32, c::Int64)
    mod = state.mod
    code = mod.functions
    u64 = emit_type_int!(mod, UInt32(64), UInt32(0))
    u32 = emit_type_int!(mod, UInt32(32), UInt32(0))
    pair = emit_type_vector!(mod, u32, UInt32(2))
    bits = reinterpret(UInt64, c)
    halves, lo, hi = fresh_id!(mod), fresh_id!(mod), fresh_id!(mod)
    encode_instruction!(code, Op.OpBitcast, pair, halves, x)
    encode_instruction!(code, Op.OpCompositeExtract, u32, lo, halves, UInt32(0))
    encode_instruction!(code, Op.OpCompositeExtract, u32, hi, halves, UInt32(1))
    newlo, wrapped, carry, hisum, newhi = (fresh_id!(mod) for _ in 1:5)
    encode_instruction!(code, Op.OpIAdd, u32, newlo, lo, emit_constant_u32!(mod, bits % UInt32))
    encode_instruction!(code, Op.OpULessThan, emit_type_bool!(mod), wrapped, newlo, lo)
    encode_instruction!(code, Op.OpSelect, u32, carry, wrapped,
                        emit_constant_u32!(mod, UInt32(1)), emit_constant_u32!(mod, UInt32(0)))
    encode_instruction!(code, Op.OpIAdd, u32, hisum, hi, emit_constant_u32!(mod, (bits >> 32) % UInt32))
    encode_instruction!(code, Op.OpIAdd, u32, newhi, hisum, carry)
    joined, packed = fresh_id!(mod), fresh_id!(mod)
    encode_instruction!(code, Op.OpCompositeConstruct, pair, joined, newlo, newhi)
    encode_instruction!(code, Op.OpBitcast, u64, packed, joined)
    return packed
end

"""
    AddressSum

A PSB address taken apart as `root + Σ multiplier · term + offset`: the one
pointer it starts from, the integer values added to it with their byte
multipliers, and every constant. Exact in 64-bit arithmetic. A term narrower
than 64 bits is a GEP index, which LLVM sign-extends, and is sign-extended when
rebuilt.
"""
mutable struct AddressSum
    root::Union{Nothing,LLVM.Value}
    const terms::Vector{Pair{LLVM.Value,Int64}}
    offset::Int64
end

"`ptr` as an `AddressSum`, or `nothing` when it does not start from exactly one pointer."
function addresssum(dl::LLVM.DataLayout, ptr::LLVM.Value)
    sum = AddressSum(nothing, Pair{LLVM.Value,Int64}[], 0)
    addterms!(sum, dl, ptr, 1) && sum.root !== nothing || return nothing
    return sum
end

"Add `m · v` to `sum`, taking `v` apart where it is a sum itself; `false` where it cannot be."
function addterms!(sum::AddressSum, dl::LLVM.DataLayout, v::LLVM.Value, m::Int64)
    v.value_type isa LLVM.VectorType && return false
    if v isa LLVM.ConstantInt
        sum.offset += m * convert(Int64, v)
    elseif v isa LLVM.GetElementPtrInst
        ops = LLVM.operands(v)
        ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(v))
        addterms!(sum, dl, ops[1], m) || return false
        addindex!(sum, dl, ops[2], m * Int64(compute_type_size(ty, dl))) || return false
        for idx in ops[3:end]
            if ty isa LLVM.StructType
                field = convert(Int, idx::LLVM.ConstantInt)
                sum.offset += m * compute_struct_field_offset(ty, field; dl)
                ty = ty.elements[field + 1]
            else
                ty = ty.element_type
                addindex!(sum, dl, idx, m * Int64(compute_type_size(ty, dl))) || return false
            end
        end
    elseif v isa LLVM.BitCastInst || v isa LLVM.AddrSpaceCastInst || v isa LLVM.PtrToIntInst
        return addterms!(sum, dl, LLVM.operands(v)[1], m)
    elseif v isa LLVM.IntToPtrInst
        x = LLVM.operands(v)[1]
        x.value_type == LLVM.Int64Type() || return false
        return addterms!(sum, dl, x, m)
    elseif v.value_type isa LLVM.PointerType
        m == 1 && sum.root === nothing || return false
        sum.root = v
    elseif v isa LLVM.AddInst || disjointor(v)
        a, b = LLVM.operands(v)
        return addterms!(sum, dl, a, m) && addterms!(sum, dl, b, m)
    elseif v isa LLVM.SubInst
        a, b = LLVM.operands(v)
        return addterms!(sum, dl, a, m) && addterms!(sum, dl, b, -m)
    elseif v isa LLVM.ShlInst && LLVM.operands(v)[2] isa LLVM.ConstantInt
        a, b = LLVM.operands(v)
        return addterms!(sum, dl, a, m << convert(Int64, b))
    elseif v isa LLVM.MulInst && any(o -> o isa LLVM.ConstantInt, LLVM.operands(v))
        a, b = LLVM.operands(v)
        a isa LLVM.ConstantInt && ((a, b) = (b, a))
        return addterms!(sum, dl, a, m * convert(Int64, b))
    else
        addterm!(sum, v, m)
    end
    return true
end

"A GEP index: taken apart when it is 64 bits wide, a sign-extended term otherwise."
function addindex!(sum::AddressSum, dl::LLVM.DataLayout, idx::LLVM.Value, m::Int64)
    (idx.value_type::LLVM.IntegerType).width == 64 && return addterms!(sum, dl, idx, m)
    if idx isa LLVM.ConstantInt
        sum.offset += m * convert(Int64, idx)
    else
        addterm!(sum, idx, m)
    end
    return true
end

function addterm!(sum::AddressSum, v::LLVM.Value, m::Int64)
    i = findfirst(t -> t.first == v, sum.terms)
    if i === nothing
        push!(sum.terms, v => m)
    else
        sum.terms[i] = v => sum.terms[i].second + m
    end
    return sum
end

"Whether `sum`'s root and terms were all emitted as what LLVM says they are."
function rebuildable(state::SPIRVEmitterState, sum::AddressSum)
    rootid = get(state.value_map, sum.root, nothing)
    is_psb_pointer(sum.root) && rootid !== nothing &&
        get(state.spirv_id_llvm_type, rootid, sum.root.value_type) isa LLVM.PointerType || return false
    return all(sum.terms) do (v, _)
        id = get(state.value_map, v, nothing)
        id !== nothing && v.value_type isa LLVM.IntegerType &&
            get(state.spirv_id_llvm_type, id, v.value_type) == v.value_type
    end
end

"Emit `opaque(root + offset) + Σ multiplier · term` as a PSB pointer of type `ptrty`."
function rebuiltaddress!(state::SPIRVEmitterState, sum::AddressSum, ptrty::UInt32)
    mod = state.mod
    code = mod.functions
    u64 = emit_type_int!(mod, UInt32(64), UInt32(0))
    root = fresh_id!(mod)
    encode_instruction!(code, Op.OpConvertPtrToU, u64, root, state.value_map[sum.root])
    acc = sum.offset == 0 ? root : opaqueadd!(state, root, sum.offset)
    for (v, m) in sum.terms
        m == 0 && continue
        term = state.value_map[v]
        if (v.value_type::LLVM.IntegerType).width < 64
            wide = fresh_id!(mod)
            encode_instruction!(code, Op.OpSConvert, u64, wide, term)
            term = wide
        end
        if m != 1
            scaled = fresh_id!(mod)
            encode_instruction!(code, Op.OpIMul, u64, scaled, term, emit_constant_u64!(mod, reinterpret(UInt64, m)))
            term = scaled
        end
        next = fresh_id!(mod)
        encode_instruction!(code, Op.OpIAdd, u64, next, acc, term)
        acc = next
    end
    result = fresh_id!(mod)
    encode_instruction!(code, Op.OpConvertUToPtr, ptrty, result, acc)
    return result
end
