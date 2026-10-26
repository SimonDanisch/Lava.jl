# LLVM pass: no 64-bit integer division reaches the driver.
#
# NVIDIA's compiler miscompiles code around its own 64-bit integer division. Two
# Mantle tests pin it, both measured on an RTX 4000 Ada (driver 595.99):
#
#   - `test_shared_index_division.jl`: a shared-memory store indexed by
#     `idx ÷ 96`, in a loop with a cooperative-matrix `muladd`, loses 2832 of
#     3072 stores. The same kernel dividing in Int32 or UInt32 keeps all 3072,
#     and so does a 64-bit multiply-high division (no division instruction).
#   - `test_int32_cartesian_miscompile.jl`: two dependent 64-bit divisions by
#     runtime sizes (`CartesianIndices` under an Int32 index) deliver the second
#     quotient where the first is used again. The Int32 forms are exact.
#
# The divisor 96 above is a compile-time constant, so the driver's own handling
# of constant divisors does not avoid it. So Lava divides in 64 bits nowhere:
#
#   - by a power of two: shifts;
#   - by another constant: the multiply-high sequence (Granlund and Montgomery;
#     the constants are `Base.MultiplicativeInverses`'), straight-line, built
#     from 32-bit partial products in 64-bit registers;
#   - by a runtime value: an always-inline helper that divides in 32 bits when
#     both magnitudes fit, which index arithmetic always does, and in 64 bits
#     otherwise.
#
# On every device: one path. Both forms are also cheaper than the 64-bit
# division every GPU emulates in software. Vector divisions are left alone;
# nothing in Lava's kernels produces them.

iswidedivision(inst::LLVM.Instruction) =
    LLVM.opcode(inst) in (LLVM.API.LLVMUDiv, LLVM.API.LLVMSDiv, LLVM.API.LLVMURem, LLVM.API.LLVMSRem) &&
    LLVM.value_type(inst) == LLVM.IntType(64)

issigned(op) = op == LLVM.API.LLVMSDiv || op == LLVM.API.LLVMSRem
isremainder(op) = op == LLVM.API.LLVMURem || op == LLVM.API.LLVMSRem

"""
    lower_wide_divisions!(mod::LLVM.Module; runtime = true) -> Bool

Replace every scalar 64-bit `udiv`/`sdiv`/`urem`/`srem`: by a constant with the
multiply-high sequence, and (with `runtime`) by anything else with a call to an
always-inline helper that divides in 32 bits when it can, then inline the
helpers. `runtime = false` touches straight-line code only, so it can run after
the CFG is structurized. Returns whether anything changed.
"""
function lower_wide_divisions!(mod::LLVM.Module; runtime::Bool = true)
    found = LLVM.Instruction[]
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        iswidedivision(inst) || continue
        bymagic(inst) || runtime || continue
        push!(found, inst)
    end
    isempty(found) && return false
    helpers = Dict{LLVM.API.LLVMOpcode, LLVM.Function}()
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        for inst in found
            op = LLVM.opcode(inst)
            a, d = LLVM.operands(inst)
            LLVM.position!(builder, LLVM.before(inst))
            q = if bymagic(inst)
                divide_by_constant!(builder, op, a, d)
            else
                fn = get!(() -> divhelper!(mod, op), helpers, op)
                LLVM.call!(builder, LLVM.function_type(fn), fn, [a, d])
            end
            LLVM.replace_uses!(inst, q)
            LLVM.erase!(inst)
        end
    end
    if !isempty(helpers)
        names = [LLVM.name(fn) for fn in values(helpers)]
        LLVM.run!(LLVM.AlwaysInlinerPass(), mod)
        # The inliner deletes a local `alwaysinline` function it emptied of
        # callers; whatever is left has none either.
        fns = LLVM.functions(mod)
        foreach(n -> haskey(fns, n) && LLVM.erase!(fns[n]), names)
    end
    return true
end

"""
Whether `inst` divides by a constant the multiply-high sequence covers: any but
zero (undefined; left to the driver) and, signed, `typemin(Int64)`.
"""
function bymagic(inst::LLVM.Instruction)
    d = LLVM.operands(inst)[2]
    d isa LLVM.ConstantInt || return false
    v = convert(Int64, d)
    return v != 0 && !(issigned(LLVM.opcode(inst)) && v == typemin(Int64))
end

k64(x::Integer) = LLVM.ConstantInt(LLVM.IntType(64), x % Int64)

"""
    mulhi!(builder, a, m::UInt64) -> LLVM.Value

The high 64 bits of the unsigned 128-bit product `a * m`, from four 32 x 32-bit
products, none of which overflows 64 bits.
"""
function mulhi!(builder::LLVM.IRBuilder, a::LLVM.Value, m::UInt64)
    lo32 = k64(0xffff_ffff)
    al, ah = LLVM.and!(builder, a, lo32), LLVM.lshr!(builder, a, k64(32))
    ml, mh = k64(m & 0xffff_ffff), k64(m >> 32)
    ll = LLVM.mul!(builder, al, ml)
    t = LLVM.add!(builder, LLVM.mul!(builder, ah, ml), LLVM.lshr!(builder, ll, k64(32)))
    w = LLVM.add!(builder, LLVM.and!(builder, t, lo32), LLVM.mul!(builder, al, mh))
    hh = LLVM.mul!(builder, ah, mh)
    return LLVM.add!(builder, LLVM.add!(builder, hh, LLVM.lshr!(builder, t, k64(32))),
                     LLVM.lshr!(builder, w, k64(32)))
end

"The high 64 bits of the signed product: the unsigned one, corrected for signs."
function smulhi!(builder::LLVM.IRBuilder, a::LLVM.Value, m::Int64)
    hi = mulhi!(builder, a, reinterpret(UInt64, m))
    # `(a >> 63) & m`: subtract `m` where `a` is negative.
    hi = LLVM.sub!(builder, hi, LLVM.and!(builder, LLVM.ashr!(builder, a, k64(63)), k64(m)))
    return m < 0 ? LLVM.sub!(builder, hi, a) : hi
end

"""
    divide_by_constant!(builder, op, a, d) -> LLVM.Value

`a op d` for a constant `d`: by a power of two with shifts, by anything else as
`Base.div(a, ::MultiplicativeInverse)` computes it; a remainder is
`a - (a ÷ d) * d`.

Powers of two never take the multiply-high: their multipliers have a
power-of-two half, InstCombine turns those partial products into
`(a >> 32) + ((a << 31) & 0x7fffffff80000000)`, and NVIDIA (RTX 4000 Ada,
driver 595.99) computes that sum's high word as `0x7fffffff & ~hi(a << 31)`:
`(j - 4) ÷ 4` in DNNKernels' Q4_K scales came out 0x3fffffff for `j = 5`.
"""
function divide_by_constant!(builder::LLVM.IRBuilder, op, a::LLVM.Value, d::LLVM.ConstantInt)
    v = convert(Int64, d)
    q = if issigned(op)
        if abs(v) == 1
            LLVM.mul!(builder, a, k64(v))
        elseif ispow2(abs(v))
            # Toward zero: a negative dividend is biased by `|d| - 1` first.
            k = trailing_zeros(abs(v))
            bias = LLVM.lshr!(builder, LLVM.ashr!(builder, a, k64(63)), k64(64 - k))
            x = LLVM.ashr!(builder, LLVM.add!(builder, a, bias), k64(k))
            v < 0 ? LLVM.sub!(builder, k64(0), x) : x
        else
            inv = Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64}(v)
            x = smulhi!(builder, a, inv.multiplier)
            x = inv.addmul == 1 ? LLVM.add!(builder, x, a) :
                inv.addmul == -1 ? LLVM.sub!(builder, x, a) : x
            # Rounds toward zero: one more where `x` is negative.
            LLVM.add!(builder, LLVM.ashr!(builder, x, k64(inv.shift)), LLVM.lshr!(builder, x, k64(63)))
        end
    else
        u = reinterpret(UInt64, v)
        if u == 1
            a
        elseif ispow2(u)
            LLVM.lshr!(builder, a, k64(trailing_zeros(u)))
        else
            inv = Base.MultiplicativeInverses.UnsignedMultiplicativeInverse{UInt64}(u)
            x = mulhi!(builder, a, inv.multiplier)
            if inv.add
                x = LLVM.add!(builder, LLVM.lshr!(builder, LLVM.sub!(builder, a, x), k64(1)), x)
            end
            LLVM.lshr!(builder, x, k64(inv.shift))
        end
    end
    return isremainder(op) ? LLVM.sub!(builder, a, LLVM.mul!(builder, q, d)) : q
end

"""
    divhelper!(mod, op) -> LLVM.Function

`i64 (i64 a, i64 b)` computing `a op b`, marked `alwaysinline`: in 32 bits when
both magnitudes fit, in 64 bits otherwise. A signed operation divides the
magnitudes and puts the sign back, so `typemin(Int32) ÷ -1` is not a 32-bit
overflow.
"""
function divhelper!(mod::LLVM.Module, op)
    i64, i32 = LLVM.IntType(64), LLVM.IntType(32)
    ft = LLVM.FunctionType(i64, [i64, i64])
    fn = LLVM.Function(mod, "lava.wide_division.$(Int(op))", ft)
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline"))
    entry, narrow, wide, join = (LLVM.BasicBlock(fn, n) for n in ("entry", "narrow", "wide", "join"))
    a, b = LLVM.parameters(fn)
    signed = issigned(op)
    LLVM.@dispose builder = LLVM.IRBuilder() begin
        LLVM.position!(builder, LLVM.at_end(entry))
        sa = signed ? LLVM.ashr!(builder, a, k64(63)) : nothing
        sb = signed ? LLVM.ashr!(builder, b, k64(63)) : nothing
        ua = signed ? LLVM.sub!(builder, LLVM.xor!(builder, a, sa), sa) : a
        ub = signed ? LLVM.sub!(builder, LLVM.xor!(builder, b, sb), sb) : b
        fits = LLVM.icmp!(builder, LLVM.API.LLVMIntEQ,
                          LLVM.lshr!(builder, LLVM.or!(builder, ua, ub), k64(32)), k64(0))
        LLVM.br!(builder, fits, narrow, wide)

        rem = isremainder(op)
        LLVM.position!(builder, LLVM.at_end(narrow))
        a32, b32 = LLVM.trunc!(builder, ua, i32), LLVM.trunc!(builder, ub, i32)
        r32 = LLVM.zext!(builder, rem ? LLVM.urem!(builder, a32, b32) : LLVM.udiv!(builder, a32, b32), i64)
        LLVM.br!(builder, join)

        LLVM.position!(builder, LLVM.at_end(wide))
        r64 = rem ? LLVM.urem!(builder, ua, ub) : LLVM.udiv!(builder, ua, ub)
        LLVM.br!(builder, join)

        LLVM.position!(builder, LLVM.at_end(join))
        r = LLVM.phi!(builder, i64)
        append!(LLVM.incoming(r), [(r32, narrow), (r64, wide)])
        # A quotient's sign is the operands' sign product; a remainder's is the
        # dividend's.
        s = !signed ? nothing : rem ? sa : LLVM.xor!(builder, sa, sb)
        LLVM.ret!(builder, signed ? LLVM.sub!(builder, LLVM.xor!(builder, r, s), s) : r)
    end
    return fn
end
