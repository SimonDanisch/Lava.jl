# SPIR-V → WGSL for modules from Lava's WebGPU target.
#
# The input is what `TargetFeatures(webgpu = true)` produces: logical addressing,
# one entry point, every helper inlined, memory as an array of heaps and arguments
# as the `args` binding. The output is the WGSL a WebGPU device compiles, for a
# device with a given number of heaps: WGSL has no arrays of bindings, so each heap
# is a binding of its own, reached through functions that switch on the heap
# (`HEAP` below).
#
# Values are assigned where SPIR-V defines them. The module printer turns single
# assignments with lexically scoped uses into `let` bindings. Phi variables and
# values that cross WGSL scopes keep function-scope storage. Pointer results
# print as the reference expression they stand for (`heap.m0[i]`) at each use.
#
# Integers: every 32-bit integer type is `u32`; signed operations bitcast through
# `i32`. 64-bit integers, which WGSL lacks, are `vec2<u32>` (low, high) with the
# operations in `WGSL_I64_PRELUDE`.
#
# Control flow: SPIR-V's structured constructs (selection and loop merges) map to
# `if`, `switch` and `loop { continuing { break if } }`. A branch to an enclosing
# loop's merge or continue target becomes `break` or `continue`. Phi nodes become
# assignments to the phi's variable on each incoming edge.

"""Printer state for one module."""
mutable struct WGSLPrinter
    m::SpvModule
    out::IOBuffer
    indent::Int
    structs::Dict{UInt32, String}       # struct type id -> WGSL struct name
    builtins::Dict{UInt32, String}      # Input variable id -> entry parameter name
    refs::Dict{UInt32, String}          # pointer result id -> reference expression
    blocks::Dict{UInt32, SpvBlock}
    uses_i64::Bool
    uses_f16::Bool
    uses_subgroups::Bool
    # Whether an infinity or NaN constant is printed, through `NONFINITE_PRELUDE`.
    uses_nonfinite::Bool
    assigned::Set{UInt32}               # results that got a variable
    roots::Dict{UInt32, UInt32}         # pointer id -> the variable it points into
    atomics::Set{UInt32}                # variables an atomic touches: declared atomic<…>
    heaps::Int                          # the device's heaps, the bindings printed
    heaparray::UInt32                   # the array-of-heaps variable, 0 if the module has none
    heapwords::Dict{UInt32, Tuple{String, String}}   # pointer id -> (heap, word) expressions
    heapops::Set{String}                # the heap functions the body calls
    derived::Vector{String}             # entry-point lines defining derived builtins
    uniformloads::Set{UInt32}           # heap loads read through `workgroupUniformLoad`
    outputs::Vector{String}             # fields of the graphics entry point's result
    flip_position::UInt32              # vertex position variable when its Y must be mirrored
end

struct WGSLError <: Exception
    msg::String
end
Base.showerror(io::IO, e::WGSLError) = print(io, "WGSL printer: ", e.msg)
refuse(msg) = throw(WGSLError(msg))

executionmodel(p::WGSLPrinter) = operand(p.m.entry, 1)

line!(p::WGSLPrinter, s::AbstractString) = (print(p.out, "    "^p.indent, s, "\n"); nothing)

function indented(f, p::WGSLPrinter)
    p.indent += 1
    f()
    p.indent -= 1
    return nothing
end

# ── Types ───────────────────────────────────────────────────────────────────

spvtype(p::WGSLPrinter, id::UInt32) = p.m.types[id]

wgsl(p::WGSLPrinter, id::UInt32) = wgsl(p, spvtype(p, id), id)
wgsl(::WGSLPrinter, ::SpvVoid, id) = refuse("void has no WGSL value type")
wgsl(::WGSLPrinter, ::SpvBool, id) = "bool"
function wgsl(p::WGSLPrinter, t::SpvInt, id)
    t.width in (8, 16, 32) && return "u32"     # narrow ones masked to their width
    t.width == 64 && (p.uses_i64 = true; return "vec2<u32>")
    refuse("$(t.width)-bit integers have no WGSL type")
end

"""The width of a narrow (8- or 16-bit) integer scalar or vector type, else 0."""
function narrow(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, scalartype(p, type))
    return t isa SpvInt && t.width < 32 ? t.width : 0
end

"""`expr` cut to the narrow width of `type`, as narrow integers are kept in a u32."""
function cut(p::WGSLPrinter, type::UInt32, expr)
    w = narrow(p, type)
    w == 0 && return expr
    return "(($expr) & $(UInt32(1) << w - 1)u)"
end
function wgsl(p::WGSLPrinter, t::SpvFloat, id)
    t.width == 32 && return "f32"
    t.width == 16 && (p.uses_f16 = true; return "f16")
    refuse("$(t.width)-bit floats have no WGSL type")
end
function wgsl(p::WGSLPrinter, t::SpvVector, id)
    is64(p, t.elem) && refuse("a vector of 64-bit integers has no WGSL type")
    return "vec$(t.n)<$(wgsl(p, t.elem))>"
end
wgsl(p::WGSLPrinter, t::SpvArray, id) = "array<$(wgsl(p, t.elem)), $(t.len)>"
wgsl(p::WGSLPrinter, t::SpvRuntimeArray, id) = "array<$(wgsl(p, t.elem))>"
wgsl(p::WGSLPrinter, t::SpvStruct, id) = p.structs[id]
wgsl(p::WGSLPrinter, t::SpvPointer, id) = refuse("a pointer is not a WGSL value")

is64(p::WGSLPrinter, type::UInt32) = (t = spvtype(p, type); t isa SpvInt && t.width == 64)
isint32(p::WGSLPrinter, type::UInt32) = (t = spvtype(p, type); t isa SpvInt && t.width == 32)

"""The scalar type of a scalar or vector type."""
function scalartype(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, type)
    return t isa SpvVector ? t.elem : type
end

"""Declare the module's struct types, members `m0, m1, …`, in dependency order."""
function declare_structs!(p::WGSLPrinter)
    order = UInt32[]
    visit(id) = begin
        t = spvtype(p, id)
        if t isa SpvStruct
            haskey(p.structs, id) && return
            foreach(visit, t.members)
            p.structs[id] = "S$(id)"
            push!(order, id)
        elseif t isa Union{SpvArray, SpvRuntimeArray, SpvVector}
            visit(t.elem)
        elseif t isa SpvPointer
            visit(t.pointee)
        end
    end
    foreach(visit, sort!(collect(keys(p.m.types))))
    for id in order
        t = spvtype(p, id)::SpvStruct
        line!(p, "struct $(p.structs[id]) {")
        indented(p) do
            for (k, mem) in enumerate(t.members)
                line!(p, "m$(k - 1): $(wgsl(p, mem)),")
            end
        end
        line!(p, "}")
    end
end

# ── Values ──────────────────────────────────────────────────────────────────

valname(id::UInt32) = "v$(id)"

"""The WGSL expression for value `id`: a constant, a builtin, or its variable."""
function value(p::WGSLPrinter, id::UInt32)
    haskey(p.m.constants, id) && return constant(p, p.m.constants[id])
    haskey(p.refs, id) && return p.refs[id]
    inst = get(p.m.results, id, nothing)
    inst !== nothing && inst.opcode == 1 && return zerovalue(p, operand(inst, 1))  # OpUndef
    return valname(id)
end

function zerovalue(p::WGSLPrinter, type::UInt32)
    return "$(wgsl(p, type))()"
end

function constant(p::WGSLPrinter, c::SpvScalar)
    t = spvtype(p, c.type)
    if t isa SpvBool
        return c.bits == 0 ? "false" : "true"
    elseif t isa SpvInt && t.width <= 32
        return "$(UInt32(c.bits & (UInt64(1) << t.width - 1)))u"
    elseif t isa SpvInt && t.width == 64
        p.uses_i64 = true
        return "vec2<u32>($(UInt32(c.bits & 0xffffffff))u, $(UInt32(c.bits >> 32))u)"
    elseif t isa SpvFloat && t.width == 32
        x = reinterpret(Float32, UInt32(c.bits))
        isfinite(x) || (p.uses_nonfinite = true; return "lava_nonfinite_f32($(UInt32(c.bits))u)")
        return floatliteral(Float64(x), "f")
    elseif t isa SpvFloat && t.width == 16
        p.uses_f16 = true
        x = reinterpret(Float16, UInt16(c.bits & 0xffff))
        isfinite(x) || (p.uses_nonfinite = true; return "lava_nonfinite_f16($(UInt32(c.bits & 0xffff))u)")
        return floatliteral(Float64(x), "h")
    end
    refuse("constant of type $t")
end
constant(p::WGSLPrinter, c::SpvComposite) =
    "$(wgsl(p, c.type))($(join((value(p, x) for x in c.parts), ", ")))"
constant(p::WGSLPrinter, c::SpvNull) = zerovalue(p, c.type)

"""A WGSL float literal that reads back as `x`: shortest round-trip digits."""
function floatliteral(x::Float64, suffix::String)
    s = string(x)
    s = replace(s, "e" => "e")
    occursin('.', s) || occursin('e', s) || (s *= ".0")
    return s * suffix
end

# ── Statement helpers ──────────────────────────────────────────────────────

"""Assign `expr` to result `id`'s variable."""
function assign!(p::WGSLPrinter, id::UInt32, expr::AbstractString)
    push!(p.assigned, id)
    line!(p, "$(valname(id)) = $(expr);")
end

"""`expr`, of 32-bit integer type, as `i32`."""
signed(expr) = "bitcast<i32>($(expr))"
"""`expr`, of `i32` type, as the `u32` all integers are."""
unsigned(expr) = "bitcast<u32>($(expr))"

function lanes(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, type)
    return t isa SpvVector ? t.n : 1
end

"""`T` for `i32` reinterpretation of a `u32` scalar or vector type."""
function signedtype(p::WGSLPrinter, type::UInt32)
    n = lanes(p, type)
    return n == 1 ? "i32" : "vec$(n)<i32>"
end
function signed(p::WGSLPrinter, type::UInt32, expr)
    w = narrow(p, type)
    w == 0 && return "bitcast<$(signedtype(p, type))>($(expr))"
    s = 32 - w        # sign-extend from the narrow width
    return "(bitcast<$(signedtype(p, type))>(($expr) << $(s)u) >> $(s)u)"
end
unsigned(p::WGSLPrinter, type::UInt32, expr) =
    lanes(p, type) == 1 ? "bitcast<u32>($(expr))" : "bitcast<vec$(lanes(p, type))<u32>>($(expr))"

# ── Instructions ────────────────────────────────────────────────────────────

const WGSL_BINARY = Dict{UInt16, String}(
    128 => "+", 129 => "+", 130 => "-", 131 => "-", 132 => "*", 133 => "*",
    134 => "/", 136 => "/", 137 => "%", 140 => "%",
    197 => "|", 198 => "^", 199 => "&",
    164 => "==", 165 => "!=", 166 => "||", 167 => "&&",
    170 => "==", 171 => "!=", 172 => ">", 174 => ">=", 176 => "<", 178 => "<=",
    180 => "==", 184 => "<", 186 => ">", 188 => "<=", 190 => ">=",
    196 => "<<", 194 => ">>",
    # NaN-aware comparisons, spelled out in `binary!`
    181 => "", 182 => "", 183 => "", 185 => "", 187 => "", 189 => "", 191 => "")

# Signed integer operations: operands as i32, a u32 (or bool) result.
const WGSL_SIGNED = Dict{UInt16, String}(
    135 => "/", 138 => "%", 173 => ">", 175 => ">=", 177 => "<", 179 => "<=", 195 => ">>")

const WGSL_COMPARE = Set{UInt16}([164, 165, 170:191...])

# 64-bit integer operations: the prelude function of each.
const WGSL_I64_BINARY = Dict{UInt16, String}(
    128 => "i64_add", 130 => "i64_sub", 132 => "i64_mul",
    134 => "u64_div", 135 => "i64_div", 137 => "u64_rem", 138 => "i64_rem",
    197 => "i64_or", 198 => "i64_xor", 199 => "i64_and",
    170 => "i64_eq", 171 => "i64_ne", 172 => "u64_gt", 173 => "i64_gt", 174 => "u64_ge",
    175 => "i64_ge", 176 => "u64_lt", 177 => "i64_lt", 178 => "u64_le", 179 => "i64_le",
    196 => "i64_shl", 194 => "u64_shr", 195 => "i64_shr")

# GLSL.std.450: the WGSL builtin, and whether its integer operands are signed.
const WGSL_GLSL = Dict{UInt32, Tuple{String, Bool}}(
    1 => ("round", false), 2 => ("round", false), 3 => ("trunc", false), 4 => ("abs", false),
    5 => ("abs", true), 6 => ("sign", false), 7 => ("sign", true), 8 => ("floor", false),
    9 => ("ceil", false), 10 => ("fract", false), 11 => ("radians", false), 12 => ("degrees", false),
    13 => ("sin", false), 14 => ("cos", false), 15 => ("tan", false), 16 => ("asin", false),
    17 => ("acos", false), 18 => ("atan", false), 19 => ("sinh", false), 20 => ("cosh", false),
    21 => ("tanh", false), 22 => ("asinh", false), 23 => ("acosh", false), 24 => ("atanh", false),
    25 => ("atan2", false), 26 => ("pow", false), 27 => ("exp", false), 28 => ("log", false),
    29 => ("exp2", false), 30 => ("log2", false), 31 => ("sqrt", false), 32 => ("inverseSqrt", false),
    37 => ("min", false), 38 => ("min", false), 39 => ("min", true), 40 => ("max", false),
    41 => ("max", false), 42 => ("max", true), 43 => ("clamp", false), 44 => ("clamp", false),
    45 => ("clamp", true), 46 => ("mix", false), 48 => ("step", false), 49 => ("smoothstep", false),
    50 => ("fma", false), 53 => ("ldexp", false), 58 => ("pack2x16float", false),
    62 => ("unpack2x16float", false), 66 => ("length", false), 67 => ("distance", false),
    68 => ("cross", false), 69 => ("normalize", false), 71 => ("reflect", false),
    73 => ("firstTrailingBit", false), 74 => ("firstLeadingBit", true), 75 => ("firstLeadingBit", false),
    79 => ("min", false), 80 => ("max", false), 81 => ("clamp", false))

function emit!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    if op == 61                                       # OpLoad
        id, ptr = operand(inst, 2), operand(inst, 3)
        if haskey(p.builtins, ptr)
            p.refs[id] = p.builtins[ptr]
        elseif spvtype(p, operand(inst, 1)) isa SpvSampledImage
            p.refs[id] = reference(p, ptr)
        elseif id in p.uniformloads
            line!(p, "if (local_invocation_index == 0u) { lava_uload$id = $(heapcall(p, "load", ptr)); }")
            assign!(p, id, "workgroupUniformLoad(&lava_uload$id)")
        elseif haskey(p.heapwords, ptr)
            assign!(p, id, heapcall(p, "load", ptr))
        else
            assign!(p, id, isatomic(p, ptr) ? "atomicLoad(&$(reference(p, ptr)))" : reference(p, ptr))
        end
    elseif op == 62                                   # OpStore
        ptr, v = operand(inst, 1), value(p, operand(inst, 2))
        ptr == p.flip_position && (v = "vec4<f32>(($v).x, -($v).y, ($v).z, ($v).w)")
        if haskey(p.heapwords, ptr)
            if executionmodel(p) == ExecModel.GLCompute
                line!(p, heapcall(p, "store", ptr, v) * ";")
            elseif p.heapwords[ptr] == ("0u", "0u") && v == "1u"
                rasterthrow!(p)
            else
                refuse("graphics stages require read-only heaps")
            end
        else
            line!(p, isatomic(p, ptr) ? "atomicStore(&$(reference(p, ptr)), $v);" : "$(reference(p, ptr)) = $v;")
        end
    elseif op in (65, 66)                             # OpAccessChain, OpInBoundsAccessChain
        id, base = operand(inst, 2), operand(inst, 3)
        if base == p.heaparray
            heapword!(p, inst)
        else
            p.refs[id] = accesschain(p, base, inst.operands[4:end])
        end
        p.roots[id] = get(p.roots, base, base)
    elseif 227 <= op <= 242
        atomic!(p, inst)
    elseif op == 124 && spvtype(p, operand(inst, 1)) isa SpvPointer
        pointercast!(p, inst)
    elseif op == 83                                   # OpCopyObject
        id, x = operand(inst, 2), operand(inst, 3)
        haskey(p.refs, x) ? (p.refs[id] = p.refs[x]) : assign!(p, id, value(p, x))
    elseif op == 1                                    # OpUndef: printed as a zero value
    elseif haskey(WGSL_BINARY, op) || haskey(WGSL_SIGNED, op)
        binary!(p, inst)
    elseif op in (126, 127, 200, 168)                 # SNegate, FNegate, Not, LogicalNot
        unary!(p, inst)
    elseif op in (204, 205)                           # OpBitReverse, OpBitCount
        builtin = op == 204 ? "reverseBits" : "countOneBits"
        assign!(p, operand(inst, 2), "$builtin($(value(p, operand(inst, 3))))")
    elseif op in 109:124 || op == 124
        conversion!(p, inst)
    elseif op == 169                                  # OpSelect
        type, id = operand(inst, 1), operand(inst, 2)
        c, a, b = value(p, operand(inst, 3)), value(p, operand(inst, 4)), value(p, operand(inst, 5))
        assign!(p, id, "select($b, $a, $c)")
    elseif op in (156, 157)                           # OpIsNan, OpIsInf
        src = operand(inst, 3)
        type = p.m.typeof[src]
        x = value(p, src)
        n = lanes(p, type)
        if (spvtype(p, scalartype(p, type))::SpvFloat).width == 16
            # Widening a half preserves whether it is finite, infinite or NaN.
            x = n == 1 ? "f32($x)" : "vec$(n)<f32>($x)"
        end
        bits = n == 1 ? "bitcast<u32>($x)" : "bitcast<vec$(n)<u32>>($x)"
        cmp = op == 156 ? ">" : "=="
        assign!(p, operand(inst, 2), "(($bits & 0x7fffffffu) $cmp 0x7f800000u)")
    elseif op in (154, 155)                           # OpAny, OpAll
        assign!(p, operand(inst, 2), "$(op == 154 ? "any" : "all")($(value(p, operand(inst, 3))))")
    elseif op == 81                                   # OpCompositeExtract
        assign!(p, operand(inst, 2), extract(p, operand(inst, 3), inst.operands[4:end]))
    elseif op == 80                                   # OpCompositeConstruct
        parts = join((value(p, x) for x in inst.operands[3:end]), ", ")
        assign!(p, operand(inst, 2), "$(wgsl(p, operand(inst, 1)))($parts)")
    elseif op == 82                                   # OpCompositeInsert
        id = operand(inst, 2)
        assign!(p, id, value(p, operand(inst, 4)))
        line!(p, "$(componentpath(p, valname(id), operand(inst, 1), inst.operands[5:end])) = $(value(p, operand(inst, 3)));")
    elseif op == 79                                   # OpVectorShuffle
        shuffle!(p, inst)
    elseif op == 77                                   # OpVectorExtractDynamic
        assign!(p, operand(inst, 2), "$(value(p, operand(inst, 3)))[$(value(p, operand(inst, 4)))]")
    elseif op == 87                                   # OpImageSampleImplicitLod
        texture = reference(p, operand(inst, 3))
        coord = value(p, operand(inst, 4))
        assign!(p, operand(inst, 2), "textureSample($texture, $(texture)_sampler, $coord)")
    elseif op == 12                                   # OpExtInst
        extinst!(p, inst)
    elseif op in (207, 208, 209)                       # OpDPdx, OpDPdy, OpFwidth
        builtin = op == 207 ? "dpdx" : op == 208 ? "dpdy" : "fwidth"
        assign!(p, operand(inst, 2), "$builtin($(value(p, operand(inst, 3))))")
    elseif op == 5380                                 # OpDemoteToHelperInvocation
        line!(p, "discard;")
    elseif op == 224                                  # OpControlBarrier
        barrier!(p, inst)
    elseif 333 <= op <= 366 || op == 4431
        subgroup!(p, inst)
    elseif op == 4450                                 # OpSDotKHR, packed 4x8
        a, b = value(p, operand(inst, 3)), value(p, operand(inst, 4))
        assign!(p, operand(inst, 2), unsigned("dot4I8Packed($a, $b)"))
    elseif op == 4451
        a, b = value(p, operand(inst, 3)), value(p, operand(inst, 4))
        assign!(p, operand(inst, 2), "dot4U8Packed($a, $b)")
    else
        refuse("SPIR-V opcode $(op) is not supported yet")
    end
    return nothing
end

"""A graphics invocation cannot publish a failure through a writable heap."""
function rasterthrow!(p::WGSLPrinter)
    if executionmodel(p) == ExecModel.Fragment
        line!(p, "discard;")
    else
        for g in p.m.globals
            operand(g, 3) == 3 || continue
            id = operand(g, 2)
            bi = decoration(p.m, id, 11)
            (bi === nothing || bi[1] != BuiltIn.Position) && continue
            line!(p, "$(p.refs[id]) = vec4<f32>(0.0f, 0.0f, 2.0f, 1.0f);")
            break
        end
    end
    line!(p, isempty(p.outputs) ? "return;" : "return lava_out;")
    return nothing
end

"""The reference expression a pointer stands for."""
function reference(p::WGSLPrinter, ptr::UInt32)
    haskey(p.refs, ptr) && return p.refs[ptr]
    refuse("pointer %$(ptr) has no reference expression (a pointer selected or passed by value)")
end

"""The type a pointer points to."""
pointee(p::WGSLPrinter, ptr::UInt32) = (spvtype(p, p.m.typeof[ptr])::SpvPointer).pointee

function accesschain(p::WGSLPrinter, base::UInt32, indices)
    expr = reference(p, base)
    type = pointee(p, base)
    for idx in indices
        expr, type = step(p, expr, type, idx)
    end
    return expr
end

"""One access-chain step from `expr` of type `type` by index `idx` (a value id)."""
function step(p::WGSLPrinter, expr, type::UInt32, idx::UInt32)
    t = spvtype(p, type)
    if t isa SpvStruct
        k = Int((p.m.constants[idx]::SpvScalar).bits)
        return "$(expr).m$(k)", t.members[k + 1]
    elseif t isa Union{SpvArray, SpvRuntimeArray, SpvVector}
        return "$(expr)[$(index(p, idx))]", t.elem
    end
    refuse("access chain into $t")
end

"""An index operand as a WGSL `u32` (64-bit indices are truncated, as Vulkan does)."""
function index(p::WGSLPrinter, idx::UInt32)
    is64(p, p.m.typeof[idx]) && return "$(value(p, idx)).x"
    return value(p, idx)
end

"""Literal-indexed component of a composite value."""
function extract(p::WGSLPrinter, composite::UInt32, literals)
    return componentpath(p, value(p, composite), p.m.typeof[composite], literals)
end

function componentpath(p::WGSLPrinter, expr, type::UInt32, literals)
    for k in literals
        t = spvtype(p, type)
        if t isa SpvStruct
            expr, type = "$(expr).m$(k)", t.members[k + 1]
        elseif t isa Union{SpvArray, SpvVector}
            expr, type = "$(expr)[$(k)]", t.elem
        else
            refuse("composite extract from $t")
        end
    end
    return expr
end

function binary!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    type, id = operand(inst, 1), operand(inst, 2)
    x, y = operand(inst, 3), operand(inst, 4)
    a, b = value(p, x), value(p, y)
    if is64(p, p.m.typeof[x])
        fn = get(WGSL_I64_BINARY, op, nothing)
        fn === nothing && refuse("64-bit integer opcode $op")
        # A shift amount is a scalar of its own width.
        if op in (194, 195, 196)
            b = is64(p, p.m.typeof[y]) ? "$(b).x" : b
        end
        p.uses_i64 = true
        return assign!(p, id, "$fn($a, $b)")
    end
    if op in (194, 195, 196) && is64(p, p.m.typeof[y])
        b = "$(b).x"
    end
    if haskey(WGSL_SIGNED, op)
        sym = WGSL_SIGNED[op]
        xt = p.m.typeof[x]
        bexpr = op == 195 ? b : signed(p, xt, b)        # an arithmetic shift's amount stays u32
        expr = "($(signed(p, xt, a)) $sym $(bexpr))"
        return assign!(p, id, op in WGSL_COMPARE ? expr : cut(p, type, unsigned(p, type, expr)))
    end
    sym = WGSL_BINARY[op]
    if op in (181, 183, 185, 187, 189, 191)           # unordered: true when either is NaN
        neg = Dict(181 => "!=", 183 => "!=", 185 => ">=", 187 => "<=", 189 => ">", 191 => "<")[op]
        expr = op == 181 ? "!(($a < $b) || ($a > $b))" : op == 183 ? "($a != $b)" : "!($a $neg $b)"
        return assign!(p, id, expr)
    elseif op == 182                                   # FOrdNotEqual
        return assign!(p, id, "(($a < $b) || ($a > $b))")
    end
    assign!(p, id, op in WGSL_COMPARE ? "($a $sym $b)" : cut(p, type, "($a $sym $b)"))
end

function unary!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    type, id, x = operand(inst, 1), operand(inst, 2), operand(inst, 3)
    a = value(p, x)
    if is64(p, type)
        p.uses_i64 = true
        op == 126 && return assign!(p, id, "i64_neg($a)")
        op == 200 && return assign!(p, id, "~$a")
        refuse("64-bit unary opcode $op")
    end
    op == 126 && return assign!(p, id, cut(p, type, unsigned(p, type, "-$(signed(p, type, a))")))
    op == 127 && return assign!(p, id, "-$a")
    op == 200 && return assign!(p, id, cut(p, type, "~$a"))
    assign!(p, id, "!$a")
end

function conversion!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    type, id, x = operand(inst, 1), operand(inst, 2), operand(inst, 3)
    from = p.m.typeof[x]
    a = value(p, x)
    to64, from64 = is64(p, type), is64(p, from)
    if op == 124                                       # OpBitcast
        (wgsl(p, type) == wgsl(p, from)) && return assign!(p, id, a)
        to64 || from64 && refuse("bitcast between 64-bit and $(wgsl(p, type))")
        # A 16-bit integer is a masked u32, and WGSL bitcasts f16 only as half of a
        # vec2<f16>: the other half is the integer's zero upper bits.
        if narrow(p, from) == 16 || narrow(p, type) == 16
            lanes(p, type) == 1 || refuse("bitcast between vectors of 16-bit integers and $(wgsl(p, type))")
            narrow(p, from) == 16 && return assign!(p, id, "bitcast<vec2<f16>>($a).x")
            return assign!(p, id, "bitcast<u32>(vec2<f16>($a, 0.0h))")
        end
        return assign!(p, id, "bitcast<$(wgsl(p, type))>($a)")
    elseif op == 113 || op == 114                      # OpUConvert, OpSConvert
        # A narrow source is kept masked: zero-extending is free, sign-extending is not.
        wide = op == 114 && narrow(p, from) != 0 ? unsigned(p, from, signed(p, from, a)) : a
        if to64 && !from64
            p.uses_i64 = true
            return assign!(p, id, op == 113 ? "vec2<u32>($wide, 0u)" : "i64_from_i32($wide)")
        elseif from64 && !to64
            return assign!(p, id, cut(p, type, "$(a).x"))
        end
        return assign!(p, id, cut(p, type, wide))
    elseif op == 112                                   # OpConvertUToF
        from64 && return assign!(p, id, "$(wgsl(p, type))(f32($a.y) * 4294967296.0f + f32($a.x))")
        return assign!(p, id, "$(wgsl(p, type))($a)")
    elseif op == 111                                   # OpConvertSToF
        from64 && (p.uses_i64 = true; return assign!(p, id, "$(wgsl(p, type))(i64_to_f32($a))"))
        return assign!(p, id, "$(wgsl(p, type))($(signed(p, from, a)))")
    elseif op == 109                                   # OpConvertFToU
        to64 && (p.uses_i64 = true; return assign!(p, id, "u64_from_f32(f32($a))"))
        # Cut to a narrow width: a narrow integer is a masked u32.
        return assign!(p, id, cut(p, type, "$(wgsl(p, type))($a)"))
    elseif op == 110                                   # OpConvertFToS
        to64 && (p.uses_i64 = true; return assign!(p, id, "i64_from_f32(f32($a))"))
        # A negative narrow result would otherwise keep its sign bits above its width.
        return assign!(p, id, cut(p, type, unsigned(p, type, "$(signedtype(p, type))($a)")))
    elseif op == 115                                   # OpFConvert
        return assign!(p, id, "$(wgsl(p, type))($a)")
    end
    refuse("conversion opcode $op")
end

function shuffle!(p::WGSLPrinter, inst::SpvInst)
    type, id = operand(inst, 1), operand(inst, 2)
    v1, v2 = operand(inst, 3), operand(inst, 4)
    n1 = lanes(p, p.m.typeof[v1])
    parts = map(inst.operands[5:end]) do k
        k == 0xffffffff && return "$(wgsl(p, scalartype(p, type)))()"
        k < n1 ? "$(value(p, v1))[$(k)]" : "$(value(p, v2))[$(k - n1)]"
    end
    assign!(p, id, "$(wgsl(p, type))($(join(parts, ", ")))")
end

function extinst!(p::WGSLPrinter, inst::SpvInst)
    type, id, set, n = operand(inst, 1), operand(inst, 2), operand(inst, 3), operand(inst, 4)
    p.m.extinst[set] == "GLSL.std.450" || refuse("extended instruction set $(p.m.extinst[set])")
    fn, issigned = get(WGSL_GLSL, n) do
        refuse("GLSL.std.450 instruction $n")
    end
    args = [value(p, x) for x in inst.operands[5:end]]
    if is64(p, type)
        p.uses_i64 = true
        fn64 = get(WGSL_I64_GLSL, n) do
            refuse("GLSL.std.450 instruction $n on 64-bit integers")
        end
        return assign!(p, id, "$fn64($(join(args, ", ")))")
    end
    if issigned
        args = [signed(p, p.m.typeof[x], a) for (x, a) in zip(inst.operands[5:end], args)]
        return assign!(p, id, unsigned(p, type, "$fn($(join(args, ", ")))"))
    end
    if n == 53                                         # Ldexp: exponent as i32
        args[2] = signed(p, p.m.typeof[operand(inst, 6)], args[2])
    end
    assign!(p, id, "$fn($(join(args, ", ")))")
end

# Memory semantics bits (SPIR-V 3.25).
const SEM_UNIFORM_MEMORY = 0x40
const SEM_WORKGROUP_MEMORY = 0x100

function barrier!(p::WGSLPrinter, inst::SpvInst)
    exec = (p.m.constants[operand(inst, 1)]::SpvScalar).bits
    sem = (p.m.constants[operand(inst, 3)]::SpvScalar).bits
    exec == 2 || refuse("a control barrier of execution scope $exec; WGSL barriers are workgroup-wide")
    sem & SEM_UNIFORM_MEMORY != 0 && line!(p, "storageBarrier();")
    (sem & SEM_WORKGROUP_MEMORY != 0 || sem & SEM_UNIFORM_MEMORY == 0) && line!(p, "workgroupBarrier();")
end

# GLSL.std.450 on 64-bit integers: the prelude function of each.
const WGSL_I64_GLSL = Dict{UInt32, String}(5 => "i64_abs", 38 => "u64_min", 39 => "i64_smin",
    41 => "u64_max", 42 => "i64_smax", 44 => "u64_clamp", 45 => "i64_sclamp")

const WGSL_SUBGROUP_REDUCE = Dict{UInt16, Tuple{String, Bool}}(
    349 => ("Add", false), 350 => ("Add", false), 351 => ("Mul", false), 352 => ("Mul", false),
    353 => ("Min", true), 354 => ("Min", false), 355 => ("Min", false),
    356 => ("Max", true), 357 => ("Max", false), 358 => ("Max", false),
    359 => ("And", false), 360 => ("Or", false), 361 => ("Xor", false),
    362 => ("And", false), 363 => ("Or", false), 364 => ("Xor", false))

function subgroup!(p::WGSLPrinter, inst::SpvInst)
    p.uses_subgroups = true
    op = inst.opcode
    type, id = operand(inst, 1), operand(inst, 2)
    scope = (p.m.constants[operand(inst, 3)]::SpvScalar).bits
    scope == 3 || refuse("group operation of scope $scope; WGSL has subgroup operations only")
    arg(k) = value(p, operand(inst, k))
    if op == 333
        assign!(p, id, "subgroupElect()")
    elseif op == 334 || op == 335
        assign!(p, id, "$(op == 334 ? "subgroupAll" : "subgroupAny")($(arg(4)))")
    elseif op == 337
        lane = operand(inst, 5)
        fn = haskey(p.m.constants, lane) ? "subgroupBroadcast" : "subgroupShuffle"
        assign!(p, id, "$fn($(arg(4)), $(arg(5)))")
    elseif op == 338
        assign!(p, id, "subgroupBroadcastFirst($(arg(4)))")
    elseif op == 339
        assign!(p, id, "subgroupBallot($(arg(4)))")
    elseif op in (345, 346, 347, 348)
        fn = Dict(345 => "subgroupShuffle", 346 => "subgroupShuffleXor",
                  347 => "subgroupShuffleUp", 348 => "subgroupShuffleDown")[op]
        assign!(p, id, "$fn($(arg(4)), $(arg(5)))")
    elseif haskey(WGSL_SUBGROUP_REDUCE, op)
        name, issigned = WGSL_SUBGROUP_REDUCE[op]
        groupop = operand(inst, 4)
        prefix = groupop == 0 ? "subgroup" : groupop == 1 ? "subgroupInclusive" :
                 groupop == 2 ? "subgroupExclusive" : refuse("group operation $groupop")
        groupop != 0 && !(name in ("Add", "Mul")) && refuse("subgroup $(name) scan has no WGSL builtin")
        x = operand(inst, 5)
        a = issigned ? signed(p, p.m.typeof[x], value(p, x)) : value(p, x)
        expr = "$(prefix)$(name)($a)"
        assign!(p, id, issigned ? unsigned(p, type, expr) : expr)
    else
        refuse("subgroup opcode $op")
    end
end

# ── Control flow ────────────────────────────────────────────────────────────

"""An enclosing construct while printing: a selection or switch with its merge, or a loop."""
struct Construct
    kind::Symbol        # :select, :switch, :guard, :loop, :continuing
    merge::UInt32
    cont::UInt32
    header::UInt32
end

"""
    exitstatement(target, stack) -> String, or nothing

What a branch to `target` is in WGSL when it leaves the current construct: "" for
falling through to the innermost selection's merge or reaching the loop's
continue target at the end of its body, "break;" or "continue;" for the innermost
loop's merge or continue target. `nothing` when `target` is an ordinary successor.
"""
function exitstatement(target::UInt32, stack::Vector{Construct}, atbodyend::Bool)
    inswitch = false
    for (depth, c) in enumerate(Iterators.reverse(stack))
        if c.kind == :guard
            # The rest of a loop body, printed inside `if` (see `terminator!`): its
            # merge is the loop's continue target, so from deeper in it is `continue`.
            target == c.merge && return depth == 1 ? "" : "continue;"
        elseif c.kind in (:select, :switch)
            target == c.merge && return depth == 1 ? "" : refuse("a branch out of a nested selection to %$(target)")
            inswitch |= c.kind == :switch
        elseif c.kind == :loop
            if target == c.merge || target == c.cont
                # In WGSL a `break` inside a `switch` leaves the switch, not the loop.
                inswitch && refuse("a branch from inside a switch to its loop's merge or continue target")
                target == c.merge && return "break;"
                return (depth == 1 && atbodyend) ? "" : "continue;"
            end
            return nothing
        elseif c.kind == :continuing
            target == c.header && return ""   # the back edge, printed by the loop
            target == c.merge && return "break;"
            return nothing
        end
    end
    return nothing
end

"""Assignments to `to`'s phi variables for the edge `from → to`, as a parallel copy."""
function phis!(p::WGSLPrinter, from::UInt32, to::UInt32)
    blk = get(p.blocks, to, nothing)
    (blk === nothing || isempty(blk.phis)) && return
    moves = Tuple{UInt32, String}[]
    for phi in blk.phis
        ops = phi.operands
        for k in 3:2:length(ops)
            ops[k + 1] == from && push!(moves, (ops[2], value(p, ops[k])))
        end
    end
    if length(moves) == 1
        assign!(p, moves[1][1], moves[1][2])
    else
        for (k, (id, expr)) in enumerate(moves)
            line!(p, "let phi$(from)_$(k) = $(expr);")
        end
        for (k, (id, _)) in enumerate(moves)
            assign!(p, id, "phi$(from)_$(k)")
        end
    end
end

"""Leave block `from` for `target`: its phi moves, then the exit statement if it is one."""
function jump!(p::WGSLPrinter, from::UInt32, target::UInt32, stack::Vector{Construct})
    phis!(p, from, target)
    stmt = exitstatement(target, stack, false)
    stmt === nothing && refuse("unstructured branch %$(from) -> %$(target)")
    isempty(stmt) || line!(p, stmt)
end

"""
    region!(p, start, stack)

Print the blocks from `start` on until the walk leaves the innermost construct.
"""
function region!(p::WGSLPrinter, start::UInt32, stack::Vector{Construct})
    b = start
    while true
        stmt = exitstatement(b, stack, true)
        if stmt !== nothing
            isempty(stmt) || line!(p, stmt)
            return
        end
        blk = p.blocks[b]
        if blk.merge !== nothing && blk.merge.loop
            loop!(p, blk, stack)
            b = blk.merge.merge
            continue
        end
        foreach(inst -> emit!(p, inst), blk.body)
        next = terminator!(p, blk, stack)
        next === nothing && return
        b = next
    end
end

"""Print `blk`'s terminator; return the block the walk continues with, or nothing."""
function terminator!(p::WGSLPrinter, blk::SpvBlock, stack::Vector{Construct})
    t = blk.terminator
    op = t.opcode
    if op == 249                                       # OpBranch
        target = operand(t, 1)
        stmt = exitstatement(target, stack, true)
        phis!(p, blk.label, target)
        if stmt !== nothing
            isempty(stmt) || line!(p, stmt)
            return nothing
        end
        return target
    elseif op == 250                                   # OpBranchConditional
        cond, tt, ft = value(p, operand(t, 1)), operand(t, 2), operand(t, 3)
        if blk.merge !== nothing && !blk.merge.loop    # a selection header
            m = blk.merge.merge
            inner = [stack; Construct(:select, m, 0, 0)]
            line!(p, "if ($cond) {")
            indented(() -> arm!(p, blk.label, tt, m, inner), p)
            line!(p, "} else {")
            indented(() -> arm!(p, blk.label, ft, m, inner), p)
            line!(p, "}")
            return m
        end
        st, sf = exitstatement(tt, stack, false), exitstatement(ft, stack, false)
        # A branch from the innermost loop's body to its continue target: print the
        # rest of the body inside `if` rather than a `continue` before it. Both mean
        # the same, but WGSL's uniformity analysis treats everything after a
        # divergent `continue` as divergent, the continuing block included, where a
        # barrier usually is (`if lane < s ... end; @synchronize` in a reduction).
        if !isempty(stack) && stack[end].kind == :loop && (st == "continue;") != (sf == "continue;") &&
           (st === nothing || sf === nothing)
            cont = stack[end].cont
            body, guard = st === nothing ? (tt, cond) : (ft, "!($cond)")
            line!(p, "if ($guard) {")
            indented(p) do
                phis!(p, blk.label, body)
                region!(p, body, [stack; Construct(:guard, cont, 0, 0)])
            end
            line!(p, "} else {")
            indented(() -> phis!(p, blk.label, cont), p)
            line!(p, "}")
            return nothing
        end
        if st !== nothing && sf !== nothing
            line!(p, "if ($cond) {")
            indented(() -> jump!(p, blk.label, tt, stack), p)
            line!(p, "} else {")
            indented(() -> jump!(p, blk.label, ft, stack), p)
            line!(p, "}")
            return nothing
        elseif st !== nothing
            line!(p, "if ($cond) {")
            indented(() -> jump!(p, blk.label, tt, stack), p)
            line!(p, "}")
            phis!(p, blk.label, ft)
            return ft
        elseif sf !== nothing
            line!(p, "if (!($cond)) {")
            indented(() -> jump!(p, blk.label, ft, stack), p)
            line!(p, "}")
            phis!(p, blk.label, tt)
            return tt
        end
        refuse("conditional branch in %$(blk.label) without a merge to two inner blocks")
    elseif op == 251                                   # OpSwitch
        blk.merge === nothing && refuse("switch without a selection merge")
        return switch!(p, blk, stack)
    elseif op in (252, 253, 255)                      # OpKill, OpReturn, OpUnreachable
        op == 252 && line!(p, "discard;")
        line!(p, isempty(p.outputs) ? "return;" : "return lava_out;")
        return nothing
    elseif op == 254
        line!(p, "return $(value(p, operand(t, 1)));")
        return nothing
    end
    refuse("terminator opcode $op")
end

"""One arm of an `if`: the edge's phi moves, then the arm's blocks up to the merge."""
function arm!(p::WGSLPrinter, from::UInt32, target::UInt32, merge::UInt32, stack::Vector{Construct})
    phis!(p, from, target)
    target == merge && return
    stmt = exitstatement(target, stack[1:end-1], false)
    if stmt !== nothing
        isempty(stmt) || line!(p, stmt)
        return
    end
    region!(p, target, stack)
end

function switch!(p::WGSLPrinter, blk::SpvBlock, stack::Vector{Construct})
    t = blk.terminator
    m = blk.merge.merge
    selector = operand(t, 1)
    sel = is64(p, p.m.typeof[selector]) ? "$(value(p, selector)).x" : value(p, selector)
    default = operand(t, 2)
    cases = Dict{UInt32, Vector{UInt32}}()
    order = UInt32[]
    for k in 3:2:length(t.operands)
        target = t.operands[k + 1]
        haskey(cases, target) || push!(order, target)
        push!(get!(Vector{UInt32}, cases, target), t.operands[k])
    end
    inner = [stack; Construct(:switch, m, 0, 0)]
    line!(p, "switch ($sel) {")
    indented(p) do
        for target in order
            target == default && continue
            line!(p, "case $(join(("$(x)u" for x in cases[target]), ", ")): {")
            indented(() -> arm!(p, blk.label, target, m, inner), p)
            line!(p, "}")
        end
        label = haskey(cases, default) ? "case $(join(("$(x)u" for x in cases[default]), ", ")), default" : "default"
        line!(p, "$label: {")
        indented(() -> arm!(p, blk.label, default, m, inner), p)
        line!(p, "}")
    end
    line!(p, "}")
    return m
end

"""
Print a loop: the header and body in `loop { }`, the continue construct in its
`continuing { }`, the back edge as `break if`.
"""
function loop!(p::WGSLPrinter, header::SpvBlock, stack::Vector{Construct})
    m, c = header.merge.merge, header.merge.cont
    inloop = [stack; Construct(:loop, m, c, header.label)]
    line!(p, "loop {")
    indented(p) do
        foreach(inst -> emit!(p, inst), header.body)
        if c == header.label
            # One block is the whole loop: its terminator is the back edge.
            continuing!(p, header, inloop)
            return
        end
        next = terminator!(p, header, inloop)
        next === nothing || region!(p, next, inloop)
        line!(p, "continuing {")
        indented(p) do
            region_continuing!(p, c, [stack; Construct(:continuing, m, c, header.label)])
        end
        line!(p, "}")
    end
    line!(p, "}")
end

"""The continue construct from `start` to the back edge."""
function region_continuing!(p::WGSLPrinter, start::UInt32, stack::Vector{Construct})
    b = start
    while true
        blk = p.blocks[b]
        blk.merge !== nothing && blk.merge.loop && refuse("a loop inside a continue construct")
        foreach(inst -> emit!(p, inst), blk.body)
        t = blk.terminator
        header = stack[end].header
        if t.opcode == 249 && operand(t, 1) == header
            phis!(p, b, header)
            return
        elseif t.opcode == 250 && (operand(t, 2) == header || operand(t, 3) == header)
            backedge!(p, blk, stack)
            return
        end
        next = terminator!(p, blk, stack)
        next === nothing && refuse("continue construct of %$(header) ends without a back edge")
        b = next
    end
end

"""The conditional back edge of `blk`: phi moves only when looping, then `break if`."""
function backedge!(p::WGSLPrinter, blk::SpvBlock, stack::Vector{Construct})
    t = blk.terminator
    header, merge = stack[end].header, stack[end].merge
    cond = value(p, operand(t, 1))
    loops = operand(t, 2) == header
    exit = operand(t, loops ? 3 : 2)
    exit == merge || refuse("back edge of %$(header) exits to %$(exit), not the loop merge")
    again = loops ? cond : "!($cond)"
    line!(p, "if ($again) {")
    indented(() -> phis!(p, blk.label, header), p)
    line!(p, "} else {")
    indented(() -> phis!(p, blk.label, merge), p)
    line!(p, "}")
    line!(p, "break if !($again);")
end

"""A one-block loop: its body is printed, its terminator is the back edge."""
function continuing!(p::WGSLPrinter, header::SpvBlock, stack::Vector{Construct})
    t = header.terminator
    line!(p, "continuing {")
    indented(p) do
        if t.opcode == 249
            phis!(p, header.label, header.label)
        else
            backedge!(p, header, [stack[1:end-1]; Construct(:continuing, stack[end].merge, header.label, header.label)])
        end
    end
    line!(p, "}")
end

# ── Atomics ─────────────────────────────────────────────────────────────────

"""Whether `ptr` points into a variable that atomics touch, so every access is atomic."""
isatomic(p::WGSLPrinter, ptr::UInt32) = get(p.roots, ptr, ptr) in p.atomics

"""The pointer operand of an atomic instruction."""
atomicpointer(inst::SpvInst) = inst.opcode == 228 ? operand(inst, 1) : operand(inst, 3)

"""
Find the variables atomics touch, following access chains back to their variable,
before anything is declared: WGSL types those variables `atomic<…>`.
"""
function findatomics!(p::WGSLPrinter, fn::SpvFunction)
    base = Dict{UInt32, UInt32}()
    for b in fn.blocks, inst in b.body
        if inst.opcode in (65, 66) || (inst.opcode == 124 && spvtype(p, operand(inst, 1)) isa SpvPointer)
            base[operand(inst, 2)] = operand(inst, 3)
        end
    end
    for b in fn.blocks, inst in b.body
        227 <= inst.opcode <= 242 || continue
        ptr = atomicpointer(inst)
        while haskey(base, ptr)
            ptr = base[ptr]
        end
        push!(p.atomics, ptr)
    end
end

const WGSL_ATOMIC_RMW = Dict{UInt16, String}(229 => "atomicExchange", 234 => "atomicAdd",
    235 => "atomicSub", 237 => "atomicMin", 239 => "atomicMax", 240 => "atomicAnd",
    241 => "atomicOr", 242 => "atomicXor")

function atomic!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    haskey(p.heapwords, atomicpointer(inst)) && return heapatomic!(p, inst)
    if op == 228                                       # OpAtomicStore
        line!(p, "atomicStore(&$(reference(p, operand(inst, 1))), $(value(p, operand(inst, 4))));")
        return
    end
    type, id, ptr = operand(inst, 1), operand(inst, 2), operand(inst, 3)
    isint32(p, type) || refuse("atomic on a $(wgsl(p, type)); WGSL atomics are 32-bit integers")
    ref = reference(p, ptr)
    if op == 227                                       # OpAtomicLoad
        assign!(p, id, "atomicLoad(&$ref)")
    elseif op == 230                                   # OpAtomicCompareExchange, strong
        value_, cmp = value(p, operand(inst, 7)), value(p, operand(inst, 8))
        push!(p.assigned, id)
        line!(p, "loop {")
        indented(p) do
            line!(p, "let r$(id) = atomicCompareExchangeWeak(&$ref, $cmp, $value_);")
            line!(p, "if (r$(id).exchanged || r$(id).old_value != $cmp) { $(valname(id)) = r$(id).old_value; break; }")
        end
        line!(p, "}")
    elseif haskey(WGSL_ATOMIC_RMW, op)
        assign!(p, id, "$(WGSL_ATOMIC_RMW[op])(&$ref, $(value(p, operand(inst, 6))))")
    else
        refuse("atomic opcode $op")
    end
end

# ── Uniform heap loads ──────────────────────────────────────────────────────
#
# WGSL requires every barrier to be in control flow it proves uniform, and takes a
# load from read-write storage to be non-uniform. A kernel that branches on a
# value it loads from the heap at an address every invocation shares (a count, a
# position, `if p0 > pos return` before its barriers) is uniform but not provably
# so. Such a load, where its value decides a branch, is made by invocation 0 into
# a workgroup variable and read back with `workgroupUniformLoad`, which WGSL takes
# as uniform; it costs two barriers. The load must itself be in uniform control
# flow, which WGSL then checks.

const WGSL_UNIFORM_BUILTINS = ("workgroup_id", "num_workgroups", "subgroup_size")

"""The variable a pointer points into, through access chains and pointer casts."""
function rootvariable(p::WGSLPrinter, ptr::UInt32)
    while haskey(p.m.results, ptr)
        inst = p.m.results[ptr]
        (inst.opcode in (65, 66) || inst.opcode == 124) || break
        ptr = operand(inst, 3)
    end
    return ptr
end

"""Whether a load from `ptr` gives every invocation the same value, given uniform operands."""
function uniformsource(p::WGSLPrinter, ptr::UInt32, args::UInt32)
    haskey(p.builtins, ptr) && return p.builtins[ptr] in WGSL_UNIFORM_BUILTINS
    root = rootvariable(p, ptr)
    return root == p.heaparray || root == args
end

"""
    uniformloads!(p, fn, params, globals)

Find the heap loads to read through `workgroupUniformLoad` (see above): in a
kernel with a barrier, those with an address that is the same for every
invocation, in a block every invocation reaches together, in the backward slice of
a branch condition. Declares their workgroup variables and the builtin they need.
"""
function uniformloads!(p::WGSLPrinter, fn::SpvFunction, params::Vector{String}, globals::IOBuffer)
    p.heaparray == 0 && return nothing
    any(b -> any(inst -> inst.opcode == 224, b.body), fn.blocks) || return nothing   # no barrier
    m = p.m
    k = findfirst(g -> get(m.names, operand(g, 2), "") == "args", m.globals)
    args = k === nothing ? UInt32(0) : operand(m.globals[k], 2)
    block = Dict{UInt32, UInt32}()            # result id -> its block
    for b in fn.blocks, inst in Iterators.flatten((b.phis, b.body))
        inst.opcode in SPV_TYPED_RESULT && (block[operand(inst, 2)] = b.label)
    end
    heapload(id) = haskey(m.results, id) && m.results[id].opcode == 61 &&
                   rootvariable(p, operand(m.results[id], 3)) == p.heaparray
    slice = conditionslice(p, fn)
    # Optimistic first: every heap load read uniformly. A load that is not (outside
    # the slice, or in a block not reached together) stays non-uniform, which can
    # make a later condition, and so later blocks, non-uniform: to a fixed point.
    chosen = Set{UInt32}(id for id in keys(m.results) if heapload(id))
    while true
        nonuniform = nonuniformvalues(p, fn, args, id -> heapload(id) && !(id in chosen))
        together = uniformblocks(fn, nonuniform)
        next = Set{UInt32}(id for id in slice if heapload(id) && !(id in nonuniform) &&
                           get(block, id, UInt32(0)) in together)
        next == chosen && break
        chosen = next
    end
    union!(p.uniformloads, chosen)
    isempty(p.uniformloads) && return nothing
    builtinparam!(p, params, "local_invocation_index", "u32")
    for id in sort!(collect(p.uniformloads))
        println(globals, "var<workgroup> lava_uload$id: u32;")
    end
    return nothing
end

"""
The values not the same for every invocation: from their sources (invocation ids,
loads but those of uniform builtins, the arguments and the heap, atomics, subgroup
operations, and the heap loads `notuniform(id)` names) forward to a fixed point.
"""
function nonuniformvalues(p::WGSLPrinter, fn::SpvFunction, args::UInt32, notuniform)
    insts = [inst for b in fn.blocks for inst in Iterators.flatten((b.phis, b.body))
             if inst.opcode in SPV_TYPED_RESULT]
    nonuniform = Set{UInt32}()
    changed = true
    while changed
        changed = false
        for inst in insts
            id = operand(inst, 2)
            id in nonuniform && continue
            source = (inst.opcode == 61 && (!uniformsource(p, operand(inst, 3), args) || notuniform(id))) ||
                     227 <= inst.opcode <= 242 || 333 <= inst.opcode <= 366 || inst.opcode == 4431
            if source || any(x -> x in nonuniform, @view inst.operands[3:end])
                push!(nonuniform, id)
                changed = true
            end
        end
    end
    return nonuniform
end

"""The blocks a block's terminator may branch to."""
function successorblocks(b::SpvBlock)
    t = b.terminator
    t.opcode == 249 && return UInt32[operand(t, 1)]
    t.opcode == 250 && return UInt32[operand(t, 2), operand(t, 3)]
    t.opcode == 251 && return UInt32[operand(t, 2); t.operands[4:2:end]]
    return UInt32[]
end

"""
Whether invocations can leave the construct `header` opens other than through its
merge: a return (or kill), or a branch to an enclosing loop's merge or continue
target (`loopexits`), a break or continue.
"""
function escapes(blocks::Dict{UInt32, SpvBlock}, header::SpvBlock, loopexits::Set{UInt32})
    own = header.merge.loop ? header.merge.cont : UInt32(0)
    seen = Set{UInt32}([header.merge.merge])
    work = successorblocks(header)
    while !isempty(work)
        l = pop!(work)
        (l in seen || !haskey(blocks, l)) && continue
        l in loopexits && l != own && return true
        push!(seen, l)
        blocks[l].terminator.opcode in (252, 253, 254) && return true    # OpKill, OpReturn(Value)
        append!(work, successorblocks(blocks[l]))
    end
    return false
end

"""
The blocks every invocation reaches together: the entry; a block a reached one
branches to unconditionally or on a uniform condition; the merge of a construct
entered together, unless invocations can leave it early ([`escapes`](@ref)), which
leaves the others behind.
"""
function uniformblocks(fn::SpvFunction, nonuniform::Set{UInt32})
    blocks = Dict(b.label => b for b in fn.blocks)
    loopexits = Set{UInt32}()
    for b in fn.blocks
        b.merge !== nothing && b.merge.loop && push!(loopexits, b.merge.merge, b.merge.cont)
    end
    together = Set{UInt32}([fn.blocks[1].label])
    changed = true
    while changed
        changed = false
        for b in fn.blocks
            b.label in together || continue
            t = b.terminator
            targets = t.opcode == 249 || (t.opcode in (250, 251) && !(operand(t, 1) in nonuniform)) ?
                      successorblocks(b) : UInt32[]
            b.merge !== nothing && !escapes(blocks, b, loopexits) && push!(targets, b.merge.merge)
            for l in targets
                l in together && continue
                push!(together, l)
                changed = true
            end
        end
    end
    return together
end

"""The values branch conditions are computed from, loads and their addresses included."""
function conditionslice(p::WGSLPrinter, fn::SpvFunction)
    m = p.m
    work = UInt32[]
    for b in fn.blocks
        t = b.terminator
        t.opcode in (250, 251) && push!(work, operand(t, 1))
    end
    seen = Set{UInt32}()
    while !isempty(work)
        id = pop!(work)
        (id in seen || !haskey(m.results, id)) && continue
        push!(seen, id)
        append!(work, @view m.results[id].operands[3:end])
    end
    return seen
end

# ── Heaps ───────────────────────────────────────────────────────────────────

"""
`heaps[h].data[w]`, an access chain into the array of heaps, as the expressions of
its heap and word: what the heap functions take.
"""
function heapword!(p::WGSLPrinter, inst::SpvInst)
    idx = inst.operands[4:end]
    length(idx) == 3 || refuse("an access chain into the heaps of $(length(idx)) indices; Lava emits heap, 0, word")
    p.heapwords[operand(inst, 2)] = (value(p, idx[1]), value(p, idx[3]))
    return nothing
end

"""A call of heap function `op` on the heap word `ptr` points to, with `args` after heap and word."""
function heapcall(p::WGSLPrinter, op::String, ptr::UInt32, args...)
    h, w = p.heapwords[ptr]
    push!(p.heapops, op)
    return "lava_heap_$(op)($(join((h, w, args...), ", ")))"
end

function heapatomic!(p::WGSLPrinter, inst::SpvInst)
    op = inst.opcode
    if op == 228                                       # OpAtomicStore
        line!(p, heapcall(p, "store", operand(inst, 1), value(p, operand(inst, 4))) * ";")
        return nothing
    end
    type, id, ptr = operand(inst, 1), operand(inst, 2), operand(inst, 3)
    isint32(p, type) || refuse("atomic on a $(wgsl(p, type)); WGSL atomics are 32-bit integers")
    if op == 227                                       # OpAtomicLoad
        assign!(p, id, heapcall(p, "load", ptr))
    elseif op == 230                                   # OpAtomicCompareExchange, strong
        assign!(p, id, heapcall(p, "cas", ptr, value(p, operand(inst, 8)), value(p, operand(inst, 7))))
    elseif haskey(WGSL_ATOMIC_RMW, op)
        assign!(p, id, heapcall(p, WGSL_ATOMIC_RMW[op], ptr, value(p, operand(inst, 6))))
    else
        refuse("atomic opcode $op")
    end
    return nothing
end

"""The binding of each of the device's heaps: the array of heaps' binding, then on."""
function declareheaps!(p::WGSLPrinter, id::UInt32, set::UInt32, binding::UInt32)
    p.heaparray = id
    compute = executionmodel(p) == ExecModel.GLCompute
    !compute && id in p.atomics && refuse("graphics stages require read-only heaps")
    access = compute ? "read_write" : "read"
    T = id in p.atomics ? "array<atomic<u32>>" : "array<u32>"
    for k in 0:p.heaps - 1
        line!(p, "@group($set) @binding($(binding + k)) var<storage, $access> lava_heap$k: $T;")
    end
    return nothing
end

"""
The body of a heap function: `stmt(k)` for heap `k`, chosen by the parameter `h`.
The last heap is the default, so a function with a result returns on every path.
"""
function heapswitch(p::WGSLPrinter, stmt)
    p.heaps == 1 && return "    $(stmt(0))\n"
    cases = join(("        case $(k)u: { $(stmt(k)) }\n" for k in 0:p.heaps - 2), "")
    return "    switch h {\n$(cases)        default: { $(stmt(p.heaps - 1)) }\n    }\n"
end

"""The heap functions the body calls (`heapcall`), each switching on its heap."""
function heapfunctions(p::WGSLPrinter)
    atomic = p.heaparray in p.atomics
    out = IOBuffer()
    for op in sort!(collect(p.heapops))
        if op == "load"
            print(out, "fn lava_heap_load(h: u32, w: u32) -> u32 {\n",
                  heapswitch(p, k -> atomic ? "return atomicLoad(&lava_heap$k[w]);" : "return lava_heap$k[w];"), "}\n")
        elseif op == "store"
            executionmodel(p) == ExecModel.GLCompute || refuse("graphics stages require read-only heaps")
            print(out, "fn lava_heap_store(h: u32, w: u32, v: u32) {\n",
                  heapswitch(p, k -> atomic ? "atomicStore(&lava_heap$k[w], v);" : "lava_heap$k[w] = v;"), "}\n")
        elseif op == "cas"
            print(out, "fn lava_heap_cas(h: u32, w: u32, cmp: u32, v: u32) -> u32 {\n",
                  heapswitch(p, k -> "loop { let r = atomicCompareExchangeWeak(&lava_heap$k[w], cmp, v); " *
                                     "if (r.exchanged || r.old_value != cmp) { return r.old_value; } }"), "}\n")
        else
            print(out, "fn lava_heap_$(op)(h: u32, w: u32, v: u32) -> u32 {\n",
                  heapswitch(p, k -> "return $(op)(&lava_heap$k[w], v);"), "}\n")
        end
    end
    return String(take!(out))
end

"""`type` with its 32-bit integers atomic, for a variable atomics touch."""
function atomictype(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, type)
    t isa SpvInt && t.width == 32 && return "atomic<u32>"
    t isa SpvArray && return "array<$(atomictype(p, t.elem)), $(t.len)>"
    t isa SpvRuntimeArray && return "array<$(atomictype(p, t.elem))>"
    if t isa SpvStruct
        name = "$(p.structs[type])_atomic"
        line!(p, "struct $name {")
        indented(p) do
            for (k, mem) in enumerate(t.members)
                line!(p, "m$(k - 1): $(atomictype(p, mem)),")
            end
        end
        line!(p, "}")
        return name
    end
    refuse("a variable of type $(wgsl(p, type)) used by atomics")
end

"""
A pointer bitcast. Logical addressing has none in general; the one Lava emits points
at the first element of what the source points to (an array's or a struct's first
member, any depth), which is that element's reference.
"""
function pointercast!(p::WGSLPrinter, inst::SpvInst)
    id, src = operand(inst, 2), operand(inst, 3)
    want = (spvtype(p, operand(inst, 1))::SpvPointer).pointee
    haskey(p.heapwords, src) && refuse("a pointer cast of a heap word")
    expr, type = reference(p, src), pointee(p, src)
    while type != want && wgsl_or_nothing(p, type) != wgsl_or_nothing(p, want)
        t = spvtype(p, type)
        if t isa SpvStruct
            expr, type = "$(expr).m0", t.members[1]
        elseif t isa Union{SpvArray, SpvRuntimeArray, SpvVector}
            expr, type = "$(expr)[0]", t.elem
        else
            refuse("a pointer cast from $(wgsl_or_nothing(p, pointee(p, src))) to $(wgsl_or_nothing(p, want)) that is not to a first element")
        end
    end
    p.refs[id] = expr
    p.roots[id] = get(p.roots, src, src)
    return nothing
end

"""The WGSL spelling of `type`, or `nothing` for one WGSL has no value type for."""
function wgsl_or_nothing(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, type)
    t isa Union{SpvPointer, SpvVoid, SpvFunctionType} && return nothing
    return wgsl(p, type)
end
