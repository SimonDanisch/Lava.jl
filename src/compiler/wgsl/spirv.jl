# Reading a SPIR-V module back: the instruction stream, names, decorations, types,
# constants and functions, as the WGSL printer needs them. Only what Lava's WebGPU
# target emits (`TargetFeatures(webgpu = true)`) has to be understood here; anything
# else is refused by the printer with its opcode.

"""One SPIR-V instruction: its opcode and the words after the first."""
struct SpvInst
    opcode::UInt16
    operands::Vector{UInt32}
end

"""The `k`-th operand word of `inst`, 1-based."""
operand(inst::SpvInst, k::Integer) = inst.operands[k]

"""A literal string starting at operand `k`: UTF-8, NUL-terminated, packed four bytes a word."""
function literal_string(inst::SpvInst, k::Integer)
    bytes = UInt8[]
    for w in inst.operands[k:end], s in 0:8:24
        b = UInt8((w >> s) & 0xff)
        b == 0x00 && return String(bytes)
        push!(bytes, b)
    end
    return String(bytes)
end

"""How many operand words the string starting at operand `k` takes."""
literal_string_words(inst::SpvInst, k::Integer) = ncodeunits(literal_string(inst, k)) ÷ 4 + 1

# ── Types ───────────────────────────────────────────────────────────────────

abstract type SpvType end
struct SpvVoid <: SpvType end
struct SpvBool <: SpvType end
struct SpvInt <: SpvType
    width::Int
    signed::Bool
end
struct SpvFloat <: SpvType
    width::Int
end
struct SpvVector <: SpvType
    elem::UInt32
    n::Int
end
struct SpvImage <: SpvType end
struct SpvSampledImage <: SpvType end
struct SpvArray <: SpvType
    elem::UInt32
    len::Int
end
struct SpvRuntimeArray <: SpvType
    elem::UInt32
end
struct SpvStruct <: SpvType
    members::Vector{UInt32}
end
struct SpvPointer <: SpvType
    storage::UInt32
    pointee::UInt32
end
struct SpvFunctionType <: SpvType
    ret::UInt32
    params::Vector{UInt32}
end

# ── Constants ───────────────────────────────────────────────────────────────

abstract type SpvConstant end
"""A scalar constant: its type and its bits (two words for 64-bit types, low first)."""
struct SpvScalar <: SpvConstant
    type::UInt32
    bits::UInt64
end
struct SpvComposite <: SpvConstant
    type::UInt32
    parts::Vector{UInt32}
end
struct SpvNull <: SpvConstant
    type::UInt32
end

# ── Functions ───────────────────────────────────────────────────────────────

"""The merge instruction of a header block: a selection or a loop."""
struct SpvMerge
    loop::Bool
    merge::UInt32
    cont::UInt32   # loops only, 0 for a selection
end

mutable struct SpvBlock
    label::UInt32
    phis::Vector{SpvInst}
    body::Vector{SpvInst}
    merge::Union{Nothing, SpvMerge}
    terminator::SpvInst
end

mutable struct SpvFunction
    id::UInt32
    type::UInt32
    params::Vector{Tuple{UInt32, UInt32}}   # (type, id)
    locals::Vector{SpvInst}                 # its OpVariable Function
    blocks::Vector{SpvBlock}
end

"""
    SpvModule(words)

The parts of a SPIR-V module the WGSL printer reads. `results` maps every result id
to its defining instruction and `typeof` to its type id.
"""
mutable struct SpvModule
    names::Dict{UInt32, String}
    decorations::Dict{UInt32, Vector{SpvInst}}
    member_decorations::Dict{Tuple{UInt32, UInt32}, Vector{SpvInst}}
    types::Dict{UInt32, SpvType}
    constants::Dict{UInt32, SpvConstant}
    globals::Vector{SpvInst}                 # OpVariable outside functions
    extinst::Dict{UInt32, String}
    entry::Union{Nothing, SpvInst}
    local_size::NTuple{3, Int}
    functions::Vector{SpvFunction}
    typeof::Dict{UInt32, UInt32}
    results::Dict{UInt32, SpvInst}
end

# Opcodes with a result type and a result id, so `typeof`/`results` can record them:
# everything the printer reads as a value, plus the ones it refuses by name.
const SPV_TYPED_RESULT = Set{UInt16}([
    1, 12, 57, 59, 61, 65, 66, 67, 70, 77, 78, 79, 80, 81, 82, 83, 87,
    109:126..., 126:205..., 207, 208, 209, 227, 229:242..., 245,
    333:366..., 4431, 4450, 4451, 4452, 6035, 5614, 5615])

function SpvModule(words::AbstractVector{UInt32})
    words[1] == 0x07230203 || error("not a SPIR-V module (magic $(repr(words[1])))")
    m = SpvModule(Dict(), Dict(), Dict(), Dict(), Dict(), SpvInst[], Dict(), nothing, (1, 1, 1),
                  SpvFunction[], Dict(), Dict())
    fn = nothing
    block = nothing
    i = 6
    while i <= length(words)
        n = Int(words[i] >> 16)
        op = UInt16(words[i] & 0xffff)
        n == 0 && error("SPIR-V instruction of length 0 at word $i")
        inst = SpvInst(op, words[i + 1:i + n - 1])
        i += n
        if op in SPV_TYPED_RESULT && length(inst.operands) >= 2
            m.typeof[inst.operands[2]] = inst.operands[1]
            m.results[inst.operands[2]] = inst
        end
        if op == 5                                   # OpName
            m.names[operand(inst, 1)] = literal_string(inst, 2)
        elseif op == 71                              # OpDecorate
            push!(get!(Vector{SpvInst}, m.decorations, operand(inst, 1)), inst)
        elseif op == 72                              # OpMemberDecorate
            push!(get!(Vector{SpvInst}, m.member_decorations, (operand(inst, 1), operand(inst, 2))), inst)
        elseif op == 11                              # OpExtInstImport
            m.extinst[operand(inst, 1)] = literal_string(inst, 2)
        elseif op == 15                              # OpEntryPoint
            m.entry === nothing || error("more than one entry point")
            m.entry = inst
        elseif op == 16                              # OpExecutionMode
            if operand(inst, 2) == 17                # LocalSize
                m.local_size = (Int(operand(inst, 3)), Int(operand(inst, 4)), Int(operand(inst, 5)))
            end
        elseif 19 <= op <= 33
            readtype!(m, inst)
        elseif op in (41, 42, 43, 44, 46)
            readconstant!(m, inst)
        elseif op == 59 && fn === nothing            # OpVariable at module scope
            push!(m.globals, inst)
        elseif op == 54                              # OpFunction
            fn = SpvFunction(operand(inst, 2), operand(inst, 4), Tuple{UInt32, UInt32}[], SpvInst[], SpvBlock[])
            m.typeof[fn.id] = operand(inst, 1)
        elseif op == 55                              # OpFunctionParameter
            push!(fn.params, (operand(inst, 1), operand(inst, 2)))
            m.typeof[operand(inst, 2)] = operand(inst, 1)
        elseif op == 56                              # OpFunctionEnd
            push!(m.functions, fn)
            fn = nothing
        elseif op == 248                             # OpLabel
            block = SpvBlock(operand(inst, 1), SpvInst[], SpvInst[], nothing, inst)
        elseif fn !== nothing && block !== nothing
            if op == 59
                push!(fn.locals, inst)
            elseif op == 245
                push!(block.phis, inst)
            elseif op == 247                         # OpSelectionMerge
                block.merge = SpvMerge(false, operand(inst, 1), 0)
            elseif op == 246                         # OpLoopMerge
                block.merge = SpvMerge(true, operand(inst, 1), operand(inst, 2))
            elseif op in (249, 250, 251, 252, 253, 254, 255)   # terminators
                block.terminator = inst
                push!(fn.blocks, block)
                block = nothing
            elseif op in (8, 317)                    # OpLine, OpNoLine
            else
                push!(block.body, inst)
            end
        end
    end
    m.entry === nothing && error("SPIR-V module has no entry point")
    return m
end

function readtype!(m::SpvModule, inst::SpvInst)
    op = inst.opcode
    id = operand(inst, 1)
    m.types[id] = if op == 19
        SpvVoid()
    elseif op == 20
        SpvBool()
    elseif op == 21
        SpvInt(Int(operand(inst, 2)), operand(inst, 3) == 1)
    elseif op == 22
        SpvFloat(Int(operand(inst, 2)))
    elseif op == 23
        SpvVector(operand(inst, 2), Int(operand(inst, 3)))
    elseif op == 25
        SpvImage()
    elseif op == 27
        SpvSampledImage()
    elseif op == 28
        len = m.constants[operand(inst, 3)]::SpvScalar
        SpvArray(operand(inst, 2), Int(len.bits))
    elseif op == 29
        SpvRuntimeArray(operand(inst, 2))
    elseif op == 30
        SpvStruct(inst.operands[2:end])
    elseif op == 32
        SpvPointer(operand(inst, 2), operand(inst, 3))
    elseif op == 33
        SpvFunctionType(operand(inst, 2), inst.operands[3:end])
    else
        error("SPIR-V type opcode $op has no WGSL form")
    end
    return nothing
end

function readconstant!(m::SpvModule, inst::SpvInst)
    op = inst.opcode
    type, id = operand(inst, 1), operand(inst, 2)
    m.typeof[id] = type
    m.constants[id] = if op == 41
        SpvScalar(type, 1)
    elseif op == 42
        SpvScalar(type, 0)
    elseif op == 43
        lo = UInt64(operand(inst, 3))
        hi = length(inst.operands) >= 4 ? UInt64(operand(inst, 4)) : UInt64(0)
        SpvScalar(type, lo | (hi << 32))
    elseif op == 44
        SpvComposite(type, inst.operands[3:end])
    else
        SpvNull(type)
    end
    return nothing
end

"""The decoration `dec` of `id` with its operands, or `nothing`."""
function decoration(m::SpvModule, id::UInt32, dec::Integer)
    for d in get(m.decorations, id, SpvInst[])
        operand(d, 2) == dec && return d.operands[3:end]
    end
    return nothing
end

function member_decoration(m::SpvModule, struct_id::UInt32, member::Integer, dec::Integer)
    for d in get(m.member_decorations, (struct_id, UInt32(member)), SpvInst[])
        operand(d, 3) == dec && return d.operands[4:end]
    end
    return nothing
end
