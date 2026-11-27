# The module level of the WGSL printer: bindings, workgroup and private variables,
# the entry point with its builtins, declarations, and the 64-bit prelude.

# SPIR-V BuiltIn → the WGSL builtin of a compute entry point.
const WGSL_COMPUTE_BUILTINS = Dict{UInt32, String}(
    28 => "global_invocation_id", 27 => "local_invocation_id", 26 => "workgroup_id",
    24 => "num_workgroups", 29 => "local_invocation_index",
    41 => "subgroup_invocation_id", 36 => "subgroup_size")

const WGSL_VERTEX_BUILTINS = Dict(
    BuiltIn.VertexIndex => "vertex_index", BuiltIn.InstanceIndex => "instance_index")
const WGSL_FRAGMENT_BUILTINS = Dict(
    BuiltIn.FragCoord => "position", BuiltIn.FrontFacing => "front_facing")

const WGSL_RESERVED = Set(["array", "bool", "break", "case", "const", "continue", "continuing",
    "default", "diagnostic", "discard", "else", "enable", "f16", "f32", "false", "fn", "for", "i32",
    "if", "let", "loop", "override", "requires", "return", "struct", "switch", "true", "u32", "var",
    "while", "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4", "atomic", "ptr", "sampler",
    "texture_2d", "main"])

"""A WGSL identifier for global `id`: its SPIR-V name when that is one, else `g<id>`."""
function globalname(p::WGSLPrinter, id::UInt32)
    name = get(p.m.names, id, "")
    ok = occursin(r"^[A-Za-z][A-Za-z0-9_]*$", name) && !(name in WGSL_RESERVED) && !startswith(name, "__")
    return ok ? name : "g$(id)"
end

"""Whether every member of the block struct `type` is `NonWritable`."""
function readonly_block(p::WGSLPrinter, type::UInt32)
    t = spvtype(p, type)
    t isa SpvStruct || return false
    return all(k -> member_decoration(p.m, type, k - 1, 24) !== nothing, eachindex(t.members))
end

"""
SPIR-V builtins WGSL lacks, computed in the entry point from builtins it has: the
name, the builtins (and their types) the expression reads, and the expression.
Subgroups are taken to be consecutive runs of `local_invocation_index`, which is
how GPUs form them for a 1-D workgroup; WGSL leaves the grouping to the
implementation.
"""
const WGSL_DERIVED_BUILTINS = Dict{UInt32, Tuple{String, Vector{Tuple{String, String}}, String}}(
    40 => ("lava_subgroup_id", [("local_invocation_index", "u32"), ("subgroup_size", "u32")],
           "local_invocation_index / subgroup_size"),
    38 => ("lava_num_subgroups", [("subgroup_size", "u32")],
           "(LAVA_WORKGROUP_INVOCATIONS + subgroup_size - 1u) / subgroup_size"))

"""The entry parameter of builtin `name`, once."""
function builtinparam!(p::WGSLPrinter, params::Vector{String}, name::String, type::String)
    name in ("subgroup_invocation_id", "subgroup_size") && (p.uses_subgroups = true)
    any(s -> startswith(s, "@builtin($name) "), params) || push!(params, "@builtin($name) $name: $type")
    return nothing
end

"""A derived builtin (`WGSL_DERIVED_BUILTINS`): its parameters, its definition in the entry point, its name."""
function derivedbuiltin!(p::WGSLPrinter, params::Vector{String}, bi::UInt32)
    name, reads, expr = WGSL_DERIVED_BUILTINS[bi]
    foreach(((n, t),) -> builtinparam!(p, params, n, t), reads)
    ls = p.m.local_size
    line = "let $name = $(replace(expr, "LAVA_WORKGROUP_INVOCATIONS" => "$(ls[1] * ls[2] * ls[3])u"));"
    line in p.derived || push!(p.derived, line)
    return name
end

"""An input or output location, including the producer/consumer interpolation."""
function location_attribute(p::WGSLPrinter, id::UInt32, varying::Bool)
    loc = decoration(p.m, id, 30)
    loc === nothing && refuse("interface variable %$id without a Location decoration")
    attr = "@location($(loc[1]))"
    if varying
        decoration(p.m, id, 14) !== nothing && return attr * " @interpolate(flat)"
        decoration(p.m, id, 13) !== nothing && return attr * " @interpolate(linear)"
    end
    return attr
end

"""Declare the module-scope variables and collect the entry point's interface."""
function declare_globals!(p::WGSLPrinter; flip_y::Bool)
    params = String[]
    model = executionmodel(p)
    for g in p.m.globals
        ptrtype, id, storage = operand(g, 1), operand(g, 2), operand(g, 3)
        type = (spvtype(p, ptrtype)::SpvPointer).pointee
        if storage == 1                                    # Input
            bi = decoration(p.m, id, 11)
            if bi === nothing
                model == ExecModel.GLCompute && refuse("compute input without a BuiltIn decoration")
                name = "lava_in$id"
                attr = location_attribute(p, id, model == ExecModel.Fragment)
                push!(params, "$attr $name: $(wgsl(p, type))")
            elseif model == ExecModel.GLCompute && haskey(WGSL_DERIVED_BUILTINS, bi[1])
                p.builtins[id] = derivedbuiltin!(p, params, bi[1])
                continue
            else
                builtins = model == ExecModel.Vertex ? WGSL_VERTEX_BUILTINS :
                           model == ExecModel.Fragment ? WGSL_FRAGMENT_BUILTINS : WGSL_COMPUTE_BUILTINS
                name = get(builtins, bi[1]) do
                    refuse("builtin $(bi[1]) has no WGSL equivalent for execution model $model")
                end
                builtinparam!(p, params, name, wgsl(p, type))
            end
            p.builtins[id] = name
            p.refs[id] = name
        elseif storage == 3                                # Output
            bi = decoration(p.m, id, 11)
            attr = if bi === nothing
                location_attribute(p, id, model == ExecModel.Vertex)
            elseif model == ExecModel.Vertex && bi[1] == BuiltIn.Position
                flip_y && (p.flip_position = id)
                "@builtin(position)"
            else
                refuse("output builtin $(bi[1]) for execution model $model")
            end
            push!(p.outputs, "$attr m$id: $(wgsl(p, type)),")
            p.refs[id] = "lava_out.m$id"
        elseif storage == 0                                # UniformConstant: combined sampler
            spvtype(p, type) isa SpvSampledImage || refuse("unsupported UniformConstant type %$type")
            binding = decoration(p.m, id, 33)
            binding === nothing && refuse("a sampled image without a binding")
            slot = Int(binding[1])
            name = "lava_texture$slot"
            line!(p, "@group(1) @binding($(2slot)) var $name: texture_2d<f32>;")
            line!(p, "@group(1) @binding($(2slot + 1)) var $(name)_sampler: sampler;")
            p.refs[id] = name
        elseif storage == 12                               # StorageBuffer
            set = decoration(p.m, id, 34)
            binding = decoration(p.m, id, 33)
            (set === nothing || binding === nothing) && refuse("storage buffer without a binding")
            # The array of heaps, the one array of blocks Lava's WebGPU target declares.
            spvtype(p, type) isa SpvArray && (declareheaps!(p, id, set[1], binding[1]); continue)
            name = globalname(p, id)
            access = readonly_block(p, type) ? "read" : "read_write"
            T = id in p.atomics ? atomictype(p, type) : wgsl(p, type)
            line!(p, "@group($(set[1])) @binding($(binding[1])) var<storage, $access> $name: $T;")
            p.refs[id] = name
        elseif storage == 4 || storage == 6                # Workgroup, Private
            name = globalname(p, id)
            space = storage == 4 ? "workgroup" : "private"
            # WebGPU zero-initializes workgroup memory, and WGSL allows no initializer there.
            init = (storage == 6 && length(g.operands) >= 4) ? " = $(value(p, operand(g, 4)))" : ""
            T = id in p.atomics ? atomictype(p, type) : wgsl(p, type)
            line!(p, "var<$space> $name: $T$init;")
            p.refs[id] = name
        else
            refuse("module variable in storage class $storage")
        end
    end
    return params
end

const WGSL_RESULT_USE = r"\bv([0-9]+)\b"
const WGSL_RESULT_ASSIGNMENT = r"^ +v([0-9]+) = "

"""
Find the single assignments whose results are read only within the scope of
the assignment. Follow the emitted body because pointer expressions can
include operands defined earlier, and structured branches can emit a block
more than once. The printer indents every lexical scope by four spaces.
"""
function lexical_result_bindings(lines, assigned, results)
    bindings = Dict{UInt32, Tuple{Int, Int}}() # result => (definition line, scope)
    mutable_results = Set{UInt32}()
    for id in assigned
        # Composite inserts write a member after the initial assignment. Phi
        # nodes and strong CAS may be assigned on more than one control path.
        results[id].opcode in (82, 230, 245) && push!(mutable_results, id)
    end
    scopes = [1]
    parents = [0]
    for (number, line) in enumerate(lines)
        start = findfirst(!=(' '), line)
        start === nothing && continue
        level = (start - 1) ÷ 4
        while length(scopes) > level
            pop!(scopes)
        end
        while length(scopes) < level
            push!(parents, last(scopes))
            push!(scopes, length(parents))
        end
        scope = last(scopes)
        definition = match(WGSL_RESULT_ASSIGNMENT, line)
        reads = definition === nothing ? line : SubString(line, ncodeunits(definition.match) + 1)
        for operand in eachmatch(WGSL_RESULT_USE, reads)
            id = parse(UInt32, operand.captures[1])
            id in assigned && !(id in mutable_results) || continue
            _, declared_in = get(bindings, id, (0, 0))
            # Parents always have smaller ids. A use before its assignment
            # or in a sibling scope requires the function-scope variable.
            ancestor = scope
            while ancestor > declared_in
                ancestor = parents[ancestor]
            end
            if iszero(declared_in) || ancestor != declared_in
                delete!(bindings, id)
                push!(mutable_results, id)
            end
        end
        definition === nothing && continue
        id = parse(UInt32, definition.captures[1])
        id in assigned && !(id in mutable_results) || continue
        if haskey(bindings, id)
            delete!(bindings, id)
            push!(mutable_results, id)
        else
            bindings[id] = (number, scope)
        end
    end
    return bindings
end

function write_wgsl_body(out::IO, lines, bindings, types)
    definitions = Dict(line => id for (id, (line, _)) in bindings)
    for (number, line) in enumerate(lines)
        id = get(definitions, number, UInt32(0))
        if iszero(id)
            println(out, line)
        else
            start = findfirst(!=(' '), line)
            name = valname(id)
            println(out, SubString(line, 1, start - 1), "let ", name, ": ", types[id],
                    SubString(line, start + ncodeunits(name)))
        end
    end
    return nothing
end

"""
    spirv_to_wgsl(words; heaps = 1, linear = false, flip_y = true) -> String

WGSL for a compute, vertex or fragment module of Lava's WebGPU target, on a device with `heaps` heaps
(at most `WEBGPU_MAX_HEAPS`), bound after the module's array of heaps' binding.

`linear` is for a one-dimensional grid of workgroups dispatched as a 2- or 3-D one,
because it has more workgroups than a dimension holds (65535 in WebGPU): the
kernel's `workgroup_id`, `num_workgroups` and `global_invocation_id` are the 1-D
ones, recovered from the dispatched grid. A kernel built through
KernelAbstractions does not need it (Lava's `linear_block_index` reads the grid);
a KernelInterface one reads its ids straight from the builtins.

Vertex entry points mirror clip Y by default, preserving Lava's Y-down
convention in a WebGPU viewport. A draw with a flipped viewport disables this.
"""
function spirv_to_wgsl(words::AbstractVector{UInt32}; heaps::Integer = 1, linear::Bool = false,
                       flip_y::Bool = true)
    1 <= heaps <= WEBGPU_MAX_HEAPS || throw(ArgumentError("$heaps heaps; a module binds 1 to $WEBGPU_MAX_HEAPS"))
    return persisted_wgsl(words, heaps, linear, flip_y) do
        print_wgsl(words, heaps, linear, flip_y)
    end
end

function print_wgsl(words, heaps, linear, flip_y)
    m = SpvModule(words)
    p = WGSLPrinter(m, IOBuffer(), 0, Dict(), Dict(), Dict(), Dict(), false, false, false, false, Set{UInt32}(),
                    Dict{UInt32, UInt32}(), Set{UInt32}(), Int(heaps), UInt32(0),
                    Dict{UInt32, Tuple{String, String}}(), Set{String}(), String[], Set{UInt32}(),
                    String[], UInt32(0))
    model = executionmodel(p)
    model in (ExecModel.GLCompute, ExecModel.Vertex, ExecModel.Fragment) ||
        refuse("entry point of execution model $model")
    compute = model == ExecModel.GLCompute
    linear && !compute && throw(ArgumentError("only compute entry points have a dispatch grid"))
    length(m.functions) == 1 || refuse("$(length(m.functions)) functions; a WebGPU module is one inlined function")
    fn = only(m.functions)
    for b in fn.blocks
        p.blocks[b.label] = b
    end
    findatomics!(p, fn)

    # Declarations first into their own buffer: they decide which enables are needed.
    declare_structs!(p)
    params = declare_globals!(p; flip_y)
    compute && uniformloads!(p, fn, params, p.out)
    globals = String(take!(p.out))

    # The body, so that `assigned` knows every variable it needs.
    p.indent = 1
    for l in fn.locals
        id = operand(l, 2)
        p.refs[id] = "l$(id)"
    end
    region!(p, fn.blocks[1].label, Construct[])
    body = String(take!(p.out))
    lines = split(body, '\n'; keepempty=false)
    bindings = lexical_result_bindings(lines, p.assigned, m.results)
    result_types = Dict(id => wgsl(p, m.typeof[id]) for id in p.assigned)

    ls = m.local_size
    preamble = linear ? lineargrid!(params, Set(values(p.builtins)), ls) : String[]

    out = IOBuffer()
    p.uses_f16 && println(out, "enable f16;")
    p.uses_subgroups && println(out, "enable subgroups;")
    # Subgroup operations act on the invocations that reach them, as in SPIR-V,
    # which is what the kernels assume; WGSL would demand uniform control flow.
    p.uses_subgroups && println(out, "diagnostic(off, subgroup_uniformity);")
    # Lava's SPIR-V inputs can use derivatives in control flow WGSL cannot
    # prove uniform. Keep the shader's existing implicit-derivative semantics.
    model == ExecModel.Fragment && println(out, "diagnostic(off, derivative_uniformity);")
    print(out, globals)
    print(out, heapfunctions(p))
    p.uses_i64 && print(out, WGSL_I64_PRELUDE)
    p.uses_nonfinite && print(out, NONFINITE_PRELUDE)
    p.uses_nonfinite && p.uses_f16 && print(out, NONFINITE_F16_PRELUDE)
    if !isempty(p.outputs)
        println(out, "struct LavaOutput {")
        foreach(l -> println(out, "    ", l), p.outputs)
        println(out, "}")
    end
    stage = compute ? "@compute @workgroup_size($(ls[1]), $(ls[2]), $(ls[3]))" :
            model == ExecModel.Vertex ? "@vertex" : "@fragment"
    result = isempty(p.outputs) ? "" : " -> LavaOutput"
    println(out, stage)
    println(out, "fn main($(join(params, ", ")))$result {")
    isempty(p.outputs) || println(out, "    var lava_out: LavaOutput;")
    foreach(l -> println(out, "    ", l), preamble)
    foreach(l -> println(out, "    ", l), p.derived)
    for l in fn.locals
        id = operand(l, 2)
        type = (spvtype(p, operand(l, 1))::SpvPointer).pointee
        init = length(l.operands) >= 4 ? " = $(value(p, operand(l, 4)))" : ""
        println(out, "    var l$(id): $(wgsl(p, type))$(init);")
    end
    for id in sort!(collect(p.assigned))
        haskey(bindings, id) && continue
        println(out, "    var $(valname(id)): $(result_types[id]);")
    end
    write_wgsl_body(out, lines, bindings, result_types)
    println(out, "}")
    return String(take!(out))
end

"""
Replace the entry parameters of the grid builtins (`spirv_to_wgsl`'s `linear`) by
the dispatched ones under names of their own, and return the lines that define the
kernel's from them, for the builtins it `used`.
"""
function lineargrid!(params::Vector{String}, used::Set{String}, ls)
    grid = ("workgroup_id", "num_workgroups", "global_invocation_id", "local_invocation_id")
    filter!(s -> !any(n -> startswith(s, "@builtin($n) "), grid), params)
    push!(params, "@builtin(workgroup_id) lava_wid: vec3<u32>", "@builtin(num_workgroups) lava_nwg: vec3<u32>",
          "@builtin(local_invocation_id) lava_lid: vec3<u32>")
    lines = ["let lava_lin = lava_wid.x + lava_nwg.x * (lava_wid.y + lava_nwg.y * lava_wid.z);"]
    "workgroup_id" in used && push!(lines, "let workgroup_id = vec3<u32>(lava_lin, 0u, 0u);")
    "num_workgroups" in used &&
        push!(lines, "let num_workgroups = vec3<u32>(lava_nwg.x * lava_nwg.y * lava_nwg.z, 1u, 1u);")
    "global_invocation_id" in used &&
        push!(lines, "let global_invocation_id = vec3<u32>(lava_lin * $(ls[1])u + lava_lid.x, lava_lid.y, lava_lid.z);")
    "local_invocation_id" in used && push!(lines, "let local_invocation_id = lava_lid;")
    return lines
end

"""
An infinity or NaN from its bits. WGSL refuses a constant expression that
evaluates to one, `bitcast<f32>(0x7f800000u)` included; a function call is not a
constant expression, so the value is made at run time.
"""
const NONFINITE_PRELUDE = """
fn lava_nonfinite_f32(bits: u32) -> f32 { return bitcast<f32>(bits); }
"""
const NONFINITE_F16_PRELUDE = """
fn lava_nonfinite_f16(bits: u32) -> f16 { return bitcast<vec2<f16>>(bits).x; }
"""

const WGSL_I64_PRELUDE = """
// 64-bit integers as vec2<u32>(low, high): WGSL has none.
fn i64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn i64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn u32_mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u; let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
fn i64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = u32_mul_wide(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}
fn i64_and(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return a & b; }
fn i64_or(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return a | b; }
fn i64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return a ^ b; }
fn i64_eq(a: vec2<u32>, b: vec2<u32>) -> bool { return all(a == b); }
fn i64_ne(a: vec2<u32>, b: vec2<u32>) -> bool { return any(a != b); }
fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }
fn u64_le(a: vec2<u32>, b: vec2<u32>) -> bool { return !u64_lt(b, a); }
fn u64_gt(a: vec2<u32>, b: vec2<u32>) -> bool { return u64_lt(b, a); }
fn u64_ge(a: vec2<u32>, b: vec2<u32>) -> bool { return !u64_lt(a, b); }
fn i64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
    let ah = bitcast<i32>(a.y); let bh = bitcast<i32>(b.y);
    return ah < bh || (ah == bh && a.x < b.x);
}
fn i64_le(a: vec2<u32>, b: vec2<u32>) -> bool { return !i64_lt(b, a); }
fn i64_gt(a: vec2<u32>, b: vec2<u32>) -> bool { return i64_lt(b, a); }
fn i64_ge(a: vec2<u32>, b: vec2<u32>) -> bool { return !i64_lt(a, b); }
fn i64_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
    let s = n & 63u;
    if (s == 0u) { return a; }
    if (s >= 32u) { return vec2<u32>(0u, a.x << (s - 32u)); }
    return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}
fn u64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    let s = n & 63u;
    if (s == 0u) { return a; }
    if (s >= 32u) { return vec2<u32>(a.y >> (s - 32u), 0u); }
    return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}
fn i64_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
    let s = n & 63u;
    let hi = bitcast<i32>(a.y);
    if (s == 0u) { return a; }
    if (s >= 32u) { return vec2<u32>(bitcast<u32>(hi >> (s - 32u)), bitcast<u32>(hi >> 31u)); }
    return vec2<u32>((a.x >> s) | (a.y << (32u - s)), bitcast<u32>(hi >> s));
}
fn i64_neg(a: vec2<u32>) -> vec2<u32> { return i64_sub(vec2<u32>(0u, 0u), a); }
// Quotient and remainder, by shift and subtract; one 32-bit division when both fit.
fn u64_divrem(a: vec2<u32>, b: vec2<u32>) -> array<vec2<u32>, 2> {
    if (a.y == 0u && b.y == 0u) {
        return array<vec2<u32>, 2>(vec2<u32>(a.x / b.x, 0u), vec2<u32>(a.x % b.x, 0u));
    }
    var q = vec2<u32>(0u, 0u);
    var r = vec2<u32>(0u, 0u);
    for (var i = 63u; ; i = i - 1u) {
        r = i64_shl(r, 1u);
        r.x = r.x | (select(a.x >> i, a.y >> (i - 32u), i >= 32u) & 1u);
        if (u64_ge(r, b)) {
            r = i64_sub(r, b);
            if (i >= 32u) { q.y = q.y | (1u << (i - 32u)); } else { q.x = q.x | (1u << i); }
        }
        if (i == 0u) { break; }
    }
    return array<vec2<u32>, 2>(q, r);
}
fn u64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return u64_divrem(a, b)[0]; }
fn u64_rem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return u64_divrem(a, b)[1]; }
fn i64_abs(a: vec2<u32>) -> vec2<u32> { return select(a, i64_neg(a), bitcast<i32>(a.y) < 0); }
fn i64_smin(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, i64_lt(a, b)); }
fn i64_smax(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, i64_gt(a, b)); }
fn u64_min(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, u64_lt(a, b)); }
fn u64_max(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(b, a, u64_gt(a, b)); }
fn i64_sclamp(x: vec2<u32>, lo: vec2<u32>, hi: vec2<u32>) -> vec2<u32> { return i64_smin(i64_smax(x, lo), hi); }
fn u64_clamp(x: vec2<u32>, lo: vec2<u32>, hi: vec2<u32>) -> vec2<u32> { return u64_min(u64_max(x, lo), hi); }
fn i64_div(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let q = u64_div(i64_abs(a), i64_abs(b));
    return select(q, i64_neg(q), (bitcast<i32>(a.y) < 0) != (bitcast<i32>(b.y) < 0));
}
fn i64_rem(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let r = u64_rem(i64_abs(a), i64_abs(b));
    return select(r, i64_neg(r), bitcast<i32>(a.y) < 0);
}
fn i64_from_i32(x: u32) -> vec2<u32> { return vec2<u32>(x, select(0u, 0xffffffffu, bitcast<i32>(x) < 0)); }
fn u64_from_f32(x: f32) -> vec2<u32> {
    let high = floor(x * (1.0f / 4294967296.0f));
    let low = x - high * 4294967296.0f;
    return vec2<u32>(u32(low), u32(high));
}
fn i64_from_f32(x: f32) -> vec2<u32> {
    let magnitude = u64_from_f32(abs(x));
    return select(magnitude, i64_neg(magnitude), x < 0.0f);
}
fn i64_to_f32(a: vec2<u32>) -> f32 {
    if (bitcast<i32>(a.y) < 0) { let n = i64_neg(a); return -(f32(n.y) * 4294967296.0 + f32(n.x)); }
    return f32(a.y) * 4294967296.0 + f32(a.x);
}
"""
