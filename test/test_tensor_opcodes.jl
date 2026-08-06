using Test, Lava

# `Op` and `Cap` in src/compiler/spirv/module.jl are HAND-MAINTAINED tables. A
# wrong constant there does not fail loudly: it produces a module the driver
# rejects, or worse accepts and misreads, a long way from the typo. So the tensor
# addressing numbers are re-derived here from a shader glslang compiles, which is
# where they came from in the first place — the same instrument the `test/glsl`
# README calls "Rule 0's strongest".
#
# Skips when glslang is absent rather than failing: this guards a transcription,
# and a machine without the compiler simply cannot check it.
"""Opcodes and capabilities actually present in `spv`, by parsing the binary.

Every opcode goes into a `Set` — deliberately not a `result id => opcode` map.
The first operand is the RESULT id for `OpType*` but the TYPE id for
value-producing ops, so keying on it makes the three instructions that share a
`tensorLayoutNV` type overwrite one another and the test silently checks a
third of what it claims to."""
function spirvfacts(spv::Vector{UInt32})
    @assert spv[1] == 0x07230203 "not a SPIR-V module"
    caps, opcodes, i = UInt32[], Set{UInt16}(), 6
    consts = Dict{UInt32,UInt32}()      # OpConstant result id => value
    layoutmodes = UInt32[]              # OpTypeTensorLayoutNV's clamp-mode operand
    while i <= length(spv)
        wc = UInt16(spv[i] >> 16); op = UInt16(spv[i] & 0xffff)
        wc == 0 && break
        ops = spv[(i + 1):(i + wc - 1)]
        push!(opcodes, op)
        op == 17 && push!(caps, ops[1])                        # OpCapability
        op == Lava.Op.OpConstant && (consts[ops[2]] = ops[3])  # %type %res %value
        op == Lava.Op.OpTypeTensorLayoutNV && push!(layoutmodes, ops[3])
        i += wc
    end
    (caps, opcodes, consts, layoutmodes)
end

@testset "tensor addressing opcodes match what glslang emits" begin
    glsl = Sys.which("glslang")
    glsl === nothing && (glsl = Sys.which("glslangValidator"))
    src = joinpath(@__DIR__, "glsl", "tensor_addressing_opcodes.comp")
    if glsl === nothing
        @info "glslang not found — skipping the opcode differential" src
    else
        out = tempname() * ".spv"
        p = run(pipeline(`$glsl -V --target-env vulkan1.3 -S comp $src -o $out`;
                         stdout = devnull, stderr = devnull); wait = false)
        wait(p)
        @test success(p)
        spv = reinterpret(UInt32, read(out))
        caps, opcodes, consts, layoutmodes = spirvfacts(Vector{UInt32}(spv))

        # Both capabilities, from their two different extensions. Requiring only
        # the coopmat2 one produces a module the driver rejects.
        @test Lava.Cap.TensorAddressingNV in caps
        @test Lava.Cap.CooperativeMatrixTensorAddressingNV in caps

        for (name, val) in (("OpTypeTensorLayoutNV", Lava.Op.OpTypeTensorLayoutNV),
                            ("OpTypeTensorViewNV", Lava.Op.OpTypeTensorViewNV),
                            ("OpCreateTensorLayoutNV", Lava.Op.OpCreateTensorLayoutNV),
                            ("OpTensorLayoutSetDimensionNV", Lava.Op.OpTensorLayoutSetDimensionNV),
                            ("OpTensorLayoutSliceNV", Lava.Op.OpTensorLayoutSliceNV),
                            ("OpCreateTensorViewNV", Lava.Op.OpCreateTensorViewNV),
                            ("OpCooperativeMatrixLoadTensorNV", Lava.Op.OpCooperativeMatrixLoadTensorNV))
            @test (name, val in opcodes) == (name, true)
        end

        # Bool constants are their own opcodes. The shader declares a view with
        # HasDimensions both true and false, so both must appear.
        @test Lava.Op.OpConstantTrue in opcodes
        @test Lava.Op.OpConstantFalse in opcodes

        # The clamp modes, by VALUE and not just by opcode — this is the part
        # that decides whether an unpadded extent is legal, so a wrong enum here
        # would silently read out of bounds rather than fail to compile. The
        # shader declares Undefined, Constant and ClampToEdge layouts.
        modes = Set(get(consts, m, typemax(UInt32)) for m in layoutmodes)
        @test Lava.TENSOR_CLAMP_UNDEFINED in modes
        @test Lava.TENSOR_CLAMP_CONSTANT in modes
        @test Lava.TENSOR_CLAMP_TO_EDGE in modes
        rm(out; force = true)
    end
end
