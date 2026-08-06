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
    while i <= length(spv)
        wc = UInt16(spv[i] >> 16); op = UInt16(spv[i] & 0xffff)
        wc == 0 && break
        push!(opcodes, op)
        op == 17 && push!(caps, spv[i + 1])             # OpCapability
        i += wc
    end
    (caps, opcodes)
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
        caps, opcodes = spirvfacts(Vector{UInt32}(spv))

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
        rm(out; force = true)
    end
end
