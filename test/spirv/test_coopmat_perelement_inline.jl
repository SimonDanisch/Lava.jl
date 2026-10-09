# Tier 1: a per-element callback compiles INLINABLE.
#
# `KernelInterface.coopmat_perelement` lowers to `OpCooperativeMatrixPerElementOpNV`,
# which names a function: `coopmat_perelement_thunk`, into which the user's callback
# must melt. A callback that stays out of line is a separate `OpFunction` with
# `DontInline` control, the driver honours that and calls it once per element —
# measured at 8.5x, 0.756 ms against 0.082 for 500 rescales of one tile across 4096
# workgroups. A performance regression with no correctness symptom at all, so it is
# asserted on the disassembly. Moved here from Mantle's suite, which runs the
# operation itself on devices that have it.

using Test
import KernelInterface
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

# Depends on both indices, and is not symmetric under swapping them.
perelement_rowcol(row::UInt32, col::UInt32, e::Float32) = e * Float32(row + 1) + Float32(col)

@testset "a per-element callback is inlinable, not DontInline" begin
    function peplain(out, inp)
        m = Lava.AcceleratedMatrix{Float32,16,16,Lava.Accumulator}(pointer(inp), 1, 16)
        copyto!(pointer(out), 1, 16, KernelInterface.coopmat_perelement(perelement_rowcol, m))
        return
    end
    d = first(compile_and_disasm(peplain,
            Tuple{Lava.LavaDeviceArray{Float32,2}, Lava.LavaDeviceArray{Float32,2}}))
    @test occursin("OpCooperativeMatrixPerElementOpNV", d)
    # Exactly one function carries the callback's signature, and it is the thunk,
    # marked `Inline`.
    @test occursin(r"OpFunction %float Inline", d)
    @test !occursin(r"OpFunction %float DontInline", d)
    # And nothing calls out of it per element.
    @test count("OpFunctionCall", d) == 0
end
