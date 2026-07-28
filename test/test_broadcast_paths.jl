# Every path through `GPUArrays._copyto!(::AnyLavaArray, ::Broadcasted)`.
#
# Lava overrides that method to launch over a flat range and, when every operand
# already has the destination's shape, to flatten the expression tree as well
# (see `array/gpuarrays.jl` — worth 6x on elementwise bandwidth). Three kernels
# come out of that decision, and the cases below are one per kernel plus the two
# wrapper shapes that pick the awkward ones.
#
# These exist because narrowing the linear->Cartesian `divrem` to `Int32` — a
# change that is worth 9% of an inference step in a *different* kernel — made
# the view and permuted operands silently return wrong values here (off by 252
# and 18) while the same-shape cases stayed exact. Nothing else in the suite
# caught it.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "broadcast paths" begin
    be = LavaBackend()
    host = reshape(collect(1f0:105f0), 7, 5, 3)
    A = KA.allocate(be, Float32, 7, 5, 3); copyto!(A, host)
    hostb = reshape(collect(1f0:21f0), 7, 1, 3)
    B = KA.allocate(be, Float32, 7, 1, 3); copyto!(B, hostb)

    # flat kernel: every operand the destination's shape
    same = KA.allocate(be, Float32, 7, 5, 3)
    same .= A .* A .+ 1f0
    KA.synchronize(be)
    @test Array(same) ≈ host .^ 2 .+ 1

    # extruded operand: dest dense, one argument broadcast along a dim
    mixed = KA.allocate(be, Float32, 7, 5, 3)
    mixed .= A .+ B .* 2f0
    KA.synchronize(be)
    @test Array(mixed) ≈ host .+ hostb .* 2

    # non-linear operand: a strided view is not linearly indexable
    vout = KA.allocate(be, Float32, 5, 5, 3)
    vout .= view(A, 2:6, :, :) .* 3f0
    KA.synchronize(be)
    @test Array(vout) ≈ host[2:6, :, :] .* 3

    # non-linear *destination*
    dst = KA.allocate(be, Float32, 7, 5, 3); copyto!(dst, host)
    view(dst, 2:6, :, :) .= 7f0
    KA.synchronize(be)
    @test all(Array(dst)[2:6, :, :] .== 7f0)
    @test Array(dst)[1, :, :] == host[1, :, :]

    # permuted operand
    pout = KA.allocate(be, Float32, 5, 7, 3)
    pout .= PermutedDimsArray(A, (2, 1, 3)) .+ 1f0
    KA.synchronize(be)
    @test Array(pout) ≈ permutedims(host, (2, 1, 3)) .+ 1
end
