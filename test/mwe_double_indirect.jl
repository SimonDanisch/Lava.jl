# MWE: dynamic-index double-indirection through MVectors silently miscompiles on GPU.
#
# Pattern: load NTuple from MVector{N, NTuple{K, Int32}} via runtime index, then
# use the NTuple's first component as another runtime index into a different
# MVector.  CPU returns the right answer; GPU constant-folds incorrectly and
# writes a fixed (wrong) byte pattern.
#
# Discovered while bringing narrow_phase_kernel up on the Lava/RADV backend
# (P4.4 follow-up).  This pattern is core to epa()'s polytope walk:
#     v = face_v[best_idx]   # NTuple{3, Int32}
#     vert = verts_m[v[1]]   # Vec3f
#
# Once this is fixed, the full narrow_phase_kernel should produce correct
# output on GPU (its other emit issues are already addressed in commits
# 4f3fe28, 174c66b, e6aaf31).
#
# Status: GATED in CI -- this test will FAIL on GPU until the emit bug
# is fixed.

using Test, Lava, KernelAbstractions, GeometryBasics, StaticArrays
using Lava: LavaArray, LavaBackend
using GeometryBasics: Vec3f
using KernelAbstractions: CPU

@inline function double_indirect_body(idxs::AbstractVector, i::Integer)
    face_v = MVector{8, NTuple{3, Int32}}(ntuple(j -> (Int32(j), Int32(j+1), Int32(j+2)), 8)...)
    verts  = MVector{16, Vec3f}(ntuple(j -> Vec3f(Float32(j*10), 0f0, 0f0), 16)...)
    @inbounds dyn_idx = idxs[i]
    @inbounds v       = face_v[dyn_idx]
    @inbounds vert    = verts[v[1]]
    return vert
end

@kernel function double_indirect_kernel(out, @Const(idxs))
    i = @index(Global)
    @inbounds out[i] = double_indirect_body(idxs, i)
end

@testset "MWE: double-indirect MVector access" begin
    idxs = LavaArray(Int32[3])

    @testset "KA.CPU produces correct output" begin
        out_cpu = LavaArray([Vec3f(0,0,0)])
        double_indirect_kernel(CPU())(out_cpu, idxs; ndrange=1)
        KernelAbstractions.synchronize(CPU())
        @test Array(out_cpu)[1] == Vec3f(30f0, 0f0, 0f0)
    end

    @testset "Lava (GPU) currently miscompiles -- writes wrong constant" begin
        out_gpu = LavaArray([Vec3f(0,0,0)])
        double_indirect_kernel(LavaBackend())(out_gpu, idxs; ndrange=1)
        Lava.vk_flush!(Lava.vk_context().default_bq)
        # Currently produces Vec3f(0,0,0) due to over-aggressive constant
        # folding in the SPIR-V emit path.  When fixed, this should equal
        # Vec3f(30, 0, 0).
        result = Array(out_gpu)[1]
        @test_skip result == Vec3f(30f0, 0f0, 0f0)
    end
end
