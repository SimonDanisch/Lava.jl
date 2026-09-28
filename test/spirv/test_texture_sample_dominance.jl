# Tier 1: a texture sample is reused only where it dominates.
#
# `tex[uv]` is four `_lava_gfx_sample_2d` calls, one per component, and the
# emitter answers them with ONE `OpImageSampleImplicitLod` by caching the sample
# under its (binding, u, v). The cache was module-wide, so a later sample of the
# same texture at the same coordinates reused the first one wherever it was: in
# a branch that does not dominate it, or in another function. spirv-val refused
# the module ("does not dominate its use"), which is how RayMakie's textured
# mesh shader failed to build. The reuse is for the calls of one `tex[uv]`, which
# are in one block, so the cache is per block.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm
using GeometryBasics

function tsd_frag()
    tex = Lava.GfxTexture2D(UInt32(0))
    uv = Vec2f(KernelInterface.frag_coord_x() / 64f0, KernelInterface.frag_coord_y() / 64f0)
    c = Vec4f(0f0, 0f0, 0f0, 1f0)
    if uv[1] > 0.5f0
        c = tex[uv]
    end
    # The same texture at the same coordinates, after the branch: sampled again,
    # not read out of a block that may not have run.
    d = tex[uv]
    Lava.gfx_output(0, c + d)
    return nothing
end

@testset "a texture sample is reused only where it dominates" begin
    # `validate = true` is the assertion that matters: spirv-val checks dominance.
    d, _ = compile_and_disasm(tsd_frag, Tuple{}; stage = :fragment, validate = true)
    # One sample per `tex[uv]`, not one per component and not one for both.
    @test count("OpImageSampleImplicitLod", d) == 2
end
