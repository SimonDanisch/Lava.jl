"""
A workgroup above 256 silently runs fewer invocations than it declares.

The device accepts `LocalSize 512 1 1`, reports no resource problem, and then
executes only local invocations 1–256. Lava computes the grid from the size it
asked for, so the launch comes back with three quarters or half of its output
never written — and a kernel that skips work looks like a speed-up, which is how
this was found (a "3.4x" permuted copy that had written a quarter of its
destination).

`Lava.WORKGROUP_LIMIT` turns that into a throw. These tests pin both halves: the
limit rejects the size, and lifting the limit reproduces the truncation, so if a
driver update fixes it the second testset fails and says so.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

# 64 simultaneously live values. The count matters and is not a round number by
# accident: 32 stays in registers and is fine at every size, 128 spills and is
# also fine, and 64 is the shape that truncates — with the *same* driver-reported
# Register Count (40) as the 128 case, which is why no statistic can predict it.
@kernel cpu=false function wglimit_probe!(out, ::Val{K}) where {K}
    I = @index(Global, Linear)
    acc = ntuple(k -> Int32(I) * Int32(k) + Int32(k * k), Val(K))
    s = zero(Int32)
    for k in 1:K
        s = s ⊻ acc[k]
    end
    @inbounds out[I] = Float16((s & Int32(3)) + Int32(1))   # never zero
end

"Fraction of a single workgroup's output that actually got written."
function groupcoverage(backend, K, wg)
    out = KA.allocate(backend, Float16, wg)
    fill!(out, zero(Float16))
    wglimit_probe!(backend)(out, Val(K); ndrange = wg, workgroupsize = wg)
    KA.synchronize(backend)
    count(!=(zero(Float16)), Array(out)) / wg
end

@testset "workgroup limit" begin
    backend = LavaBackend()

    @testset "the limit rejects what the device will not honour" begin
        @test Lava.WORKGROUP_LIMIT[] == 256
        out = KA.allocate(backend, Float16, 512)
        @test_throws ArgumentError wglimit_probe!(backend)(out, Val(64);
                                                          ndrange = 512, workgroupsize = 512)
        # At or below the limit every lane runs, whatever the body costs.
        for K in (32, 64, 128), wg in (64, 128, 256)
            @test groupcoverage(backend, K, wg) == 1.0
        end
    end

    @testset "above it, invocations go missing (driver behaviour, pinned)" begin
        old = Lava.WORKGROUP_LIMIT[]
        try
            Lava.WORKGROUP_LIMIT[] = 1024
            # Exactly 256 lanes participate regardless of the size asked for, so
            # the coverage is 256/wg. If a driver update fixes this, these two
            # become 1.0 and the limit above can be raised.
            @test groupcoverage(backend, 64, 512) == 0.5
            @test groupcoverage(backend, 64, 1024) == 0.25
            # Not every body triggers it — which is the reason the limit is a
            # flat cap rather than a per-kernel prediction.
            @test groupcoverage(backend, 128, 512) == 1.0
        finally
            Lava.WORKGROUP_LIMIT[] = old
        end
    end
end
