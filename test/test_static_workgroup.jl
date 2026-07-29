"""
A workgroup passed in the kernel's TYPE computes wrong global indices when it has
an INTERIOR unit extent — and `Lava.WORKGROUP_FALLBACK` now catches that.

The fault: with such a workgroup baked into the kernel's type, the block index
decode takes the wrong divisor for dimension 2 and exactly
`min(1, blocks[3] / blocks[2])` of the output is written, silently. A *trailing*
unit extent is harmless, which is why ranks 1-3 look clean. It is Lava's codegen,
not KernelAbstractions': the same kernel, workgroup and ndrange are correct on
`KA.CPU()`, which runs the same `expand`/`NDRange` machinery.

`(obj::KA.Kernel{LavaBackend})` now detects the shape and re-launches through the
dynamic path, which computes correct indices. That costs ~2x on the affected
kernels (the index arithmetic stops being compile-time constant) and cannot
affect any launch that was already right. This file checks BOTH halves: that the
guard fixes it, and — with the guard switched off — that the underlying fault is
still there, so nobody deletes a guard that is still load-bearing.

Superseded framing, kept because the API rule still holds:

    kernel(backend, wg)(args...; ndrange)                      # WRONG at rank 4
    kernel(backend)(args...; ndrange, workgroupsize = wg)      # correct
    kernel(backend, wg, ndrange)(args...)                      # correct

Found because `permutedims!` used the first spelling: `ndrange = (72, 256, 8, 16)`
with `wg = (32, 4, 1, 1)` left 2 064 384 of 2 359 296 destination elements never
written, with no error anywhere. Every group does run — the per-lane global
indices simply collide, so 8 groups land on the same region and the rest of the
array is never visited.

The failing configuration is the only one whose iteration space pairs a real
`blocks` field with a zero-size `workitems::Nothing`; both-static has two
`Nothing`s and the keyword form has two real fields, and both are correct. Ranks
1-3 and rank 5 are also correct, which is what makes this easy to walk past.

Kept as a test rather than a comment so that whoever fixes the layout finds out.
Until then the rule is: pass workgroup sizes as the launch keyword.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function wgmark!(d)
    I = @index(Global, NTuple)
    @inbounds d[I...] = 1.0f0
end

"Fraction of `ndrange` actually written, launching `mode`."
function coverage(nd, wg, mode; fallback::Bool = true)
    old = Lava.WORKGROUP_FALLBACK[]
    Lava.WORKGROUP_FALLBACK[] = fallback
    try
        return coverage_(nd, wg, mode)
    finally
        Lava.WORKGROUP_FALLBACK[] = old
    end
end

function coverage_(nd, wg, mode)
    back = LavaBackend()
    d = KA.allocate(back, Float32, nd...)
    fill!(d, 0.0f0)
    if mode === :typed
        wgmark!(back, wg)(d; ndrange = nd)
    elseif mode === :keyword
        wgmark!(back)(d; ndrange = nd, workgroupsize = wg)
    else
        wgmark!(back, wg, nd)(d)
    end
    KA.synchronize(back)
    n = count(==(1.0f0), Array(d))
    d = nothing
    GC.gc()
    return n / prod(nd)
end

@testset "workgroup size: launch keyword vs kernel type" begin
    @testset "the keyword form is correct at every rank" begin
        for (nd, wg) in (((1000,), (64,)),
                         ((250, 250), (32, 4)),
                         ((72, 64, 64), (32, 4, 1)),
                         ((72, 256, 8, 16), (32, 4, 1, 1)),   # rank 4: the failing shape
                         ((72, 8, 1, 1), (32, 4, 1, 1)),
                         ((8, 8, 8, 8, 8), (8, 1, 1, 1, 1)))
            @test coverage(nd, wg, :keyword) == 1.0
        end
    end

    @testset "both-static is correct too" begin
        @test coverage((72, 256, 8, 16), (32, 4, 1, 1), :bothstatic) == 1.0
    end

    @testset "the typed form is correct only outside rank 4" begin
        @test coverage((1000,), (64,), :typed) == 1.0
        @test coverage((250, 250), (32, 4), :typed) == 1.0
        @test coverage((72, 64, 64), (32, 4, 1), :typed) == 1.0
        @test coverage((8, 8, 8, 8, 8), (8, 1, 1, 1, 1), :typed) == 1.0
        # Documented failure. Flip these to `== 1.0` when the layout is fixed.
        @test coverage((72, 256, 8, 16), (32, 4, 1, 1), :typed; fallback = false) < 1.0
        @test coverage((72, 8, 1, 1), (32, 4, 1, 1), :typed; fallback = false) < 1.0
    end

    @testset "the trigger is workitems[3] == 1" begin
        # Sharpest form of the bug. At rank 4 the typed spelling is wrong IFF the
        # THIRD workgroup extent is 1; `wg[4]` is irrelevant. With `wg[3] > 1`
        # every block grid is correct, so a unit interior workitem extent is what
        # breaks the block decode — consistent with LLVM folding away that
        # dimension's `(g-1)*1 + 1` term and taking dimension 2's divisor with it.
        ND = (64, 256, 8, 2)
        for wg in ((32, 4, 2, 1), (32, 4, 4, 1))
            @test coverage(ND, wg, :typed; fallback = false) == 1.0   # wg[3] > 1: fine
        end
        for wg in ((32, 4, 1, 1), (32, 4, 1, 2), (32, 1, 1, 1), (32, 8, 1, 1))
            b = cld.(ND, wg)
            @test coverage(ND, wg, :typed; fallback = false) ≈ min(1.0, b[3] / b[2])
        end
    end

    @testset "an INTERIOR unit extent is the trigger, a trailing one is harmless" begin
        # Generalises past rank 4 and explains the whole pattern: ranks 1-3 look
        # clean because `(32,4,1)`'s unit extent is trailing, and the rank-5 case
        # that passed earlier did so because it satisfied b2 <= b3, not because
        # rank 5 is immune.
        ND5 = (32, 128, 8, 4, 2)
        @test coverage(ND5, (16, 4, 1, 1, 1), :typed; fallback = false) < 1.0   # dims 3,4 unit, interior
        @test coverage(ND5, (16, 4, 2, 1, 1), :typed; fallback = false) < 1.0   # dim 4 unit, interior
        @test coverage(ND5, (16, 4, 2, 2, 1), :typed) == 1.0  # only dim 5, trailing
    end

    @testset "the failure obeys an exact law: written = min(1, b3/b2)" begin
        # Not decoration — this is the sharpest statement of the bug and the
        # thing a fix has to explain. Dimension 2 of the BLOCK grid is decoded
        # with dimension 3's extent as its divisor, so its component ranges over
        # `0:b3-1` instead of `0:b2-1` and exactly `b3/b2` of the output is
        # written. Verified over a 6x5 grid; a few representative cells here.
        law(b2, b3) = coverage((64, 4b2, b3, 2), (32, 4, 1, 1), :typed; fallback = false)
        for (b2, b3) in ((2, 1), (4, 2), (8, 1), (16, 4), (64, 8))
            @test law(b2, b3) ≈ min(1.0, b3 / b2)
        end
        # …and it is correct exactly when b2 <= b3.
        for (b2, b3) in ((2, 2), (4, 4), (8, 16), (2, 8))
            @test law(b2, b3) == 1.0
        end
    end

    @testset "WORKGROUP_FALLBACK makes the typed form correct everywhere" begin
        # The guard is the point of all the above. With it on, every shape that
        # used to be silently wrong is right, and nothing that worked changes.
        for (nd, wg) in (((64, 256, 8, 2), (32, 4, 1, 1)), ((64, 256, 8, 2), (32, 4, 1, 2)),
                         ((64, 256, 8, 2), (32, 1, 1, 1)), ((72, 256, 8, 16), (32, 4, 1, 1)),
                         ((72, 8, 1, 1), (32, 4, 1, 1)), ((32, 128, 8, 4, 2), (16, 4, 1, 1, 1)),
                         ((64, 256, 8, 2), (32, 4, 2, 1)), ((250, 250), (32, 4)),
                         ((72, 64, 64), (32, 4, 1)), ((1000,), (64,)))
            @test coverage(nd, wg, :typed) == 1.0
        end
    end

    @testset "the guard names exactly the dangerous shapes" begin
        @test Lava.interior_unit_workgroup((32, 4, 1, 1))
        @test Lava.interior_unit_workgroup((32, 4, 1, 2))
        @test Lava.interior_unit_workgroup((16, 4, 2, 1, 1))
        @test !Lava.interior_unit_workgroup((32, 4, 2, 1))     # trailing only
        @test !Lava.interior_unit_workgroup((32, 4, 1))        # trailing only
        @test !Lava.interior_unit_workgroup((32, 4))
        @test !Lava.interior_unit_workgroup((64,))
    end

    @testset "launchgroup fills the fast axis first" begin
        @test Lava.launchgroup((4, 4, 288, 1024)) == (4, 4, 16, 1)
        @test Lava.launchgroup((4096, 72, 8, 1)) == (256, 1, 1, 1)
        @test Lava.launchgroup((16, 72, 4, 1024)) == (16, 16, 1, 1)
        @test Lava.launchgroup((10,)) == (10,)
        for sz in ((4, 4, 288, 1024), (4096, 72, 8, 1), (16, 72, 4, 1024), (2, 2, 576, 1024))
            wg = Lava.launchgroup(sz)
            @test all(wg .<= sz)                    # never larger than the axis
            @test prod(wg) <= 256
        end
    end

    @testset "no shaping rule may exceed the thread budget" begin
        # The one invariant that is not a performance preference: a workgroup
        # larger than `maxComputeWorkGroupInvocations` does not run slowly, it
        # does not complete. `staticgroup` used to give every interior axis 2
        # threads unconditionally — `2^(N-2)`, i.e. 65 536 at rank 18 — and
        # GPUArrays' 18-d `permutedims` hung for the full 120 s flush timeout,
        # failing the 354 assertions that came after it.
        shapes = Any[]
        for n in 1:20
            push!(shapes, ntuple(d -> d == 1 ? 4 : 2, n))       # the hanging shape
            push!(shapes, ntuple(_ -> 2, n))
            push!(shapes, ntuple(d -> d == n ? 1024 : 3, n))
            push!(shapes, ntuple(d -> isodd(d) ? 1 : 64, n))
        end
        for sz in shapes, f in (Lava.launchgroup, Lava.staticgroup)
            wg = f(sz)
            @test prod(wg) <= 256
            @test all(wg .>= 1)
            @test all(wg .<= sz)
        end
    end

    @testset "high rank stays correct via the dynamic fallback" begin
        # Above the rank where 2-per-interior-axis fits, `staticgroup` has to
        # leave interior extents at 1 — which is precisely the shape
        # `WORKGROUP_FALLBACK` re-launches dynamically, so the result is still
        # right. Asserted here so the two rules stay aware of each other.
        for n in 12:18
            sz = ntuple(d -> d == 1 ? 4 : 2, n)
            wg = Lava.staticgroup(sz)
            @test prod(wg) <= 256
            @test Lava.interior_unit_workgroup(wg)   # so the fallback fires
        end
    end
end
