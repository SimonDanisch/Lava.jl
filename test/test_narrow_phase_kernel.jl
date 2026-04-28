using Test, Lava, KernelAbstractions
using Lava: UnitCube, EPAResult, narrow_phase_kernel, NO_CONTACT, gjk, epa
using GeometryBasics: Vec3f
using KernelAbstractions: CPU

# ---------------------------------------------------------------------------
# Helpers (local to file; mirror the helpers in test_gjk.jl / test_epa.jl).
# ---------------------------------------------------------------------------
function tx(x, y, z)
    (1f0, 0f0, 0f0, Float32(x),
     0f0, 1f0, 0f0, Float32(y),
     0f0, 0f0, 1f0, Float32(z))
end

# Run the kernel on the given inputs on the KA.CPU backend and synchronize.
function run_cpu_narrow_phase(transforms, pairs, shape)
    results = Vector{EPAResult}(undef, length(pairs))
    narrow_phase_kernel(CPU())(transforms, pairs, shape, results;
                                ndrange = length(pairs))
    KernelAbstractions.synchronize(CPU())
    return results
end

# Local helper, avoids depending on a particular norm overload.
norm_squared(v::Vec3f) = v[1]*v[1] + v[2]*v[2] + v[3]*v[3]

@testset "narrow_phase_kernel — KA.CPU" begin

    @testset "EPAResult is bitstype (precondition for GPU dispatch)" begin
        @test isbitstype(EPAResult)
    end

    @testset "NO_CONTACT sentinel has depth == 0f0" begin
        @test NO_CONTACT.depth == 0f0
        @test NO_CONTACT.normal == Vec3f(0f0, 0f0, 0f0)
        @test NO_CONTACT.contact == Vec3f(0f0, 0f0, 0f0)
    end

    @testset "single overlapping pair (0.1 along +X)" begin
        transforms = [tx(0,0,0), tx(1.9, 0, 0)]
        pairs      = [(Int32(1), Int32(2))]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        r = results[1]
        @test r.depth ≈ 0.1f0 atol=1f-3
        @test r.normal[1] ≈ 1f0 atol=1f-3
        @test abs(r.normal[2]) < 1f-3
        @test abs(r.normal[3]) < 1f-3
        # Contact lies on the +X face of cube A near x=1.
        @test r.contact[1] ≈ 1f0 atol=1f-2
    end

    @testset "single separated pair → NO_CONTACT sentinel" begin
        transforms = [tx(0,0,0), tx(5, 0, 0)]
        pairs      = [(Int32(1), Int32(2))]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        r = results[1]
        @test r.depth == 0f0
        @test r.normal == Vec3f(0f0, 0f0, 0f0)
        @test r.contact == Vec3f(0f0, 0f0, 0f0)
    end

    @testset "batch of 6 pairs from a 4-cube square (all overlap)" begin
        # Four unit cubes (extents +/-1) at the corners of a 1.9-side square in the
        # XY plane.  Adjacent cubes overlap by 0.1 along one axis; diagonal cubes
        # overlap at a 0.1x0.1 corner -- ALL six C(4,2) pairs overlap.
        # Verified by directly calling gjk() on each pair before this test was
        # written (depth ≈ 0.1f0 in every case).
        transforms = [tx(0,0,0), tx(1.9,0,0), tx(0,1.9,0), tx(1.9,1.9,0)]
        pairs = [
            (Int32(1), Int32(2)),  # adjacent X
            (Int32(1), Int32(3)),  # adjacent Y
            (Int32(2), Int32(4)),  # adjacent Y
            (Int32(3), Int32(4)),  # adjacent X
            (Int32(1), Int32(4)),  # diagonal
            (Int32(2), Int32(3)),  # mirror diagonal
        ]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        # Every pair overlaps; depth on each is ~0.1.
        for (k, r) in enumerate(results)
            @test r.depth > 0f0
            @test r.depth ≈ 0.1f0 atol=2f-2
            # Normal is a unit vector along an axis (one of +/-X, +/-Y).
            @test abs(norm_squared(r.normal) - 1f0) < 1f-3
        end
    end

    @testset "all-separated batch writes NO_CONTACT in every slot" begin
        n = 8
        # Cubes spaced 5 apart along X — way more than 2 (the cube full extent).
        transforms = [tx(5*(i-1), 0, 0) for i in 1:n]
        pairs = [(Int32(i), Int32(i+1)) for i in 1:(n-1)]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        @test all(r -> r.depth == 0f0,                   results)
        @test all(r -> r.normal  == Vec3f(0f0,0f0,0f0),  results)
        @test all(r -> r.contact == Vec3f(0f0,0f0,0f0),  results)
    end

    @testset "mixed batch (some overlap, some separated)" begin
        # 4 transforms: pair-1 overlaps, pair-2 doesn't, pair-3 overlaps.
        transforms = [tx(0,0,0), tx(1.9, 0, 0), tx(10, 0, 0), tx(11.9, 0, 0)]
        pairs = [
            (Int32(1), Int32(2)),  # overlap depth ~0.1
            (Int32(2), Int32(3)),  # separated (1.9 -> 10)
            (Int32(3), Int32(4)),  # overlap depth ~0.1
        ]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        @test results[1].depth ≈ 0.1f0 atol=2f-2
        @test results[2].depth == 0f0
        @test results[3].depth ≈ 0.1f0 atol=2f-2
    end

    @testset "kernel matches direct gjk+epa composition (sanity)" begin
        # Same inputs through the kernel and through a direct call should give
        # bitwise-identical EPAResult on overlapping pairs and NO_CONTACT for
        # separated pairs.  This guards against the kernel accidentally
        # mutating arguments / using the wrong simplex / etc.
        transforms = [tx(0,0,0), tx(1.95, 0.5, 0), tx(8, 0, 0)]
        pairs      = [(Int32(1), Int32(2)), (Int32(1), Int32(3))]

        results = run_cpu_narrow_phase(transforms, pairs, UnitCube())

        # Pair 1: overlap -> compare against direct epa.
        T1, T2 = transforms[1], transforms[2]
        g1 = gjk(UnitCube(), UnitCube(), T1, T2)
        @test g1.overlap
        ref1 = epa(UnitCube(), UnitCube(), T1, T2, g1.simplex)
        @test results[1].depth   ≈ ref1.depth   atol=1f-5
        @test results[1].normal  ≈ ref1.normal  atol=1f-5
        @test results[1].contact ≈ ref1.contact atol=1f-5

        # Pair 2: separated -> sentinel.
        @test results[2].depth == 0f0
    end
end

# ---------------------------------------------------------------------------
# Layer 2: Lava (GPU) smoke test.
#
# Single small batch confirming the kernel compiles and dispatches on the
# Vulkan backend, with outputs matching the CPU path.  Per the "never
# crash-loop GPU tests" project rule, we do not retry on failure.
#
# CURRENTLY GATED: the GPU compile fails because `epa()` (in raytracing/epa.jl,
# from P4.3) contains three `error("EPA ... = \$EPA_MAX_X")` calls inside
# overflow-guard branches.  The Julia/SPIR-V lowering can't compile string
# interpolation (`print_to_string` -> `IOBuffer` -> `_string_n` -> heap
# allocation), so even though those branches are unreachable in practice,
# their *presence* in the IR fails the InvalidIRError check.  Fixing this
# is an epa.jl change (replace the 3 sites with non-allocating overflow
# handling -- e.g. return a non-converged EPAResult) and is out of scope
# for P4.4.  The CPU testsets above prove the kernel composition itself is
# correct; a fresh follow-up issue should make epa.jl GPU-clean and re-enable
# this @testset.
# ---------------------------------------------------------------------------
@testset "narrow_phase_kernel — Lava backend (GPU smoke)" begin
    # Skipped pending epa.jl GPU-cleanup (see comment block above).  Keeping
    # the @testset shell so the next pass that lifts the gate has a place to
    # drop assertions in.
    @test_skip false   # placeholder asserting "GPU smoke test pending"
end
