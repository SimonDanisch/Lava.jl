# Empty pool blocks must be returned to the driver without waiting for an OOM.
#
# `GPU_LIVE_BYTES` tracks pool *capacity*, and blocks used to be handed back only
# on an allocation-failure retry. `maybe_collect`'s gate is a ratio against the
# device heap, so on an iGPU with a large shared heap a few GB of dead blocks is
# only ~20 % and never trips it — the pool ratchets up to the high-water mark of
# the largest workload and stays there. Across a multi-scene run that is
# gigabytes of system RAM the rest of the machine still needs, and it showed up
# as a driver timeout partway through a 5-scene sweep.
#
# `maybe_trim_pool!` adds an absolute-capacity trigger so dead capacity is
# released on its own terms.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "empty pool blocks are trimmed without an OOM" begin
    be = LavaBackend()
    ctx = Lava.vk_context()

    # Grow the pool past the trim threshold, then drop every reference.
    target = Lava.pool(Lava.vk_context()).trim_threshold + 256 * 1024 * 1024
    let arrays = Lava.LavaArray[]
        while Lava.GPU_LIVE_BYTES[] < target
            a = KA.allocate(be, Float32, 4_000_000)   # 16 MB each
            fill!(a, 1.0f0)
            push!(arrays, a)
        end
        KA.synchronize(be)
        empty!(arrays)
    end

    grown = Lava.GPU_LIVE_BYTES[]
    @test grown >= Lava.pool(Lava.vk_context()).trim_threshold

    # Defeat the rate limiter so the test doesn't depend on wall-clock timing.
    Lava.pool(Lava.vk_context()).last_trim = 0.0
    Lava.maybe_trim_pool!(ctx)

    trimmed = Lava.GPU_LIVE_BYTES[]
    @test trimmed < grown                     # capacity actually came back
    @test trimmed < Lava.pool(Lava.vk_context()).trim_threshold

    # And the allocator still works afterwards — blocks were returned, not corrupted.
    b = KA.allocate(be, Float32, 1024)
    fill!(b, 2.0f0)
    KA.synchronize(be)
    @test all(Array(b) .== 2.0f0)
end

@testset "trim is rate-limited" begin
    ctx = Lava.vk_context()
    Lava.pool(Lava.vk_context()).last_trim = time()            # just trimmed
    before = Lava.GPU_LIVE_BYTES[]
    Lava.maybe_trim_pool!(ctx)                # must be a no-op, not a stall
    @test Lava.GPU_LIVE_BYTES[] == before
end
