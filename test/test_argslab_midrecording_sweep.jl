using Test
using Lava
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# Regression: `sweep_retired_batches!` reset the arg/indirect slab cursors
# whenever `in_flight` drained to empty — including when called
# opportunistically from the ALLOCATION path in the middle of recording a
# batch. The active batch's already-recorded dispatches keep their packed
# args at slab offsets below the cursor; after the reset, the next
# dispatch's args overwrite them, so the earlier dispatches read garbage
# arg buffers (wrong buffer pointers) when the batch submits.
#
# Latent because `in_flight` only drains mid-recording when the GPU runs
# ahead of the host. First surfaced through Hikari's volpath bounce-loop
# early-exit (2026-06-10): the mid-loop synchronize let every subsequent
# sample's first trace dispatch read a clobbered queue pointer — whole
# samples rendered black, nondeterministically.
#
# The sequence below reproduces it deterministically:
#   1. drain, then record dispatch A and submit its batch (→ in_flight)
#   2. record dispatch B into the NEW active batch (args at cursor > 0)
#   3. wait (WITHOUT sweeping) until the submitted batch completes
#   4. allocate → opportunistic sweep retires it → in_flight empties →
#      the old code reset the cursors HERE, with B already recorded
#   5. record dispatch C — its args land at offset 0, clobbering B's
#   6. submit + wait: B's kernel then ran with C's argument buffer

@kernel function fill_value_kernel!(arr, v::Int32)
    i = @index(Global)
    @inbounds arr[i] = v
end

# Heavy variant: keeps the GPU busy for tens of milliseconds so the submitted
# batch is still in flight while the host records the next dispatches — that
# in-flight-ness is what holds the hazard window open (see sequence below).
@kernel function heavy_fill_kernel!(arr, v::Int32)
    i = @index(Global)
    acc = Float32(i)
    for _ in 1:20_000
        acc = muladd(acc, 1.0000001f0, 0.5f0)
    end
    @inbounds arr[i] = v + (acc > 0f0 ? Int32(0) : Int32(1))
end

@testset "arg-slab pool reset must not fire mid-recording" begin
    backend = Lava.LavaBackend()
    bq = backend.dispatch_bq
    n = 4096

    a = KA.allocate(backend, Int32, n); KA.fill!(a, Int32(0))
    b = KA.allocate(backend, Int32, n); KA.fill!(b, Int32(0))
    c = KA.allocate(backend, Int32, n); KA.fill!(c, Int32(0))
    KA.synchronize(backend)

    k! = fill_value_kernel!(backend, 256)
    heavy! = heavy_fill_kernel!(backend, 256)

    # 1. Dispatch A (slow — keeps the GPU busy), then force-submit its batch
    #    so it lands in in_flight and STAYS there while we record below.
    heavy!(a, Int32(1); ndrange=n)
    target = bq.active_batch.signal_value
    Lava.submit!(bq)

    # 2. Dispatch B into the fresh active batch. (ensure_active_batch!'s own
    #    sweep may run before the submitted batch completes; the deterministic
    #    hazard window is AFTER this recording.)
    k!(b, Int32(2); ndrange=n)

    # 3. Wait for the submitted batch to complete WITHOUT sweeping, so it
    #    sits completed-but-unreclaimed in in_flight.
    deadline = time() + 10.0
    while Lava.query_timeline(bq) < target
        time() > deadline && error("timeout waiting for submitted batch")
        sleep(0.001)
    end

    # 4. Allocation → vk_alloc/pool_alloc → sweep_retired_batches! retires
    #    the completed batch → in_flight empties. The buggy reset fired here
    #    even though the active batch already holds dispatch B. (Must be a
    #    real allocation — ≤64 B takes the unified fast path without a sweep.)
    big = KA.allocate(backend, Int32, 1 << 20)

    # 5. Two more dispatches. With the buggy cursor reset, C's args repack at
    #    slab offset 0 and D's args land exactly on B's slot — so when the
    #    batch finally submits, B's kernel reads D's argument buffer (writes
    #    D's value into D's array; `b` never gets written).
    k!(c, Int32(3); ndrange=n)
    d = KA.allocate(backend, Int32, n); KA.fill!(d, Int32(0))
    k!(d, Int32(4); ndrange=n)

    # 6. Submit everything and verify each dispatch wrote ITS OWN array.
    KA.synchronize(backend)
    @test all(==(1), Array(a))
    @test all(==(2), Array(b))   # fails pre-fix: B ran with D's args
    @test all(==(3), Array(c))
    @test all(==(4), Array(d))
end
