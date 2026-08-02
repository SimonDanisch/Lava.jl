"""
A backend knows which device it runs on.

This is the one field the multi-device work rests on, and it did not exist. Both
project briefs asserted it did — "`LavaBackend(ctx)` pins a context, `BatchQueue`
carries `ctx`" — and neither was true: `LavaBackend(ctx)` kept `ctx.default_bq`
and **discarded `ctx`**, while a `BatchQueue` holds a `Vulkan.Device`, not the
`VkContext` that owns it. So nothing above Lava could ask a backend which device
it meant, and the four module-scope caches that hold device-owned handles
(`PIPELINE_CACHE`, `LINKED_KERNEL_CACHE`, `LAUNCH_PLAN_CACHE`,
`GFX_PIPELINE_CACHE`) had no key to be keyed by.

The asymmetry worth remembering: a **buffer** always knew. `KA.get_backend` has
been reading `a.buf[].ctx` all along. It was only the backend that forgot, which
is why this looked like a design problem and was a one-field omission.

Nothing here needs a second GPU. What it pins is that the *distinction* between
"this device" and "whichever device is current" is expressible and survives a
round trip — which is the part a second device would rely on, and the part that
cannot be added later without touching every call site.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a backend knows its device" begin
    ctx = Lava.vk_context()

    unpinned = LavaBackend()
    pinned   = LavaBackend(ctx)

    # ── Pinned means pinned: the context handed in is the one kept.
    @test getfield(pinned, :ctx) !== nothing
    @test Lava.vk_context(pinned) === ctx

    # ── Unpinned means "whichever is current", and stores nothing, so a
    #    module-level `const BACKEND = LavaBackend()` still survives
    #    `vk_reset_device!()` — the reason the queues resolve lazily too.
    @test getfield(unpinned, :ctx) === nothing
    @test Lava.vk_context(unpinned) === ctx

    # ── A queue-only backend is NOT thereby pinned to a device. Pinning an
    #    upload queue is a statement about scheduling, not about which GPU.
    @test getfield(LavaBackend(ctx.default_bq), :ctx) === nothing
    @test getfield(LavaBackend(ctx.default_bq, ctx.default_bq), :ctx) === nothing

    # ── The round trip that makes it useful: an array knows its device, and the
    #    backend derived from it agrees. This is the path a per-device cache key
    #    would travel.
    a = Lava.LavaArray(zeros(Float32, 8))
    @test Lava.vk_context(a) === ctx
    @test Lava.vk_context(KA.get_backend(a)) === ctx

    # ── And it still works as a backend.
    fill!(a, 2.0f0)
    KA.synchronize(pinned)
    @test Array(a) == fill(2.0f0, 8)
end
