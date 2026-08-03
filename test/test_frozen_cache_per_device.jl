# The frozen kernel memo is per device, and the profiler can see it.
#
# `FROZEN_MEM` caches `LavaLinkedKernel`, which owns a `VkPipeline` —
# "session-dependent, NOT serializable" by its own field comment. GUARDRAILS §8
# forbids a device-owned handle in a module-level cache, and every other
# pipeline-owning cache was keyed by `ctx.id` in the per-device sweep. This one
# was missed, because `frozen_load` LOOKS like it respects the device: it takes a
# `ctx` and hands it to `link_kernel`. It does that only on a MISS. On a hit it
# returned the first device's pipeline and ignored its `ctx` argument.
#
# Why nothing caught it:
#
#   * `twodevice_probe.jl` runs with no `FROZEN_VERSION`, so `frozen_load`
#     returns at its first line and this cache is never populated at all.
#   * the shipped runners are the exact opposite — they call
#     `use_frozen_kernels` in `__init__`, so for them this is the ONLY kernel
#     cache in play and `LINKED_KERNEL_CACHE` stays empty.
#
# That second point is also a profiling hole, and it is asserted here too:
# `list_compiled_kernels` walked `LINKED_KERNEL_CACHE` only, so on a frozen run
# it reported ZERO kernels while dozens dispatched — taking `kernel_stats`,
# `pipeline_exec_stats` and every register/scratch number with it. Measured on a
# Depth Anything forward before the fix: 0 kernels reported, 45 live dispatch
# names.
#
# No second device is needed to pin the shape: the cache being keyed by `ctx.id`
# is a property of the container, and a second device is simulated by asking for
# the memo under a different id.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function fcpd_scale!(out, @Const(x), s)
    i = @index(Global)
    @inbounds out[i] = x[i] * s
end

@testset "frozen kernel memo is per device" begin
    ctx = Lava.vk_context()

    saved_version = Lava.FROZEN_VERSION[]
    saved_mem = copy(Lava.FROZEN_MEM)
    try
        empty!(Lava.FROZEN_MEM)
        # A version of our own, so this cannot read or write the real cache.
        Lava.FROZEN_VERSION[] = "testpd_" * string(ctx.id; base = 16)
        Lava.FROZEN_RECORDING[] = true

        back = LavaBackend()
        x = Lava.LavaArray(collect(Float32, 1:256))
        out = Lava.LavaArray(zeros(Float32, 256))
        fcpd_scale!(back, 64)(out, x, 2.0f0; ndrange = 256)
        KA.synchronize(back)
        @test Array(out) == Float32.(2 .* (1:256))

        # ── the shape: outer level is device id, inner is the kernel key ──────
        @test Lava.FROZEN_MEM isa Dict{UInt64, <:Dict}
        @test haskey(Lava.FROZEN_MEM, ctx.id)
        mine = Lava.FROZEN_MEM[ctx.id]
        @test all(v -> v isa Lava.LavaLinkedKernel, values(mine))
        # NOT `!isempty` after one launch: a frozen MISS compiles and
        # `frozen_store` writes the SPIR-V to disk, but only a later `frozen_load`
        # disk HIT populates the in-memory memo. So the second launch of the same
        # kernel in a fresh session is what fills it — which is exactly the
        # shipped shape, and worth pinning rather than asserting away.
        fcpd_scale!(back, 64)(out, x, 3.0f0; ndrange = 256)
        KA.synchronize(back)
        @test Array(out) == Float32.(3 .* (1:256))

        # ── a different device id gets its OWN memo, not this one's pipelines ─
        # The bug was that the key had no device in it at all, so every context
        # read the same entries. `frozen_mem` is the accessor `frozen_load` uses.
        other = ctx.id + 0x1000
        try
            othermem = Lava.frozen_mem_byid(other)
            @test othermem !== mine
            @test isempty(othermem)
            # And the real device's entries are untouched by that.
            @test !isempty(Lava.FROZEN_MEM[ctx.id])
        finally
            delete!(Lava.FROZEN_MEM, other)
        end

        # ── the profiler can see frozen kernels ──────────────────────────────
        # This is the half that made a frozen run unprofilable: with
        # LINKED_KERNEL_CACHE empty, `list_compiled_kernels` returned nothing.
        ks = Lava.list_compiled_kernels()
        @test !isempty(ks)
        # `.source`, not `.name`: every Lava compute entry point is called
        # "main", so `name` cannot tell two kernels apart. That is what made this
        # list unusable for attribution before `kernel_source_name` existed.
        @test all(k -> k.name == "main", ks)
        @test any(k -> occursin("fcpd_scale", k.source), ks)
        # Every entry is a real kernel record, not a Dict from walking the wrong
        # level — the failure mode the docstring already records for the other
        # cache.
        @test all(k -> k isa Lava.KernelStats, ks)
        @test all(k -> k.spirv.bytes > 0 && k.spirv.n_instructions > 0, ks)
    finally
        Lava.FROZEN_RECORDING[] = false
        Lava.frozen_clear!()
        Lava.FROZEN_VERSION[] = saved_version
        empty!(Lava.FROZEN_MEM)
        merge!(Lava.FROZEN_MEM, saved_mem)
    end
end
