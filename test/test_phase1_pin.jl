using Test, Lava

@testset "Phase 1 — unified pin!/sync_access! + dead-code sweep" begin

@testset "symbol presence/absence" begin
    # Old API must be gone
    @test !isdefined(Lava, :record_one!)
    @test !isdefined(Lava, :record_arg_accesses!)
    @test !isdefined(Lava, :track_buffer_access!)
    @test !isdefined(Lava, :pin_args!)
    @test !isdefined(Lava, :pin_fields!)
    @test !hasmethod(Lava.vk_flush!, Tuple{})

    # New API must be present
    @test isdefined(Lava, :pin!)
    @test isdefined(Lava, :sync_access!)

    # Struct must match
    @test !hasfield(Lava.CommandBatch, :data_refs)
    @test !hasfield(Lava.CommandBatch, :fence)
    @test !hasfield(Lava.CommandBatch, :reaper_task)
    @test !hasfield(Lava.CommandBatch, :retired)
    @test !hasfield(Lava.CommandBatch, :error)
    @test hasfield(Lava.CommandBatch, :pinned)
    @test hasfield(Lava.CommandBatch, :bq)

    # LavaAdaptor must carry the batch (no zero-arg constructor)
    @test fieldnames(Lava.LavaAdaptor) == (:batch,)
    @test_throws MethodError Lava.LavaAdaptor()
end

@testset "pin! deduplication within a batch" begin
    ctx = Lava.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    batch = Lava.ensure_active_batch!(bq)
    @test batch.bq === bq
    empty_size = length(batch.pinned)

    # Same LavaArray pinned 5 times → still one entry (IdSet dedup)
    for _ in 1:5
        Lava.pin!(batch, a)
    end
    @test length(batch.pinned) == empty_size + 1

    Lava.vk_flush!(bq)
end

@testset "sync_access! default is no-op; VkManagedBuffer specialization updates last_write" begin
    ctx = Lava.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))

    batch = Lava.ensure_active_batch!(bq)
    Lava.pin!(batch, a)
    @test a.buf[] in batch.pinned
    # Pre-submit, sync_access! hasn't fired yet; last_write may be anything.
    # After flush, the buffer's last_write must reference this bq.
    Lava.vk_flush!(bq)
    lw = a.buf[].last_write
    @test lw !== nothing
    @test lw[1] === bq

    # Default no-op branch: pin something non-buffer, sync_access! should return nothing.
    batch2 = Lava.ensure_active_batch!(bq)
    @test Lava.sync_access!(batch2, "not a buffer") === nothing
    Lava.vk_flush!(bq)
end

@testset "LavaAdaptor pins during strip (KA-style Adapt path)" begin
    ctx = Lava.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    batch = Lava.ensure_active_batch!(bq)
    adaptor = Lava.LavaAdaptor(batch)

    # adapt strips LavaArray → LavaDeviceArray AND pins the original in the same pass
    import Adapt
    dev_a = Adapt.adapt(adaptor, a)
    @test dev_a isa Lava.LavaDeviceArray
    @test a.buf[] in batch.pinned     # pin fired at the strip point

    Lava.vk_flush!(bq)
end

@testset "wrapper struct: adaptor recurses, pins nested LavaArray" begin
    ctx = Lava.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))
    b = LavaArray{Int32,1}(zeros(Int32, 4))

    struct TestWrapper{A, B}
        items::A
        sizes::B
    end

    w = TestWrapper(a, b)

    batch = Lava.ensure_active_batch!(bq)
    adaptor = Lava.LavaAdaptor(batch)

    import Adapt
    wc = Adapt.adapt(adaptor, w)
    @test wc isa TestWrapper
    @test a.buf[] in batch.pinned
    @test b.buf[] in batch.pinned

    Lava.vk_flush!(bq)
end

end  # @testset
