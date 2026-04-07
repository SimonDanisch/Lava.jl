# Caching, GC, and Allocation Tests for Lava.jl
#
# Tests kernel cache eviction, pipeline cache, staging buffer lifecycle,
# CB batching, and reset_device cleanup of global state.

using Test
using Lava
using KernelAbstractions

@testset "Caching & Allocations" begin

    # ── 1. Kernel cache eviction ──
    @testset "kernel cache eviction" begin
        @testset "FIFO eviction at max size" begin
            # Save original max
            old_max = Lava.MAX_KERNEL_CACHE_SIZE[]
            Lava.MAX_KERNEL_CACHE_SIZE[] = 5

            try
                # Create 7 distinct kernels by varying workgroup size
                results = Lava.LavaArray{Float32}(undef, 16)
                for wg in (16, 32, 64, 128, 256, 512, 1024)
                    @kernel function fill_wg!(a)
                        i = @index(Global, Linear)
                        @inbounds a[i] = Float32(i)
                    end
                    fill_wg!(Lava.LavaBackend())(results; ndrange=16, workgroupsize=wg)
                end
                Lava.vk_flush!()

                # Cache should have at most 5 entries
                @test length(Lava.LINKED_KERNEL_CACHE) <= 5
                @test length(Lava.KERNEL_INSERTION_ORDER) <= 5

                # Result should still be correct (latest kernel worked)
                r = Array(results)
                @test r[1] ≈ 1.0f0
                @test r[16] ≈ 16.0f0

                Lava.unsafe_free!(results)
            finally
                Lava.MAX_KERNEL_CACHE_SIZE[] = old_max
            end
        end

        @testset "cache hit produces correct results" begin
            a = Lava.LavaArray{Float32}(undef, 64)
            @kernel function double_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i) * 2.0f0
            end

            # First call: compile
            double_k!(Lava.LavaBackend())(a; ndrange=64)
            Lava.vk_flush!()
            r1 = Array(a)

            # Second call: cache hit
            fill!(a, 0.0f0)
            double_k!(Lava.LavaBackend())(a; ndrange=64)
            Lava.vk_flush!()
            r2 = Array(a)

            @test r1 == r2
            @test r1[32] ≈ 64.0f0

            Lava.unsafe_free!(a)
        end
    end

    # ── 2. Pipeline cache eviction ──
    @testset "pipeline cache eviction" begin
        old_max = Lava.MAX_PIPELINE_CACHE_SIZE[]
        Lava.MAX_PIPELINE_CACHE_SIZE[] = 3

        try
            a = Lava.LavaArray{Float32}(undef, 16)

            # Generate distinct pipelines via different kernel functions
            @kernel function pk1!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 1.0f0
            end
            @kernel function pk2!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 2.0f0
            end
            @kernel function pk3!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 3.0f0
            end
            @kernel function pk4!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 4.0f0
            end
            @kernel function pk5!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 5.0f0
            end

            pk1!(Lava.LavaBackend())(a; ndrange=16)
            pk2!(Lava.LavaBackend())(a; ndrange=16)
            pk3!(Lava.LavaBackend())(a; ndrange=16)
            pk4!(Lava.LavaBackend())(a; ndrange=16)
            pk5!(Lava.LavaBackend())(a; ndrange=16)
            Lava.vk_flush!()

            @test length(Lava.PIPELINE_CACHE) <= 3
            @test length(Lava.PIPELINE_INSERTION_ORDER) <= 3

            # Latest pipeline still works
            @test Array(a)[1] ≈ 5.0f0

            Lava.unsafe_free!(a)
        finally
            Lava.MAX_PIPELINE_CACHE_SIZE[] = old_max
        end
    end

    # ── 3. Staging buffer grows correctly ──
    @testset "staging buffer lifecycle" begin
        @testset "grows on demand" begin
            # Upload small then large data — staging buffer should grow
            small = Lava.LavaArray(Float32.(ones(16)))
            r_small = Array(small)
            @test all(r_small .≈ 1.0f0)

            large = Lava.LavaArray(Float32.(2.0f0 .* ones(4096)))
            r_large = Array(large)
            @test all(r_large .≈ 2.0f0)

            # Staging buf should exist and be at least 4096 * 4 bytes
            staging = Lava.STAGING_BUF[]
            if staging !== nothing
                @test staging[4] >= 4096 * sizeof(Float32)
            end

            Lava.unsafe_free!(small)
            Lava.unsafe_free!(large)
        end
    end

    # ── 4. CB batching data_refs lifecycle ──
    @testset "command batch data_refs" begin
        @testset "data_refs cleared after flush" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            @kernel function noop_cb!(x)
                i = @index(Global, Linear)
            end
            noop_cb!(Lava.LavaBackend())(a; ndrange=3)

            # After dispatch, batch should have data_refs
            ctx = Lava.vk_context()

            Lava.vk_flush!()

            # After flush, if batch was reclaimed, data_refs should be cleared
            # (The batch may be recycled into free_batches with empty data_refs)
            batch = ctx.active_batch
            if batch !== nothing
                @test isempty(batch.data_refs)
            end

            Lava.unsafe_free!(a)
        end
    end

    # ── 5. Arg slab allocator reuse ──
    @testset "arg slab allocator reuse" begin
        @testset "slabs reused across flushes" begin
            a = Lava.LavaArray{Float32}(undef, 16)
            @kernel function slab_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i)
            end

            # First batch of dispatches
            for _ in 1:100
                slab_k!(Lava.LavaBackend())(a; ndrange=16)
            end
            Lava.vk_flush!()
            n_slabs_after_first = length(Lava.ARG_SLABS)

            # Second batch — should reuse same slabs
            for _ in 1:100
                slab_k!(Lava.LavaBackend())(a; ndrange=16)
            end
            Lava.vk_flush!()
            n_slabs_after_second = length(Lava.ARG_SLABS)

            @test n_slabs_after_second == n_slabs_after_first
            @test Lava.ARG_SLAB_OFFSET[] == 0  # Reset after flush

            Lava.unsafe_free!(a)
        end
    end

    # ── 6. Indirect buffer slab reuse ──
    @testset "indirect buffer slab reuse" begin
        @testset "indirect slabs reset after flush" begin
            a = Lava.LavaArray{Float32}(undef, 16)

            @kernel function indirect_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i)
            end

            # Several dispatches that use indirect buffers
            for _ in 1:50
                indirect_k!(Lava.LavaBackend())(a; ndrange=16)
            end
            Lava.vk_flush!()

            @test Lava.INDIRECT_SLAB_OFFSET[] == 0
            @test Lava.INDIRECT_SLAB_IDX[] == 1

            Lava.unsafe_free!(a)
        end
    end

    # ── 7. GC pressure tracking ──
    @testset "GC pressure tracking" begin
        @testset "GPU_BYTES_SINCE_LAST_GC increments" begin
            before = Lava.GPU_BYTES_SINCE_LAST_GC[]

            buf = Lava.vk_alloc(1024)
            after = Lava.GPU_BYTES_SINCE_LAST_GC[]

            @test after >= before + 1024

            Lava.vk_free!(buf)
            Lava.flush_deferred_frees!()
        end
    end

    # ── 8. Correctness across many compile-dispatch cycles ──
    @testset "correctness across many dispatch cycles" begin
        @testset "varied kernels produce correct results" begin
            N = 256
            a = Lava.LavaArray{Float32}(undef, N)
            b = Lava.LavaArray{Float32}(undef, N)

            # Kernel 1: identity
            @kernel function id_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i]
            end

            # Kernel 2: negate
            @kernel function neg_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = -src[i]
            end

            # Kernel 3: add constant
            @kernel function add_k!(dst, src, c)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i] + c
            end

            fill!(a, 7.0f0)

            id_k!(Lava.LavaBackend())(b, a; ndrange=N)
            Lava.vk_flush!()
            @test all(Array(b) .≈ 7.0f0)

            neg_k!(Lava.LavaBackend())(b, a; ndrange=N)
            Lava.vk_flush!()
            @test all(Array(b) .≈ -7.0f0)

            add_k!(Lava.LavaBackend())(b, a, 3.0f0; ndrange=N)
            Lava.vk_flush!()
            @test all(Array(b) .≈ 10.0f0)

            Lava.unsafe_free!(a)
            Lava.unsafe_free!(b)
        end
    end

    # ── 9. Broadcast allocation stability ──
    @testset "broadcast allocation stability" begin
        @testset "repeated broadcasts don't leak" begin
            GC.gc(true)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            baseline = length(Lava.LIVE_BUFFERS)

            for _ in 1:20
                a = Lava.LavaArray(Float32.(rand(128)))
                b = Lava.LavaArray(Float32.(rand(128)))
                c = a .+ b  # broadcast creates a new array
                Lava.vk_flush!()
                Lava.unsafe_free!(a)
                Lava.unsafe_free!(b)
                Lava.unsafe_free!(c)
            end

            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            after = length(Lava.LIVE_BUFFERS)
            @test after == baseline
        end
    end

    # ── 10. Unified buffer allocation ──
    @testset "unified buffer allocation" begin
        @testset "mapped ptr is non-null" begin
            buf = Lava.vk_alloc_unified(256)
            @test buf.mapped_ptr != Ptr{UInt8}(0)
            @test buf.address != 0
            @test buf.size >= 256
            Lava.vk_free!(buf)
            Lava.flush_deferred_frees!()
        end
    end
end
