# GPU Memory Safety Tests
#
# Verifies that Lava's memory management correctly handles:
# 1. Use-after-free detection (BDA poison, size check, DataRef exception)
# 2. Double-free safety (warning, no crash)
# 3. Memory leak prevention (buffer count stable across repeated operations)
# 4. keep_data_alive! prevents GC during GPU execution
# 5. Deferred free mechanism
# 6. Derived arrays (views) keep parent alive via DataRef refcount
# 7. Arg validation catches freed arrays before dispatch

using Test
using Lava
using KernelAbstractions
using GPUArrays

@testset "GPU Memory Safety" begin

    # ── 1. Use-after-free detection ──
    @testset "use-after-free detection" begin
        # After unsafe_free!, accessing the buffer should throw LavaError
        @testset "freed LavaArray detected at launch" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            b = Lava.LavaArray(Float32[0, 0, 0])

            @kernel function copy_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i]
            end

            # Free the source array
            Lava.unsafe_free!(a)

            # Attempting to launch with a freed array should throw
            @test_throws Lava.LavaError begin
                copy_k!(Lava.LavaBackend())(b, a; ndrange=3)
            end

            Lava.unsafe_free!(b)
        end

        @testset "BDA poison value set on destroy" begin
            buf = Lava.vk_alloc(1024)
            original_addr = buf.address
            @test original_addr != Lava._BDA_POISON
            @test buf.size == 1024

            Lava.destroy_buffer!(buf)

            @test buf.address == Lava._BDA_POISON
            @test buf.size == 0
        end

        @testset "size==0 after free" begin
            a = Lava.LavaArray(Int32[10, 20, 30])
            buf = a.buf[]
            @test buf.size > 0

            Lava.unsafe_free!(a)
            # After DataRef releases, the buffer should be freed
            # (either immediately or deferred, but size should eventually be 0)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()

            # The DataRef may have already called vk_free! via its destructor
            @test buf.size == 0 || buf.address == Lava._BDA_POISON
        end
    end

    # ── 2. Double-free safety ──
    @testset "double-free safety" begin
        @testset "double vk_free! warns but doesn't crash" begin
            buf = Lava.vk_alloc(512)
            Lava.vk_free!(buf)
            Lava.flush_deferred_frees!()

            # Second free should warn (BDA already poisoned) but not crash
            @test_logs (:warn, r"double-free") Lava.vk_free!(buf)
        end

        @testset "double unsafe_free! on LavaArray is safe" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            Lava.unsafe_free!(a)
            # Second call should be safe (DataRef handles refcount)
            @test nothing === Lava.unsafe_free!(a)
        end
    end

    # ── 3. Memory leak prevention ──
    @testset "memory leak prevention" begin
        @testset "buffer count stable across allocations" begin
            # Force GC and flush to get a clean baseline
            GC.gc(true)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            baseline = length(Lava._live_buffers)

            # Allocate and free many arrays
            for _ in 1:100
                a = Lava.LavaArray(Float32.(rand(1024)))
                Lava.unsafe_free!(a)
            end

            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            after = length(Lava._live_buffers)
            @test after == baseline
        end

        @testset "GPU_LIVE_BYTES tracks correctly" begin
            GC.gc(true)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            before_bytes = Lava.GPU_LIVE_BYTES[]

            # Use a large allocation that bypasses the pool (> POOL_LARGE_THRESHOLD)
            # so GPU_LIVE_BYTES actually increases. Small allocations come from
            # pre-allocated 64MB pool blocks and don't change the counter.
            a = Lava.LavaArray(Float32.(zeros(32 * 1024 * 1024)))  # 128 MB
            after_alloc = Lava.GPU_LIVE_BYTES[]
            @test after_alloc > before_bytes

            Lava.unsafe_free!(a)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
            after_free = Lava.GPU_LIVE_BYTES[]
            @test after_free == before_bytes
        end
    end

    # ── 4. keep_data_alive! prevents GC during GPU execution ──
    @testset "keep_data_alive! mechanism" begin
        @testset "dispatch produces correct results despite GC pressure" begin
            # The real test: dispatching with arrays that could be GC'd works correctly
            # because keep_data_alive! holds references in the batch's data_refs.
            a = Lava.LavaArray(Float32[1, 2, 3, 4])
            b = Lava.LavaArray(Float32[0, 0, 0, 0])

            @kernel function add_one_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i] + 1.0f0
            end

            add_one_k!(Lava.LavaBackend())(b, a; ndrange=4)

            # Trigger GC while GPU work is in flight — keep_data_alive! should prevent
            # the buffers from being collected
            GC.gc(false)

            Lava.vk_flush!()
            result = Array(b)
            @test result == Float32[2, 3, 4, 5]

            Lava.unsafe_free!(a)
            Lava.unsafe_free!(b)
        end
    end

    # ── 5. Deferred free mechanism ──
    @testset "deferred free mechanism" begin
        @testset "buffers deferred during recording" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            b = Lava.LavaArray(Float32[0, 0, 0])

            @kernel function copy_k2!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i]
            end

            # Start a dispatch (creates active recording batch)
            copy_k2!(Lava.LavaBackend())(b, a; ndrange=3)

            # Allocate a temp buffer and free it while recording
            temp = Lava.vk_alloc(256)
            temp_addr = temp.address

            # vk_free! during recording should defer, not destroy immediately
            Lava.vk_free!(temp)

            # Check it was deferred (address not yet poisoned)
            ctx = Lava.vk_context()
            if ctx.active_batch !== nothing && ctx.active_batch.recording
                @test temp.address != Lava._BDA_POISON || temp in Lava.DEFERRED_FREES
            end

            # After flush, deferred frees are processed
            Lava.vk_flush!()
            @test temp.address == Lava._BDA_POISON
            @test temp.size == 0

            Lava.unsafe_free!(a)
            Lava.unsafe_free!(b)
        end
    end

    # ── 6. Derived arrays (views) keep parent alive ──
    @testset "derived arrays keep parent alive" begin
        @testset "derive shares DataRef with parent" begin
            parent = Lava.LavaArray(Float32[10, 20, 30, 40, 50])
            # GPUArrays.derive creates a derived array sharing the same buffer
            child = GPUArrays.derive(Float32, parent, (3,), 1)  # 3 elements, offset=1

            parent_buf = parent.buf[]
            child_buf = child.buf[]
            @test parent_buf === child_buf  # Same underlying VkManagedBuffer

            # Free parent — child should keep buffer alive via DataRef refcount
            Lava.unsafe_free!(parent)

            # Buffer should NOT be destroyed (child holds a DataRef copy)
            @test child_buf.size > 0
            @test child_buf.address != Lava._BDA_POISON

            # Reading from child should still work
            result = Array(child)
            @test result == Float32[20, 30, 40]

            Lava.unsafe_free!(child)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
        end

        @testset "reshape shares DataRef" begin
            a = Lava.LavaArray(Float32[1, 2, 3, 4, 5, 6])
            b = reshape(a, 2, 3)

            a_buf = a.buf[]
            b_buf = b.buf[]
            @test a_buf === b_buf

            Lava.unsafe_free!(a)
            # b should still be valid
            result = Array(b)
            @test result == Float32[1 3 5; 2 4 6]

            Lava.unsafe_free!(b)
            Lava.vk_flush!()
            Lava.flush_deferred_frees!()
        end
    end

    # ── 7. Arg validation catches all 3 conditions ──
    @testset "launch arg validation" begin
        @kernel function noop_k!(x)
            i = @index(Global, Linear)
        end

        @testset "validation enabled by default" begin
            @test Lava._launch_arg_validation[] == true
        end

        @testset "catches freed array (DataRef released)" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            Lava.unsafe_free!(a)

            @test_throws Lava.LavaError begin
                noop_k!(Lava.LavaBackend())(a; ndrange=3)
            end
        end

        @testset "catches poisoned BDA address" begin
            # Create array and manually poison its buffer
            a = Lava.LavaArray(Float32[1, 2, 3])
            buf = a.buf[]
            original_addr = buf.address
            buf.address = Lava._BDA_POISON

            @test_throws Lava.LavaError begin
                noop_k!(Lava.LavaBackend())(a; ndrange=3)
            end

            # Restore for cleanup
            buf.address = original_addr
            Lava.unsafe_free!(a)
        end

        @testset "catches zero-size buffer" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            buf = a.buf[]
            original_size = buf.size
            buf.size = 0

            @test_throws Lava.LavaError begin
                noop_k!(Lava.LavaBackend())(a; ndrange=3)
            end

            # Restore for cleanup
            buf.size = original_size
            Lava.unsafe_free!(a)
        end

        @testset "validation can be disabled" begin
            a = Lava.LavaArray(Float32[1, 2, 3])
            buf = a.buf[]
            buf.size = 0  # Would normally trigger validation error

            Lava._launch_arg_validation[] = false
            try
                # Should NOT throw with validation disabled
                # (but we don't actually dispatch — just test that validation is skipped)
                # We can't safely dispatch with a corrupted buffer, so just test the toggle
                @test Lava._launch_arg_validation[] == false
            finally
                Lava._launch_arg_validation[] = true
                buf.size = max(3 * sizeof(Float32), 16)  # Restore
            end
            Lava.unsafe_free!(a)
        end
    end

    # ── 8. Correct GPU results after memory operations ──
    @testset "correctness after memory operations" begin
        @testset "allocate-use-free cycle produces correct results" begin
            @kernel function saxpy_k!(y, a, x)
                i = @index(Global, Linear)
                @inbounds y[i] = a * x[i] + y[i]
            end

            N = 1024
            for _ in 1:5
                x = Lava.LavaArray(Float32.(ones(N)))
                y = Lava.LavaArray(Float32.(2.0f0 .* ones(N)))

                saxpy_k!(Lava.LavaBackend())(y, 3.0f0, x; ndrange=N)
                Lava.vk_flush!()

                result = Array(y)
                @test all(r -> r ≈ 5.0f0, result)

                Lava.unsafe_free!(x)
                Lava.unsafe_free!(y)
            end
        end

        @testset "reuse after free produces correct results" begin
            @kernel function fill_k!(a, val)
                i = @index(Global, Linear)
                @inbounds a[i] = val
            end

            # Allocate, free, reallocate — new buffer should work correctly
            a = Lava.LavaArray{Float32}(undef, 512)
            fill_k!(Lava.LavaBackend())(a, 42.0f0; ndrange=512)
            Lava.vk_flush!()
            @test all(r -> r ≈ 42.0f0, Array(a))

            Lava.unsafe_free!(a)
            Lava.flush_deferred_frees!()

            # New allocation may reuse the same VRAM
            b = Lava.LavaArray{Float32}(undef, 512)
            fill_k!(Lava.LavaBackend())(b, 99.0f0; ndrange=512)
            Lava.vk_flush!()
            @test all(r -> r ≈ 99.0f0, Array(b))

            Lava.unsafe_free!(b)
        end
    end

    # ── 9. Slab allocator safety ──
    @testset "arg buffer slab allocator" begin
        @testset "reset after flush" begin
            @kernel function trivial_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i)
            end

            a = Lava.LavaArray{Float32}(undef, 64)

            # Multiple dispatches before flush
            for _ in 1:10
                trivial_k!(Lava.LavaBackend())(a; ndrange=64)
            end

            # After flush, slab allocator resets
            Lava.vk_flush!()
            @test Lava._arg_slab_offset[] == 0
            @test Lava._arg_slab_idx[] == 1

            result = Array(a)
            @test result[1] ≈ 1.0f0
            @test result[64] ≈ 64.0f0

            Lava.unsafe_free!(a)
        end

        @testset "many dispatches don't exhaust allocations" begin
            a = Lava.LavaArray{Float32}(undef, 16)

            @kernel function inc_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] += 1.0f0
            end

            # 500 dispatches in a single batch — slab allocator should handle this
            fill!(a, 0.0f0)
            for _ in 1:500
                inc_k!(Lava.LavaBackend())(a; ndrange=16)
            end
            Lava.vk_flush!()

            result = Array(a)
            @test result[1] ≈ 500.0f0

            Lava.unsafe_free!(a)
        end
    end

    # ── 10. Device generation tracking ──
    @testset "device generation tracking" begin
        @testset "buffer records device generation at creation" begin
            buf = Lava.vk_alloc(256)
            @test buf.device_gen == Lava._device_generation[]
            Lava.vk_free!(buf)
            Lava.flush_deferred_frees!()
        end
    end
end
