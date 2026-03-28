# Test rapid GPU allocation/deallocation cycles that mirror the reference test pattern:
# - Many LavaArrays created per "test" (positions, colors, quad_offsets, etc.)
# - Each "test" creates and discards arrays in rapid succession
# - GC pressure from old arrays while new dispatches are recording
# - Simulates the pattern that crashes RayMakie reference tests on APU

using Test, Lava, KernelAbstractions

@kernel function fill_kernel!(dst, val)
    i = @index(Global, Linear)
    @inbounds dst[i] = val
end

@testset "Rapid allocation/free cycles" begin
    backend = Lava.LavaBackend()

    @testset "many small arrays created and discarded" begin
        # Simulate 50 "test iterations" each creating 10 arrays
        for iter in 1:50
            arrays = [Lava.LavaArray(rand(Float32, 100)) for _ in 1:10]

            # Dispatch on each array
            for a in arrays
                fill_kernel!(backend)(a, Float32(iter); ndrange=100)
            end
            Lava.vk_flush!()

            # Verify last array
            result = Array(arrays[end])
            @test all(x -> x == Float32(iter), result)

            # Let arrays go out of scope — GC should handle cleanup
        end
        GC.gc(true)
        Lava.flush_deferred_frees!()
        @test true  # Didn't crash
    end

    @testset "allocation during active recording" begin
        # Create arrays and dispatch without flushing between iterations
        for iter in 1:20
            a = Lava.LavaArray(rand(Float32, 200))
            b = Lava.LavaArray(zeros(Float32, 200))
            fill_kernel!(backend)(b, 1f0; ndrange=200)
            # Don't flush — batches accumulate
        end
        Lava.vk_flush!()
        GC.gc(true)
        Lava.flush_deferred_frees!()
        @test true
    end

    @testset "interleaved alloc/free/dispatch" begin
        # Mirrors the reference test pattern: create figure data, render, discard, repeat
        for iter in 1:30
            # "Scatter plot" data
            positions = Lava.LavaArray(rand(Float32, 50))
            colors = Lava.LavaArray(rand(Float32, 50))
            sizes = Lava.LavaArray(rand(Float32, 50))

            # "Line plot" data
            line_pos = Lava.LavaArray(rand(Float32, 100))
            line_colors = Lava.LavaArray(rand(Float32, 100))

            # Dispatch
            fill_kernel!(backend)(positions, Float32(iter); ndrange=50)
            fill_kernel!(backend)(line_pos, Float32(iter); ndrange=100)

            # Flush every 5 iterations (like the reference test GC between test files)
            if iter % 5 == 0
                Lava.vk_flush!()
                Lava.flush_deferred_frees!()
                GC.gc(true)
            end
        end
        Lava.vk_flush!()
        Lava.flush_deferred_frees!()
        GC.gc(true)
        @test true
    end

    @testset "graphics pipeline alloc pattern" begin
        # Simulate the geometry shader prep: create temp arrays, dispatch, discard
        for iter in 1:20
            # Mimic prep_sprite_gfx allocations
            n = 50
            positions = Lava.LavaArray(rand(Float32, n * 3))
            quad_offsets = Lava.LavaArray(rand(Float32, n * 2))
            quad_scales = Lava.LavaArray(rand(Float32, n * 2))
            rotations = Lava.LavaArray(rand(Float32, n * 4))
            colors = Lava.LavaArray(rand(Float32, n * 4))
            uv_rects = Lava.LavaArray(rand(Float32, n * 4))
            shapes = Lava.LavaArray(rand(UInt8, n))
            stroke_colors = Lava.LavaArray(zeros(Float32, n * 4))
            glow_colors = Lava.LavaArray(zeros(Float32, n * 4))

            # Dispatch a compute kernel on some of them
            fill_kernel!(backend)(positions, 1f0; ndrange=n*3)
            fill_kernel!(backend)(colors, 0.5f0; ndrange=n*4)

            Lava.vk_flush!()

            # Explicit cleanup (what the reference tests should do)
            for arr in [positions, quad_offsets, quad_scales, rotations, colors,
                        uv_rects, stroke_colors, glow_colors]
                Lava.unsafe_free!(arr)
            end
            # shapes is UInt8, unsafe_free! it too
            Lava.unsafe_free!(shapes)

            Lava.flush_deferred_frees!()
        end
        @test true
    end

    @testset "deferred free count stays bounded" begin
        initial_deferred = length(Lava.DEFERRED_FREES)

        for iter in 1:100
            a = Lava.LavaArray(rand(Float32, 50))
            fill_kernel!(backend)(a, 1f0; ndrange=50)
            # Don't explicitly free — let GC handle it
        end

        Lava.vk_flush!()
        GC.gc(true)
        Lava.flush_deferred_frees!()

        final_deferred = length(Lava.DEFERRED_FREES)
        @test final_deferred <= initial_deferred + 10  # Should be mostly cleaned up
    end

    @testset "proactive deferred free flush in vk_alloc" begin
        # Deferred frees should be flushed automatically by vk_alloc when safe
        # (not recording, no in-flight batches)
        for iter in 1:30
            a = Lava.LavaArray(rand(Float32, 100))
            fill_kernel!(backend)(a, Float32(iter); ndrange=100)
            Lava.vk_flush!()
            # Let a go out of scope without explicit free — GC finalizer should handle it
        end
        GC.gc(true)
        # Next allocation should proactively flush any accumulated deferred frees
        b = Lava.LavaArray(rand(Float32, 10))
        @test length(Lava.DEFERRED_FREES) == 0
        Lava.unsafe_free!(b)
    end

    @testset "no crash after many flush cycles" begin
        for cycle in 1:50
            arrays = [Lava.LavaArray(rand(Float32, 100)) for _ in 1:5]
            for a in arrays
                fill_kernel!(backend)(a, Float32(cycle); ndrange=100)
            end
            Lava.vk_flush!()
        end
        GC.gc(true)
        Lava.flush_deferred_frees!()
        @test true
    end
end
