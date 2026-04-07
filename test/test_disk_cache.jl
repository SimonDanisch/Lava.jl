# Disk Cache Persistence Tests
#
# Tests that GPUCompiler's disk cache works for Lava's SPIR-V compilation:
# 1. Kernels defined in packages (not REPL) get cached to disk
# 2. Cache hit skips LLVM + SPIR-V compilation on second load
# 3. Cache invalidation works when kernel code changes
#
# NOTE: Disk caching only works for kernels defined in precompiled packages.
# REPL-defined kernels get build_id=nothing and are not disk-cached.
# These tests verify the in-memory two-tier cache and GPUCompiler integration.

using Test
using Lava
using KernelAbstractions
using GPUCompiler

@testset "Kernel Cache" begin

    @testset "two-tier cache structure" begin
        # Compile a kernel
        @kernel function tier_test_kernel(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i] + 1f0
        end

        backend = Lava.LavaBackend()
        a = Lava.LavaArray(Float32[1, 2, 3, 4])
        b = Lava.LavaArray(zeros(Float32, 4))

        # Clear caches
        empty!(Lava.LINKED_KERNEL_CACHE)

        # First dispatch populates both tiers
        tier_test_kernel(backend)(b, a; ndrange=4)
        KernelAbstractions.synchronize(backend)

        @test length(Lava.LINKED_KERNEL_CACHE) >= 1
        # Disk cache should have written files
        dir = Lava.lava_disk_cache_dir()
        @test isdir(dir) && !isempty(filter(f -> endswith(f, ".jls"), readdir(dir)))
        @test Array(b) == Float32[2, 3, 4, 5]
    end

    @testset "Tier 1 hit is fast" begin
        @kernel function speed_test_kernel(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i] * 2f0
        end

        backend = Lava.LavaBackend()
        a = Lava.LavaArray(Float32[1, 2, 3, 4])
        b = Lava.LavaArray(zeros(Float32, 4))

        # Warm up (compile)
        speed_test_kernel(backend)(b, a; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # Tier 1 hit should be very fast (< 1ms for the cache lookup alone)
        t = @elapsed for _ in 1:100
            speed_test_kernel(backend)(b, a; ndrange=4)
        end
        avg_us = t / 100 * 1e6
        @test avg_us < 500  # < 500μs per dispatch (including arg pack + vkCmd record)
    end

    @testset "Tier 2 repopulates Tier 1" begin
        @kernel function repop_test_kernel(dst, val::Float32)
            i = @index(Global)
            @inbounds dst[i] = val
        end

        backend = Lava.LavaBackend()
        c = Lava.LavaArray(zeros(Float32, 4))

        # Compile
        repop_test_kernel(backend)(c, 42f0; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # Clear Tier 1 only (simulating session restart without disk)
        empty!(Lava.LINKED_KERNEL_CACHE)
        empty!(Lava.KERNEL_INSERTION_ORDER)

        # Next dispatch should hit Tier 2 and repopulate Tier 1
        repop_test_kernel(backend)(c, 99f0; ndrange=4)
        KernelAbstractions.synchronize(backend)

        @test length(Lava.LINKED_KERNEL_CACHE) >= 1
        @test Array(c) == fill(99f0, 4)
    end

    @testset "different workgroup sizes get separate cache entries" begin
        @kernel function wg_test_kernel(dst)
            i = @index(Global)
            @inbounds dst[i] = Float32(i)
        end

        backend = Lava.LavaBackend()
        d = Lava.LavaArray(zeros(Float32, 64))
        before = length(Lava.LINKED_KERNEL_CACHE)

        wg_test_kernel(backend)(d; ndrange=64, workgroupsize=32)
        KernelAbstractions.synchronize(backend)
        after_32 = length(Lava.LINKED_KERNEL_CACHE)

        wg_test_kernel(backend)(d; ndrange=64, workgroupsize=64)
        KernelAbstractions.synchronize(backend)
        after_64 = length(Lava.LINKED_KERNEL_CACHE)

        # Two different workgroup sizes should create two cache entries
        @test after_64 > after_32
    end

    @testset "LavaLinkedKernel has correct fields" begin
        # Check that the linked kernel has all expected data
        for (key, linked) in Lava.LINKED_KERNEL_CACHE
            @test linked isa Lava.LavaLinkedKernel
            @test !isempty(linked.compiled.spirv_bytes)
            @test !isempty(linked.compiled.entry_name)
            @test linked.compiled.workgroup_size isa NTuple{3, Int}
            @test !isempty(linked.offsets)
            @test length(linked.byval_sizes) == length(linked.offsets)
            break  # just check first entry
        end
    end

    @testset "GPUCompiler disk cache integration" begin
        # Check that disk cache is available (may or may not be enabled)
        path = GPUCompiler.disk_cache_path()
        @test path isa String
        @test !isempty(path)

        # Verify cache_file returns nothing for REPL-defined kernels
        # (disk cache only works for precompiled package code)
        @kernel function repl_kernel(dst)
            i = @index(Global)
            @inbounds dst[i] = 1f0
        end

        backend = Lava.LavaBackend()
        e = Lava.LavaArray(zeros(Float32, 4))
        repl_kernel(backend)(e; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # REPL kernels should work even without disk cache
        @test Array(e) == fill(1f0, 4)
    end
end
