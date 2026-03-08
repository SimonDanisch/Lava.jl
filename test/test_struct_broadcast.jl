# Test struct broadcast correctness — regression tests for LLVM/Julia struct size mismatch.
#
# Bug: LLVM's struct layout can be larger than Julia's sizeof for types with zero-sized
# fields (Nothing, type parameters). The host-side arg packing used Julia's sizeof to
# allocate inline struct data, but LLVM-generated kernel code reads at LLVM's offsets.
# When LLVM's struct is larger, inline data for successive args overlaps, corrupting values.
#
# The fix: use LLVM byval type sizes (from function parameter attributes) for inline
# struct allocation instead of Julia's sizeof. See entry_wrapper.jl and launch.jl.
#
# These tests specifically target the n=10 case where CompilerMetadata has LLVM size 24
# but Julia sizeof 16, causing the Broadcasted closure's inline data to be read incorrectly.

using Test
using Lava
using KernelAbstractions

# ── Test structs ──

struct TestS12
    a::Float32
    b::Float32
    c::Float32
end

struct TestS52
    f1::Float32; f2::Float32; f3::Float32; f4::Float32
    f5::Float32; f6::Float32; f7::Float32; f8::Float32
    f9::Float32; f10::Float32; f11::Float32; f12::Float32
    f13::Float32
end

struct TestS24
    a::Float64
    b::Float64
    c::Float64
end

@testset "Struct Broadcast Correctness" begin
    # The key sizes that trigger different CompilerMetadata LLVM types:
    # - n=10 → workgroup_size=10, CompilerMetadata LLVM=24 vs Julia=16 (the original bug)
    # - n=1,5 → small sizes
    # - n=32,64,256,1000 → typical sizes with workgroup_size=32/64/256
    @testset "S12 broadcast n=$n" for n in [1, 5, 10, 32, 64, 100, 256, 1000]
        src = Lava.LavaArray(TestS12[TestS12(Float32(i), Float32(i+0.5), Float32(i+0.25)) for i in 1:n])
        dst = Lava.LavaArray{TestS12}(undef, n)
        dst .= src
        Lava.vk_flush!()
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    @testset "S52 broadcast n=$n" for n in [1, 10, 64, 256]
        src = Lava.LavaArray(TestS52[TestS52(ntuple(j -> Float32(i*100 + j), 13)...) for i in 1:n])
        dst = Lava.LavaArray{TestS52}(undef, n)
        dst .= src
        Lava.vk_flush!()
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    @testset "S24 (Float64 fields) broadcast n=$n" for n in [1, 10, 64]
        src = Lava.LavaArray(TestS24[TestS24(Float64(i), Float64(i+0.5), Float64(i+0.25)) for i in 1:n])
        dst = Lava.LavaArray{TestS24}(undef, n)
        dst .= src
        Lava.vk_flush!()
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    # Test that byval_llvm_sizes are populated during compilation
    @testset "byval_llvm_sizes populated" begin
        # After running broadcasts above, cache should have entries
        @test !isempty(Lava._byval_sizes_cache)
        # All byval_sizes should be non-negative
        for (k, v) in Lava._byval_sizes_cache
            @test all(s -> s >= 0, v)
        end
    end
end

@testset "KA Kernel with Struct Args" begin
    @kernel function copy_structs_ka(dst, src)
        i = @index(Global)
        @inbounds dst[i] = src[i]
    end

    @testset "KA copy S12 n=$n" for n in [10, 64, 256]
        src = Lava.LavaArray(TestS12[TestS12(Float32(i), Float32(2i), Float32(3i)) for i in 1:n])
        dst = Lava.LavaArray{TestS12}(undef, n)
        kernel = copy_structs_ka(Lava.LavaBackend())
        kernel(dst, src; ndrange=n)
        Lava.vk_flush!()
        @test Array(dst) == Array(src)
    end

    @kernel function transform_structs_ka(dst, src, scale::Float32)
        i = @index(Global)
        s = @inbounds src[i]
        @inbounds dst[i] = TestS12(s.a * scale, s.b * scale, s.c * scale)
    end

    @testset "KA transform S12 with scalar arg n=$n" for n in [10, 64]
        src = Lava.LavaArray(TestS12[TestS12(1f0, 2f0, 3f0) for _ in 1:n])
        dst = Lava.LavaArray{TestS12}(undef, n)
        kernel = transform_structs_ka(Lava.LavaBackend())
        kernel(dst, src, 2f0; ndrange=n)
        Lava.vk_flush!()
        result = Array(dst)
        @test all(r -> r == TestS12(2f0, 4f0, 6f0), result)
    end
end
