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

# ── Bool-padding alignment regression tests ──
# Bug: structs with Bool fields followed by Float32/Int32 have alignment padding
# (3 bytes after Bool). The SPIR-V emitter computed byte offsets for struct fields
# by summing _compute_type_size without padding, producing addresses shifted by
# 1-3 bytes. This caused GPUVM faults on RADV and unaligned BDA access errors
# in GPU-assisted validation.

struct BoolPadStruct
    x::Float32
    flag::Bool      # offset 4, 1 byte + 3 padding
    y::Float32      # offset 8
end

struct TwoBoolStruct
    a::Float32
    b::Float32
    c::Float32
    flag1::Bool     # offset 12
    flag2::Bool     # offset 13, 2 bytes padding
    d::Float32      # offset 16
end

struct BoolHeavyStruct
    v1::Float32; v2::Float32; v3::Float32
    active::Bool     # offset 12, padding to 16
    w1::Float32; w2::Float32; w3::Float32; w4::Float32
    done::Bool       # offset 32, padding to 36
    result::Float32  # offset 36
end

@testset "Bool Padding Alignment" begin
    backend = Lava.LavaBackend()

    @kernel function read_bool_pad(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            dst[i] = s.flag ? s.x + s.y : s.x - s.y
        end
    end

    @testset "BoolPadStruct read n=$n" for n in [64, 256, 1024]
        src = Lava.LavaArray([BoolPadStruct(Float32(i), isodd(i), Float32(i+1)) for i in 1:n])
        dst = Lava.LavaArray(zeros(Float32, n))
        read_bool_pad(backend)(dst, src; ndrange=n)
        Lava.vk_flush!()
        result = Array(dst)
        expected = [isodd(i) ? Float32(2i+1) : Float32(-1) for i in 1:n]
        @test result ≈ expected
    end

    @kernel function write_bool_pad(dst, scale::Float32)
        i = @index(Global)
        @inbounds dst[i] = BoolPadStruct(Float32(i) * scale, isodd(i), Float32(i) + scale)
    end

    @testset "BoolPadStruct write n=$n" for n in [64, 256]
        dst = Lava.LavaArray{BoolPadStruct}(undef, n)
        write_bool_pad(backend)(dst, 2f0; ndrange=n)
        Lava.vk_flush!()
        result = Array(dst)
        for i in 1:n
            @test result[i].x ≈ Float32(i) * 2f0
            @test result[i].flag == isodd(i)
            @test result[i].y ≈ Float32(i) + 2f0
        end
    end

    @kernel function read_two_bools(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.a + s.b + s.c + s.d
            if s.flag1; val += 100f0; end
            if s.flag2; val += 1000f0; end
            dst[i] = val
        end
    end

    @testset "TwoBoolStruct (consecutive bools) n=$n" for n in [64, 256]
        src = Lava.LavaArray([TwoBoolStruct(1f0, 2f0, 3f0, isodd(i), i % 3 == 0, 4f0) for i in 1:n])
        dst = Lava.LavaArray(zeros(Float32, n))
        read_two_bools(backend)(dst, src; ndrange=n)
        Lava.vk_flush!()
        result = Array(dst)
        for i in 1:n
            expected = 10f0 + (isodd(i) ? 100f0 : 0f0) + (i % 3 == 0 ? 1000f0 : 0f0)
            @test result[i] ≈ expected
        end
    end

    @kernel function read_bool_heavy(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.v1 + s.v2 + s.v3 + s.w1 + s.w2 + s.w3 + s.w4 + s.result
            if s.active; val *= 2f0; end
            if s.done; val *= -1f0; end
            dst[i] = val
        end
    end

    @testset "BoolHeavyStruct (bools at different offsets) n=$n" for n in [64, 256]
        src = Lava.LavaArray([BoolHeavyStruct(1f0,2f0,3f0, isodd(i), 4f0,5f0,6f0,7f0, i%3==0, 8f0) for i in 1:n])
        dst = Lava.LavaArray(zeros(Float32, n))
        read_bool_heavy(backend)(dst, src; ndrange=n)
        Lava.vk_flush!()
        result = Array(dst)
        for i in 1:n
            base = 36f0  # 1+2+3+4+5+6+7+8
            val = isodd(i) ? base * 2f0 : base
            val = i % 3 == 0 ? val * -1f0 : val
            @test result[i] ≈ val
        end
    end
end
