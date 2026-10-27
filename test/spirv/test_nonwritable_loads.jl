# Tier 1: a PSB load LLVM marks `!invariant.load` is read through a NonWritable member
#
# SPIR-V decorates memory objects and struct members, never pointer values, so the
# only way to tell a driver that memory behind a PSB pointer is read-only is to load
# through member 0 of a struct whose member is `NonWritable` (glslang's `readonly
# buffer_reference` has the same shape). RADV first showed why it matters: it only
# uses the scalar cache (SMEM) for a global load it knows is not written, so a loop
# reading the same address in every lane issued vector loads instead.
#
# The struct must be its own type: the cached `{T}` struct is shared with every other
# `{T}` in the module (the push-constant block is `{ulong}`), and decorating that one
# would claim those are read-only too.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_count, compile_and_disasm

@inline function nw_load_invariant(p::Ptr{UInt32})
    Base.llvmcall(("""
        define i32 @entry(ptr %p) #0 {
            %v = load i32, ptr %p, align 4, !invariant.load !0
            ret i32 %v
        }
        attributes #0 = { alwaysinline }
        !0 = !{}
        """, "entry"), UInt32, Tuple{Ptr{UInt32}}, p)
end

@inline function nw_load_invariant(p::Ptr{UInt64})
    Base.llvmcall(("""
        define i64 @entry(ptr %p) #0 {
            %v = load i64, ptr %p, align 8, !invariant.load !0
            ret i64 %v
        }
        attributes #0 = { alwaysinline }
        !0 = !{}
        """, "entry"), UInt64, Tuple{Ptr{UInt64}}, p)
end

function nw_invariant_kernel(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = nw_load_invariant(src.ptr + 4 * (i - 1))
    return nothing
end

function nw_plain_kernel(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[i]
    return nothing
end

function nw_invariant_u64_kernel(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = nw_load_invariant(src.ptr + 8 * (i - 1))
    return nothing
end

function nw_const_2d!(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[1, i] + src[2, i]
    return nothing
end

function nw_const_cartesian!(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[CartesianIndex(1, i)]
    return nothing
end

function nw_const_linear!(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[i]
    return nothing
end

function nw_const_0d!(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[]
    return nothing
end

struct NWPair
    a::Float32
    b::Float32
end

function nw_const_composite!(out, src)
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = src[i].a
    return nothing
end

"The struct ids carrying a `NonWritable` member decoration in disassembly `d`."
nonwritable_structs(d) = [m[1] for m in eachmatch(r"OpMemberDecorate (%\S+) 0 NonWritable", d)]

@testset "NonWritable loads" begin
    @testset "an invariant load goes through a NonWritable member" begin
        d, _ = compile_and_disasm(nw_invariant_kernel,
                                  Tuple{Lava.LavaDeviceArray{UInt32,1}, Lava.LavaDeviceArray{UInt32,1}})
        structs = nonwritable_structs(d)
        @test length(structs) == 1
        s = only(structs)
        check(d, "OpMemberDecorate $s 0 Offset 0")
        # The load's pointer is an access chain into a PSB pointer to that struct.
        ptr_ty = match(Regex("(%\\S+) = OpTypePointer PhysicalStorageBuffer $s\\b"), d)
        @test ptr_ty !== nothing
        check(d, "OpConvertUToPtr $(ptr_ty[1])")
        check(d, "OpAccessChain %_ptr_PhysicalStorageBuffer_uint")
    end

    @testset "a plain load is not marked" begin
        d, _ = compile_and_disasm(nw_plain_kernel,
                                  Tuple{Lava.LavaDeviceArray{UInt32,1}, Lava.LavaDeviceArray{UInt32,1}})
        check_not(d, "NonWritable")
    end

    @testset "the NonWritable struct is not a shared {T} struct" begin
        # `{ulong}` is also the push-constant block; the read-only struct must not be it.
        d, _ = compile_and_disasm(nw_invariant_u64_kernel,
                                  Tuple{Lava.LavaDeviceArray{UInt64,1}, Lava.LavaDeviceArray{UInt64,1}})
        s = only(nonwritable_structs(d))
        check_not(d, "OpDecorate $s Block")
        check_not(d, "OpTypePointer PushConstant $s\n")
        check_count(d, "OpMemberDecorate $s 0 ", 2)   # Offset and NonWritable, nothing else
    end

    # Every indexing form `LavaDeviceArray` has, read through `Lava.Const`.
    @testset "Lava.Const reads through a NonWritable member: $(nameof(f))" for (f, src) in (
            (nw_const_2d!,        Lava.Const{Float32,2}),
            (nw_const_cartesian!, Lava.Const{Float32,2}),
            (nw_const_linear!,    Lava.Const{Float16,1}),
            (nw_const_0d!,        Lava.Const{Float32,0}))
        d, _ = compile_and_disasm(f, Tuple{Lava.LavaDeviceArray{eltype(src),1}, src})
        @test length(nonwritable_structs(d)) == 1
    end

    @testset "a composite element loads as the array does" begin
        # As CUDA.jl's `unsafe_cached_load` falls back: the metadata would have to be
        # carried field by field.
        d, _ = compile_and_disasm(nw_const_composite!,
                                  Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.Const{NWPair,1}})
        check_not(d, "NonWritable")
    end

    @testset "reinterpreting keeps the promise" begin
        a = Lava.LavaDeviceArray{UInt32,1}(Ptr{UInt32}(UInt(64)), (4,))
        r = reinterpret(UInt8, Lava.Const(a))
        @test r isa Lava.Const{UInt8,1}
        @test size(r) == (16,)
        @test pointer(r) == Ptr{UInt8}(UInt(64))
    end
end
