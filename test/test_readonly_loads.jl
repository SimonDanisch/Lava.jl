# Arguments a compile job marks read-only (`LavaCompilerParams.readonly`): a load from
# memory they reach, at an address every invocation of a subgroup shares, is invariant,
# and the emitter reads it through a NonWritable member. The flags replaced compiling
# those arguments as `Lava.Const`, which gave the compile other argument types than
# Mantle's access walk and so a second inference of the whole kernel. Pinned here: a
# flag marks the uniform loads and leaves the per-invocation ones alone, through an
# argument passed by value or by reference; every piece of a load read in pieces gets the
# same access; and a kernel that writes a flagged argument does not compile.
#
# Device-free, like the rest of this suite.

using Test, Lava

const D = Lava.LavaDeviceArray{Float32,1}

# `a[i]` differs per invocation, `b[j]` is the same address in all of them.
function rl_dot!(out, a, b)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        s = 0f0
        for j in 1:length(b)
            @inbounds s += a[i] * b[j]
        end
        @inbounds out[i] = s
    end
    return nothing
end

function rl_pair_dot!(out, pair)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        s = 0f0
        for j in 1:length(pair.b)
            @inbounds s += pair.a[i] * pair.b[j]
        end
        @inbounds out[i] = s
    end
    return nothing
end

# New code every run, so the object cache cannot answer from an earlier one.
const RL_SALT = Float32(rand(10_000:99_999))
function rl_cached_sum!(out, a, b)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    i <= length(out) && (@inbounds out[i] = a[i] + b[1] + RL_SALT)
    return nothing
end

function rl_write!(out, a)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    i <= length(out) && (@inbounds a[i] = out[1])
    return nothing
end

function rl_bytes!(out, a)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    # An Int32 at a byte offset, loaded at alignment 1: the emitter reads it byte by byte.
    i <= length(out) && (@inbounds out[i] = unsafe_load(Ptr{Int32}(pointer(a, 2))))
    return nothing
end

compiled(f, tt; readonly = 0) =
    Lava.lava_compile_gpu_from_job(Lava.lava_kernel_job(f, tt; workgroup_size = (64, 1, 1), readonly))

# The loads of SPIR-V type `type` in a kernel, and how many of them read through a
# NonWritable member.
function loads(kernel, type)
    dis = Lava.disassemble_spirv(kernel.spirv_bytes)
    blocks = Set(m[1] for m in eachmatch(r"OpMemberDecorate (%\w+) 0 NonWritable", dis))
    pointertypes = Set(m[1] for m in eachmatch(r"(%\w+) = OpTypePointer PhysicalStorageBuffer (%\w+)", dis)
                       if m[2] in blocks)
    pointers = Set(m[1] for m in eachmatch(r"(%\w+) = Op\w+ (%\w+)", dis) if m[2] in pointertypes)
    members = Set(m[1] for m in eachmatch(r"(%\w+) = OpAccessChain %\w+ (%\w+)", dis) if m[2] in pointers)
    found = [m[1] for m in eachmatch(Regex("= OpLoad %$type (%\\w+)"), dis)]
    return (; total = length(found), nonwritable = count(in(members), found))
end
nonwritable(kernel, type = "float") = loads(kernel, type).nonwritable

@testset "read-only arguments" begin
    @testset "a flag marks the loads every invocation shares" begin
        @test nonwritable(compiled(rl_dot!, Tuple{D,D,D}; readonly = 0b100)) > 0
        # `a[i]` is per invocation: NonWritable would only free the driver to move it.
        @test nonwritable(compiled(rl_dot!, Tuple{D,D,D}; readonly = 0b010)) == 0
        @test nonwritable(compiled(rl_dot!, Tuple{D,D,D})) == 0
    end

    @testset "through an argument passed by reference" begin
        Pair = NamedTuple{(:a, :b),Tuple{D,D}}
        @test nonwritable(compiled(rl_pair_dot!, Tuple{D,Pair}; readonly = 0b10)) > 0
        @test nonwritable(compiled(rl_pair_dot!, Tuple{D,Pair})) == 0
    end

    @testset "the kernel cache keeps a kernel compiled with flags apart from one without" begin
        # The flags act after LLVM's optimization, so a key on the optimized module alone
        # served the flagged kernel's SPIR-V to the plain launch, or the other way round.
        cached(readonly) = Lava.compile_or_lookup(Lava.lava_kernel_job(rl_cached_sum!, Tuple{D,D,D};
                                                                       workgroup_size = (64, 1, 1), readonly))
        @test nonwritable(cached(0b110)) > 0
        @test nonwritable(cached(0)) == 0
    end

    @testset "every piece of a load read in pieces is NonWritable" begin
        # RADV merges neighbouring loads only when their access flags agree. The emitter
        # read the pieces of an under-aligned invariant load as plain loads, so a
        # read-only first field next to plain byte loads of the rest split a 28-byte
        # element into fifteen loads, and a trace kernel lost half its occupancy.
        U = Lava.LavaDeviceArray{UInt8,1}
        I = Lava.LavaDeviceArray{Int32,1}
        @test loads(compiled(rl_bytes!, Tuple{I,U}; readonly = 0b10), "uchar") == (; total = 4, nonwritable = 4)
        @test loads(compiled(rl_bytes!, Tuple{I,U}), "uchar") == (; total = 4, nonwritable = 0)
    end

    @testset "a kernel that writes a flagged argument does not compile" begin
        @test_throws "marks read-only" compiled(rl_write!, Tuple{D,D}; readonly = 0b10)
        @test nonwritable(compiled(rl_write!, Tuple{D,D}; readonly = 0b01)) > 0
    end
end
