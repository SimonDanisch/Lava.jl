# KNOWN BUG: whole-struct copy of a 6-byte struct drops a field on every other element.
#
#     struct S3I16; a::Int16; b::Int16; c::Int16; end   # sizeof 6, offsets 0/2/4
#     dst .= src        # b == 0 for even elements
#     d[i] = s[i]       # same, in a hand-written kernel
#
# NOT broadcast-specific — any whole-struct copy is affected, which is the part
# that makes it dangerous: a plain `d[i] = s[i]` is silently wrong.
#
# Element i sits at byte 6(i-1), so its middle field is at ≡2 (mod 4) for odd i
# and ≡0 (mod 4) for even i. Only elements whose fields flip 4-byte phase are
# corrupted.
#
# What has been ruled out, each verified on the 8060S:
#
#   sizeof 4 (2xInt16), 8 (4xInt16), 12 (3xFloat32), 12 (Int16+2xInt32)  correct
#   sizeof 3 (3xInt8)                                                    correct
#       -> needs 16-bit fields whose absolute offset alternates 4-byte phase;
#          not merely "element size is not a multiple of 4"
#   copyto!(dst, src)                                    correct (different path)
#   dst .= Ref(S3I16(11,22,33))  constant, no load       correct (not the store)
#   read src[i], write .a/.b/.c to three arrays          correct (not the load)
#       -> it needs a loaded AGGREGATE forwarded into a store; neither half alone
#   rebuilding the struct from its fields before storing correct (typed stores)
#
# REJECTED HYPOTHESIS, recorded so it is not re-tried: that `lower_memcpy!` in
# compiler/compilation.jl over-aligns. Its non-alloca fallback copies 4-byte
# chunks and stamps `alignment!(..., 4)`, which really is wrong for a 6-byte
# element stride — but deriving the chunk width from the largest power of two
# dividing the copy size (i32/i16/i8) changed nothing: all 16 even elements stayed
# corrupted. So either this copy does not reach that fallback (the destination is
# an alloca, taking the typed load/store branch above it), or the fault is further
# down in SPIR-V emission. Next step is to dump the emitted SPIR-V for
# `aggregate_copy!` and look at the store sequence.
#
# Registered with @test_broken rather than left unregistered: a gated test nobody
# runs rots. Three tests in this suite were fixed without anyone noticing because
# they were gated AND absent from runtests.jl. @test_broken reports "Unexpected
# Pass" the moment this is fixed.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

struct CopyS6;  a::Int16; b::Int16; c::Int16; end                    # 6  — the bug
struct CopyS2;  a::Int16; b::Int16; end                              # 4
struct CopyS4;  a::Int16; b::Int16; c::Int16; d::Int16; end          # 8
struct CopyS3B; a::Int8;  b::Int8;  c::Int8; end                     # 3
struct CopyS12; a::Float32; b::Float32; c::Float32; end              # 12
struct CopyS10; a::Int16; b::Int32; c::Int32; end                    # 12 (padded)

@kernel function aggregate_copy!(d, @Const(s))
    i = @index(Global)
    @inbounds d[i] = s[i]          # whole-struct load then store -> memcpy
end

function check_copy(::Type{S}, mk) where {S}
    n = 16
    src_data = [mk(i) for i in 1:n]
    src = Lava.LavaArray(src_data)

    bcast = Lava.LavaArray{S}(undef, n)
    bcast .= src

    kern = Lava.LavaArray{S}(undef, n)
    aggregate_copy!(LavaBackend(), 64)(kern, src; ndrange = n)
    KA.synchronize(LavaBackend())

    return Array(bcast) == src_data, Array(kern) == src_data
end

@testset "whole-struct copy preserves every field" begin
    # The size that broke: 6 bytes, so even elements are only 2-aligned.
    @test sizeof(CopyS6) == 6
    b, k = check_copy(CopyS6, i -> CopyS6(Int16(i), Int16(i * 3), Int16(i * 7)))
    @test_broken b        # broadcast
    @test_broken k        # hand-written kernel — same fault

    # Sizes that ARE correct. They pin the boundary of the fault, and a fix that
    # changes how copies are chunked must not regress them.
    for (S, mk) in (
            (CopyS2,  i -> CopyS2(Int16(i), Int16(i * 3))),
            (CopyS4,  i -> CopyS4(Int16(i), Int16(i * 3), Int16(i * 7), Int16(i * 11))),
            (CopyS3B, i -> CopyS3B(Int8(i), Int8(i * 3), Int8(i * 7))),
            (CopyS12, i -> CopyS12(Float32(i), Float32(i * 3), Float32(i * 7))),
            (CopyS10, i -> CopyS10(Int16(i), Int32(i * 3), Int32(i * 7))),
        )
        b, k = check_copy(S, mk)
        @test b
        @test k
    end

    # Paths that never used the memcpy fallback; kept so a future change to it
    # cannot quietly start routing them through a broken one.
    n = 16
    sd = [CopyS6(Int16(i), Int16(i * 3), Int16(i * 7)) for i in 1:n]
    src = Lava.LavaArray(sd)
    d = Lava.LavaArray{CopyS6}(undef, n)
    copyto!(d, src)
    Lava.vk_flush!(Lava.vk_context())
    @test Array(d) == sd
end
