# KNOWN BUG: broadcasting a 6-byte struct drops a 16-bit field on every other element.
#
#   struct S3I16; a::Int16; b::Int16; c::Int16; end     # sizeof 6, offsets 0/2/4
#   dst .= src                                          # b == 0 for even elements
#
# Element i starts at byte 6(i-1), so its middle field lands at ≡2 (mod 4) for odd
# i and ≡0 (mod 4) for even i. Only the elements whose fields flip 4-byte phase are
# corrupted, which points at an aggregate copy that derives the intra-word offset
# from the STATIC field offset instead of the dynamic element base.
#
# What has been ruled out, each verified on the 8060S:
#
#   sizeof 4 (2xInt16), 8 (4xInt16), 8 (Int16+Int32 padded)   all correct
#   sizeof 3 (3xInt8)                                          correct
#       -> not simply "stride not a multiple of 4"; it needs 16-bit fields whose
#          absolute offset alternates phase across elements
#   copyto!(dst, src)                                          correct
#       -> not the array copy path
#   dst .= Ref(S3I16(11,22,33))   (constant store, no load)    correct
#       -> not the store on its own
#   reading src[i] on GPU and writing .a/.b/.c to three arrays correct
#       -> not the load on its own
#
# So it is specifically the broadcast forwarding a loaded aggregate into a store.
#
# Marked @test_broken deliberately rather than left unregistered: a gated test
# that nobody runs rots. Two such tests in this suite (mwe_double_indirect.jl and
# test_int32_cartesian_miscompile.jl) were fixed without anyone noticing, because
# they were both gated AND absent from runtests.jl. @test_broken reports
# "Unexpected Pass" the moment this is fixed, which is exactly how that was found.
#
# The broader test_struct_alignment_systematic.jl fails on this same case (S09)
# and stays unregistered until this is fixed.

using Test, Lava

struct BcastStride6; a::Int16; b::Int16; c::Int16; end

@testset "broadcast of a 6-byte struct keeps every field" begin
    @test sizeof(BcastStride6) == 6          # the precondition the bug needs

    n = 8
    src_data = [BcastStride6(Int16(i), Int16(i * 3), Int16(i * 7)) for i in 1:n]
    src = Lava.LavaArray(src_data)
    dst = Lava.LavaArray{BcastStride6}(undef, n)
    dst .= src
    Lava.vk_flush!(Lava.vk_context())
    @test_broken Array(dst) == src_data

    # The paths that must stay correct, so a future fix cannot regress them.
    dst2 = Lava.LavaArray{BcastStride6}(undef, n)
    copyto!(dst2, src)
    Lava.vk_flush!(Lava.vk_context())
    @test Array(dst2) == src_data

    dst3 = Lava.LavaArray{BcastStride6}(undef, n)
    dst3 .= Ref(BcastStride6(Int16(11), Int16(22), Int16(33)))
    Lava.vk_flush!(Lava.vk_context())
    @test all(==(BcastStride6(Int16(11), Int16(22), Int16(33))), Array(dst3))
end
