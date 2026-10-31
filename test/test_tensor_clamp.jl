using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMc = Lava.AcceleratedMatrix

# A CLAMPING layout makes an out-of-range block legal, and this is the claim the
# whole tensor-addressing port rests on.
#
# `gemm_padn` pads N 1500 -> 1536, `GEMM_BLOCK` rounds M, `padtile`/`crsextent`
# round K, and `gemm_divides` declines shapes outright — every one of them exists
# because `OpCooperativeMatrixLoadKHR` has no bounds checking, so extents must be
# arranged to divide the tile. Under `TENSOR_CLAMP_CONSTANT` the driver
# bounds-checks the load itself.
#
# MEASURED, on a 17x17 tensor (an extent that divides nothing) read with a 16x16
# tile at offsets that run off the end:
#
#     offset 0   0 out-of-range    in-range correct
#     offset 4  87 out-of-range    in-range correct   out-of-range == 0.0 exactly
#     offset 8 175 out-of-range    in-range correct   out-of-range == 0.0 exactly
#
# No fault, no garbage, exact zeros — and a zero contributes nothing to a dot
# product, so an unpadded extent Just Works for a GEMM.
#
# THE OBJECT IS NOT WHAT OUT-OF-RANGE ELEMENTS KEEP. An earlier note here said it
# was. The `%object` operand is loaded full of a sentinel below precisely so the
# two answers are distinguishable, and the out-of-range elements come back 0, not
# the sentinel. Zeroing the object would have made this untestable — zero is also
# the plausible clamp constant.
const EXT_C = 17
const TC = 16
const SENT = -999.0f0

@kernel cpu = false function tensorclamp_kernel!(out, @Const(src), @Const(sentinel), off::Int32)
    lay = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT_C), Int32(EXT_C))),
            (off, off), (Int32(TC), Int32(TC)))
    obj = AMc{Float32,TC,TC,Lava.Accumulator}(pointer(sentinel), 1, TC)
    Lava.copyto!(pointer(out), 1, TC,
                 Lava.tensor_load(obj, UInt64(pointer(src)), lay))
end

@testset "a clamping layout makes an unpadded extent legal" begin
    ctx = Lava.vk_context()
    if !ctx.coopmat2.tensor_addressing
        @info "device has no coopmat2 tensor addressing — skipping"
    else
        back = LavaBackend()
        WG = Lava.device_subgroup_size(ctx)
        src = KA.allocate(back, Float32, EXT_C, EXT_C)
        s = Float32.(reshape(1:(EXT_C * EXT_C), EXT_C, EXT_C))
        copyto!(src, s)
        sentinel = KA.allocate(back, Float32, TC, TC); fill!(sentinel, SENT)
        out = KA.allocate(back, Float32, TC, TC)

        for off in Int32.((0, 4, 8))
            fill!(out, NaN32)
            tensorclamp_kernel!(back, (Int(WG),))(out, src, sentinel, off;
                                                  ndrange = (Int(WG),))
            KA.synchronize(back)
            o = Array(out)
            # The load transposes (dims are reversed for a column-major array),
            # so tile (i,j) comes from s[off+j, off+i].
            inr = [(off + j <= EXT_C && off + i <= EXT_C) for i in 1:TC, j in 1:TC]
            @test all(o[i, j] == s[off + j, off + i]
                      for i in 1:TC, j in 1:TC if inr[i, j])
            oo = [o[i, j] for i in 1:TC, j in 1:TC if !inr[i, j]]
            if !isempty(oo)
                @test all(==(0.0f0), oo)      # the clamp constant
                @test !any(==(SENT), oo)      # NOT the object's value
                @test !any(isnan, oo)         # and it was written at all
            end
        end
    end
end
