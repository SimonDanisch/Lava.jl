# Prototype port of `mul_mm.comp`'s scalar (non-COOPMAT) branch: a register-
# blocked, shared-memory-staged fp32 GEMM. Reference vendored at
# dev/Lava/reference/mul_mm/.
#
# Scoped to exact multiples (M%BM == N%BN == K%BK == 0) for now. The reference
# carries bounds checks in staging and store; those come next, not dropped.
#
# ── Two deliberate deviations from the reference, both noted rather than silent
#
# 1. LAYOUT. ggml holds A row-major over (m, k) so its staging walks k
#    contiguously; a `LavaArray` is column-major M x K, so the contiguous axis is
#    m. The SHARED layout is kept identical (`bufa[m * SHF + k]`); only which
#    index runs fastest in the staging loop changes, so consecutive lanes still
#    read consecutive global addresses on both sides.
#
# 2. NO V2 ACCUMULATOR PACKING. The reference keeps `sums` as `ACC_TYPEV2` and
#    indexes `.x`/`.y` for two adjacent output ROWS, with a flattened `sums_idx`.
#    That packing is a GLSL vector-type artifact: its shared loads and
#    `dot_product` are vector-typed, and a flat array is the only way to index a
#    register block in GLSL. Here each accumulator is named by its four loop
#    indices via `@nexprs`. Identical arithmetic, identical register count,
#    identical store order (rows `drw + 0 .. drw + TM-1`); what goes away is the
#    index arithmetic, which cannot be expressed in an `@nexprs` symbol anyway.
module GemmProto

using KernelAbstractions, Lava
const KA = KernelAbstractions

# The reference's shipped F32 configuration, at a 64-wide subgroup.
const BM, BN, BK = 64, 64, 32        # BK = 32 with BK_STEP = 4 is the F32/F16 pair
const WM, WN = 32, 32
const WMITER, TM, TN = 2, 4, 2
const WARP = 64
const BKSTEP = 4

const NUM_WARPS = (BM ÷ WM) * (BN ÷ WN)                # 4
const WG = NUM_WARPS * WARP                            # 256
const WNITER = (WM * WN) ÷ (WARP * TM * TN * WMITER)   # 1
const WSUBM, WSUBN = WM ÷ WMITER, WN ÷ WNITER          # 16, 32
const SHF = BK + 2                  # SHMEM_STRIDE = BK/2 + 1 V2s, expressed in floats
const AREPS, BREPS = (BM * BK) ÷ WG, (BK * BN) ÷ WG    # 8, 8
const NKSTEP = BK ÷ BKSTEP                             # 8 k-steps of 4

@eval begin
    # `unsafe_indices=true` for the reason gemm.jl:609 records: KA's __validindex
    # guard is dead when the launch is an exact multiple and the kernel derives
    # indices from group/local ids, and leaving it in kept the inner muladd under
    # an OpSelectionMerge (3x).
    @kernel cpu=false unsafe_indices=true function gemm_fp32_staged!(
            C, @Const(A), @Const(B), ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
        bufa = @localmem Float32 ($BM * $SHF,)
        bufb = @localmem Float32 ($BN * $SHF,)

        tid = @index(Local, Linear) - 1
        blk = @index(Group, Linear) - 1
        nblk_m = M ÷ $BM
        ir = (blk % nblk_m) * $BM          # this workgroup's first row of C
        ic = (blk ÷ nblk_m) * $BN          # ...and its first column

        tiw    = tid % $WARP
        warp_i = tid ÷ $WARP
        tiwr   = tiw % ($WSUBM ÷ $TM)
        tiwc   = tiw ÷ ($WSUBM ÷ $TM)
        warp_r = warp_i % ($BM ÷ $WM)
        warp_c = warp_i ÷ ($BM ÷ $WM)

        Base.Cartesian.@nexprs $WNITER wsic -> Base.Cartesian.@nexprs $TN cc ->
            Base.Cartesian.@nexprs $WMITER wsir -> Base.Cartesian.@nexprs $TM jj ->
                s_wsic_cc_wsir_jj = 0.0f0

        for kb in 0:(K ÷ $BK - 1)
            k0 = kb * $BK
            # A block: rows [ir, ir+BM) x cols [k0, k0+BK); `m` runs fastest.
            @inbounds for r in 0:($AREPS - 1)
                idx = tid + r * $WG
                m, kk = Lava.splitidx(idx, Val($BM))
                bufa[1 + m * $SHF + kk] = A[1 + (ir + m) + (k0 + kk) * M]
            end
            # B block: rows [k0, k0+BK) x cols [ic, ic+BN); `k` runs fastest.
            @inbounds for r in 0:($BREPS - 1)
                idx = tid + r * $WG
                kk, n = Lava.splitidx(idx, Val($BK))
                bufb[1 + n * $SHF + kk] = B[1 + (k0 + kk) + (ic + n) * K]
            end
            @synchronize

            @inbounds Base.Cartesian.@nexprs $NKSTEP ii -> begin
                kof = (ii - 1) * $BKSTEP
                # cache_a[WMITER*TM], four k-values each (the reference's V4).
                Base.Cartesian.@nexprs $WMITER wsir -> Base.Cartesian.@nexprs $TM jj -> begin
                    ab_wsir_jj = 1 + (warp_r * $WM + (wsir - 1) * $WSUBM +
                                      tiwr * $TM + (jj - 1)) * $SHF + kof
                    a_wsir_jj_0 = bufa[ab_wsir_jj]
                    a_wsir_jj_1 = bufa[ab_wsir_jj + 1]
                    a_wsir_jj_2 = bufa[ab_wsir_jj + 2]
                    a_wsir_jj_3 = bufa[ab_wsir_jj + 3]
                end
                # One B vector live at a time, reloaded in the innermost loop:
                # the reference's ordering, and the register discipline gemm.jl:48
                # records losing 64 registers to when the first port hoisted it.
                Base.Cartesian.@nexprs $WNITER wsic -> Base.Cartesian.@nexprs $TN cc -> begin
                    bb = 1 + (warp_c * $WN + (wsic - 1) * $WSUBN +
                              tiwc * $TN + (cc - 1)) * $SHF + kof
                    b0 = bufb[bb]; b1 = bufb[bb + 1]
                    b2 = bufb[bb + 2]; b3 = bufb[bb + 3]
                    Base.Cartesian.@nexprs $WMITER wsir -> Base.Cartesian.@nexprs $TM jj ->
                        s_wsic_cc_wsir_jj =
                            muladd(a_wsir_jj_0, b0,
                            muladd(a_wsir_jj_1, b1,
                            muladd(a_wsir_jj_2, b2,
                            muladd(a_wsir_jj_3, b3, s_wsic_cc_wsir_jj))))
                end
            end
            @synchronize
        end

        @inbounds Base.Cartesian.@nexprs $WNITER wsic -> Base.Cartesian.@nexprs $WMITER wsir -> begin
            drw_wsic_wsir = ir + warp_r * $WM + (wsir - 1) * $WSUBM + tiwr * $TM
            dcw_wsic_wsir = ic + warp_c * $WN + (wsic - 1) * $WSUBN + tiwc * $TN
            Base.Cartesian.@nexprs $TN cc -> Base.Cartesian.@nexprs $TM jj ->
                C[1 + (drw_wsic_wsir + (jj - 1)) +
                      (dcw_wsic_wsir + (cc - 1)) * M] = s_wsic_cc_wsir_jj
        end
    end
end

function gemm!(C, A, B)
    M, K = size(A); N = size(B, 2)
    (M % BM == 0 && N % BN == 0 && K % BK == 0) ||
        error("prototype requires M%$BM==0, N%$BN==0, K%$BK==0; got ($M,$N,$K)")
    gemm_fp32_staged!(KA.get_backend(C), WG)(C, A, B, Val(M), Val(N), Val(K);
                                             ndrange = (M ÷ BM) * (N ÷ BN) * WG)
    C
end

end # module
