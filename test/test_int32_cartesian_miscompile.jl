# Regression test for a narrow-index miscompile.
#
# ── Answer: it is NVIDIA's shader compiler, and that is now demonstrated ──
#
# The same SPIR-V module, on two independent Vulkan implementations
# (`tools/narrow_index_second_driver.jl`):
#
#                                    rank 2 (control)   rank 3 (the bug)
#   NVIDIA RTX 4000 Ada                   exact              WRONG
#   lavapipe (llvmpipe, LLVM 22.1.8)      exact              exact
#
# The rank-2 control passes on both, so the drivers are comparable. **The module
# is correct and NVIDIA compiles it wrongly.** Everything below — every value
# checked and found right, the pass pipeline bisected, the LLVM IR verified — was
# the long way round to a question only a second driver could answer, and it is
# recorded because each step also rules something out for the next bug.
#
# That conclusion was twice refused earlier on weaker evidence ("valid SPIR-V,
# therefore the driver"), which is exactly how one talks oneself into a wrong
# answer. What makes it hold here is the control: an unrelated implementation
# running the identical bytes and getting it right.
#
# Reportable upstream as it stands: the reproducer is `identity` over a plain
# rank-3 array, preprocessed, indexed with a narrow linear index — see the shrink
# table below.
#
# ── Status: still broken here, and there is a workaround ──
#
# A previous revision of this header said "FIXED — this test now guards the fix".
# It does not: both narrow cases fail today, and they fail identically with every
# source change from the staged-GEMM session stashed, so this is not a fresh
# regression — the fix either never landed or was undone long before. Left as
# `@test_broken` rather than as two red assertions, because a permanently failing
# suite is a suite people stop reading.
#
# **It is `CartesianIndices` itself, not the division, and not the 32-bit width.**
# Four decompositions of the same linear index into the same `Broadcasted`, all
# asserted below:
#
#   cis[I]                                    64-bit, Base            correct
#   cis[I % Int32]                            32-bit, Base            WRONG
#   cart32(Int32(I-1), Int32 extents)         32-bit, hand-rolled ÷/% correct
#   cart32(UInt32(I-1), FastDiv32 extents)    32-bit, magic-number    correct
#
# The third row is the one that settles it. It is 32 bits *and* it divides, and
# it is exact — so neither the narrow width nor the division is the fault, and
# the earlier note about `OpSDiv` versus `OpUDiv` in the disassembly really was
# a red herring. What is left is Base's `CartesianIndices` decomposition when the
# index handed to it is narrow.
#
# Worth stating because the first guess was that this shared a cause with
# `test_shared_index_division.jl` — a real division in a shared-store index,
# found the same week, also silently wrong, also vanishing under instrumentation,
# also fixed by magic-number arithmetic. It does not: that one *is* the division,
# this one is not. Two bugs that look alike and are not.
#
# ── The emitted decomposition arithmetic is correct ──
#
# Dumping both modules (they differ only in the decomposition) and reading the
# narrow one's `%ulong` sequence by hand:
#
#   %141 = sext(I)                 ; the 1-based linear index
#   %143 = %141 - 1                ; 0-based
#   %146 = %143 / d1               ; q
#   %148 = %141 - %146*d1          ; = (I-1) % d1 + 1   <- the +1 is folded in
#   %152 = (%146 + 1) - %149*d2    ; = q % d2 + 1
#   %153 = %149 + 1
#
# All three components are right. The remainder looks wrong at a glance — it
# divides `%143` and subtracts from `%141` — and is not: that fold *is* the
# 1-based conversion.
#
# ── And the consumption is identical ──
#
# The two modules carry 8 and 34 `OpSelect`, which looks like the answer and is
# not. Re-running both with the components **written out** instead of used to
# index anything gives 5 and 31: the whole 26-select gap is `cart32`'s own
# divide-by-zero guards, unrolled. Indexing the `Broadcasted` adds exactly
# **three** selects to each — one per dimension of `Base.newindex` over
# `Extruded` — so the consumption is the same shape in the correct and the
# incorrect path.
#
# So: same emitted components, same consumption, different answers. Nothing
# structural is left, which agrees with the Heisenbug and points at liveness or
# scheduling around those values rather than at any instruction being wrong.
#
# ── A trap worth naming, because it looks like a result ──
#
# The obvious next move is to write `J[1]`, `J[2]`, `J[3]` out and check them.
# Done: they are exactly right in both paths. **That proves nothing about the
# failing kernel**, because storing the components *is* the second consumer that
# the Heisenbug note above says makes the fault disappear — the probe measures a
# kernel that no longer has the bug. Same for putting the stores and `bc[J]` in
# one kernel: the header's own history says that combination comes out correct.
#
# There is no runtime observation of this bug that does not destroy it. Whatever
# comes next has to be static, not another probe kernel.
#
# ── It is not the pipeline, and the wrong answer has a shape ──
#
# `LAVA_SKIP_PASSES` (see `passes/structurize_cfg.jl`) drops one structurization
# pass at a time. Every one of the eleven leaves the fault in place, and only
# `StructurizeCFG` is load-bearing enough to break compilation — so no pass in
# that pipeline introduces it. The post-pass LLVM IR is correct too, and the
# SPIR-V mirrors it 1:1: `%14 = ashr(shl(I,32),32)` is `sext(trunc(I))`, then
# `sdiv`/`mul`/`sub` with the same +1 fold verified above.
#
# What finally says something is the *shape* of the wrong output rather than the
# fact of it. For the view case, `got[i,j,k] == want[i,k,1]`, exactly:
#
#   delivered index   (c1, c3, 1)
#   correct index     (c1, c2, c3)
#
# The second component reads the **third**, and the third falls to its default.
# The component *values* are right — they are being extracted at the wrong
# offsets. That is a tuple/aggregate-extraction fault, not arithmetic, and it
# explains the Heisenbug exactly: storing `J[1]`, `J[2]`, `J[3]` forces the
# `CartesianIndex` to exist as a real aggregate, and the offsets come out right.
#
# Tracing that through the dump: the three `OpSelect`s are wired correctly
# (`%148`->dim1, `%152`->dim2, `%153`->dim3, and those are `c1`, `c2`, `c3` by
# the derivation above), and so is the address they feed —
# `(c2-1)*%71 + (c3-1)*%74*%71 + c1` with `%71 = 7`, `%74 = 5`, the right strides
# for a `(7,5,3)` view. Every step in isolation is right.
#
# ── The struct layout, so the next person does not have to derive it ──
#
# The device-side argument is
# `Broadcasted{…, Tuple{Extruded{SubArray{Float32,3,LavaDeviceArray{Float32,3},…}}, Float32}}`,
# and the byte offsets in the dump pin down every field — the `keeps` and
# `Idefault`s land exactly where this predicts, which is what confirms it:
#
#   +0         parent ptr              +48, +56    the two `Slice` axes
#   +8,16,24   parent dims (7,5,3)     +64, +72    offset1, stride1
#   +32, +40   UnitRange start, stop   +80,81,82   keeps
#                                      +88,96,104  Idefaults
#                                      +112        the Float32 operand
#
# The two modules load slightly different sets — `narrowci` reads +24 and never
# +40, `handdiv` the reverse — and that is **not** a misread: +24 is `dims[3]`
# and +40 is the range's `stop`, so it is two different constant-foldings of the
# same information. (Recorded because it looks like a smoking gun and is not.)
#
# With the layout known, `narrowci`'s address decodes as *correct*:
# `c1 + (c2-1)*7 + (c3-1)*35`, which is exactly the flat offset of element
# `(c1,c2,c3)` in a `(7,5,3)` parent viewed from row 2.
#
# ── Diffed against the true minimal pair, and everything agrees ──
#
# The right comparison is `:wide` against `:narrowci` — same Base
# `CartesianIndices`, only the index width differs — not `:handdiv`, which
# changes the decomposition as well. Doing that:
#
#   * the address computation is **instruction-for-instruction identical**, the
#     same nine-op sequence in both;
#   * it is fed from the **same struct offsets** in both: +16, +8, +32;
#   * the three `OpSelect`s pair each dimension with its own `keep` and its own
#     `Idefault` in both (the ids run in opposite order — the working module
#     emits its loads back to front — which looks like a reversal and is not);
#   * the divisors are loaded from the `cis` argument at +0 and +8, i.e. 5 and 5,
#     in both.
#
# So every value in the failing module is correct and matches the working one
# wherever the two can be compared. The only part that *must* differ is the
# decomposition itself, and that one was hand-verified correct above.
#
# That is where reading the disassembly runs out. The fault is not visible in
# any value, which is exactly what the Heisenbug has been saying all along.
#
# ── Shrinking it instead: rank >= 3 and `Extruded`, nothing else ──
#
# Sweeping rank, operand kind, whether `Broadcast.preprocess` is applied, and
# whether there is an arithmetic operand at all (`ok` = the narrow index is
# exact):
#
#   case              pre+op   pre,noop   nopre+op   nopre,noop
#   rank3 view         WRONG     WRONG        ok          ok
#   rank3 plain        WRONG     WRONG        ok          ok
#   rank3 plain 9x8x7  WRONG     WRONG        ok          ok
#   rank4 plain        WRONG     WRONG        ok          ok
#   rank2 view/plain      ok        ok        ok          ok
#   rank2 permuted        ok        ok        ok          ok
#   rank1 view            ok        ok        ok          ok
#
# **Rank >= 3 and `preprocess`.** Nothing else is needed: no view, no
# permutation, no arithmetic — `identity` over a plain rank-3 array is enough —
# and the extents do not matter. Rank 2 is never wrong, at any combination.
#
# `preprocess` is what wraps the operand in `Extruded`, and `Extruded` is what
# makes `bc[J]` **destructure the index tuple and rebuild it** through
# `newindex`. A plain rank-3 operand without it indexes directly and is exact.
# So the fault needs a >= 3-element index tuple to be taken apart and put back
# together — which is the same shape as the observed `(c1, c3, 1)`.
#
# That is a far smaller module than the original view-based case, and it is
# where a fresh attempt should start: `identity` over a plain rank-3 array,
# preprocessed, narrow index.
#
# Recovering a `CartesianIndex` from an `Int32` linear index, then using it to
# index a `Broadcasted`, produces wrong values on device. The same expression is
# correct in every simpler setting, which is what makes it a compiler bug rather
# than a misuse:
#
#   cis[I % Int32] -> write the components out          correct
#   cis[I % Int32] -> index a dense LavaArray           correct
#   cis[I % Int32] -> index a SubArray of a LavaArray   correct
#   cis[I % Int32] -> index a PermutedDimsArray         correct
#   cis[I % Int32] -> index a Broadcasted               WRONG
#
# It does not depend on the axes' element type: a `CartesianIndices` with `Int`
# axes indexed by an `Int32` fails identically, and widening the recovered index
# back to `Int` immediately afterwards does not help. Only the *index* being
# `Int32` matters.
#
# Why it is worth fixing rather than avoiding: narrowing this arithmetic is a
# real speedup — the same narrowing inside `DNNKernels.im2col_kernel!` (four
# integer divisions per element) moved a whole inference step from 35.7 ms to
# 32.7. Lava's broadcast kernels cannot take it until this is fixed, and any
# other kernel that reaches for a 32-bit index is exposed to the same silent
# wrong answer.
#
# Symptom when it regresses in the other direction: `Lava.lava_broadcast_flat_*`
# in `array/gpuarrays.jl` may go back to `cis[I % Int32]`.
#
# ── What is known about the mechanism ──
#
# It is a *codegen* fault, not a semantic one: adding a second consumer of the
# recovered index (storing `J[1]`, `J[2]`, `J[3]` to a side buffer) makes the
# wrong answer disappear while changing nothing about the computation. A bug
# that vanishes when you observe it is register allocation, liveness or DCE.
#
# Two hypotheses were tested and eliminated:
#   * `Base.newindex` does `ifelse(keep::Bool, I[k]::Int32, Idefault[k]::Int)`,
#     a mixed-width `ifelse` — but a direct probe of that compiles correctly.
#   * The narrowed `CartesianIndices` itself — the decomposition returns exactly
#     the right components when written straight out.
#
# In the emitted SPIR-V the two variants differ in their division: the narrow
# path uses `OpSDiv` (and computes the remainder as `a - (a/b)*b`) where the wide
# path uses `OpUDiv` / `OpUMod`. That difference is *expected* and not the bug —
# it mirrors LLVM's own `sdiv` vs `udiv` (`emit.jl` maps them 1:1), and SPIR-V's
# signed opcodes carry the signedness independently of the type's signedness bit,
# so `OpSDiv` on a type declared `OpTypeInt 64 0` is legal. Checked so the next
# person does not chase it.
#
# The narrow path is also *not* actually narrow: the `Int32` is `OpUConvert`ed
# back to 64 bits immediately, so the arithmetic happens at 64 bits either way
# and the only real difference is which division opcode and how the remainder is
# formed. Combined with the Heisenbug behaviour above, that points at the
# scheduling/liveness of those extra `OpIMul`/`OpISub` values rather than at any
# single instruction being wrong. Dumping both with
# `Lava.SPIRV_DUMP_DIR[] = "..."` and diffing is how to pick this up again.

using Test, Lava, KernelAbstractions
using Lava: FastDiv32, cart32
const KA = KernelAbstractions

# Four decompositions of the same linear index into the same `Broadcasted`. The
# point of running all four is to separate the three things that could be at
# fault — the 32-bit width, the division, and `CartesianIndices` — since only
# `:narrowci` has all three.
@kernel function bcast_index!(dest, bc, cis, n, fd, szi, ::Val{MODE}) where {MODE}
    I = @index(Global, Linear)
    if I <= n
        J = MODE === :wide     ? (@inbounds cis[I]) :                        # 64-bit, Base
            MODE === :narrowci ? (@inbounds cis[I % Int32]) :                # 32-bit, Base
            MODE === :handdiv  ? CartesianIndex(cart32(Int32(I) - Int32(1), szi)) :
                                 CartesianIndex(cart32(UInt32(I) - UInt32(1), fd))
        @inbounds dest[I] = bc[J]
    end
end

@testset "Int32 linear index into a Broadcasted" begin
    be = LavaBackend()
    host = reshape(collect(1f0:105f0), 7, 5, 3)
    A = KA.allocate(be, Float32, 7, 5, 3); copyto!(A, host)

    for (name, src, sz, want) in (
            ("view",     view(A, 2:6, :, :),          (5, 5, 3), host[2:6, :, :] .* 3),
            ("permuted", PermutedDimsArray(A, (2, 1, 3)), (5, 7, 3),
             permutedims(host, (2, 1, 3)) .* 3))
        n = prod(sz)
        cis = CartesianIndices(sz)
        fd = map(FastDiv32, sz)
        szi = map(Int32, sz)
        # `preprocess` is what `_copyto!` does, and it is load-bearing here: it
        # wraps the operands in `Extruded`, and without it one of these two
        # shapes computes the right answer even with the narrow index.
        dummy = KA.allocate(be, Float32, sz...)
        bc = Broadcast.preprocess(dummy,
                Broadcast.instantiate(Broadcast.broadcasted(*, src, 3f0)))
        run(mode) = begin
            out = KA.allocate(be, Float32, n); fill!(out, 0f0)
            bcast_index!(be, 64)(out, bc, cis, n, fd, szi, Val(mode); ndrange = n)
            KA.synchronize(be)
            reshape(Array(out), sz)
        end
        @test run(:wide) ≈ want              # 64-bit through Base: correct
        # Base's `CartesianIndices` under a narrow index: still wrong. Turns into
        # a failure the day it is fixed.
        @test_broken run(:narrowci) ≈ want
        # The two that isolate it. `:handdiv` is 32-bit AND divides and is exact,
        # which rules out both the width and the division; `:magic` is the form
        # Lava's broadcast kernels actually use.
        @test run(:handdiv) ≈ want
        @test run(:magic) ≈ want
    end
end

@testset "the boundary: rank 2 is exact, rank 3 preprocessed is not" begin
    # The smallest form of the bug — no view, no permutation, no arithmetic —
    # and the rank below it, which must stay correct. Pinning both means a change
    # in *either* direction shows up: a fix (rank 3 starts passing) or a spread
    # (rank 2 starts failing).
    be = LavaBackend()
    narrow(src, hsrc, sz) = begin
        n = prod(sz)
        dummy = KA.allocate(be, Float32, sz...)
        bc = Broadcast.preprocess(dummy, Broadcast.instantiate(
                 Broadcast.broadcasted(identity, src)))
        out = KA.allocate(be, Float32, n); fill!(out, 0f0)
        bcast_index!(be, 64)(out, bc, CartesianIndices(sz), n,
                             map(FastDiv32, sz), map(Int32, sz), Val(:narrowci);
                             ndrange = n)
        KA.synchronize(be)
        reshape(Array(out), sz) ≈ hsrc
    end
    h2 = reshape(collect(1f0:25f0), 5, 5)
    A2 = KA.allocate(be, Float32, 5, 5); copyto!(A2, h2)
    @test narrow(A2, h2, (5, 5))                 # rank 2: exact

    h3 = reshape(collect(1f0:75f0), 5, 5, 3)
    A3 = KA.allocate(be, Float32, 5, 5, 3); copyto!(A3, h3)
    @test_broken narrow(A3, h3, (5, 5, 3))       # rank 3 + Extruded: the bug
end
