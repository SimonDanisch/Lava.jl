# Regression test for a Lava miscompile, currently EXPECTED TO FAIL.
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
# real speedup — the same narrowing inside `LavaDNN.im2col_kernel!` (four
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
const KA = KernelAbstractions

@kernel function bcast_narrow_index!(dest, bc, cis, n, ::Val{NARROW}) where {NARROW}
    I = @index(Global, Linear)
    if I <= n
        J = NARROW ? (@inbounds cis[I % Int32]) : (@inbounds cis[I])
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
        # `preprocess` is what `_copyto!` does, and it is load-bearing here: it
        # wraps the operands in `Extruded`, and without it one of these two
        # shapes computes the right answer even with the narrow index.
        dummy = KA.allocate(be, Float32, sz...)
        bc = Broadcast.preprocess(dummy,
                Broadcast.instantiate(Broadcast.broadcasted(*, src, 3f0)))
        results = map((false, true)) do narrow
            out = KA.allocate(be, Float32, n); fill!(out, 0f0)
            bcast_narrow_index!(be, 64)(out, bc, cis, n, Val(narrow); ndrange = n)
            KA.synchronize(be)
            reshape(Array(out), sz)
        end
        @test results[1] ≈ want                  # wide index: correct
        @test_broken results[2] ≈ want           # narrow index: the miscompile
    end
end
