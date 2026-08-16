# Where the tensor GEMM's time actually goes — measured, not inferred.
#
# WHY THIS FILE EXISTS. Five explanations for the tensor-addressed GEMM being
# 2.3x slower than the staged one have each failed a predictive test: register
# pressure, occupancy, instruction count, global traffic, and per-slice
# addressing registers (see bench_tensor_gemm_full.jl). Every one of the five was
# read off a STATIC property of the kernel — a count, a limit, a ratio. Not one
# measured how the kernel spends time. That is the common factor in five misses,
# and it is what this file changes.
#
# Two probes, chosen because neither perturbs register pressure — which the
# obvious probe (double the muladds, double the loads) does, and every variant of
# this kernel past 2x2 is already pegged at the 255-register cap, so a
# register-perturbing ablation cannot be read.
#
# PROBE 1 — K-sweep. Time against K at fixed M, N. The slope is the cost of one
# k-iteration in steady state; the intercept is launch + prologue + epilogue.
# Whether the 2.3x lives in the loop or in the setup is not currently known, and
# it decides where to look next. Fit both kernels.
#
# PROBE 2 — L1-resident arm. The same kernel with the k offset removed, so every
# iteration re-reads the SAME tile and the working set collapses into L1. The
# delta against the real kernel is time spent waiting on memory.
#
# THE CONTROL THAT MAKES PROBE 2 MEAN ANYTHING: run it on the STAGED kernel too.
# An earlier session measured the tensor kernel's L1-resident arm at 58.9 TF/s
# against 18.0 honest and correctly refused to quote it as a speedup — but that
# 3.3x is a real measurement of memory stall, and it was never compared against
# the staged kernel's own L1-resident arm. If staged gains the same 3.3x, both
# are memory-bound and this says nothing about the difference. If staged gains
# much less, the tensor kernel is stalling where staged is not, and that is the
# first positive finding in this whole investigation.
#
# A NOTE ON THE ARITHMETIC CEILING. Neither kernel can beat the tensor cores'
# issue rate for its muladd count, and the 4x2 tensor kernel issues 8 muladds per
# k-step against the staged kernel's own count at a different tile — so "muladds
# per unit output" is NOT equal between them and the comparison has to be per
# output element, not per k-step. Getting that denominator wrong is what made an
# earlier maxpool result read as a 15% win that did not exist.
using Lava, KernelAbstractions, Printf, Statistics
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

"""
    ksweep(run, M, N, Ks; reps) -> (slope_ms_per_kstep, intercept_ms, points)

Time `run(M, N, K)` across `Ks` and least-squares fit time against the number of
k-steps. `run` must dispatch many times per sync — a per-call sync measures the
clock ramp, not the kernel (the same staged kernel reads 3.3 TF/s cold and 40.5
warm).
"""
function ksweep(run, M, N, Ks; reps = 20)
    pts = Tuple{Float64,Float64}[]
    for K in Ks
        run(M, N, K)                                     # warm this shape
        ts = [batched(() -> run(M, N, K), reps) for _ in 1:7]
        push!(pts, (K / 16, median(ts)))
    end
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    x̄, ȳ = mean(xs), mean(ys)
    slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄) .^ 2)
    (slope, ȳ - slope * x̄, pts)
end

"""Median of `reps` dispatches per sync, in ms per dispatch."""
function batched(f, reps)
    KA.synchronize(LavaBackend())
    t0 = time_ns()
    for _ in 1:reps; f(); end
    KA.synchronize(LavaBackend())
    (time_ns() - t0) / 1e6 / reps
end

"""
    memory_stall_fraction(real_run, l1_run; reps) -> (t_real, t_l1, fraction)

`1 - t_l1/t_real` — the share of runtime that disappears when the working set
fits in L1. Run this on BOTH kernels or it is uninterpretable: a large number
means "memory-bound", which is unremarkable on its own and only becomes evidence
as a DIFFERENCE between the two kernels.
"""
function memory_stall_fraction(real_run, l1_run; reps = 20)
    real_run(); l1_run()
    tr = median([batched(real_run, reps) for _ in 1:7])
    tl = median([batched(l1_run, reps) for _ in 1:7])
    (tr, tl, 1 - tl / tr)
end

function report(name, slope, intercept, pts)
    @printf("%-22s  %.5f ms/k-step   intercept %.4f ms\n", name, slope, intercept)
    for (x, y) in pts
        @printf("      %5d k-steps  %8.4f ms\n", Int(x), y)
    end
end

# ── MEASURED: pending. Filled in from a run with the GPU otherwise idle —
# a concurrent test suite or a second REPL on this shared card invalidates
# absolute times, and the K-sweep's intercept is exactly the quantity that
# background load corrupts first.
