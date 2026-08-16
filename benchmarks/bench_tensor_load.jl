# Is a tensor-addressed load actually FASTER than the strided load Lava already
# has? This decides whether the coopmat2 port is worth finishing, and it is
# cheap to answer, so it comes before building a GEMM around it.
#
# Both arms load the same tiles from the same array and feed them to the same
# `coopmat_muladd`, so the arithmetic is identical and only the LOAD differs:
#
#   strided   AcceleratedMatrix(ptr, offset, stride)  — OpCooperativeMatrixLoadKHR
#   tensor    tensor_load(zero, addr, sliced-layout)  — OpCooperativeMatrixLoadTensorNV
#
# The accumulator is stored at the end so nothing is dead, and the loop is long
# enough that the card stays clocked up — a short kernel measures the clock ramp
# instead of the kernel, which has produced several wrong numbers in this repo
# (`tools/measure.jl` documents them).
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMb = Lava.AcceleratedMatrix

const TB = 16            # tile
const EXT = 1024         # array is EXT x EXT
const REPS = 512         # tiles loaded per thread-group, per arm

@kernel cpu = false function strided_load_bench!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    for r in 0:(REPS - 1)
        off = 1 + (r % (EXT ÷ TB)) * TB * EXT
        a = AMb{Float16,TB,TB,Lava.MatrixA}(pointer(src), off, EXT)
        acc = Lava.coopmat_muladd(a, b, acc)
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

@kernel cpu = false function tensor_load_bench!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    base = Lava.tensor_setdim(
               Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
               (Int32(EXT), Int32(EXT)))
    za = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixA})
    for r in 0:(REPS - 1)
        o = Int32((r % (EXT ÷ TB)) * TB)
        l = Lava.tensor_slice(base, (Int32(0), o), (Int32(TB), Int32(TB)))
        a = Lava.tensor_load(za, UInt64(pointer(src)), l)
        acc = Lava.coopmat_muladd(a, b, acc)
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

# ── The two controls that separate "the slice is expensive" from "the load is
# expensive". Without them a single ratio blames whichever half you expected.
@kernel cpu = false function strided_fixed_bench!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    for _ in 0:(REPS - 1)
        a = AMb{Float16,TB,TB,Lava.MatrixA}(pointer(src), 1, EXT)
        acc = Lava.coopmat_muladd(a, b, acc)
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

@kernel cpu = false function tensor_fixed_bench!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    # slice HOISTED: the layout is built once, so the loop measures only the
    # load instruction against the strided one.
    l = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT), Int32(EXT))),
            (Int32(0), Int32(0)), (Int32(TB), Int32(TB)))
    za = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixA})
    for _ in 0:(REPS - 1)
        a = Lava.tensor_load(za, UInt64(pointer(src)), l)
        acc = Lava.coopmat_muladd(a, b, acc)
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

# ── Does the slice cost AMORTISE? The arms above do one muladd per load, the
# lowest intensity a GEMM can have. A 64x64 tile issues 16 muladds per pair of
# slices, so the prediction from the numbers above (slice/load ~= 2.24) is that
# the slice falls to 2.24/(16+2.24) ~= 12% of the loop. Predicted is not
# measured, so: same loop, 16 muladds per slice.
const MULS = 16

@kernel cpu = false function tensor_amortised_bench!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    base = Lava.tensor_setdim(
               Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
               (Int32(EXT), Int32(EXT)))
    za = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixA})
    for r in 0:((REPS ÷ MULS) - 1)
        o = Int32((r % (EXT ÷ TB)) * TB)
        l = Lava.tensor_slice(base, (Int32(0), o), (Int32(TB), Int32(TB)))
        a = Lava.tensor_load(za, UInt64(pointer(src)), l)
        for _ in 1:MULS
            acc = Lava.coopmat_muladd(a, b, acc)
        end
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

# The control for it: identical muladd count, slice hoisted. The ratio of these
# two is the real per-block slice overhead at GEMM intensity.
@kernel cpu = false function tensor_amortised_fixed!(out, @Const(src))
    acc = Lava.coopmat_zero(AMb{Float32,TB,TB,Lava.Accumulator})
    b = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixB})
    l = Lava.tensor_slice(
            Lava.tensor_setdim(
                Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                (Int32(EXT), Int32(EXT))),
            (Int32(0), Int32(0)), (Int32(TB), Int32(TB)))
    za = Lava.coopmat_zero(AMb{Float16,TB,TB,Lava.MatrixA})
    for _ in 0:((REPS ÷ MULS) - 1)
        a = Lava.tensor_load(za, UInt64(pointer(src)), l)
        for _ in 1:MULS
            acc = Lava.coopmat_muladd(a, b, acc)
        end
    end
    Lava.copyto!(pointer(out), 1, TB, acc)
end

ctx = Lava.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device"
else
    back = LavaBackend()
    WG = Lava.device_subgroup_size(ctx)
    src = KA.allocate(back, Float16, EXT, EXT)
    copyto!(src, Float16.(reshape(1:(EXT * EXT), EXT, EXT) ./ (EXT * EXT)))
    out = KA.allocate(back, Float32, TB, TB)

    arms = ("strided varying" => () -> strided_load_bench!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)),
            "strided fixed"   => () -> strided_fixed_bench!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)),
            "tensor varying"  => () -> tensor_load_bench!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)),
            "tensor fixed"    => () -> tensor_fixed_bench!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)),
            "amort varying"   => () -> tensor_amortised_bench!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)),
            "amort fixed"     => () -> tensor_amortised_fixed!(back, (Int(WG),))(out, src; ndrange = (Int(WG),)))
    run(f) = (f(); KA.synchronize(back))
    for (_, f) in arms; run(f); end       # compile every arm before timing any

    # INTERLEAVED, one sample of each arm per round, never a block per arm: a
    # block gives each arm its own stretch of the clock ramp, which is how a
    # 1.75x got invented here once.
    samples = Dict(name => Float64[] for (name, _) in arms)
    for _ in 1:15, (name, f) in arms
        t0 = time_ns(); run(f); push!(samples[name], (time_ns() - t0) / 1e6)
    end
    med(v) = (w = sort(v); w[cld(length(w), 2)])
    res = Dict(name => (med(v), minimum(v), maximum(v)) for (name, v) in samples)
    for (name, _) in arms
        m, lo, hi = res[name]
        @printf("%-16s %7.3f ms   (min %.3f, max %.3f)\n", name, m, lo, hi)
    end
    @printf("\nload instruction alone   tensor/strided = %.3fx  (fixed arms)\n",
            res["tensor fixed"][1] / res["strided fixed"][1])
    @printf("cost of re-slicing       tensor varying/fixed = %.3fx\n",
            res["tensor varying"][1] / res["tensor fixed"][1])
    @printf("cost of re-offsetting    strided varying/fixed = %.3fx\n",
            res["strided varying"][1] / res["strided fixed"][1])
    sl = res["tensor varying"][1] / res["tensor fixed"][1] - 1     # slice/load
    @printf("\nat %d muladds per slice  measured %.3fx overhead, predicted %.3fx\n",
            MULS, res["amort varying"][1] / res["amort fixed"][1],
            1 + sl / MULS)
    println("\nThe fixed arms isolate the LOAD; the varying/fixed ratios price the")
    println("address arithmetic each path needs per k-block. A single overall ratio")
    println("blames whichever half you already suspected.")
end
