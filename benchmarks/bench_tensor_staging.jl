# Does removing the shared-memory STAGING pay?
#
# This is the last of tensor addressing's three claims. The load is free
# (0.998x) and the clamping layout makes unpadded extents legal (exact zeros out
# of range); both are measured. Staging is the one left, and it is the one the
# benchmarks so far could not touch, because none of them wrote shared memory.
#
# Staging only earns its cost when SEVERAL subgroups read the same block — one
# subgroup staging for itself is pure overhead, and a comparison built that way
# would hand the win to tensor addressing for free. So: four subgroups per
# workgroup, all consuming the same B block, differing only in how they get it.
#
#   staged   one cooperative fill into @localmem, @synchronize, four reads from
#            shared — one global read of B per k-step, plus two barriers
#   tensor   each subgroup tensor-loads B itself — four global reads of B per
#            k-step, no shared memory, no barriers
#
# A is loaded identically in both arms, so only B's path differs.
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMs = Lava.AcceleratedMatrix

const TS = 16
const EXTS = 1024
const KSTEPS = 256
const NSUB = 4              # subgroups per workgroup — the reuse factor

@kernel cpu = false unsafe_indices = true function staged_bench!(out, @Const(A), @Const(B))
    sh = @localmem Float16 (TS * TS,)
    tid = @index(Local, Linear) - 1
    nt = NSUB * 32
    acc = Lava.coopmat_zero(AMs{Float32,TS,TS,Lava.Accumulator})
    @inbounds for k in 0:(KSTEPS - 1)
        base = (k % (EXTS ÷ TS)) * TS * EXTS
        for i in tid:nt:(TS * TS - 1)
            sh[1 + i] = B[1 + base + i]
        end
        @synchronize
        b = AMs{Float16,TS,TS,Lava.MatrixB}(sh, 1, TS)
        a = AMs{Float16,TS,TS,Lava.MatrixA}(pointer(A), 1 + base, EXTS)
        acc = Lava.coopmat_muladd(a, b, acc)
        @synchronize
    end
    tid < 32 && Lava.copyto!(pointer(out), 1, TS, acc)
end

# The staged arm above pays TWO barriers per k-step for one 16x16 block, which
# is the worst work-per-barrier ratio a staged kernel can have — Lava's real GEMM
# stages 64x128. Beating a strawman would prove nothing, so this arm stages
# `KGROUP` k-blocks at once and pays the same two barriers across all of them.
const KGROUP = 4

@kernel cpu = false unsafe_indices = true function staged_amortised_bench!(out, @Const(A), @Const(B))
    sh = @localmem Float16 (TS * TS * KGROUP,)
    tid = @index(Local, Linear) - 1
    nt = NSUB * 32
    acc = Lava.coopmat_zero(AMs{Float32,TS,TS,Lava.Accumulator})
    @inbounds for kg in 0:((KSTEPS ÷ KGROUP) - 1)
        # Nested over (block, element) rather than one flat loop with `÷` and `%`
        # per element. The flat version measured SLOWER than staging one block
        # per k-step — the integer division cost more than the barriers it saved,
        # which is a confound, not a result.
        for j in 0:(KGROUP - 1)
            kk = (kg * KGROUP + j) % (EXTS ÷ TS)
            src = 1 + kk * TS * EXTS
            dst = j * TS * TS
            for i in tid:nt:(TS * TS - 1)
                sh[1 + dst + i] = B[src + i]
            end
        end
        @synchronize
        for j in 0:(KGROUP - 1)
            b = AMs{Float16,TS,TS,Lava.MatrixB}(sh, 1 + j * TS * TS, TS)
            base = ((kg * KGROUP + j) % (EXTS ÷ TS)) * TS * EXTS
            a = AMs{Float16,TS,TS,Lava.MatrixA}(pointer(A), 1 + base, EXTS)
            acc = Lava.coopmat_muladd(a, b, acc)
        end
        @synchronize
    end
    tid < 32 && Lava.copyto!(pointer(out), 1, TS, acc)
end

@kernel cpu = false unsafe_indices = true function tensorstage_bench!(out, @Const(A), @Const(B))
    tid = @index(Local, Linear) - 1
    lay = Lava.tensor_setdim(
              Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
              (Int32(EXTS), Int32(EXTS)))
    zb = Lava.coopmat_zero(AMs{Float16,TS,TS,Lava.MatrixB})
    acc = Lava.coopmat_zero(AMs{Float32,TS,TS,Lava.Accumulator})
    @inbounds for k in 0:(KSTEPS - 1)
        o = Int32((k % (EXTS ÷ TS)) * TS)
        sl = Lava.tensor_slice(lay, (Int32(0), o), (Int32(TS), Int32(TS)))
        b = Lava.tensor_load(zb, UInt64(pointer(B)), sl)
        a = AMs{Float16,TS,TS,Lava.MatrixA}(pointer(A), 1 + (k % (EXTS ÷ TS)) * TS * EXTS, EXTS)
        acc = Lava.coopmat_muladd(a, b, acc)
    end
    tid < 32 && Lava.copyto!(pointer(out), 1, TS, acc)
end

ctx = Lava.vk_context()
if !ctx.coopmat2.tensor_addressing
    @info "no coopmat2 tensor addressing on this device"
else
    back = LavaBackend()
    sg = Lava.device_subgroup_size(ctx)
    WGs = NSUB * Int(sg)
    A = KA.allocate(back, Float16, EXTS, EXTS)
    B = KA.allocate(back, Float16, EXTS, EXTS)
    copyto!(A, Float16.(reshape(1:(EXTS * EXTS), EXTS, EXTS) ./ (EXTS * EXTS)))
    copyto!(B, Float16.(reshape((EXTS * EXTS):-1:1, EXTS, EXTS) ./ (EXTS * EXTS)))
    out = KA.allocate(back, Float32, TS, TS)

    arms = ("staged (2 bar / k-step)"  => () -> staged_bench!(back, (WGs,))(out, A, B; ndrange = (WGs,)),
            "staged (2 bar / $KGROUP steps)" => () -> staged_amortised_bench!(back, (WGs,))(out, A, B; ndrange = (WGs,)),
            "tensor (no staging)"      => () -> tensorstage_bench!(back, (WGs,))(out, A, B; ndrange = (WGs,)))
    run(f) = (f(); KA.synchronize(back))
    for (_, f) in arms; run(f); end

    samples = Dict(n => Float64[] for (n, _) in arms)
    for _ in 1:15, (n, f) in arms
        t0 = time_ns(); run(f); push!(samples[n], (time_ns() - t0) / 1e6)
    end
    med(v) = (w = sort(v); w[cld(length(w), 2)])
    for (n, _) in arms
        v = samples[n]
        @printf("%-26s %7.3f ms   (min %.3f)\n", n, med(v), minimum(v))
    end
    t = med(samples["tensor (no staging)"])
    for key in ("staged (2 bar / k-step)", "staged (2 bar / $KGROUP steps)")
        r = t / med(samples[key])
        @printf("\ntensor / %-24s = %.3fx  %s\n", key, r,
                r < 1 ? "(dropping the staging WINS)" : "(staging wins)")
    end
    println("\n$NSUB subgroups share each B block: the staged arms read B once per k-step,")
    println("the tensor arm reads it $NSUB times. The second staged arm amortises its")
    println("barriers over $KGROUP k-steps — compare against THAT one, since a real GEMM")
    println("stages a 64x128 block and does not pay two barriers per 16x16.")
end
