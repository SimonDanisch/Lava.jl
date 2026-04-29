# Lava.jl vs AMDGPU.jl benchmark suite
#
# Usage: include this file from a Julia session with Lava and AMDGPU loaded.
#   using Lava, AMDGPU, KernelAbstractions, AcceleratedKernels
#   include("dev/Lava/benchmarks/run_benchmarks.jl")
#
# Results are printed to stdout and saved to benchmarks/results_<date>.md

using Dates
import KernelAbstractions as KA
import GeometryBasics

const SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
const SORT_SIZES = [1_000, 10_000, 100_000, 1_000_000]
const SEARCH_SIZES = [10_000, 100_000, 1_000_000]
const WARMUP = 3
const TRIALS = 10

# ── Benchmark helper ──

struct BenchResult
    label::String
    n::Int
    median_us::Float64
    min_us::Float64
end

function bench_gpu(f::Function, label::String, n::Int; warmup=WARMUP, trials=TRIALS)
    for _ in 1:warmup
        f()
    end
    times = Float64[]
    for _ in 1:trials
        t = @elapsed f()
        push!(times, t)
    end
    med = sort(times)[div(length(times), 2) + 1]
    mn = minimum(times)
    med_us = round(med * 1e6, digits=1)
    min_us = round(mn * 1e6, digits=1)
    println("  $label N=$n: median=$(med_us)μs  min=$(min_us)μs")
    return BenchResult(label, n, med_us, min_us)
end

# ── Benchmarks ──

function run_all_benchmarks()
    results = BenchResult[]
    has_amdgpu = isdefined(Main, :AMDGPU) && AMDGPU.functional()

    println("=" ^ 70)
    println("  Lava.jl Benchmark Suite — $(Dates.today())")
    println("  GPU: AMD RX 7900 XTX, Vulkan 1.4")
    has_amdgpu && println("  AMDGPU: $(pkgversion(AMDGPU)) (ROCBackend comparison)")
    println("=" ^ 70)

    # ── Broadcast ──
    println("\n▶ BROADCAST: a .= b .+ c .* d (Float32)")
    for N in SIZES
        b_cpu = rand(Float32, N); c_cpu = rand(Float32, N); d_cpu = rand(Float32, N)

        a_l = Lava.LavaArray(zeros(Float32, N))
        b_l = Lava.LavaArray(b_cpu); c_l = Lava.LavaArray(c_cpu); d_l = Lava.LavaArray(d_cpu)
        push!(results, bench_gpu("lava_broadcast", N) do
            a_l .= b_l .+ c_l .* d_l
            KA.synchronize(Lava.LavaBackend())
        end)

        if has_amdgpu
            a_r = ROCArray(zeros(Float32, N))
            b_r = ROCArray(b_cpu); c_r = ROCArray(c_cpu); d_r = ROCArray(d_cpu)
            push!(results, bench_gpu("amdgpu_broadcast", N) do
                a_r .= b_r .+ c_r .* d_r
                AMDGPU.synchronize()
            end)
        end
    end

    # ── Reduction ──
    println("\n▶ REDUCTION: sum(a) (Float32)")
    for N in SIZES
        a_cpu = rand(Float32, N)

        a_l = Lava.LavaArray(a_cpu)
        push!(results, bench_gpu("lava_sum", N) do
            sum(a_l)
        end)

        if has_amdgpu
            a_r = ROCArray(a_cpu)
            push!(results, bench_gpu("amdgpu_sum", N) do
                sum(a_r)
            end)
        end
    end

    # ── Math map ──
    println("\n▶ MATH MAP: a .= sin.(b) .+ cos.(c) (Float32)")
    for N in SIZES
        b_cpu = rand(Float32, N); c_cpu = rand(Float32, N)

        a_l = Lava.LavaArray(zeros(Float32, N))
        b_l = Lava.LavaArray(b_cpu); c_l = Lava.LavaArray(c_cpu)
        push!(results, bench_gpu("lava_sincos", N) do
            a_l .= sin.(b_l) .+ cos.(c_l)
            KA.synchronize(Lava.LavaBackend())
        end)

        if has_amdgpu
            a_r = ROCArray(zeros(Float32, N))
            b_r = ROCArray(b_cpu); c_r = ROCArray(c_cpu)
            push!(results, bench_gpu("amdgpu_sincos", N) do
                a_r .= sin.(b_r) .+ cos.(c_r)
                AMDGPU.synchronize()
            end)
        end
    end

    # ── Dot product ──
    println("\n▶ DOT PRODUCT: mapreduce(*, +, a, b) (Float32)")
    for N in SIZES
        a_cpu = rand(Float32, N); b_cpu = rand(Float32, N)

        a_l = Lava.LavaArray(a_cpu); b_l = Lava.LavaArray(b_cpu)
        push!(results, bench_gpu("lava_dot", N) do
            mapreduce(*, +, a_l, b_l)
        end)

        if has_amdgpu
            a_r = ROCArray(a_cpu); b_r = ROCArray(b_cpu)
            push!(results, bench_gpu("amdgpu_dot", N) do
                mapreduce(*, +, a_r, b_r)
            end)
        end
    end

    # ── KA SAXPY ──
    println("\n▶ KA SAXPY: a[i] = a[i] + α*b[i] (Float32)")
    @kernel function _saxpy_bench!(a, b, α)
        i = @index(Global)
        @inbounds a[i] = a[i] + α * b[i]
    end

    for N in SIZES
        a_cpu = rand(Float32, N); b_cpu = rand(Float32, N)
        α = Float32(2.5)

        a_l = Lava.LavaArray(copy(a_cpu)); b_l = Lava.LavaArray(b_cpu)
        push!(results, bench_gpu("lava_saxpy", N) do
            _saxpy_bench!(Lava.LavaBackend())(a_l, b_l, α; ndrange=N)
            KA.synchronize(Lava.LavaBackend())
        end)

        if has_amdgpu
            a_r = ROCArray(copy(a_cpu)); b_r = ROCArray(b_cpu)
            push!(results, bench_gpu("amdgpu_saxpy", N) do
                _saxpy_bench!(ROCBackend())(a_r, b_r, α; ndrange=N)
                AMDGPU.synchronize()
            end)
        end
    end

    # ── Grain instances (MVector + Vec3f + dual TLAS instance write) ──
    # Exercises the MVector{N, Vec3f} and LavaInstanceRecord-write paths.
    # This is the load-bearing benchmark for the alloca-retyping pass: if the
    # retype rewrites Function-storage MVector allocas, this kernel's compile
    # and runtime perf should stay within ~5% of baseline.
    println("\n▶ Grain instances: write_grain_instances_kernel (Vec3f + Vec4f)")
    let
        for N in (10_000, 100_000, 1_000_000)
            positions = Lava.LavaArray([GeometryBasics.Point3f(Float32(i), 0f0, 0f0) for i in 1:N])
            quats     = Lava.LavaArray([GeometryBasics.Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:N])
            instances = Lava.LavaArray{Lava.LavaInstanceRecord}(undef, 2 * N;
                            extra_usage=Lava.AS_INPUT_USAGE)
            push!(results, bench_gpu("lava_grain_instances", N) do
                Lava.write_grain_instances_kernel(Lava.LavaBackend())(
                    positions, quats, 1f0, UInt64(0), UInt64(0), instances; ndrange=N)
                KA.synchronize(Lava.LavaBackend())
            end)
        end
    end

    # ── AK sort ──
    println("\n▶ AK.sort! (Float32)")
    for N in SORT_SIZES
        data_cpu = rand(Float32, N)

        push!(results, bench_gpu("lava_ak_sort", N) do
            a = Lava.LavaArray(copy(data_cpu))
            AcceleratedKernels.sort!(a)
            KA.synchronize(Lava.LavaBackend())
        end)

        if has_amdgpu
            try
                bench_gpu("amdgpu_ak_sort", N) do
                    a = ROCArray(copy(data_cpu))
                    AcceleratedKernels.sort!(a)
                    AMDGPU.synchronize()
                end
            catch e
                println("  amdgpu_ak_sort N=$N: FAILED ($(typeof(e)))")
                push!(results, BenchResult("amdgpu_ak_sort", N, NaN, NaN))
            end
        end
    end

    # ── AK searchsortedfirst ──
    println("\n▶ AK.searchsortedfirst! (Float32)")
    for N in SEARCH_SIZES
        sorted_cpu = sort(rand(Float32, N))
        needles_cpu = rand(Float32, N ÷ 10)

        push!(results, bench_gpu("lava_ak_search", N) do
            sorted = Lava.LavaArray(sorted_cpu)
            needles = Lava.LavaArray(needles_cpu)
            result = Lava.LavaArray(zeros(Int, length(needles_cpu)))
            AcceleratedKernels.searchsortedfirst!(result, sorted, needles)
            KA.synchronize(Lava.LavaBackend())
        end)
    end

    return results
end

# ── Run and save ──

results = run_all_benchmarks()

# Save raw data as CSV for tracking
outdir = joinpath(@__DIR__)
csv_path = joinpath(outdir, "results_$(Dates.today()).csv")
open(csv_path, "w") do io
    println(io, "date,label,n,median_us,min_us")
    for r in results
        println(io, "$(Dates.today()),$(r.label),$(r.n),$(r.median_us),$(r.min_us)")
    end
end
println("\nResults saved to: $csv_path")
