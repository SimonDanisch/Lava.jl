# GPUArrays TestSuite runner for LavaArray
# Runs all test groups in-process with error isolation and device health monitoring.

using Lava
import GPUArrays
using Test

gpuarrays_testsuite = joinpath(dirname(dirname(pathof(GPUArrays))), "test", "testsuite.jl")
include(gpuarrays_testsuite)

# Supported element types — Lava supports a broad set via SPIR-V
TestSuite.supported_eltypes(::Type{<:LavaArray}) = (Int16, Int32, Int64,
                                                     Float16, Float32, Float64,
                                                     ComplexF16, ComplexF32, ComplexF64,
                                                     Complex{Int16}, Complex{Int32}, Complex{Int64})

# Disallow scalar indexing — matches CUDA/Metal/AMDGPU behavior
GPUArrays.allowscalar(false)

function count_results(ts::Test.DefaultTestSet)
    pass = 0; fail = 0; err = 0; broken = 0
    for r in ts.results
        if r isa Test.DefaultTestSet
            sub = count_results(r)
            pass += sub[1]; fail += sub[2]; err += sub[3]; broken += sub[4]
        elseif r isa Test.Pass
            pass += 1
        elseif r isa Test.Fail
            fail += 1
        elseif r isa Test.Error
            err += 1
        elseif r isa Test.Broken
            broken += 1
        end
    end
    return (pass, fail, err, broken)
end

# Check if the Vulkan device is still alive
function device_alive()
    try
        x = LavaArray(Float32[1.0])
        r = Array(x)
        return r[1] == 1.0f0
    catch
        return false
    end
end

# Skip tests that need features we don't support
const SKIP = Set([
    "sparse",       # needs sparse array types
    "ext/jld2",     # needs JLD2 extension
    "alloc cache",  # needs alloc_cache support
    "random",       # needs RNG on GPU (not implemented)
])

function run_gpuarrays_tests()
    test_names = sort(collect(keys(TestSuite.tests)))
    filter!(n -> n ∉ SKIP, test_names)

    all_results = Vector{Tuple{String,Int,Int,Int}}()
    device_dead = false

    for name in test_names
        if device_dead
            push!(all_results, (name, 0, 0, 1))
            println("$name ... SKIPPED (device lost)")
            continue
        end

        print("$name ... ")
        flush(stdout)
        try
            ts = @testset "$name" begin
                TestSuite.tests[name](LavaArray)
            end
            p, f, e, _ = count_results(ts)
            push!(all_results, (name, p, f, e))
            if f + e > 0
                println("$(p)p $(f)f $(e)e")
            else
                println("$(p) pass")
            end
            if e > 0 && !device_alive()
                device_dead = true
                println("  *** Device lost during $name — skipping remaining tests ***")
            end
        catch ex
            push!(all_results, (name, 0, 0, 1))
            msg = sprint(showerror, ex)
            println("CRASH: ", msg[1:min(end,150)])
            if occursin("DEVICE_LOST", msg) || !device_alive()
                device_dead = true
                println("  *** Device lost — skipping remaining tests ***")
            end
        end
    end

    println("\n" * "="^70)
    println("  GPUArrays TestSuite Results (LavaArray)")
    println("="^70)
    total_p = 0; total_f = 0; total_e = 0
    for (name, p, f, e) in all_results
        total_p += p; total_f += f; total_e += e
        status = (f + e == 0) ? "PASS" : "FAIL"
        detail = (f + e == 0) ? "$(p) pass" : "$(p)p $(f)f $(e)e"
        println("  $status  $(rpad(name, 40)) $detail")
    end
    println("="^70)
    println("  Total: $total_p passed, $total_f failed, $total_e errors")
    n_groups_pass = count(x -> x[3] + x[4] == 0, all_results)
    println("  $n_groups_pass/$(length(all_results)) test groups fully passing")
    println("  Skipped: $(join(sort(collect(SKIP)), ", "))")
    return all_results
end

run_gpuarrays_tests()
