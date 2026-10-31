# The kernel cache: compiled SPIR-V kept with each kernel's `CodeInstance`
# (`compiler/cache.jl`), the way CUDA.jl keeps its images.
#
# Each property is checked with `compile_stats`, which counts compiles, rather
# than with the answer alone, which is the same whether anything was cached:
#
#   * a second lookup of a kernel compiles nothing;
#   * neither does one after an unrelated method definition, which moves the
#     world age but leaves the kernel's code valid;
#   * an edit to a function the kernel inlines compiles it again, from the new
#     code;
#   * code compiled before, in this process or an earlier one, is found in the
#     object cache when its `CodeInstance` is new: an edit undone compiles nothing;
#   * the kernels Lava's own precompile workload compiled are found in this
#     process without compiling, because they went into the package image.
#
# Device-free, like the rest of this suite.

using Test, Lava

# Drawn per run: kernels outlive the process, so code an earlier run compiled is
# found, not compiled, and only code no run has seen is a compile.
const CC_FACTOR = Float32(rand(10_000:99_999))
cc_scale(x) = x * CC_FACTOR

function cc_kernel!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        @inbounds out[i] = cc_scale(Float32(i))
    end
    return nothing
end

const CC_TT = Tuple{Lava.LavaDeviceArray{Float32, 1}}

cc_lookup() = Lava.compile_or_lookup(Lava.lava_kernel_job(cc_kernel!, CC_TT; workgroup_size = (64, 1, 1)))

@testset "compile cache" begin
    @testset "a lookup compiles once" begin
        Lava.reset_compile_stats!()
        first = cc_lookup()
        @test Lava.compile_stats() == (; hits = 0, misses = 1)
        @test cc_lookup() === first
        @test Lava.compile_stats() == (; hits = 1, misses = 1)
    end

    @testset "an unrelated definition compiles nothing" begin
        before = cc_lookup()
        Lava.reset_compile_stats!()
        @eval cc_unrelated() = 1
        @test Base.invokelatest(cc_lookup) === before
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
    end

    @testset "an edit to an inlined callee compiles again" begin
        before = cc_lookup()
        Lava.reset_compile_stats!()
        @eval cc_scale(x) = x * $(CC_FACTOR + 1f0)
        after = Base.invokelatest(cc_lookup)
        @test Lava.compile_stats() == (; hits = 0, misses = 1)
        @test after !== before
        @test after.spirv_bytes != before.spirv_bytes
    end

    # The edit undone makes a new `CodeInstance` of code compiled before, which
    # Julia numbers differently: only the object cache, keyed by the module without
    # those numbers (`canonical_ir`), knows it.
    @testset "code compiled before is not compiled again" begin
        @eval cc_scale(x) = x * $(CC_FACTOR + 2f0)
        first = Base.invokelatest(cc_lookup)
        @eval cc_unrelated_2() = 2
        @eval cc_scale(x) = x * $(CC_FACTOR + 3f0)
        Base.invokelatest(cc_lookup)
        Lava.reset_compile_stats!()
        @eval cc_scale(x) = x * $(CC_FACTOR + 2f0)
        again = Base.invokelatest(cc_lookup)
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
        @test again !== first
        @test again.spirv_bytes == first.spirv_bytes
        @test again.push_info.arg_layout == first.push_info.arg_layout
    end

    @testset "the package image's kernels compile nothing" begin
        Lava.reset_compile_stats!()
        Lava.compile_or_lookup(Lava.lava_kernel_job(
            Lava.precompile_warmup_kernel!,
            Tuple{Lava.LavaDeviceArray{Float32, 1}, Lava.LavaDeviceArray{Float32, 1}};
            workgroup_size = (64, 1, 1)))
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
    end
end
