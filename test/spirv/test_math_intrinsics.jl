# Tier 1: Math intrinsic emission tests
# Tests GLSL.std.450 mapping, no f64 leakage for f32 math

using Test
using Lava
using KernelAbstractions
using KernelAbstractions: @kernel, @index
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

@testset "Math Intrinsics" begin

    # GLSL.std.450 extended instruction mappings for f32
    f32_math = [
        (sin,   "Sin"),
        (cos,   "Cos"),
        (sqrt,  "Sqrt"),
        (abs,   "FAbs"),
        (floor, "Floor"),
        (ceil,  "Ceil"),
        (exp,   "Exp"),
        (log,   "Log"),
        (exp2,  "Exp2"),
        (log2,  "Log2"),
    ]

    @testset "f32 $jl_name → GLSL.std.450 $spirv_name" for (jl_name, spirv_name) in f32_math
        fn = let f = jl_name
            function(A, B)
                i = Lava.lava_global_invocation_id_x()
                @inbounds B[i] = f(A[i])
                return nothing
            end
        end
        d, _ = compile_and_disasm(fn, Tuple{Lava.LavaDeviceArray{Float32,1},
                                             Lava.LavaDeviceArray{Float32,1}})
        # GLSL.std.450 import and OpExtInst are on separate lines
        check(d, "GLSL.std.450")
        check_regex(d, "OpExtInst.*$spirv_name")
        check_not(d, "OpTypeFloat 64")
    end

    @testset "min/max" begin
        function minmax(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = min(A[i], B[i])
            return nothing
        end
        d, _ = compile_and_disasm(minmax, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                 Lava.LavaDeviceArray{Float32,1},
                                                 Lava.LavaDeviceArray{Float32,1}})
        # min should map to FMin or NMin
        @test occursin("FMin", d) || occursin("NMin", d)
    end

    @testset "clamp" begin
        function clamp_kernel(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = clamp(A[i], 0.0f0, 1.0f0)
            return nothing
        end
        d, _ = compile_and_disasm(clamp_kernel, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                       Lava.LavaDeviceArray{Float32,1}})
        @test occursin("FClamp", d) || occursin("NClamp", d) ||
              (occursin("FMax", d) && occursin("FMin", d)) ||
              (occursin("NMax", d) && occursin("NMin", d))
    end

    @testset "sign" begin
        function sign_kernel(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = sign(A[i])
            return nothing
        end
        d, _ = compile_and_disasm(sign_kernel, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                      Lava.LavaDeviceArray{Float32,1}})
        @test occursin("FSign", d) || occursin("OpFOrdGreaterThan", d) || occursin("OpSelect", d)
    end

    @testset "copysign zero-preserving (bitwise sign-copy)" begin
        # copysign(x, 0) must return +|x|, not 0. GLSL.std.450 FSign(0) is 0,
        # so `FAbs(x) * FSign(y)` produces 0 for any y==0 — breaks ray-tracing
        # `safe_invdir` clamps and similar IEEE-dependent code.
        #
        # The emitter errors on any `llvm.copysign` that slips past the
        # `Base.copysign` overlay in `device/math.jl`. Each kernel below
        # compiles (so no leak to the emitter) AND returns the IEEE result.

        @kernel function copysign_kernel!(out, x_arr, y_arr)
            i = @index(Global, Linear)
            @inbounds out[i] = copysign(x_arr[i], y_arr[i])
        end

        x = Float32[1f-5,  1f-5,  1f-5, -3f0, 2f0, 2f0]
        y = Float32[ 0f0, -0f0, -1f0,  2f0, Inf32, -Inf32]
        expected = Float32[1f-5, -1f-5, -1f-5, 3f0, 2f0, -2f0]

        x_arr = Lava.LavaArray(x)
        y_arr = Lava.LavaArray(y)
        out = Lava.LavaArray(zeros(Float32, length(x)))
        copysign_kernel!(Lava.LavaBackend())(out, x_arr, y_arr; ndrange=length(x))
        Lava.vk_flush!(Lava.vk_context())
        @test reinterpret(UInt32, Array(out)) == reinterpret(UInt32, expected)

        # Float64
        xd = Float64[1e-10,  1e-10,  1e-10, -3.0]
        yd = Float64[  0.0,   -0.0,   -1.0,  2.0]
        expected_d = Float64[1e-10, -1e-10, -1e-10, 3.0]
        x_arr64 = Lava.LavaArray(xd); y_arr64 = Lava.LavaArray(yd)
        out64 = Lava.LavaArray(zeros(Float64, length(xd)))
        copysign_kernel!(Lava.LavaBackend())(out64, x_arr64, y_arr64; ndrange=length(xd))
        Lava.vk_flush!(Lava.vk_context())
        @test reinterpret(UInt64, Array(out64)) == reinterpret(UInt64, expected_d)
    end

    @testset "copysign leak paths (must not reach emitter)" begin
        # Exercises the ways `llvm.copysign` could slip past the Base.copysign
        # overlay. Each kernel here will FAIL TO COMPILE if it does, because
        # the emitter errors on `llvm.copysign`. Numeric checks additionally
        # confirm correctness.

        # 1. @fastmath: fastmath rewrites `copysign` but still hits Base.copysign.
        @kernel function fastmath_cs!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath copysign(x[i], y[i])
        end
        x = Lava.LavaArray(Float32[1f-5, 2f0])
        y = Lava.LavaArray(Float32[0f0, -1f0])
        out = Lava.LavaArray(zeros(Float32, 2))
        fastmath_cs!(Lava.LavaBackend())(out, x, y; ndrange=2)
        Lava.vk_flush!(Lava.vk_context())
        @test Array(out) == Float32[1f-5, -2f0]

        # 2. Vectorizable inner loop — gives LLVM a chance to recognize the
        #    bitwise sign-copy pattern and canonicalize back to `llvm.copysign`.
        @kernel function loop_cs!(out, xs, ys, n::Int32)
            i = @index(Global, Linear)
            acc = 0f0
            @inbounds for j in Int32(1):n
                acc += copysign(xs[i, j], ys[i, j])
            end
            @inbounds out[i] = acc
        end
        xs = Lava.LavaArray(fill(1f-5, 16, 8))
        ys = Lava.LavaArray(zeros(Float32, 16, 8))  # all zero y's — the bug case
        lout = Lava.LavaArray(zeros(Float32, 16))
        loop_cs!(Lava.LavaBackend())(lout, xs, ys, Int32(8); ndrange=16)
        Lava.vk_flush!(Lava.vk_context())
        @test all(Array(lout) .≈ 8f0 * 1f-5)  # would be 0 if old bug reappeared

        # 3. Exact `safe_invdir` pattern — the original bug trigger.
        @kernel function safe_invdir_kernel!(out, d)
            i = @index(Global, Linear)
            ooeps = 1f-5
            @inbounds dv = d[i]
            clamped = abs(dv) > ooeps ? dv : copysign(ooeps, dv)
            @inbounds out[i] = 1f0 / clamped
        end
        d = Lava.LavaArray(Float32[0f0, -0f0, 1f0, -1f0])
        sout = Lava.LavaArray(zeros(Float32, 4))
        safe_invdir_kernel!(Lava.LavaBackend())(sout, d; ndrange=4)
        Lava.vk_flush!(Lava.vk_context())
        result = Array(sout)
        # +0 → clamp to +1e-5 → 1/1e-5 = 1e5   (NOT Inf)
        # -0 → clamp to -1e-5 → 1/-1e-5 = -1e5 (NOT Inf)
        @test result[1] == 1f5
        @test result[2] == -1f5
        @test result[3] == 1f0
        @test result[4] == -1f0
    end

    @testset "round halfway (round-to-nearest-even)" begin
        # Julia's `round(::Float32)` lowers to `llvm.rint` (round-half-to-even).
        # GLSL.std.450 opcode 1 (`Round`) has implementation-defined halfway
        # behavior per SPIR-V spec; opcode 2 (`RoundEven`) is the IEEE-correct
        # round-to-nearest-even. Mapping `llvm.rint → RoundEven` keeps GPU
        # parity with Julia CPU across drivers.
        @kernel function round_kernel!(out, xs)
            i = @index(Global, Linear)
            @inbounds out[i] = round(xs[i])
        end

        xs = Lava.LavaArray(Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5])
        out = Lava.LavaArray(zeros(Float32, 8))
        round_kernel!(Lava.LavaBackend())(out, xs; ndrange=8)
        Lava.vk_flush!(Lava.vk_context())
        gpu = Array(out)
        cpu = Float32[round(x) for x in Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5]]
        # Expected (round-to-even): 0, 2, 2, 4, -0, -2, -2, 4
        @test gpu == cpu
    end
end


