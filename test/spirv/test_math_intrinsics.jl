# Tier 1: Math intrinsic emission tests
# Tests GLSL.std.450 mapping, no f64 leakage for f32 math

using Test
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
end


