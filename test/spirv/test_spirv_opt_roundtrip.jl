# Tier 1: spirv-opt roundtrip validation
#
# Compiles kernels via our emitter, runs through spirv-opt -O --validate-after-all,
# and re-validates output with spirv-val. Proves our SPIR-V is semantically clean
# enough for a production optimizer to parse and transform.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "spirv-opt Roundtrip" begin

    # Helper: compile and roundtrip, return (disasm, optimized_bytes)
    function compile_and_optimize(f, tt; stage=:compute, kwargs...)
        d, bytes = compile_and_disasm(f, tt; stage, kwargs...)
        opt = spirv_opt_roundtrip(bytes)
        return (d, opt)
    end

    @testset "vadd (f32)" begin
        function vadd(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] + B[i]
            return nothing
        end
        d, opt = compile_and_optimize(vadd, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                   Lava.LavaDeviceArray{Float32,1},
                                                   Lava.LavaDeviceArray{Float32,1}})
        @test !isempty(opt)
    end

    @testset "integer arithmetic" begin
        function int_arith(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] * B[i] + A[i]
            return nothing
        end
        _, opt = compile_and_optimize(int_arith, Tuple{Lava.LavaDeviceArray{Int32,1},
                                                        Lava.LavaDeviceArray{Int32,1},
                                                        Lava.LavaDeviceArray{Int32,1}})
        @test !isempty(opt)
    end

    @testset "branching" begin
        function branching(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                if A[i] > 0.0f0
                    B[i] = 1.0f0
                else
                    B[i] = 0.0f0
                end
            end
            return nothing
        end
        _, opt = compile_and_optimize(branching, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                        Lava.LavaDeviceArray{Float32,1}})
        @test !isempty(opt)
    end

    @testset "loop with break" begin
        function looping(A, n)
            i = Lava.lava_global_invocation_id_x()
            s = 0.0f0
            for j in Int32(1):n
                s += Float32(j)
                if s > 100.0f0
                    break
                end
            end
            @inbounds A[i] = s
            return nothing
        end
        _, opt = compile_and_optimize(looping, Tuple{Lava.LavaDeviceArray{Float32,1}, Int32})
        @test !isempty(opt)
    end

    @testset "atomics" begin
        function atomic_kernel(counter)
            Lava.Atomix.@atomic counter[1] += Int32(1)
            return nothing
        end
        _, opt = compile_and_optimize(atomic_kernel, Tuple{Lava.LavaDeviceArray{Int32,1}})
        @test !isempty(opt)
    end

    @testset "conversions" begin
        function conv(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = Float32(A[i]) * 2.0f0
            return nothing
        end
        _, opt = compile_and_optimize(conv, Tuple{Lava.LavaDeviceArray{Int32,1},
                                                    Lava.LavaDeviceArray{Float32,1}})
        @test !isempty(opt)
    end

    @testset "vertex shader" begin
        function vert()
            idx = Lava.vertex_index()
            Lava.set_position!(GeometryBasics.Vec4f(Float32(idx), 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        _, opt = compile_and_optimize(vert, Tuple{}; stage=:vertex)
        @test !isempty(opt)
    end

    @testset "fragment shader" begin
        function frag()
            xy = Lava.frag_coord_xy()
            Lava.gfx_output(0, GeometryBasics.Vec4f(xy[1], xy[2], 0.0f0, 1.0f0))
            return nothing
        end
        _, opt = compile_and_optimize(frag, Tuple{}; stage=:fragment)
        @test !isempty(opt)
    end

end
