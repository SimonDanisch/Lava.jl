# Tier 1: Compute shader control flow emission tests
# Tests if/else, loops, nested CF, break, structured merge blocks

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "Compute Control Flow" begin

    @testset "if/else branch" begin
        # Simple if/else may be optimized to OpSelect — both are valid
        function if_else(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds if A[i] > 0.0f0
                B[i] = 1.0f0
            else
                B[i] = -1.0f0
            end
            return nothing
        end
        d, _ = compile_and_disasm(if_else, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                  Lava.LavaDeviceArray{Float32,1}})
        @test occursin("OpSelectionMerge", d) || occursin("OpSelect", d)
    end

    @testset "for loop" begin
        function for_loop(A, n)
            i = Lava.lava_global_invocation_id_x()
            s = 0.0f0
            for j in Int32(1):n
                s += Float32(j)
            end
            @inbounds A[i] = s
            return nothing
        end
        d, _ = compile_and_disasm(for_loop, Tuple{Lava.LavaDeviceArray{Float32,1}, Int32})
        check(d, "OpLoopMerge")
        check_regex(d, "OpPhi")
    end

    @testset "nested if (two selection merges)" begin
        function nested_if(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                x = A[i]
                if x > 0.0f0
                    if x > 1.0f0
                        B[i] = 2.0f0
                    else
                        B[i] = 1.0f0
                    end
                else
                    B[i] = 0.0f0
                end
            end
            return nothing
        end
        d, _ = compile_and_disasm(nested_if, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                    Lava.LavaDeviceArray{Float32,1}})
        # Either branches or OpSelect (LLVM may optimize simple branches)
        n_merges = count("OpSelectionMerge", d)
        n_selects = count("OpSelect", d)
        @test n_merges >= 2 || n_selects >= 2 || (n_merges + n_selects >= 2)
    end

    @testset "while loop with break" begin
        function while_break(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                val = A[i]
                count = Int32(0)
                while count < Int32(100)
                    val *= 0.5f0
                    count += Int32(1)
                    val < 0.001f0 && break
                end
                B[i] = val
            end
            return nothing
        end
        d, _ = compile_and_disasm(while_break, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                      Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpLoopMerge")
        # The break should produce a branch to the merge block
        check(d, "OpBranchConditional")
    end

    @testset "loop + if combo" begin
        function loop_if(A, B, n)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = 0.0f0
                for j in Int32(1):n
                    if A[i] > Float32(j)
                        s += 1.0f0
                    else
                        s -= 1.0f0
                    end
                end
                B[i] = s
            end
            return nothing
        end
        d, _ = compile_and_disasm(loop_if, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                  Lava.LavaDeviceArray{Float32,1}, Int32})
        # Both loop and selection merges present
        check(d, "OpLoopMerge")
        check(d, "OpSelectionMerge")
    end

    @testset "ternary expression" begin
        function ternary(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = A[i] > 0.0f0 ? 1.0f0 : 0.0f0
            return nothing
        end
        d, _ = compile_and_disasm(ternary, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                  Lava.LavaDeviceArray{Float32,1}})
        # Ternary may compile to OpSelect or branch+merge
        @test occursin("OpSelect", d) || occursin("OpSelectionMerge", d)
    end
end
