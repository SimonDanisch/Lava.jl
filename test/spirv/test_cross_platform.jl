# Tier 1: Cross-platform SPIR-V correctness checks
# Targeted at known vendor crash patterns (NVIDIA, AMD)

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc
using GeometryBasics

@testset "Cross-Platform Checks" begin

    # Helper: compile a representative kernel for reuse across checks
    function compile_vadd()
        function vadd(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] + B[i]
            return nothing
        end
        compile_and_disasm(vadd, Tuple{Lava.LavaDeviceArray{Float32,1},
                                       Lava.LavaDeviceArray{Float32,1},
                                       Lava.LavaDeviceArray{Float32,1}})
    end

    @testset "BDA alignment — PSB loads/stores have Aligned" begin
        d, _ = compile_vadd()
        # Every PhysicalStorageBuffer load/store must have Aligned operand
        # NVIDIA crashes without it
        for line in split(d, '\n')
            if occursin("PhysicalStorageBuffer", line) &&
               (occursin("OpLoad", line) || occursin("OpStore", line))
                @test occursin("Aligned", line)
            end
        end
    end

    @testset "no OpUnreachable" begin
        d, _ = compile_vadd()
        check_not(d, "OpUnreachable")
    end

    @testset "no CrossDevice scope in atomics" begin
        function atomic_kernel(counter)
            Lava.Atomix.@atomic counter[1] += Int32(1)
            return nothing
        end
        d, _ = compile_and_disasm(atomic_kernel,
                                   Tuple{Lava.LavaDeviceArray{Int32,1}})
        check_not(d, "CrossDevice")
    end

    @testset "structured CF — selection merges paired" begin
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
        d, _ = compile_and_disasm(branching, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                    Lava.LavaDeviceArray{Float32,1}})
        # Every OpSelectionMerge should be followed (within a few lines) by
        # OpBranchConditional or OpSwitch
        lines = split(d, '\n')
        for (i, line) in enumerate(lines)
            if occursin("OpSelectionMerge", line)
                # Search next few lines for branch
                found = false
                for j in (i+1):min(i+3, length(lines))
                    if occursin("OpBranchConditional", lines[j]) || occursin("OpSwitch", lines[j])
                        found = true
                        break
                    end
                end
                @test found
            end
        end
    end

    @testset "structured CF — loop merges paired" begin
        function looping(A, n)
            i = Lava.lava_global_invocation_id_x()
            s = 0.0f0
            for j in Int32(1):n
                s += Float32(j)
            end
            @inbounds A[i] = s
            return nothing
        end
        d, _ = compile_and_disasm(looping, Tuple{Lava.LavaDeviceArray{Float32,1}, Int32})
        lines = split(d, '\n')
        for (i, line) in enumerate(lines)
            if occursin("OpLoopMerge", line)
                # Should be followed by OpBranch or OpBranchConditional
                found = false
                for j in (i+1):min(i+3, length(lines))
                    if occursin("OpBranch", lines[j])
                        found = true
                        break
                    end
                end
                @test found
            end
        end
    end

    @testset "complex CF stress test — nested loop + if + break" begin
        function complex_cf(A, B, n)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = 0.0f0
                for j in Int32(1):n
                    x = A[i] * Float32(j)
                    if x > 10.0f0
                        s += x
                        if x > 100.0f0
                            break
                        end
                    else
                        s -= 1.0f0
                    end
                end
                B[i] = s
            end
            return nothing
        end
        # This should compile and validate without errors
        d, _ = compile_and_disasm(complex_cf,
                                   Tuple{Lava.LavaDeviceArray{Float32,1},
                                         Lava.LavaDeviceArray{Float32,1}, Int32})
        # Verify it has both loop and selection merges
        check(d, "OpLoopMerge")
        check(d, "OpSelectionMerge")
        # And no OpUnreachable
        check_not(d, "OpUnreachable")
    end

    @testset "vertex shader cross-platform" begin
        function vert()
            idx = Lava.vertex_index()
            Lava.set_position!(GeometryBasics.Vec4f(Float32(idx), 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(vert, Tuple{}; stage=:vertex)
        check_not(d, "OpUnreachable")
        # No compute-specific decorations
        check_not(d, "LocalSize")
    end
end


