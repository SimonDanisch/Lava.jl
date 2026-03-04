# Tier 1: Compute shader memory + atomics + barriers
# Tests shared memory, atomics, barriers, CAS loop

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "Compute Memory" begin

    @testset "shared memory" begin
        function shared_mem_kernel(A)
            ptr = Lava.lava_alloc_shared(Val(:test_shared), Float32, Val(64))
            shared = Lava.LavaSharedArray{Float32}(ptr, 64)
            lid = Lava.lava_local_invocation_id_x()
            gid = Lava.lava_global_invocation_id_x()
            @inbounds shared[lid] = A[gid]
            Lava.lava_workgroup_barrier()
            @inbounds A[gid] = shared[lid]
            return nothing
        end
        d, _ = compile_and_disasm(shared_mem_kernel,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "Workgroup")  # Workgroup storage class
        check(d, "OpControlBarrier")
    end

    @testset "atomic int add" begin
        function atomic_add(counter, A)
            i = Lava.lava_global_invocation_id_x()
            @inbounds Lava.Atomix.@atomic counter[1] += A[i]
            return nothing
        end
        d, _ = compile_and_disasm(atomic_add,
                                   Tuple{Lava.LavaDeviceArray{Int32,1},
                                         Lava.LavaDeviceArray{Int32,1}})
        check(d, "OpAtomicIAdd")
        # Must use Device scope, not CrossDevice (invalid in Vulkan SPIR-V)
        check_not(d, "CrossDevice")
    end

    @testset "atomic uint add" begin
        function atomic_uadd(counter, val)
            Lava.Atomix.@atomic counter[1] += val
            return nothing
        end
        d, _ = compile_and_disasm(atomic_uadd,
                                   Tuple{Lava.LavaDeviceArray{UInt32,1}, UInt32})
        check(d, "OpAtomicIAdd")
    end

    @testset "f32 atomic add (CAS loop)" begin
        function f32_atomic(counter)
            Lava.Atomix.@atomic counter[1] += 1.0f0
            return nothing
        end
        d, _ = compile_and_disasm(f32_atomic,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}})
        # Float32 atomics use a CAS loop (OpAtomicCompareExchange)
        check(d, "OpAtomicCompareExchange")
    end

    @testset "barrier" begin
        function barrier_kernel(A)
            Lava.lava_workgroup_barrier()
            return nothing
        end
        d, _ = compile_and_disasm(barrier_kernel,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpControlBarrier")
    end
end
