# Tier 1: Compute shader memory + atomics + barriers
# Tests shared memory, atomics, barriers, CAS loop

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

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

    @testset "f32 atomic add (hardware OpAtomicFAddEXT)" begin
        function f32_atomic(counter)
            Lava.Atomix.@atomic counter[1] += 1.0f0
            return nothing
        end
        d, _ = compile_and_disasm(f32_atomic,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}})
        # Float32 atomics now use hardware atomicrmw fadd, emitted as
        # OpAtomicFAddEXT + AtomicFloat32AddEXT capability +
        # SPV_EXT_shader_atomic_float_add extension.
        check(d, "OpAtomicFAddEXT")
        check(d, "AtomicFloat32AddEXT")
        check(d, "SPV_EXT_shader_atomic_float_add")
        # No CAS loop any more.
        check_not(d, "OpAtomicCompareExchange")
    end

    @testset "CAS loop carries the { old, success } aggregate" begin
        # A hash map's linear probing: LLVM rotates the loop and carries the
        # cmpxchg result `{ T, i1 }` through a phi. That aggregate has to be a
        # real SPIR-V struct, or the phi is emitted over the bare old value and
        # spirv-val rejects it (`compile_and_disasm` validates).
        function cas_insert(keys, vals, key, val, cap)
            slot = Int32(1)
            while true
                prev = (Lava.Atomix.@atomicreplace keys[slot] typemax(UInt64) => key).old
                if prev == typemax(UInt64) || prev == key
                    @inbounds vals[slot] = val
                    return nothing
                end
                slot = slot == cap ? Int32(1) : slot + Int32(1)
            end
        end
        d, _ = compile_and_disasm(cas_insert,
                                   Tuple{Lava.LavaDeviceArray{UInt64,1},
                                         Lava.LavaDeviceArray{Int32,1}, UInt64, Int32, Int32})
        check(d, "OpAtomicCompareExchange")
        check(d, "Int64Atomics")
        # The aggregate: `OpCompositeConstruct %struct %old %success`.
        check_regex(d, "OpCompositeConstruct %_struct_\\d+ %\\d+ %\\d+")
    end

    @testset "a struct with nested arrays in workgroup memory" begin
        # A reduction's partial result: a `Point3f` (`[1 x [3 x float]]`) and a
        # flag, stored into and loaded from shared memory field by field, each
        # field through a byte-offset GEP. The second float sits 4 bytes into
        # element 0 of the outer array; resolving that offset stopped at the
        # inner array and fell back to dividing by the struct's size, an
        # `OpPtrAccessChain` spirv-val rejects (`compile_and_disasm` validates).
        function lane_kernel(out, src)
            ptr = Lava.lava_alloc_shared(Val(:test_lanes), Tuple{NTuple{1,NTuple{3,Float32}}, Bool}, Val(64))
            shared = Lava.LavaSharedArray{Tuple{NTuple{1,NTuple{3,Float32}}, Bool}}(ptr, 64)
            lid = Lava.lava_local_invocation_id_x() + 1
            @inbounds shared[lid] = (((src[lid], 2f0 * src[lid], 3f0 * src[lid]),), true)
            Lava.lava_workgroup_barrier()
            v, ok = @inbounds shared[65 - lid]
            @inbounds out[lid] = ok ? v[1][2] : 0f0
            return nothing
        end
        d, _ = compile_and_disasm(lane_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.LavaDeviceArray{Float32,1}})
        check(d, "Workgroup")
        check_not(d, "OpPtrAccessChain %_ptr_Workgroup")
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


