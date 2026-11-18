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
            ptr = Lava.lava_alloc_shared(Float32, Val(64))
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
            @inbounds Lava.Atomix.@atomic :monotonic counter[1] += A[i]
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
            Lava.Atomix.@atomic :monotonic counter[1] += val
            return nothing
        end
        d, _ = compile_and_disasm(atomic_uadd,
                                   Tuple{Lava.LavaDeviceArray{UInt32,1}, UInt32})
        check(d, "OpAtomicIAdd")
    end

    @testset "f32 atomic add (hardware OpAtomicFAddEXT)" begin
        function f32_atomic(counter)
            Lava.Atomix.@atomic :monotonic counter[1] += 1.0f0
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
                prev = (Lava.Atomix.@atomicreplace :monotonic keys[slot] typemax(UInt64) => key).old
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
            ptr = Lava.lava_alloc_shared(Tuple{NTuple{1,NTuple{3,Float32}}, Bool}, Val(64))
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

    @testset "a struct array in workgroup memory at computed and constant indices" begin
        # AcceleratedKernels' block-scan up-sweep over `Optional{Bounds3}`, laid out as
        # `{ [2 x [1 x [3 x float]]], i32 }` (28 bytes): `totals[ai + 1]` with
        # `ai = offset * (2i + 1) - 1`, then a field of `totals[64]`. LLVM 19 and later
        # write the fields of element `ai` as -28, -24, … bytes off element `ai + 1`,
        # and that field of element 63 as 1768 bytes off the array; taken as byte
        # offsets they became `OpPtrAccessChain`s by -24 ÷ 28 and byte loads from
        # element 0, which spirv-val rejects (`compile_and_disasm` validates).
        combine(a, b) = (((min.(a[1][1][1], b[1][1][1]),), (max.(a[1][2][1], b[1][2][1]),)), a[2] | b[2])
        function upsweep_kernel(out)
            totals = Lava.KI.localmemory(Tuple{NTuple{2,Tuple{NTuple{3,Float32}}}, Int32}, Val((64,)))
            ithread = UInt32(Lava.lava_local_invocation_id_x())
            gid = Lava.lava_global_invocation_id_x() + 1
            @inbounds totals[ithread + 0x1] = (((Float32(gid), 0f0, 0f0),), ((1f0, 2f0, 3f0),)), Int32(1)
            offset = one(ithread)
            half = UInt32(64) >> 0x1
            while half > 0x0
                Lava.lava_workgroup_barrier()
                if ithread < half
                    ai = offset * (0x2 * ithread + 0x1) - 0x1
                    bi = offset * (0x2 * ithread + 0x2) - 0x1
                    @inbounds totals[bi + 0x1] = combine(totals[ai + 0x1], totals[bi + 0x1])
                end
                offset = offset << 0x1
                half = half >> 0x1
            end
            Lava.lava_workgroup_barrier()
            @inbounds out[gid] = totals[64][1][1][1][2]
            return nothing
        end
        d, _ = compile_and_disasm(upsweep_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "Workgroup")
    end

    @testset "a device array stored in device memory" begin
        # A table of device-array handles, the way Raycore's `store_texture`
        # keeps its textures and Mantle's `test_stored_device_array.jl` reads one:
        # the argument's address points at a handle, and the handle's address
        # points at the data. After SROA that is a chain of `load ptr` with no
        # struct in it, so the middle pointer's pointee is an opaque `ptr`, and
        # it only maps through the value loaded from it. It failed with "Cannot
        # map opaque pointer type". `compile_and_disasm` validates, so each
        # compile below is also a spirv-val pass.
        DA = Lava.LavaDeviceArray
        function stored_read(out, table)
            i = Lava.lava_global_invocation_id_x() + 1
            @inbounds out[i] = table[1][i]
            return nothing
        end
        d, _ = compile_and_disasm(stored_read, Tuple{DA{Float32,1}, DA{DA{Float32,1},1}})
        check(d, "OpLoad %float")
        # Julia 1.13 may load a stored array's address as an integer; a table of raw
        # pointers is a `load ptr` chain on every version. Three levels of address,
        # every one a typed PhysicalStorageBuffer pointer.
        function stored_ptr_read(out, table)
            i = Lava.lava_global_invocation_id_x() + 1
            @inbounds out[i] = unsafe_load(table[1], i)
            return nothing
        end
        d, _ = compile_and_disasm(stored_ptr_read, Tuple{DA{Float32,1}, DA{Ptr{Float32},1}})
        check(d, "%_ptr_PhysicalStorageBuffer__ptr_PhysicalStorageBuffer__ptr_PhysicalStorageBuffer_float")

        # A handle at a computed slot, and a handle WRITTEN by the kernel.
        function stored_dyn(out, table, j::Int32)
            i = Lava.lava_global_invocation_id_x() + 1
            @inbounds out[i] = table[j][i] + table[j + Int32(1)][i]
            return nothing
        end
        d, _ = compile_and_disasm(stored_dyn, Tuple{DA{Float32,1}, DA{DA{Float32,1},1}, Int32})
        check(d, "%_ptr_PhysicalStorageBuffer__ptr_PhysicalStorageBuffer_float")
        function stored_write(table, src)
            @inbounds table[2] = src
            return nothing
        end
        d, _ = compile_and_disasm(stored_write, Tuple{DA{DA{Float32,1},1}, DA{Float32,1}})
        check(d, "OpStore")

        # A table of tables, indexed past the first slot at both levels so both
        # are struct GEPs. Every `LavaDeviceArray{T,1}` is the same LLVM struct
        # `{ ptr, [1 x i64] }`, so the struct-member map recorded that struct as
        # pointing to ITSELF and mapping it recursed until the stack overflowed.
        function stored_deep(out, t)
            i = Lava.lava_global_invocation_id_x() + 1
            @inbounds out[i] = t[2][3][i]
            return nothing
        end
        d, _ = compile_and_disasm(stored_deep, Tuple{DA{Float32,1}, DA{DA{DA{Float32,1},1},1}})
        check(d, "%_ptr_PhysicalStorageBuffer_float")
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


