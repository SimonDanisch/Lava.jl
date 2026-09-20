# Tier 1: signed 8-bit cooperative matrices.
#
# A cooperative matrix loads out of an ordinary array, and SPIR-V requires an
# access chain's result type to match the base's element type. This emitter
# makes every integer type SIGNLESS — `OpTypeInt <width> 0`, in `types.jl` —
# except that `coopmat_component_type!` used to make `i8` and `i32` signed. The
# two then disagreed and the module failed validation before it reached a
# driver:
#
#     OpAccessChain result type (OpTypeInt) does not match the type that
#     results from indexing into the base <id> (OpTypeInt)
#     %211 = OpAccessChain %_ptr_Workgroup_char %37 %uint_0
#
# Signedness for an integer product is a property of the OPERATION, and
# `OpCooperativeMatrixMulAddKHR` has an operands word for exactly that. Absent,
# the components are read as unsigned, which for int8 weights is not a rounding
# difference but a different number — so both halves are checked here.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_regex, compile_and_disasm

@testset "Cooperative matrix, 8-bit" begin
    # int8 x int8 -> int32 out of shared memory, which is where the mismatch was.
    function i8_coopmat(C)
        pa = Lava.lava_alloc_shared(Val(:i8a), Int8, Val(256))
        pb = Lava.lava_alloc_shared(Val(:i8b), Int8, Val(256))
        pc = Lava.lava_alloc_shared(Val(:i32c), Int32, Val(256))
        sa = Lava.LavaSharedArray{Int8}(pa, 256)
        sb = Lava.LavaSharedArray{Int8}(pb, 256)
        sc = Lava.LavaSharedArray{Int32}(pc, 256)
        lid = Lava.lava_local_invocation_id_x()
        @inbounds sa[lid] = Int8(lid % 7)
        @inbounds sb[lid] = Int8(lid % 5)
        Lava.lava_workgroup_barrier()
        a = Lava.CoopMatrix{Int8,16,16,Lava.MatrixA,Lava.SubgroupScope}(sa, 1, 16)
        b = Lava.CoopMatrix{Int8,16,16,Lava.MatrixB,Lava.SubgroupScope}(sb, 1, 16)
        c = zero(Lava.CoopMatrix{Int32,16,16,Lava.Accumulator,Lava.SubgroupScope})
        c = muladd(a, b, c)
        copyto!(sc, 1, 16, c)
        Lava.lava_workgroup_barrier()
        @inbounds C[lid] = sc[lid]
        return nothing
    end
    d, _ = compile_and_disasm(i8_coopmat, Tuple{Lava.LavaDeviceArray{Int32,1}})

    @testset "the module is one the validator accepts" begin
        # `compile_and_disasm` validates; reaching here at all is the assertion
        # the original failure would have broken.
        check(d, "OpTypeCooperativeMatrixKHR")
        check(d, "Workgroup")
    end

    @testset "integer components are signless, like every other integer" begin
        check_not(d, "OpTypeInt 8 1")
        check_not(d, "OpTypeInt 32 1")
        check(d, "OpTypeInt 8 0")
    end

    @testset "the product says which operands are signed" begin
        # The trailing operands word, which the disassembler prints by name.
        check(d, "MatrixASignedComponentsKHR")
        check(d, "MatrixBSignedComponentsKHR")
        check(d, "MatrixCSignedComponentsKHR")
        check(d, "MatrixResultSignedComponentsKHR")
    end
end

@testset "Cooperative matrix, unsigned 8-bit" begin
    # The same product over unsigned components emits no signed bits, which is
    # what says the flags come from the dtype rather than from being an integer.
    function u8_coopmat(C)
        pa = Lava.lava_alloc_shared(Val(:u8a), UInt8, Val(256))
        pb = Lava.lava_alloc_shared(Val(:u8b), UInt8, Val(256))
        pc = Lava.lava_alloc_shared(Val(:u32c), UInt32, Val(256))
        sa = Lava.LavaSharedArray{UInt8}(pa, 256)
        sb = Lava.LavaSharedArray{UInt8}(pb, 256)
        sc = Lava.LavaSharedArray{UInt32}(pc, 256)
        lid = Lava.lava_local_invocation_id_x()
        @inbounds sa[lid] = UInt8(lid % 7)
        @inbounds sb[lid] = UInt8(lid % 5)
        Lava.lava_workgroup_barrier()
        a = Lava.CoopMatrix{UInt8,16,16,Lava.MatrixA,Lava.SubgroupScope}(sa, 1, 16)
        b = Lava.CoopMatrix{UInt8,16,16,Lava.MatrixB,Lava.SubgroupScope}(sb, 1, 16)
        c = zero(Lava.CoopMatrix{UInt32,16,16,Lava.Accumulator,Lava.SubgroupScope})
        c = muladd(a, b, c)
        copyto!(sc, 1, 16, c)
        Lava.lava_workgroup_barrier()
        @inbounds C[lid] = sc[lid]
        return nothing
    end
    d, _ = compile_and_disasm(u8_coopmat, Tuple{Lava.LavaDeviceArray{UInt32,1}})
    check(d, "OpCooperativeMatrixMulAddKHR")
    check_not(d, "OpTypeInt 8 1")
    check_not(d, "SignedComponentsKHR")
end
