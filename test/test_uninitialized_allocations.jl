using Test, Lava, LLVM, StaticArrays
using Lava: LavaDeviceArray, lava_compile

# Julia's allocation optimizer zero-fills an object it moves to the stack unless
# the allocation is marked uninitialized, which codegen does only for a
# constructor that writes every field. `MArray(undef)`, and `@private` with it,
# then stored the whole array each time the allocation ran: for Hikari's 16
# BSDF records, 432 stores on every shading hit, ~19% of a frame on NVIDIA.

struct UninitializedRecord
    kind::UInt32
    data::NTuple{15, Float32}
    weight::Float32
end

# Records are written, then read, only below a run-time count.
function counted_records!(out::LavaDeviceArray{Float32,1}, src::LavaDeviceArray{Float32,1}, n::Int32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    records = MVector{16, UninitializedRecord}(undef)
    count = Int32(0)
    @inbounds for j in Int32(1):n
        v = src[i + UInt32(j)]
        if v > 0f0
            count += Int32(1)
            records[count] = UninitializedRecord(UInt32(j), ntuple(k -> v * k, Val(15)), v)
        end
    end
    total = 0f0
    @inbounds for j in Int32(1):count
        total += records[j].weight + records[j].data[3]
    end
    @inbounds out[i] = total
    return nothing
end

# The same array with an initializer: its stores are the program's own.
function initialized_records!(out::LavaDeviceArray{Float32,1}, src::LavaDeviceArray{Float32,1}, n::Int32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    records = MVector{16, UninitializedRecord}(ntuple(_ -> UninitializedRecord(UInt32(0), ntuple(_ -> 0f0, Val(15)), 0f0), Val(16)))
    @inbounds for j in Int32(1):n
        v = src[i + UInt32(j)]
        v > 0f0 && (records[j] = UninitializedRecord(UInt32(j), ntuple(k -> v * k, Val(15)), v))
    end
    total = 0f0
    @inbounds for j in Int32(1):Int32(16)
        total += records[j].weight
    end
    @inbounds out[i] = total
    return nothing
end

"Stores at constant indices into the kernel's one private array (float or retyped to words)."
function constant_array_stores(asm::AbstractString)
    var = match(r"(%\w+) = OpVariable %_ptr_Function__arr_\w+_uint_\d+ Function", asm)
    var === nothing && error("no private array in the kernel")
    chains = Set(m[1] for m in eachmatch(Regex("(%\\w+) = OpAccessChain %_ptr_Function_\\w+ $(var[1]) %uint_\\w+\n"), asm))
    return count(m -> m[1] in chains, eachmatch(r"OpStore (%\w+) ", asm))
end

@testset "allocations are uninitialized" begin
    @testset "every allocation call is marked" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, """
                declare ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10)) allockind("alloc")
                define void @owner() {
                  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) allockind("alloc")
                  %b = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null) allockind("alloc,uninitialized")
                  %c = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null)
                  ret void
                }
                """)
            Lava.uninitialized_allocations!(mod)
            @test LLVM.verify(mod) === nothing
            calls = [i for b in LLVM.blocks(mod.functions["owner"]) for i in LLVM.instructions(b) if i isa LLVM.CallInst]
            kinds = [[LLVM.value(a) for a in collect(LLVM.function_attributes(c))
                      if a isa LLVM.EnumAttribute && LLVM.kind(a) === :allockind] for c in calls]
            @test kinds == [[UInt64(9)], [UInt64(9)], [UInt64(9)]]  # alloc | uninitialized
        end
    end

    args = Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Int32}
    @testset "MArray(undef) is not filled" begin
        r = lava_compile(counted_records!, args)
        @test !occursin("call void @llvm.memset", r.pre_pass_ir)
        @test constant_array_stores(r.spirv_disasm) == 0
    end
    @testset "an initializer still stores" begin
        r = lava_compile(initialized_records!, args)
        @test constant_array_stores(r.spirv_disasm) >= 16
    end
end
