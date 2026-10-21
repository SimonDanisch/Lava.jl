# F33: ordering is an observable contract, not a code-generation hint.
module AtomicOrderingTests
using Test, Lava, LLVM
const UA = Lava.UnsafeAtomics

function emit(body; bits=32, storage=:buffer)
    prefix, ptrty, ptr = if storage === :workgroup
        ("@flag = internal addrspace(3) global i$bits 0\n", "ptr addrspace(3)",
         "%p = getelementptr i$bits, ptr addrspace(3) @flag, i32 0")
    else
        ("", "ptr", "%p = inttoptr i64 4096 to ptr")
    end
    occursin("%p",body) || (ptr = "")
    ir = prefix * "define void @main() {\nentry:\n$ptr\n" *
         replace(body, "PTR" => ptrty, "INT" => "i$bits") * "\nret void\n}"
    LLVM.Context() do _
        mod = parse(LLVM.Module, ir)
        try
            mod.datalayout = "e-p:64:64-i64:64"
            LLVM.verify(mod)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (1,1,1))
            Lava.validate_spirv(bytes)
            return bytes
        finally
            LLVM.dispose(mod)
        end
    end
end

# Read numeric operands, so tests assert the actual scope and semantics IDs,
# rather than just the existence of an opcode or a conveniently named constant.
function instructions(bytes, opcode)
    words = reinterpret(UInt32, bytes)
    constants = Dict{UInt32,UInt32}()
    found = Vector{UInt32}[]
    i = 6
    while i <= length(words)
        n = Int(words[i] >> 16)
        op = UInt16(words[i] & 0xffff)
        operands = collect(words[i+1:i+n-1])
        op == Lava.Op.OpConstant && length(operands) == 3 &&
            (constants[operands[2]] = operands[3])
        op == opcode && push!(found, operands)
        i += n
    end
    return found, constants
end
function semantics(bytes, opcode; scopeindex=4, semindex=5)
    found, constants = instructions(bytes, opcode)
    return [(constants[ops[scopeindex]], constants[ops[semindex]]) for ops in found]
end

function message(p::Ptr{Int32}, q::Ptr{Int32})
    value = UA.load(p, UA.acquire, UA.device)
    unsafe_store!(q, value)
    UA.store!(p, Int32(1), UA.release, UA.device)
    nothing
end
function ordered_fence(p::Ptr{Int32})
    unsafe_store!(p, Int32(2))
    UA.fence(UA.acq_rel, UA.device)
    nothing
end
function array_add(a)
    Lava.Atomix.@atomic :monotonic a[1] += one(eltype(a))
    nothing
end
function array_release(a)
    Lava.Atomix.@atomic :release a[1] += Int32(1)
    nothing
end
function array_default(a)
    Lava.Atomix.@atomic a[1] += Int32(1)
    nothing
end

function float_min(p::Ptr{Float32}, x::Float32)
    UA.modify!(p,min,x,UA.acq_rel,UA.device)
    nothing
end
function float_max(p::Ptr{Float32}, x::Float32)
    UA.modify!(p,max,x,UA.acq_rel,UA.device)
    nothing
end

@testset "LLVM atomic ordering is preserved" begin
    # UniformMemory | WorkgroupMemory | ImageMemory: the protected payload
    # need not reside in the same memory class as its synchronization flag.
    acquire = UInt32(0x4942)
    release = UInt32(0x2944)
    acqrel = UInt32(0x6948)
    @testset "atomic loads and stores, $bits-bit $storage" for bits in (32,64), storage in (:buffer,:workgroup)
        for (order, expected) in (("unordered",0), ("monotonic",0), ("acquire",acquire))
            bytes = emit("%v = load atomic INT, PTR %p syncscope(\"device\") $order, align $(bits÷8)"; bits,storage)
            @test semantics(bytes,Lava.Op.OpAtomicLoad) == [(1,expected)]
            @test isempty(first(instructions(bytes,Lava.Op.OpLoad)))
        end
        for (order, expected) in (("unordered",0), ("monotonic",0), ("release",release))
            bytes = emit("store atomic INT 1, PTR %p syncscope(\"device\") $order, align $(bits÷8)"; bits,storage)
            @test semantics(bytes,Lava.Op.OpAtomicStore;scopeindex=2,semindex=3) == [(1,expected)]
            @test isempty(first(instructions(bytes,Lava.Op.OpStore)))
        end
    end
    @testset "bit-preserving floating atomic accesses" begin
        for ty in ("float","double")
            bytes=emit("%v = load atomic $ty, ptr %p syncscope(\"device\") acquire, align 8\nstore atomic $ty %v, ptr %p syncscope(\"device\") release, align 8")
            @test semantics(bytes,Lava.Op.OpAtomicLoad) == [(1,acquire)]
            @test semantics(bytes,Lava.Op.OpAtomicStore;scopeindex=2,semindex=3) == [(1,release)]
            @test length(first(instructions(bytes,Lava.Op.OpBitcast))) == 2
        end
        for (f,opcode) in ((float_min,Lava.Op.OpAtomicFMinEXT),(float_max,Lava.Op.OpAtomicFMaxEXT))
            result=Lava.lava_compile_gpu(f,Tuple{Ptr{Float32},Float32})
            @test !isempty(first(instructions(result.spirv_bytes,opcode)))
            @test all(==((1,acqrel)),semantics(result.spirv_bytes,opcode))
        end
    end
    @testset "RMW, CAS success/failure and fences" begin
        for (order, expected) in (("monotonic",0), ("acquire",acquire), ("release",release), ("acq_rel",acqrel))
            bytes=emit("%v = atomicrmw add PTR %p, INT 1 syncscope(\"device\") $order")
            @test semantics(bytes,Lava.Op.OpAtomicIAdd) == [(1,expected)]
            for (failure,failsem) in (("monotonic",0),("acquire",acquire))
                # LLVM allows a stronger failure ordering since LLVM 8.
                bytes=emit("%v = cmpxchg PTR %p, INT 0, INT 1 syncscope(\"device\") $order $failure")
                successsem = failure == "acquire" && order == "monotonic" ? acquire :
                    failure == "acquire" && order == "release" ? acqrel : expected
                @test semantics(bytes,Lava.Op.OpAtomicCompareExchange) == [(1,successsem)]
                @test semantics(bytes,Lava.Op.OpAtomicCompareExchange;semindex=6) == [(1,failsem)]
            end
        end
        for (order, expected) in (("acquire",acquire),("release",release),("acq_rel",acqrel))
            bytes=emit("fence syncscope(\"device\") $order")
            @test semantics(bytes,Lava.Op.OpMemoryBarrier;scopeindex=1,semindex=2) == [(1,expected)]
        end
    end
    @testset "atomics through byte-offset pointers" begin
        for (body,opcode) in (
            ("%v = cmpxchg ptr %slot, i32 0, i32 1 syncscope(\"device\") acq_rel acquire",Lava.Op.OpAtomicCompareExchange),
            ("%v = atomicrmw add ptr %slot, i32 1 syncscope(\"device\") acq_rel",Lava.Op.OpAtomicIAdd))
            bytes=emit("%slot = getelementptr i8, ptr %p, i64 4\n"*body)
            @test semantics(bytes,opcode) == [(1,acqrel)]
        end
    end
    @testset "scope and volatile" begin
        for (scope,expected) in (("device",1),("workgroup",2),("subgroup",3),("singlethread",4),("queuefamily",5))
            order = scope == "singlethread" ? "monotonic" : "acquire"
            bytes=emit("%v = load atomic i32, ptr %p syncscope(\"$scope\") $order, align 4")
            @test semantics(bytes,Lava.Op.OpAtomicLoad) == [(expected,scope == "singlethread" ? 0 : acquire)]
            d=Lava.disassemble_spirv(bytes)
            @test occursin("OpCapability VulkanMemoryModelDeviceScope",d) == (scope == "device")
        end
        bytes=emit("%v = load atomic volatile i32, ptr %p syncscope(\"device\") acquire, align 4")
        @test semantics(bytes,Lava.Op.OpAtomicLoad) == [(1,acquire|0x8000)]
        bytes=emit("store atomic volatile i32 1, ptr %p syncscope(\"device\") release, align 4")
        @test semantics(bytes,Lava.Op.OpAtomicStore;scopeindex=2,semindex=3) == [(1,release|0x8000)]
    end
    @testset "unsupported requests fail at compilation" begin
        for body in (
            "%v = load atomic i32, ptr %p syncscope(\"device\") seq_cst, align 4",
            "store atomic i32 1, ptr %p syncscope(\"device\") seq_cst, align 4",
            "%v = atomicrmw add ptr %p, i32 1 syncscope(\"device\") seq_cst",
            "%v = cmpxchg ptr %p, i32 0, i32 1 syncscope(\"device\") seq_cst acquire",
            "%v = cmpxchg ptr %p, i32 0, i32 1 syncscope(\"device\") acq_rel seq_cst",
            "fence syncscope(\"device\") seq_cst")
            @test_throws r"cannot preserve sequentially consistent" emit(body)
        end
        @test_throws r"Unsupported ordered singlethread" emit("%v = load atomic i32, ptr %p syncscope(\"singlethread\") acquire, align 4")
        @test_throws r"Under-aligned atomic load" emit("%v = load atomic i32, ptr %p syncscope(\"device\") acquire, align 2")
        for scope in ("system","agent","unknown")
            @test_throws r"Unsupported LLVM atomic syncscope" emit("%v = load atomic i32, ptr %p syncscope(\"$scope\") acquire, align 4")
        end
        @test_throws r"Unsupported atomic access type" emit("%v = load atomic i16, ptr %p syncscope(\"device\") acquire, align 2")
    end
    @testset "Julia compiler and Atomix entry points" begin
        r=Lava.lava_compile_gpu(message,Tuple{Ptr{Int32},Ptr{Int32}})
        @test semantics(r.spirv_bytes,Lava.Op.OpAtomicLoad) == [(1,acquire)]
        @test semantics(r.spirv_bytes,Lava.Op.OpAtomicStore;scopeindex=2,semindex=3) == [(1,release)]
        r=Lava.lava_compile_gpu(ordered_fence,Tuple{Ptr{Int32}})
        @test semantics(r.spirv_bytes,Lava.Op.OpMemoryBarrier;scopeindex=1,semindex=2) == [(1,acqrel)]
        for T in (Int32,UInt32,Int64,UInt64,Float32)
            r=Lava.lava_compile_gpu(array_add,Tuple{Lava.LavaDeviceArray{T,1}})
            opcode=T===Float32 ? Lava.Op.OpAtomicFAddEXT : Lava.Op.OpAtomicIAdd
            @test semantics(r.spirv_bytes,opcode) == [(1,0)]
        end
        r=Lava.lava_compile_gpu(array_add,Tuple{Lava.LavaDeviceArray{Float64,1}})
        @test !isempty(first(instructions(r.spirv_bytes,Lava.Op.OpAtomicCompareExchange)))
        @test isempty(first(instructions(r.spirv_bytes,Lava.Op.OpAtomicFAddEXT)))
        r=Lava.lava_compile_gpu(array_release,Tuple{Lava.LavaDeviceArray{Int32,1}})
        @test semantics(r.spirv_bytes,Lava.Op.OpAtomicIAdd) == [(1,release)]
        @test_throws r"cannot preserve sequentially consistent" Lava.lava_compile_gpu(array_default,Tuple{Lava.LavaDeviceArray{Int32,1}})
    end
end
end # module
