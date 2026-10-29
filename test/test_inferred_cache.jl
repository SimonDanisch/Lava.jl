module InferredCacheTests
using Test, Lava, GPUCompiler

const FACTOR = Float32(rand(10_000:99_999))
const TABLE = Float32[2, 3, 5]
const OPAQUE = Ref(2f0)
struct Nested{T}
    value::T
end
@noinline helper(x::Float32) = x*x + FACTOR
function kernel!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    i <= length(out) && (@inbounds out[i] = helper(Float32(i)))
    nothing
end
function global_kernel!(out)
    @inbounds out[1] = TABLE[1]
    nothing
end
function opaque_kernel!(out)
    @inbounds out[1] = OPAQUE[]
    nothing
end
function vararg_kernel!(out, values...)
    @inbounds out[1] = Float32(length(values))
    nothing
end
const TT = Tuple{Lava.LavaDeviceArray{Float32,1}}
job(f=kernel!; group=64) = Lava.lava_kernel_job(f, TT; workgroup_size=(group,1,1))

@static if Lava.HAS_OBJECT_CACHE && VERSION >= v"1.12.0-DEV.1823" &&
           isdefined(GPUCompiler.CompilerCaching, :get_codeinfos)

function valuekey(x)
    s = Lava.IRKeyState(Pair{Core.CodeInstance,Core.CodeInfo}[], Base.get_world_counter())
    Lava.irkey!(s, x)
    Lava.ObjCache.keyhash(1, take!(s.io))
end

@testset "inferred kernel cache" begin
    original_job = job()
    key = Lava.inferred_kernel_key(original_job)
    @test key !== nothing
    @test key == Lava.inferred_kernel_key(original_job)
    @test key != Lava.inferred_kernel_key(job(; group=128))
    @test Lava.inferred_kernel_key(job(opaque_kernel!)) === nothing
    @eval unrelated(x::String) = sizeof(x)
    @test key == Lava.inferred_kernel_key(Base.invokelatest(job))

    @testset "entry signatures and feature flags" begin
        first = Lava.lava_kernel_job(vararg_kernel!, TT; workgroup_size=(64,1,1))
        second = Lava.lava_kernel_job(vararg_kernel!, Tuple{TT.parameters...,Float32}; workgroup_size=(64,1,1))
        @test Lava.inferred_kernel_key(first) != Lava.inferred_kernel_key(second)
        query = Lava.lava_kernel_job(kernel!, TT; workgroup_size=(64,1,1), enable_ray_query=true,
                                    features=Lava.TargetFeatures(;ray_query=true))
        @test key != Lava.inferred_kernel_key(query)
        config = GPUCompiler.CompilerConfig(original_job.config; entry_abi=:func)
        @test key != Lava.inferred_kernel_key(GPUCompiler.CompilerJob(original_job; config))
    end

    @testset "global contents and aliasing" begin
        global_job = job(global_kernel!)
        initial = Lava.inferred_kernel_key(global_job)
        @test initial !== nothing
        try
            TABLE[1] = 7f0
            @test Lava.inferred_kernel_key(global_job) != initial
        finally
            TABLE[1] = 2f0
        end
        @test Lava.inferred_kernel_key(global_job) == initial
        a, b = Float32[2, 3], Float32[2, 3]
        @test valuekey((a,a)) != valuekey((a,b))
        @test_throws Lava.IRKeyUnsupported valuekey(Ptr{Int}(UInt(0x1234)))
        nested = Float32
        for _ in 1:160
            nested = Nested{nested}
        end
        @test_throws Lava.IRKeyUnsupported valuekey(nested)
    end

    @testset "type layouts with the same qualified name" begin
        # Runtime-created modules have the same name but distinct build nonces.
        # Only the layout change should distinguish their nominally equal types.
        a, b, c = Module(:LayoutProbe), Module(:LayoutProbe), Module(:LayoutProbe)
        Core.eval(a, :(struct Record; x::Float32; y::Int32; end))
        Core.eval(b, :(struct Record; x::Float32; y::Int32; end))
        Core.eval(c, :(struct Record; x::Float64; y::Int32; end))
        @test valuekey(a.Record) == valuekey(b.Record)
        @test valuekey(a.Record) != valuekey(c.Record)

        plain, constant, atomic = Module(:FieldProbe), Module(:FieldProbe), Module(:FieldProbe)
        Core.eval(plain, :(mutable struct Record; x::Float32; y::Int32; end))
        Core.eval(constant, :(mutable struct Record; const x::Float32; y::Int32; end))
        Core.eval(atomic, :(mutable struct Record; @atomic x::Float32; y::Int32; end))
        @test valuekey(plain.Record) != valuekey(constant.Record)
        @test valuekey(plain.Record) != valuekey(atomic.Record)

        left, right = Module(:HierarchyProbe), Module(:HierarchyProbe)
        for m in (left, right)
            Core.eval(m, :(abstract type ParentA end))
            Core.eval(m, :(abstract type ParentB end))
        end
        Core.eval(left, :(struct Record <: ParentA; x::Float32; end))
        Core.eval(right, :(struct Record <: ParentB; x::Float32; end))
        @test valuekey(left.Record) != valuekey(right.Record)
    end

    if Lava.ObjCache.enabled()
        original = Lava.persisted_kernel(original_job)
        # Bypass the session-tier kernel result: a persistent hit must skip
        # LLVM emission itself, not merely the optimization and SPIR-V stages.
        @test timedwait(() -> Lava.ObjCache.get("Lava/inferred-kernel", key) !== nothing, 10) == :ok
        Lava.reset_compile_stats!()
        Lava.reset_phase_log!()
        previous = Lava.enable_phase_capture!()
        hit = try
            Lava.persisted_kernel(original_job)
        finally
            Lava.enable_phase_capture!(previous)
        end
        @test hit.spirv_bytes == original.spirv_bytes
        @test hit.push_info.arg_layout == original.push_info.arg_layout
        @test Lava.compile_stats() == (;hits=1,misses=0)
        @test Lava.llvm_compile_stats() == (;hits=0,misses=0)
        @test isempty(Lava.phase_records())

        @eval @noinline helper(x::Float32) = x*x + $(FACTOR + 1f0)
        changed_job = Base.invokelatest(job)
        @test Lava.inferred_kernel_key(changed_job) != key
        changed = Lava.persisted_kernel(changed_job)
        @test changed.spirv_bytes != original.spirv_bytes
        @test changed.spirv_bytes == Lava.persisted_kernel_from_llvm(changed_job).spirv_bytes

        mktempdir() do dir
            Lava.reset_phase_log!()
            previous = Lava.enable_phase_capture!()
            try
                withenv("LAVA_DUMP_KERNELS"=>"1", "LAVA_DUMP_KERNELS_DIR"=>dir) do
                    Lava.persisted_kernel(changed_job)
                end
            finally
                Lava.enable_phase_capture!(previous)
            end
            @test any(endswith(".spv"), readdir(dir))
            @test any(r -> r.label == "GPUCompiler.compile(:llvm)", Lava.phase_records())
        end
    end
end

end
end
