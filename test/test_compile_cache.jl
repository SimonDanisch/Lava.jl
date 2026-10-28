# The kernel cache: compiled SPIR-V kept with each kernel's `CodeInstance`
# (`compiler/cache.jl`), the way CUDA.jl keeps its images.
#
# Each property is checked with `compile_stats`, which counts compiles, rather
# than with the answer alone, which is the same whether anything was cached:
#
#   * a second lookup of a kernel compiles nothing;
#   * neither does one after an unrelated method definition, which moves the
#     world age but leaves the kernel's code valid;
#   * an edit to a function the kernel inlines compiles it again, from the new
#     code;
#   * code compiled before, in this process or an earlier one, is found in the
#     object cache when its `CodeInstance` is new: an edit undone compiles nothing;
#   * the kernels Lava's own precompile workload compiled are found in this
#     process without compiling, because they went into the package image.
#
# Device-free, like the rest of this suite.

using Test, Lava, GPUCompiler, LLVM

# Drawn per run: kernels outlive the process, so code an earlier run compiled is
# found, not compiled, and only code no run has seen is a compile.
const CC_FACTOR = Float32(rand(10_000:99_999))
cc_scale(x) = x * CC_FACTOR

function cc_kernel!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        @inbounds out[i] = cc_scale(Float32(i))
    end
    return nothing
end

const CC_TT = Tuple{Lava.LavaDeviceArray{Float32, 1}}

@noinline cc_shared(x::Float32) = x*x + CC_FACTOR + Float32(Lava.lava_global_invocation_id_x())
function cc_first!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    i <= length(out) && (@inbounds out[i] = cc_shared(Float32(i)))
    return nothing
end
function cc_second!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    i <= length(out) && (@inbounds out[i] = cc_shared(Float32(i)) + 7f0)
    return nothing
end

function cc_optimization!(out)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        @inbounds out[i] = sin(Float32(i)) + CC_FACTOR + 3f0
    end
    return nothing
end

cc_lookup() = Lava.compile_or_lookup(Lava.lava_kernel_job(cc_kernel!, CC_TT; workgroup_size = (64, 1, 1)))

function cc_source_locations_ir(line)
    """
    source_filename = "function_locations.jl"
    define i32 @helper(i32 %x) noinline !dbg !4 {
      %y = add i32 %x, $(Int(CC_FACTOR)), !dbg !7
      ret i32 %y, !dbg !7
    }
    define void @main() !dbg !8 {
      %r = call i32 @helper(i32 2), !dbg !9
      ret void, !dbg !9
    }
    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2}
    !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "cache test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
    !1 = !DIFile(filename: "function_locations.jl", directory: "/cache-test")
    !2 = !{i32 2, !"Debug Info Version", i32 3}
    !3 = !{}
    !4 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: $line, type: !5, scopeLine: $line, spFlags: DISPFlagDefinition, unit: !0)
    !5 = !DISubroutineType(types: !3)
    !7 = !DILocation(line: $line, column: 1, scope: !4)
    !8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 200, type: !5, scopeLine: 200, spFlags: DISPFlagDefinition, unit: !0)
    !9 = !DILocation(line: 200, column: 1, scope: !8)
    """
end

function cc_global_layout_ir(reverse, value)
    a="@a = private addrspace(1) constant [1 x i32] [i32 $value]"
    b="@b = private addrspace(1) constant [1 x i32] [i32 $(value+1)]"
    join(reverse ? (b,a) : (a,b),'\n') * """

    define i32 @helper() noinline {
      %p = getelementptr [1 x i32], ptr addrspace(1) @a, i32 0, i32 0
      %v = load i32, ptr addrspace(1) %p
      ret i32 %v
    }
    define void @main() {
      %p = getelementptr [1 x i32], ptr addrspace(1) @b, i32 0, i32 0
      %v = load i32, ptr addrspace(1) %p
      %h = call i32 @helper()
      ret void
    }
    """
end
function cc_builtin_layout_ir(reverse)
    a="@__spirv_BuiltInGlobalInvocationId = external addrspace(7) global <3 x i32>"
    b="@__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32"
    join(reverse ? (b,a) : (a,b),'\n') * """

    define i32 @helper() noinline {
      %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInGlobalInvocationId
      %x = extractelement <3 x i32> %v, i32 0
      %r = add i32 %x, $(Int(CC_FACTOR))
      ret i32 %r
    }
    define void @main() {
      %v = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex
      %h = call i32 @helper()
      ret void
    }
    """
end
@testset "compile cache" begin
    @testset "a lookup compiles once" begin
        Lava.reset_compile_stats!()
        first = cc_lookup()
        @test Lava.compile_stats() == (; hits = 0, misses = 1)
        @test cc_lookup() === first
        @test Lava.compile_stats() == (; hits = 1, misses = 1)
    end

    @testset "an unrelated definition compiles nothing" begin
        before = cc_lookup()
        Lava.reset_compile_stats!()
        @eval cc_unrelated() = 1
        @test Base.invokelatest(cc_lookup) === before
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
    end

    @testset "an edit to an inlined callee compiles again" begin
        before = cc_lookup()
        Lava.reset_compile_stats!()
        @eval cc_scale(x) = x * $(CC_FACTOR + 1f0)
        after = Base.invokelatest(cc_lookup)
        @test Lava.compile_stats() == (; hits = 0, misses = 1)
        @test after !== before
        @test after.spirv_bytes != before.spirv_bytes
    end

    # The edit undone makes a new `CodeInstance` of code compiled before, which
    # Julia numbers differently: only the object cache, keyed by the module without
    # those numbers (`canonical_ir`), knows it.
    @testset "code compiled before is not compiled again" begin
        @eval cc_scale(x) = x * $(CC_FACTOR + 2f0)
        first = Base.invokelatest(cc_lookup)
        @eval cc_unrelated_2() = 2
        @eval cc_scale(x) = x * $(CC_FACTOR + 3f0)
        Base.invokelatest(cc_lookup)
        Lava.reset_compile_stats!()
        @eval cc_scale(x) = x * $(CC_FACTOR + 2f0)
        again = Base.invokelatest(cc_lookup)
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
        @test again !== first
        @test again.spirv_bytes == first.spirv_bytes
        @test again.push_info.arg_layout == first.push_info.arg_layout
    end

    @testset "the package image's kernels compile nothing" begin
        Lava.reset_compile_stats!()
        Lava.compile_or_lookup(Lava.lava_kernel_job(
            Lava.precompile_warmup_kernel!,
            Tuple{Lava.LavaDeviceArray{Float32, 1}, Lava.LavaDeviceArray{Float32, 1}};
            workgroup_size = (64, 1, 1)))
        @test Lava.compile_stats() == (; hits = 1, misses = 0)
    end
    @testset "launch settings share inference and keep separate code" begin
        ordinary = Lava.lava_kernel_job(cc_kernel!, CC_TT; workgroup_size = (64, 1, 1))
        query = Lava.lava_kernel_job(cc_kernel!, CC_TT; workgroup_size = (32, 1, 1),
                                    enable_ray_query = true,
                                    features = Lava.TargetFeatures(; ray_query = true))
        a = GPUCompiler.drive_inference!(GPUCompiler.get_interpreter(ordinary), ordinary.source)
        b = GPUCompiler.drive_inference!(GPUCompiler.get_interpreter(query), query.source)
        @test a === b
        @test GPUCompiler.cache_owner(ordinary) === GPUCompiler.cache_owner(query)
        first = cc_lookup()
        other = Lava.compile_or_lookup(query)
        @test other !== first
        @test first.workgroup_size == (64, 1, 1)
        @test other.workgroup_size == (32, 1, 1)
        @test !first.enable_ray_query && other.enable_ray_query
        @test other.spirv_bytes != first.spirv_bytes
    end

    @testset "different kernels reuse a compiled helper with a shared builtin" begin
        lookup(f) = Lava.compile_or_lookup(Lava.lava_kernel_job(f, CC_TT; workgroup_size = (64, 1, 1)))
        Lava.reset_compile_stats!()
        first = lookup(cc_first!)
        @test Lava.function_compile_stats() == (; hits = 0, misses = 1)
        second = lookup(cc_second!)
        @test Lava.function_compile_stats() == (; hits = 1, misses = 1)
        @test second.spirv_bytes != first.spirv_bytes
        # A callee edit must invalidate the helper and both callers.
        @eval @noinline cc_shared(x::Float32) = x*x + $(CC_FACTOR + 1f0) + Float32(Lava.lava_global_invocation_id_x())
        Lava.reset_compile_stats!()
        changed = Base.invokelatest(lookup, cc_first!)
        @test Lava.function_compile_stats() == (; hits = 0, misses = 1)
        @test changed.spirv_bytes != first.spirv_bytes
        Base.invokelatest(lookup, cc_second!)
        @test Lava.function_compile_stats() == (; hits = 1, misses = 1)
    end

    @testset "identical helpers have one linked definition" begin
        GPUCompiler.JuliaContext() do _
            source = """
                define i32 @first(i32 %x) noinline {
                    %y = add i32 %x, $(Int(CC_FACTOR))
                    ret i32 %y
                }
                define i32 @second(i32 %x) noinline {
                    %y = add i32 %x, $(Int(CC_FACTOR))
                    ret i32 %y
                }
                define void @main() {
                    %a = call i32 @first(i32 1)
                    %b = call i32 @second(i32 2)
                    ret void
                }
                """
            mod = parse(LLVM.Module, source)
            Lava.reset_compile_stats!()
            bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            Lava.validate_spirv(bytes, source, locations)
            @test Lava.function_compile_stats() == (; hits = 1, misses = 1)
            dis = Lava.disassemble_spirv(bytes)
            @test count(r"OpFunction %uint DontInline", dis) == 1
            @test count("OpFunctionCall", dis) == 2
        end
    end
    @testset "helpers compile without the optional object cache" begin
        GPUCompiler.JuliaContext() do _
            ir = cc_source_locations_ir(10)
            mod = parse(LLVM.Module, ir)
            Lava.reset_compile_stats!()
            bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1);
                                                        cache_functions=false)
            Lava.validate_spirv(bytes, ir, locations)
            @test Lava.function_compile_stats() == (; hits=0, misses=0)
            @test ("/cache-test/function_locations.jl", 10) in values(locations)
            @test count("OpFunctionCall", Lava.disassemble_spirv(bytes)) == 1
            LLVM.dispose(mod)
        end
    end
    @testset "linked helpers keep current source locations on a cache hit" begin
        GPUCompiler.JuliaContext() do _
            Lava.reset_compile_stats!()
            for line in (10, 110)
                ir = cc_source_locations_ir(line)
                mod = parse(LLVM.Module, ir)
                bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                Lava.validate_spirv(bytes, ir, locations)
                @test ("/cache-test/function_locations.jl", line) in values(locations)
                @test ("/cache-test/function_locations.jl", 200) in values(locations)
                @test !occursin("lava_link_source_", Lava.disassemble_spirv(bytes))
                LLVM.dispose(mod)
            end
            @test Lava.function_compile_stats() == (; hits = 1, misses = 1)
        end
    end

    @testset "helpers bind the current global layout and initializers" begin
        GPUCompiler.JuliaContext() do _
            Lava.reset_compile_stats!()
            for (reverse, value) in ((false, Int(CC_FACTOR)),
                                     (true, Int(CC_FACTOR)),
                                     (true, Int(CC_FACTOR) + 2))
                Lava.reset_compile_stats!()
                mod = parse(LLVM.Module, cc_global_layout_ir(reverse, value))
                bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                @test Lava.function_compile_stats() == (; hits = 0, misses = 1)
                fresh, fresh_locations = withenv("LAVA_DUMP_KERNELS" => "1") do
                    Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                end
                # A reused import bound to a different global still validates.
                # Compare against an emission that cannot read the helper cache.
                @test bytes == fresh
                Lava.validate_spirv(bytes, "", locations)
                Lava.validate_spirv(fresh, "", fresh_locations)
                LLVM.dispose(mod)
            end
        end
    end

    @testset "builtin declarations remain part of the global interface" begin
        GPUCompiler.JuliaContext() do _
            Lava.reset_compile_stats!()
            for reverse in (false, true)
                mod = parse(LLVM.Module, cc_builtin_layout_ir(reverse))
                bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                Lava.validate_spirv(bytes, "", locations)
                @test !isempty(bytes)
                LLVM.dispose(mod)
            end
            @test Lava.function_compile_stats() == (; hits = 0, misses = 2)
        end
    end

    @testset "identical LLVM contexts emit identical decorations" begin
        types=join(["%s$i = type { i32, [$i x float], i8 }" for i in 1:12],"\n")
        loads=join(["%p$i = inttoptr i64 $(4096i) to ptr addrspace(1)\n%v$i = load %s$i, ptr addrspace(1) %p$i, align 4" for i in 1:12],"\n")
        source="target datalayout = \"e-p:64:64-i64:64-n8:16:32:64\"\n$types\ndefine void @main() {\n$loads\nret void\n}"
        binaries=Vector{UInt8}[]
        for _ in 1:8
            bytes=GPUCompiler.JuliaContext() do _
                mod=parse(LLVM.Module,source)
                bytes,locations=Lava.emit_spirv_from_llvm(mod,"main",(64,1,1))
                Lava.validate_spirv(bytes,source,locations)
                bytes
            end
            push!(binaries,bytes)
        end
        @test all(==(first(binaries)),binaries)
    end

    @testset "pointer arguments share their pointee and stride across functions" begin
        source = """
        target datalayout = "e-p:64:64-i64:64-n8:16:32:64"
        define float @helper(ptr addrspace(1) %a) noinline {
          %p = getelementptr float, ptr addrspace(1) %a, i32 1
          %v = load float, ptr addrspace(1) %p
          ret float %v
        }
        define void @main() {
          %p = inttoptr i64 4096 to ptr addrspace(1)
          %seed = load float, ptr addrspace(1) %p
          %r = call float @helper(ptr addrspace(1) %p)
          ret void
        }
        """
        indirect=replace(source,"%seed = load float, ptr addrspace(1) %p\n  %r = call float @helper(ptr addrspace(1) %p)"=>"%a = load ptr addrspace(1), ptr addrspace(1) %p\n  %r = call float @helper(ptr addrspace(1) %a)")
        for ir in (source,indirect)
            GPUCompiler.JuliaContext() do _
                mod=parse(LLVM.Module,ir)
                bytes,locs=Lava.emit_spirv_from_llvm(mod,"main",(64,1,1))
                Lava.validate_spirv(bytes,ir,locs)
                @test count("OpFunctionCall", Lava.disassemble_spirv(bytes)) == 1
            end
        end
    end

    @testset "LLVM optimization is reused before running its pass pipeline" begin
        Lava.reset_compile_stats!()
        job = Lava.lava_kernel_job(cc_optimization!, CC_TT; workgroup_size = (64, 1, 1))
        ir = Vector{UInt8}[]
        for _ in 1:2
            GPUCompiler.JuliaContext() do _
                mod, meta = Lava.lava_llvm(job)
                LLVM.verify(mod)
                push!(ir, Lava.canonical_ir(mod))
                compiled = Lava.lava_gpu_kernel(mod, meta, job)
                @test !isempty(compiled.spirv_bytes)
            end
        end
        @test ir[1] == ir[2]
        @test Lava.llvm_compile_stats() == (; hits = 1, misses = 1)
        changed = Lava.lava_kernel_job(cc_optimization!, CC_TT; workgroup_size = (128, 1, 1))
        GPUCompiler.JuliaContext() do _
            mod, _ = Lava.lava_llvm(changed)
            LLVM.verify(mod)
        end
        @test Lava.llvm_compile_stats() == (; hits = 1, misses = 2)
        Lava.reset_compile_stats!()
        @test Lava.llvm_compile_stats() == (; hits = 0, misses = 0)
    end

    @testset "optimization keys follow global uses rather than insertion order" begin
        GPUCompiler.JuliaContext() do _
            a = "@a = private global i32 17"
            b = "@b = private global i32 29"
            body = """
            define i32 @main() {
              %x = load i32, ptr @a
              %y = load i32, ptr @b
              %z = sub i32 %x, %y
              ret i32 %z
            }
            """
            first = parse(LLVM.Module, a * "\n" * b * "\n" * body)
            second = parse(LLVM.Module, b * "\n" * a * "\n" * body)
            changed = parse(LLVM.Module, replace(a, "17" => "18") * "\n" * b * "\n" * body)
            @test Lava.optimization_input(first) == Lava.optimization_input(second)
            @test Lava.optimization_input(first) != Lava.optimization_input(changed)
            @test Lava.optimization_symbols(first) == Lava.optimization_symbols(second)
        end
    end

end
