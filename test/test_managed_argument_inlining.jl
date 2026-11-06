using Test, Lava, LLVM

@testset "local allocations inline their helper calls before GC lowering" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
            declare ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
            declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
            define float @managed(ptr addrspace(10) %object) noinline {
              ret float 1.0
            }
            define float @physical(ptr %data) noinline {
              ret float 2.0
            }
            define float @nested(ptr addrspace(10) %object) noinline {
              ret float 3.0
            }
            define float @derived(ptr addrspace(11) %wrapper) noinline {
              %object = load ptr addrspace(10), ptr addrspace(11) %wrapper
              %r = call float @nested(ptr addrspace(10) %object)
              ret float %r
            }
            define float @derived_copy(ptr addrspace(11) %wrapper) noinline {
              %object = load ptr addrspace(10), ptr addrspace(11) %wrapper
              %r = call float @nested(ptr addrspace(10) %object)
              ret float %r
            }
            define ptr addrspace(10) @factory() noinline {
              %object = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null)
              ret ptr addrspace(10) %object
            }
            define float @returned(ptr addrspace(10) %object) noinline {
              ret float 5.0
            }
            define float @unrelated(ptr addrspace(11) %value) noinline {
              ret float 4.0
            }
            declare float @external(ptr addrspace(10))
            define void @owner(ptr addrspace(11) %argument) {
              %object = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null)
              %a = call float @managed(ptr addrspace(10) %object)
              %wrapper = alloca ptr addrspace(10)
              store ptr addrspace(10) %object, ptr %wrapper
              %view = addrspacecast ptr %wrapper to ptr addrspace(11)
              %b = call float @derived(ptr addrspace(11) %view)
              %data = addrspacecast ptr addrspace(10) %object to ptr
              %c = call float @physical(ptr %data)
              %d = call float @unrelated(ptr addrspace(11) %argument)
              %copy = alloca ptr addrspace(10)
              call void @llvm.memcpy.p0.p0.i64(ptr %copy, ptr %wrapper, i64 8, i1 false)
              %copied_view = addrspacecast ptr %copy to ptr addrspace(11)
              %e = call float @derived_copy(ptr addrspace(11) %copied_view)
              %made = call ptr addrspace(10) @factory()
              %f = call float @returned(ptr addrspace(10) %made)
              ret void
            }
            """)
        Lava.inline_managed_arguments!(mod)
        @test LLVM.verify(mod) === nothing
        for name in ("managed", "physical", "derived", "nested", "derived_copy", "factory", "returned")
            body = string(mod.functions[name])
            @test !occursin("noinline", body)
            @test occursin("alwaysinline", body)
        end
        # A Julia aggregate passed by reference is not necessarily a mutable
        # allocation. Do not override all such helpers' inlining decisions.
        unrelated = string(mod.functions["unrelated"])
        @test occursin("noinline", unrelated)
        @test !occursin("alwaysinline", unrelated)
        @test isempty(mod.functions["external"].blocks)
    end
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
            define float @argument_only(ptr addrspace(10) %object) noinline {
              ret float 1.0
            }
            """)
        before = string(mod)
        Lava.inline_managed_arguments!(mod)
        @test string(mod) == before
    end
end

# Julia tags a load through a by-reference argument as reading no stack memory
# of its caller (`!noalias`, TBAA, `!invariant.load`), which holds for a function
# compiled alone. Inlined into a caller that passes its own stack object, GVN
# trusted the claim and folded the load to `undef`: the pbrt CoatedDiffuse guide
# lost every return and its kernel failed SPIR-V validation.
const SEPARATE_COMPILATION_CLAIMS_IR = """
    declare ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
    declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
    define internal i64 @reader(ptr addrspace(10) %object, ptr addrspace(11) %term) noinline {
      %field = getelementptr inbounds i8, ptr addrspace(11) %term, i64 8
      %value = load i64, ptr addrspace(11) %field, align 8, !tbaa !5, !invariant.load !7, !alias.scope !4, !noalias !3
      ret i64 %value
    }
    define i64 @owner(i64 %x) {
      %object = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 16, ptr addrspace(10) null)
      %term = alloca [2 x i64], align 8
      %source = alloca [2 x i64], align 8
      %slot = getelementptr inbounds i8, ptr %source, i64 8
      store i64 %x, ptr %slot, align 8, !alias.scope !3, !noalias !4
      call void @llvm.memcpy.p0.p0.i64(ptr %term, ptr %source, i64 16, i1 false), !alias.scope !3, !noalias !4
      %view = addrspacecast ptr %term to ptr addrspace(11)
      %r = call i64 @reader(ptr addrspace(10) %object, ptr addrspace(11) %view)
      ret i64 %r
    }
    !0 = distinct !{!0, !"jnoalias"}
    !1 = distinct !{!1, !0, !"jnoalias_stack"}
    !2 = distinct !{!2, !0, !"jnoalias_const"}
    !3 = !{!1}
    !4 = !{!2}
    !5 = !{!6, !6, i64 0}
    !6 = !{!"jtbaa_const", !8, i64 0}
    !7 = !{}
    !8 = !{!"jtbaa"}
    """

@testset "force-inlined helpers drop their separate-compilation alias claims" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, SEPARATE_COMPILATION_CLAIMS_IR)
        Lava.inline_managed_arguments!(mod)
        reader = mod.functions["reader"]
        loads = [i for b in LLVM.blocks(reader) for i in LLVM.instructions(b) if i isa LLVM.LoadInst]
        @test all(i -> !haskey(LLVM.metadata(i), LLVM.MD_noalias) && !haskey(LLVM.metadata(i), LLVM.MD_tbaa) &&
                       !haskey(LLVM.metadata(i), LLVM.MD_invariant_load), loads)
        LLVM.run!(LLVM.AlwaysInlinerPass(), mod)
        LLVM.run!(LLVM.GVNPass(), mod)
        @test LLVM.verify(mod) === nothing
        owner = mod.functions["owner"]
        ret = LLVM.terminator(last(collect(LLVM.blocks(owner))))
        returned = first(LLVM.operands(ret))
        @test !(returned isa LLVM.UndefValue) && !(returned isa LLVM.PoisonValue)
    end
end
