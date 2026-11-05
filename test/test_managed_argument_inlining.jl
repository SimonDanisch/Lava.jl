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
