using Test, Lava, LLVM
@testset "Private constant globals retain caller storage class" begin
    for cast in (false, true)
        pointer = cast ? "ptr" : "ptr addrspace(1)"
        argument = cast ? "addrspacecast (ptr addrspace(1) @constant to ptr)" : "@constant"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                @constant = internal addrspace(1) constant [2 x float] [float 1.0, float 2.0]
                @out = internal addrspace(3) global [64 x float] undef
                define internal float @read_constant($pointer nocapture readonly %p) noinline {
                entry:
                  %q = getelementptr float, $pointer %p, i64 1
                  %v = load float, $pointer %q
                  ret float %v
                }
                define void @main() {
                entry:
                  %v = call float @read_constant($pointer $argument)
                  %outp = getelementptr [64 x float], ptr addrspace(3) @out, i64 0, i64 0
                  store float %v, ptr addrspace(3) %outp
                  ret void
                }
                """
            )
            @test Lava.has_alloca_type_mismatch_at_callsites(mod.functions["read_constant"], mod)
            Lava.force_inline_all!(mod, mod.functions["main"])
            @test !haskey(mod.functions, "read_constant") || isempty(mod.functions["read_constant"].blocks)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end
end
@testset "Function parameters retain caller container" begin
    for pun in (false, true), offset in (0, 4)
        element = pun ? "i32" : "float"
        constant = pun ? "1065353216" : "1.0"
        load = offset == 0 ? "%v = load float, ptr %p" : "%q = getelementptr i8, ptr %p, i64 4\n %v = load float, ptr %q"
        ir = """
        @out = internal addrspace(3) global [64 x float] undef
        define internal float @read_second(ptr nocapture readonly %p) noinline {
        entry:
          $load
          ret float %v
        }
        define void @main() {
        entry:
          %a = alloca [2 x $element]
          %p = getelementptr [2 x $element], ptr %a, i64 0, i64 $(offset ÷ 4)
          store $element $constant, ptr %p
          %v = call float @read_second(ptr %a)
          %outp = getelementptr [64 x float], ptr addrspace(3) @out, i64 0, i64 0
          store float %v, ptr addrspace(3) %outp
          ret void
        }
        """
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, ir)
            @test !Lava.has_alloca_type_mismatch_at_callsites(mod.functions["read_second"], mod)
            Lava.force_inline_all!(mod, mod.functions["main"])
            @test !isempty(mod.functions["read_second"].blocks)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end
end

@testset "Subobjects and incompatible caller layouts still inline" begin
    ir = """
    define internal float @read_second(ptr nocapture readonly %p) noinline {
    entry:
      %q = getelementptr i8, ptr %p, i64 4
      %v = load float, ptr %q
      ret float %v
    }
    define float @caller() {
    entry:
      %a = alloca [4 x i32]
      %b = alloca [8 x i32]
      %x = call float @read_second(ptr %a)
      %y = call float @read_second(ptr %b)
      %v = fadd float %x, %y
      ret float %v
    }
    """
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, ir)
        @test Lava.has_alloca_type_mismatch_at_callsites(mod.functions["read_second"], mod)
    end
    LLVM.Context() do ctx
        subobject = replace(ir, "%y = call float @read_second(ptr %b)" => "%sub = getelementptr i8, ptr %a, i64 4\n %y = call float @read_second(ptr %sub)")
        mod = parse(LLVM.Module, subobject)
        @test Lava.has_alloca_type_mismatch_at_callsites(mod.functions["read_second"], mod)
    end
end

@testset "Pointer-bearing descriptors keep context-local lowering" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            define internal float @descriptor(ptr %p) noinline {
            entry:
              %address = load i64, ptr %p
              %data = inttoptr i64 %address to ptr addrspace(1)
              %value = load float, ptr addrspace(1) %data
              ret float %value
            }
            define float @caller() {
            entry:
              %descriptor = alloca [2 x i64]
              %value = call float @descriptor(ptr %descriptor)
              ret float %value
            }
            """
        )
        @test Lava.has_alloca_type_mismatch_at_callsites(mod.functions["descriptor"], mod)
    end
end

@testset "Forwarded resource descriptors are not numeric arrays" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            define internal float @read_resource(ptr nocapture readonly %p) noinline {
            entry:
              %address = load i64, ptr %p
              %data = inttoptr i64 %address to ptr addrspace(1)
              %value = load float, ptr addrspace(1) %data
              ret float %value
            }
            define internal float @relay(ptr nocapture readonly %p) noinline {
            entry:
              %value = call float @read_resource(ptr %p)
              ret float %value
            }
            define float @caller() {
            entry:
              %descriptor = alloca [2 x i64]
              %value = call float @relay(ptr %descriptor)
              ret float %value
            }
            """
        )
        @test !Lava.private_numeric_helper(mod.functions["read_resource"])
        @test !Lava.private_numeric_helper(mod.functions["relay"])
        @test Lava.has_alloca_type_mismatch_at_callsites(mod.functions["relay"], mod)
    end
end
