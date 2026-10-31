using Test, Lava, LLVM

@testset "Wide reads from private numeric arguments" begin
    for offset in (0, 4)
        ir = """
        @out = internal addrspace(3) global [64 x i64] undef
        define internal i64 @read_pair(ptr nocapture readonly %p) noinline {
        entry:
          %q = getelementptr i8, ptr %p, i64 $offset
          %v = load i64, ptr %q, align 4
          ret i64 %v
        }
        define void @main() {
        entry:
          %a = alloca [3 x float]
          %p0 = getelementptr [3 x float], ptr %a, i64 0, i64 0
          %p1 = getelementptr [3 x float], ptr %a, i64 0, i64 1
          %p2 = getelementptr [3 x float], ptr %a, i64 0, i64 2
          store float 1.0, ptr %p0
          store float 2.0, ptr %p1
          store float 3.0, ptr %p2
          %v = call i64 @read_pair(ptr %a)
          %outp = getelementptr [64 x i64], ptr addrspace(3) @out, i64 0, i64 0
          store i64 %v, ptr addrspace(3) %outp
          ret void
        }
        """
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, ir)
            Lava.run_llvm_passes!(mod, mod.functions["main"])
            @test LLVM.verify(mod) === nothing
            @test haskey(mod.functions, "read_pair")
            @test !any(
                i -> i isa LLVM.LoadInst && i.value_type == LLVM.Int64Type(),
                (i for b in mod.functions["read_pair"].blocks for i in b.instructions)
            )
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end
end

@testset "Byte views cannot discard partially covered fields" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            define i64 @unaligned() {
            entry:
              %a = alloca [3 x float]
              %p = getelementptr i8, ptr %a, i64 2
              %v = load i64, ptr %p, align 1
              ret i64 %v
            }
            """
        )
        Lava.decompose_typepun_gep_loads!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        # The whole-field join cannot represent this unaligned access. It must
        # leave it for another lowering rather than silently lose its edge bytes.
        @test occursin("load i64", string(mod))
    end
end

@testset "Owned result fields exclude unspecified padding" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            %R = type { float, i8, float }
            @out = internal addrspace(3) global [64 x i32] undef
            define internal void @fill(ptr noalias nocapture sret(%R) %p) noinline {
            entry:
              store float 1.0, ptr %p
              %flag = getelementptr i8, ptr %p, i64 4
              store i8 7, ptr %flag
              %padding = getelementptr i8, ptr %p, i64 5
              store i8 9, ptr %padding
              %last = getelementptr i8, ptr %p, i64 8
              store float 2.0, ptr %last
              ret void
            }
            define void @main() {
            entry:
              %a = alloca %R
              call void @fill(ptr %a)
              %flag = getelementptr %R, ptr %a, i64 0, i32 1
              %value = load i8, ptr %flag
              %wide = zext i8 %value to i32
              %outp = getelementptr [64 x i32], ptr addrspace(3) @out, i64 0, i64 0
              store i32 %wide, ptr addrspace(3) %outp
              ret void
            }
            """
        )
        Lava.materialize_private_numeric_arguments!(mod)
        @test LLVM.verify(mod) === nothing
        @test count(
            i -> i isa LLVM.StoreInst && i.operands[1].value_type == LLVM.Int8Type(),
            (i for b in mod.functions["fill"].blocks for i in b.instructions)
        ) == 1
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end

@testset "Unqualified mutable arguments keep aliasing semantics" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            define internal void @update(ptr %p) noinline {
            entry:
              %v = load float, ptr %p
              store float %v, ptr %p
              ret void
            }
            define void @main() {
            entry:
              %a = alloca [3 x float]
              call void @update(ptr %a)
              ret void
            }
            """
        )
        before = string(mod)
        Lava.materialize_private_numeric_arguments!(mod)
        @test string(mod) == before
    end
end
