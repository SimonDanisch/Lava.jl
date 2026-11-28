using Test, Lava
import Lava.LLVM as LLVM

# Independent original traversal: LLVM views and source-block reachability,
# without the indexed graph introduced for large shaders.
function reference_shared_merges!(f)
    sources=Dict{LLVM.BasicBlock,Vector{LLVM.BasicBlock}}()
    targets=LLVM.BasicBlock[]
    for bb in f.blocks
        term=bb.terminator
        term isa LLVM.BrInst && LLVM.isconditional(term) || continue
        for target in term.successors
            target==bb && continue
            haskey(sources,target) || push!(targets,target)
            entries=get!(() -> LLVM.BasicBlock[],sources,target)
            bb in entries || push!(entries,bb)
        end
    end
    for target in targets, source in sources[target][2:end]
        Lava.insert_cfg_trampoline!(f,source,target)
    end
    return nothing
end

function shared_merge_fixture(n,cycle)
    out=IOBuffer()
    println(out,"define i32 @fixture(i1 %condition, i1 %repeat) {\nentry:\n br label %b1")
    for i in 1:n
        next=i<n ? "b$(i+1)" : cycle==2 ? "b$i" : "exit"
        println(out,"b$i:\n br i1 %condition, label %join, label %$next")
    end
    println(out,"exit:")
    println(out,cycle==1 ? " br i1 %repeat, label %b1, label %join" : " br label %join")
    incoming=join(["[ $i, %b$i ]" for i in 1:n],", ")
    println(out,"join:\n %result = phi i32 $incoming, [ 0, %exit ]\n ret i32 %result\n}")
    return String(take!(out))
end

@testset "Indexed CFG traversal preserves every edge and phi" begin
    LLVM.Context() do context
        for n in (1,2,8,32,128),cycle in 0:2
            ir=shared_merge_fixture(n,cycle)
            old=parse(LLVM.Module,ir)
            new=parse(LLVM.Module,ir)
            try
                reference_shared_merges!(old.functions["fixture"])
                Lava.isolate_shared_merge_targets!(new.functions["fixture"])
                @test LLVM.verify(old)===nothing
                @test LLVM.verify(new)===nothing
                @test string(new)==string(old)
            finally
                LLVM.dispose(old);LLVM.dispose(new)
            end
        end
        ir=shared_merge_fixture(512,1)
        old=parse(LLVM.Module,ir);new=parse(LLVM.Module,ir)
        try
            old_seconds=@elapsed reference_shared_merges!(old.functions["fixture"])
            new_seconds=@elapsed Lava.isolate_shared_merge_targets!(new.functions["fixture"])
            @test string(new)==string(old)
            println((cfg_blocks=512,original_seconds=old_seconds,indexed_seconds=new_seconds,
                speedup=old_seconds/new_seconds))
        finally
            LLVM.dispose(old);LLVM.dispose(new)
        end
    end
end

const CFG_TRAV_TEST_FIXTURES = (
    ("void", "ret void", "ret void"),
    ("i32", "ret i32 1", "ret i32 2"),
    ("void", "unreachable", "unreachable"),
    ("void", "ret void", "unreachable"),
    ("{ i32, i32 }",
     "ret { i32, i32 } { i32 1, i32 2 }",
     "ret { i32, i32 } { i32 3, i32 4 }"),
)

cfg_trav_test_ir(ty, a, b) =
    """
    declare void @external()

    define $ty @fixture(i1 %c) {
        entry:
            br i1 %c, label %a, label %b
        a:
            $a
        b:
            $b
    }

    define void @forever() {
        entry:
            br label %loop
        loop:
            br label %loop
    }
    """

@testset "Function exit unification" begin

    LLVM.Context() do context
        for (ty, a, b) in CFG_TRAV_TEST_FIXTURES

            ir = cfg_trav_test_ir(ty, a, b)

            LLVM.@dispose mod=parse(LLVM.Module, ir) begin

                values = [string(v) for bb in mod.functions["fixture"].blocks
                          for v in bb.terminator.operands if bb.terminator isa LLVM.RetInst]

                Lava.unify_function_exit_nodes!(mod)
                @test LLVM.verify(mod) === nothing

                terms = [bb.terminator for bb in mod.functions["fixture"].blocks]
                @test count(t -> t isa LLVM.RetInst, terms) == (a == "unreachable" ? 0 : 1)
                @test count(t -> t isa LLVM.UnreachableInst, terms) == (b == "unreachable" ? 1 : 0)

                if ty != "void"
                    phi = first(last(mod.functions["fixture"].blocks).instructions)
                    @test phi isa LLVM.PHIInst
                    @test [bb.name for (_, bb) in phi.incoming] == ["a", "b"]
                    @test [string(v) for (v, _) in phi.incoming] == values
                end

                unified = string(mod)
                Lava.unify_function_exit_nodes!(mod)
                @test string(mod) == unified
            end
        end
    end
end

