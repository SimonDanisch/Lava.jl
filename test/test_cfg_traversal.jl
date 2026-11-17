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
