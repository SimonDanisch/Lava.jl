using Test, Lava

@testset "Phase 4 — single-writer enforcement + GC counter fix" begin

@testset "BatchQueue has owning_thread set to construction thread" begin
    bq = Lava.vk_context().default_bq
    @test hasfield(Lava.BatchQueue, :owning_thread)
    @test bq.owning_thread == Threads.threadid()
end

@testset "cross-thread dispatch trips the assert" begin
    # Only meaningful under `julia -t N` with N > 1.
    if Threads.nthreads() > 1
        bq = Lava.vk_context().default_bq
        result = Ref{Any}(nothing)
        # Run ensure_active_batch! from a different thread.
        #
        # `Threads.@spawn begin ... end |> wait` does NOT wait: a macro consumes
        # as much of the expression as it can, so that parses as
        # `@spawn (begin ... end |> wait)` — the wait runs INSIDE the task and
        # nothing joins it. `result[]` was then read before the task had run and
        # came back `nothing`, failing against a Lava that behaves correctly.
        task = Threads.@spawn begin
            try
                Lava.ensure_active_batch!(bq)
                result[] = :no_assert   # should not happen
            catch e
                result[] = e
            end
        end
        wait(task)
        @test result[] isa AssertionError
    else
        @info "Skipping cross-thread test; run with `julia -t 2+` to exercise"
    end
end

@testset "GPU_LIVE_BYTES is atomic" begin
    @test Lava.GPU_LIVE_BYTES isa Threads.Atomic{Int}
end

@testset "VkContext has no public nothing-default_bq path" begin
    # The inner constructor takes a raw primary queue and builds default_bq
    # itself; there's no way to get a VkContext with default_bq unset after
    # the constructor returns.
    @test fieldtype(Lava.VkContext, :default_bq) === Lava.BatchQueue
end

end  # @testset
