using Test, Lava, KernelAbstractions
using Lava: AcceleratedMatrix, MatrixA, MatrixB, Accumulator, coopmat_gemm_available

# Can a cooperative matrix be loaded out of `@localmem`?
#
# It could not, until `loadw`/`storew`: `coopmat_base_pointer!` hardcoded
# `SC.PhysicalStorageBuffer`, so `OpCooperativeMatrixLoadKHR` could only address
# global memory. That is the one thing standing between `coopmat_gemm_kernel!`
# and a shared-memory staged GEMM, which is what every high-performance
# implementation does and what ours does not — see `perf-plan.md`.
#
# Smallest possible statement of it: stage a tile into shared, load a matrix
# from there, and compare against the same tile loaded from global. Anything
# larger would confound a codegen fault with a tiling bug.

const TILE = 16

@kernel cpu=false function stage_and_roundtrip!(out, @Const(inp))
    # One subgroup, one tile. `Val`-free literal sizes on purpose: a `@localmem`
    # extent that comes from a local variable miscompiles silently and the
    # kernel writes nothing at all.
    tile = @localmem Float16 (TILE * TILE,)
    i = @index(Local, Linear)
    @inbounds for k in 0:7                      # 32 lanes stage 256 elements
        j = i + k * 32
        tile[j] = inp[j]
    end
    @synchronize
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, 1, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

# Same, but reading the tile from `OFF` rather than from the start — the case
# that decides which SPIR-V construct the pointer comes from.
@kernel cpu=false function shared_at_offset!(out, @Const(inp), ::Val{OFF}) where {OFF}
    tile = @localmem Float16 (512,)
    i = @index(Local, Linear)
    @inbounds for k in 0:15
        tile[i + k * 32] = inp[i + k * 32]
    end
    @synchronize
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, OFF, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

@kernel cpu=false function global_roundtrip!(out, @Const(inp))
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(pointer(inp), 1, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

# A x B accumulated, with A staged through shared and B read from global: the
# shape the GEMM will actually use, where a matrix loaded from `Workgroup`
# memory has to be a valid operand of `OpCooperativeMatrixMulAddKHR` and not
# merely storable.
@kernel cpu=false function shared_a_gemm!(out, @Const(a), @Const(b))
    tile = @localmem Float16 (TILE * TILE,)
    i = @index(Local, Linear)
    @inbounds for k in 0:7
        j = i + k * 32
        tile[j] = a[j]
    end
    @synchronize
    A = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, 1, TILE)
    B = AcceleratedMatrix{Float16,TILE,TILE,MatrixB}(pointer(b), 1, TILE)
    C = zero(AcceleratedMatrix{Float32,TILE,TILE,Accumulator})
    C = muladd(A, B, C)
    copyto!(pointer(out), 1, TILE, C)
end

@testset "cooperative matrix from Workgroup memory" begin
    if !coopmat_gemm_available()
        @info "no cooperative-matrix support on this device — skipping"
    else
        backend = LavaBackend()
        h = Float16.(reshape(1:(TILE * TILE), TILE, TILE) ./ 16)

        @testset "load from shared == load from global" begin
            inp = LavaArray(vec(h))
            fromshared = LavaArray(zeros(Float16, TILE * TILE))
            fromglobal = LavaArray(zeros(Float16, TILE * TILE))
            stage_and_roundtrip!(backend, 32)(fromshared, inp; ndrange = 32)
            global_roundtrip!(backend, 32)(fromglobal, inp; ndrange = 32)
            KernelAbstractions.synchronize(backend)
            # Against the global path rather than against `h` directly: that
            # isolates the storage class, which is the only thing new here.
            @test Array(fromshared) == Array(fromglobal)
            @test Array(fromshared) == vec(h)          # …and both are right
        end

        @testset "at an offset into the tile" begin
            # The offset is what the whole lowering turns on. Passed as a GEP it
            # vanished at 1 (leaving a pointer to the array) and survived at 17
            # as a *constant expression* rather than an instruction, so no test
            # on the LLVM node kind separated the two — and the wrong branch
            # emitted an `OpBitcast` between Workgroup pointers, which is
            # illegal in Vulkan's Logical addressing model, validates clean, and
            # **segfaults NVIDIA's shader compiler**. The index is an argument
            # now; these three offsets are why.
            for off in (1, 17, 33)
                inp = LavaArray(Float16.(collect(1:512)))
                out = LavaArray(zeros(Float16, TILE * TILE))
                shared_at_offset!(backend, 32, 32)(out, inp, Val(off))
                KernelAbstractions.synchronize(backend)
                @test Array(out) == Float16.(collect(off:(off + TILE * TILE - 1)))
            end
        end

        @testset "a shared-loaded matrix multiplies" begin
            b = Float16.(reshape(1:(TILE * TILE), TILE, TILE) ./ 32)
            A = LavaArray(vec(h)); B = LavaArray(vec(b))
            C = LavaArray(zeros(Float32, TILE * TILE))
            shared_a_gemm!(backend, 32)(C, A, B; ndrange = 32)
            KernelAbstractions.synchronize(backend)
            want = Float32.(h) * Float32.(b)           # column-major throughout
            @test reshape(Array(C), TILE, TILE) ≈ want rtol = 1e-2
        end
    end
end
