# Regression: the same NVIDIA stride-product miscompile as
# test_repeat_inner_3d.jl, reached through *multi-index* `a[i, j, k, …]` rather
# than through `a[::CartesianIndex]`.
#
# `linear_index` was Horner-factored to dodge the miscompile, but only the
# CartesianIndex and linear overrides routed through it. A multi-index read had
# no `@lava_device_override`, so it fell back to Base's generic
# `_to_linear_index` → `_sub2ind`, which emits exactly the naive expansion
#   i1 + d1*(i2-1) + d1*d2*(i3-1) + d1*d2*d3*(i4-1)
# that the driver evaluates wrong when it feeds a PhysicalStorageBuffer offset.
# The SPIR-V is valid; the reads land at bogus addresses and return garbage.
#
# It only misfires when the indices are *computed*: `a[1,2,2,1]` constant-folds
# and is correct, while the same indices arriving from a delinearised loop index
# are not — which is why it hid until GPUArrays' `vectorized_getindex`
# (host/indexing.jl:84, `getindex_generated`) hit it. That silently truncated
# every strided slice: on a (3,2,2,1) array, `a[:, :, 2:2, :]` returned
# [7,8,9,0,0,0] instead of [7,8,9,10,11,12].
#
# Fixed in `array/ka_backend.jl` by adding multi-index getindex/setindex!
# overrides that route through the Horner `linear_index`.

using Test, Lava
using KernelAbstractions
const KA = KernelAbstractions

@testset "multi-index a[i,j,k,…] — NVIDIA stride-product miscompile" begin
    @testset "strided slices materialise correctly" begin
        h4 = reshape(Float32.(1:12), 3, 2, 2, 1)
        g4 = Lava.LavaArray(h4)
        @test Array(g4[:, :, 2:2, :]) == h4[:, :, 2:2, :]
        @test Array(g4[:, :, 2, :]) == h4[:, :, 2, :]
        @test Array(g4[:, :, 2, 1]) == h4[:, :, 2, 1]
        @test Array(g4[:, :, 1:1, :]) == h4[:, :, 1:1, :]
        @test Array(copy(view(g4, :, :, 2:2, :))) == h4[:, :, 2:2, :]

        h3 = reshape(Float32.(1:24), 2, 3, 4)
        g3 = Lava.LavaArray(h3)
        @test Array(g3[:, :, 2]) == h3[:, :, 2]
        @test Array(g3[:, 2:2, :]) == h3[:, 2:2, :]
        @test Array(g3[:, :, 2:3]) == h3[:, :, 2:3]

        h5 = reshape(Float32.(1:24), 2, 2, 3, 2, 1)
        g5 = Lava.LavaArray(h5)
        @test Array(g5[:, :, :, 1, :]) == h5[:, :, :, 1, :]
        @test Array(g5[:, :, 2, :, :]) == h5[:, :, 2, :, :]
    end

    # The direct form: indices computed inside the kernel, which is what the
    # driver miscompiled. Literal indices constant-fold and never reproduced it.
    @testset "computed multi-index reads" begin
        @kernel function readcomputed!(out, src, idims)
            i = @index(Global, Linear)
            is = @inbounds CartesianIndices(idims)[i]
            @inbounds out[i] = src[is[1], is[2], 2, 1]
        end
        h = reshape(Float32.(1:12), 3, 2, 2, 1)
        g = Lava.LavaArray(h)
        out = KA.allocate(Lava.LavaBackend(), Float32, 3, 2, 1, 1)
        fill!(out, -1.0f0)
        readcomputed!(Lava.LavaBackend())(out, g, (3, 2, 1, 1); ndrange=size(out))
        KA.synchronize(Lava.LavaBackend())
        @test vec(Array(out)) == vec(h[:, :, 2:2, :])
    end

    @testset "computed multi-index writes" begin
        @kernel function writecomputed!(dst, idims)
            i = @index(Global, Linear)
            is = @inbounds CartesianIndices(idims)[i]
            @inbounds dst[is[1], is[2], is[3], is[4]] = Float32(i)
        end
        dims = (3, 2, 2, 1)
        dst = KA.allocate(Lava.LavaBackend(), Float32, dims...)
        fill!(dst, -1.0f0)
        writecomputed!(Lava.LavaBackend())(dst, dims; ndrange=dims)
        KA.synchronize(Lava.LavaBackend())
        @test vec(Array(dst)) == Float32.(1:prod(dims))
    end
end
