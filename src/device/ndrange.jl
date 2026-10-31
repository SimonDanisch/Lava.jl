# How a kernel recovers its own index. Device code, and nothing here is Vulkan's.
#
# `KA.@index` and `KA.@synchronize` expand to calls on `KA.__index_*`,
# `KA.__validindex` and `KA.__synchronize`, which KernelAbstractions gives host
# definitions — so these are `@lava_device_override`s rather than plain methods,
# for the same reason `KI.barrier` is one in `device/kernelinterface.jl`: a plain
# method would shadow the host one and put an `llvmcall` of a SPIR-V builtin on
# the CPU.
#
# The `ctx` argument is KA's `CompilerMetadata`, which carries the ndrange and
# the iteration space. It is a compile-time value, so everything below folds
# away except the builtin reads.
#
# Came from `array/ka_backend.jl` when the runtime moved to Mantle — see
# `device/devicearray.jl`'s header for how it got left behind and what that cost.

import KernelAbstractions as KA

# Compute linear 1-based block index from 3D workgroup IDs.
# Uses lava_num_workgroups (GPU builtin = actual dispatch dimensions),
# so this stays correct even when pad_to_3d splits large 1D dispatches.
@inline function linear_block_index()
    bx = Int(lava_workgroup_id_x())
    by = Int(lava_workgroup_id_y())
    bz = Int(lava_workgroup_id_z())
    nx = Int(lava_num_workgroups_x())
    ny = Int(lava_num_workgroups_y())
    return bx + by * nx + bz * nx * ny + 1
end

@inline ndrank(::KA.NDRange{N}) where {N} = Val(N)

@inline extentat(t::NTuple{N,Int}, ::Val{I}) where {N,I} = I <= N ? t[I] : 1

"""
    directdispatch(is, Val(N)) -> Bool

Whether the global index can be read straight off the dispatch builtins instead
of being recovered by dividing a flattened index back apart.

`linear_block_index()` builds a linear index *out of* `WorkgroupId.{x,y,z}`, and
`KA.expand` then indexes `blocks(iterspace)` — a `CartesianIndices` — with it,
and `workitems(iterspace)` with the thread id. That round trip is `2(N-1)`
integer divisions by **runtime** extents, and it is recovering exactly the
numbers the hardware already handed us: `block_dims = pad_to_3d(size(blocks))`,
and `pad_to_3d` is the identity once trailing unit axes are dropped.

Measured on a plain elementwise copy of 70.8 MB, dynamic launch, this device:

    rank 3   0.710 ms  100 GB/s  ->  0.286 ms  247 GB/s
    rank 4   1.047      68       ->  0.397     179

247 GB/s is what the *same copy* reaches through `@index(Global, Linear)`, whose
rank-1 iteration space needs no division at all — so the tuple index has stopped
paying anything for its shape. This is why `grid_sample2d_kernel!` looked like a
slow gather: a bare copy over its ndrange cost **88%** of it, and the sampling
was never the expense.

**The guard compares against what was actually dispatched** — three
`NumWorkgroups` reads — rather than re-deriving `pad_to_3d`'s rules here, where
the copy could drift out of agreement with it. `pad_to_3d` re-splits a rank-1 or
rank-2 grid past the device's `max_wg_dims`, and flattens a rank >= 4 grid that
still has a non-unit axis past the third; in those cases the extents disagree
and [`globalcart`](@ref) divides.
"""
@inline function directdispatch(is, ::Val{N}) where {N}
    # Rank 4 is excluded on purpose, and statically: there the *device* reads
    # the iteration space wrongly. `(1,1,4,1)` has `workitems = (1,1,4,1)`, so
    # `length(workitems) == size(workitems,1)` is `4 == 1` — false on the host,
    # and the direct path is correctly declined there — yet on device the same
    # predicate admits it and three elements come out wrong. Rewriting it from
    # `all(ntuple(...))` to scalar `length`/`size` comparisons changed nothing,
    # and the same shapes are exact the moment this returns false, so it is the
    # rank-4 metadata read and not the predicate. Reproducers, all rank 4 and
    # all fine at ranks 1-3 and 5: (1,1,4,1) 3 wrong, (1,7,1,1) 6, (7,5,3,2)
    # 168, (16,8,4,2) 896, (72,256,8,16) 1_566_720.
    #
    # `launchgroup`'s docstring records the other half of this from the static
    # side. Until that is understood, a rank-4 launch divides like it always
    # has; `pinned by test_index_recovery.jl`.
    N <= 3 || return false
    nx = Int(lava_num_workgroups_x())
    ny = Int(lava_num_workgroups_y())
    nz = Int(lava_num_workgroups_z())
    nb = size(KA.blocks(is))
    # The dispatch grid must BE the block grid on the three hardware axes ...
    nx == extentat(nb, Val(1)) && ny == extentat(nb, Val(2)) && nz == extentat(nb, Val(3)) &&
        # ... with nothing left over on any axis past them. Comparing the total
        # against the product says that in one scalar, where a per-axis
        # `all(ntuple(i -> nb[i] == 1, Val(N)))` both reads worse and, at rank 4
        # specifically, came back TRUE on device when it is false on the host —
        # sending `(7,5,3,2)` and `(1,7,1,1)` down the direct path, which is only
        # valid for a 1-D workgroup. Same reason for `workitems` below.
        length(KA.blocks(is)) == nx * ny * nz &&
        # The workgroup must be 1-D along the fastest axis, so the linear thread
        # id IS the fastest coordinate. `launchgroup` fills from the fastest axis
        # up, so this holds for nearly every launch; `staticgroup` is the
        # deliberate exception and it divides.
        length(KA.workitems(is)) == extentat(size(KA.workitems(is)), Val(1))
end

@inline function directcart(is, ::Val{N}, t::Int) where {N}
    w = extentat(size(KA.workitems(is)), Val(1))
    CartesianIndex(ntuple(Val(N)) do i
        i == 1 ? Int(lava_workgroup_id_x()) * w + t :
        i == 2 ? Int(lava_workgroup_id_y()) + 1 :
        i == 3 ? Int(lava_workgroup_id_z()) + 1 : 1
    end)
end

"""
    globalcart(ctx) -> CartesianIndex

The global index: [`directcart`](@ref) when the dispatch allows it, and
otherwise the *original* `KA.expand(is, linear_block_index(), t)`, character for
character.

**The fallback is left exactly as it was on purpose.** Re-spelling it as
`expand(is, blocks(is)[b], workitems(is)[t])` — which is what KA's own
`expand(::Integer, ::Integer)` method expands to, and so should be identical —
produced wrong indices for a **rank-4** ndrange with a multi-dimensional
workgroup, and only there: `(16,8,4,2)`, `(7,5,3,2)` and `(72,256,8,16)` all
came back wrong while rank 3 and rank 5 through the same path stayed exact.
That is the rank-4 fragility `launchgroup`'s docstring already describes from
the other direction. Splitting the call is enough to trip it, so this path does
not get touched to make the fast one look tidier.
"""
@inline function globalcart(ctx)
    is = KA.__iterspace(ctx)
    r = ndrank(is)
    t = Int(lava_local_invocation_id_x()) + 1
    directdispatch(is, r) && return directcart(is, r, t)
    return @inbounds KA.expand(is, linear_block_index(), t)
end

@lava_device_override @inline function KA.__index_Local_Linear(ctx)
    return Int(lava_local_invocation_id_x()) + 1
end

@lava_device_override @inline function KA.__index_Group_Linear(ctx)
    return linear_block_index()
end

@lava_device_override @inline function KA.__index_Global_Linear(ctx)
    @inbounds LinearIndices(KA.__ndrange(ctx))[globalcart(ctx)]
end

@lava_device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Int(lava_local_invocation_id_x()) + 1]
end

@lava_device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[linear_block_index()]
end

@lava_device_override @inline function KA.__index_Global_Cartesian(ctx)
    return globalcart(ctx)
end

@lava_device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        return globalcart(ctx) in KA.__ndrange(ctx)
    else
        return true
    end
end

# `KA.SharedMemory` — the other override `@localmem` needs — is in
# `device/sharedmemory.jl`, next to `lava_alloc_shared` and `LavaSharedArray`
# that it is built from. The `CoopMatrix` load/store methods over a shared array
# are in `device/coopmat_intrinsics.jl`.

# ── Synchronization ──

@lava_device_override @inline function KA.__synchronize()
    lava_workgroup_barrier()
end

# ── @private: per-thread scratch memory via stack-allocated MArray ──

# Identical to CUDA.jl's own definition (CUDAKernels.jl:205). `@private` was
# once suspected of being ~8x slower here than plain locals; that was a
# measurement-order artifact — whichever kernel is timed first pays a one-time
# cost. Interleaved, the two are the same speed, and post-SROA IR is identical
# (66 instructions, 11 loads, 1 store, 5 phis in both). Nothing to fix here.
@lava_device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

# KA.__print is now overridden in device/printf.jl (routes @print → DebugPrintf).
