# ============================================================================
# Lava.HWTLAS — concrete hardware-accelerated TLAS living in Lava
# ============================================================================
#
# Single GPU-resident-instances code path: every push! produces an
# `InstanceBatch{Tri}` that owns a `LavaArray{LavaInstanceRecord, 1}`.
# Mutations (update_transform!/update_transforms!) write GPU-side via
# compute kernels and flag `transforms_dirty`.  `sync!` decides
# rebuild-vs-refit from the dirty flags.

import Raycore
import Adapt
using GeometryBasics
using StaticArrays: SVector, SMatrix
using Base: @propagate_inbounds
import KernelAbstractions
const KA = KernelAbstractions

const Mat4f = SMatrix{4, 4, Float32, 16}

# `unsafe_free!(::Nothing)` lives in `raytracing/acceleration.jl` alongside
# the other `unsafe_free!` methods so the method table is easy to audit.

# ============================================================================
# InstanceBatch struct
# ============================================================================

"""
    InstanceBatch{Tri}

A batch of N TLAS instances all referencing the same BLAS, with instance
records read from a GPU-resident `LavaArray{LavaInstanceRecord, 1}` at
sync! / refit time. Returned `handle` lets callers track the batch in
`HWTLAS.handle_to_batch_idx` for delete!/refit.

`triangles` holds the per-triangle metadata for the BLAS (typically
`Vector{Triangle{UInt32}}`). Pass an empty vector for rayQuery-only
callers that don't need Hikari's per-triangle TriangleMeta lookup.
"""
struct InstanceBatch{Tri}
    blas::LavaBLAS
    instance_buf::LavaArray{LavaInstanceRecord, 1}
    n::Int
    instance_mask::UInt8
    handle::Raycore.TLASHandle
    triangles::Vector{Tri}
end

# ============================================================================
# HWTLAS struct
# ============================================================================

"""
    HWTLAS{Tri} <: Raycore.AbstractAccel

Lava-native hardware-accelerated TLAS.  Concretely typed on `Tri` (the
per-primitive triangle type, typically `Raycore.Triangle{UInt32}`).

Build geometry with `push!(hwtlas, mesh, transform)`, then call
`Raycore.sync!(hwtlas)` to upload and build the Vulkan AS.  The adapted form
lives in `hwtlas.static_tlas` as a `HWAdaptedAccel{HWTLAS{Tri}}`.

# Mutation contract

`update_transform!` / `update_transforms!` write directly to the batch's
GPU-resident `instance_buf` via a compute kernel and flag
`transforms_dirty`.  The next `sync!` decides between full rebuild
(topology change, `dirty=true`) and `MODE_UPDATE_KHR` refit
(`transforms_dirty=true`).  No CPU-side staging.

# Adapted-form invariant

`sync!(hwtlas)` is the sole owner of `hwtlas.static_tlas`.  It rebuilds as
efficiently as possible — in place via `resize!`/`copyto!` where the backing
buffer can be reused, reallocated only when a buffer grew — and stores the
result in `hwtlas.static_tlas`.  `sync!` MAY reassign `hwtlas.static_tlas`
when a buffer was reallocated.

Every consumer that hands the accel to a raytracing dispatch MUST go through
`hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` (which reads /
refreshes `hwtlas.static_tlas`) per dispatch.  Both are cheap; `sync!` did
the heavy lifting.  **NEVER cache the `HWAdaptedAccel` returned by `adapt`
across mutations** — consumers that cache silently see stale geometry.

# Non-blocking sync!

`sync!(hwtlas)` does NOT call `KA.synchronize(backend)`.  Old backings are
dropped via `Lava.unsafe_free!`, which defers destruction through
`hwtlas.bq`'s timeline (`bq.deferred_as_frees` / `bq.deferred_frees`) when
prior dispatches are still in flight.  Phase-B pinning of RT closure leaves
(tri_gpu, off_gpu, hw_tlas, hw_accel) is what makes the timeline tracking
correct on the BDA path.

For a CPU-blocking drain use `Raycore.wait_for_gpu!(hwtlas)`, which calls
`vk_flush!(hwtlas.bq)` (waits on the HWTLAS's own queue specifically, not
the backend-wide queue the Raycore default `wait_for_gpu!` uses).
"""
mutable struct HWTLAS{Tri} <: Raycore.AbstractAccel
    backend::LavaBackend

    # BatchQueue for all AS builds and RT dispatches on this HWTLAS.
    bq::BatchQueue

    # Geometry (accumulated on push!)
    blas_list::Vector{LavaBLAS}
    blas_triangles::Vector{Vector{Tri}}
    blas_offsets::Vector{UInt32}

    # Instance batches -- N instances of one BLAS each, transforms in a GPU buffer.
    # Every push! produces one batch; per-mesh push! batches simply have n=1.
    instance_batches::Vector{InstanceBatch{Tri}}

    # Handle management: handle -> index into instance_batches.
    handle_to_batch_idx::Dict{Raycore.TLASHandle, Int}
    next_handle_id::UInt32

    # Bounding box (CPU-side, updated on push!)
    root_aabb::Raycore.Bounds3

    # Built on sync! — Lava-specific
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, HardwareAccel{Vector{Tri}}}
    tri_gpu::Union{Nothing, LavaArray{Tri, 1}}
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}
    # Combined instance buffer: concatenation of every batch's instance_buf.
    # Allocated/grown in rebuild_hw_tlas_from_batch! and reused across syncs +
    # refits.
    combined_instance_buf::Union{Nothing, LavaArray{LavaInstanceRecord, 1}}

    # GPU-adapted form, owned by sync!.  Consumers read this via
    # `hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` per dispatch.
    # `nothing` until first sync!.
    static_tlas::Any   # Union{Nothing, HWAdaptedAccel{HWTLAS{Tri}}}

    # Topology dirty: a push!/delete! changed the batch list, full rebuild needed.
    dirty::Bool
    # Transforms dirty: a kernel just wrote new records into a batch's
    # instance_buf, refit (MODE_UPDATE_KHR) is enough.
    transforms_dirty::Bool
end

"""
    HWTLAS{Tri}(backend::LavaBackend; bq=backend.bq) -> HWTLAS{Tri}

Construct an empty HWTLAS parametrised on triangle type `Tri`.
"""
function HWTLAS{Tri}(backend::LavaBackend; bq::BatchQueue=backend.bq) where {Tri}
    HWTLAS{Tri}(
        backend, bq,
        LavaBLAS[], Vector{Tri}[], UInt32[],
        InstanceBatch{Tri}[],
        Dict{Raycore.TLASHandle, Int}(),
        UInt32(1),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing,
        nothing,                  # combined_instance_buf
        nothing,
        true,                     # dirty
        false,                    # transforms_dirty
    )
end

"""
    HWTLAS(backend::LavaBackend; bq=backend.bq) -> HWTLAS{Triangle{UInt32}}

Default constructor — narrows to `Triangle{UInt32}`.
"""
HWTLAS(backend::LavaBackend; bq::BatchQueue=backend.bq) =
    HWTLAS{Raycore.Triangle{UInt32}}(backend; bq)

# ============================================================================
# HWAdaptedAccel
# ============================================================================

"""
    HWAdaptedAccel{H<:HWTLAS} <: Raycore.AbstractAdaptedAccel

GPU-adapted form of `HWTLAS`.  Dispatches ray tracing to hardware.
"""
struct HWAdaptedAccel{H<:HWTLAS} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
end

# ============================================================================
# PrecomputedHitsAccel (struct defined here so adapt_structure can reference it)
# ============================================================================

"""
    PrecomputedHitsAccel{R,T,O,Tri} <: Raycore.AbstractAdaptedAccel

Wraps a batch-traced result buffer plus the GPU triangle + offset arrays for
CPU-side `closest_hit` dispatch after `batch_trace_indirect`.

`empty` is a sentinel triangle returned on ray miss (instead of indexing
into `triangles[1]` which is UB when the geometry is empty).
"""
struct PrecomputedHitsAccel{R, T, O, Tri} <: Raycore.AbstractAdaptedAccel
    results::R
    triangles::T
    offsets::O
    empty::Tri
end

# ============================================================================
# Adapt.adapt_structure
# ============================================================================

"""
    Adapt.adapt_structure(to, hwtlas::HWTLAS) -> HWAdaptedAccel

Ensure `sync!` has run then return `hwtlas.static_tlas`.  Cheap on a clean
HWTLAS — do NOT cache the return across mutations.
"""
function Adapt.adapt_structure(to, hwtlas::HWTLAS{Tri}) where Tri
    Raycore.sync!(hwtlas)
    # `static_tlas` is typed `::Any` in the struct body (forward-reference to
    # HWAdaptedAccel{HWTLAS{Tri}}); narrow via return-assertion so callers see
    # a concrete type and the dispatch hot path stays type-stable.
    return hwtlas.static_tlas::HWAdaptedAccel{HWTLAS{Tri}}
end

"""
    Adapt.adapt_structure(to, p::PrecomputedHitsAccel) -> PrecomputedHitsAccel

Adapt all array fields to the target context; copy `empty` verbatim.
"""
function Adapt.adapt_structure(to, p::PrecomputedHitsAccel)
    PrecomputedHitsAccel(
        Adapt.adapt(to, p.results),
        Adapt.adapt(to, p.triangles),
        Adapt.adapt(to, p.offsets),
        p.empty,
    )
end

# ============================================================================
# Accessors
# ============================================================================

Raycore.world_bound(hwtlas::HWTLAS)    = hwtlas.root_aabb
Raycore.n_geometries(hwtlas::HWTLAS)   = length(hwtlas.blas_list)
Raycore.n_instances(hwtlas::HWTLAS)    = sum(b.n for b in hwtlas.instance_batches; init=0)

"""
    Raycore.instance_buffer(hwtlas::HWTLAS, handle::Raycore.TLASHandle) -> LavaArray{LavaInstanceRecord, 1}

Return the GPU instance buffer for the batch registered under `handle`. The
caller can write new instance records into the returned LavaArray (typically
via a compute kernel) and then call `Raycore.sync!(hwtlas)` to commit
the change to the underlying LavaTLAS via MODE_UPDATE_KHR.

Errors if the handle is not registered.
"""
function Raycore.instance_buffer(hwtlas::HWTLAS, handle::Raycore.TLASHandle)
    idx = get(hwtlas.handle_to_batch_idx, handle, nothing)
    idx === nothing && error("instance_buffer: invalid or deleted handle.")
    return hwtlas.instance_batches[idx].instance_buf
end

# RayMakie compat: hwtlas.instances -> lightweight length-only view
struct HWTLASInstances
    n::Int
end
Base.isempty(x::HWTLASInstances) = x.n == 0
Base.length(x::HWTLASInstances) = x.n

function Base.getproperty(hwtlas::HWTLAS, s::Symbol)
    if s === :instances
        n = sum(b.n for b in getfield(hwtlas, :instance_batches); init=0)
        return HWTLASInstances(n)
    end
    return getfield(hwtlas, s)
end

# ============================================================================
# wait_for_gpu!
# ============================================================================

"""
    Raycore.wait_for_gpu!(hwtlas::HWTLAS) -> hwtlas

Block until all pending GPU work on `hwtlas.bq` has completed.
Uses Lava's `vk_flush!` rather than `KA.synchronize(backend)`.
"""
function Raycore.wait_for_gpu!(hwtlas::HWTLAS)
    vk_flush!(hwtlas.bq)
    return hwtlas
end

# ============================================================================
# Mutation API
# ============================================================================

function hwtlas_add_geometry!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh) where {Tri}
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    verts = decompose(Point3f, nmesh)
    norms = Raycore.Normal3f.(Raycore.decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? Point2f[] : Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))

    has_meta = hasproperty(nmesh, :face_meta)
    n_faces = length(fs)

    cpu_triangles = let tris = Tri[]
        for i in 1:n_faces
            Raycore.is_degenerate_face(verts, indices, i) && continue
            meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
            push!(tris, Raycore.build_triangle(verts, norms, uvs, indices, i, meta))
        end
        tris
    end
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertex positions for BLAS build
    n_tris = length(cpu_triangles)
    blas_vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        vs = cpu_triangles[i].vertices
        for j in 1:3
            v = vs[j]
            blas_vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    blas_indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        blas_indices[i+1] = UInt32(i)
    end

    hw_blas = as_build() do ctx
        build_blas(ctx, blas_vertices, blas_indices)
    end

    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = Raycore.union(hwtlas.root_aabb, Raycore.Bounds3(Point3f(v)))
        end
    end

    hwtlas.dirty = true
    return blas_idx
end

# Internal: register an InstanceBatch for `blas` with `n` instances initially
# populated from the CPU-side `records` Vector.  Returns the new handle.
function _register_batch!(hwtlas::HWTLAS{Tri}, blas::LavaBLAS,
                          records::Vector{LavaInstanceRecord},
                          triangles::Vector{Tri},
                          instance_mask::UInt8) where {Tri}
    n = length(records)
    instance_buf = LavaArray{LavaInstanceRecord, 1}(undef, n; extra_usage=AS_INPUT_USAGE)
    Base.copyto!(instance_buf, records)
    handle = Raycore.TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    push!(hwtlas.instance_batches,
          InstanceBatch{Tri}(blas, instance_buf, n, instance_mask, handle, triangles))
    hwtlas.handle_to_batch_idx[handle] = lastindex(hwtlas.instance_batches)
    hwtlas.dirty = true
    return handle
end

function Base.push!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff)) where {Tri}
    blas_idx  = hwtlas_add_geometry!(hwtlas, mesh)
    blas      = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    record    = LavaInstanceRecord(mat4_to_vk_transform(transform), blas.address;
                                   custom_index=instance_id, mask=instance_mask)
    return _register_batch!(hwtlas, blas, [record], triangles, instance_mask)
end

function Base.push!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                    instance_mask::UInt8=UInt8(0xff)) where {Tri}
    if instance_ids !== nothing && length(instance_ids) != length(transforms)
        throw(ArgumentError("instance_ids length $(length(instance_ids)) != transforms length $(length(transforms))"))
    end
    blas_idx  = hwtlas_add_geometry!(hwtlas, mesh)
    blas      = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    addr      = blas.address
    records = Vector{LavaInstanceRecord}(undef, length(transforms))
    @inbounds for i in eachindex(transforms)
        iid = instance_ids === nothing ? UInt32(0) : UInt32(instance_ids[i])
        records[i] = LavaInstanceRecord(mat4_to_vk_transform(transforms[i]), addr;
                                        custom_index=iid, mask=instance_mask)
    end
    return _register_batch!(hwtlas, blas, records, triangles, instance_mask)
end

"""
    push!(hwtlas::HWTLAS, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
          instance_id::UInt32=UInt32(0)) -> TLASHandle

Register a pre-built `LavaBLAS` (e.g. from `build_blas_aabb`) as a new
geometry + instance.  No triangle data is associated; accordingly the
`hw_accel` / ray-tracing pipeline path is not usable on this HWTLAS after
this call.  Use the compute-rayQuery path (`lava_launch!` with `tlas=hwtlas`)
instead.
"""
function Base.push!(hwtlas::HWTLAS{Tri}, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff)) where {Tri}
    # Register the pre-built BLAS — no triangles.
    push!(hwtlas.blas_list, blas)
    push!(hwtlas.blas_triangles, Tri[])
    blas_idx = length(hwtlas.blas_list)
    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    record = LavaInstanceRecord(mat4_to_vk_transform(transform), blas.address;
                                custom_index=instance_id, mask=instance_mask)
    return _register_batch!(hwtlas, blas, [record], Tri[], instance_mask)
end

"""
    push!(tlas::HWTLAS{Tri}, blas::LavaBLAS,
          instance_buf::LavaArray{LavaInstanceRecord, 1};
          n::Integer, instance_mask::UInt8,
          triangles::Vector{Tri}) -> Raycore.TLASHandle

Register an N-instance batch in the TLAS. All N instances reference the
same `blas`; per-instance transforms / custom_indices live in `instance_buf`
and are written by a GPU compute kernel (see `write_grain_instances_kernel`).

`n` defaults to `length(instance_buf)`. Pass a smaller value when the buffer
is pre-allocated larger than the current live instance count.

`triangles` supplies the BLAS's per-triangle metadata for Hikari's path tracer
(`tri_gpu` / `off_gpu` lookup). Pass empty (the default) for rayQuery-only
callers -- those don't need per-triangle metadata.

Returns one `TLASHandle` for the whole batch. Subsequent `sync!` builds
the underlying `LavaTLAS` with `allow_update=true` so per-frame refits work.
"""
function Base.push!(tlas::HWTLAS{Tri}, blas::LavaBLAS,
                    instance_buf::LavaArray{LavaInstanceRecord, 1};
                    n::Integer = length(instance_buf),
                    instance_mask::UInt8 = UInt8(0xff),
                    triangles::Vector{Tri} = Tri[]) where {Tri}
    n_int = Int(n)
    n_int <= length(instance_buf) || error(
        "push!: n=$n_int exceeds instance_buf length $(length(instance_buf))")
    handle = Raycore.TLASHandle(tlas.next_handle_id)
    tlas.next_handle_id += UInt32(1)
    push!(tlas.instance_batches, InstanceBatch{Tri}(blas, instance_buf, n_int, instance_mask, handle, triangles))
    tlas.handle_to_batch_idx[handle] = lastindex(tlas.instance_batches)
    tlas.dirty = true
    return handle
end

"""
    push!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh,
          instance_buf::LavaArray{LavaInstanceRecord, 1};
          n::Integer, instance_mask::UInt8) -> Raycore.TLASHandle

Build a BLAS from `mesh` (or reuse a cached one) and register an N-instance
batch backed by the GPU-resident `instance_buf`. The per-instance transforms
and custom_indices are read from `instance_buf` at `sync!` time -- typically
written by a compute kernel such as `write_meshscatter_instances_kernel`.

`n` defaults to `length(instance_buf)`. Pass a smaller value when the buffer
is pre-allocated larger than the current live instance count.

Returns one `TLASHandle` for the whole batch.
"""
function Base.push!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh,
                    instance_buf::LavaArray{LavaInstanceRecord, 1};
                    n::Integer = length(instance_buf),
                    instance_mask::UInt8 = UInt8(0xff)) where {Tri}
    n_int = Int(n)
    n_int <= length(instance_buf) || error(
        "push!: n=$n_int exceeds instance_buf length $(length(instance_buf))")
    blas_idx = hwtlas_add_geometry!(hwtlas, mesh)
    blas = hwtlas.blas_list[blas_idx]
    triangles = hwtlas.blas_triangles[blas_idx]
    handle = Raycore.TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    push!(hwtlas.instance_batches, InstanceBatch{Tri}(blas, instance_buf, n_int, instance_mask, handle, triangles))
    hwtlas.handle_to_batch_idx[handle] = lastindex(hwtlas.instance_batches)
    hwtlas.dirty = true
    return handle
end

# ============================================================================
# GPU update kernel + update_transform!/update_transforms!
# ============================================================================

"""
    update_instance_records_kernel!(records, transforms, blas_address, cim, sof)

One thread per instance.  Reads transforms[i] (any indexable 4×4-ish matrix),
packs the upper 3×4 row-major into a `Mat3x4f`, and writes
`records[i] = LavaInstanceRecord(T, cim, sof, blas_address)`.

`cim` packs custom_index (low 24 bits) and mask (high 8 bits); `sof` packs
sbt_offset (low 24 bits) and flags (high 8 bits).
"""
KA.@kernel function update_instance_records_kernel!(
        records,
        @Const(transforms),
        blas_address::UInt64,
        cim::UInt32,
        sof::UInt32)
    i = @index(Global, Linear)
    @inbounds m = transforms[i]
    # Pack row-major 3×4 floats into the Mat3x4f tuple ordering. The
    # NTuple{12,Float32} -> LavaInstanceRecord constructor handles the
    # Mat3x4f conversion (which is byte-identical, no transpose).
    T = (
        Float32(m[1,1]), Float32(m[1,2]), Float32(m[1,3]), Float32(m[1,4]),
        Float32(m[2,1]), Float32(m[2,2]), Float32(m[2,3]), Float32(m[2,4]),
        Float32(m[3,1]), Float32(m[3,2]), Float32(m[3,3]), Float32(m[3,4]),
    )
    @inbounds records[i] = LavaInstanceRecord(T, cim, sof, blas_address)
end

"""
    Raycore.update_transforms!(hwtlas::HWTLAS, handle::TLASHandle,
                               transforms::AbstractVector{<:AbstractMatrix})

Bulk update every instance in `handle`'s batch. `length(transforms)` must
equal the batch size. Writes new records into the batch's GPU-resident
`instance_buf` via a compute kernel and flags `transforms_dirty`. The next
`sync!` issues a `MODE_UPDATE_KHR` refit (no rebuild).
"""
function Raycore.update_transforms!(hwtlas::HWTLAS, handle::Raycore.TLASHandle,
                                    transforms::AbstractVector{<:AbstractMatrix})
    haskey(hwtlas.handle_to_batch_idx, handle) || error("Invalid handle")
    batch = hwtlas.instance_batches[hwtlas.handle_to_batch_idx[handle]]
    length(transforms) == batch.n || error(
        "Transform count $(length(transforms)) != batch.n $(batch.n)")
    backend = KA.get_backend(batch.instance_buf)
    backend_transforms = Adapt.adapt(backend, transforms)
    cim = (UInt32(0) & 0x00FFFFFF) | (UInt32(batch.instance_mask) << 24)
    sof = UInt32(0)
    update_instance_records_kernel!(backend)(
        batch.instance_buf, backend_transforms,
        batch.blas.address, cim, sof;
        ndrange = batch.n)
    hwtlas.transforms_dirty = true
    return nothing
end

"""
    update_transform!(hwtlas::HWTLAS, handle::TLASHandle, transform::Mat4f)

Set every instance in `handle`'s batch to the same transform. Returns true
if the handle was valid (matches the previous return semantics).
"""
function Raycore.update_transform!(hwtlas::HWTLAS, handle::Raycore.TLASHandle, transform::Mat4f)
    haskey(hwtlas.handle_to_batch_idx, handle) || return false
    batch = hwtlas.instance_batches[hwtlas.handle_to_batch_idx[handle]]
    Raycore.update_transforms!(hwtlas, handle, fill(transform, batch.n))
    return true
end

# ============================================================================
# delete!
# ============================================================================

function Base.delete!(hwtlas::HWTLAS, handle::Raycore.TLASHandle)::Bool
    idx = get(hwtlas.handle_to_batch_idx, handle, nothing)
    idx === nothing && return false
    deleteat!(hwtlas.instance_batches, idx)
    delete!(hwtlas.handle_to_batch_idx, handle)
    # Reindex other handles whose batch index shifted left by one.
    for (h, j) in hwtlas.handle_to_batch_idx
        j > idx && (hwtlas.handle_to_batch_idx[h] = j - 1)
    end
    hwtlas.dirty = true
    return true
end

# ============================================================================
# Rebuild / refit helpers + sync!
# ============================================================================

# Try to reuse `prev` as the GPU sink for `data` via capacity-aware resize+copyto.
function _reuse_or_alloc(prev, data::AbstractArray{T}) where T
    if prev isa LavaArray{T}
        resize!(prev, length(data))
        copyto!(prev, data)
        return prev
    end
    return LavaArray(data)
end

# Allocate-or-reuse a combined LavaArray{LavaInstanceRecord} that fits all
# `total` records, then GPU-copy each batch's instance_buf into the right
# offset.  The combined buffer is what build_tlas / refit_tlas! sees.
function _concat_batch_instances!(hwtlas::HWTLAS{Tri}) where {Tri}
    total = 0
    for batch in hwtlas.instance_batches
        total += batch.n
    end
    combined = hwtlas.combined_instance_buf
    if combined === nothing || length(combined) < total
        combined = LavaArray{LavaInstanceRecord, 1}(undef, total;
                                                       extra_usage=AS_INPUT_USAGE)
        hwtlas.combined_instance_buf = combined
    end
    inst_offset = 0
    for batch in hwtlas.instance_batches
        # GPU->GPU copy at the right offset.  copyto! on LavaArray uses
        # vkCmdCopyBuffer (no CPU staging).
        Base.copyto!(combined, inst_offset + 1, batch.instance_buf, 1, batch.n)
        inst_offset += batch.n
    end
    return combined, total
end

# Compact `blas_list` / `blas_triangles` / `blas_offsets` to drop entries
# no longer referenced by any live batch.  Returns the dropped BLASes so the
# caller can `unsafe_free!` them after the rebuild.
function _compact_blas_list!(hwtlas::HWTLAS{Tri}) where {Tri}
    n_blas = length(hwtlas.blas_list)
    n_blas == 0 && return LavaBLAS[]
    used = Set{Int}()
    # Identify each BLAS by reference identity (===) since multiple batches
    # may share the same LavaBLAS object.
    for batch in hwtlas.instance_batches
        for (i, blas) in enumerate(hwtlas.blas_list)
            blas === batch.blas && push!(used, i)
        end
    end
    length(used) == n_blas && return LavaBLAS[]

    dropped = LavaBLAS[]
    new_blas       = LavaBLAS[]
    new_triangles  = Vector{Tri}[]
    new_offsets    = UInt32[]
    running_offset = UInt32(0)
    for i in 1:n_blas
        if i in used
            push!(new_blas, hwtlas.blas_list[i])
            push!(new_triangles, hwtlas.blas_triangles[i])
            push!(new_offsets, running_offset)
            running_offset += UInt32(length(hwtlas.blas_triangles[i]))
        else
            push!(dropped, hwtlas.blas_list[i])
        end
    end
    hwtlas.blas_list      = new_blas
    hwtlas.blas_triangles = new_triangles
    hwtlas.blas_offsets   = new_offsets
    return dropped
end

function rebuild_hw_tlas_from_batch!(hwtlas::HWTLAS{Tri}) where {Tri}
    # Drop unused BLASes (deleted batches may have left some unreferenced).
    dropped_blases = _compact_blas_list!(hwtlas)

    if isempty(hwtlas.instance_batches)
        # Caller (sync!) handles the empty path before us; assert defensively.
        return (nothing, nothing, nothing, nothing, dropped_blases)
    end

    # Rebuild always starts with a fresh combined buffer: the previous one
    # (if any) is now solely owned by the soon-to-be-freed `hw_tlas.preserves`.
    # Reusing it across builds shares a single LavaArray identity between
    # multiple TLAS preserves lists, and the older TLAS's `unsafe_free!` would
    # then free the buffer out from under the live TLAS.
    hwtlas.combined_instance_buf = nothing
    combined, total_n = _concat_batch_instances!(hwtlas)

    hw_tlas = as_build() do ctx
        build_tlas(ctx, combined, total_n; allow_update=true)
    end

    # Concatenate triangle metadata across batches.  Each batch's instances all
    # reference one BLAS, so all instances of batch i share the same offset
    # into all_tris (= the running tri_offset where batch i's triangles begin).
    all_tris = Tri[]
    per_inst_offsets = UInt32[]
    blas_offsets = UInt32[]
    tri_offset = UInt32(0)
    for batch in hwtlas.instance_batches
        push!(blas_offsets, tri_offset)
        append!(all_tris, batch.triangles)
        for _ in 1:batch.n
            push!(per_inst_offsets, tri_offset)
        end
        tri_offset += UInt32(length(batch.triangles))
    end

    hw_accel = if hwtlas.hw_accel isa HardwareAccel
        accel_prev = hwtlas.hw_accel
        accel_prev.tlas = hw_tlas
        accel_prev.triangle_data = all_tris
        accel_prev.blas_offsets = blas_offsets
        accel_prev.per_instance_tri_offsets = per_inst_offsets
        accel_prev
    else
        HardwareAccel(hw_tlas, all_tris, blas_offsets, per_inst_offsets; bq=hwtlas.bq)
    end

    tri_gpu = _reuse_or_alloc(hwtlas.tri_gpu, all_tris)
    off_gpu = _reuse_or_alloc(hwtlas.off_gpu, per_inst_offsets)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu, dropped_blases)
end

"""
    Raycore.sync!(hwtlas::HWTLAS)

Single commit boundary. Decides between full rebuild (topology change) and
in-place refit (transforms-only update) from the dirty flags. No-op when
nothing changed.

Does NOT call KA.synchronize — uses Lava's timeline-based deferred free path.
"""
function Raycore.sync!(hwtlas::HWTLAS)
    # True no-op when nothing changed and static_tlas is already built.
    if !hwtlas.dirty && !hwtlas.transforms_dirty && hwtlas.static_tlas !== nothing
        return hwtlas
    end

    # Empty-topology path: drop the prior AS without rebuilding.
    if hwtlas.dirty && isempty(hwtlas.instance_batches)
        # Even with no batches we may still have unreferenced BLASes (e.g.,
        # the user delete!'d every handle).  Compact + free.
        dropped_blases = _compact_blas_list!(hwtlas)
        old_hw_tlas = hwtlas.hw_tlas
        old_tri_gpu = hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu
        hwtlas.hw_tlas    = nothing
        hwtlas.hw_accel   = nothing
        hwtlas.tri_gpu    = nothing
        hwtlas.off_gpu    = nothing
        hwtlas.dirty            = false
        hwtlas.transforms_dirty = false
        hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
        unsafe_free!(old_hw_tlas)
        unsafe_free!(old_tri_gpu)
        unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    if hwtlas.dirty
        # Topology change: full rebuild (also handles the "first sync after
        # construct" case since the constructor leaves dirty=true).
        hw_tlas, hw_accel, tri_gpu, off_gpu, dropped_blases =
            rebuild_hw_tlas_from_batch!(hwtlas)

        old_hw_tlas = hwtlas.hw_tlas
        # tri_gpu / off_gpu get reused-in-place when sizes permit -- do NOT
        # release unless the rebuild returned a different object.
        old_tri_gpu = hwtlas.tri_gpu === tri_gpu ? nothing : hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu === off_gpu ? nothing : hwtlas.off_gpu

        hwtlas.hw_tlas   = hw_tlas
        hwtlas.hw_accel  = hw_accel
        hwtlas.tri_gpu   = tri_gpu
        hwtlas.off_gpu   = off_gpu
        hwtlas.dirty            = false
        hwtlas.transforms_dirty = false
        hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
        unsafe_free!(old_hw_tlas)
        unsafe_free!(old_tri_gpu)
        unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    # Transforms-only path: MODE_UPDATE_KHR refit.
    if hwtlas.transforms_dirty
        if hwtlas.hw_tlas === nothing || !hwtlas.hw_tlas.allow_update
            # Defensive: if the prior build wasn't refit-capable, fall back to
            # full rebuild.  Should not happen on normal use since
            # rebuild_hw_tlas_from_batch! always sets allow_update=true.
            hwtlas.dirty = true
            return Raycore.sync!(hwtlas)
        end
        combined, total_n = _concat_batch_instances!(hwtlas)
        as_build() do ctx
            Lava.refit_tlas!(ctx, hwtlas.hw_tlas, combined, total_n)
        end
        hwtlas.transforms_dirty = false
        # static_tlas wraps the same hw_tlas — reuse, just make sure it
        # exists (first-sync edge case).
        hwtlas.static_tlas === nothing && (hwtlas.static_tlas = HWAdaptedAccel(hwtlas))
        return hwtlas
    end

    # Both flags false but static_tlas was nothing — first sync on a fresh
    # HWTLAS with no batches yet.
    hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
    return hwtlas
end

# ============================================================================
# Trace dispatch
# ============================================================================

function trace_closest_hits!(results, rays, accel::HWAdaptedAccel, n;
                              cull_mask::UInt32 = UInt32(0xFF))
    trace_closest_hits!(results, rays, accel.hwtlas.hw_accel, n; cull_mask=cull_mask)
end

function trace_closest_hits_indirect!(results, rays, accel::HWAdaptedAccel, n_buf;
                                       cull_mask::UInt32 = UInt32(0xFF))
    trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf; cull_mask=cull_mask)
end

function batch_trace_indirect(results, rays, accel::HWAdaptedAccel, n_buf;
                               cull_mask::UInt32 = UInt32(0xFF))
    trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf; cull_mask=cull_mask)
    Tri = eltype(eltype(accel.hwtlas.blas_triangles))
    empty = Raycore.empty_triangle(Tri)
    return PrecomputedHitsAccel(results, accel.hwtlas.tri_gpu, accel.hwtlas.off_gpu, empty)
end

function set_custom_anyhit!(accel::HWAdaptedAccel, anyhit_fn, raygen_fn)
    hw = accel.hwtlas.hw_accel
    hw === nothing && error("HWTLAS not synced")
    set_anyhit_pipeline!(hw, anyhit_fn, raygen_fn)
end

# RT shader intrinsics forwarded from HWAdaptedAccel -> Lava intrinsics
rt_primitive_id(::HWAdaptedAccel)          = lava_rt_primitive_id()
rt_instance_custom_index(::HWAdaptedAccel) = lava_rt_instance_custom_index()
rt_instance_id(::HWAdaptedAccel)           = lava_rt_instance_id()
rt_launch_id_x(::HWAdaptedAccel)           = lava_rt_launch_id_x()
rt_global_invocation_id_x(::HWAdaptedAccel) = lava_global_invocation_id_x()
rt_ignore_intersection(::HWAdaptedAccel)   = lava_rt_ignore_intersection()
rt_terminate_ray(::HWAdaptedAccel)         = lava_rt_terminate_ray()
rt_payload_store!(::HWAdaptedAccel, val, slot) = lava_rt_payload_store_f32_at(val, slot)
rt_payload_load(::HWAdaptedAccel, slot)    = lava_rt_payload_load_f32_at(slot)

function rt_trace_ray!(::HWAdaptedAccel, flags, mask, sbt_offset, sbt_stride, miss_idx,
                       ox, oy, oz, tmin, dx, dy, dz, tmax)
    lava_rt_trace_ray(flags, mask, sbt_offset, sbt_stride, miss_idx,
                       ox, oy, oz, tmin, dx, dy, dz, tmax)
end

@propagate_inbounds function Raycore.closest_hit(accel::PrecomputedHitsAccel, ray)
    tid = lava_global_invocation_id_x() + UInt32(1)
    result = accel.results[tid]

    if result.hit == UInt32(0)
        return (false, accel.empty, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end

    # `accel.offsets` is keyed by gl_InstanceID (0-based).  `result.instance_id`
    # is that builtin; `result.instance_custom_index` carries the interface
    # override and is forwarded as the 5th return value.
    tri_idx = Int(accel.offsets[result.instance_id + UInt32(1)]) +
              Int(result.primitive_id) + 1
    tri = accel.triangles[tri_idx]

    w = 1f0 - result.bary_u - result.bary_v
    bary = SVector{3,Float32}(w, result.bary_u, result.bary_v)

    return (true, tri, result.t, bary, result.instance_custom_index)
end
