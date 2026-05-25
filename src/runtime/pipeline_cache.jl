# VkPipelineCache persistence.
#
# A single VkPipelineCache per device, seeded from disk and saved back so
# AMDVLK / RADV / etc. don't recompile SPIR-V → ISA every session. Lives in the
# same scratchspace as the SPIR-V cache (`lava_disk_cache_dir()`) for cohesion.
#
# Safety: the spec's pInitialData header carries vendor/device/driver IDs and
# the driver rejects any mismatch — so loading stale or wrong-driver data is a
# harmless empty start. We just pass the raw bytes through.

"""
Disk path for this device's pipeline-cache blob. Co-located with the SPIR-V
disk cache; one file per (device, driver version) pair.
"""
function lava_pipeline_cache_path(device_name::AbstractString, driver_version::AbstractString)
    sanitized_dev = replace(device_name, r"[^a-zA-Z0-9]+" => "_")
    sanitized_drv = replace(driver_version, r"[^a-zA-Z0-9]+" => "_")
    return joinpath(lava_disk_cache_dir(),
                    "vk_pipeline_cache_$(sanitized_dev)_drv$(sanitized_drv).bin")
end

"""
Read the saved pipeline-cache blob from disk; return empty bytes on miss/error.
The driver will validate the header at create-time and reject bad data, so
returning whatever's on disk is safe.
"""
function load_pipeline_cache_data(path::String)
    isfile(path) || return UInt8[]
    try
        return read(path)
    catch ex
        @debug "Lava: pipeline cache read failed" path exception=ex
        return UInt8[]
    end
end

"""
Create a `VkPipelineCache` for `device`, seeded from `path` if it exists.
Vulkan.jl's `PipelineCacheCreateInfo(initial_data::Ptr{Cvoid}; initial_data_size)`
takes a raw pointer — the bytes vector is GC-pinned for the duration of the call.
"""
function create_lava_pipeline_cache(device::Vulkan.Device, path::String)
    initial = load_pipeline_cache_data(path)
    return GC.@preserve initial begin
        ptr = isempty(initial) ? C_NULL : Ptr{Cvoid}(pointer(initial))
        ci = Vulkan.PipelineCacheCreateInfo(ptr; initial_data_size=UInt64(length(initial)))
        Vulkan.PipelineCache(device, ci)
    end
end

"""
Snapshot `ctx.pipeline_cache` to its on-disk path. Atomic write via mktemp+rename
so concurrent processes can't observe a partial blob. No-op if the device was
lost (handles are invalid).
"""
function save_pipeline_cache!(ctx)
    ctx.device_lost && return nothing
    isdefined(ctx, :pipeline_cache) || return nothing
    try
        # Vulkan.jl returns (size::UInt, ptr::Ptr{Cvoid}) where ptr was Libc.malloc'd
        # on our behalf; we must `Libc.free` after copying out.
        size, ptr = unwrap(
            Vulkan.get_pipeline_cache_data(ctx.device, ctx.pipeline_cache))
        try
            size == 0 && return nothing
            bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), Int(size); own=false)
            path = lava_pipeline_cache_path(ctx.device_name, ctx.driver_version)
            mkpath(dirname(path))
            tmppath, io = mktemp(dirname(path); cleanup=false)
            try
                write(io, bytes)
            finally
                close(io)
            end
            mv(tmppath, path; force=true)
        finally
            Libc.free(ptr)
        end
    catch ex
        @debug "Lava: pipeline cache save failed" exception=ex
    end
    return nothing
end

# atexit hook: persist on Julia shutdown. Single-shot (registered once).
const _PIPELINE_CACHE_ATEXIT_REGISTERED = Ref(false)
function _register_pipeline_cache_atexit!()
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] && return
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] = true
    atexit() do
        ctx = VK_CONTEXT_REF[]
        ctx === nothing && return
        save_pipeline_cache!(ctx)
    end
end
