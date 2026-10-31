"""
Frozen kernel cache: SPIR-V on disk under a key that never changes by itself.

The existing disk cache (`lava_disk_cache_*`) keys on a `Core.MethodInstance`,
which means obtaining the key already costs what the cache was meant to save:
`get_compiled_kernel_and_pipeline` calls `GPUCompiler.methodinstance(typeof(f),
tt)`, and that infers the kernel. Measured on SAM 2's encoder, a cold call spends
70 s in Julia's compiler with every one of its 99 kernels already on disk as
SPIR-V. The bytes were there; getting permission to use them was the expensive
part.

So this cache is keyed on nothing but `typeof(f)`, `tt` and the workgroup size —
all of them types the caller already holds, none of them requiring inference —
plus a **version string the programmer sets by hand**. There is no hashing of
source, no dependency scan, no staleness check. An entry written under `v3` is
read back under `v3` forever. Change a kernel and the cache is wrong until you
bump the version; that is the trade, and it is deliberate, because every
automatic invalidation scheme so far has cost more than it saved.

Two levels, because SPIR-V is portable and machine code is not:

  * `<key>.spirv` — the compiled module. Valid on any device, so it survives a
    driver update or a different GPU.
  * the driver's own `VkPipelineCache` (see `pipeline_cache.jl`) — SPIR-V to
    ISA. Device- and driver-specific, header-validated before the driver is
    allowed to touch it, and simply absent after an update, at which point the
    driver recompiles from level 1.

The module in the key is the one the kernel is **defined** in, not the one that
launched it. A broadcast kernel over `LavaArray` belongs to Lava whoever calls
it, so DNNKernels and VideoEditor share Lava's entry instead of each writing their
own copy under their own name.

Ray-tracing stages go through `frozen_rt_load` / `frozen_rt_store`, keyed the
same way plus `stage` and `payload_type` — one function is compiled once per
stage, and the payload changes the emitted module. They are the expensive half
of an hw_accel=true scene: crown's startup is ~1063 s cold against ~123 s
frozen, for an unchanged 7.5 s frame.

Note that only level 1 applies to them. Their ISA is already covered by the
device-wide `ctx.pipeline_cache`, which `create_rt_pipeline` passes and which
persists across sessions, so there is no per-stage `.bin`. That device-wide
cache is also why the RT compile looked cached when it was not: it caches the
driver's SPIR-V -> ISA step, and everything above it — GPUCompiler, the LLVM
passes, structurize, the emitter — ran on every start regardless.
"""

"""
    FROZEN_VERSION[]

The cache generation. Empty disables the frozen cache entirely.

Set by `@compile_workload`, and the only thing that invalidates an entry. Bump
it while developing kernels; leave it alone otherwise.
"""
const FROZEN_VERSION = Ref("")

"""
    cache_io_error(ex) -> Bool

Whether `ex` is a failure of the cache MEDIUM rather than of this library.

A cache entry that is missing, truncated, or written by an older version is a
recompile — never a failed session. Everything else is a bug here, and a bare
`catch` that turns a `MethodError` in the deserializer into "cache miss" is how
a cache silently stops working: every launch pays a recompile, every run is
correct, and nothing ever says why.
"""
# No `Serialization.SerializationError` — the stdlib defines no such type, and
# naming it made this function throw on its first call. `deserialize` on a
# damaged or version-skewed entry surfaces as one of these instead: truncation is
# `EOFError`, a type that no longer exists is `UndefVarError`, a changed field
# layout is `TypeError` or `MethodError`.
cache_io_error(ex) = ex isa Union{SystemError, Base.IOError, EOFError,
                                  ArgumentError, UndefVarError, TypeError,
                                  MethodError}

"""When true, kernels compiled the slow way are written to the frozen cache."""
const FROZEN_RECORDING = Ref(false)

# Session-local, keyed by the same types the disk key is derived from, so a
# repeat launch costs a tuple hash rather than building a filename.

"""
The per-kernel `VkPipelineCache` the pipeline for the kernel being recorded was
built with, so `frozen_store` can snapshot exactly that kernel's ISA rather than
the device-wide accumulation.
"""

const FROZEN_HITS = Ref(0)
const FROZEN_STORES = Ref(0)
const FROZEN_MISSES = Ref(0)

"""
    ctx.diag.frozen_log_misses

Print the key of every kernel that falls through to a compile.

A miss is invisible otherwise — the answer is still right, it just cost seconds
— so this is how a workload is debugged into covering everything. What it prints
is exactly the filename that *would* have been read, which is directly
comparable against a directory listing.
"""

"""Whether to log keys, from the env so precompilation can be traced too."""
frozen_logging(ctx::VkContext) =
    ctx.diag.frozen_log_misses || get(ENV, "LAVA_FROZEN_LOG", "") == "1"
# The RT path compiles before a device is chosen — `lava_compile_rt_shader` takes
# no context — so there is genuinely none to ask. Fall back to the current one if
# it exists, and to the environment variable if it does not.
frozen_logging() = let c = VK_CONTEXT_REF[]
    (c !== nothing && c.diag.frozen_log_misses) || get(ENV, "LAVA_FROZEN_LOG", "") == "1"
end

"""Where frozen entries live: one directory, shared by every package."""
function frozen_cache_dir()
    dir = FROZEN_CACHE_DIR[]
    isempty(dir) || return dir
    dir = joinpath(first(Base.DEPOT_PATH), "scratchspaces", "lava_frozen_kernels")
    FROZEN_CACHE_DIR[] = dir
    return dir
end
const FROZEN_CACHE_DIR = Ref("")

"""
    typestring(T) -> String

`T` printed with every name fully qualified, whatever is in scope.

`string(T)` abbreviates names that are visible from the *current module*, and
which module that is differs between precompilation and the session that loads
the result: the same kernel signature printed
`Lava.LavaDeviceArray{Int64,1}` while precompiling and `LavaDeviceArray{Int64,1}`
at run time. Two spellings, two digests, and every entry the workload froze
missed — 168 stored, 168 missed, with the right bytes sitting on disk the whole
time. `:module => nothing` turns the abbreviation off.
"""
typestring(@nospecialize(T)) = sprint(show, T; context = :module => nothing)

"""
The layout of what an entry *is*, folded into every key.

A frozen entry is one serialized `LavaGPUKernel`, and the only type nested in one
is `PushConstantInfo` — so these two describe the format completely. `Serialization`
reconstructs a struct field by field against the *current* definition, so a field
added to either one makes every entry ever written unreadable. It is caught
(`cache_io_error`) and each kernel recompiles, which is correct but silent enough
to look like the cache simply stopped helping.

Deriving it beats a hand-set number because the person who adds the field is not
the person who remembers the cache exists: `PushConstantInfo` gained `arg_offsets`
on 2026-08-04 and poisoned 456 JuliaVision entries and 58 RT entries at once, in a
repo where every consumer would have had to bump its own version string.

Hashed from the printed form, not the types: `hash` of a type is object identity
and differs between sessions.
"""
const FROZEN_LAYOUT = hash(string(fieldnames(LavaGPUKernel), fieldtypes(LavaGPUKernel),
                                  fieldnames(PushConstantInfo), fieldtypes(PushConstantInfo)))

"""
    frozen_key(f, tt, workgroup_size) -> String

`<module>_<kernel>_<signature digest>_v<version>`.

The digest is over `typestring(tt)`, the workgroup size and `FROZEN_LAYOUT` — a
*string*, because `hash` of a type is object identity and changes between
sessions, while its printed form does not.

**It also covers the build id of the module defining the kernel, and of `Lava`.**
The two additions are orthogonal and both are needed: `FROZEN_LAYOUT` invalidates
when the *stored record's* shape changes, this when the *kernel body* may have.
Without the second the key describes a signature and not a body, so editing a
kernel without bumping the version leaves the OLD SPIR-V reachable — and the
symptom is not a wrong number, it is `device was lost ... a dispatch wrote out of
bounds`, reported from whatever submit is in flight and nowhere near the edit.
That cost an hour to diagnose once and then recurred immediately: nine hundred
entries frozen during a bisecting session went stale the moment the code was
restored.

`Base.module_build_id` changes exactly when a module is recompiled, which is
exactly when its kernels may have changed, so this is the invalidation signal
already sitting in the runtime. `Lava`'s own id is included because a change in
the emitter alters the SPIR-V of *every* kernel, whoever defined it.

The cost is a re-freeze after any recompilation of `Lava` or the defining
package. In development that is precisely what is wanted; for an installed
package the id comes from the `.ji` and is stable across processes.

So a changed body under an unchanged signature **is** now detected — this
docstring used to say the opposite, and `KERNELS_VERSION` remains only for the
deliberate, cross-package generation bump rather than as the sole guard.
"""
function frozen_key(@nospecialize(f), @nospecialize(tt), workgroup_size)
    F = typeof(f)
    # `parentmodule(F)` is where the `@kernel` was written; `Lava` is where the
    # SPIR-V for it is produced. A change in either invalidates the entry.
    bids = hash(Base.module_build_id(parentmodule(F)),
                hash(Base.module_build_id(@__MODULE__)))
    h = hash(typestring(tt), hash(typestring(F),
             hash(workgroup_size, hash(FROZEN_LAYOUT, bids))))
    sanitize(s) = replace(s, r"[^A-Za-z0-9_]" => "_")
    mod = sanitize(string(parentmodule(F)))
    fn = sanitize(string(nameof(F)))
    # Filesystems cap names near 255 bytes and a mangled kernel name can run
    # far past that on its own; the digest is what makes the entry unique, the
    # rest is for a human reading the directory.
    length(fn) > 72 && (fn = fn[1:72])
    return string(mod, '_', fn, '_', string(h; base = 16, pad = 16),
                  "_v", FROZEN_VERSION[])
end

frozen_path(key::AbstractString) = joinpath(frozen_cache_dir(), key * ".spirv")
frozen_binpath(key::AbstractString) = joinpath(frozen_cache_dir(), key * ".bin")

"""
    frozen_pipeline_cache(ctx, key) -> VkPipelineCache | nothing

A `VkPipelineCache` seeded with this kernel's own `<key>.bin`, or `nothing` when
there is none for this device.

This is level 2, per kernel: SPIR-V is portable, the ISA the driver derives from
it is not. The blob is validated against vendor ID, device ID and cache UUID
before `vkCreatePipelineCache` sees a byte of it — a foreign or truncated blob
is a documented way to crash *inside* the driver, and a crash there takes the
process with it before any Julia `try` can run. After a driver update the UUID
changes, every `.bin` stops matching, and the kernel is rebuilt from its
`.spirv`, which is exactly the fallback the two levels exist for.
"""
function frozen_pipeline_cache(ctx::VkContext, key::AbstractString)
    path = frozen_binpath(key)
    isfile(path) || return nothing
    bytes = try
        read(path)
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen pipeline blob unreadable; the driver will recompile its ISA" path exception = ex maxlog = 1
        return nothing
    end
    pipeline_cache_compatible(bytes, ctx.physical_device) || return nothing
    try
        return GC.@preserve bytes begin
            ci = Vulkan.PipelineCacheCreateInfo(Ptr{Cvoid}(pointer(bytes));
                                                initial_data_size = UInt64(length(bytes)))
            Vulkan.PipelineCache(ctx.device, ci)
        end
    catch ex
        # Passed the header check and still refused: keep the session, drop the
        # blob so the next run does not retry it.
        @warn "Lava: frozen pipeline blob rejected; rebuilding from SPIR-V" path exception = ex maxlog = 1
        rm(path; force = true)
        return nothing
    end
end

"""Snapshot `pcache` to `<key>.bin`, atomically. Best effort."""
function frozen_store_bin(ctx::VkContext, key::AbstractString, pcache)
    try
        size, ptr = unwrap(Vulkan.get_pipeline_cache_data(ctx.device, pcache))
        try
            size == 0 && return nothing
            bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), Int(size); own = false)
            dir = frozen_cache_dir(); mkpath(dir)
            tmppath, io = mktemp(dir; cleanup = false)
            write(io, bytes); close(io)
            mv(tmppath, frozen_binpath(key); force = true)
        finally
            Libc.free(ptr)
        end
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen pipeline blob store failed; level-2 cache will stay cold" key exception = ex maxlog = 1
    end
    return nothing
end

"""
    frozen_load(ctx, f, tt, workgroup_size) -> LavaLinkedKernel | nothing

The linked kernel for this signature, from memory or from disk, without ever
asking Julia to infer anything.
"""
function frozen_load(ctx::VkContext, @nospecialize(f), @nospecialize(tt), workgroup_size)
    isempty(FROZEN_VERSION[]) && return nothing
    memkey = (typeof(f), tt, workgroup_size)
    hit = get(ctx.caches.frozen_mem, memkey, nothing)
    hit === nothing || return hit
    key = frozen_key(f, tt, workgroup_size)
    path = frozen_path(key)
    if !isfile(path)
        frozen_logging(ctx) && println("frozen MISS: ", key, " || ", typestring(tt))
        return nothing
    end
    compiled = try
        open(Serialization.deserialize, path)::LavaGPUKernel
    catch ex
        # A damaged entry costs a recompile, never the session — but it says so.
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen cache entry unreadable; recompiling this kernel" path exception = ex maxlog = 1
        return nothing
    end
    # Level 2: hand the pipeline this kernel's own driver blob when there is a
    # matching one, so the driver reuses its ISA instead of recompiling.
    linked = link_kernel(ctx, compiled; pipeline_cache = frozen_pipeline_cache(ctx, key))
    ctx.caches.frozen_mem[memkey] = linked
    FROZEN_HITS[] += 1
    return linked
end

"""
    frozen_store(ctx, f, tt, workgroup_size, compiled)

Write a compiled kernel under its frozen key. Only while recording.
"""
function frozen_store(ctx::VkContext, @nospecialize(f), @nospecialize(tt), workgroup_size,
                      compiled::LavaGPUKernel)
    (FROZEN_RECORDING[] && !isempty(FROZEN_VERSION[])) || return nothing
    dir = frozen_cache_dir()
    mkpath(dir)
    key = frozen_key(f, tt, workgroup_size)
    path = frozen_path(key)
    # The LLVM IR string is session-specific and large; the frozen entry is the
    # SPIR-V and what it takes to build a pipeline from it, nothing else.
    entry = LavaGPUKernel(compiled.spirv_bytes, compiled.entry_name,
                          compiled.workgroup_size, compiled.push_info, "",
                          compiled.enable_ray_query)
    try
        tmppath, io = mktemp(dir; cleanup = false)
        Serialization.serialize(io, entry)
        close(io)
        mv(tmppath, path; force = true)          # atomic: no half-written entry
        FROZEN_STORES[] += 1
        # …and the driver's ISA for it, from a cache holding only this kernel.
        # Recording builds the pipeline through `frozen_link_recording`, which
        # leaves the per-kernel cache in `ctx.caches.frozen_last_pcache`.
        let pc = ctx.caches.frozen_last_pcache
            pc === nothing || frozen_store_bin(ctx, key, pc)
        end
        frozen_logging(ctx) &&
            println("frozen STORE: ", basename(path)[1:end-6], " || ", typestring(tt))
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen cache store failed; this kernel will recompile next session" path exception = ex maxlog = 1
    end
    return nothing
end

"""
    frozen_stats() -> NamedTuple

Hits, stores and misses since the counters were last reset, plus how many
entries are on disk for the current version. What a test asserts on.
"""
function frozen_stats()
    dir = frozen_cache_dir()
    suffix = "_v" * FROZEN_VERSION[] * ".spirv"
    ondisk = isdir(dir) ? count(f -> endswith(f, suffix), readdir(dir)) : 0
    (hits = FROZEN_HITS[], stores = FROZEN_STORES[], misses = FROZEN_MISSES[],
     ondisk = ondisk, version = FROZEN_VERSION[])
end

"""Reset the hit/store/miss counters (not the cache itself)."""
frozen_reset_stats!() = (FROZEN_HITS[] = 0; FROZEN_STORES[] = 0; FROZEN_MISSES[] = 0; nothing)

"""
    frozen_clear!(; version = FROZEN_VERSION[])

Delete every on-disk entry for `version`, and the session's memory of them.
The one supported way to invalidate, short of bumping the version.
"""
function frozen_clear!(; version::AbstractString = FROZEN_VERSION[],
                        ctx::VkContext = vk_context())
    empty!(ctx.caches.frozen_mem)
    empty!(FROZEN_RT_MEM)
    dir = frozen_cache_dir()
    isdir(dir) || return 0
    suffix = "_v" * version * ".spirv"
    n = 0
    for f in readdir(dir)
        endswith(f, suffix) || continue
        rm(joinpath(dir, f); force = true)
        # The pipeline blob beside it, or it is orphaned forever: nothing else
        # names a `.bin`, so one left behind is disk that is never reclaimed and
        # never read. Measured: 56 of them had accumulated this way.
        rm(joinpath(dir, replace(f, r"\.spirv$" => ".bin")); force = true)
        n += 1
    end
    return n
end

"""
    frozen_prune!(; keep = 2000, ctx = vk_context()) -> Int

Delete all but the `keep` most recently modified frozen entries. Returns how many
were removed.

**Orphans are the normal outcome of editing kernels, not a malfunction.** An
entry is keyed by signature, workgroup, version and the build id of the defining
module (see [`frozen_key`](@ref)); recompiling that module makes every one of its
entries unreachable but does not delete them. A development session that edits
kernels a few times leaves hundreds behind — one measured session accumulated
**6066**.

They are harmless except for disk, because an unreachable entry is never loaded.
This exists so the directory can be bounded when that matters, and is deliberately
**not** called automatically: deleting a cache another package is about to hit
turns a hit into a recompile, and that choice belongs to whoever knows the
machine.

`Lava.frozen_clear!(; version)` is the other tool — it removes one generation
outright, which is what a deliberate `KERNELS_VERSION` bump obsoletes.
"""
function frozen_prune!(; keep::Int = 2000, ctx::VkContext = vk_context())
    empty!(ctx.caches.frozen_mem)
    empty!(FROZEN_RT_MEM)
    dir = frozen_cache_dir()
    isdir(dir) || return 0
    spirv = filter(f -> endswith(f, ".spirv"), readdir(dir))
    length(spirv) <= keep && return 0
    # Oldest first, by the entry's own mtime.
    sort!(spirv; by = f -> mtime(joinpath(dir, f)))
    n = 0
    for f in spirv[1:(length(spirv) - keep)]
        rm(joinpath(dir, f); force = true)
        rm(joinpath(dir, replace(f, r"\.spirv$" => ".bin")); force = true)
        n += 1
    end
    # Sweep `.bin` blobs whose `.spirv` is already gone. `frozen_clear!` used to
    # leave these behind, and nothing else names them.
    live = Set(readdir(dir))
    for f in readdir(dir)
        endswith(f, ".bin") || continue
        replace(f, r"\.bin$" => ".spirv") in live && continue
        rm(joinpath(dir, f); force = true)
        n += 1
    end
    return n
end

# RT entries key on (F, tt, stage, payload, push_size) — five fields, so they
# cannot share `frozen_mem`, whose key type is fixed at three.
#
# **Module-level on purpose, unlike `frozen_mem`.** A `LavaRTShader` is SPIR-V
# bytes, a stage, push-constant info and IR text — no `VkPipeline`, no device
# handle of any kind — and the key is device-independent. So this is a
# compile-result memo, and sharing it across devices is correct rather than a
# §8 defect. `frozen_mem` holds `LavaLinkedKernel`s, which own a pipeline; that
# is why only that one moved onto the context.
const FROZEN_RT_MEM = Dict{Tuple{DataType, DataType, Symbol, Symbol, Int}, Any}()

# ── Ray-tracing shaders ──
#
# The compute path freezes in `launch.jl`; the RT path had no equivalent, so an
# hw_accel=true scene rebuilt every raygen/miss/chit from scratch in every
# session. That is the expensive half: crown spends ~610 s in Lava-side compile
# (of which `run_structurize_cfg_pipeline!` alone is 57.8 s and 54.6 s on the two
# fat chits) against ~7.8 s of actual rendering.
#
# The device-wide `ctx.pipeline_cache` does NOT cover this. It caches the
# driver's SPIR-V → ISA step, which happens *after* everything above; the
# SPIR-V has to exist before the driver is asked for anything.
#
# Keyed like the compute entries plus `stage` and `payload_type`, since one
# function is compiled once per stage and payloads change the emitted module.

function frozen_rt_key(@nospecialize(f), @nospecialize(tt), stage::Symbol,
                       payload_type::Symbol, push_constant_size::Integer)
    F = typeof(f)
    h = hash(typestring(tt),
             hash(typestring(F),
                  hash(stage, hash(payload_type, hash(push_constant_size)))))
    sanitize(s) = replace(s, r"[^A-Za-z0-9_]" => "_")
    mod = sanitize(string(parentmodule(F)))
    fn = sanitize(string(nameof(F)))
    length(fn) > 64 && (fn = fn[1:64])
    return string(mod, '_', fn, "_rt", sanitize(string(stage)), '_',
                  string(h; base = 16, pad = 16), "_v", FROZEN_VERSION[])
end

"""
    frozen_rt_load(f, tt, stage, payload_type, push_constant_size) -> LavaRTShader | nothing

The frozen SPIR-V for an RT stage, without running the compiler.
"""
function frozen_rt_load(@nospecialize(f), @nospecialize(tt), stage::Symbol,
                        payload_type::Symbol, push_constant_size::Integer)
    isempty(FROZEN_VERSION[]) && return nothing
    memkey = (typeof(f), tt, stage, payload_type, Int(push_constant_size))
    hit = get(FROZEN_RT_MEM, memkey, nothing)
    hit === nothing || return hit
    key = frozen_rt_key(f, tt, stage, payload_type, push_constant_size)
    path = frozen_path(key)
    if !isfile(path)
        frozen_logging() && println("frozen RT MISS: ", key, " || ", typestring(tt))
        return nothing
    end
    shader = try
        open(Serialization.deserialize, path)::LavaRTShader
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen RT entry unreadable; recompiling this stage" path exception = ex maxlog = 1
        return nothing
    end
    FROZEN_RT_MEM[memkey] = shader
    FROZEN_HITS[] += 1
    return shader
end

"""
    frozen_rt_store(f, tt, stage, payload_type, push_constant_size, shader)

Write an RT stage's SPIR-V under its frozen key. Only while recording.
"""
function frozen_rt_store(@nospecialize(f), @nospecialize(tt), stage::Symbol,
                         payload_type::Symbol, push_constant_size::Integer,
                         shader::LavaRTShader)
    (FROZEN_RECORDING[] && !isempty(FROZEN_VERSION[])) || return nothing
    dir = frozen_cache_dir()
    mkpath(dir)
    key = frozen_rt_key(f, tt, stage, payload_type, push_constant_size)
    path = frozen_path(key)
    # Drop the LLVM IR: session-specific, and by far the largest field.
    entry = LavaRTShader(shader.spirv_bytes, shader.stage, shader.push_info, "")
    try
        tmppath, io = mktemp(dir; cleanup = false)
        Serialization.serialize(io, entry)
        close(io)
        mv(tmppath, path; force = true)
        FROZEN_STORES[] += 1
        frozen_logging() &&
            println("frozen RT STORE: ", basename(path)[1:end-6], " || ", typestring(tt))
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen RT store failed; this stage will recompile next session" path exception = ex maxlog = 1
    end
    return nothing
end
