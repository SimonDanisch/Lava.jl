# The frozen cache, COMPILER half: what a kernel's identity is, where it is
# written, and the ray-tracing entries.
#
# Nothing in this file names a `VkContext`. That is the point of the split — the
# compiler consults the cache before it compiles, and used to reach into the
# Vulkan runtime to do it. The device half is `runtime/frozen_pipeline.jl`; see
# its header for what stayed there and why.
#
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


"""
Modules WITHOUT a package identity that may use the frozen cache anyway.

[`frozen_eligible`](@ref) refuses `Main` because its build id is constant across
sessions, so an edited kernel is served stale SPIR-V — the measurement is over
there. A test does not have that problem: it writes under a version string it
mints fresh on every run and clears afterwards, so nothing it stores outlives it.
Without this the store path is simply unreachable from a test file, since every
kernel defined at the top level of one belongs to `Main` — which is how the
frozen-cache tests came to assert `stores >= 2` against a constant 0.

Names MODULES rather than being a boolean on purpose: opting one test file in
must not opt in every script in the session.

    push!(Lava.FROZEN_UNPACKAGED, @__MODULE__)
"""
const FROZEN_UNPACKAGED = Set{Module}()

"""
    frozen_eligible(f) -> Bool

Is this kernel's defining module one whose build id actually MOVES when its
source changes?

`frozen_key` mixes in `Base.module_build_id(parentmodule(typeof(f)))`, and that
is the whole reason a changed kernel body produces a different key. It only holds
for modules loaded from a package image. `Main` — a script, `include`, or the
REPL — keeps ONE build id for the entire session AND across sessions, so a
redefined kernel there hits the previous entry and is served stale SPIR-V.

Measured, not assumed: a Main-defined kernel edited from `2i` to `3i` came back
with the old `2i` result, `build_id(Main)` byte-identical across both runs. That
is why the cache cannot simply be switched on globally.

Packages have a UUID and Main does not, which is exactly the distinction needed.

See [`FROZEN_UNPACKAGED`](@ref) for the one deliberate way past this.
"""
function frozen_eligible(@nospecialize(f))
    m = Base.moduleroot(parentmodule(typeof(f)))
    m in FROZEN_UNPACKAGED && return true
    return Base.PkgId(m).uuid !== nothing
end


# ── Automatic bounding ──────────────────────────────────────────────────────
#
# `frozen_prune!` is documented as deliberately manual, and that was right while
# the cache was opt-in. It is on by default now, so "grows forever" became the
# DEFAULT behaviour: every source edit mints a new build id and therefore a fresh
# set of entries, and nothing reclaims the old ones. Measured here mid-session:
# 1541 entries / 1.6 GB, from one project.
#
# cuTile bounds its disk cache the same way (`DiskCache`: prune at a 90 % high
# water mark down to 75 %). This is that, adapted to plain files: check ONCE per
# session, on the first store, and only when over the cap. One readdir+stat pass
# against a compile that costs seconds is free.
#
# Recency is the right proxy for usefulness: a live entry is re-read (and its
# `.bin` rewritten) whenever it is hit, while an entry orphaned by an edit is
# never touched again.
const FROZEN_PRUNED = Ref(false)

"""Byte budget for the frozen cache. `LAVA_FROZEN_MAX_BYTES` overrides; 0 disables."""
frozen_max_bytes() = parse(Int, get(ENV, "LAVA_FROZEN_MAX_BYTES", string(2 * 1024^3)))


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

"""
Whether the bound device asked for miss logging. Pushed by `bind_context!`, the
same way `targetfeatures()` is, and reset when the device is released.

It used to be read off `ctx.diag.frozen_log_misses` through `VK_CONTEXT_REF[]`,
which meant this file — the half the COMPILER consults before it compiles — named
a `VkContext` for a log line. The flag is a boolean, so the runtime pushes the
boolean.
"""
const FROZEN_LOG_MISSES = Ref(false)

"""
Whether to log keys. The env var is read every time so precompilation can be
traced without a device, and the RT path genuinely has none — it compiles before
a device is chosen, because `lava_compile_rt_shader` takes no context.
"""
frozen_logging() = FROZEN_LOG_MISSES[] || get(ENV, "LAVA_FROZEN_LOG", "") == "1"

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
    # Same build-id mix as `frozen_key`, and for the same reason: without it a
    # changed shader body under an unchanged signature keeps its key and is
    # served stale SPIR-V. This key used to hash only types, stage, payload and
    # push-constant size, which was survivable while the cache was opt-in and is
    # not now that it is on by default.
    bids = hash(Base.module_build_id(parentmodule(F)),
                hash(Base.module_build_id(@__MODULE__)))
    h = hash(typestring(tt),
             hash(typestring(F),
                  hash(stage, hash(payload_type,
                       hash(push_constant_size, hash(FROZEN_LAYOUT, bids))))))
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
    frozen_eligible(f) || return nothing
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
    frozen_eligible(f) || return nothing
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

"""
    frozen_rt_clear!(; version = FROZEN_VERSION[]) -> Int

Drop this version's frozen entries: the ray-tracing memo, and the files on disk.
Returns how many entries were removed.

The COMPILER's half of clearing the cache, and the half that needs no device —
the memo holds `LavaRTShader`s, which are SPIR-V bytes and a stage, and the files
are what this package wrote. `Mantle.frozen_clear!` calls this and additionally
empties the per-device memo of linked kernels, which own a `VkPipeline` and are
therefore its business.

Split for the reason everything else here was: an emitter test that needs to bust
the memo between two compiles should not have to have a device.
"""
function frozen_rt_clear!(; version::AbstractString = FROZEN_VERSION[])
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
