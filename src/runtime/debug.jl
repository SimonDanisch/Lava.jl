# Debug / validation knobs.
#
# These functions toggle Vulkan validation + GPU-Assisted Validation at runtime
# and self-test that the layer is actually firing. They exist because:
#   1. Env vars (LAVA_VALIDATION, LAVA_GPU_AV) only take effect at Vulkan instance
#      creation. To turn validation on/off in an already-running session we have
#      to reset the device.
#   2. GPU-AV's BDA OOB tracking is per-VkBuffer, but Lava's pool puts many
#      LavaArrays into one shared 64 MiB VkBuffer. With the pool on, GPU-AV
#      cannot see sub-pool overruns. enable_gpu_av(pool_disabled=true) switches
#      to one-VkBuffer-per-LavaArray for finer bounds checking.
#   3. "GPU-AV is enabled" and "GPU-AV is actually catching errors" are not the
#      same thing on every driver. verify_gpu_av runs a known-OOB write and
#      asserts the layer fires, so a passing call is proof end-to-end.

"""
    Lava.enable_gpu_av(; pool_disabled=false)

Turn on the Vulkan validation layer + GPU-Assisted Validation and reset the
Vulkan device so the new settings take effect.

`pool_disabled=false` (default) keeps the buffer pool on — GPU-AV will catch
writes past the entire pool block (~64 MiB) but is *blind to sub-pool overruns*
because it sees only one VkBuffer per pool block. This is the recommended
mode for general validation.

Set `pool_disabled=true` to also force every `LavaArray` onto its own
`VkBuffer`, so sub-pool BDA out-of-bounds are caught. This is debug-only:
allocation is significantly slower, and the path is less well-tested than the
pooled one (the host-upload + flush-after-error paths may misbehave).

The validation layer adds 10-100× shader execution overhead. Pair with
`verify_gpu_av()` after this call to prove the layer is actually firing on
the current driver.
"""
function enable_gpu_av(; pool_disabled::Bool=false)
    ENV["LAVA_VALIDATION"] = "1"
    ENV["LAVA_GPU_AV"]     = "1"
    POOL_DISABLED[] = pool_disabled
    # vk_reset_device! tears down the old instance and lazily creates a new one
    # via vk_context(), which re-reads the env vars above.
    vk_reset_device!()
    ctx = vk_context()
    if !ctx.gpu_assisted
        @warn "Lava.enable_gpu_av: layer accepted but gpu_assisted flag is false — VK_EXT_validation_features may be unavailable on this loader/driver"
    end
    @info "Lava: GPU-AV enabled" device=ctx.device_name pool_disabled validation=true gpu_assisted=ctx.gpu_assisted
    return nothing
end

"""
    Lava.disable_gpu_av()

Turn off validation + GPU-AV, re-enable the buffer pool, and reset the
Vulkan device.
"""
function disable_gpu_av()
    ENV["LAVA_VALIDATION"] = "0"
    ENV["LAVA_GPU_AV"]     = "0"
    ENV["LAVA_SYNC_VAL"]   = "0"
    ENV["LAVA_BEST"]       = "0"
    POOL_DISABLED[] = false
    vk_reset_device!()
    @info "Lava: validation + GPU-AV disabled, pool re-enabled" device=vk_context().device_name
    return nothing
end

"""
    Lava.verify_gpu_av(; timeout=10.0) -> Bool

Run a known out-of-bounds device write and assert that the GPU-AV layer
reports it via its debug-utils callback. Returns `true` on success; throws
`LavaError` if the layer did not catch the OOB within `timeout` seconds.

Uses a write at index `100_000_000_000` (~400 GB past the array start), so
the destination address is past every possible underlying `VkBuffer` —
this proves the layer is alive whether `POOL_DISABLED` is true or false.

Implementation: the flush after a GPU-AV-caught dispatch can hang in cleanup
on some drivers (observed on AMDVLK Windows), so the flush is spawned on a
worker task and the main task polls `VALIDATION_MESSAGES` until either the
expected message appears or the timeout expires. On success, the function
calls `vk_reset_device!` to clear the half-flushed batch state.
"""
function verify_gpu_av(; timeout::Float64=60.0)
    ctx = vk_context()
    if !ctx.gpu_assisted
        throw(LavaError("verify_gpu_av",
            "GPU-AV is not enabled (gpu_assisted=false)",
            "Call Lava.enable_gpu_av() first."))
    end
    arr = LavaArray{Int32,1}(undef, (4,))
    empty!(VALIDATION_MESSAGES)
    @kernel inbounds = true function _lava_gpuav_check_kernel!(out, bad_idx::Int)
        i = @index(Global)
        if i == 1
            out[bad_idx] = Int32(0xBAD)
        end
    end
    k = _lava_gpuav_check_kernel!(LavaBackend(), 1)
    Base.invokelatest(k, arr, 100_000_000_000; ndrange=4)
    # Submit on main thread. The validation callback fires asynchronously when
    # the driver processes the submission — but that requires the host to be
    # waiting on the semaphore (to pump the driver event loop). Lava's flush!
    # does an infinite wait (command.jl:785), and on AMDVLK Windows the
    # semaphore never signals after a GPU-AV-caught fault (vkWaitSemaphores
    # ignores its finite timeout). So spawn the wait on a worker thread — it
    # will hang there indefinitely, but its presence is enough to pump the
    # driver, which fires our callback that pushes into VALIDATION_MESSAGES.
    # Main thread polls the buffer; on success we call vk_reset_device! to
    # tear down the hung worker.
    bq = vk_context().default_bq
    submit!(bq)
    if !isempty(bq.in_flight)
        target = maximum(b.signal_value for b in bq.in_flight)
        Threads.@spawn try
            Vulkan.wait_semaphores(bq.device,
                Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [target]),
                typemax(UInt64))
        catch
        end
    end
    caught_msg = ""
    deadline = time() + timeout
    while time() < deadline
        for m in VALIDATION_MESSAGES
            if occursin("Out of bounds access", m) || occursin("device address", m)
                caught_msg = m
                break
            end
        end
        isempty(caught_msg) || break
        sleep(0.1)
    end
    if isempty(caught_msg)
        n = length(VALIDATION_MESSAGES)
        throw(LavaError("verify_gpu_av",
            "GPU-AV did not fire on a known out-of-bounds write within $(timeout)s " *
            "($(n) messages captured, none matching 'Out of bounds access').",
            "Layer is silently broken: check Vulkan SDK install + VK_LAYER_KHRONOS_validation availability."))
    end
    @info "Lava: GPU-AV verified working" sample=first(caught_msg, 240)
    # The flush task may still be hung waiting on a fence the GPU never signals;
    # reset the device so subsequent code starts from a clean state.
    vk_reset_device!()
    return true
end

"""
    Lava.activate_all_debugging(; verify=true, pool_disabled=true)

One call that turns on **every** validation facility Lava supports, in the
correct configuration, and then *proves* the strongest one (GPU-AV) is
actually catching errors on the current driver. Use this instead of poking
the individual `LAVA_*` env vars / `enable_gpu_av` by hand — it removes the
"I half-configured it and got a false-clean run" trap.

Enables together, then resets the Vulkan device so they take effect:
  * `VK_LAYER_KHRONOS_validation`  — core spec checks         (`LAVA_VALIDATION`)
  * GPU-Assisted Validation        — shader OOB instrumentation (`LAVA_GPU_AV`)
  * Synchronization validation     — races / missing barriers (`LAVA_SYNC_VAL`)
  * Best-practices layer           — API-misuse warnings      (`LAVA_BEST`)
  * Pool **disabled** (`pool_disabled=true`, the default here) — one
    `VkBuffer` per `LavaArray` so GPU-AV sees *sub-pool* BDA out-of-bounds.
    With the pool on, GPU-AV is blind to overruns that stay inside the shared
    64 MiB block — i.e. exactly the bugs you reach for this function to find.

`verify=true` (default) runs `verify_gpu_av()` after setup: a known
out-of-bounds device write that the layer must report. This is the crucial
guard — GPU-AV can *attach* (`gpu_assisted=true`) yet silently fail to
instrument shaders on some driver/loader combos (observed on RADV / Mesa 26
locally: the layer attaches but never fires, so the probe's OOB is not
caught and **segfaults the process**). A segfault or thrown `LavaError` at
the verify step is therefore a definitive "GPU-AV is non-functional on this
driver" signal — do GPU-AV bug-hunting on a driver where this call returns
cleanly. Pass `verify=false` to skip the probe (and keep the session alive)
when you only want the non-instrumenting layers (core / sync / best).

Expect 10-100× slowdown. Turn everything back off with `disable_gpu_av()`.

Returns `true` if all layers activated and (when `verify=true`) GPU-AV was
proven to fire; throws `LavaError` if a layer failed to attach.
"""
function activate_all_debugging(; verify::Bool=true, pool_disabled::Bool=true)
    ENV["LAVA_VALIDATION"] = "1"
    ENV["LAVA_GPU_AV"]     = "1"
    ENV["LAVA_SYNC_VAL"]   = "1"
    ENV["LAVA_BEST"]       = "1"
    POOL_DISABLED[]        = pool_disabled
    vk_reset_device!()
    ctx = vk_context()
    if !ctx.gpu_assisted
        throw(LavaError("activate_all_debugging",
            "validation instance created but gpu_assisted=false — GPU-AV did not attach",
            "Check VK_EXT_validation_features support + VK_LAYER_KHRONOS_validation install on this driver/loader."))
    end
    @info "Lava: ALL debugging active" device=ctx.device_name validation=true gpu_av=true sync_val=true best_practices=true pool_disabled=pool_disabled
    if verify
        @info "Lava: proving GPU-AV actually fires (known-OOB probe; a segfault here means GPU-AV is non-functional on this driver)"
        verify_gpu_av()
    else
        @warn "Lava: GPU-AV NOT verified (verify=false) — a clean run does not prove the layer is catching anything. Call verify_gpu_av() to confirm."
    end
    return true
end
