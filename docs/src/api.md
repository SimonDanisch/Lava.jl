# API Reference

This page lists the public surface of Lava. Items marked stable belong to the GPUArrays / KernelAbstractions interface and follow semver. Graphics and ray-tracing items are still evolving.

## Backend & arrays

```@docs
LavaBackend
LavaArray
LavaDeviceArray
```

## Compute kernels

```@docs
Lava.lava_launch!
Lava.vk_flush!
Lava.clear_kernel_cache!
Lava.clear_spirv_disk_cache!
```

## Hardware ray tracing

```@docs
HWTLAS
HardwareAccel
RayTracingPipeline
trace_closest_hits!
```

## Graphics

```@docs
Rasterizer
TrianglePipeline
LinePipeline
GraphicsPipeline
RenderWindow
LavaFramebuffer
LavaTexture2D
LavaSampler
present_frame!
```

## Debugging

```@docs
Lava.vk_context
vk_reset_device!
dump_state
gpu_memory_usage
set_dispatch_logging!
get_dispatch_log
```

### Printing from kernels

```@docs
@lava_printf
Lava.enable_debug_printf!
Lava.disable_debug_printf!
Lava.get_printf_output
Lava.clear_printf_output!
```

## Index

```@index
```
