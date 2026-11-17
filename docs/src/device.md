# Device code

What a kernel or shader can call, and what it compiles to.

## Indices, barriers, workgroup memory

KernelInterface's device API is implemented for Vulkan: `KI.get_global_id`,
`KI.get_local_id`, `KI.get_group_id`, `KI.get_num_groups`, the subgroup queries,
`KI.barrier` and `KI.localmemory`. KernelAbstractions' `@index`, `@synchronize`,
`@localmem` and `@private` lower onto the same intrinsics.

`KI.localmemory(T, Val(dims))` (and `@localmem`) gives every call site its own
`Workgroup` allocation, shared by all invocations of a workgroup.

## Subgroups

`KI.shfl`, `KI.shfl_down`, `KI.sub_group_ballot` and `KI.sub_group_reduce` lower to
`GroupNonUniform` instructions. `KI.sub_group_reduce(+, x)` is one `OpGroupNonUniformIAdd`
or `FAdd` for `Float32`, `Float64`, `Int32`, `UInt32`, `Int64` and `UInt64`; other
operators and types use KernelInterface's shuffle-based reduction. A device's
subgroup width is `Lava.caps(device).subgroup`.

## Atomics

Atomix's `@atomic`, `Atomix.modify!`, `Atomix.replace!` and `Atomix.swap!` compile
to SPIR-V atomics on device and workgroup memory. The requested memory ordering is
kept. Vulkan has no sequential consistency, so `:sequentially_consistent` lowers to
the strongest ordering Vulkan has for the operation: acquire for a load, release for
a store, acquire-release for a read-modify-write. `Float64` addition is a
compare-and-swap loop.

## Math

`Float32` and `Float16` functions lower to `GLSL.std.450` extended instructions where
one exists with Julia's semantics. `GLSL.std.450` has no 64-bit `exp`, `log`, `asin`,
`acos`, `atan` and `atan2`; for `Float64` these are table-free fdlibm algorithms,
accurate to within an ulp. A domain error returns `NaN`, because a kernel cannot
throw.

`min`, `max`, `^`, `round`, `clamp` and `hypot` follow Julia: `min` and `max`
propagate `NaN` and order `-0.0` below `0.0`, `round` rounds ties to even, and `^`
handles negative bases, zero, infinities and integer powers as Base does. The
relaxed hardware forms are opt-in through `Base.FastMath`:

| Julia | relaxed form | lowers to |
|---|---|---|
| `min`, `max` | `Base.FastMath.min_fast`, `max_fast` | `NMin`, `NMax` |
| `x ^ y` | `Base.FastMath.pow_fast` | `Pow` |

`@fastmath` rewrites a block to the relaxed forms.

## Printing

`@lava_printf "fmt" args...` prints with a C format string; KernelAbstractions'
`@print` chooses the specifiers from the argument types. Both emit
`NonSemantic.DebugPrintf`, which the Vulkan validation layer executes. With Mantle,
create the device with `debug = Mantle.DebugConfig(printf = true)`; each line arrives
as an `@info` log message.

| argument | specifier |
|---|---|
| `Int32` and narrower, `Bool` | `%d` |
| `UInt32` and narrower | `%u`, `%x` |
| `Int64` | `%ld` |
| `UInt64` | `%lu` |
| `Float16`, `Float32` | `%f`, `%e`, `%g` |
| `Float64` | `%lf` |

## Integer dot product

`Lava.dot4x8(a::UInt32, b::UInt32)` is the dot product of the four signed bytes
packed in each word, accumulated in `Int32`: one `OpSDotKHR`.

## Cooperative matrices

The matrix types and operations are the [device vocabulary](vocabulary.md)'s.
`AcceleratedMatrix{T,M,N,Use}` is a subgroup-scope `M x N` tile, `Use` one of
`MatrixA`, `MatrixB`, `Accumulator`; `WorkgroupMatrix` is the workgroup-scope form
of `VK_NV_cooperative_matrix2`.

```julia
using Lava: AcceleratedMatrix, MatrixA, MatrixB, Accumulator

a = AcceleratedMatrix{Float16,16,16,MatrixA}(pointer(A), 1, 16)   # load, column-major, stride 16
b = AcceleratedMatrix{Float16,16,16,MatrixB}(pointer(B), 1, 16)
c = zero(AcceleratedMatrix{Float32,16,16,Accumulator})
copyto!(pointer(C), 1, 16, muladd(a, b, c))
```

Matrices support `muladd`, `*`, `+`, `zero`, `convert` and per-component access
(`coopmat_getcomp`, `coopmat_setcomp`, `coopmat_length`). With
`VK_NV_cooperative_matrix2`, `coopmat_perelement(f, m, extras...)` calls
`f(row, col, value, extras...)` for every element, `coopmat_reduce` combines along
rows or columns, and `TensorLayout`/`TensorView` address a tile inside a larger
tensor with clamping at its edges.

Which shapes a device runs is a host query. `Lava.caps(device).shapes` lists them,
`Lava.supports(device, shape)` checks one, and `Lava.bestshape(device, A, C)` picks
the squarest, then largest, shape for operand type `A` and accumulator `C`, or
returns `nothing`.

## Ray queries

With `VK_KHR_ray_query`, a kernel or shader traces a ray against the top-level
acceleration structure the runtime binds:

```julia
Lava.lava_ray_query_init(origin::Point3f, direction::Vec3f, tmin, tmax)
while Lava.lava_ray_query_proceed()
    Lava.lava_ray_query_confirm()            # accept a candidate triangle
end
if Lava.lava_ray_query_get_type(true) == UInt32(1)   # committed: a triangle
    t = Lava.lava_ray_query_get_t(true)
end
```

The committed hit also gives `lava_ray_query_get_instance_id`,
`lava_ray_query_get_primitive_index` and `lava_ray_query_get_barycentrics`. Mantle
builds the acceleration structures.
