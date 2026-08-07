# What a live tensor slice costs in registers — and the control that makes it an
# attribution instead of a correlation.
#
# This exists because it is the answer to "why is the tensor-addressed GEMM
# slower than the staged one" (see bench_tensor_gemm_full.jl). Four other
# explanations — registers-in-aggregate, occupancy, instruction count, global
# traffic — each made a prediction and each failed it. This one survived.
#
# THE TRAP THIS FILE IS SHAPED AROUND: the obvious sweep varies the number of
# hoisted slices, but hoisting S slices also puts S loads in flight. Two things
# move at once and the register growth can be pinned on either. The control arm
# holds ONE slice and varies only the loads, by putting the offset in the
# pointer. Run both or believe neither.
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix
const TT = 16

"""
    slice_sweep_kernel(S) -> function

S slices held live across the k-loop, S loads per iteration, one accumulator.
Varies live slices AND loads together — the arm that needs a control.
"""
function slice_sweep_kernel(S::Int)
    name = Symbol("tensor_slice_sweep_", S)
    @eval begin
        @kernel cpu = false unsafe_indices = true function $name(Ct, @Const(A), @Const(B),
                                                                  M::Int32, N::Int32, K::Int32)
            grp = @index(Group, Linear) - 1
            p0 = Int32(grp) * Int32(TT)
            lb = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (N, K))
            la = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (K, M))
            zb = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixA})
            za = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixB})
            acc = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})
            Base.Cartesian.@nexprs $S i -> (sb_i = Lava.tensor_slice(lb,
                (p0 + Int32((i - 1) * TT), Int32(0)), (Int32(TT), Int32(TT))))
            sa = Lava.tensor_slice(la, (Int32(0), Int32(0)), (Int32(TT), Int32(TT)))
            pb = UInt64(pointer(B)); pa = UInt64(pointer(A))
            nk = (K + Int32(TT) - Int32(1)) ÷ Int32(TT)
            for kk in Int32(0):(nk - Int32(1))
                ko = UInt64(kk) * UInt64(TT); bo = pb + ko * 0x2; ao = pa + ko * UInt64(M) * 0x2
                a = Lava.tensor_load(za, ao, sa)
                Base.Cartesian.@nexprs $S i -> (acc = Lava.coopmat_muladd(
                    Lava.tensor_load(zb, bo, sb_i), a, acc))
            end
            Lava.copyto!(pointer(Ct), 1 + p0, N, acc)
        end
    end
    launch_sweep(kernel_binding(@__MODULE__, name), S)
end

"""
    load_sweep_kernel(S) -> function

CONTROL. S loads per iteration like the arm above, but ONE live slice: the tile
offset moves in the base pointer instead. Isolates loads-in-flight.
"""
function load_sweep_kernel(S::Int)
    name = Symbol("tensor_load_sweep_", S)
    @eval begin
        @kernel cpu = false unsafe_indices = true function $name(Ct, @Const(A), @Const(B),
                                                                  M::Int32, N::Int32, K::Int32)
            grp = @index(Group, Linear) - 1
            p0 = Int32(grp) * Int32(TT)
            lb = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (N, K))
            la = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (K, M))
            zb = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixA})
            za = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixB})
            acc = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})
            sb = Lava.tensor_slice(lb, (p0, Int32(0)), (Int32(TT), Int32(TT)))
            sa = Lava.tensor_slice(la, (Int32(0), Int32(0)), (Int32(TT), Int32(TT)))
            pb = UInt64(pointer(B)); pa = UInt64(pointer(A))
            nk = (K + Int32(TT) - Int32(1)) ÷ Int32(TT)
            for kk in Int32(0):(nk - Int32(1))
                ko = UInt64(kk) * UInt64(TT); bo = pb + ko * 0x2; ao = pa + ko * UInt64(M) * 0x2
                a = Lava.tensor_load(za, ao, sa)
                Base.Cartesian.@nexprs $S i -> (acc = Lava.coopmat_muladd(
                    Lava.tensor_load(zb, bo + UInt64((i - 1) * TT) * 0x2, sb), a, acc))
            end
            Lava.copyto!(pointer(Ct), 1 + p0, N, acc)
        end
    end
    launch_sweep(kernel_binding(@__MODULE__, name), S)
end

# `@nexprs` needs a literal count, so each S needs its own @eval'd kernel — and
# reading the binding back with `getfield` in the SAME world that defined it is
# the Julia 1.12 world-age trap (a warning today, an error later). A second
# `Core.eval` sees the definition world, so the lookup is legal.
kernel_binding(mod, name::Symbol) = Core.eval(mod, name)

launch_sweep(kern, S) = (back, A, B, M, N, K) -> begin
    Ct = KA.allocate(back, Float32, N, M)
    fill!(Ct, 0f0)
    kern(back, (32,))(Ct, A, B, Int32(M), Int32(N), Int32(K); ndrange = (cld(N, TT * S) * 32,))
    Ct
end

"""Registers per (workgroup size) for whatever `f` dispatches, against a cleared cache."""
function sweep_registers(ctx, f)
    Lava.clear_kernel_cache!()
    f()
    KA.synchronize(Lava.backend(ctx))
    [(k.workgroup_size, k.registers) for k in Lava.kernel_stats(ctx)]
end

function run_register_sweeps(; M = 1280, N = 1536, K = 1280, sizes = (1, 2, 4, 6, 8))
    Lava.enable_pipeline_executable_properties!()      # MUST precede device creation
    ctx = Lava.vk_reset_device!()
    back = Lava.backend(ctx)
    A = KA.allocate(back, Float16, M, K); fill!(A, Float16(0.01))
    B = KA.allocate(back, Float16, K, N); fill!(B, Float16(0.01))
    @printf("%-6s %14s %14s\n", "S", "slices live", "loads only")
    for S in sizes
        rs = sweep_registers(ctx, () -> slice_sweep_kernel(S)(back, A, B, M, N, K))
        rl = sweep_registers(ctx, () -> load_sweep_kernel(S)(back, A, B, M, N, K))
        @printf("%-6d %14d %14d\n", S, rs[1][2], rl[1][2])
    end
    println("""
    MEASURED 2026-08-07, 32 threads, one accumulator:
        slices live   81  80  133  168  255     ~20 registers per live slice
        loads only    81  82   84   85   87     ~0.75 registers per load
    255 is the hardware cap, so S=8 is truncated and the real slope is steeper.
    A whole 16x16 fp16 coopmat operand is 4 registers per lane; a slice
    descriptor costs five of them.""")
end
