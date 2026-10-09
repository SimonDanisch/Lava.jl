# SPV_NV_tensor_addressing front end.
#
# The staging half of coopmat2: instead of copying a block of an operand into
# shared memory and loading the matrix from there, a TENSOR LAYOUT describes the
# array in memory and the matrix is loaded straight out of it. The driver does
# the striding, the bounds-checking and the coalescing.
#
# Shaped exactly like `coopmat_intrinsics.jl`: each operation is an `llvmcall` to
# a declared-but-undefined function whose name carries everything that must be a
# compile-time constant, returning an `i32` handle that the emitter maps to a
# SPIR-V result id. A layout is an SSA value in SPIR-V just as it is in LLVM, so
# the two graphs line up and nothing needs a variable.
#
#     _lava_tensor_<op>_<dim>_<clamp>
#
# `dim` and `clamp` are in the NAME because they are operands of
# `OpTypeTensorLayoutNV`, which is built from constants — see
# `compiler/spirv/coopmat.jl` and `test/glsl/tensor_addressing_opcodes.comp` for
# where the numbers come from.

# `TensorLayout`, `TensorView`, the clamp modes and the operations themselves are
# KernelInterface's (`KernelInterface/src/tensor.jl`): a kernel library names them
# without naming this compiler. What follows is Lava's lowering of each, and the
# overlays at the end of the file that make KernelInterface's functions reach it.
# A layout's storage here is the `Int32` handle the emitter maps to a result id.
import KernelInterface: TensorLayout, TensorView, tensor_layout, tensor_setdim,
                        tensor_setstride, tensor_setclampvalue, tensor_clampbits,
                        tensor_slice, tensor_view, tensor_load, tensor_store

tensor_intrinsic_name(op::AbstractString, dim::Integer, clamp::UInt32) =
    "_lava_tensor_$(op)_$(dim)_$(clamp)"

# An unregistered name is rejected as "unsupported call to an unknown function"
# by GPUCompiler's IR validation, long before Lava's emitter sees it.
# `GPUCompiler.isintrinsic(::LavaCompilerJob, …)` now whitelists the whole
# `_lava_tensor_` prefix, so nothing here needs enumerating — which is what lets
# rank and clamp mode vary freely instead of only in the combinations someone
# remembered to list.

"""Name of the tensor LOAD, which additionally carries the matrix shape.

`U` is a `MatrixUse` TYPE, not a number, so it goes through `COOPMAT_USE_SUFFIX`
exactly as `coopmat_intrinsic_name` does — interpolating the type itself yields
`Lava.Accumulator` and the emitter's `parse` then chokes on the `L`."""
tensor_load_name(dim::Integer, clamp::UInt32, ::Type{T}, M, N,
                 ::Type{U}, ::Type{S} = SubgroupScope) where {T,U<:MatrixUse,S<:MatrixScope} =
    "_lava_tensor_load_$(dim)_$(clamp)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])" *
    COOPMAT_SCOPE_SUFFIX[S]

# Not enumerated: `isintrinsic` accepts the `_lava_tensor_` prefix, so the load
# works at any cooperative-matrix shape the device supports — including the
# flexible dimensions a fixed (16,16)/(16,8)/(8,8) list silently excluded.

"""
    lava_tensor_layout(Val(DIM), Val(CLAMP)) -> TensorLayout

`OpCreateTensorLayoutNV`. Takes no operands — the rank and clamp mode live in
the type, and the dimensions are set separately by [`tensor_setdim`](@ref)
because they are runtime values.
"""
@generated function lava_tensor_layout(::Val{DIM}, ::Val{CLAMP}) where {DIM,CLAMP}
    fname = tensor_intrinsic_name("create", DIM, UInt32(CLAMP))
    ir = """
        declare i32 @$fname() #0
        define i32 @entry() #0 {
            %r = call i32 @$fname()
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{})
        TensorLayout{$DIM,$CLAMP}(h)
    end
end

"""
    lava_tensor_setdim(layout, dims::NTuple{DIM,Int32}) -> TensorLayout

`OpTensorLayoutSetDimensionNV` — the extent of the whole tensor, one runtime
`<id>` per dimension. Returns a NEW layout; these are values, not mutable state.
"""
@generated function lava_tensor_setdim(l::TensorLayout{DIM,CLAMP},
                                  dims::NTuple{DIM,Int32}) where {DIM,CLAMP}
    fname = tensor_intrinsic_name("setdim", DIM, UInt32(CLAMP))
    params = join(("i32" for _ in 1:(DIM + 1)), ", ")
    argl = join(("i32 %a$i" for i in 0:DIM), ", ")
    callargs = join(("i32 %a$i" for i in 0:DIM), ", ")
    ir = """
        declare i32 @$fname($params) #0
        define i32 @entry($argl) #0 {
            %r = call i32 @$fname($callargs)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    argtypes = Expr(:curly, :Tuple, :Int32, (:Int32 for _ in 1:DIM)...)
    args = [:(dims[$i]) for i in 1:DIM]
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, $argtypes, l.handle, $(args...))
        TensorLayout{$DIM,$CLAMP}(h)
    end
end

"""
    lava_tensor_setstride(layout, strides::NTuple{DIM,Int32}) -> TensorLayout

`OpTensorLayoutSetStrideNV` — the distance in ELEMENTS between successive indices
of each dimension, innermost last, in the same order as [`tensor_setdim`](@ref).

Without it a layout describes a **packed** tensor. That is enough for a slab that
happens to be contiguous, and not enough for anything else: attention's `q`, `k`
and `v` are a permuted view of one packed `(E, L, H, B)` block, so a layout over
one head's `(E, L)` slab has to be told that consecutive `L` are `stride(q, 2)`
apart rather than `E`. Both coopmat2 reference shaders set it on every layout.

Returns a NEW layout; these are values, not mutable state.
"""
@generated function lava_tensor_setstride(l::TensorLayout{DIM,CLAMP},
                                     strides::NTuple{DIM,Int32}) where {DIM,CLAMP}
    fname = tensor_intrinsic_name("setstride", DIM, UInt32(CLAMP))
    params = join(("i32" for _ in 1:(DIM + 1)), ", ")
    argl = join(("i32 %a$i" for i in 0:DIM), ", ")
    callargs = join(("i32 %a$i" for i in 0:DIM), ", ")
    ir = """
        declare i32 @$fname($params) #0
        define i32 @entry($argl) #0 {
            %r = call i32 @$fname($callargs)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    argtypes = Expr(:curly, :Tuple, :Int32, (:Int32 for _ in 1:DIM)...)
    args = [:(strides[$i]) for i in 1:DIM]
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, $argtypes, l.handle, $(args...))
        TensorLayout{$DIM,$CLAMP}(h)
    end
end

"""
    lava_tensor_setclampvalue(layout, value::Int32) -> TensorLayout

`OpTensorLayoutSetClampValueNV` — what a `TENSOR_CLAMP_CONSTANT` load
substitutes for elements outside the tensor. The default is zero.

Zero is not always the useful identity. Attention loads `V` as `Bc x EP` from a
slab only `E` wide, so `EP - E` columns of every `V` tile are fill; with a fill
of **one** those columns of `P x V` come out as the ROW SUM of `P`, which is the
quantity the online softmax otherwise spends a `coopmat_reduce` per key block
computing. The padding is discarded by the clamping store either way.

`value` is a single 32-bit operand whatever the component type — one operand,
not one per dimension. What its bits mean for a narrower component is NOT stated
by the GLSL signature (a `uint` for an fp16 matrix); `mwe_tensor_clampvalue.jl`
determines it on the device rather than assuming, and
[`tensor_clampbits`](@ref) is the answer in the form callers should use.
"""
@generated function lava_tensor_setclampvalue(l::TensorLayout{DIM,CLAMP},
                                         value::Int32) where {DIM,CLAMP}
    fname = tensor_intrinsic_name("setclampvalue", DIM, UInt32(CLAMP))
    ir = """
        declare i32 @$fname(i32, i32) #0
        define i32 @entry(i32 %a0, i32 %a1) #0 {
            %r = call i32 @$fname(i32 %a0, i32 %a1)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,Int32}, l.handle, value)
        TensorLayout{$DIM,$CLAMP}(h)
    end
end


"""
    lava_tensor_slice(layout, offsets, sizes) -> TensorLayout

`OpTensorLayoutSliceNV` — the sub-block this workgroup owns.

Operands are OFFSET/SIZE **pairs**, one pair per dimension, not all offsets
followed by all sizes. Read off glslang's output for
`sliceTensorLayoutNV(tl, 0, 16, 0, 16)`, which emits `%0 %16 %0 %16`.
"""
@generated function lava_tensor_slice(l::TensorLayout{DIM,CLAMP},
                                 offsets::NTuple{DIM,Int32},
                                 sizes::NTuple{DIM,Int32}) where {DIM,CLAMP}
    fname = tensor_intrinsic_name("slice", DIM, UInt32(CLAMP))
    n = 2DIM + 1
    params = join(("i32" for _ in 1:n), ", ")
    argl = join(("i32 %a$i" for i in 0:(n - 1)), ", ")
    callargs = join(("i32 %a$i" for i in 0:(n - 1)), ", ")
    ir = """
        declare i32 @$fname($params) #0
        define i32 @entry($argl) #0 {
            %r = call i32 @$fname($callargs)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    argtypes = Expr(:curly, :Tuple, :Int32, (:Int32 for _ in 1:(2DIM))...)
    # interleaved, because the instruction wants pairs
    args = Expr[]
    for i in 1:DIM
        push!(args, :(offsets[$i]))
        push!(args, :(sizes[$i]))
    end
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, $argtypes, l.handle, $(args...))
        TensorLayout{$DIM,$CLAMP}(h)
    end
end

# A `TensorView` (KernelInterface's) permutes a layout's dimensions at a load.
#
# Handle for an `OpTypeTensorViewNV` value. `PERM` is the dimension permutation as
# a tuple; `(1, 0)` is the transpose.
#
# **This is what replaces staging a transposed copy.** A flash-attention kernel
# wants `S = Q·Kᵀ`, and with `(E, L, H, B)` slabs both Q and K sit in memory as
# `(E, L)` — so one of the two operands is always the wrong way round for
# `coopmat_muladd`, whose operand shapes are pinned by
# `mwe_tensor_gemm_nonsquare.jl`: a logical `M x K` operand must live as `K x M`.
# Loading K through a transposing view reads it in place instead.
tensor_view_name(dim::Integer, perm) =
    "_lava_tensor_view_$(dim)_0_" * join(perm, "x")

"""
    lava_tensor_view(Val(DIM), Val(PERM)) -> TensorView

`OpCreateTensorViewNV`. Takes no operands — rank and permutation live in the
type, like [`tensor_layout`](@ref)'s rank and clamp mode.

The clamp field in the intrinsic's name is a fixed `0`: a view has no clamp mode,
and it is present only so the shared `parse_tensor_name` still applies.
"""
@generated function lava_tensor_view(::Val{DIM}, ::Val{PERM}) where {DIM,PERM}
    fname = tensor_view_name(DIM, PERM)
    ir = """
        declare i32 @$fname() #0
        define i32 @entry() #0 {
            %r = call i32 @$fname()
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{})
        TensorView{$DIM,$PERM}(h)
    end
end

"""Name of the tensor load THROUGH A VIEW. Distinct from `tensor_load_name`
because the two have different arity under the same matrix type; see the method."""
tensor_loadview_name(dim::Integer, clamp::UInt32, ::Type{T}, M, N,
                     ::Type{U}, ::Type{S} = SubgroupScope) where {T,U<:MatrixUse,S<:MatrixScope} =
    "_lava_tensor_loadv_$(dim)_$(clamp)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])" *
    COOPMAT_SCOPE_SUFFIX[S]

"""
    lava_tensor_load(A::AcceleratedMatrix, addr::UInt64, layout) -> AcceleratedMatrix

`OpCooperativeMatrixLoadTensorNV` — fill a matrix straight from memory through
`layout`, with no shared-memory staging.

`A` is the `%object` operand and it is **not** a destination: it is the matrix's
existing value, which out-of-range elements KEEP. That only matters under a
clamping layout, and it is why this takes a matrix in as well as returning one —
glslang emits an `OpLoad` of the target for exactly this reason. Pass a zeroed
matrix when every element is in range.
"""
@generated function lava_tensor_load(a::CoopMatrix{T,M,N,U,SC}, addr::UInt64,
                                l::TensorLayout{DIM,CLAMP}) where {T,M,N,U,SC,DIM,CLAMP}
    fname = tensor_load_name(DIM, UInt32(CLAMP), T, M, N, U, SC)
    ir = """
        declare i32 @$fname(i64, i32, i32) #0
        define i32 @entry(i64 %p, i32 %o, i32 %l) #0 {
            %r = call i32 @$fname(i64 %p, i32 %o, i32 %l)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{UInt64,Int32,Int32},
                          addr, a.handle, l.handle)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
    lava_tensor_load(A, addr::UInt64, layout, view::TensorView) -> AcceleratedMatrix

As [`tensor_load`](@ref), but reading through a `TensorView` — the permuted form.

The view rides in the instruction's `TensorAddressingOperands` mask rather than
as a plain operand, which is why this is a separate method and not a default
argument: the encoding differs, not just the argument list.
"""
@generated function lava_tensor_load(a::CoopMatrix{T,M,N,U,SC}, addr::UInt64,
                                l::TensorLayout{DIM,CLAMP},
                                v::TensorView{DIM,PERM}) where {T,M,N,U,SC,DIM,CLAMP,PERM}
    # `loadv`, NOT `load`: the two differ in ARITY, and a kernel that loads two
    # matrices of the same type — one plain, one through a view — would otherwise
    # declare one LLVM name with two signatures. That is exactly what happened in
    # the flash MWE, where K (viewed) and V (plain) are both MatrixB{f16,16,16}:
    # the view arrived at the instruction as a constant and spirv-val rejected it
    # with "does not have a tensor view type".
    fname = tensor_loadview_name(DIM, UInt32(CLAMP), T, M, N, U, SC)
    ir = """
        declare i32 @$fname(i64, i32, i32, i32) #0
        define i32 @entry(i64 %p, i32 %o, i32 %l, i32 %v) #0 {
            %r = call i32 @$fname(i64 %p, i32 %o, i32 %l, i32 %v)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{UInt64,Int32,Int32,Int32},
                          addr, a.handle, l.handle, v.handle)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""Name of the tensor STORE. Same shape as `tensor_load_name`; see it for why
`U` goes through `COOPMAT_USE_SUFFIX` rather than being interpolated."""
tensor_store_name(dim::Integer, clamp::UInt32, ::Type{T}, M, N,
                  ::Type{U}, ::Type{S} = SubgroupScope) where {T,U<:MatrixUse,S<:MatrixScope} =
    "_lava_tensor_store_$(dim)_$(clamp)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])" *
    COOPMAT_SCOPE_SUFFIX[S]

"""
    lava_tensor_store(a::AcceleratedMatrix, addr::UInt64, layout) -> nothing

`OpCooperativeMatrixStoreTensorNV` — write a matrix straight to memory through
`layout`, the mirror of [`tensor_load`](@ref).

**This is the half that makes a ragged OUTPUT legal.** A clamping layout
bounds-checks writes the same way it bounds-checks reads, so a tile straddling
the edge of an `M x N` destination writes only the elements inside it. Without
it a tensor-addressed GEMM can consume unpadded operands but still cannot
produce an unpadded result — the loads clamp and the store runs off the end.

Unlike the load, `a` here IS the value being written: there is no `%object`
distinction to make, because nothing is left over to keep.
"""
@generated function lava_tensor_store(a::CoopMatrix{T,M,N,U,SC}, addr::UInt64,
                                 l::TensorLayout{DIM,CLAMP}) where {T,M,N,U,SC,DIM,CLAMP}
    fname = tensor_store_name(DIM, UInt32(CLAMP), T, M, N, U, SC)
    ir = """
        declare void @$fname(i64, i32, i32) #0
        define void @entry(i64 %p, i32 %m, i32 %l) #0 {
            call void @$fname(i64 %p, i32 %m, i32 %l)
            ret void
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt64,Int32,Int32},
                      addr, a.handle, l.handle)
        nothing
    end
end

"""Name of the tensor store THROUGH A VIEW. A distinct name for the same reason
`tensor_loadview_name` is distinct from `tensor_load_name`: the two have
different arity under one matrix type, and a single LLVM symbol cannot carry both
signatures — which is how a view once reached an instruction as a constant."""
tensor_storeview_name(dim::Integer, clamp::UInt32, ::Type{T}, M, N,
                      ::Type{U}, ::Type{S} = SubgroupScope) where {T,U<:MatrixUse,S<:MatrixScope} =
    "_lava_tensor_storev_$(dim)_$(clamp)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])" *
    COOPMAT_SCOPE_SUFFIX[S]

"""
    lava_tensor_store(a, addr::UInt64, layout, view::TensorView) -> nothing

As [`tensor_store`](@ref), but writing through a `TensorView` — the permuted form.

`mul_mm_cm2.comp` stores its result this way rather than transposing it first:
the accumulator is `M x N` and the destination is laid out `N x M`, so the view
turns the write around instead of a staging pass. That makes this the last
primitive a tensor-addressed GEMM needs, and the reason it exists.

The view rides in the `TensorAddressingOperands` mask rather than as a plain
operand, so this is a separate method and not a default argument — the encoding
differs, not just the argument list.
"""
@generated function lava_tensor_store(a::CoopMatrix{T,M,N,U,SC}, addr::UInt64,
                                 l::TensorLayout{DIM,CLAMP},
                                 v::TensorView{DIM,PERM}) where {T,M,N,U,SC,DIM,CLAMP,PERM}
    fname = tensor_storeview_name(DIM, UInt32(CLAMP), T, M, N, U, SC)
    ir = """
        declare void @$fname(i64, i32, i32, i32) #0
        define void @entry(i64 %p, i32 %m, i32 %l, i32 %v) #0 {
            call void @$fname(i64 %p, i32 %m, i32 %l, i32 %v)
            ret void
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt64,Int32,Int32,Int32},
                      addr, a.handle, l.handle, v.handle)
        nothing
    end
end

# ── KernelInterface's operations, lowered ────────────────────────────────────
#
# Overlays rather than methods: a plain method on KernelInterface's function has no
# backend in its signature, so it would answer for every backend in the session.
@lava_device_override @inline tensor_layout(d::Val, c::Val) = lava_tensor_layout(d, c)
@lava_device_override @inline tensor_setdim(l::TensorLayout, dims::Tuple) =
    lava_tensor_setdim(l, dims)
@lava_device_override @inline tensor_setstride(l::TensorLayout, strides::Tuple) =
    lava_tensor_setstride(l, strides)
@lava_device_override @inline tensor_setclampvalue(l::TensorLayout, value::Int32) =
    lava_tensor_setclampvalue(l, value)
@lava_device_override @inline tensor_slice(l::TensorLayout, offsets::Tuple, sizes::Tuple) =
    lava_tensor_slice(l, offsets, sizes)
@lava_device_override @inline tensor_view(d::Val, p::Val) = lava_tensor_view(d, p)
@lava_device_override @inline tensor_load(a::CoopMatrix, addr::UInt64, l::TensorLayout) =
    lava_tensor_load(a, addr, l)
@lava_device_override @inline tensor_load(a::CoopMatrix, addr::UInt64, l::TensorLayout,
                                          v::TensorView) =
    lava_tensor_load(a, addr, l, v)
@lava_device_override @inline tensor_store(a::CoopMatrix, addr::UInt64, l::TensorLayout) =
    lava_tensor_store(a, addr, l)
@lava_device_override @inline tensor_store(a::CoopMatrix, addr::UInt64, l::TensorLayout,
                                           v::TensorView) =
    lava_tensor_store(a, addr, l, v)
