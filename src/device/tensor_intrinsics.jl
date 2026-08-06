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

"""
    TensorLayout{DIM,CLAMP}

Handle for an `OpTypeTensorLayoutNV` value. `DIM` is the rank; `CLAMP` is one of
`TENSOR_CLAMP_UNDEFINED` / `TENSOR_CLAMP_CONSTANT` / `TENSOR_CLAMP_TO_EDGE`.

**`TENSOR_CLAMP_CONSTANT` is the interesting one.** Under it the driver
bounds-checks the load and substitutes a constant outside the tensor, so an
extent that does not divide the tile is legal — which is what would retire
`gemm_padn`, the `GEMM_BLOCK` pad on M and `padtile`/`crsextent` on K, and the
`gemm_divides` gate that declines shapes outright.
"""
struct TensorLayout{DIM,CLAMP}
    handle::Int32
end

tensor_intrinsic_name(op::AbstractString, dim::Integer, clamp::UInt32) =
    "_lava_tensor_$(op)_$(dim)_$(clamp)"

# Every name has to be registered, or GPUCompiler's IR validation rejects the
# call as "unsupported call to an unknown function" long before Lava's emitter
# ever sees it — `GPUCompiler.isintrinsic(::LavaCompilerJob, …)` whitelists a few
# prefixes and consults `KNOWN_INTRINSICS` for everything else. The coopmat
# intrinsics enumerate their shapes the same way; there is no prefix rule.
#
# Ranks 1-4 and all three clamp modes is the whole reachable set: rank comes from
# the operand's dimensionality (2 for a GEMM, 4 for a convolution's im2col view)
# and the mode from `TENSOR_CLAMP_*`.
for dim in 1:4, clamp in (TENSOR_CLAMP_UNDEFINED, TENSOR_CLAMP_CONSTANT, TENSOR_CLAMP_TO_EDGE),
    op in ("create", "setdim", "slice")
    push!(KNOWN_INTRINSICS, tensor_intrinsic_name(op, dim, clamp))
end

"""Name of the tensor LOAD, which additionally carries the matrix shape.

`U` is a `MatrixUse` TYPE, not a number, so it goes through `COOPMAT_USE_SUFFIX`
exactly as `coopmat_intrinsic_name` does — interpolating the type itself yields
`Lava.Accumulator` and the emitter's `parse` then chokes on the `L`."""
tensor_load_name(dim::Integer, clamp::UInt32, ::Type{T}, M, N,
                 ::Type{U}) where {T,U<:MatrixUse} =
    "_lava_tensor_load_$(dim)_$(clamp)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])"

for dim in 1:4, clamp in (TENSOR_CLAMP_UNDEFINED, TENSOR_CLAMP_CONSTANT, TENSOR_CLAMP_TO_EDGE),
    T in (Float16, Float32), U in (MatrixA, MatrixB, Accumulator),
    (M, N) in ((16, 16), (16, 8), (8, 8))
    push!(KNOWN_INTRINSICS, tensor_load_name(dim, clamp, T, M, N, U))
end

"""
    tensor_layout(Val(DIM), Val(CLAMP)) -> TensorLayout

`OpCreateTensorLayoutNV`. Takes no operands — the rank and clamp mode live in
the type, and the dimensions are set separately by [`tensor_setdim`](@ref)
because they are runtime values.
"""
@generated function tensor_layout(::Val{DIM}, ::Val{CLAMP}) where {DIM,CLAMP}
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
    tensor_setdim(layout, dims::NTuple{DIM,Int32}) -> TensorLayout

`OpTensorLayoutSetDimensionNV` — the extent of the whole tensor, one runtime
`<id>` per dimension. Returns a NEW layout; these are values, not mutable state.
"""
@generated function tensor_setdim(l::TensorLayout{DIM,CLAMP},
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
    tensor_slice(layout, offsets, sizes) -> TensorLayout

`OpTensorLayoutSliceNV` — the sub-block this workgroup owns.

Operands are OFFSET/SIZE **pairs**, one pair per dimension, not all offsets
followed by all sizes. Read off glslang's output for
`sliceTensorLayoutNV(tl, 0, 16, 0, 16)`, which emits `%0 %16 %0 %16`.
"""
@generated function tensor_slice(l::TensorLayout{DIM,CLAMP},
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

"""
    tensor_load(A::AcceleratedMatrix, addr::UInt64, layout) -> AcceleratedMatrix

`OpCooperativeMatrixLoadTensorNV` — fill a matrix straight from memory through
`layout`, with no shared-memory staging.

`A` is the `%object` operand and it is **not** a destination: it is the matrix's
existing value, which out-of-range elements KEEP. That only matters under a
clamping layout, and it is why this takes a matrix in as well as returning one —
glslang emits an `OpLoad` of the target for exactly this reason. Pass a zeroed
matrix when every element is in range.
"""
@generated function tensor_load(a::AcceleratedMatrix{T,M,N,U}, addr::UInt64,
                                l::TensorLayout{DIM,CLAMP}) where {T,M,N,U,DIM,CLAMP}
    fname = tensor_load_name(DIM, UInt32(CLAMP), T, M, N, U)
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
        AcceleratedMatrix{$T,$M,$N,$U}(h)
    end
end
