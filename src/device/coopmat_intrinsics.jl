# llvmcall stubs for SPV_KHR_cooperative_matrix.
#
# Bottom layer only: `acceleratedmatrix.jl` exposes these through ordinary Base
# functions and nothing here is user-facing. The SPIR-V emitter dispatches on
# the LLVM function name, so the matrix's shape has to be encoded *in the name*
# — `_lava_coopmat_load_f16_16x16_a` — because a cooperative matrix type is
# built from literal constants (component type, rows, columns, use) and none of
# that can travel as a runtime argument.
#
# The value itself never appears in LLVM IR. Each stub returns an `Int32`
# handle, and the emitter maps handle SSA value -> `OpCooperativeMatrix*` result
# id. A cooperative matrix is an SSA value in SPIR-V too, so the two graphs have
# the same shape; the handle exists only because LLVM needs *something* to keep
# distinct matrices distinct.

const COOPMAT_DTYPE_SUFFIX = Dict(
    Float16 => "f16", Float32 => "f32", Float64 => "f64",
    Int8 => "i8", UInt8 => "u8", Int32 => "i32", UInt32 => "u32",
)

const COOPMAT_USE_SUFFIX = Dict(MatrixA => "a", MatrixB => "b", Accumulator => "acc")

const COOPMAT_IR_TYPE = Dict(
    Float16 => "half", Float32 => "float", Float64 => "double",
    Int8 => "i8", UInt8 => "i8", Int32 => "i32", UInt32 => "i32",
)

"""
    coopmat_intrinsic_name(op, T, M, N, Use) -> String

`_lava_coopmat_<op>_<dtype>_<M>x<N>_<use>`. The emitter parses this back out;
keep the two in step.
"""
function coopmat_intrinsic_name(op::String, ::Type{T}, M::Integer, N::Integer,
                                ::Type{U}) where {T,U<:MatrixUse}
    haskey(COOPMAT_DTYPE_SUFFIX, T) ||
        throw(ArgumentError("no cooperative-matrix component type for $T"))
    "_lava_coopmat_$(op)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])"
end

# Each distinct (op, T, M, N, Use) needs its own LLVM declaration, so the stubs
# are generated on demand from the type parameters rather than enumerated.

@generated function coopmat_load(::Type{AcceleratedMatrix{T,M,N,U}}, ptr::Ptr{S},
                                 offset::Integer, stride::Integer) where {T,M,N,U,S}
    fname = coopmat_intrinsic_name("load", T, M, N, U)
    ir = """
        declare i32 @$fname(i64, i32) #0
        define i32 @entry(i64 %p, i32 %s) #0 {
            %r = call i32 @$fname(i64 %p, i32 %s)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        base = reinterpret(UInt64, ptr) + UInt64((offset - 1) * sizeof($S))
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{UInt64,UInt32},
                          base, UInt32(stride))
        AcceleratedMatrix{$T,$M,$N,$U}(h)
    end
end

@generated function coopmat_store(ptr::Ptr{S}, offset::Integer, stride::Integer,
                                  m::AcceleratedMatrix{T,M,N,U}) where {S,T,M,N,U}
    fname = coopmat_intrinsic_name("store", T, M, N, U)
    ir = """
        declare void @$fname(i64, i32, i32) #0
        define void @entry(i64 %p, i32 %s, i32 %h) #0 {
            call void @$fname(i64 %p, i32 %s, i32 %h)
            ret void
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        base = reinterpret(UInt64, ptr) + UInt64((offset - 1) * sizeof($S))
        Base.llvmcall(($ir, "entry"), Cvoid, Tuple{UInt64,UInt32,Int32},
                      base, UInt32(stride), m.handle)
        nothing
    end
end

@generated function coopmat_zero(::Type{AcceleratedMatrix{T,M,N,U}}) where {T,M,N,U}
    fname = coopmat_intrinsic_name("zero", T, M, N, U)
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
        AcceleratedMatrix{$T,$M,$N,$U}(h)
    end
end

# The accumulator's shape and type name the instruction; A and B follow from it.
@generated function coopmat_muladd(a::AcceleratedMatrix{TA,M,K,MatrixA},
                                   b::AcceleratedMatrix{TB,K,N,MatrixB},
                                   c::AcceleratedMatrix{TC,M,N,Accumulator}) where {TA,TB,TC,M,N,K}
    fname = coopmat_intrinsic_name("muladd", TC, M, N, Accumulator)
    ir = """
        declare i32 @$fname(i32, i32, i32) #0
        define i32 @entry(i32 %a, i32 %b, i32 %c) #0 {
            %r = call i32 @$fname(i32 %a, i32 %b, i32 %c)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,Int32,Int32},
                          a.handle, b.handle, c.handle)
        AcceleratedMatrix{$TC,$M,$N,Accumulator}(h)
    end
end

# The emitter matches on the prefix, so individual names need no registration;
# this keeps GPUCompiler's unknown-intrinsic check happy for the common shapes.
for T in (Float16, Float32), U in (MatrixA, MatrixB, Accumulator),
    (M, N) in ((16, 16), (16, 8)), op in ("load", "store", "zero", "muladd")
    push!(KNOWN_INTRINSICS, coopmat_intrinsic_name(op, T, M, N, U))
end

