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
                                ::Type{U}; rowmajor::Bool = false) where {T,U<:MatrixUse}
    haskey(COOPMAT_DTYPE_SUFFIX, T) ||
        throw(ArgumentError("no cooperative-matrix component type for $T"))
    base = "_lava_coopmat_$(op)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])"
    return rowmajor ? base * "_row" : base
end

# Each distinct (op, T, M, N, Use) needs its own LLVM declaration, so the stubs
# are generated on demand from the type parameters rather than enumerated.

@generated function coopmat_load(::Type{AcceleratedMatrix{T,M,N,U}}, ptr::Ptr{S},
                                 offset::Integer, stride::Integer,
                                 ::Val{RM} = Val(false)) where {T,M,N,U,S,RM}
    fname = coopmat_intrinsic_name("load", T, M, N, U; rowmajor = RM)
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

# ── Workgroup (shared) memory ────────────────────────────────────────────────
#
# The methods above take a `Ptr`, i.e. an integer device address, which the
# emitter turns into a `PhysicalStorageBuffer` pointer. Shared memory has no
# such address: `@localmem` is an LLVM global in `addrspace(3)` and the only way
# to name it is to keep the pointer, so these take a `Core.LLVMPtr{S,3}` and the
# emitter reads the storage class off the pointer instead of assuming one.
#
# Separate intrinsic names (`loadw`/`storew`) rather than the same name with a
# different argument type, because the emitter dispatches on the LLVM function
# name and nothing else — the same reason the shape is in the name.
#
# This is what a shared-memory staged GEMM needs. Loading cooperative matrices
# straight from global, as `coopmat_gemm_kernel!` does, re-reads each tile once
# per register-block row and column; every high-performance GEMM stages a block
# into shared once and loads from there.

"""
The element offset is an *argument*, not a GEP.

Doing it as a GEP made the emitter's job depend on whether LLVM had folded it:
at offset 1 the `getelementptr` by zero disappeared and what arrived was a
pointer to the whole array, while at offset 17 it survived — as a constant
expression rather than an instruction, so no test on the LLVM node kind
distinguished the two reliably. Passing the index through means the emitter
always sees the same shape and always emits exactly one `OpAccessChain`.
"""
# Whether a `@localmem` element type is a 2-wide vector of the matrix's component
# type, i.e. the `f16vec2` staging buffer `mul_mm.comp` uses.
#
# `OpCooperativeMatrixLoadKHR` allows a pointer to a *vector* whose component type
# matches the matrix's, with the stride counted in those vectors — which is how the
# reference gets 32-bit shared accesses out of an fp16 tile. `@localmem Float16`
# gives 16-bit ones, and widening only the global load without widening the shared
# array loses to bank conflicts (scalar 1.04-1.09x, 2-wide 0.80-0.92x, 4-wide
# 0.69-0.81x — see the note in `array/gemm.jl`).
#
# A comment, not a docstring: this sits between the docstring below and the
# function it documents, and two adjacent string literals make `@doc` try to
# document the second one.
coopmat_vec2(::Type{NTuple{2,VecElement{T}}}, ::Type{T}) where {T} = true
coopmat_vec2(::Type, ::Type) = false

@generated function coopmat_load(::Type{AcceleratedMatrix{T,M,N,U}},
                                 ptr::Core.LLVMPtr{S,3},
                                 offset::Integer, stride::Integer,
                                 ::Val{RM} = Val(false)) where {T,M,N,U,S,RM}
    S === T || coopmat_vec2(S, T) ||
        throw(ArgumentError("coopmat_load: a Workgroup array of $S cannot back a \
                             cooperative matrix of $T; use $T or NTuple{2,VecElement{$T}}"))
    # `loadw2` when the shared array is a vector of `T`: same instruction, but the
    # emitter has to build a pointer to the *vector* rather than to `T`.
    fname = coopmat_intrinsic_name(S === T ? "loadw" : "loadw2", T, M, N, U;
                                   rowmajor = RM)
    ir = """
        declare i32 @$fname(ptr addrspace(3), i32, i32) #0
        define i32 @entry(ptr addrspace(3) %p, i32 %o, i32 %s) #0 {
            %r = call i32 @$fname(ptr addrspace(3) %p, i32 %o, i32 %s)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32,
                          Tuple{Core.LLVMPtr{$S,3},UInt32,UInt32},
                          ptr, UInt32(offset - 1), UInt32(stride))
        AcceleratedMatrix{$T,$M,$N,$U}(h)
    end
end

@generated function coopmat_store(ptr::Core.LLVMPtr{S,3}, offset::Integer,
                                  stride::Integer,
                                  m::AcceleratedMatrix{T,M,N,U}) where {S,T,M,N,U}
    fname = coopmat_intrinsic_name("storew", T, M, N, U)
    ir = """
        declare void @$fname(ptr addrspace(3), i32, i32, i32) #0
        define void @entry(ptr addrspace(3) %p, i32 %o, i32 %s, i32 %h) #0 {
            call void @$fname(ptr addrspace(3) %p, i32 %o, i32 %s, i32 %h)
            ret void
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), Cvoid,
                      Tuple{Core.LLVMPtr{$S,3},UInt32,UInt32,Int32},
                      ptr, UInt32(offset - 1), UInt32(stride), m.handle)
        nothing
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

@generated function coopmat_convert(::Type{AcceleratedMatrix{T,M,N,U}},
                                    m::AcceleratedMatrix{S,M,N,U}) where {T,S,M,N,U}
    fname = coopmat_intrinsic_name("convert", T, M, N, U)
    ir = """
        declare i32 @$fname(i32) #0
        define i32 @entry(i32 %h) #0 {
            %r = call i32 @$fname(i32 %h)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32}, m.handle)
        AcceleratedMatrix{$T,$M,$N,$U}(h)
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
    (M, N) in ((16, 16), (16, 8)),
    op in ("load", "store", "zero", "muladd", "loadw", "loadw2", "storew", "convert")
    push!(KNOWN_INTRINSICS, coopmat_intrinsic_name(op, T, M, N, U))
    op in ("load", "loadw", "loadw2", "store", "storew") &&
        push!(KNOWN_INTRINSICS, coopmat_intrinsic_name(op, T, M, N, U; rowmajor = true))
end

