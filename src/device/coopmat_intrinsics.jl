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

# Subgroup scope adds NOTHING to the name. It is the default and by far the
# common case, so every name a kernel emitted before workgroup scope existed is
# unchanged — which keeps the frozen SPIR-V cache keys, the emitter tests and the
# disassembly in the notes all still matching.
const COOPMAT_SCOPE_SUFFIX = Dict(SubgroupScope => "", WorkgroupScope => "_wg")

const COOPMAT_IR_TYPE = Dict(
    Float16 => "half", Float32 => "float", Float64 => "double",
    Int8 => "i8", UInt8 => "i8", Int32 => "i32", UInt32 => "i32",
)

"""
    coopmat_intrinsic_name(op, T, M, N, Use, Scope) -> String

`_lava_coopmat_<op>_<dtype>_<M>x<N>_<use>[_wg][_row]`. The emitter parses this
back out; keep the two in step.

The two optional trailing flags are order-independent to the parser but written
scope-then-layout here, and a subgroup-scope name carries no scope flag at all —
see `COOPMAT_SCOPE_SUFFIX`.
"""
function coopmat_intrinsic_name(op::String, ::Type{T}, M::Integer, N::Integer,
                                ::Type{U}, ::Type{S} = SubgroupScope;
                                rowmajor::Bool = false) where {T,U<:MatrixUse,S<:MatrixScope}
    haskey(COOPMAT_DTYPE_SUFFIX, T) ||
        throw(ArgumentError("no cooperative-matrix component type for $T"))
    base = "_lava_coopmat_$(op)_$(COOPMAT_DTYPE_SUFFIX[T])_$(M)x$(N)_$(COOPMAT_USE_SUFFIX[U])" *
           COOPMAT_SCOPE_SUFFIX[S]
    return rowmajor ? base * "_row" : base
end

# Each distinct (op, T, M, N, Use) needs its own LLVM declaration, so the stubs
# are generated on demand from the type parameters rather than enumerated.

@generated function coopmat_load(::Type{CoopMatrix{T,M,N,U,SC}}, ptr::Ptr{S},
                                 offset::Integer, stride::Integer,
                                 ::Val{RM} = Val(false)) where {T,M,N,U,SC,S,RM}
    fname = coopmat_intrinsic_name("load", T, M, N, U, SC; rowmajor = RM)
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
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
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
#
# Generalised from a 2-only predicate on 2026-08-12. The width is whatever the
# `@localmem` element is, and SPIR-V vectors are 2, 3 or 4 components — so 4-wide
# (64-bit) shared accesses are reachable and 8-wide are not. That matters: the
# staged GEMM's scalar -> vec2 step is worth **+45% to +54%** per shape on this
# card, measured against cuBLAS beside it, so the width is on the critical path
# and the next notch is worth having.
#
# (A separate, earlier experiment that widened only the GLOBAL load while leaving
# `@localmem` as `Float16` lost monotonically — 2-wide 0.80-0.92x, 4-wide
# 0.69-0.81x — because each lane then wrote V scalar stores strided by V across
# the warp, a V-way bank conflict. That result says nothing about widening BOTH
# sides, which is what this is. See the note in `array/gemm.jl`.)
coopmat_vecwidth(::Type{NTuple{N,VecElement{T}}}, ::Type{T}) where {N,T} = N
coopmat_vecwidth(::Type, ::Type) = 0

"""Whether a `@localmem` element type may back a cooperative matrix of `T`:
either `T` itself, or a SPIR-V-legal vector of it."""
coopmat_sharedok(::Type{S}, ::Type{T}) where {S,T} =
    S === T || coopmat_vecwidth(S, T) in (2, 3, 4)

@generated function coopmat_load(::Type{CoopMatrix{T,M,N,U,SC}},
                                 ptr::Core.LLVMPtr{S,3},
                                 offset::Integer, stride::Integer,
                                 ::Val{RM} = Val(false)) where {T,M,N,U,SC,S,RM}
    coopmat_sharedok(S, T) ||
        throw(ArgumentError("coopmat_load: a Workgroup array of $S cannot back a \
                             cooperative matrix of $T; use $T or \
                             NTuple{2|3|4,VecElement{$T}}"))
    # `loadw<W>` when the shared array is a W-wide vector of `T`: same
    # instruction, but the emitter has to build a pointer to the *vector* rather
    # than to `T`, and `Stride` is then counted in vectors.
    fname = coopmat_intrinsic_name(S === T ? "loadw" : "loadw$(coopmat_vecwidth(S, T))",
                                   T, M, N, U, SC; rowmajor = RM)
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
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

@generated function coopmat_store(ptr::Core.LLVMPtr{S,3}, offset::Integer,
                                  stride::Integer,
                                  m::CoopMatrix{T,M,N,U,SC},
                                  ::Val{RM} = Val(false)) where {S,T,M,N,U,SC,RM}
    fname = coopmat_intrinsic_name("storew", T, M, N, U, SC; rowmajor = RM)
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
                                  m::CoopMatrix{T,M,N,U,SC}) where {S,T,M,N,U,SC}
    fname = coopmat_intrinsic_name("store", T, M, N, U, SC)
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

"""
    coopmat_convert(::Type{CoopMatrix{T,M,N,U,S}}, m) -> CoopMatrix

Convert a cooperative matrix's component type, and optionally its **use**.

The use may differ from the source's: an fp32 `Accumulator` becoming an fp16
`MatrixA` is what feeds the second product of a flash-attention kernel, and
`flash_attn_cm2.comp` does exactly that twice (`Qf16` and `P_A`). The emitter
picks the instruction — `OpFConvert` when the uses agree, and
`OpCooperativeMatrixConvertNV` when they do not, since `OpFConvert` is only legal
between matrix types agreeing on scope, rows, columns AND use.
"""
@generated function coopmat_convert(::Type{CoopMatrix{T,M,N,U,SC}},
                                    m::CoopMatrix{S,M,N,V,SC}) where {T,S,M,N,U,V,SC}
    fname = coopmat_intrinsic_name("convert", T, M, N, U, SC)
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
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
    coopmat_undef(::Type{CoopMatrix{T,M,N,U,SC}}) -> CoopMatrix

An **uninitialised** matrix (`OpUndef`), for a destination whose prior contents
cannot be observed.

[`tensor_load`](@ref) takes its destination as an argument because glslang emits
an `OpLoad` of the target, and under a CONSTANT clamp mode the out-of-range
elements come from the layout's clamp value rather than from that target — so
nothing about the destination is readable afterwards. Passing
[`coopmat_zero`](@ref) there is a real value the allocator must hold until the
load overwrites it; GLSL declares `coopmat mat_a;` and assigns nothing.

**Only where the clamp mode makes the destination unobservable.** With
`TENSOR_CLAMP_UNDEFINED` an out-of-range element keeps whatever was in the
destination, and `OpUndef` there is genuine garbage in the output rather than
the zero the caller expected.
"""
@generated function coopmat_undef(::Type{CoopMatrix{T,M,N,U,SC}}) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("undef", T, M, N, U, SC)
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
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

@generated function coopmat_zero(::Type{CoopMatrix{T,M,N,U,SC}}) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("zero", T, M, N, U, SC)
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
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

# The accumulator's shape and type name the instruction; A and B follow from it.
# All three share one `S`, so a subgroup-scope operand cannot reach a
# workgroup-scope product: that is a method error here rather than a module the
# driver rejects.
@generated function coopmat_muladd(a::CoopMatrix{TA,M,K,MatrixA,S},
                                   b::CoopMatrix{TB,K,N,MatrixB,S},
                                   c::CoopMatrix{TC,M,N,Accumulator,S}) where {TA,TB,TC,M,N,K,S}
    fname = coopmat_intrinsic_name("muladd", TC, M, N, Accumulator, S)
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
        CoopMatrix{$TC,$M,$N,Accumulator,$S}(h)
    end
end

"""
    coopmat_length(::Type{CoopMatrix{T,M,N,U,S}}) -> Int32

Components of the matrix **this invocation** holds — `OpCooperativeMatrixLengthKHR`.

A cooperative matrix is distributed across the subgroup and the split is the
implementation's business, so this is a runtime query even though the type is
static. It is what makes an elementwise epilogue expressible at all: a GEMM that
wants `gelu` on its accumulator has to reach the components, and the only portable
way to say "all of mine" is to walk `0:length-1`.
"""
@generated function coopmat_length(::Type{CoopMatrix{T,M,N,U,SC}}) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("length", T, M, N, U, SC)
    ir = """
        declare i32 @$fname() #0
        define i32 @entry() #0 {
            %r = call i32 @$fname()
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), Int32, Tuple{})
    end
end

"""
    coopmat_getcomp(m, i) -> T

Component `i` (**0-based**) of this invocation's share of `m`.

SPIR-V has no extract-from-cooperative-matrix instruction. The access is
`OpAccessChain` into a `Function`-storage variable of the matrix type, which is
what GLSL's `mat[i]` lowers to as well; the emitter keeps one such variable per
matrix type and writes `m` into it before reading. See `coopmat.jl`.
"""
@generated function coopmat_getcomp(m::CoopMatrix{T,M,N,U,SC}, i::Int32) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("getcomp", T, M, N, U, SC)
    irty = COOPMAT_IR_TYPE[T]
    ir = """
        declare $irty @$fname(i32, i32) #0
        define $irty @entry(i32 %m, i32 %i) #0 {
            %r = call $irty @$fname(i32 %m, i32 %i)
            ret $irty %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), $T, Tuple{Int32,Int32}, m.handle, i)
    end
end

"""
    coopmat_setcomp(m, i, v) -> AcceleratedMatrix

`m` with component `i` (**0-based**) of this invocation's share replaced by `v`.

Functional, because a cooperative matrix is an SSA value in SPIR-V and in the IR
alike: the emitter stores `m` into its `Function` variable, writes the component
through an `OpAccessChain`, and loads the whole matrix back as the result.
"""
@generated function coopmat_setcomp(m::CoopMatrix{T,M,N,U,SC}, i::Int32,
                                    v::T) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("setcomp", T, M, N, U, SC)
    irty = COOPMAT_IR_TYPE[T]
    ir = """
        declare i32 @$fname(i32, i32, $irty) #0
        define i32 @entry(i32 %m, i32 %i, $irty %v) #0 {
            %r = call i32 @$fname(i32 %m, i32 %i, $irty %v)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,Int32,$T},
                          m.handle, i, v)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
    coopmat_mul(a, b) -> AcceleratedMatrix

Component-wise product of two cooperative matrices of the same type — plain
`OpFMul`, which SPV_KHR_cooperative_matrix defines to act component-wise.

This is the cheap way to apply a per-row factor to an accumulator, and unlike
`coopmat_getcomp`/`coopmat_setcomp` or `coopmat_perelement` it never asks for a
component: both operands stay wherever the implementation keeps them and the
result is one instruction. Build the factor matrix with a **stride-0** load, so
every column reads the same vector — see `AcceleratedMatrix`'s shared-memory
constructor.

Portable: this is the KHR extension, not `VK_NV_cooperative_matrix2`, so it is
also what AMD's RDNA3 WMMA path gets.
"""
@generated function coopmat_mul(a::CoopMatrix{T,M,N,U,SC},
                                b::CoopMatrix{T,M,N,U,SC}) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("mul", T, M, N, U, SC)
    ir = """
        declare i32 @$fname(i32, i32) #0
        define i32 @entry(i32 %a, i32 %b) #0 {
            %r = call i32 @$fname(i32 %a, i32 %b)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,Int32}, a.handle, b.handle)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
    coopmat_add(a, b) -> CoopMatrix

Component-wise sum — plain `OpFAdd`, which `SPV_KHR_cooperative_matrix` defines
to act component-wise exactly as `OpFMul` does.

The reason it exists: a GEMM's accumulator can be started from `bias + residual`
rather than from zero, which makes a transformer's residual add the tensor
cores' own accumulate instead of a separate pass over `M x N`. The two operands
are both accumulator loads — the bias at **stride 0** so every column reads the
same vector, the residual at its real leading dimension. See `accinit`.

Same portability as `coopmat_mul`: KHR, not `VK_NV_cooperative_matrix2`.
"""
@generated function coopmat_add(a::CoopMatrix{T,M,N,U,SC},
                                b::CoopMatrix{T,M,N,U,SC}) where {T,M,N,U,SC}
    fname = coopmat_intrinsic_name("add", T, M, N, U, SC)
    ir = """
        declare i32 @$fname(i32, i32) #0
        define i32 @entry(i32 %a, i32 %b) #0 {
            %r = call i32 @$fname(i32 %a, i32 %b)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,Int32}, a.handle, b.handle)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""LLVM type for a value that travels through `coopmat_keepparam`."""
function coopmat_keep_irtype(::Type{T}) where {T}
    T <: Core.LLVMPtr && return ("ptr addrspace($(T.parameters[2]))", "p$(T.parameters[2])")
    haskey(COOPMAT_IR_TYPE, T) || throw(ArgumentError("no LLVM type for $T"))
    (COOPMAT_IR_TYPE[T], COOPMAT_DTYPE_SUFFIX[T])
end

"""
    coopmat_keepparam(x)

Make `x` unremovable: a call to an undefined function, so no LLVM pass may
conclude the value is unused.

Needed because `OpCooperativeMatrixPerElementOpNV` names a *function* and the
SPIR-V rules fix that function's signature — `(u32 row, u32 col, T element,
extras...)` — while LLVM is free to rewrite the signature of a Julia function it
can see every caller of. Two passes do exactly that to a plausible callback:

  * **Dead-argument elimination.** A rescale that only needs the row ignores
    `col`, so the parameter goes, and the validator reports
    "second parameter type must be a 32-bit integer" — it is now the element.
  * **Interprocedural constant propagation.** A `@localmem` pointer is a
    compile-time constant, so its uses inside the callee are replaced by the
    constant and the parameter then dies the same way. The extra operand at the
    call site has nothing left to bind to.

Both were observed, in that order, on the first callback written against this
instruction. The fix is not to guess which parameters survive: it is to make
every one of them used. The emitter drops these calls, so nothing is emitted.
"""
@generated function coopmat_keepparam(x::T) where {T}
    irty, suffix = coopmat_keep_irtype(T)
    name = "_lava_keepparam_$(suffix)"
    ir = """
        declare void @$name($irty) #0
        define void @entry($irty %x) #0 {
            call void @$name($irty %x)
            ret void
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        Base.llvmcall(($ir, "entry"), Cvoid, Tuple{$T}, x)
        nothing
    end
end

for T in (Float16, Float32, Float64, Int8, UInt8, Int32, UInt32)
    push!(KNOWN_INTRINSICS, "_lava_keepparam_$(COOPMAT_DTYPE_SUFFIX[T])")
end
for as in 0:7
    push!(KNOWN_INTRINSICS, "_lava_keepparam_p$as")
end

"""
    coopmat_perelement_thunk(f, row, col, element, extras...)

The `OpFunction` that `OpCooperativeMatrixPerElementOpNV` points at.

It exists so the *user's* callback need not care about any of this: the thunk
pins its own parameters with [`coopmat_keepparam`](@ref) and then calls `f`
normally, so `f` may ignore `col`, ignore the element, or use a `@localmem`
pointer that LLVM knows is constant, and the emitted signature is still the one
the instruction requires.

`f` is a singleton — a top-level function's type has no fields — so it costs no
parameter at all and the thunk's LLVM signature is exactly
`(i32, i32, T, extras...)`.
"""
@noinline function coopmat_perelement_thunk(f::F, row::UInt32, col::UInt32, e::T,
                                            extras::Vararg{Any,NE}) where {F,T,NE}
    coopmat_keepparam(row)
    coopmat_keepparam(col)
    coopmat_keepparam(e)
    ntuple(i -> coopmat_keepparam(extras[i]), Val(NE))
    # The barrier is on the RESULT, and it is what lets a callback that ignores
    # its element work at all. `@noinline` stops inlining, not interprocedural
    # constant propagation: a callback returning a literal — `(r, c, e) -> -3f38`,
    # which is how a flash kernel's running maximum is initialised — lets Julia
    # replace the *use* of this call with that constant, the marker then reaches
    # the emitter as a constant rather than a call, and the error says the
    # callback "did not survive as a call". Which is true and unhelpful.
    Base.compilerbarrier(:const, f(row, col, e, extras...))
end

"""
    coopmat_perelement(f, m) -> AcceleratedMatrix

`m` with `f(row, col, element)` applied to every element —
`OpCooperativeMatrixPerElementOpNV`, from `VK_NV_cooperative_matrix2`.

`row` and `col` are `UInt32` and **0-based**, because they index the matrix as
SPIR-V numbers it, not as Julia would.

This is the instruction that makes a row-dependent transform of an accumulator
affordable. The portable way to write one is `coopmat_setcomp(m, i, g(coopmat_getcomp(m, i)))`
over `0:coopmat_length-1`, and that is fine for a shape-independent epilogue like
`gelu` — but it cannot see the row, so a flash-attention rescale first has to
discover which row each component belongs to, and worse, every access spills the
whole tile into a `Function` variable. On SAM 2's attention that took the kernel
from 123 registers to 192, halving resident workgroups per SM. Here the driver
walks its own layout and nothing is spilled.

Anything the callback needs beyond `(row, col, element)` goes in `extras`, which
the instruction appends to the call — `f(row, col, element, extras...)`. A shared
array's `.ptr` and an offset are the shape this was built for: a flash-attention
rescale is `element * cs[base + row]`, and both `cs` and `base` travel that way.
The values in `extras` are taken from the marker call, so they are the real ones
even though `row`, `col` and `element` there are dummies.

`f` must be a **top-level** function, not a closure: a closure is passed an
environment pointer, which is not one of the operand slots. Pass what it would
capture through `extras` instead, or dispatch on a zero-size `Val` parameter
(those cost no argument at all).

**Do not mark `f` `@noinline`.** [`coopmat_perelement_thunk`](@ref) already is,
and it is the function the instruction names; `f` should melt into it. A
`@noinline` callback stays a separate `OpFunction` with `DontInline` control, the
driver honours that, and the per-element loop then pays a real function call per
element — measured at **8.5x**, 0.756 ms against 0.082 for 500 rescales of one
tile across 4096 workgroups. That is the difference between this instruction
being a wash against `getcomp`/`setcomp` and being seven times worse than it.

Check `vk_context().coopmat2.per_element_operations` before compiling a kernel
that uses this: it is NVIDIA-only, and on AMD the `getcomp`/`setcomp` path is
still the right answer.

The `f(...)` call below is a marker, not the work — see `coopmat_perelement_marker`.
"""
@generated function coopmat_perelement(f::F, m::CoopMatrix{T,M,N,U,SC},
                                       extras::Vararg{Any,NE}) where {F,T,M,N,U,SC,NE}
    fname = coopmat_intrinsic_name("perelem", T, M, N, U, SC)
    irty = COOPMAT_IR_TYPE[T]
    ir = """
        declare i32 @$fname(i32, $irty) #0
        define i32 @entry(i32 %m, $irty %p) #0 {
            %r = call i32 @$fname(i32 %m, $irty %p)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    # `compilerbarrier` is what makes the marker survive, and it is not optional.
    # Written plainly, `f(UInt32(0), UInt32(0), zero(T))` is a call to a pure
    # function on compile-time constants, and Julia's concrete evaluation folds
    # it to a literal — `@noinline` does not stop that, it only stops inlining.
    # The callee then has no call site at all, so no `OpFunction`, and
    # `emit_coopmat_call!` reports "did not survive as a call".
    #
    # Deriving the arguments from `m.handle` instead does defeat the fold, and is
    # WRONG: the handle is a fiction that stands for the matrix itself, so
    # `reinterpret(Float32, handle)` reaches the emitter as `OpBitcast %float
    # %coopmat` and `spirv-val` rejects the module ("Cooperative matrix can only
    # be cast to another cooperative matrix"). Block the fold, do not disguise it.
    quote
        probe = coopmat_perelement_thunk(f,
                  Base.compilerbarrier(:const, UInt32(0)),
                  Base.compilerbarrier(:const, UInt32(0)),
                  Base.compilerbarrier(:const, zero($T)),
                  extras...)
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,$T}, m.handle, probe)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
    coopmat_perelement(f, m, o::CoopMatrix) -> CoopMatrix

`m` with `f(row, col, m_element, o_element)` applied to every element — the
per-element op with a **matrix** as its extra operand.

`OpCooperativeMatrixPerElementOpNV` allows a cooperative matrix in the operand
list, and the callback is then handed that matrix's CORRESPONDING ELEMENT rather
than the matrix. `flash_attn_cm2.comp` uses it twice — `coopMatPerElementNV(M,
rowmax, Max, Mold)` is the elementwise maximum of two matrices in ONE pass.

Without it the same result takes three: negate, relu, add. That is not
bookkeeping — an ablation of the flash kernel puts **70% of its time in the
softmax's per-element passes**, for one eightieth of the arithmetic, so the pass
count is the thing that costs.

`o` must have the same shape and scope as `m`; the instruction reads them
element-for-element.

**Why this is a separate method and not another `extras` entry.** The extras of
the scalar form ride on the marker call, where they are real values; here the
instruction needs the matrix's `<id>` while the CALLBACK must be compiled against
an element. So the matrix goes through the `llvmcall` — where the emitter can
resolve its handle to the matrix id — and the marker gets a dummy element of the
right type. One kernel may use both forms; they are different LLVM symbols.
"""
@generated function coopmat_perelement(f::F, m::CoopMatrix{T,M,N,U,SC},
                                       o::CoopMatrix{T,M,N,U2,SC}) where {F,T,M,N,U,U2,SC}
    fname = coopmat_intrinsic_name("perelemm", T, M, N, U, SC)
    irty = COOPMAT_IR_TYPE[T]
    ir = """
        declare i32 @$fname(i32, $irty, i32) #0
        define i32 @entry(i32 %m, $irty %p, i32 %o) #0 {
            %r = call i32 @$fname(i32 %m, $irty %p, i32 %o)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        probe = coopmat_perelement_thunk(f,
                  Base.compilerbarrier(:const, UInt32(0)),
                  Base.compilerbarrier(:const, UInt32(0)),
                  Base.compilerbarrier(:const, zero($T)),
                  Base.compilerbarrier(:const, zero($T)))
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,$T,Int32},
                          m.handle, probe, o.handle)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

"""
`OpCooperativeMatrixReduceNV`'s reduce mask. `Row` and `Column` are bits, so
`RowAndColumn` is their union — which `flash_attn_cm2.comp` uses as
`gl_CooperativeMatrixReduceRowAndColumnNV`, and which is how `Column = 2` is
known without the extension spec to hand. `TwoByTwo` is the remaining bit and is
**unverified**; nothing here uses it yet.
"""
module CoopMatReduce
const Row          = UInt32(1)
const Column       = UInt32(2)
const RowAndColumn = UInt32(3)
const TwoByTwo     = UInt32(4)
end

"""
The combiner the instruction actually names, wrapping the user's `f`.

Same job as [`coopmat_perelement_thunk`](@ref) and the same reason: the SPIR-V
rules fix this function's signature at `(T, T) -> T`, while LLVM is free to
rewrite the signature of a Julia function it can see every caller of. A combiner
that ignores one argument — `smearReduce(x, y) = x`, which the reference uses to
broadcast rather than reduce — would otherwise lose that parameter to
dead-argument elimination and emit a module the validator rejects.
"""
@noinline function coopmat_reduce_thunk(f::F, x::T, y::T) where {F,T}
    coopmat_keepparam(x)
    coopmat_keepparam(y)
    # Result barrier for the same reason as `coopmat_perelement_thunk`'s.
    Base.compilerbarrier(:const, f(x, y))
end

"""
    coopmat_reduce(f, AcceleratedMatrix{T,M,N,U}, m, mask) -> AcceleratedMatrix

Combine `m` along `mask` with the binary function `f`, into a matrix of the
**destination** type — `OpCooperativeMatrixReduceNV`, from
`VK_NV_cooperative_matrix2`.

Two things about the semantics, both taken from `flash_attn_cm2.comp` rather than
from the extension spec, and both easy to guess wrong:

  * **`f` is a binary combiner**, `(x, y) -> z`, not the `(row, col, element)` of
    [`coopmat_perelement`](@ref). A row maximum is `max`.
  * **The result is not a vector.** It has the destination's shape with the
    reduced value repeated along the reduced axis. The reference leans on that
    twice: `rowmax` is `Br x Bc` and gets compared elementwise against the whole
    score tile, and `smearReduce(x, y) = x` reduces nothing at all — it exists so
    a `Br x Bc` matrix can be *resized* to `Br x HSV_pad` with the value smeared.
    That second use also needs a destination whose extent differs, i.e. flexible
    dimensions; a same-shape reduce does not.

The destination type is explicit because it is not derivable from `m`.

`f` must be a **top-level** function, not a closure, for the reason spelled out
on `coopmat_perelement`: a closure is passed an environment pointer and there is
no operand slot for it. Do not mark it `@noinline` — the thunk already is, and a
`DontInline` callback costs a real function call per element.

Check `vk_context().coopmat2.reductions` before compiling a kernel that uses this.
"""
@generated function coopmat_reduce(f::F, ::Type{CoopMatrix{T,M,N,U,SC}},
                                   m::CoopMatrix{S,MM,NN,UU,SC},
                                   ::Val{MASK}) where {F,T,M,N,U,SC,S,MM,NN,UU,MASK}
    fname = coopmat_intrinsic_name("reduce$(UInt32(MASK))", T, M, N, U, SC)
    irty = COOPMAT_IR_TYPE[T]
    ir = """
        declare i32 @$fname(i32, $irty) #0
        define i32 @entry(i32 %m, $irty %p) #0 {
            %r = call i32 @$fname(i32 %m, $irty %p)
            ret i32 %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    # `compilerbarrier` for the same reason as `coopmat_perelement`: without it
    # Julia concrete-evaluates `f(zero(T), zero(T))` to a literal, the callee
    # loses its only call site, no `OpFunction` is emitted, and the emitter
    # reports "did not survive as a call".
    quote
        probe = coopmat_reduce_thunk(f,
                  Base.compilerbarrier(:const, zero($T)),
                  Base.compilerbarrier(:const, zero($T)))
        h = Base.llvmcall(($ir, "entry"), Int32, Tuple{Int32,$T}, m.handle, probe)
        CoopMatrix{$T,$M,$N,$U,$SC}(h)
    end
end

# No shape enumeration here. `GPUCompiler.isintrinsic` accepts the whole
# `_lava_coopmat_` prefix, because the shape rides in the name and the emitter
# parses it back out — enumerating shapes made the GATE decide which shapes
# exist, and coopmat2 FLEXIBLE DIMENSIONS allows any M/N within the device's
# max-dimension limit. A 64x16 matrix (legal here: `flexible_dimensions` is
# enabled at device creation, and no device-reported KHR shape has M != 16) was
# rejected before reaching an emitter that handles it correctly — measured
# 1.1e-6 against CPU once the gate was lifted.
#
# Shapes the device cannot actually run still fail, just later and more
# precisely: the emitter errors on a malformed name, and the driver rejects a
# shape outside its limits. `coopmat_shape(ctx, T, M, N, K)` is the host-side
# query for deciding whether to take this path at all.

