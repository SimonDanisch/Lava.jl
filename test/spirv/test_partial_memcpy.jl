# Tier 1: a memcpy that fills only PART of its destination.
#
# `lower_memcpy!` turns `llvm.memcpy` into a typed load + store, and it typed
# both by the DESTINATION alloca without looking at the copy's length. That is
# right for the common case — Julia copies a whole object — and wrong for the
# one below.
#
# Julia builds a tuple field by field. `(device_array, vec)` is a 32-byte alloca
# filled by a 16-byte copy of the array and then a 12-byte copy of the vector,
# so the FIRST copy fills a prefix. Typed by the destination it became
#
#     %v = load { { ptr, [1 x i64] }, [1 x [3 x float]] }, ptr %arg1_alloca
#
# — 32 bytes read from a 16-byte alloca, and 16 bytes of the destination
# clobbered with whatever followed it. What reached the emitter was a load with
# no member to address: with a pointer at offset 0 it cannot be typed at all
# ("Cannot map opaque pointer type without context"), and after the typepun pass
# the out-of-range half is `OpLoad %uchar` on a struct pointer, which spirv-val
# rejects.
#
# Found through RayMakie's scatter stage, which is where an argument tuple gets
# materialised: indexing an immutable vector at a RUNTIME index forces the tuple
# into memory instead of registers. That is what the shader below does.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, compile_and_disasm
using GeometryBasics

# Top level, not inside the `@testset`: a shader defined in a local scope is a
# closure, and every name it reaches for is a global access GPUCompiler rejects.
#
# `u[i]` with a runtime `i`: an immutable vector cannot be indexed in registers,
# so the argument tuple is written to memory first — which is what produces the
# partial copy.
#
# Through `VertexWrapper`, because THAT is what builds the tuple: it calls
# `F.instance(args...)`, and a stage body written against the intrinsics
# directly never materialises one. The same body compiled bare passes either
# way, which is what made the first attempt at this test worthless.
function partial_copy_body(xs::Lava.LavaDeviceArray{Vec3f,1}, u::Vec3f)
    i = KernelInterface.vertex_index()
    v = xs[i]
    return (position = Vec4f(v[1], v[2], v[3], 1f0), out = Vec3f(u[i], u[i], u[i]))
end

@testset "a memcpy filling part of its destination" begin
    # `validate = true` is the assertion: before the fix this threw, either from
    # spirv-val or from the type mapper, and never produced a module.
    d, _ = compile_and_disasm(Lava.VertexWrapper{typeof(partial_copy_body), ()}(),
                              Tuple{Lava.LavaDeviceArray{Vec3f,1}, Vec3f};
                              stage = :vertex, validate = true)

    # Reaching here at all is the assertion: before the fix `compile_and_disasm`
    # threw, from spirv-val ("OpLoad Result Type %uchar does not match Pointer
    # %N's type") or, when the over-wide load started at a pointer member, from
    # the type mapper. A module came out either way only after the fix.
    @testset "a module comes out" begin
        check(d, "OpEntryPoint Vertex")
    end
end
