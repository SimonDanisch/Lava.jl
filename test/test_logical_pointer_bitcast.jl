# A logical pointer may never be OpBitcast.
#
# Vulkan mandates the Logical addressing model, under which `OpBitcast` may
# neither produce nor consume a pointer. spirv-val rejects it outright:
#
#     Instruction may not have a logical pointer operand
#       %99 = OpBitcast %_ptr_Workgroup__arr_float_uint_128 %98
#
# The construct is a clamped ternary over shared memory,
#     l = lid == 1 ? sh[lid] : sh[lid - 1]
# which LLVM lowers to a select between the array base pointer and an element
# pointer. Reconciling those two operand types by bitcasting the element pointer
# UP to the array pointer type is illegal; drilling the aggregate DOWN to its
# first element with OpAccessChain is the legal direction.
#
# ── Why this test is shaped the way it is ─────────────────────────────────────
#
# Two false starts, both worth stating so nobody repeats them:
#
# 1. A pure runtime assertion does not work. The driver ACCEPTS the invalid
#    module and returns the correct answer. `test_select_width_mismatch.jl` and
#    `test_shared_memory_stress.jl` both passed for as long as the bug existed;
#    only a validation-layer device (`DebugConfig(validation = true)`), which is
#    not the default, ever objected.
#
# 2. A Tier 1 `compile_and_disasm` check on an equivalent hand-written function
#    does not work either — it emits a VALID module. The illegal bitcast depends
#    on the exact lowering KernelAbstractions' `@kernel` wrapper produces, not on
#    the ternary alone. A Tier 1 version of this test passed with the fix
#    reverted, i.e. it asserted nothing.
#
# So: launch the real KA kernel, capture what the compiler actually emitted via
# LAVA_SPIRV_DUMP_DIR, and assert on that. Verified to fail with the fix
# disabled.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "no OpBitcast on a logical pointer" begin
    M = 128

    @kernel function lpb_clamped_ternary!(out)
        lid = @index(Local)
        sh = @localmem Float32 (128,)
        @inbounds sh[lid] = Float32(lid)
        @synchronize()
        @inbounds begin
            l = lid == 1   ? sh[lid] : sh[lid - 1]
            r = lid == 128 ? sh[lid] : sh[lid + 1]
            out[lid] = (l + sh[lid] + r) / 3f0
        end
    end

    dumpdir = mktempdir()
    prev = get(ENV, "LAVA_SPIRV_DUMP_DIR", nothing)
    ENV["LAVA_SPIRV_DUMP_DIR"] = dumpdir
    try
        out = Lava.LavaArray(zeros(Float32, M))
        lpb_clamped_ternary!(Lava.LavaBackend())(out; ndrange = M, workgroupsize = M)
        KA.synchronize(Lava.LavaBackend())

        # The answer must still be right — the fix reconciles the select's
        # operands, it does not change what the kernel computes.
        v = Float32.(1:M)
        ref = [ (i == 1 ? v[i] : v[i-1]) + v[i] + (i == M ? v[i] : v[i+1]) for i in 1:M ] ./ 3f0
        @test Array(out) ≈ ref

        spvs = filter(f -> endswith(f, ".spv"), readdir(dumpdir; join = true))
        @test !isempty(spvs)          # a dump we never wrote would assert nothing

        for f in spvs
            words = reinterpret(UInt32, read(f))
            # Walk the instruction stream: OpBitcast = 124. Its result type is the
            # first operand, so flag any whose result type id was declared by
            # OpTypePointer = 32. Done on the binary rather than on disassembler
            # text so it does not depend on spirv-dis being installed.
            ptr_types = Set{UInt32}()
            bad = 0
            i = 6                      # first word past the 5-word header
            while i <= length(words)
                wc = words[i] >> 16
                op = words[i] & 0xFFFF
                wc == 0 && break
                op == 32 && push!(ptr_types, words[i+1])          # OpTypePointer result id
                op == 124 && words[i+1] in ptr_types && (bad += 1) # OpBitcast to a pointer
                i += Int(wc)
            end
            @test bad == 0
        end
    finally
        prev === nothing ? delete!(ENV, "LAVA_SPIRV_DUMP_DIR") : (ENV["LAVA_SPIRV_DUMP_DIR"] = prev)
        rm(dumpdir; recursive = true, force = true)
    end
end
