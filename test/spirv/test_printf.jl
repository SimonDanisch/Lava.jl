# Tier 1: on-kernel printing through NonSemantic.DebugPrintf.
#
# `@lava_printf` takes a C format string; `KernelAbstractions.@print` takes
# literals and values and has the specifiers chosen from the argument types
# (`device/printf.jl`). Both emit `OpExtInst ... DebugPrintf` with the format as
# an `OpString`, and the module has to pass spirv-val.
#
# Moved here from Mantle's `test/vulkan/test_lava_printf.jl`. Its device half
# read the printed lines back through the validation layer's debug-printf, which
# resets the device; Mantle's `test/test_kernel_print.jl` runs printing kernels
# on every backend and checks that they still compute.

using Test
import KernelAbstractions
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, compile_and_disasm

@testset "printf" begin
    @testset "@lava_printf" begin
        # Format + args of several widths exercise i32 / i64 / float / double operands.
        function pf_dev(out::Lava.LavaDeviceArray{Float32,1})
            i = Lava.lava_global_invocation_id_x()
            @lava_printf "i=%d u=%u big=%ld f=%f d=%lf\n" Int32(i) UInt32(i) Int64(i)*7 out[i] Float64(i)
            @inbounds out[i] = Float32(i)
            return nothing
        end
        d, _ = compile_and_disasm(pf_dev, Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "SPV_KHR_non_semantic_info")    # extension declared
        check(d, "NonSemantic.DebugPrintf")      # ext-inst set imported
        check(d, "OpString")                     # format string present
        check(d, "i=%d u=%u big=%ld f=%f d=%lf") # exact format recovered

        # No-arg format must also compile + validate.
        function pf_noarg(out::Lava.LavaDeviceArray{Float32,1})
            Lava.lava_global_invocation_id_x()
            @lava_printf "hello from a kernel\n"
            return nothing
        end
        d, _ = compile_and_disasm(pf_noarg, Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "hello from a kernel")
    end

    @testset "KernelAbstractions.@print" begin
        # The portable KA API: string literals interleaved with values. Lava's
        # __print override auto-selects specifiers from the arg types and routes
        # through the same DebugPrintf path.
        function kp_dev(out::Lava.LavaDeviceArray{Float32,1})
            i = Lava.lava_global_invocation_id_x()
            KernelAbstractions.@print("tid=", UInt32(i), " big=", Int64(i)*3,
                                      " val=", out[i], "\n")
            @inbounds out[i] = Float32(i)
            return nothing
        end
        d, _ = compile_and_disasm(kp_dev, Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "NonSemantic.DebugPrintf")
        # literals + auto-selected specifiers (u32→%u, i64→%ld, f32→%f) assembled in order
        check(d, "tid=%u big=%ld val=%f")
    end
end
