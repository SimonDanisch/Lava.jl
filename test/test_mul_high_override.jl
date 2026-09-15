# ── The i128-free mul_high override must stay attached to Base's real name ──
#
# `SignedMultiplicativeInverse`/`UnsignedMultiplicativeInverse` — what `div` by a
# loop-invariant divisor lowers to — take the high half of a 64x64 multiply via
# `widen`, i.e. Int128. SPIR-V has no 128-bit integer type, so Lava overlays that
# function with a 32-bit decomposition.
#
# The overlay is attached by NAME, and the name is a Base internal that moves:
# Julia <=1.12 spells it `Base.MultiplicativeInverses._mul_high`, Julia 1.13
# renamed it to `Base.mul_hi` and moved it to int.jl, generalised to
# `T<:Integer`. An overlay on a name Base no longer has is not an error — the
# `@overlay` simply defines a method nobody calls, the compiler reaches the
# widening original, and the failure surfaces far away as an i128 type the
# backend cannot lower. So pin two things: that the name still resolves, and
# that the replacement computes what the original does.

using Random: MersenneTwister

@testset "mul_high override (no i128)" begin
    # 1. Whichever spelling this Julia has, Lava must overlay THAT one.
    target = isdefined(Base, :mul_hi) ? Base.mul_hi :
                                        Base.MultiplicativeInverses._mul_high
    mt = Lava.lava_method_table
    for T in (UInt64, Int64)
        sig = Tuple{typeof(target),T,T}
        ms = Base._methods_by_ftype(sig, mt, -1, Base.get_world_counter())
        @test ms !== nothing && ms !== false && !isempty(ms)
        # and it must be Lava's, not Base's leaking through
        @test all(m -> parentmodule(m.method) === Lava, ms)
    end

    # 2. The decomposition agrees with Base's widening version.
    @test Lava._mul_high_u64(UInt64(0), UInt64(0)) == 0
    @test Lava._mul_high_i64(Int64(0), Int64(0)) == 0

    edges_u = UInt64[0, 1, 0xFFFFFFFF, UInt64(1) << 32, UInt64(1) << 63,
                     typemax(UInt64) - 1, typemax(UInt64)]
    for a in edges_u, b in edges_u
        @test Lava._mul_high_u64(a, b) == target(a, b)
    end

    edges_i = Int64[0, 1, -1, Int64(1) << 32, -(Int64(1) << 32),
                    typemin(Int64), typemax(Int64)]
    for a in edges_i, b in edges_i
        @test Lava._mul_high_i64(a, b) == target(a, b)
    end

    rng = MersenneTwister(0xBEEF)
    @test all(1:20_000) do _
        a, b = rand(rng, UInt64), rand(rng, UInt64)
        Lava._mul_high_u64(a, b) == target(a, b)
    end
    @test all(1:20_000) do _
        a, b = rand(rng, Int64), rand(rng, Int64)
        Lava._mul_high_i64(a, b) == target(a, b)
    end

    # 3. End-to-end: `div` by a SignedMultiplicativeInverse is the Base path that
    #    reaches mul_high. With the overlay attached it emits valid SPIR-V using
    #    only 64/32/8-bit integers; without it the module needs an i128 the
    #    backend cannot lower.
    SMI = Base.MultiplicativeInverses.SignedMultiplicativeInverse
    function _divinv!(out, inv)
        @inbounds out[1] = Int32(div(Int64(1234567), inv))
        return nothing
    end
    res = Lava.lava_compile_gpu(_divinv!,
        Tuple{Lava.LavaDeviceArray{Int32,1},SMI{Int64}};
        workgroup_size = (64, 1, 1), validate = true)
    dis = Lava.disassemble_spirv(res.spirv_bytes)
    @test !occursin("OpTypeInt 128", dis)
end

