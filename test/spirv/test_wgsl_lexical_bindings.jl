using Test, Lava

@testset "WGSL lexical result bindings" begin
    body = """
        v1 = 1u;
        if (condition) {
            v2 = v1 + 1u;
            v3 = 0u;
            v6 = 1u;
            v7 = v2 + 1u;
            store(v7);
        } else {
            v3 = v6;
        }
        v4 = aggregate;
        v4.m0 = v2;
        v5 = 1u;
        store(v3 + v5);
    """
    # Preserve the entry-point indentation removed by the Julia string literal.
    lines = ["    " * line for line in split(body, '\n'; keepempty=false)]
    ids = Set(UInt32.(1:7))
    results = Dict(id => Lava.SpvInst(0x0080, UInt32[]) for id in ids)
    results[0x00000004] = Lava.SpvInst(0x0052, UInt32[]) # OpCompositeInsert
    results[0x00000005] = Lava.SpvInst(0x00f5, UInt32[]) # OpPhi
    bindings = Lava.lexical_result_bindings(lines, ids, results)
    @test Set(keys(bindings)) == Set(UInt32[1, 7])

    out = IOBuffer()
    Lava.write_wgsl_body(out, lines, bindings, Dict(id => "u32" for id in ids))
    wgsl = String(take!(out))
    @test occursin("    let v1: u32 = 1u;", wgsl)
    @test occursin("        let v7: u32 = v2 + 1u;", wgsl)
    @test occursin("        v2 = v1 + 1u;", wgsl)
    @test occursin("    v4.m0 = v2;", wgsl)
end
