# SPIR-V intrinsics in Lava — what is wired, what is not

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


Companion to `dev/JuliaVision/plans/kernels-to-port.md`. That file ranks *kernels* by what they would
buy; this one is the layer underneath: which SPIR-V instructions Lava can
actually emit, and what each open item needs before it can start.

Written 2026-08-02 from an audit, not from memory. Reproduce it with the script
at the bottom — the counts below are `229` opcodes and `36` capabilities declared
in `src/compiler/spirv/module.jl`, of which `172` and `30` are reachable.

**The distinction that matters and is easy to miss:** a constant declared in
`module.jl`, a capability the *device* reports, and an instruction Lava can
*emit* are three different things. `ctx.coopmat2.reductions` is `true` today and
nothing can emit a reduction. A kernel that gates on that field compiles and then
fails in the emitter. Check this file, not the device.

## Rule 0 — the driver is not the suspect

Lava is a few months old and was written fast. The NVIDIA Vulkan driver is years
old and ships to millions of machines. When a kernel here misbehaves, the prior
is overwhelmingly that **the bug is ours**. "Driver bug" is a conclusion that
needs a mountain of evidence, never a working hypothesis.

This is not a style preference. It has been wrong here, expensively, and
recently. On 2026-08-02 a workgroup above 256 threads "silently ran fewer
invocations than the shader declared" — recorded as a hardware lane cap for
months, with a page of supporting evidence: `spirv-val` passed, the dump showed
`LocalSize 512`, `VK_KHR_pipeline_executable_properties` reported identical
register counts for a body that failed and one that worked, and it was
body-dependent so no statistic could predict it. Every one of those statements
was true. The conclusion was wrong. The cause was one line of **ours**:
`hash(spirv_bytes)` samples a large `Vector`, so two modules differing at a
single byte collided in our pipeline cache and the 512-thread launch ran the
256-thread shader.

**Consequence for anything currently labelled a driver bug: it is a suspect, not
a finding.** The two open items so labelled — `OpUDiv` in a shared-store index,
and narrow-index + rank≥3 `Extruded` — have exactly the shape the workgroup cap
had: root-caused, mitigated, "cannot be settled on this hardware". Re-open both
against our own compiler before anyone reports anything upstream.

Instruments, in this order, before the word "driver" is used at all:

1. `spirv-val --target-env vulkan1.3`, then GPU-assisted validation
   (`vk_reset_device!(debug = DebugConfig(gpu_av = true))`). Cheap, and neither
   is run by default.
2. Hunt undefined behaviour **in our own output**: out-of-range access,
   uninitialised reads, missing `NonPrivatePointer` or memory semantics on shared
   access, and signed-vs-unsigned comparisons that LLVM canonicalised under
   `nuw`/`nsw` flags our index arithmetic supplied. A module can be valid and
   still depend on something the spec leaves undefined.
3. **Write the same kernel in GLSL, compile it with `glslangValidator`, and run
   that module through the same Lava dispatch.** If glslang's SPIR-V is correct
   and ours is not, the bug is ours and the disassembly diff localises it. This
   is the strongest instrument available, it needs no second machine, and it is
   the same technique that produced the coopmat2 opcodes in section D.
4. Only if all three come back clean *and* the behaviour reproduces on a second
   NVIDIA device is "driver" even on the table — and it is still more likely to
   be our reliance on something unspecified than a defect in theirs.

A corollary worth stating because it inverts an easy assumption: when our module
behaves *differently* on another vendor or another NVIDIA chip, that difference
is evidence about **our** code — it means we are depending on unspecified
behaviour — not evidence against the driver.


---

## A. Wired end to end — Julia binding, emitter case, test

### Compute builtins (`src/runtime/intrinsics.jl`)

| Julia | SPIR-V |
|---|---|
| `lava_global_invocation_id`, `_local_invocation_id`, `_workgroup_id`, `_num_workgroups`, `_workgroup_size` | the `<3 x i32>` `BuiltIn` inputs |
| `lava_local_invocation_index` | `LocalInvocationIndex` |
| `lava_subgroup_size`, `lava_subgroup_local_id` | `SubgroupSize`, `SubgroupLocalInvocationId` — **new 2026-08-02** |
| `lava_workgroup_barrier` | `OpControlBarrier Workgroup Workgroup` |

### Subgroup (`src/device/subgroup.jl`)

| Julia | SPIR-V | capability |
|---|---|---|
| `subgroup_add/mul/min/max/and/or/xor` | `OpGroupNonUniform{I,F}Add` … `BitwiseXor`, `GroupOperation Reduce` | `GroupNonUniformArithmetic` |
| `subgroup_inclusive_scan_add`, `subgroup_exclusive_scan_add` | same, `InclusiveScan` / `ExclusiveScan` | same |
| `subgroup_elect` | `OpGroupNonUniformElect` | `GroupNonUniform` |
| `subgroup_broadcast_first` | `OpGroupNonUniformBroadcastFirst` | `GroupNonUniform` |
| `subgroup_all`, `subgroup_any` | `OpGroupNonUniformAll/Any` | `GroupNonUniformVote` |
| `subgroup_shuffle`, `_xor` | `OpGroupNonUniformShuffle`, `ShuffleXor` | `GroupNonUniformShuffle` |
| `subgroup_shuffle_up`, `_down` | `ShuffleUp`, `ShuffleDown` | `GroupNonUniformShuffleRelative` |
| `subgroup_broadcast` | `OpGroupNonUniformBroadcast` | `GroupNonUniformBallot` |
| `subgroup_rotate` | `OpGroupNonUniformRotateKHR` | `GroupNonUniformRotateKHR` + `SPV_KHR_subgroup_rotate` |

The shuffle family and rotate are **new 2026-08-02**; the opcodes had been
declared since the beginning with no emitter case and no binding.

### Cooperative matrix (`src/device/coopmat_intrinsics.jl`)

`coopmat_load` / `loadw` / `loadw2`, `coopmat_store` / `storew`, `coopmat_zero`,
`coopmat_muladd`, `coopmat_convert`, `coopmat_length`, `coopmat_getcomp` /
`setcomp`, and **new 2026-08-02** `coopmat_mul` (`OpFMul`, component-wise, plain
KHR), `coopmat_add` (`OpFAdd`, likewise) and `coopmat_perelement`
(`OpCooperativeMatrixPerElementOpNV`).

`coopmat_add` exists so a GEMM accumulator can start from `bias + residual`
instead of zero — the bias at **stride 0** so one vector broadcasts across the
tile, the residual at the destination's leading dimension — which would make a
transformer's residual add the tensor cores' own accumulate. The GEMM plumbing
is not written: of SAM 2's 98 residual adds the 51 that are structurally
foldable turned out to be the *cheap* half, so the measurement did not justify
the surgery. The instruction is portable and tested either way.

**New 2026-08-11: scope is a type parameter.** `CoopMatrix{T,M,N,Use,Scope}` with
two aliases — `AcceleratedMatrix` (subgroup, the portable KHR one, and what every
existing kernel spells) and `WorkgroupMatrix` (`Scope = Workgroup`, which
`VK_NV_cooperative_matrix2` enables). One matrix then spans all `NT` invocations
instead of one subgroup: a `32 x 96` fp32 accumulator is 12 components a lane at
256 invocations against 96 at subgroup scope. Every operation requires its
operands to agree on scope, so mixing them is a method error at the call site
rather than a module the driver rejects.

The SPIR-V difference is **one constant** — `Scope = 2` instead of `3` on
`OpTypeCooperativeMatrixKHR` — and nothing else: same capabilities, same
extensions, same instructions, verified against glslang in
`test/glsl/workgroup_scope_coopmat.comp`.

**What the GEMM measurement then said about tensor addressing itself**, since it
is the strongest evidence this file has on the subject: a tensor-addressed load
costs **1.5x to 4x the registers** of the equivalent `@localmem`-staged code, in
all four kernel structures tried (workgroup scope at five tilings, subgroup scope
with a 4x4 register block, with and without unrolling). The driver targets 128
registers and caps at 255, so that is decisive — `src/array/gemm_cm2.jl` has the
table. Attention still wins with it because there the gain is structural rather
than arithmetic. Two more things worth carrying: an **unclamped** layout is 2.8x
SLOWER here than a clamping one — the inverse of `mul_mm_cm2.comp`'s own fast
path — and batching several loads before their products extends live ranges the
scheduler cannot shorten.

Two tensor-addressing gaps closed alongside it, both wanted by the GEMM:
`tensor_setstride` (`OpTensorLayoutSetStrideNV`, 5374 — without it a layout
describes a PACKED tensor, and every attention operand is a permuted view of one
block) and a **viewed store**, `tensor_store(m, addr, layout, view)`, which is
how `mul_mm_cm2.comp` writes an `N x M` destination from an `M x N` accumulator
without a staging pass. Each has an MWE that keeps the un-viewed / un-strided
form beside it as a negative control, because a packed test array passes either
way and proves nothing. What is NOT free is the shape: legal
`(M, N, K)` multiples depend on the workgroup size, and the device reports the
table through
`get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv`
(RTX 4000 Ada, fp16 x fp16 -> fp32: 16/16/16 at 32 and 64 invocations, 32/16/16
at 128, 32/32/16 at 256). Launch a shape off that table and pipeline creation
fails.

Extents registered: `16x16`, `16x8`, and `8x8`. Nothing ships `8x8` — it is what
**lavapipe** offers, and having it lets a cooperative-matrix reproducer run on a
second, independent consumer. That is what settled `test_shared_index_division.jl`.

Component types: **`f16 f32 f64 i8 u8 i32 u32`**. Note `i8`/`u8` are wired, and
the device offers int8 at **K32** — so the "does K32 double throughput" question
is answerable with no new code (see `dev/JuliaVision/plans/kernels-to-port.md` item 20).

### Other

`_lava_glsl_*` → `GLSL.std.450`; `_lava_debug_printf_*` → `NonSemantic.DebugPrintf`;
atomics via LLVM `atomicrmw` → `OpAtomic*` / `OpAtomicFAddEXT`; ray-query and
ray-tracing intrinsics (`lava_rt_*`, `lava_ray_query_*`), graphics stage I/O
(`_lava_gfx_*`, `_lava_geom_*`).

---

## B. Emitter has it, no Julia binding — small patches

| SPIR-V | note |
|---|---|
| `OpGroupNonUniformBallot` | emitter case exists, nothing can reach it. Its own comment is the warning: ballot yields a **uvec4**, and a wave64 subgroup (RDNA3) puts lanes 32–63 in `.y`, so a binding that reads only the low dword silently drops half the subgroup |
| scans for `mul/min/max/and/or/xor` | the `GroupOperation` plumbing is generic; only `add` is generated in `subgroup.jl` |

---

## C. Declared in `module.jl`, emits nothing — the trap

These read as "supported" when grepping and are not. Six capabilities and 57
opcodes are in this state; the ones that matter here:

| constant | why it is there | status |
|---|---|---|
| `CooperativeMatrixReductionsNV`, `OpCooperativeMatrixReduceNV` (5366) | I added them 2026-08-02 from the glslang reference dump while implementing per-element ops | **declared only** — blocks item 17 |
| `GroupNonUniformClustered`, `GroupNonUniformQuad` | never used | no emitter case, no binding |
| `OpGroupNonUniformLogicalAnd/Or/Xor` | boolean reductions | no emitter case |
| `OpMemoryBarrier` | Lava only emits `OpControlBarrier` | **prerequisite for item 21** — a spin-wait needs an acquire/release memory barrier separate from the control barrier |
| `OpBitFieldInsert/SExtract/UExtract`, `OpSNegate`, `OpFwidth`, `OpTypeMatrix`, `OpTypeRuntimeArray`, `OpTypeSampler`, `OpAtomicLoad/Store/IIncrement/IDecrement`, `OpInBounds*`, `OpImageSampleExplicitLod`, `OpLogicalEqual`, `OpNop`, `OpSource`, `OpMemberName`, `OpMemoryModel` | assorted | unused, harmless |
| ~30 `OpHitObject*NV`, `OpReorderThreadWithHintNV`, `OpExecuteCallableKHR`, `OpReportIntersectionKHR`, `OpRayQueryGenerateIntersectionKHR` | SER / RT surface declared ahead of use | out of scope here |

---

## D. Not present at all — grouped by the item that needs it

### ~~Item 17 — coopmat2 reductions~~ — **DONE 2026-08-03**, see section A

`coopmat_reduce`, `CoopMatReduce.{Row,Column,RowAndColumn,TwoByTwo}`,
`test_coopmat_reduce.jl` (5 asserts). It was the cheapest of the five as
predicted: a new `op ==` branch plus a binding, reusing the marker /
`coopmat_keepparam` / thunk machinery that `coopmat_perelement` had already paid
for.

Three things it established that the notes had wrong or open, all measured:

- **Subgroup scope is accepted.** Every use in `flash_attn_cm2.comp` is
  `gl_ScopeWorkgroup`, and if the driver had refused a `Subgroup`-scoped reduce
  this item would have needed workgroup-scope matrices first — a much larger
  change, and it would have taken K3 with it. It does not refuse.
- **The callback is a binary combiner** `(x, y) -> z`, not the
  `(row, col, element)` of the per-element op.
- **The combine order is unspecified per output element**, so `f` must be
  associative and commutative. `max` and `+` are; the reference's
  `smearReduce(x, y) = x` is not, and on a row of `200..215` it returns
  `[200..207, 200..207]` — the whole row *is* reduced (a `sum` returns the exact
  full-row total) but different output positions combine it in different orders.
  That does not contradict the reference: it only ever hands `smearReduce` a
  matrix already uniform along the row, where it reduces nothing and exists purely
  to RESIZE. Verified both ways.

Mask values: `Row = 1`, `Column = 2`, `RowAndColumn = 3` — the reference uses the
union, which is how `Column` is known without the spec. `TwoByTwo = 4` is the
remaining bit and is **unverified**.

The resize half of the reference's usage — `Br x Bc` reduced into `Br x HSV_pad`
— still needs **item 18**, flexible dimensions. A same-shape reduce does not.

### Item 18 — flexible dimensions

- capability `CooperativeMatrixFlexibleDimensionsNV`, not declared
- `emit_coopmat_type!` currently always emits the fixed KHR type; the shape check
  in `coopmat_shape` matches against the driver's 15 advertised combinations,
  all `M16`, and would have to be relaxed rather than matched

### Item 19 — tensor addressing / block loads

Nothing exists. The largest missing surface: `OpTypeTensorLayoutNV`,
`OpTypeTensorViewNV`, `OpCreateTensorLayoutNV`, `OpTensorLayoutSetDimensionNV`,
`OpCooperativeMatrixLoadTensorNV` / `StoreTensorNV`, plus new SPIR-V *types*,
which is a bigger change to `SPIRVTypeContext` than any of the others.

### Item 20 — fp8 (and bf16)

Verified by assembling and validating a real module against `vulkan1.3` with our
own spirv-tools, so these numbers are known-good rather than recalled:

- extension `SPV_EXT_float8`
- capabilities `Float8EXT` = **4212**, `Float8CooperativeMatrixEXT` = **4213**
- `OpTypeFloat 8 <encoding>` with `Float8E4M3EXT` = **4214**, `Float8E5M2EXT` = **4215**
- device: `VK_EXT_shader_float8` present, `shaderFloat8` and
  `shaderFloat8CooperativeMatrix` both true — **Lava does not request it**
- `VK_KHR_shader_bfloat16` likewise present and unrequested (bf16 is K16, so it
  buys range, not speed)

Lava needs: the device request; `emit_type_float!` extended to carry an FP
encoding operand; component-type cases; and a 1-byte Julia primitive type with
host-side conversion. **Not** an fp8 arithmetic type — a GEMM loads fp8, muladds,
accumulates fp32 and stores fp32, so no fp8 value ever appears in LLVM IR.

### Item 21 — maximal reconvergence

- extension `SPV_KHR_maximal_reconvergence`, execution mode
  `MaximallyReconvergesKHR` — neither declared nor emitted
- also needs `OpMemoryBarrier` (section C) for the acquire/release around a
  shared-flag spin-wait

The *device* feature is enabled (Lava `955e0b8`), which is necessary and not
sufficient: without the execution mode in the module, the reconvergence guarantee
does not apply and the spin-wait is not well-defined.

### Cooperative vector — no item yet

`VK_NV_cooperative_vector` is enabled on the device with zero SPIR-V support
(`OpCooperativeVectorMatMulNV` and friends). Left unlisted in
`dev/JuliaVision/plans/kernels-to-port.md` because item 1 (flash-decoding) addresses the same shape
portably and should be tried first.

---

## E. How to add one — and the three traps

Adding a plain instruction is mechanical: a constant in `module.jl`, a case in
the relevant emitter file, a `@generated` llvmcall stub whose LLVM function name
the emitter pattern-matches, a `push!(KNOWN_INTRINSICS, ...)`, and a test.
`coopmat_mul` is the smallest complete example (~30 lines total).

**Get the reference from glslang, do not recall opcode numbers.**
`glslangValidator --target-env vulkan1.3 -S comp x.comp -o x.spv`, then parse the
words in Python. That is where 5366 / 5369 / 5430 / 5432 came from, and it also
showed that glslang emits a per-element callback as `OpFunction None` — which
turned out to be the whole performance story. Local glslang is 16.4.0 and does
**not** know fp8; for that, assemble SPIR-V text with the jll's `spirv-as` and
validate it, which is how section D's fp8 numbers were obtained.

An instruction that takes a **function operand** is the hard case, and all three
traps below cost real time on `OpCooperativeMatrixPerElementOpNV`:

1. **The marker call gets constant-folded.** There is no way to pass an LLVM
   function to `Base.llvmcall`, so the callback is called once with dummy
   arguments purely to put it in the call graph. `f(0, 0, zero(T))` is a pure
   call on constants and Julia's *concrete evaluation* folds it to a literal —
   `@noinline` does not stop that. Use `Base.compilerbarrier(:const, x)` on each
   argument. Deriving the arguments from the coopmat handle instead defeats the
   fold and is wrong: the handle *is* the matrix, so it reaches the emitter as
   `OpBitcast %float %coopmat` and `spirv-val` rejects the module.
2. **LLVM rewrites the callee's signature.** SPIR-V fixes it; LLVM does not know
   that. Dead-argument elimination removes an unused `col`; interprocedural
   constant propagation removes a `@localmem` pointer it proved constant. Do not
   guess which survive — pin every parameter with `coopmat_keepparam`, a call to
   an undefined function that the emitter drops.
3. **`@noinline` on the user's callback costs 8.5x, silently.** It becomes SPIR-V
   `DontInline`, the driver honours it, and the per-element loop pays a real
   function call per element: 0.756 ms against 0.082 for the same work, with no
   correctness symptom at all. Only Lava's thunk is `@noinline`; the emitter marks
   resolved callbacks `Inline` (`perelement_callbacks`).

And the one that is not about intrinsics at all but bit hardest: **a kernel that
skips work looks like a speed-up.** Assert output coverage, not just values — see
`gpu-measurement-traps`.

---

## Reproducing this audit

```bash
cd dev/Lava
# opcodes declared vs emitted
grep -oE 'const Op[A-Za-z0-9]+' src/compiler/spirv/module.jl | sed 's/const //' | sort -u > /tmp/d
for o in $(cat /tmp/d); do grep -qrE "Op\.$o\b" src/compiler/ && echo $o; done > /tmp/u
comm -23 /tmp/d /tmp/u            # declared, never emitted

# capabilities, same shape
sed -n '/^module Cap/,/^end/p' src/compiler/spirv/module.jl \
  | grep -oE 'const [A-Za-z0-9]+' | sed 's/const //' | sort -u > /tmp/dc
for c in $(cat /tmp/dc); do grep -qrE "Cap\.$c\b" src/compiler/ && echo $c; done > /tmp/uc
comm -23 /tmp/dc /tmp/uc

# what the emitter dispatches on
sed -n '/^function emit_call!/,/^function emit_lava_glsl!/p' src/compiler/spirv/emit.jl \
  | grep -oE '"[a-z_0-9]+"'
```

Device-side capability, which is a *different* question, comes from
`Lava.vk_context()`: `coopmat_available`, `coopmat_shapes`, `coopmat2`,
`coopvec_available`, `maximal_reconvergence_available`,
`subgroup_uniform_control_flow_available`, `subgroup_rotate_available`.
