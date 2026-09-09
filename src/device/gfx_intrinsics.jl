# DELETED in phase 1.4: see Mantle/docs/mantle-owns-it.md
#
# 39 device-side graphics intrinsics: `vertex_index`, `instance_index`, the
# `frag_coord` family, `set_position!`, `set_point_size!`, `dFdx`/`dFdy`,
# `emit_vertex!`/`end_primitive!`/`primitive_id_in`, `sample_texture_2d`, the
# `gfx_input`/`gfx_output` varying access and the geometry-stage inputs.
#
# They were defined here as Lava's OWN functions, so Mantle declared the same
# names separately and `MantleVulkanExt` bridged the two. Phase 2.1 declares
# them once in KernelInterface — which Lava already depends on — and this file
# comes back as `@lava_device_override KernelInterface.<name>`, overriding
# rather than defining.
