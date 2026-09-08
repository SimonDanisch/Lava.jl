# DELETED in phase 1.4: see Mantle/docs/mantle-owns-it.md
#
# The 28 `lava_rt_*` ray-tracing intrinsics. The vendor is in the NAME, which
# CLAUDE.md forbids in a code path, and Hikari imported all thirteen of them by
# name — the single reason a renderer that goes through Mantle for everything
# else still had `import Lava` at the top of a file.
#
# Phase 2.1 declares them in KernelInterface under vendor-free names
# (`rt_trace_ray`, `rt_launch_id_x`, …) and this file comes back as overrides.
