"""Yantra: AI agents build, test and fix physical systems; packs: factory, cube lab."""

# Bump when anything an agent can observe or the scoring changes (physics, sound, features, rules).
# 0.1: hand-tuned modal sound, 0.05 s room tail.  0.2: FEM modes, cited materials, damping fitted to
# real recordings, 0.01 s room tail, one-sided feature window.  0.3: yantra.core episodes; pushes and
# the act require a prediction of their result, scored separately from success.
BENCHMARK_VERSION = "0.3"
