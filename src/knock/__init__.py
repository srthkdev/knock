"""Knock: probe hidden physical properties, then make one flick."""

# Bump when anything an agent can observe or the scoring changes (physics, sound, features, rules).
# 0.1: hand-tuned modal sound, 0.05 s room tail.  0.2: FEM modes, cited materials, damping fitted to
# real recordings, 0.01 s room tail, one-sided feature window.
BENCHMARK_VERSION = "0.2"
