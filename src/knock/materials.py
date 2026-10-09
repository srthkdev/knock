"""Material constants shared by physics (density) and sound (modes, damping)."""

# c: bending-wave speed scale (m/s), a + b*f: modal decay rate (1/s),
# tau: contact duration (s), density: kg/m^3.
MATERIALS = {
    "metal":   dict(c=5200, a=2.5, b=0.0012, tau=0.00008, density=7800),
    "glass":   dict(c=5400, a=6.0, b=0.0025, tau=0.00010, density=2500),
    "plastic": dict(c=1700, a=35.0, b=0.012, tau=0.00060, density=1100),
    "wood":    dict(c=2600, a=55.0, b=0.030, tau=0.00040, density=650),
}

WALL = 0.003  # shell thickness of hollow objects (m)

# Loose or fixed contents of hollow objects.
CONTENTS = {
    "empty":  dict(mass=0.0),
    "marble": dict(mass=0.005, radius=0.008, count=1, solref="0.002 1", tau=None),
    "beads":  dict(mass=0.0008, radius=0.004, count=8, solref="0.002 1", tau=None),
    "foam":   dict(mass=0.004, radius=0.012, count=1, solref="0.03 0.3", tau=0.004),
    "weight": dict(mass=0.25),  # glued to the floor of the box: no rattle
}
