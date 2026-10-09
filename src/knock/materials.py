"""Material constants shared by physics (density) and sound (elastic constants, damping, contact time).

Every number carries its source in `SOURCES`. Keys per material:

    density  kg/m^3   mass in MuJoCo and the FEM scaling sqrt(E/rho)
    E        Pa       Young's modulus (isotropic; for wood the longitudinal modulus E_L)
    nu       -        Poisson ratio (for wood nu_LR)
    E_contact Pa      modulus seen by a ball striking the face (wood: across the grain, E_R)
    alpha    1/s      Rayleigh mass-proportional damping
    beta     s        Rayleigh stiffness-proportional damping; decay rate d = 0.5*(alpha + beta*w^2)
    alpha_lit, beta_lit  Ren, Yeh & Lin (2013) Table I estimates, kept for reference
    tau      s        contact duration of the tap, from Hertz theory (see `hertz_contact_time`)

Densities are deliberately left at the values the committed levels (levels/dev) were generated
with; each is within a few percent of the cited handbook value (noted per material), so that
masses, flick distances and stored oracle answers do not change.
"""
import numpy as np

# ---------------------------------------------------------------- sources
REN2013 = ("Ren, Yeh & Lin, 'Example-Guided Physically Based Modal Sound Synthesis', ACM TOG 32(1) 2013, "
           "Table I (alpha, beta estimated from recordings; d = 0.5*(alpha + beta*w^2))")
REALIMPACT = ("Clarke et al., 'RealImpact: A Dataset of Impact Sound Fields for Real Objects', CVPR 2023: "
              "per-mode decay rates fitted on 1500 recordings of 50 objects as d = a + b*f "
              "(knock real-data check); alpha, beta = least-squares fit of 0.5*(alpha + beta*(2*pi*f)^2) "
              "to a + b*f in relative error over 300-8000 Hz")
CALLISTER = "Callister & Rethwisch, Materials Science and Engineering: An Introduction, 9th ed., App. B (Tables B.1-B.3)"
WOOD_HB = "USDA FPL Wood Handbook, FPL-GTR-282 (2021), ch. 5: Table 5-3a (SG, MOE at 12% MC), Tables 5-1/5-2 (E_R/E_L, nu_LR)"
NEURALSOUND = "NeuralSound (Jin et al., SIGGRAPH 2022) material table, src/classic/fem/femModel.py"
JOHNSON = "K. L. Johnson, Contact Mechanics (CUP 1985), sec. 11.4: Hertz impact time t = 2.87 (m^2/(R E*^2 v))^(1/5)"

# RealImpact fits (a [1/s], b [s]) used for calibration, with 90% bootstrap ranges over objects.
REALIMPACT_FITS = {
    "wood":    dict(a=80.8, b=0.0082, a_ci=(71.3, 94.2), b_ci=(0.0060, 0.0098)),
    "plastic": dict(a=54.8, b=0.0118, a_ci=(36.0, 71.5), b_ci=(0.0094, 0.0150)),
    "metal":   dict(a=5.0, b=0.0028, a_ci=(-0.2, 9.0), b_ci=(0.0019, 0.0051)),
    "glass":   dict(a=52.2, b=0.0020, a_ci=(21.3, 91.2), b_ci=(-0.0017, 0.0052)),
}

# ---------------------------------------------------------------- striker / contact
STRIKER = dict(radius=0.008, mass=0.01, E=207e9, nu=0.30, speed=1.2)  # steel ball, as in world.object_scene
# (speed: middle of the 0.8-1.6 m/s range used by world.simulate_tap)


def hertz_contact_time(E, nu, striker=STRIKER):
    """Hertz contact duration of a sphere hitting a flat face of a much heavier body [JOHNSON]."""
    e_star = 1.0 / ((1 - nu ** 2) / E + (1 - striker["nu"] ** 2) / striker["E"])
    m, R, v = striker["mass"], striker["radius"], striker["speed"]
    return float(2.87 * (m ** 2 / (R * e_star ** 2 * v)) ** 0.2)


# ---------------------------------------------------------------- materials
_RAW = {
    # carbon steel (AISI 1020): E 207 GPa, nu 0.30, rho 7850 [CALLISTER]; density kept at 7800 (-0.6%)
    "metal": dict(density=7800, E=207e9, nu=0.30, E_contact=207e9,
                  alpha=16.14, beta=1.917e-8, alpha_lit=6.3035, beta_lit=2.1160e-8),
    # soda-lime glass: E 69 GPa, nu 0.23, rho 2500 [CALLISTER]
    "glass": dict(density=2500, E=69e9, nu=0.23, E_contact=69e9,
                  alpha=110.7, beta=1.216e-8, alpha_lit=18.301, beta_lit=1.4342e-7),
    # polycarbonate: E 2.38 GPa, nu 0.36, rho 1200 [CALLISTER] (NeuralSound: 2.4 GPa, 0.37, 1190);
    # density kept at 1100 (-8%; ABS is 1050), so sqrt(E/rho) is ~4% above pure PC.
    "plastic": dict(density=1100, E=2.38e9, nu=0.36, E_contact=2.38e9,
                    alpha=140.1, beta=7.552e-8, alpha_lit=52.627, beta_lit=8.7753e-7),
    # white ash at 12% MC: SG 0.60 -> rho ~ 0.60*1.12*1000 = 672 (kept 650, -3%), E_L 12.0 GPa,
    # E_R/E_L 0.125, nu_LR 0.371 [WOOD_HB]. Wood is orthotropic; the FEM treats it as isotropic with
    # E_L, which overestimates cross-grain stiffness (frequencies are an upper estimate).
    "wood": dict(density=650, E=12.0e9, nu=0.371, E_contact=0.125 * 12.0e9,
                 alpha=184.9, beta=5.03e-8, alpha_lit=2.1364, beta_lit=3.0828e-6),
}

SOURCES = {
    "metal": dict(E=CALLISTER, nu=CALLISTER, density=CALLISTER + " (7850; kept 7800)",
                  alpha=REALIMPACT, beta=REALIMPACT, alpha_lit=REN2013, beta_lit=REN2013, tau=JOHNSON),
    "glass": dict(E=CALLISTER, nu=CALLISTER, density=CALLISTER,
                  alpha=REALIMPACT, beta=REALIMPACT, alpha_lit=REN2013, beta_lit=REN2013, tau=JOHNSON),
    "plastic": dict(E=CALLISTER + "; " + NEURALSOUND, nu=CALLISTER + "; " + NEURALSOUND,
                    density=CALLISTER + " (PC 1200, ABS 1050; kept 1100)",
                    alpha=REALIMPACT, beta=REALIMPACT, alpha_lit=REN2013, beta_lit=REN2013, tau=JOHNSON),
    "wood": dict(E=WOOD_HB, nu=WOOD_HB, density=WOOD_HB + " (672; kept 650)", E_contact=WOOD_HB,
                 alpha=REALIMPACT, beta=REALIMPACT, alpha_lit=REN2013, beta_lit=REN2013, tau=JOHNSON),
}

MATERIALS = {}
for _name, _p in _RAW.items():
    _p = dict(_p)
    _p["tau"] = hertz_contact_time(_p["E_contact"], _p["nu"])
    _p["c"] = float(np.sqrt(_p["E"] / _p["density"]))  # bar wave speed sqrt(E/rho), m/s
    _p["source"] = SOURCES[_name]
    MATERIALS[_name] = _p

WALL = 0.003  # shell thickness of hollow objects (m)

# Loose or fixed contents of hollow objects.
CONTENTS = {
    "empty":  dict(mass=0.0),
    "marble": dict(mass=0.005, radius=0.008, count=1, solref="0.002 1", tau=None),
    "beads":  dict(mass=0.0008, radius=0.004, count=8, solref="0.002 1", tau=None),
    "foam":   dict(mass=0.004, radius=0.012, count=1, solref="0.03 0.3", tau=0.004),
    "weight": dict(mass=0.25),  # glued to the floor of the box: no rattle
}
