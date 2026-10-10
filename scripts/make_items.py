"""Build toy item assets from Keenan Crane's CC0 models (Spot, Bob, Blub).

For each model: rotate into the sim frame (x forward, z up), scale to a real toy size, centre on the
solid's centre of mass. Writes primitive collision shapes (MuJoCo) to src/yantra/packs/assets/items.json
and the textured render mesh to web/public/assets/crane/<name>.json (+ texture png).
"""
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path("assets_src/crane")
MODELS = {  # obj, texture, length (m) along x after rotation
    "spot": ("spot/spot/spot_triangulated.obj", "spot/spot/spot_texture.png", 0.20),
    "bob": ("bob/bob/bob_tri.obj", "bob/bob/bob_diffuse.png", 0.18),
    "blub": ("blub/blub/blub_triangulated.obj", "blub/blub/blub_texture.png", 0.22),
}


def load_obj(p):
    V, T, faces = [], [], []
    for line in open(p):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            V.append([float(x) for x in s[1:4]])
        elif s[0] == "vt":
            T.append([float(x) for x in s[1:3]])
        elif s[0] == "f":
            faces.append([tuple(int(i) - 1 if i else -1 for i in (c.split("/") + [""])[:2]) for c in s[1:]])
    return np.array(V), np.array(T) if T else None, faces


def solid_com(V, F):
    """Volume centroid of a closed triangle mesh (signed tetrahedra)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    return ((a + b + c) / 4 * vol[:, None]).sum(0) / vol.sum()


# Collision shapes: capsules, spheres and boxes fitted to each mesh. These use MuJoCo's analytic
# colliders, so native and WASM builds agree bit for bit (mesh inertia and ellipsoid/convex
# collisions differ in the last bits and chaotic tipping amplifies that). Inertia: solid ellipsoid of the toy's extent, per kg.
PRIMS = {
    "spot": {"geoms": [{"type": "capsule", "size": [0.042], "fromto": [-0.04, 0.0, -0.005, 0.055, 0.0, -0.005]},
                       {"type": "sphere", "size": [0.035], "pos": [-0.06, 0, 0.075]},
                       *[{"type": "capsule", "size": [0.014], "fromto": [x, y, -0.03, x, y, -0.0706]}
                         for x, y in ((0.06, 0.025), (0.06, -0.025), (-0.03, 0.027), (-0.03, -0.027))]],
             "rest": 0.0846, "half": 0.1, "inertia": [0.00178, 0.0029, 0.00212]},
    "bob": {"geoms": [{"type": "box", "size": [0.08, 0.065, 0.028], "pos": [0.005, 0.0, -0.002]},
                      {"type": "sphere", "size": [0.028], "pos": [-0.06, 0, 0.035]}],
            "rest": 0.03, "half": 0.096, "inertia": [0.00138, 0.00202, 0.0026]},
    "blub": {"geoms": [{"type": "capsule", "size": [0.051], "fromto": [-0.09, 0.0, 0.0, 0.025, 0.0, 0.0]}],
             "rest": 0.051, "half": 0.143, "inertia": [0.0013, 0.00314, 0.003]},
}

items = {}
for name, (obj, tex, length) in MODELS.items():
    V, T, faces = load_obj(SRC / obj)
    V = V[:, [2, 0, 1]]                      # obj (x, y up, z fwd) -> sim (x fwd, y left, z up): cyclic, det +1
    if name == "bob":
        V = V[:, [1, 0, 2]] * [1, -1, 1]      # Bob's long axis is obj x; turn him to face +x (rotation about z)
    V *= length / np.ptp(V[:, 0])
    tri = np.array([[f[0][0], f[k][0], f[k + 1][0]] for f in faces for k in range(1, len(f) - 1)])
    V -= solid_com(V, tri)
    items[name] = {**PRIMS[name], "size": np.round(np.ptp(V, 0), 4).tolist()}
    # render mesh: unindexed corners (position + uv) so OBJ's separate uv indices survive
    pos, uv = [], []
    for f in faces:
        for k in range(1, len(f) - 1):
            for vi, ti in (f[0], f[k], f[k + 1]):
                pos += np.round(V[vi], 5).tolist()
                uv += (np.round(T[ti], 5).tolist() if T is not None and ti >= 0 else [0, 0])
    Path(f"web/public/assets/crane/{name}.json").write_text(json.dumps({"position": pos, "uv": uv}, separators=(",", ":")))
    shutil.copy(SRC / tex, f"web/public/assets/crane/{name}.png")
    print(name, "rest", items[name]["rest"], "mesh size", items[name]["size"])
Path("src/yantra/packs/assets/items.json").write_text(json.dumps(items, separators=(",", ":")))
shutil.copy("src/yantra/packs/assets/items.json", "web/public/data/factory/items.json")
Path("web/public/assets/crane/LICENSE.txt").write_text(
    "Spot, Bob and Blub by Keenan Crane, CC0 1.0 Universal.\nhttps://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/\n")
