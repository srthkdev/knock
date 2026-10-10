"""Offscreen renders of a level: the table, identical-looking objects, and the target band."""
from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .level import Level


def scene_xml(level: Level, positions: dict[str, float] | None = None) -> str:
    positions = positions or {}
    t = level.task
    xs = [t.target_x - t.tol, t.target_x + t.tol]
    geoms = []
    for o in level.objects:
        x = positions.get(o.id, 0.0)
        r, g, b = (int(o.color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        geoms.append(f'<geom type="box" pos="{x} {o.y} {o.half}" size="{o.half} {o.half} {o.half}" '
                     f'rgba="{r:.3f} {g:.3f} {b:.3f} 1" material="obj"/>')
        geoms.append(f'<geom type="box" pos="0.6 {o.y} 0.0006" size="0.75 0.004 0.0005" rgba="0.75 0.74 0.70 1" '
                     f'contype="0" conaffinity="0"/>')  # lane line
    band_w = xs[1] - xs[0]
    span = (len(level.objects) - 1) * 0.35 / 2 + 0.15
    geoms.append(f'<geom type="box" pos="{t.target_x} 0 0.0008" size="{band_w / 2} {span} 0.0006" '
                 f'rgba="0.30 0.72 0.52 0.55" contype="0" conaffinity="0"/>')
    for x in (0.0,):
        geoms.append(f'<geom type="box" pos="{x} 0 0.0007" size="0.003 {span} 0.0005" rgba="0.3 0.3 0.3 1" '
                     f'contype="0" conaffinity="0"/>')
    return f"""<mujoco>
      <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
        <quality shadowsize="4096"/><global offwidth="1600" offheight="1000"/></visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.93 0.93 0.91" rgb2="0.80 0.82 0.85" width="256" height="256"/>
        <texture name="wood" type="2d" builtin="checker" rgb1="0.86 0.83 0.77" rgb2="0.84 0.81 0.75" width="512" height="512"/>
        <material name="table" texture="wood" texrepeat="8 8" reflectance="0.05"/>
        <material name="obj" specular="0.3" shininess="0.4" reflectance="0.02"/>
      </asset>
      <worldbody>
        <light pos="0.4 -0.6 2.2" dir="-0.1 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
        <geom type="plane" size="2 2 .1" material="table"/>
        {''.join(geoms)}
        <camera name="persp" pos="-0.55 -1.35 0.95" xyaxes="0.92 -0.39 0 0.2 0.47 0.86"/>
        <camera name="top" pos="0.55 0 2.0" xyaxes="1 0 0 0 1 0"/>
      </worldbody></mujoco>"""


def render_level(level: Level, path, positions=None, camera="persp", size=(960, 600), labels=True):
    m = mujoco.MjModel.from_xml_string(scene_xml(level, positions))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, height=size[1], width=size[0]) as r:
        r.update_scene(d, camera=camera)
        img = Image.fromarray(r.render())
    if labels:
        _label(img, level, m, d, camera, positions or {})
    img.save(path)
    return path


def _label(img, level, m, d, camera, positions):
    """Draw object ids and the target band label at projected positions."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    cam = m.camera(camera).id
    for o in level.objects:
        p = np.array([positions.get(o.id, 0.0), o.y, 2 * o.half + 0.03])
        uv = _project(m, d, cam, p, img.size)
        if uv is not None:
            draw.text(uv, o.id, fill=(20, 20, 20), font=font, anchor="ms")
    t = level.task
    uv = _project(m, d, cam, np.array([t.target_x, -((len(level.objects) - 1) * 0.35 / 2 + 0.2), 0.0]), img.size)
    if uv is not None:
        draw.text(uv, "target", fill=(30, 110, 75), font=font, anchor="mm")


def _project(m, d, cam_id, p, size):
    w, h = size
    pos = d.cam_xpos[cam_id]
    R = d.cam_xmat[cam_id].reshape(3, 3)
    v = R.T @ (p - pos)
    if v[2] >= 0:
        return None
    fovy = np.deg2rad(m.cam_fovy[cam_id])
    f = 0.5 * h / np.tan(fovy / 2)
    return (w / 2 + f * v[0] / -v[2], h / 2 - f * v[1] / -v[2])
