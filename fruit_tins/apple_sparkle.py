#!/usr/bin/env python3
"""
Plain apple tin + a puffy four-point sparkle star standing upright on the lid.

Outputs (stl/apple_sparkle/):
  apple_lid.stl          single colour (star merged into the lid)
  apple_lid_body.stl     multi-colour: load both together as ONE multi-part
  apple_lid_star.stl     object and give the star its own filament
  apple_base.stl         unchanged plain base
Tweak the SPARK_* values and run:  python3 apple_sparkle.py
"""
import os
import math
import numpy as np
import manifold3d as m3d
from manifold3d import Manifold, CrossSection
import generate as g
import decor

SPARK_H = 14.0      # half height (centre -> top tip), mm  -> ~28 mm tall
SPARK_W = 9.5       # half width (centre -> side tip), mm  -> ~19 mm wide
SPARK_P = 0.62      # <1 = concave sides; smaller = skinnier arms
SPARK_T = 6.0       # thickness at the centre (mm); edges taper to SPARK_EDGE
SPARK_EDGE = 1.6
SPARK_S = 22.0      # position: arc length from the top centre (mm)
SPARK_PHI = -math.pi / 2 + 0.85   # position: angle around the apple (front-right of the stem)
SPARK_TILT = 10.0   # lean sideways (deg)
SPARK_FACE = 0.0    # which way the star faces (deg, 0 = front)
BOSS_R = 0.0        # little rounded mound the star stands on (mm)
SPARK_SINK = 7.0    # how deep the bottom tip goes into the lid (mm)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stl', 'apple_sparkle')


def sparkle_outline(n=400):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / SPARK_P
    x = SPARK_W * np.sign(c) * np.abs(c) ** e
    y = SPARK_H * np.sign(s) * np.abs(s) ** e
    cs = CrossSection([np.c_[x, y]])
    # blunt the four tips a little so they print
    return cs.offset(-0.45, m3d.JoinType.Round).offset(0.45, m3d.JoinType.Round)


def puffy_sparkle():
    """Star in the XZ plane (x = sideways, z = up), puffy in Y, centred at origin."""
    shape = sparkle_outline()
    outline = np.vstack([np.asarray(p) for p in shape.to_polygons()])
    from apple_star import poly_edge_distance
    dmax = poly_edge_distance(np.zeros((1, 2)), outline)[0]
    m = shape.extrude(1.0).translate((0, 0, -0.5)).refine_to_length(0.35)

    def f(v):
        v = np.array(v, float)
        d = poly_edge_distance(v[:, :2], outline)
        t = np.clip(d / dmax, 0, 1)
        half = 0.5 * (SPARK_EDGE + (SPARK_T - SPARK_EDGE) * np.sqrt(1 - (1 - t) ** 2))
        thick = v[:, 2] * 2 * half          # -half .. +half
        return np.c_[v[:, 0], -thick, v[:, 1]]  # 2D y -> up (z); minus keeps it right-handed
    return m.warp_batch(f)


def main():
    fr = next(f for f in g.FRUITS if f.name == 'apple')
    base, lid, _ = fr.build()

    x, y, nx, ny = fr.from_top(SPARK_S)
    star = puffy_sparkle().rotate((0, -SPARK_TILT, 0))
    # stand it on the surface: bottom tip sunk SPARK_SINK into the lid
    star = star.translate((0, 0, SPARK_H - SPARK_SINK))
    # star faces the front of the apple (-y); SPARK_FACE turns it
    star = star.rotate((0, 0, SPARK_FACE)).translate(
        (float(x) * math.cos(SPARK_PHI), float(x) * math.sin(SPARK_PHI), float(y)))
    # where the bottom tip ends up -> put a small rounded mound there
    th = math.radians(-SPARK_TILT)
    tip = np.array([-SPARK_H * math.sin(th), 0.0, -SPARK_H * math.cos(th) + SPARK_H - SPARK_SINK])
    f_ = math.radians(SPARK_FACE)
    tip = np.array([tip[0] * math.cos(f_) - tip[1] * math.sin(f_),
                    tip[0] * math.sin(f_) + tip[1] * math.cos(f_), tip[2]])
    tip += np.array([float(x) * math.cos(SPARK_PHI), float(x) * math.sin(SPARK_PHI), float(y)])
    rt = float(np.hypot(tip[0], tip[1]))
    ss = np.linspace(0, 40, 2000)
    xs, ys, _, _ = fr.from_top(ss)
    zsurf = float(ys[np.argmin(np.abs(xs - rt))])
    # drop the star so its bottom tip sits SPARK_SINK below the skin right there
    star = star.translate((0, 0, zsurf - SPARK_SINK - tip[2]))
    if BOSS_R > 0:
        boss = Manifold.sphere(BOSS_R, 48).scale((1, 1, 0.6)).translate((tip[0], tip[1], zsurf))
        star = star + boss
    # nothing may poke into the box: cut away the lid cavity (+ keep 1 mm skin)
    cavity = fr.apply_warp(Manifold.revolve(fr.inner ^ decor_rect(fr.zs - 1), fr.seg))
    star = decor.clean(star - cavity)

    overlap = (lid ^ star).volume()
    assert overlap > 10, f'star barely touches the lid ({overlap:.1f} mm3) - raise SPARK_SINK'
    print(f'star/lid contact volume {overlap:.0f} mm3')
    single = decor.clean(lid + star)
    body = decor.clean(lid - star)
    os.makedirs(OUT, exist_ok=True)
    dz = (0, 0, -fr.zs)
    for name, m in [('apple_lid.stl', single.translate(dz)),
                    ('apple_lid_body.stl', body.translate(dz)),
                    ('apple_lid_star.stl', star.translate(dz)),
                    ('apple_base.stl', base)]:
        assert m.status() == m3d.Error.NoError, name
        g.save(m, os.path.join(OUT, name))
        bb = m.bounding_box()
        print(f'{name:20s} genus {m.genus():3d}  vol {m.volume():8.0f} mm3  '
              f'{bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f}')


def decor_rect(z0):
    return g.rect(0, z0, 500, 500)


if __name__ == '__main__':
    main()
