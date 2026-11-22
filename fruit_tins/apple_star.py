#!/usr/bin/env python3
r"""
Plain apple tin + one puffy star on the top of the lid, next to the stem.

Outputs (stl/apple_star/):
  apple_lid.stl        single colour (star merged into the lid)
  apple_lid_body.stl   \  multi-colour: load both together as ONE multi-part
  apple_lid_star.stl   /  object and give the star its own filament
  apple_base.stl       unchanged plain base (copy, for convenience)
Tweak STAR_* below and run:  python3 apple_star.py
"""
import os
import math
import numpy as np
import manifold3d as m3d
from manifold3d import Manifold
import generate as g
import decor

STAR_R = 8.0        # outer radius (mm)  -> ~16 mm across
STAR_H = 2.6        # puff height at the centre (mm)
STAR_EMB = 0.8      # sunk into the lid wall (mm)
STAR_S = 19.0       # position: arc length from the top centre (mm)
STAR_PHI = -0.7     # position: angle around the apple (rad)
STAR_ROT = 18.0     # spin of the star (deg)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stl', 'apple_star')


def poly_edge_distance(pts, poly):
    """Distance from 2D points to the outline of a closed polygon."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(pts))
    for i in range(0, len(pts), 4000):
        p = pts[i:i + 4000]
        t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        out[i:i + 4000] = np.sqrt(((p[:, None] - q) ** 2).sum(2).min(1))
    return out


def puffy_star(fr):
    shape = decor.star(STAR_R, inner=0.5, round_r=0.9).rotate(STAR_ROT)
    outline = np.vstack([np.asarray(p) for p in shape.to_polygons()])
    # how far the centre is from the edge sets the dome scale
    dmax = poly_edge_distance(np.zeros((1, 2)), outline)[0]

    def post(a, b, w):
        d = poly_edge_distance(np.c_[a, b], outline)
        t = np.clip(d / (dmax * 0.9), 0, 1)
        dome = 0.35 + 0.65 * np.sqrt(1 - (1 - t) ** 2)       # rounded edge, soft top
        return np.where(w > 0, w * dome, w)

    m = shape.extrude(STAR_EMB + STAR_H).translate((0, 0, -STAR_EMB)).refine_to_length(0.35)

    def f(v):
        v = np.array(v, float)
        a, b, w = v[:, 0], v[:, 1], post(v[:, 0], v[:, 1], v[:, 2])
        s = STAR_S + a
        x = fr.from_top(s)[0]
        phi = STAR_PHI + b / np.maximum(x, 2.0)
        return decor.surf(fr, s, phi, w)
    return m.warp_batch(f)


def main():
    fr = next(f for f in g.FRUITS if f.name == 'apple')
    base, lid, _ = fr.build()
    star = fr.apply_warp(puffy_star(fr))          # follow the apple's lobes too
    star = decor.clean(star)
    body = decor.clean(lid - star)
    single = decor.clean(lid + star)
    os.makedirs(OUT, exist_ok=True)
    dz = (0, 0, -fr.zs)                           # lid printed rim-down
    for name, m in [('apple_lid.stl', single.translate(dz)),
                    ('apple_lid_body.stl', body.translate(dz)),
                    ('apple_lid_star.stl', star.translate(dz)),
                    ('apple_base.stl', base)]:
        assert m.status() == m3d.Error.NoError, name
        g.save(m, os.path.join(OUT, name))
        bb = m.bounding_box()
        print(f'{name:20s} genus {m.genus():3d}  vol {m.volume():8.0f} mm3  '
              f'{bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f}')


if __name__ == '__main__':
    main()
