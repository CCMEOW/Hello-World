#!/usr/bin/env python3
"""
Fruit tin boxes for 3D printing (apple / pear / peach / orange / strawberry)
plus a separate foam-net style sleeve for every fruit.

Every fruit is split into
  * <fruit>_base.stl  - lower bowl with an inner locating lip (print open side up)
  * <fruit>_lid.stl   - upper cap, slips over the lip (print open side down)
  * <fruit>_net.stl   - diamond lattice sleeve that the fruit sits in
Strawberry and orange also export the green calyx alone (<fruit>_calyx.stl),
already positioned on the lid, for multi-material printing / painting masks.

Design rules used so that parts print WITHOUT supports:
  * the lid's inner ceiling is a cone (<= ~49 deg from vertical)
  * the bottom of each fruit has a flat foot and is limited to ~58 deg overhang
  * net strands rise at <= ~48 deg, the bottom ring sits on the bed

Requires: pip install manifold3d numpy trimesh
Usage:    python3 generate.py [output_dir]
"""
import os
import sys
import math
import numpy as np
import manifold3d as m3d
from manifold3d import Manifold, CrossSection

# ------------------------------------------------------------------ settings
WALL = 1.6          # wall thickness (mm)
CLEAR = 0.20        # radial clearance between lip and lid
LIP_H = 6.0         # lip height
LIP_W = 1.6         # lip thickness
CONE_S = 1.15       # lid ceiling: horizontal per vertical (1.15 ~ 49 deg)
FOOT_K = 1.60       # bottom overhang limit: horizontal per vertical (~58 deg)
SEG = 192           # circular segments for fruits

NET_RS = 1.5        # net strand radius (3 mm strands)
NET_GAP = 0.6       # gap between fruit and net strand surface
NET_FLARE = 3.0     # outward flare at the top of the net


# ------------------------------------------------------------------ helpers
def catmull_rom(pts, n_per_seg=24):
    """Centripetal-ish uniform Catmull-Rom through control points."""
    p = np.asarray(pts, float)
    p = np.vstack([p[0] * 2 - p[1], p, p[-1] * 2 - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)


def poly(points):
    return CrossSection([np.asarray(points, float)])


def rect(x0, y0, x1, y1):
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def max_x_at(curves, z):
    """Largest x where any polyline in `curves` crosses height z."""
    best = 0.0
    for c in curves:
        y0, y1 = c[:-1, 1], c[1:, 1]
        x0, x1 = c[:-1, 0], c[1:, 0]
        hit = (np.minimum(y0, y1) <= z) & (np.maximum(y0, y1) >= z) & (y0 != y1)
        if hit.any():
            t = (z - y0[hit]) / (y1[hit] - y0[hit])
            best = max(best, float(np.max(x0[hit] + t * (x1[hit] - x0[hit]))))
    return best


def closed(c):
    return np.vstack([c, c[:1]])


def tube(points, r, seg=14):
    """Chain of sphere hulls along a polyline -> round tube."""
    s = Manifold.sphere(r, seg)
    parts = [Manifold.batch_hull([s.translate(tuple(a)), s.translate(tuple(b))])
             for a, b in zip(points[:-1], points[1:])]
    return Manifold.batch_boolean(parts, m3d.OpType.Add)


def save(man, path):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln == 0, 1, ln)).astype(np.float32)
    rec = np.zeros(len(f), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, tri
    with open(path, 'wb') as fh:
        fh.write(b'fruit tin - generated'.ljust(80, b' '))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())
    return len(f)


# ------------------------------------------------------------------ fruit
class Fruit:
    def __init__(self, name, ctrl, zs, color, stem=None, calyx=None,
                 warp=None, seeds=False, seg=SEG):
        self.name, self.zs, self.color = name, zs, color
        self.stem, self.calyx, self.warp_kind, self.seeds = stem, calyx, warp, seeds
        self.seg = seg

        # outer profile: bottom axis -> side -> top axis, x = radius, y = height
        c = catmull_rom(ctrl)
        c[:, 0] = np.maximum(c[:, 0], 0)
        c[0, 0] = c[-1, 0] = 0.0
        c[:, 1] -= c[:, 1].min()
        # bottom overhang limit (only bites in the lowest few mm)
        foot = ctrl[1][0]
        c[:, 0] = np.minimum(c[:, 0], foot + c[:, 1] * FOOT_K)
        self.curve = c
        self.H = c[:, 1].max()

        P = poly(np.vstack([c, [[0, c[-1, 1]], [0, 0]]]))
        self.P = P
        full = P + P.mirror((1, 0))
        self.inner = full.offset(-WALL, m3d.JoinType.Round, circular_segments=64) \
            ^ rect(0, -1, 500, 500)
        self.inner_curves = [closed(np.asarray(p)) for p in self.inner.to_polygons()]
        self.outer_curves = [closed(np.vstack([c, [[0, c[-1, 1]], [0, 0]]]))]

        # lid ceiling cone apex: keep the whole cone inside the inner offset
        pts = np.vstack(self.inner_curves)
        pts = pts[pts[:, 1] >= zs]
        self.zc = float(np.min(pts[:, 1] + pts[:, 0] / CONE_S))

        # lip radius: fits inside the lid cavity over the lip height
        zz = np.linspace(zs, zs + LIP_H, 25)
        lid_r = [min(max_x_at(self.inner_curves, z), (self.zc - z) * CONE_S) for z in zz]
        self.lip_out = min(lid_r) - CLEAR
        self.lip_in = self.lip_out - LIP_W
        if self.zc - self.lip_out / CONE_S < zs + LIP_H:
            print(f'  ! {name}: lid ceiling cone clips the lip region')

    # ---- outer radius at height z (for nets & texture masks)
    def r_out(self, z):
        return max_x_at(self.outer_curves, z)

    # ---- point on outer surface at arc length s measured from the top axis
    def from_top(self, s):
        c = self.curve[::-1]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        x = np.interp(s, d, c[:, 0])
        y = np.interp(s, d, c[:, 1])
        tx = np.interp(s, d, np.gradient(c[:, 0], d))
        ty = np.interp(s, d, np.gradient(c[:, 1], d))
        ln = np.hypot(tx, ty) + 1e-9
        # walking from the top down the side, outward normal = tangent rotated +90deg
        return x, y, -ty / ln, tx / ln

    # ---- 2D sections
    def base_2d(self):
        zs = self.zs
        outer = self.P ^ rect(0, -1, 500, zs)
        chamfer = poly([(0, -1000), (self.lip_in + zs + 1000, -1000),
                        (self.lip_in, zs), (0, zs)])
        cavity = self.inner ^ chamfer ^ rect(0, -1, 500, zs + 1)
        lip = poly([(self.lip_in, zs - 0.01), (self.lip_out, zs - 0.01),
                    (self.lip_out, zs + LIP_H - 0.6),
                    (self.lip_out - 0.6, zs + LIP_H), (self.lip_in, zs + LIP_H)])
        return (outer - cavity) + lip

    def lid_2d(self):
        zs = self.zs
        outer = self.P ^ rect(0, zs, 500, 500)
        cone = poly([(0, -1000), ((self.zc + 1000) * CONE_S, -1000), (0, self.zc)])
        cavity = self.inner ^ cone ^ rect(0, zs - 1, 500, 500)
        return outer - cavity

    # ---- surface decoration -------------------------------------------
    def apply_warp(self, man):
        if self.warp_kind is None:
            return man
        zg = np.linspace(0, self.H, 600)
        rg = np.array([self.r_out(z) for z in zg])
        kind, fr = self.warp_kind, self

        def f(v):
            v = np.array(v, float)
            r = np.hypot(v[:, 0], v[:, 1])
            th = np.arctan2(v[:, 1], v[:, 0])
            z = v[:, 2]
            if kind == 'peach':
                # suture crease down one side, deeper toward the top
                w = 0.13 + 0.10 * (1 - z / fr.H)
                d = -2.6 * np.exp(-(th / w) ** 2) * np.clip((z - 4) / (fr.H * 0.5), 0, 1)
                d *= np.clip(r / 6, 0, 1)
            elif kind == 'orange':
                outer = r > np.interp(z, zg, rg) - 0.4
                n = (np.sin(th * 41 + z * 0.9) * np.sin(z * 1.7 + th * 7)
                     + np.sin(th * 67 - z * 1.3) * np.sin(z * 2.3 - th * 11))
                d = np.where(outer, -0.18 * (n * n), 0.0)
            else:
                d = 0.0
            scale = np.where(r > 1e-6, (r + d) / np.maximum(r, 1e-6), 1.0)
            v[:, 0] *= scale
            v[:, 1] *= scale
            return v
        return man.warp_batch(f)

    def seed_pits(self):
        pits = []
        sph = Manifold.sphere(1.0, 10)
        top_skip = 18.0 if self.calyx else 4.0
        c = self.curve[::-1]
        L = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        n = 170
        ga = math.pi * (3 - math.sqrt(5))
        for i in range(n):
            s = top_skip + (L - top_skip - 6) * (i + 0.5) / n
            x, y, nx, ny = self.from_top(s)
            if x < 5 or abs(y - self.zs) < 2.4 or y < 5:
                continue
            a = i * ga
            cx, cz = x + 0.45 * nx, y + 0.45 * ny
            pit = sph.scale((1.05, 0.75, 1.25)).rotate((0, 0, 0))
            pit = pit.translate((cx, 0, cz)).rotate((0, 0, math.degrees(a)))
            pits.append(pit)
        return Manifold.batch_boolean(pits, m3d.OpType.Add)

    def make_calyx(self):
        n, L, W = self.calyx
        parts = []
        for k in range(n):
            # leaf outline in (u = arc length from top, v = sideways)
            u = np.linspace(1.5, L, 24)
            half = W / 2 * np.sin(np.pi * np.clip(u / L, 0, 1) ** 0.7) + 0.9
            half[-1] = 0.15
            outline = np.vstack([np.c_[u, -half], np.c_[u[::-1], half[::-1]]])  # CCW
            leaf = CrossSection([outline]).extrude(2.0).translate((0, 0, -1.0))
            leaf = leaf.refine_to_length(0.8)
            phi0 = 2 * math.pi * k / n + 0.2 * math.sin(k * 2.3)
            fr = self

            def f(v, phi0=phi0, k=k):
                v = np.array(v, float)
                s = v[:, 0]
                x, y, nx, ny = fr.from_top(s)
                curl = 0.6 * np.clip((s - L * 0.7) / (L * 0.3), 0, 1) ** 2 * (1 + 0.5 * math.sin(k))
                w = v[:, 2] + curl
                phi = phi0 + v[:, 1] / np.maximum(x, 3.0)
                rad = x + w * nx
                return np.c_[rad * np.cos(phi), rad * np.sin(phi), y + w * ny]
            parts.append(leaf.warp_batch(f))
        top = self.from_top(0.0)[1]
        hub = Manifold.cylinder(2.5, 4.5, 4.0, 32).translate((0, 0, top - 1.5))
        parts.append(hub)
        return Manifold.batch_boolean(parts, m3d.OpType.Add)

    def make_stem(self):
        length, lean, r = self.stem
        top = self.from_top(0.0)[1]
        t = np.linspace(0, 1, 14)
        pts = np.c_[lean * t ** 1.6, np.zeros_like(t), top - 2.5 + (length + 2.5) * t]
        return tube(pts, r, 16)

    # ---- build ---------------------------------------------------------
    def build(self):
        base = Manifold.revolve(self.base_2d(), self.seg)
        lid = Manifold.revolve(self.lid_2d(), self.seg)
        base, lid = self.apply_warp(base), self.apply_warp(lid)
        if self.seeds:
            pits = self.seed_pits()
            base, lid = base - pits, lid - pits
        calyx = None
        if self.stem:
            lid = lid + self.make_stem()
        if self.calyx:
            calyx = self.make_calyx()
            lid = lid + calyx
        return base, lid, calyx

    # ---- net -----------------------------------------------------------
    def build_net(self):
        zg = np.linspace(0, self.H, 400)
        rg = np.array([self.r_out(z) for z in zg])
        zeq = zg[np.argmax(rg)]
        drdz = np.gradient(rg, zg)
        # start the net where the fruit wall is steep enough to print on
        zb = float(zg[np.argmax((drdz < 1.15) & (zg > 1))])
        Hn = zeq + 4.0 - zb
        h = np.linspace(0, Hn, 300)
        run = np.maximum.accumulate(np.interp(h + zb, zg, rg))
        R = run + NET_GAP + NET_RS + NET_FLARE * (h / Hn) ** 2
        dR = np.gradient(R, h)
        # tangential rate so total slope stays <= 45 deg
        tan_rate = np.sqrt(np.clip(1.2 - dR ** 2, 0.12, None)) / R
        theta = np.r_[0, np.cumsum(0.5 * (tan_rate[1:] + tan_rate[:-1]) * np.diff(h))]
        D = theta[-1]
        best = None
        for N in range(14, 23):
            mcross = min(math.floor(2 * D * N / (2 * math.pi)), 4)
            if mcross < 2:
                continue
            sc = mcross * 2 * math.pi / (N * 2 * D)
            if best is None or sc > best[0]:
                best = (sc, N, mcross)
        sc, N, mcross = best
        theta *= sc

        zc0, zc1 = NET_RS, Hn - NET_RS
        hh = np.linspace(zc0, zc1, int((zc1 - zc0) / 1.6) + 2)
        Rh, Th = np.interp(hh, h, R), np.interp(hh, h, theta)
        strands = []
        for k in range(N):
            a0 = 2 * math.pi * k / N
            for sgn in (1, -1):
                a = a0 + sgn * Th
                strands.append(np.c_[Rh * np.cos(a), Rh * np.sin(a), hh])
        parts = [tube(p, NET_RS, 12) for p in strands]
        for zr, rr in ((zc0, Rh[0]), (zc1, Rh[-1])):
            ring = Manifold.revolve(CrossSection.circle(NET_RS, 16).translate((rr, zr)), 160)
            parts.append(ring)
        net = Manifold.batch_boolean(parts, m3d.OpType.Add)
        self.net_info = dict(zb=zb, N=N, rows=mcross, H=Hn, top_d=2 * Rh[-1])
        return net, zb


# ------------------------------------------------------------------ catalogue
FRUITS = [
    Fruit('apple',
          [(0, 0), (17, 0), (24, 1.6), (31, 5.5), (36, 13), (38.5, 24), (38.6, 34),
           (37, 44), (33, 52.5), (26.5, 59), (19, 62.8), (12, 62.6), (6.5, 59.8),
           (2.5, 57.6), (0, 57.2)],
          zs=26, color='#d8322b', stem=(11.0, 3.2, 1.9)),
    Fruit('pear',
          [(0, 0), (15, 0), (22, 1.8), (29, 7), (33.5, 15.5), (34.5, 25), (33, 34),
           (28.5, 43), (22.5, 51), (18.3, 59), (16.4, 66), (14.4, 72.5), (10, 77.2),
           (5, 78.6), (0, 78.4)],
          zs=26, color='#8db83a', stem=(9.0, 2.2, 1.7)),
    Fruit('peach',
          [(0, 0), (15, 0), (23, 2.2), (30, 7.5), (34.8, 16), (36.4, 27), (35.8, 38),
           (32.5, 48), (26.5, 56.5), (18, 62.5), (9, 66), (3.5, 67.6), (0, 68.6)],
          zs=29, color='#f2a1a8', warp='peach'),
    Fruit('orange',
          [(0, 0), (16, 0), (25, 2.2), (32, 7.5), (36.6, 16), (38.2, 27), (37.2, 38),
           (33.5, 48), (27, 55.6), (18, 60.3), (9, 62.2), (0, 62.6)],
          zs=26, color='#f07c25', warp='orange', calyx=(5, 5.5, 3.2)),
    Fruit('strawberry',
          [(0, 0), (7, 0), (13, 2.6), (20, 8.5), (26.5, 17), (31.5, 27), (34, 37),
           (34, 46), (31, 54), (24.5, 59.5), (15, 62.2), (6, 63), (0, 63.2)],
          zs=29, color='#d72631', seeds=True, calyx=(7, 23.0, 8.5)),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'stl')
    os.makedirs(out, exist_ok=True)
    report = []
    for fr in FRUITS:
        print(f'building {fr.name} ...')
        base, lid, calyx = fr.build()
        net, zb = fr.build_net()
        # print orientation: base open side up, lid rim on the bed, net bottom ring on bed
        files = {
            f'{fr.name}_base.stl': base,
            f'{fr.name}_lid.stl': lid.translate((0, 0, -fr.zs)),
            f'{fr.name}_net.stl': net,
        }
        if calyx is not None:
            files[f'{fr.name}_calyx.stl'] = calyx.translate((0, 0, -fr.zs))
        for fn, man in files.items():
            assert man.status() == m3d.Error.NoError, fn
            nt = save(man, os.path.join(out, fn))
            bb = man.bounding_box()
            print(f'  {fn:24s} {nt:7d} tris  {bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f} mm'
                  f'  genus {man.genus()}')
        bb = base.bounding_box()
        report.append((fr, bb, net))
    return report


if __name__ == '__main__':
    main()
