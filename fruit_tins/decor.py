#!/usr/bin/env python3
"""
Decorated versions of the fruit tins: cream drips, stars, little flowers and
leaves as raised reliefs that hug the fruit surface.

Outputs (stl/decorated/):
  <fruit>_base.stl / <fruit>_lid.stl           single colour, decorations as relief
  multicolor/<fruit>/lid_body.stl, lid_white.stl, lid_green.stl, lid_brown.stl,
                     base_body.stl, base_white.stl, ...
      -> same coordinates, no overlaps: load all of one part together as ONE
         multi-part object and give every part its own filament.
Nets are unchanged (stl/plain/<fruit>_net.stl).

Usage: python3 decor.py
"""
import os
import math
import numpy as np
import manifold3d as m3d
from manifold3d import Manifold, CrossSection
import generate as g

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stl', 'decorated')
PRIORITY = ['green', 'brown', 'white']        # who wins where decorations overlap


# ------------------------------------------------------------------ 2D shapes
def star(R, points=5, inner=0.48, round_r=0.35):
    a = np.arange(points * 2) * math.pi / points + math.pi / 2
    rr = np.where(np.arange(points * 2) % 2 == 0, R, R * inner)
    cs = CrossSection([np.c_[rr * np.cos(a), rr * np.sin(a)]])
    return cs.offset(-round_r, m3d.JoinType.Round).offset(round_r, m3d.JoinType.Round)


def flower(R, petals=5):
    parts = []
    for k in range(petals):
        a = 2 * math.pi * k / petals + math.pi / 2
        p = CrossSection.circle(R * 0.42, 40).scale((1.0, 0.82)) \
            .rotate(math.degrees(a)).translate((R * 0.55 * math.cos(a), R * 0.55 * math.sin(a)))
        parts.append(p)
    return CrossSection.batch_boolean(parts, m3d.OpType.Add)


def flower_center(R):
    return CrossSection.circle(R * 0.24, 32)


def leaf_outline(L, W):
    """Leaf pointing along +x (the meridian, downward), root at x = 0."""
    t = np.linspace(0, 1, 40)
    half = W / 2 * np.sin(np.pi * t) ** 0.85 * (1 - 0.25 * t) + 0.55 * (1 - t) ** 4
    half[-1] = 0.12
    u = t * L
    return CrossSection([np.vstack([np.c_[u, -half], np.c_[u[::-1], half[::-1]]])])


# ------------------------------------------------------------------ placing on the fruit
def surf(fr, s, phi, w):
    x, y, nx, ny = fr.from_top(s)
    rad = x + w * nx
    return np.c_[rad * np.cos(phi), rad * np.sin(phi), y + w * ny]


def decal(fr, shape, s0, phi0, emb, height, rot=0.0, taper=0.85, post=None):
    """Extrude a 2D shape (x -> down the meridian, y -> around) and wrap it onto
    the outer skin centred at arc length s0 from the top and angle phi0."""
    m = shape.rotate(rot).extrude(emb + height, scale_top=(taper, taper)) \
        .translate((0, 0, -emb)).refine_to_length(0.6)

    def f(v):
        v = np.array(v, float)
        a, b, w = v[:, 0], v[:, 1], v[:, 2]
        if post is not None:
            w = post(a, b, w)
        s = s0 + a
        x = fr.from_top(s)[0]
        phi = phi0 + b / np.maximum(x, 2.0)
        return surf(fr, s, phi, w)
    return m.warp_batch(f)


def leaf(fr, s_root, phi, L, W, lift=0.0, curl=1.2):
    def post(a, b, w):
        t = np.clip(a / L, 0, 1)
        w = w + lift + curl * np.clip((t - 0.6) / 0.4, 0, 1) ** 2
        top = v_top = w > lift + 0.4
        vein = 0.35 * np.exp(-(b / 0.45) ** 2) * ((t > 0.08) & (t < 0.85))
        return np.where(v_top, w - vein, w)
    return decal(fr, leaf_outline(L, W), s_root, phi, 0.8, 1.3, taper=1.0, post=post)


def cream(fr, y_base, drips, seed=1, emb=0.6, height=1.5):
    """Cream layer poured over the top, ending in round drips.
    y_base: height of the cream edge between drips; drips: number of drips."""
    rng = np.random.default_rng(seed)
    # arc length (from the top) where the surface reaches y_base
    ss = np.linspace(0, 200, 4000)
    ys = fr.from_top(ss)[1]
    s_base = float(ss[np.argmax(ys < y_base)])
    phi = np.linspace(0, 2 * math.pi, 1440, endpoint=False)
    edge = s_base + 0.9 * np.sin(phi * drips * 2 + 0.7)
    centers = np.sort(rng.uniform(0, 2 * math.pi, drips) + np.arange(drips) * 0)
    centers = (np.arange(drips) + rng.uniform(-0.3, 0.3, drips)) * 2 * math.pi / drips
    for c in centers:
        length = rng.uniform(4.5, 11.0)
        width = rng.uniform(0.10, 0.16) * 40 / max(fr.from_top(s_base)[0], 10)
        d = np.angle(np.exp(1j * (phi - c)))
        # drip = narrow neck with a rounder drop at the end
        edge += length * np.exp(-(np.abs(d) / width) ** 2.6)
    X, Y = edge * np.cos(phi), edge * np.sin(phi)
    cs = CrossSection([np.c_[X, Y]])
    m = cs.extrude(emb + height).translate((0, 0, -emb)).refine_to_length(0.8)

    def f(v):
        v = np.array(v, float)
        s = np.hypot(v[:, 0], v[:, 1])
        ph = np.arctan2(v[:, 1], v[:, 0])
        e = np.interp(np.mod(ph, 2 * math.pi), phi, edge, period=2 * math.pi)
        w = v[:, 2]
        fall = np.clip((e - s) / 2.5, 0.25, 1.0) ** 0.5     # puffy, rounded edge
        w = np.where(w > 0, w * fall, w)
        return surf(fr, s, ph, w)
    lowest_y = float(fr.from_top(edge.max())[1])
    return m.warp_batch(f), lowest_y


def scatter(fr, n, rmin, rmax, yranges, seed, avoid_phi=None, gap=3.0, tries=4000):
    """Pick n non-overlapping spots (s, phi, r, rot) whose y lies in yranges."""
    rng = np.random.default_rng(seed)
    c = fr.curve[::-1]
    Ltot = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    pts, out = [], []
    for _ in range(tries):
        if len(out) >= n:
            break
        r = rng.uniform(rmin, rmax)
        s = rng.uniform(0, Ltot)
        x, y, _, _ = fr.from_top(s)
        if x < 14 or not any(lo + r <= y <= hi - r for lo, hi in yranges):
            continue
        phi = rng.uniform(0, 2 * math.pi)
        if avoid_phi is not None and abs(np.angle(np.exp(1j * (phi - avoid_phi)))) < 0.35:
            continue
        p = np.array([x * math.cos(phi), x * math.sin(phi), y])
        if any(np.linalg.norm(p - q) < r + rq + gap for q, rq in pts):
            continue
        pts.append((p, r))
        out.append((s, phi, r, rng.uniform(0, 72)))
    return out


# ------------------------------------------------------------------ designs
def design(fr):
    """Return {colour: [Manifold,...]} in un-warped fruit coordinates."""
    zs, H = fr.zs, fr.H
    lid_band = (zs + 2.0, H)
    base_band = (7.0, zs - 2.0)
    grp = {'white': [], 'green': [], 'brown': []}
    W, G = grp['white'], grp['green']

    if fr.name == 'apple':
        for s, phi, r, rot in scatter(fr, 22, 3.2, 6.5, [lid_band, base_band], seed=7):
            W.append(decal(fr, star(r), s, phi, 0.6, 1.2, rot))
        G.append(leaf(fr, 2.2, 0.8, 27, 14))
    elif fr.name == 'pear':
        c, ylow = cream(fr, 62.0, 7, seed=3)
        W.append(c)
        G.append(leaf(fr, 2.0, 2.2, 21, 11, lift=1.3))
        for s, phi, r, rot in scatter(fr, 14, 4.2, 6.2, [(zs + 2.0, ylow - 1.5), base_band],
                                      seed=11):
            W.append(decal(fr, flower(r), s, phi, 0.6, 1.2, rot))
    elif fr.name == 'peach':
        G.append(leaf(fr, 3.5, 1.3, 25, 12))
        G.append(leaf(fr, 3.5, 2.5, 21, 10.5))
        for s, phi, r, rot in scatter(fr, 16, 4.8, 7.4, [lid_band, base_band],
                                      seed=5, avoid_phi=0.0, gap=2.5):
            W.append(decal(fr, flower(r), s, phi, 0.6, 1.2, rot))
    elif fr.name == 'orange':
        G.append(leaf(fr, 3.0, 0.5, 24, 12))
        G.append(leaf(fr, 3.0, 1.6, 19, 10))
        for s, phi, r, rot in scatter(fr, 14, 4.0, 5.8, [lid_band, base_band], seed=9, gap=2.5):
            W.append(decal(fr, flower(r, 5), s, phi, 0.6, 1.2, rot))
    elif fr.name == 'strawberry':
        c, ylow = cream(fr, 47.0, 8, seed=4)
        W.append(c)
        fr.calyx_lift = 1.3
        fr.seed_ymax = ylow - 2.0
    return grp


def build(fr):
    lid = Manifold.revolve(fr.lid_2d(), fr.seg)
    base = Manifold.revolve(fr.base_2d(), fr.seg)
    grp = design(fr)
    grp = {k: fr.apply_warp(Manifold.batch_boolean(v, m3d.OpType.Add)) if v else Manifold()
           for k, v in grp.items()}
    lid, base = fr.apply_warp(lid), fr.apply_warp(base)
    if fr.seeds:
        pits = fr.seed_pits()
        lid, base = lid - pits, base - pits
    if fr.stem:
        grp['brown'] = grp['brown'] + fr.make_stem()
    if fr.sepals:
        grp['brown'] = grp['brown'] + fr.make_sepals()
    if fr.calyx:
        grp['green'] = grp['green'] + fr.make_calyx()

    # resolve overlaps by priority
    taken = Manifold()
    for k in PRIORITY:
        grp[k] = grp[k] - taken
        taken = taken + grp[k]

    upper = Manifold.cube((400, 400, 300)).translate((-200, -200, fr.zs))
    lower = Manifold.cube((400, 400, 300)).translate((-200, -200, fr.zs - 300))
    parts = {
        'lid': {'body': lid - taken, **{k: v ^ upper for k, v in grp.items()}},
        'base': {'body': base - taken, **{k: v ^ lower for k, v in grp.items()}},
    }
    single = {'lid': lid + (taken ^ upper), 'base': base + (taken ^ lower)}
    return parts, single


def clean(m):
    """Drop zero-volume slivers / voids left where flat decorations meet the faceted skin."""
    comps = [c for c in m.decompose() if c.volume() > 1.0]
    return Manifold.compose(comps) if len(comps) > 1 else comps[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    for fr in g.FRUITS:
        print(f'decorating {fr.name} ...')
        parts, single = build(fr)
        mdir = os.path.join(OUT, 'multicolor', fr.name)
        os.makedirs(mdir, exist_ok=True)
        for which in ('lid', 'base'):
            dz = -fr.zs if which == 'lid' else 0.0
            man = clean(single[which]).translate((0, 0, dz))
            assert man.status() == m3d.Error.NoError
            nt = g.save(man, os.path.join(OUT, f'{fr.name}_{which}.stl'))
            print(f'  {fr.name}_{which}.stl  {nt} tris  genus {man.genus()}')
            for col, m in parts[which].items():
                if m.is_empty() or m.volume() < 1.0:
                    continue
                m = clean(m).translate((0, 0, dz))
                assert m.status() == m3d.Error.NoError
                g.save(m, os.path.join(mdir, f'{which}_{col}.stl'))
                print(f'    multicolor/{fr.name}/{which}_{col}.stl  {m.volume():.0f} mm3')


if __name__ == '__main__':
    main()
