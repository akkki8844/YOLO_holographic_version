#!/usr/bin/env python3
"""holo3d.py - a tiny quad-mesh 3D engine for the hologram studio.

Everything is drawn as translucent hard-light: solid shaded quads (painter's
algorithm) with bright silhouette-lit edges, then scan-lined and bloomed by the
caller.  All-electric BLUE - no other hue is ever produced here.

The meshes are authored once at import time and reused every frame; only the
projection is recomputed, and that is fully vectorised in numpy.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Palette (BGR) - blue only
# --------------------------------------------------------------------------- #
HOLO_WHITE = (255, 248, 225)     # hot core / spec highlight
HOLO_CYAN = (255, 225, 110)      # bright electric blue-white
HOLO_BLUE = (255, 150, 40)       # deep blue body fill
HOLO_DEEP = (170, 95, 25)        # shadow-side blue
HOLO_DIM = (120, 80, 35)         # faded blue for depth layers

# Monochrome set - used ONLY by the blueprint, which is a black-and-white
# drafting hologram rather than part of the blue hard-light kit.
MONO_WHITE = (252, 252, 252)
MONO_GREY = (176, 176, 176)
MONO_FILL = (86, 86, 86)
MONO_DIM = (54, 54, 54)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def dim(color, k: float):
    """Scale a BGR colour, clamped into range (k may exceed 1.0)."""
    return (int(min(255.0, max(0.0, color[0] * k))),
            int(min(255.0, max(0.0, color[1] * k))),
            int(min(255.0, max(0.0, color[2] * k))))


def ease_out_back(f: float) -> float:
    return 1.0 + 2.70158 * (f - 1.0) ** 3 + 1.70158 * (f - 1.0) ** 2


def smoothstep(f: float) -> float:
    f = clamp01(f)
    return f * f * (3.0 - 2.0 * f)


def rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Screen-basis rotation: row0 -> screen x, row1 -> screen y, row2 -> view."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx @ ry


FLIP_X = np.diag([1.0, -1.0, -1.0])      # mirror the model for the other hand


def _basis(axis):
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a)) or 1.0
    a = a / n
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u /= (float(np.linalg.norm(u)) or 1.0)
    v = np.cross(a, u)
    return a, u, v


# --------------------------------------------------------------------------- #
# Mesh authoring
# --------------------------------------------------------------------------- #
class Mesh:
    """Quad-faced mesh with per-part explode directions.

    Triangles are stored as quads with a repeated vertex so the whole draw
    path stays a single (F, 4) index array.
    """

    def __init__(self):
        self._v: list = []
        self._vp: list = []
        self._f: list = []
        self._fs: list = []
        self._fp: list = []
        self._dirs: dict = {}
        self._cur = 0
        self.V = None

    # -- authoring ---------------------------------------------------------- #
    def part(self, pid: int, direction) -> None:
        """Declare the explode direction for a part id."""
        self._dirs[pid] = np.asarray(direction, float)
        self._cur = pid

    def use(self, pid: int) -> None:
        self._cur = pid

    def _av(self, p) -> int:
        self._v.append((float(p[0]), float(p[1]), float(p[2])))
        self._vp.append(self._cur)
        return len(self._v) - 1

    def quad(self, a, b, c, d, shade: float = 1.0) -> None:
        self._f.append((a, b, c, d))
        self._fs.append(shade)
        self._fp.append(self._cur)

    def tri(self, a, b, c, shade: float = 1.0) -> None:
        self.quad(a, b, c, c, shade)

    # -- primitives --------------------------------------------------------- #
    def box(self, c, half, shade: float = 1.0, axes=None):
        c = np.asarray(c, float)
        hx, hy, hz = half
        ax = np.asarray(axes[0], float) if axes else np.array([1.0, 0.0, 0.0])
        ay = np.asarray(axes[1], float) if axes else np.array([0.0, 1.0, 0.0])
        az = np.asarray(axes[2], float) if axes else np.array([0.0, 0.0, 1.0])
        idx = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    idx.append(self._av(c + ax * hx * sx + ay * hy * sy + az * hz * sz))
        i = {(sx, sy, sz): idx[n] for n, (sx, sy, sz) in enumerate(
            [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])}
        self.quad(i[(-1, -1, -1)], i[(1, -1, -1)], i[(1, 1, -1)], i[(-1, 1, -1)], shade * 0.9)
        self.quad(i[(-1, -1, 1)], i[(1, -1, 1)], i[(1, 1, 1)], i[(-1, 1, 1)], shade)
        self.quad(i[(-1, -1, -1)], i[(-1, -1, 1)], i[(-1, 1, 1)], i[(-1, 1, -1)], shade * 0.85)
        self.quad(i[(1, -1, -1)], i[(1, -1, 1)], i[(1, 1, 1)], i[(1, 1, -1)], shade * 0.95)
        self.quad(i[(-1, -1, -1)], i[(1, -1, -1)], i[(1, -1, 1)], i[(-1, -1, 1)], shade * 0.8)
        self.quad(i[(-1, 1, -1)], i[(1, 1, -1)], i[(1, 1, 1)], i[(-1, 1, 1)], shade)
        return idx

    def tube(self, p0, p1, r0, r1, seg: int = 12, shade: float = 1.0,
             squash: float = 1.0, cap0: bool = True, cap1: bool = True):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        a, u, v = _basis(p1 - p0)
        ring0, ring1 = [], []
        for k in range(seg):
            th = 2.0 * math.pi * k / seg
            cu, sv = math.cos(th), math.sin(th) * squash
            ring0.append(self._av(p0 + u * (cu * r0) + v * (sv * r0)))
            ring1.append(self._av(p1 + u * (cu * r1) + v * (sv * r1)))
        for k in range(seg):
            n = (k + 1) % seg
            self.quad(ring0[k], ring0[n], ring1[n], ring1[k],
                      shade * (0.78 + 0.22 * (k % 2)))
        if cap0 and seg >= 4:
            c0 = self._av(p0)
            for k in range(seg):
                self.tri(c0, ring0[k], ring0[(k + 1) % seg], shade * 0.7)
        if cap1 and seg >= 4:
            c1 = self._av(p1)
            for k in range(seg):
                self.tri(c1, ring1[k], ring1[(k + 1) % seg], shade * 0.9)
        return ring0, ring1

    def torus(self, c, axis, radius, r_tube, major: int = 16, minor: int = 6,
              shade: float = 1.0, squash: float = 1.0):
        c = np.asarray(c, float)
        a, u, v = _basis(axis)
        rings = []
        for i in range(major):
            th = 2.0 * math.pi * i / major
            centre = c + u * (math.cos(th) * radius) + v * (math.sin(th) * radius * squash)
            out = u * math.cos(th) + v * math.sin(th) * squash
            out /= (float(np.linalg.norm(out)) or 1.0)
            ring = []
            for j in range(minor):
                ph = 2.0 * math.pi * j / minor
                ring.append(self._av(centre + out * (math.cos(ph) * r_tube)
                                     + a * (math.sin(ph) * r_tube)))
            rings.append(ring)
        for i in range(major):
            n = (i + 1) % major
            for j in range(minor):
                m = (j + 1) % minor
                self.quad(rings[i][j], rings[n][j], rings[n][m], rings[i][m],
                          shade * (0.75 + 0.25 * (j % 2)))
        return rings

    def ellipsoid(self, c, radii, seg: int = 12, rings: int = 6, shade: float = 1.0):
        c = np.asarray(c, float)
        rx, ry, rz = radii
        grid = []
        for i in range(rings + 1):
            ph = math.pi * i / rings
            row = []
            for k in range(seg):
                th = 2.0 * math.pi * k / seg
                row.append(self._av((c[0] + rx * math.sin(ph) * math.cos(th),
                                     c[1] + ry * math.cos(ph),
                                     c[2] + rz * math.sin(ph) * math.sin(th))))
            grid.append(row)
        for i in range(rings):
            for k in range(seg):
                n = (k + 1) % seg
                self.quad(grid[i][k], grid[i][n], grid[i + 1][n], grid[i + 1][k],
                          shade * (0.8 + 0.2 * (k % 2)))
        return grid

    def disc(self, c, axis, radius, seg: int = 14, shade: float = 1.0):
        c = np.asarray(c, float)
        a, u, v = _basis(axis)
        mid = self._av(c)
        ring = [self._av(c + u * (math.cos(2 * math.pi * k / seg) * radius)
                         + v * (math.sin(2 * math.pi * k / seg) * radius))
                for k in range(seg)]
        for k in range(seg):
            self.tri(mid, ring[k], ring[(k + 1) % seg], shade)
        return ring

    # -- compile ------------------------------------------------------------ #
    def compile(self) -> "Mesh":
        self.V = np.asarray(self._v, np.float64)
        self.FI = np.asarray(self._f, np.int32)
        self.FS = np.asarray(self._fs, np.float64)
        self.FP = np.asarray(self._fp, np.int32)
        self.VP = np.asarray(self._vp, np.int32)
        dirs = np.zeros((max(self._dirs) + 1 if self._dirs else 1, 3))
        for pid, d in self._dirs.items():
            dirs[pid] = d
        self.PDIR = dirs
        self.VD = dirs[self.VP]
        c = self.V.mean(axis=0)
        self.radius = float(np.abs(self.V - c).max()) or 1.0
        return self

    def part_centre(self, pid: int, explode: float = 0.0) -> np.ndarray:
        sel = self.VP == pid
        if not sel.any():
            return np.zeros(3)
        return (self.V[sel] + self.VD[sel] * explode).mean(axis=0)

    @property
    def parts(self):
        return sorted(self._dirs)


# --------------------------------------------------------------------------- #
# Projection + rendering
# --------------------------------------------------------------------------- #
# Camera distance in model units.  FOCAL fixes the on-screen size; DEPTH alone
# controls how strong the perspective is - the closer the camera sits, the more
# the near face of an object flares out over the far one.  Pulled in from 4.4 so
# every model reads as a solid with real depth instead of a flat blue decal.
# (3.5 rather than lower: past about 3.2 the near faces flare out so far that
# the models cost twice as much to fill for very little extra depth.)
DEPTH = 3.5
FOCAL = 0.66


def project(v3: np.ndarray, centre, scale: float, rot: np.ndarray) -> np.ndarray:
    """Project model-space points to pixels; returns an (N, 2) float array."""
    v3 = np.atleast_2d(np.asarray(v3, float))
    vr = v3 @ rot.T
    zc = np.maximum(DEPTH - vr[:, 2], 0.35)
    k = (DEPTH * FOCAL / zc) * scale
    return np.stack([centre[0] + vr[:, 0] * k, centre[1] + vr[:, 1] * k], axis=1)


_LIGHT = np.array([-0.42, -0.58, 0.70])
_LIGHT /= np.linalg.norm(_LIGHT)


def render_mesh(dst, mesh: Mesh, centre, scale: float, rot: np.ndarray, *,
                explode: float = 0.0, alpha: float = 1.0, fill=HOLO_DEEP,
                edge=HOLO_CYAN, wire: float = 0.0, scan: bool = True,
                hot: float = 0.0, cull: bool = False, aa: bool = True):
    """Painter's-algorithm render of one mesh into a BGR overlay.

    wire  0 -> solid shaded, 1 -> pure wireframe (fills suppressed)
    hot   extra white-hot boost on the edges (materialise / lock flashes)
    cull  drop back faces - twice as fast, slightly less see-through
    Returns the projected pixel bounding box, or None when nothing was drawn.
    """
    if mesh.V is None or alpha <= 0.01 or scale <= 0.5:
        return None
    hh, ww = dst.shape[:2]
    verts = mesh.V if explode <= 0.0 else mesh.V + mesh.VD * explode
    vr = verts @ rot.T
    zc = np.maximum(DEPTH - vr[:, 2], 0.35)
    k = (DEPTH * FOCAL / zc) * scale
    px = centre[0] + vr[:, 0] * k
    py = centre[1] + vr[:, 1] * k
    if not (np.isfinite(px).all() and np.isfinite(py).all()):
        return None
    pts = np.stack([px, py], axis=1)
    fi = mesh.FI
    quads = pts[fi]                                     # (F, 4, 2)
    a, b, c = vr[fi[:, 0]], vr[fi[:, 1]], vr[fi[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0.0] = 1.0
    nrm = nrm / ln[:, None]
    lam = nrm @ _LIGHT
    facing = nrm[:, 2]
    depth = vr[fi][:, :, 2].mean(axis=1)
    # Depth cue + specular glint.  Flat Lambert alone makes a mesh look like a
    # sticker: with the far side of the object visibly receding into the dark
    # and a hard highlight riding the near edges, the same geometry reads as a
    # solid you could walk around.
    dcue = np.clip(0.5 + depth / (2.0 * mesh.radius), 0.0, 1.0)
    spec = np.clip(lam, 0.0, 1.0) ** 14
    inten = (0.12 + 0.74 * np.abs(lam) + 0.55 * spec) * mesh.FS
    inten *= 0.52 + 0.78 * dcue
    inten = np.where(facing > 0.0, inten, inten * 0.34)
    rim = 1.0 - np.abs(facing)
    area = 0.5 * np.abs((quads[:, 2, 0] - quads[:, 0, 0]) * (quads[:, 3, 1] - quads[:, 1, 1])
                        - (quads[:, 3, 0] - quads[:, 1, 0]) * (quads[:, 2, 1] - quads[:, 0, 1]))
    xs0, xs1 = quads[:, :, 0].min(axis=1), quads[:, :, 0].max(axis=1)
    ys0, ys1 = quads[:, :, 1].min(axis=1), quads[:, :, 1].max(axis=1)
    keep = (area > 1.2) & (xs1 > 0) & (ys1 > 0) & (xs0 < ww) & (ys0 < hh)
    if cull:
        keep &= facing > 0.0
    if not keep.any():
        return None
    fcol = np.clip(np.asarray(fill, float)[None, :]
                   * (inten * alpha * (1.0 - clamp01(wire)))[:, None], 0, 255)
    ek = (0.16 + 0.40 * inten + 0.50 * rim * (0.45 + 0.55 * dcue) + hot) * alpha
    ecol = np.clip(np.asarray(edge, float)[None, :] * ek[:, None], 0, 255)
    ipts = quads.astype(np.int32)
    order = np.argsort(depth)
    order = order[keep[order]]
    solid = wire < 0.98
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    fci = fcol.astype(np.int32)
    eci = ecol.astype(np.int32)
    for i in order:
        poly = ipts[i]
        if solid:
            cv2.fillPoly(dst, [poly], (int(fci[i, 0]), int(fci[i, 1]), int(fci[i, 2])))
        cv2.polylines(dst, [poly], True,
                      (int(eci[i, 0]), int(eci[i, 1]), int(eci[i, 2])), 1, lt)
    bx0 = int(max(0, xs0[keep].min())); bx1 = int(min(ww, xs1[keep].max() + 1))
    by0 = int(max(0, ys0[keep].min())); by1 = int(min(hh, ys1[keep].max() + 1))
    if scan and bx1 - bx0 > 2 and by1 - by0 > 2:
        roi = dst[by0:by1:3, bx0:bx1]
        dst[by0:by1:3, bx0:bx1] = (roi * 0.52).astype(np.uint8)
    return (bx0, by0, bx1, by1)


def screen_radius(mesh: Mesh, scale: float) -> float:
    """Worst-case pixel radius of a mesh at this scale (for ROI sizing)."""
    r = mesh.radius
    return scale * r * (DEPTH * FOCAL / max(DEPTH - r, 0.5))


def render_glow(dst, mesh: Mesh, centre, scale: float, rot: np.ndarray, *,
                gain: float = 0.95, glow: float = 0.55, **kw):
    """Render a mesh at FULL resolution into a small scratch ROI, then add it.

    Half-resolution rendering is right for the big objects, but a wrist-worn
    model only covers ~55 px there and dissolves into mush.  The scratch buffer
    is just the object's bounding box, so full-res detail costs very little,
    and a blurred copy of the same buffer supplies the bloom that the half-res
    upscale would otherwise have provided.
    """
    if mesh.V is None or scale <= 0.5 or not np.isfinite(centre).all():
        return None
    hh, ww = dst.shape[:2]
    r = screen_radius(mesh, scale) + 8.0
    if not math.isfinite(r) or r > max(ww, hh) * 2.0:
        return None
    x0 = int(max(0, math.floor(centre[0] - r)))
    x1 = int(min(ww, math.ceil(centre[0] + r)))
    y0 = int(max(0, math.floor(centre[1] - r)))
    y1 = int(min(hh, math.ceil(centre[1] + r)))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    buf = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    box = render_mesh(buf, mesh, (centre[0] - x0, centre[1] - y0), scale, rot, **kw)
    if box is None:
        return None
    roi = dst[y0:y1, x0:x1]
    if glow > 0.01:
        cv2.addWeighted(cv2.blur(buf, (11, 11)), glow, roi, 1.0, 0, roi)
    cv2.addWeighted(buf, gain, roi, 1.0, 0, roi)
    return (x0, y0, x1, y1)
