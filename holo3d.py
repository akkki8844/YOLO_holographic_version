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

    def slim(self, k: float) -> "Mesh":
        """Scale the whole model across its long axis (y and z), before compile.

        Authoring a bracer is much easier if the cross-section can be tuned in
        one place afterwards instead of in thirty radii.
        """
        self._v = [(x, y * k, z * k) for (x, y, z) in self._v]
        return self

    # -- compile ------------------------------------------------------------ #
    def _smooth_normals(self) -> None:
        """Area-weighted vertex normals, in MODEL space.

        The renderer only ever rotates the model, never deforms it, so these
        can be baked once and rotated with the same matrix as the vertices.
        They are what turns a tube from a ring of visibly separate facets into
        a smooth shaded cylinder: the fill of each quad is graded from its own
        corner normals instead of being one flat slab of Lambert.
        """
        fi = self.FI
        a, b, c = self.V[fi[:, 0]], self.V[fi[:, 1]], self.V[fi[:, 2]]
        d = self.V[fi[:, 3]]
        # cross of the quad diagonals: for a triangle (d == c) this degrades
        # gracefully to twice the triangle normal, so tris need no special case
        fn = np.cross(c - a, d - b)
        vn = np.zeros_like(self.V)
        for k in range(4):
            np.add.at(vn, fi[:, k], fn)      # area-weighted: fn is unnormalised
        ln = np.linalg.norm(vn, axis=1)
        ln[ln == 0.0] = 1.0
        self.VN = vn / ln[:, None]
        # Blend each corner back toward its own FACE normal by how far the two
        # disagree.  Averaging normals unconditionally is what turns a crisp
        # armour box into a soft blob: at a cube corner the three faces meeting
        # there average to a diagonal and every side of the box ends up lit the
        # same.  Curved surfaces (a tube ring, a dome) have corners that agree
        # closely with their faces, so they smooth fully; anything folding by
        # more than ~40 degrees stays hard.  Dot products are rotation
        # invariant, so this whole decision bakes in at author time.
        fl = np.linalg.norm(fn, axis=1)
        fl[fl == 0.0] = 1.0
        fhat = fn / fl[:, None]
        corner = self.VN[fi]                             # (F, 4, 3)
        agree = np.einsum("fkj,fj->fk", corner, fhat)
        w = np.clip((agree - 0.76) / 0.18, 0.0, 1.0)[:, :, None]
        blend = fhat[:, None, :] * (1.0 - w) + corner * w
        bl = np.linalg.norm(blend, axis=2)
        bl[bl == 0.0] = 1.0
        self.CN = blend / bl[:, :, None]                 # (F, 4, 3)

    def _build_edges(self) -> None:
        """Unique edge list with the (up to two) faces on either side.

        Only needed so the renderer can tell a SILHOUETTE edge - where the
        surface folds away from the camera - from an interior panel crease.
        Drawing those two the same way is what makes a hologram read as a
        flat decal; a hot outline around the true occluding contour with
        quieter creases inside it is most of what sells the volume.
        """
        fi = self.FI
        ev = np.stack([fi[:, [0, 1, 2, 3]].ravel(),
                       fi[:, [1, 2, 3, 0]].ravel()], axis=1)
        ef = np.repeat(np.arange(len(fi), dtype=np.int32), 4)
        live = ev[:, 0] != ev[:, 1]          # triangles carry one null edge
        ev, ef = np.sort(ev[live], axis=1), ef[live]
        if len(ev) == 0:
            self.EV = np.zeros((0, 2), np.int32)
            self.EF = np.zeros((0, 2), np.int32)
            self.FE = np.full((len(fi), 4), -1, np.int32)
            return
        key = ev[:, 0].astype(np.int64) * (len(self.V) + 1) + ev[:, 1]
        uk, inv = np.unique(key, return_inverse=True)
        srt = np.argsort(inv, kind="stable")
        inv_s, ef_s, ev_s = inv[srt], ef[srt], ev[srt]
        counts = np.bincount(inv_s, minlength=len(uk))
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rank = np.arange(len(inv_s)) - np.repeat(start, counts)
        pair = np.full((len(uk), 2), -1, np.int32)
        pair[inv_s[rank == 0], 0] = ef_s[rank == 0]
        pair[inv_s[rank == 1], 1] = ef_s[rank == 1]
        self.EV = ev_s[rank == 0].astype(np.int32)
        self.EF = pair
        # ...and the reverse map, face-corner -> unique edge, so the draw loop
        # can ask "is side k of this face on the silhouette?" without a lookup
        fe = np.full(len(fi) * 4, -1, np.int32)
        fe[np.flatnonzero(live)] = inv
        self.FE = fe.reshape(-1, 4)

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
        # measured from the ORIGIN, not the centroid: the projection pivots on
        # the origin, so that is what bounds the model on screen.  (A long
        # model like the forearm bracer sits well off its own centroid, and a
        # centroid-based radius under-sizes its scratch buffer and clips it.)
        self.radius = float(np.abs(self.V).max()) or 1.0
        self._smooth_normals()
        self._build_edges()
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

# the second half of a quad when it is split for gradient shading
_TRI_B = np.array([0, 2, 3])


def render_mesh(dst, mesh: Mesh, centre, scale: float, rot: np.ndarray, *,
                explode: float = 0.0, alpha: float = 1.0, fill=HOLO_DEEP,
                edge=HOLO_CYAN, wire: float = 0.0, scan: bool = True,
                scan_phase: float = 0.0, hot: float = 0.0,
                cull: bool = False, aa: bool = True, smooth: bool = True,
                translucent: bool = True):
    """Painter's-algorithm render of one mesh into a BGR overlay.

    wire        0 -> solid shaded, 1 -> pure wireframe (fills suppressed)
    hot         extra white-hot boost on the edges (materialise / lock flashes)
    cull        drop back faces - twice as fast, slightly less see-through
    scan_phase  scrolls the scan-lines; feed it time and the raster CREEPS down
                the model the way a live projection does, instead of sitting
                on it like a printed texture.
    smooth      grade each big quad from its own corner normals instead of
                filling it flat
    translucent composite the far shell UNDER the near one additively, so the
                back of the object shows through the front
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
    # Fresnel on the FILL, not just the outline.  Real volumetric hard light is
    # brightest where you look along its surface rather than into it, so the
    # faces turning away toward the silhouette carry more glow than the ones
    # aimed at you.  Without this the interior reads flat and only the outline
    # says "3D"; with it the whole body has a shell to it.
    fres = rim * rim
    area = 0.5 * np.abs((quads[:, 2, 0] - quads[:, 0, 0]) * (quads[:, 3, 1] - quads[:, 1, 1])
                        - (quads[:, 3, 0] - quads[:, 1, 0]) * (quads[:, 2, 1] - quads[:, 0, 1]))
    xs0, xs1 = quads[:, :, 0].min(axis=1), quads[:, :, 0].max(axis=1)
    ys0, ys1 = quads[:, :, 1].min(axis=1), quads[:, :, 1].max(axis=1)
    keep = (area > 1.2) & (xs1 > 0) & (ys1 > 0) & (xs0 < ww) & (ys0 < hh)
    if cull:
        keep &= facing > 0.0
    if not keep.any():
        return None
    # ---- smooth shading ---------------------------------------------------- #
    # Lighting a quad from its own FLAT normal is what makes a 12-sided tube
    # read as twelve separate metal plates.  Relighting it from the averaged
    # corner normals of the surface instead costs one matmul for the whole
    # mesh and collapses the facet steps into a continuous roll of light, while
    # the flat normal is still what decides facing, culling and depth order.
    ra = rb = smod = None
    if smooth and getattr(mesh, "CN", None) is not None:
        cn = (mesh.CN.reshape(-1, 3) @ rot.T).reshape(-1, 4, 3)
        clam = cn @ _LIGHT
        crim = 1.0 - np.abs(cn[:, :, 2])
        cspec = np.clip(clam, 0.0, 1.0) ** 14
        cmod = ((0.12 + 0.74 * np.abs(clam) + 0.55 * cspec)
                * (1.0 + 0.55 * crim * crim))
        fmod = (0.12 + 0.74 * np.abs(lam) + 0.55 * spec) * (1.0 + 0.55 * fres)
        # ratios, not absolutes: everything the face already carries (its own
        # shade, the depth cue, alpha, back-face damping) survives untouched
        # and only the DISTRIBUTION of light over the surface changes.  On a
        # hard-edged face the corner normals ARE the face normal, so the ratio
        # is exactly 1 and the panel keeps its crisp flat tone.
        crat = np.clip(cmod / fmod[:, None], 0.30, 2.30)
        ra = crat[:, :3].mean(axis=1)                    # corners 0-1-2
        rb = (crat[:, 0] + crat[:, 2] + crat[:, 3]) / 3.0  # corners 0-2-3
        smod = 0.5 * (ra + rb)
    fcol = np.clip(np.asarray(fill, float)[None, :]
                   * (inten * (1.0 + 0.55 * fres) * alpha
                      * (1.0 - clamp01(wire)))[:, None], 0, 255)
    if smod is not None:
        fcol = np.clip(fcol * smod[:, None], 0, 255)
    # Interior creases run QUIETER than they used to: the silhouette pass below
    # now owns the outline, and a panel line as bright as the contour is what
    # made the old render look like a wireframe drawing rather than a lit shell.
    ek = (0.13 + 0.33 * inten + 0.40 * rim * (0.45 + 0.55 * dcue) + hot) * alpha
    # The glint runs WHITE-hot instead of merely a brighter blue.  A specular
    # highlight that keeps the body hue reads as a glowing decal; one that
    # blows out to white reads as a hard surface catching a light.
    ehot = np.clip(0.9 * spec + hot, 0.0, 1.0)
    ecol = np.clip((np.asarray(edge, float)[None, :] * (1.0 - ehot)[:, None]
                    + np.asarray(HOLO_WHITE, float)[None, :] * ehot[:, None])
                   * ek[:, None], 0, 255)
    # Near faces carry a heavier outline than far ones - the cheapest depth cue
    # there is.  Only above a size threshold: on a 40 px model a 2 px outline
    # swallows the panel lines it is supposed to separate.
    ethick = np.where((dcue > 0.62) & (scale >= 24.0), 2, 1).astype(np.int32)
    ipts = quads.astype(np.int32)
    order = np.argsort(depth)
    order = order[keep[order]]
    solid = wire < 0.98
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    bx0 = int(max(0, xs0[keep].min())); bx1 = int(min(ww, xs1[keep].max() + 1))
    by0 = int(max(0, ys0[keep].min())); by1 = int(min(hh, ys1[keep].max() + 1))

    # ---- per-corner gradient ---------------------------------------------- #
    # On top of the smoothed relight, the BIG faces get filled as two
    # differently-lit triangles so the light rolls across them instead of
    # stepping at the quad boundary.  Deliberately restricted to faces that
    # are both large and carry a real spread: below that, the second fillPoly
    # costs more than the eye gains.
    grad = None
    if solid and smod is not None:
        grad = keep & (area > 130.0) & (np.abs(ra - rb) > 0.10)
        if not grad.any():
            grad = None
        else:
            fca = np.clip(fcol * (ra / smod)[:, None], 0, 255).astype(np.int32)
            fcb = np.clip(fcol * (rb / smod)[:, None], 0, 255).astype(np.int32)
    fci = fcol.astype(np.int32)
    eci = ecol.astype(np.int32)

    # ---- translucent layering --------------------------------------------- #
    # Painter's algorithm alone paints the near shell straight over the far one,
    # so a "hologram" ends up as opaque as a plastic toy.  Rendering the faces
    # that point AWAY from us into the target first and the near ones into a
    # scratch layer that is then ADDED means the far wall keeps showing through
    # the near wall, and the two crossing each other reads brighter still -
    # exactly how a volumetric projection behaves.
    layer = None
    back = None
    if (translucent and solid and not cull
            and bx1 - bx0 > 3 and by1 - by0 > 3
            and (facing[order] <= 0.0).any() and (facing[order] > 0.0).any()):
        layer = np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8)
        lpts = ipts - np.array([bx0, by0], np.int32)
        back = facing <= 0.0
        # the far shell is quieter so the sum of the two does not blow out
        fci[back] = (fci[back] * 0.20).astype(np.int32)
        # the creases on the far wall drop back too, or the object turns into a
        # ball of wire with no near/far reading at all
        eci[back] = (eci[back] * 0.18).astype(np.int32)
        if grad is not None:
            fca[back] = (fca[back] * 0.20).astype(np.int32)
            fcb[back] = (fcb[back] * 0.20).astype(np.int32)

    # ---- silhouette edges -------------------------------------------------- #
    # An edge shared by a front face and a back face (or owned by a single face)
    # is where the surface folds out of sight: the occluding contour.  Every
    # other edge is an interior crease.  Drawing the two identically is exactly
    # what flattens a wireframe, so contours run hot and thick while creases
    # stay quiet.  It has to happen INSIDE the depth-sorted loop, though - these
    # models are dozens of overlapping primitives, and a contour pass done at
    # the end paints the outline of every buried cartridge and strap straight
    # through the housing that should be hiding it.
    sfl = None
    if getattr(mesh, "FE", None) is not None and len(mesh.EF):
        f0, f1 = mesh.EF[:, 0], mesh.EF[:, 1]
        s1 = np.where(f1 < 0, -facing[f0], facing[np.maximum(f1, 0)])
        sil = (facing[f0] * s1) <= 0.0
        fe = mesh.FE
        sfl = np.where(fe >= 0, sil[np.maximum(fe, 0)], False)   # (F, 4)
        sany = sfl.any(axis=1)
        segs4 = np.stack([ipts, np.roll(ipts, -1, axis=1)], axis=2)  # (F,4,2,2)
        lseg4 = segs4 - np.array([bx0, by0], np.int32) if layer is not None else None
        # the contour rides the same hue as the crease but far brighter, and
        # thicker on the near half of the model so the outline itself carries
        # depth rather than ringing the whole thing at one weight
        scol = np.clip(ecol * (1.28 + 0.9 * hot), 0, 255).astype(np.int32)
        if back is not None:
            # a contour on the FAR wall has to fade with the wall it belongs
            # to, or the outline of every buried part burns straight through
            # the housing that is supposed to hide it
            scol[back] = (scol[back] * 0.22).astype(np.int32)
        sthick = np.where((dcue > 0.66) & (scale >= 40.0), 2, 1).astype(np.int32)

    for i in order:
        if layer is not None and not back[i]:
            tgt, poly, segs = layer, lpts[i], lseg4
        else:
            tgt, poly, segs = dst, ipts[i], segs4
        if solid:
            if grad is not None and grad[i]:
                cv2.fillPoly(tgt, [poly[:3]],
                             (int(fca[i, 0]), int(fca[i, 1]), int(fca[i, 2])))
                cv2.fillPoly(tgt, [poly[_TRI_B]],
                             (int(fcb[i, 0]), int(fcb[i, 1]), int(fcb[i, 2])))
            else:
                cv2.fillPoly(tgt, [poly],
                             (int(fci[i, 0]), int(fci[i, 1]), int(fci[i, 2])))
        cv2.polylines(tgt, [poly], True,
                      (int(eci[i, 0]), int(eci[i, 1]), int(eci[i, 2])),
                      int(ethick[i]), lt)
        if sfl is not None and sany[i]:
            cv2.polylines(tgt, list(segs[i][sfl[i]]), False,
                          (int(scol[i, 0]), int(scol[i, 1]), int(scol[i, 2])),
                          int(sthick[i]), lt)
    if layer is not None:
        roi = dst[by0:by1, bx0:bx1]
        cv2.addWeighted(layer, 1.0, roi, 1.0, 0, roi)

    if scan and bx1 - bx0 > 2 and by1 - by0 > 2:
        _scanlines(dst, bx0, by0, bx1, by1, scan_phase)
    return (bx0, by0, bx1, by1)


# Interference fringe: period and drift of the wide bands that roll through the
# body of the projection, on top of the fine 3-pixel raster.
_FRINGE = 23


def _scanlines(dst, bx0, by0, bx1, by1, phase):
    """Fine raster + drifting interference fringes over the model's footprint.

    The fine 1-in-3 raster already said "projection".  The wide, slow fringe
    on top of it is the other half of the tell: real volumetric displays beat
    against themselves, and the banding drifts at a different rate from the
    scan.  Both are strided slices, so this touches well under half the pixels
    in the box.
    """
    ys = by0 + int(phase) % 3
    if ys < by1:
        roi = dst[ys:by1:3, bx0:bx1]
        dst[ys:by1:3, bx0:bx1] = (roi * 0.52).astype(np.uint8)
    f0 = by0 + int(by0 - phase * 0.55) % _FRINGE
    for off in range(3):
        y = f0 + off
        if y >= by1:
            break
        roi = dst[y:by1:_FRINGE, bx0:bx1]
        dst[y:by1:_FRINGE, bx0:bx1] = (roi * (0.66 + 0.11 * off)).astype(np.uint8)


def screen_radius(mesh: Mesh, scale: float) -> float:
    """Worst-case pixel radius of a mesh at this scale (for ROI sizing)."""
    r = mesh.radius
    return scale * r * (DEPTH * FOCAL / max(DEPTH - r, 0.5))


def mesh_bbox(mesh: Mesh, centre, scale: float, rot: np.ndarray,
              explode: float = 0.0):
    """Exact projected pixel bounds of a mesh - (x0, y0, x1, y1) floats.

    One matmul over the vertices, which is far cheaper than the pixels saved:
    a bounding SPHERE around a long thin model like the bracer is several
    times its real footprint, and every one of those pixels would be blurred
    and blended for nothing.
    """
    verts = mesh.V if explode <= 0.0 else mesh.V + mesh.VD * explode
    vr = verts @ rot.T
    zc = np.maximum(DEPTH - vr[:, 2], 0.35)
    k = (DEPTH * FOCAL / zc) * scale
    px = centre[0] + vr[:, 0] * k
    py = centre[1] + vr[:, 1] * k
    if not (np.isfinite(px).all() and np.isfinite(py).all()):
        return None
    return float(px.min()), float(py.min()), float(px.max()), float(py.max())


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
    box = mesh_bbox(mesh, centre, scale, rot, kw.get("explode", 0.0))
    if box is None:
        return None
    pad = 8.0
    if max(box[2] - box[0], box[3] - box[1]) > max(ww, hh) * 4.0:
        return None
    x0 = int(max(0, math.floor(box[0] - pad)))
    x1 = int(min(ww, math.ceil(box[2] + pad)))
    y0 = int(max(0, math.floor(box[1] - pad)))
    y1 = int(min(hh, math.ceil(box[3] + pad)))
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
