"""
Document scanner blueprint (CamScanner-style).

Drop-in replacement: keeps the same blueprint name (`scanner`), the same
url_prefix (/scanner) and the same template (scanner.html).

Pipeline
  1. /upload        save photo, auto-detect the page corners
  2. (browser)      user can drag the 4 corners to fix the crop
  3. /process       perspective-correct + enhance (Magic color, B&W, ...)
  4. /pdf           multi-page PDF   |  /result/<id>_out.jpg   single JPG
  Extras: /detect-frame (live camera outline)

Needs: flask, opencv-python, numpy, pillow, reportlab
"""

import io
import os
import re
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, url_for
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

scanner = Blueprint(
    "scanner",
    __name__,
    url_prefix="/scanner"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "scanner")
OUTPUT_DIR = os.path.join(BASE_DIR, "uploads", "scanner", "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024    # 25 MB per photo
MAX_FRAME_BYTES = 2 * 1024 * 1024      # live-preview frames are tiny
SRC_MAX_SIDE = 2600                    # stored original, longest edge (px)
OUT_MAX_SIDE = 2400                    # scanned page, longest edge (px)
KEEP_SECONDS = 6 * 60 * 60             # temp files are deleted after 6 h
MAX_PDF_PAGES = 60

FILTERS = ("magicpro", "magic", "document", "bw", "gray", "lighten", "original")

ID_RE = re.compile(r"^[0-9a-f]{32}$")
FILE_RE = re.compile(r"^([0-9a-f]{32})_(src|out)\.jpg$")

PDF_SIZES = {"a4": A4, "letter": letter}


# ---------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------

def _fail(message, status=400):
    return jsonify({"success": False, "error": message}), status


def _valid_id(value):
    return isinstance(value, str) and ID_RE.match(value) is not None


def _src_path(pid):
    return os.path.join(UPLOAD_DIR, pid + "_src.jpg")


def _out_path(pid):
    return os.path.join(OUTPUT_DIR, pid + "_out.jpg")


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _clamp(value, low, high):
    return max(low, min(high, value))


def _cleanup_old_files():
    """Delete scans older than KEEP_SECONDS so the folder never grows forever."""
    limit = time.time() - KEEP_SECONDS
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        try:
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if FILE_RE.match(name) and os.path.getmtime(path) < limit:
                    os.remove(path)
        except OSError:
            pass


def _decode_upload(data):
    """Bytes -> BGR image. Honours phone EXIF rotation and caps the size."""
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
    except Exception:
        return None

    width, height = pil.size
    scale = SRC_MAX_SIDE / float(max(width, height))
    if scale < 1:
        pil = pil.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.LANCZOS
        )

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@scanner.after_request
def _camera_permission(response):
    # Make sure a site-wide Permissions-Policy cannot block the camera here.
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response


@scanner.route("/")
def scanner_page():
    return render_template("scanner.html")


# ---------------------------------------------------------
# PAGE DETECTION
# ---------------------------------------------------------

DETECT_SIDE = 720


def _order_index(points):
    pts = np.array(points, dtype="float32").reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    idx = np.argsort(angles)                      # clockwise on screen
    start = int(np.argmin(pts[idx].sum(axis=1)))  # top-left = smallest x + y
    return np.roll(idx, -start)


def order_points(points):
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype="float32").reshape(4, 2)
    return pts[_order_index(pts)]


def _quad_from_contour(contour):
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    for eps in (0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32"), cv2.contourArea(hull)
        if len(approx) < 4:
            break

    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype("float32")
    return box, cv2.contourArea(hull)


def _edge_support(edge_near, quad):
    """Share of the quad outline that sits on a real image edge (0..1)."""
    h, w = edge_near.shape
    t = np.linspace(0.04, 0.96, 36, dtype=np.float32)
    samples = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        samples.append(a[None, :] + t[:, None] * (b - a)[None, :])
    pts = np.concatenate(samples)
    # where the page runs out of the frame there is no edge to find: ignore
    inside = (pts[:, 0] >= 3) & (pts[:, 0] <= w - 4) & (pts[:, 1] >= 3) & (pts[:, 1] <= h - 4)
    pts = pts[inside]
    if len(pts) < 30:
        return 0.0
    return float((edge_near[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0).mean())


def _curve_support(edge_near, ordered, mids):
    """Like _edge_support, but along the curved outline (quadratic edges)."""
    h, w = edge_near.shape
    t = np.linspace(0.04, 0.96, 36, dtype=np.float32)[:, None]
    samples = []
    for i in range(4):
        a, b = ordered[i], ordered[(i + 1) % 4]
        m = np.array(mids[i], dtype=np.float32)
        c = 2.0 * m - (a + b) / 2.0
        samples.append((1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b)
    pts = np.concatenate(samples)
    inside = (pts[:, 0] >= 3) & (pts[:, 0] <= w - 4) & (pts[:, 1] >= 3) & (pts[:, 1] <= h - 4)
    pts = pts[inside]
    if len(pts) < 30:
        return 0.0
    return float((edge_near[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0).mean())


def _score_quad(quad, hull_area, w, h, edge_near):
    """0 = reject. Higher = looks more like a photographed page."""
    ordered = order_points(quad)
    area = cv2.contourArea(ordered)
    fraction = area / float(w * h)

    if fraction < 0.10 or fraction > 0.995:
        return 0.0

    # A mask that just traces the picture frame is not a page.
    margin = 3
    on_edge = sum(
        1 for x, y in ordered
        if x <= margin or y <= margin or x >= w - 1 - margin or y >= h - 1 - margin
    )
    if on_edge >= 3:
        return 0.0

    sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 0.12 * min(w, h):
        return 0.0
    if min(sides[0], sides[2]) / max(sides[0], sides[2]) < 0.4:
        return 0.0
    if min(sides[1], sides[3]) / max(sides[1], sides[3]) < 0.4:
        return 0.0

    for i in range(4):
        a = ordered[i - 1] - ordered[i]
        b = ordered[(i + 1) % 4] - ordered[i]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        if angle < 45 or angle > 135:
            return 0.0

    support = _edge_support(edge_near, ordered)
    if support < 0.30:
        return 0.0

    fit = min(hull_area, area) / max(hull_area, area, 1.0)

    centre = ordered.mean(axis=0)
    off_centre = np.linalg.norm(centre - np.array([w / 2.0, h / 2.0])) / (0.5 * np.hypot(w, h))

    return (fraction ** 0.7) * fit * (0.25 + 0.75 * support) * (1.0 - 0.25 * min(off_centre, 1.0))


def _clean(mask, close_iter=2):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=close_iter)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _grabcut_mask(small):
    h, w = small.shape[:2]
    k = 320.0 / max(h, w)
    g = cv2.resize(small, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gh, gw = g.shape[:2]
    mask = np.zeros((gh, gw), np.uint8)
    rect = (int(gw * 0.04), int(gh * 0.04), int(gw * 0.92), int(gh * 0.92))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(12345)          # same photo -> same result every time
    cv2.grabCut(g, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def _masks(small, blur, edges, fast):
    """Several independent ways to separate the page from the background.

    Returns a list of (mask, contour-retrieval-mode)."""
    h, w = small.shape[:2]
    out = []

    # 1) closed edge outlines (inner holes = page area, outer = blob of edges)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    out.append((closed, cv2.RETR_LIST))
    out.append((closed, cv2.RETR_EXTERNAL))

    # 2) bright page on darker surface, and the reverse
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append((_clean(otsu), cv2.RETR_EXTERNAL))
    out.append((_clean(255 - otsu), cv2.RETR_EXTERNAL))

    # 3) white-ish paper: all three channels are high (coloured objects drop out)
    minc = cv2.GaussianBlur(np.min(small, axis=2), (5, 5), 0)
    _, paper = cv2.threshold(minc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append((_clean(paper), cv2.RETR_EXTERNAL))

    # 4) anything that differs in colour from the picture border
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.03 * min(h, w)))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)
    ])
    background = np.median(border, axis=0)
    dist = np.linalg.norm(lab - background, axis=2)
    dist = np.clip(dist * 255.0 / max(float(dist.max()), 1.0), 0, 255).astype(np.uint8)
    _, diff = cv2.threshold(cv2.GaussianBlur(dist, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append((_clean(diff), cv2.RETR_EXTERNAL))

    if not fast:
        # 5) colour similar to the centre of the picture (where the page usually is)
        patch = lab[int(h * 0.40):int(h * 0.60), int(w * 0.40):int(w * 0.60)].reshape(-1, 3)
        centre = np.median(patch, axis=0)
        spread = float(np.median(np.linalg.norm(patch - centre, axis=1)))
        near = (np.linalg.norm(lab - centre, axis=2) < max(14.0, 4.0 * spread)).astype(np.uint8) * 255
        out.append((_clean(near), cv2.RETR_EXTERNAL))

        # 6) GrabCut: foreground vs background model of the whole picture
        try:
            out.append((_clean(_grabcut_mask(small), 1), cv2.RETR_EXTERNAL))
        except cv2.error:
            pass

    return out


def _intersect(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return np.array([(a * (x3 - x4) - (x1 - x2) * b) / d,
                     (a * (y3 - y4) - (y1 - y2) * b) / d], dtype="float32")


def _distinct_lines(lines, min_gap):
    kept = []
    for seg in lines:
        mid = np.array([(seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0])
        d = np.array([seg[2] - seg[0], seg[3] - seg[1]], dtype=np.float64)
        d /= (np.linalg.norm(d) + 1e-9)
        duplicate = False
        for other in kept:
            od = np.array([other[2] - other[0], other[3] - other[1]], dtype=np.float64)
            od /= (np.linalg.norm(od) + 1e-9)
            if abs(d[0] * od[1] - d[1] * od[0]) > 0.1:
                continue
            gap = abs((mid[0] - other[0]) * od[1] - (mid[1] - other[1]) * od[0])
            if gap < min_gap:
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
    return kept


def _line_quad(edges, edge_near, sw, sh):
    """Build the page from the 2 best horizontal + 2 best vertical lines.

    Works when the page outline is broken or touches clutter so that no
    clean contour exists."""
    segs = cv2.HoughLinesP(
        cv2.dilate(edges, np.ones((3, 3), np.uint8)), 1, np.pi / 180, threshold=50,
        minLineLength=int(0.22 * min(sw, sh)), maxLineGap=int(0.05 * max(sw, sh))
    )
    if segs is None:
        return None, 0.0

    segs = np.asarray(segs, dtype=np.float32).reshape(-1, 4)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    horizontal = (angle < 45) | (angle > 135)
    order = np.argsort(-length)

    gap = 0.025 * min(sw, sh)
    hs = _distinct_lines([segs[i] for i in order if horizontal[i]], gap)[:8]
    vs = _distinct_lines([segs[i] for i in order if not horizontal[i]], gap)[:8]

    def ymid(s): return (s[1] + s[3]) / 2.0
    def xmid(s): return (s[0] + s[2]) / 2.0

    best_quad, best_score = None, 0.0
    for hi in range(len(hs)):
        for hj in range(hi + 1, len(hs)):
            top, bottom = (hs[hi], hs[hj]) if ymid(hs[hi]) < ymid(hs[hj]) else (hs[hj], hs[hi])
            if ymid(bottom) - ymid(top) < 0.15 * sh:
                continue
            for vi in range(len(vs)):
                for vj in range(vi + 1, len(vs)):
                    left, right = (vs[vi], vs[vj]) if xmid(vs[vi]) < xmid(vs[vj]) else (vs[vj], vs[vi])
                    if xmid(right) - xmid(left) < 0.15 * sw:
                        continue
                    corners = [_intersect(top, left), _intersect(top, right),
                               _intersect(bottom, right), _intersect(bottom, left)]
                    if any(c is None for c in corners):
                        continue
                    quad = np.array(corners, dtype="float32")
                    if (quad[:, 0] < -0.1 * sw).any() or (quad[:, 0] > 1.1 * sw).any() \
                            or (quad[:, 1] < -0.1 * sh).any() or (quad[:, 1] > 1.1 * sh).any():
                        continue
                    quad[:, 0] = np.clip(quad[:, 0], 0, sw - 1)
                    quad[:, 1] = np.clip(quad[:, 1], 0, sh - 1)
                    area = cv2.contourArea(order_points(quad))
                    score = 0.92 * _score_quad(quad, area, sw, sh, edge_near)
                    if score > best_score:
                        best_quad, best_score = quad, score
    return best_quad, best_score


def _measure_bends(image, corners_norm):
    """Trace each page side on the real image edge and fit a curve to it.

    Returns None when every side is straight, otherwise 4 side-middle points
    (normalised) lying on the traced edges. Corners are left untouched."""
    h, w = image.shape[:2]
    k = min(1.0, 1400.0 / max(h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.8).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gh, gw = gray.shape

    quad = np.array(corners_norm, dtype=np.float32) * np.array([gw, gh], dtype=np.float32)
    band = max(8, int(0.07 * min(gh, gw)))
    offsets = np.arange(-band, band + 1, dtype=np.float32)
    mids, any_curved = [], False

    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.linalg.norm(d))
        u = d / max(length, 1e-6)
        n = np.array([-u[1], u[0]], dtype=np.float32)
        mid = (a + b) / 2.0

        t = np.linspace(0.1, 0.9, 48, dtype=np.float32)
        base = a[None, :] + t[:, None] * d[None, :]
        pos = base[:, None, :] + offsets[None, :, None] * n[None, None, :]
        xs = np.rint(pos[..., 0]).astype(int)
        ys = np.rint(pos[..., 1]).astype(int)
        ok = (xs >= 1) & (xs < gw - 1) & (ys >= 1) & (ys < gh - 1)
        xs, ys = np.clip(xs, 0, gw - 1), np.clip(ys, 0, gh - 1)

        strength = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1]) * ok
        # a straight guess stays the default: nearby edges win ties
        strength = strength * (1.0 - 0.15 * np.abs(offsets)[None, :] / band)
        best = strength.argmax(axis=1)
        peak = strength[np.arange(len(t)), best]
        keep = (peak > 0.6 * np.median(peak)) & ok[np.arange(len(t)), best]

        bow = 0.0
        if keep.sum() >= 24:
            tt, oo = t[keep], offsets[best[keep]]
            basis = 4.0 * tt * (1.0 - tt)
            A = np.stack([basis, 1.0 - tt, tt], axis=1)          # bow, offset at start, offset at end
            line = np.stack([1.0 - tt, tt], axis=1)
            inlier = np.ones(len(tt), bool)
            for _ in range(3):
                coef = np.linalg.lstsq(A[inlier], oo[inlier], rcond=None)[0]
                resid = np.abs(A @ coef - oo)
                inlier = resid < max(2.5, 2.5 * np.median(resid))
                if inlier.sum() < 18:
                    break
            if inlier.sum() >= 18:
                lcoef = np.linalg.lstsq(line[inlier], oo[inlier], rcond=None)[0]
                res_curve = float(np.mean(np.abs(A[inlier] @ coef - oo[inlier])))
                res_line = float(np.mean(np.abs(line[inlier] @ lcoef - oo[inlier])))
                # a real bend is a very clean curve; clutter / weak edges never fit this well
                if abs(coef[0]) > 0.02 * length and res_curve < 1.2 and res_curve < 0.2 * res_line and inlier.mean() > 0.8:
                    bow = float(coef[0])
                    any_curved = True

        mids.append(mid + n * bow)

    if not any_curved:
        return None
    mids = np.array(mids, dtype=np.float32) / np.array([gw, gh], dtype=np.float32)
    return np.clip(mids, 0, 1)


def _refine_corners(image, corners_norm):
    """Snap the 4 sides onto the strongest real edges at higher resolution."""
    h, w = image.shape[:2]
    k = min(1.0, 1400.0 / max(h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    gh, gw = gray.shape

    quad = np.array(corners_norm, dtype=np.float32) * np.array([gw, gh], dtype=np.float32)
    band = max(6, int(0.03 * min(gh, gw)))
    offsets = np.arange(-band, band + 1, dtype=np.float32)
    lines = []

    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.linalg.norm(d))
        u = d / max(length, 1e-6)
        n = np.array([-u[1], u[0]], dtype=np.float32)

        t = np.linspace(0.08, 0.92, 60, dtype=np.float32)
        base = a[None, :] + t[:, None] * d[None, :]
        pos = base[:, None, :] + offsets[None, :, None] * n[None, None, :]
        xs = np.clip(np.rint(pos[..., 0]).astype(int), 0, gw - 1)
        ys = np.clip(np.rint(pos[..., 1]).astype(int), 0, gh - 1)

        strength = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1])
        strength = strength * (1.0 - 0.35 * np.abs(offsets)[None, :] / band)
        best = strength.argmax(axis=1)
        peak = strength[np.arange(len(t)), best]
        keep = peak > 0.5 * np.median(peak)
        if keep.sum() < 15:
            lines.append(None)
            continue

        pts = base[keep] + offsets[best[keep]][:, None] * n[None, :]
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        if (dist < 2.5).mean() < 0.5:
            lines.append(None)
            continue
        lines.append((x0 - 1000 * vx, y0 - 1000 * vy, x0 + 1000 * vx, y0 + 1000 * vy))

    refined = []
    moved = False
    for i in range(4):
        l1, l2 = lines[(i - 1) % 4], lines[i]
        point = _intersect(l1, l2) if (l1 is not None and l2 is not None) else None
        if point is None or np.linalg.norm(point - quad[i]) > 0.05 * max(gh, gw):
            refined.append(quad[i])
        else:
            refined.append(point)
            moved = True

    if not moved:
        return None
    refined = np.array(refined, dtype=np.float32) / np.array([gw, gh], dtype=np.float32)
    return np.clip(refined, 0, 1)


def detect_page(image, fast=False):
    """Find the page in a photo.

    Returns None, or {"corners": [[x,y]*4], "mids": [[x,y]*4] | None, "curved": bool}
    with everything normalised to 0..1. `mids` is only given for a bent page.
    `fast=True` is used for the live camera outline."""
    height, width = image.shape[:2]
    scale = DETECT_SIDE / float(max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blur))
    edges = cv2.bitwise_or(
        cv2.Canny(blur, max(10, 0.66 * median), min(255, 1.33 * median)),
        cv2.Canny(blur, 20, 60)
    )
    edge_near = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    best_quad, best_score, best_contour = None, 0.0, None

    for mask, mode in _masks(small, blur, edges, fast):
        contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
        for contour in contours:
            if cv2.contourArea(contour) < 0.08 * sw * sh:
                continue
            quad, hull_area = _quad_from_contour(contour)
            score = _score_quad(quad, hull_area, sw, sh, edge_near)
            if score > best_score:
                best_quad, best_score, best_contour = quad, score, contour

    if not fast:
        line_quad, line_score = _line_quad(edges, edge_near, sw, sh)
        if line_quad is not None and line_score > best_score:
            best_quad, best_score, best_contour = line_quad, line_score, None

    if best_quad is None:
        return None

    ordered = order_points(best_quad)
    mids, curved = None, False

    corners = ordered / np.array([sw, sh], dtype=np.float32)

    if not fast:
        found_mids = _measure_bends(image, corners)
        if found_mids is not None:
            curved = True
            mids = found_mids.tolist()

    if not curved and not fast:
        refined = _refine_corners(image, corners)
        if refined is not None:
            before = _edge_support(edge_near, ordered)
            after = _edge_support(edge_near, refined * np.array([sw, sh], dtype=np.float32))
            if after >= before - 0.03:
                corners = refined

    corners = np.clip(corners, 0, 1)

    return {
        "corners": [[round(float(x), 5), round(float(y), 5)] for x, y in corners],
        "mids": [[round(x, 5), round(y, 5)] for x, y in mids] if mids else None,
        "curved": bool(curved)
    }


def detect_corners(image):
    """Fast corners-only detection (live camera outline)."""
    found = detect_page(image, fast=True)
    return found["corners"] if found else None


def _parse_points(raw, count=4):
    """Validate points sent by the browser. None -> not given."""
    if raw is None:
        return None
    try:
        pts = np.array(raw, dtype="float32")
    except (TypeError, ValueError):
        raise ValueError("Corners are not valid.")
    if pts.shape != (count, 2) or not np.isfinite(pts).all():
        raise ValueError("Corners are not valid.")
    return np.clip(pts, 0.0, 1.0)


# ---------------------------------------------------------
# PERSPECTIVE CORRECTION + BENT-PAGE FLATTENING
# ---------------------------------------------------------

def _cap_size(image):
    longest = max(image.shape[:2])
    if longest > OUT_MAX_SIDE:
        ratio = OUT_MAX_SIDE / float(longest)
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    return image


def four_point_transform(image, points):
    rect = order_points(points)
    tl, tr, br, bl = rect

    max_width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    max_height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    max_width, max_height = max(max_width, 32), max(max_height, 32)

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, destination)

    warped = cv2.warpPerspective(
        image, matrix, (max_width, max_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return _cap_size(warped)


def _bezier(p0, p1, p2, t):
    t = t[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def coons_transform(image, corners, mids):
    """Flatten a bent page.

    The page outline is 4 curved edges (each passes through its two corners
    and its middle handle). Every output pixel is blended between the top/bottom
    and left/right curves, so curved text lines come out straight."""
    tl, tr, br, bl = [np.asarray(p, dtype=np.float64) for p in corners]
    mt, mr, mb, ml = [np.asarray(p, dtype=np.float64) for p in mids]

    def control(a, m, b):
        return 2.0 * m - (a + b) / 2.0

    def top(u): return _bezier(tl, control(tl, mt, tr), tr, u)
    def bottom(u): return _bezier(bl, control(bl, mb, br), br, u)
    def left(v): return _bezier(tl, control(tl, ml, bl), bl, v)
    def right(v): return _bezier(tr, control(tr, mr, br), br, v)

    def arc(curve):
        pts = curve(np.linspace(0, 1, 60))
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    width = int(round(max(arc(top), arc(bottom))))
    height = int(round(max(arc(left), arc(right))))
    ratio = min(1.0, OUT_MAX_SIDE / float(max(width, height, 1)))
    width, height = max(int(width * ratio), 32), max(int(height * ratio), 32)

    us = np.linspace(0, 1, width)
    vs = np.linspace(0, 1, height)
    u = us[None, :, None]
    v = vs[:, None, None]

    patch = (
        (1 - v) * top(us)[None, :, :] + v * bottom(us)[None, :, :]
        + (1 - u) * left(vs)[:, None, :] + u * right(vs)[:, None, :]
        - ((1 - u) * (1 - v) * tl + u * (1 - v) * tr + (1 - u) * v * bl + u * v * br)
    )

    map_x = patch[:, :, 0].astype(np.float32)
    map_y = patch[:, :, 1].astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _align_mids(raw_corners, raw_mids):
    """Corners get re-sorted (TL,TR,BR,BL); keep each middle handle on its edge.

    Returns (ordered_corners, mids_for_ordered_edges) or (ordered_corners, None)
    when the outline is twisted and the handles cannot be matched."""
    raw_corners = np.asarray(raw_corners, dtype="float32")
    idx = _order_index(raw_corners)
    ordered = raw_corners[idx]
    if raw_mids is None:
        return ordered, None

    aligned = []
    for i in range(4):
        p, q = int(idx[i]), int(idx[(i + 1) % 4])
        found = None
        for j in range(4):
            if {p, q} == {j, (j + 1) % 4}:
                found = j
                break
        if found is None:
            return ordered, None
        aligned.append(raw_mids[found])
    return ordered, np.array(aligned, dtype="float32")


# ---------------------------------------------------------
# ENHANCEMENT FILTERS
# ---------------------------------------------------------

def _flatten(image):
    """Remove shadows / uneven light by dividing out the paper's own colour.

    The paper colour is estimated on a small copy (a morphological closing
    erases thin dark text strokes), then divided out of the full image.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 400.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    k = _odd(max(9, 0.04 * max(small.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    background = cv2.medianBlur(background, 5)
    background = cv2.GaussianBlur(background, (0, 0), max(2.0, k / 4.0))
    background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)

    result = image.astype(np.float32) * 255.0 / np.maximum(background.astype(np.float32), 40.0)
    return np.clip(result, 0, 255).astype(np.uint8)


def _levels(image, low, high):
    table = np.clip((np.arange(256) - low) * 255.0 / float(high - low), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def _gamma(image, gamma):
    table = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(image, table)


def _sharpen(image, amount=0.6):
    blur = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)


def _magic_pro(image):
    """CamScanner-style "Magic Pro" colour.

    1. divide out shadows / uneven light
    2. find this page's own white and black point and stretch between them
    3. add a gentle S-curve so ink gets dark and crisp
    4. push anything that is paper-coloured to pure white (clean background)
    5. keep real colours (stamps, pens, logos) vivid, then sharpen
    """
    flat = _flatten(image)

    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    white = _clamp(float(np.percentile(gray, 88)), 150.0, 250.0)
    black = _clamp(float(np.percentile(gray, 1)), 0.0, 90.0)
    low = black + 0.06 * (white - black)
    high = max(white * 0.97, low + 40.0)
    out = _levels(flat, low, high)

    # S-curve: darker darks, brighter lights, mid-tones kept
    x = np.arange(256) / 255.0
    curve = 0.55 * (x * x * (3.0 - 2.0 * x)) + 0.45 * x
    out = cv2.LUT(out, np.clip(curve * 255.0, 0, 255).astype(np.uint8))

    # colour boost, but never on the white paper itself
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # paper -> pure white (soft edge so thin strokes do not get eaten)
    lightest = np.min(out, axis=2).astype(np.float32)
    paper = np.clip((lightest - 205.0) / 30.0, 0.0, 1.0)[:, :, None]
    out = (out.astype(np.float32) * (1.0 - paper) + 255.0 * paper).astype(np.uint8)

    return _sharpen(out, 0.9)


def apply_filter(image, filter_name, brightness=0.0, contrast=1.0):
    """brightness -100..100, contrast 0.5..2.0. Returns a BGR or gray image."""

    if filter_name == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = _flatten(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        block = _odd(max(31, max(gray.shape) // 30))
        # brightness acts as the threshold: higher = lighter page, thinner ink
        offset = _clamp(12 + brightness / 8.0, 2, 28)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, offset
        )
        return cv2.medianBlur(bw, 3)

    if filter_name == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = _levels(_flatten(gray), 30, 230)
        out = _sharpen(out)

    elif filter_name == "lighten":
        flat = _flatten(image)
        out = cv2.addWeighted(image, 0.35, flat, 0.65, 0)
        out = _gamma(out, 0.8)
        out = _sharpen(out, 0.4)

    elif filter_name == "document":
        # clean white paper, crisp dark ink, colour kept but calmer
        out = _levels(_flatten(image), 45, 215)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = _sharpen(out, 0.8)

    elif filter_name == "magicpro":
        out = _magic_pro(image)

    elif filter_name == "magic":
        out = _levels(_flatten(image), 20, 235)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = _sharpen(out)

    else:
        out = image

    if brightness or contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness + 128 * (1 - contrast))

    return out


def _paper_color(image):
    """Typical page colour: median of a ring just inside the edges (gray or BGR)."""
    h, w = image.shape[:2]
    a, b = max(2, int(0.025 * min(h, w))), max(4, int(0.08 * min(h, w)))
    ring = np.ones((h, w), bool)
    ring[b:h - b, b:w - b] = False
    ring[:a, :] = False
    ring[h - a:, :] = False
    ring[:, :a] = False
    ring[:, w - a:] = False
    return np.median(image[ring], axis=0)


def _fix_edges(image):
    """If a side has a thin band of another colour (table, hand, shadow that
    got into the crop), repaint that band in the page colour."""
    h, w = image.shape[:2]
    band = max(2, int(0.012 * min(h, w)))
    paper = _paper_color(image)
    out = image.copy()
    fill = np.clip(paper, 0, 255).astype(np.uint8)

    def differs(strip):
        diff = np.abs(np.median(strip.reshape(-1, *image.shape[2:]), axis=0) - paper)
        return float(np.max(diff)) > 22.0

    if differs(image[:band]):      out[:band] = fill
    if differs(image[h - band:]):  out[h - band:] = fill
    if differs(image[:, :band]):   out[:, :band] = fill
    if differs(image[:, w - band:]): out[:, w - band:] = fill
    return out


# ---------------------------------------------------------
# UPLOAD + AUTO DETECT
# ---------------------------------------------------------

@scanner.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("image")

    if file is None:
        return _fail("No image uploaded.")

    data = file.read()

    if not data:
        return _fail("The image is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        return _fail("That image is larger than 25 MB. Use a smaller photo.", 413)

    image = _decode_upload(data)

    if image is None:
        return _fail("That file is not a readable image.")

    _cleanup_old_files()

    pid = uuid.uuid4().hex

    cv2.imwrite(_src_path(pid), image, [cv2.IMWRITE_JPEG_QUALITY, 93])

    found = None
    try:
        found = detect_page(image)
    except Exception:
        current_app.logger.exception("Page detection failed")

    detected = found is not None

    if detected:
        corners, mids, curved = found["corners"], found["mids"], found["curved"]
    else:
        # start with a slightly inset frame so the handles are easy to grab
        corners = [[0.03, 0.03], [0.97, 0.03], [0.97, 0.97], [0.03, 0.97]]
        mids, curved = None, False

    height, width = image.shape[:2]

    return jsonify({
        "success": True,
        "id": pid,
        "width": width,
        "height": height,
        "src": url_for("scanner.result", filename=pid + "_src.jpg"),
        "corners": corners,
        "mids": mids,
        "curved": curved,
        "detected": detected
    })


@scanner.route("/detect-frame", methods=["POST"])
def detect_frame():
    """Fast detection for the live camera outline. Nothing is stored."""

    file = request.files.get("image")

    if file is None:
        return _fail("No frame.")

    data = file.read(MAX_FRAME_BYTES + 1)

    if len(data) > MAX_FRAME_BYTES:
        return _fail("Frame too large.", 413)

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return _fail("Frame is not an image.")

    try:
        corners = detect_corners(image)
    except Exception:
        corners = None

    return jsonify({"success": True, "corners": corners})


# ---------------------------------------------------------
# CROP + ENHANCE
# ---------------------------------------------------------

@scanner.route("/process", methods=["POST"])
def process_scan():

    data = request.get_json(silent=True) or {}

    pid = data.get("id")

    if not _valid_id(pid) or not os.path.exists(_src_path(pid)):
        return _fail("This photo expired. Capture or upload it again.", 404)

    filter_name = data.get("filter", "magicpro")
    if filter_name not in FILTERS:
        filter_name = "magicpro"

    try:
        brightness = _clamp(float(data.get("brightness", 0)), -100.0, 100.0)
        contrast = _clamp(float(data.get("contrast", 1)), 0.5, 2.0)
        rotate = int(data.get("rotate", 0)) % 360
        corners = _parse_points(data.get("corners"))
        mids = _parse_points(data.get("mids"))
        flatten = bool(data.get("flatten")) and mids is not None
    except (TypeError, ValueError) as error:
        return _fail(str(error) if str(error) else "Invalid settings.")

    if rotate not in (0, 90, 180, 270):
        rotate = 0

    image = cv2.imread(_src_path(pid))

    if image is None:
        return _fail("Could not read the stored photo.", 500)

    try:
        height, width = image.shape[:2]

        if corners is None:
            scanned = image
        else:
            size = np.array([width, height], dtype="float32")
            points = corners * size
            polygon_area = cv2.contourArea(order_points(points))
            if polygon_area < 0.01 * width * height:
                return _fail("The crop area is too small. Drag the corners further apart.")
            if flatten:
                ordered, edge_mids = _align_mids(points, mids * size if mids is not None else None)
                if edge_mids is None:
                    scanned = four_point_transform(image, points)
                else:
                    scanned = coons_transform(image, ordered, edge_mids)
            else:
                scanned = four_point_transform(image, points)

        if rotate == 90:
            scanned = cv2.rotate(scanned, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            scanned = cv2.rotate(scanned, cv2.ROTATE_180)
        elif rotate == 270:
            scanned = cv2.rotate(scanned, cv2.ROTATE_90_COUNTERCLOCKWISE)

        scanned = apply_filter(scanned, filter_name, brightness, contrast)
        if filter_name != "original":
            scanned = _fix_edges(scanned)

        cv2.imwrite(_out_path(pid), scanned, [cv2.IMWRITE_JPEG_QUALITY, 92])

    except Exception:
        current_app.logger.exception("Processing failed")
        return _fail("Could not process this photo.", 500)

    out_height, out_width = scanned.shape[:2]

    return jsonify({
        "success": True,
        "image": url_for("scanner.result", filename=pid + "_out.jpg"),
        "width": out_width,
        "height": out_height
    })


# ---------------------------------------------------------
# RESULT IMAGE
# ---------------------------------------------------------

@scanner.route("/result/<filename>")
def result(filename):

    match = FILE_RE.match(filename)

    if not match:
        abort(404)

    pid, kind = match.groups()

    path = _src_path(pid) if kind == "src" else _out_path(pid)

    if not os.path.exists(path):
        abort(404)

    download = request.args.get("download") == "1"

    return send_file(
        path,
        mimetype="image/jpeg",
        as_attachment=download,
        download_name="Scanned_Document.jpg",
        max_age=0
    )


@scanner.route("/delete", methods=["POST"])
def delete_page():

    data = request.get_json(silent=True) or {}
    pid = data.get("id")

    if _valid_id(pid):
        for path in (_src_path(pid), _out_path(pid)):
            try:
                os.remove(path)
            except OSError:
                pass

    return jsonify({"success": True})


# ---------------------------------------------------------
# DOWNLOAD PDF (one or many pages)
# ---------------------------------------------------------

@scanner.route("/pdf")
def create_pdf():

    ids = [i for i in request.args.get("ids", "").split(",") if _valid_id(i)][:MAX_PDF_PAGES]

    paths = []
    for pid in ids:
        if os.path.exists(_out_path(pid)):
            paths.append(_out_path(pid))
        elif os.path.exists(_src_path(pid)):
            paths.append(_src_path(pid))

    if not paths:
        return _fail("There are no scanned pages to export.", 404)

    size = request.args.get("size", "a4")
    fill = request.args.get("fill", "1") == "1"

    # Read every page size first, so ALL pages can share one size.
    dims = []
    for path in paths:
        with Image.open(path) as picture:
            dims.append(picture.size)

    if size in PDF_SIZES:
        short, long_ = sorted(PDF_SIZES[size])
        # one orientation for the whole file: whatever most pages use
        landscape = sum(1 for w, h in dims if w > h) > len(dims) / 2.0
        page_w, page_h = (long_, short) if landscape else (short, long_)
    else:
        # "Original size": 150 dpi, and every page uses the first page's size
        page_w, page_h = dims[0][0] * 72.0 / 150.0, dims[0][1] * 72.0 / 150.0

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_w, page_h))
    pdf.setTitle("Scanned Document")
    pdf.setCreator("RAUF Document Scanner")

    for path, (img_w, img_h) in zip(paths, dims):

        contain = min(page_w / img_w, page_h / img_h)
        cover = max(page_w / img_w, page_h / img_h)

        # page background = this scan's own paper colour (never a white frame
        # around a cream / grey page)
        with Image.open(path) as picture:
            small = np.array(picture.convert("RGB").resize((200, max(1, int(200 * img_h / img_w)))))
        r, g, b = [float(v) for v in _paper_color(small)]

        pdf.setPageSize((page_w, page_h))
        pdf.setFillColorRGB(r / 255.0, g / 255.0, b / 255.0)
        pdf.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        # Fill page: cover the whole sheet, but only when that trims <= 12 %
        # of the picture; otherwise fit inside (no content is ever lost).
        if fill and contain / cover >= 0.88:
            ratio = cover
        else:
            ratio = contain

        draw_w, draw_h = img_w * ratio, img_h * ratio
        pdf.drawImage(
            ImageReader(path),
            (page_w - draw_w) / 2.0,
            (page_h - draw_h) / 2.0,
            width=draw_w,
            height=draw_h
        )
        pdf.showPage()

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Scanned_Document.pdf",
        max_age=0
    )
