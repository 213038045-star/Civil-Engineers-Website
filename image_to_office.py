"""
image_to_office.py - image to Word / Excel converter (no GUI), used by routes/image_office.py
Image -> Excel / Word Extractor (table-aware)

Behaviour
---------
* Image HAS a visible table (long horizontal + vertical rules forming a grid):
    - Excel : one Excel cell per table cell, with borders.
    - Word  : a real Word table, one Word cell per table cell.
    - Pictures/logos/QR codes inside a cell go back into that same cell.
* Image has NO visible table:
    - Excel : text only, NO borders (Excel is always a grid, but nothing is
              drawn as a table); pictures go on an "Objects" sheet.
    - Word  : normal paragraphs, NO table at all; pictures placed below.

Install once:
    python -m pip install rapidocr-onnxruntime opencv-python openpyxl python-docx pillow
"""

import difflib
import os
import re
import statistics
import io
from pathlib import Path

import cv2
import numpy as np

LOW_CONF = 0.90
_ENGINE = None


# ============================================================================ IO
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    source_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if source_img is None:
        raise ValueError(f"Cannot open image: {path}")
    img = source_img
    scale = 1.0
    if img.shape[1] < 1800:
        scale = min(2.0, 1800 / img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img, source_img, scale


# ============================================================================ OCR
def _word_segments(gray, box):
    """
    Look at the real pixels of one OCR line and cut it wherever there is a
    clear white gap (= a space between words).  Returns (x0, x1, y0, y1,
    [(a, b), ...]) with segment columns relative to x0, or None.
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    H, W = gray.shape[:2]
    x0, x1 = max(0, int(min(xs))), min(W, int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(H, int(max(ys)) + 1)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return None
    crop = gray[y0:y1, x0:x1]
    _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rows = np.where((bw > 0).sum(axis=1) >= 1)[0]          # real text height
    if len(rows) == 0:
        return None
    text_h = rows[-1] - rows[0] + 1
    cols = (bw > 0).sum(axis=0) >= 1                         # ink per column
    ink = np.where(cols)[0]
    if len(ink) == 0:
        return None
    first, last = ink[0], ink[-1]

    # typical letter height (ignores dots, commas, hyphens)
    n_cc, _lab, st, _c = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [int(st[i][3]) for i in range(1, n_cc)
          if st[i][4] >= 4 and st[i][3] >= 0.35 * text_h]
    eh = float(np.median(hs)) if hs else float(text_h)

    # all white gaps between first and last ink column
    gaps, c0 = [], None
    for i in range(first, last + 1):
        if not cols[i]:
            if c0 is None:
                c0 = i
        elif c0 is not None:
            gaps.append((c0, i))
            c0 = None
    if not gaps:
        return x0, x1, y0, y1, [(first, last + 1)]
    lens = [b - a for a, b in gaps]
    med = float(np.median(lens))
    # a space is clearly wider than the usual letter gap of THIS line
    # (so letter-spaced headings and digits like "0188" are not cut up)
    thr = max(3.0, 0.33 * eh, 2.0 * med)
    passed = [ln for ln in lens if ln >= thr]
    if passed:                      # stray small gaps must not count as spaces
        thr = max(thr, 0.45 * float(np.median(passed)))
    segs, start = [], first
    for (a, b), ln in zip(gaps, lens):
        if ln >= thr:
            segs.append((start, a))
            start = b
    segs.append((start, last + 1))
    return x0, x1, y0, y1, segs


_NARROW = set("iljtf.,;:'!|I()[]/\\ ")
_WIDE = set("mwMW@%")


def _char_w(c):
    if c in _NARROW:
        return 0.45
    if c in "r-\u2013\"":
        return 0.62
    if c in _WIDE:
        return 1.6
    if c.isupper():
        return 1.15
    return 1.0


def _cuts_for(text, widths):
    """Character indices (in `text`, no spaces) where the pieces start, or None."""
    placed = _place_spaces(text.replace(" ", ""), widths)
    if not placed:
        return None
    cuts, n = [], 0
    for part in placed.split(" ")[:-1]:
        n += len(part)
        cuts.append(n)
    return cuts


def split_cells(box, text, gray, conf, stroke, ink_h):
    """
    One OCR line sometimes spans two separate cells ("= Assumed" | "Punching").
    Cut it at gaps far wider than a word space.  Returns a list of pieces.
    """
    try:
        found = _word_segments(gray, box)
        if not found:
            return None
        x0, _x1, y0, y1, segs = found
        if len(segs) < 2:
            return None
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        hs = float(y1 - y0)
        big = [i for i, g in enumerate(gaps) if g >= max(1.5 * hs, 8)]
        if not big or len(text.replace(" ", "")) < len(big) + 1:
            return None
        # merge the segments between big gaps into groups
        groups, cur = [], [segs[0]]
        for i in range(len(segs) - 1):
            if i in big:
                groups.append(cur)
                cur = [segs[i + 1]]
            else:
                cur.append(segs[i + 1])
        groups.append(cur)
        widths = [sum(b - a for a, b in g) for g in groups]
        cuts = _cuts_for(text, widths)
        if not cuts:
            return None
        bare = text.replace(" ", "")
        pieces, prev = [], 0
        for k, c in enumerate(cuts + [len(bare)]):
            pieces.append(bare[prev:c])
            prev = c
        # restore the normal spaces inside every piece (re-run on its own crop)
        out = []
        for g, txt in zip(groups, pieces):
            gx0, gx1 = x0 + g[0][0], x0 + g[-1][1]
            sub = [[gx0, y0], [gx1, y0], [gx1, y1], [gx0, y1]]
            txt = fix_text(_restore_spaces(gray, sub, txt))
            out.append({"text": txt, "conf": conf, "x0": float(gx0), "x1": float(gx1),
                        "y0": float(y0), "y1": float(y1), "stroke": stroke, "ink_h": ink_h})
        return out
    except Exception:
        return None


def _place_spaces(text, widths, forced=()):
    """
    Put (len(widths)-1) spaces into `text` (no spaces inside) so that the pieces
    have the same relative widths as the measured word widths in the image.
    Dynamic programming over the split positions; the text itself is untouched.
    """
    chars = list(text)
    n, k = len(chars), len(widths)
    if k < 2 or n < k:
        return None
    cw = np.cumsum([0.0] + [_char_w(c) for c in chars])
    tot_c = cw[-1]
    tot_w = float(sum(widths))
    if tot_c <= 0 or tot_w <= 0:
        return None
    meas = [w / tot_w for w in widths]
    INF = 1e18
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n - (k - j) + 1):
            for h in range(j - 1, i):
                if best[j - 1][h] >= INF:
                    continue
                pred = (cw[i] - cw[h]) / tot_c
                cost = best[j - 1][h] + (pred - meas[j - 1]) ** 2
                if j > 1 and h in forced:
                    cost -= 0.5                         # OCR already saw a space here
                if cost < best[j][i]:
                    best[j][i], back[j][i] = cost, h
    cuts, i = [], n
    for j in range(k, 0, -1):
        cuts.append(i)
        i = back[j][i]
    cuts = sorted(cuts)[:-1]
    out, prev = [], 0
    for c in cuts + [n]:
        out.append("".join(chars[prev:c]))
        prev = c
    return " ".join(out)


def _restore_spaces(gray, box, text):
    """
    OCR often glues words together ("AIRCONDITIONINGSYSTEM").  Measure the real
    white gaps in the picture; when there are more gaps than spaces in the
    text, re-place the spaces so every word has the width seen in the image.
    """
    try:
        found = _word_segments(gray, box)
        if not found:
            return text
        _x0, _x1, _y0, _y1, segs = found
        if len(segs) < 2 or text.count(" ") * 2 > len(segs) - 1:
            return text            # OCR already has most spaces: do not risk it
        forced, pos = set(), 0
        for ch in text:
            if ch == " ":
                forced.add(pos)
            else:
                pos += 1
        bare = text.replace(" ", "")
        fixed = _place_spaces(bare, [b - a for a, b in segs], forced)
        return fixed or text
    except Exception:
        return text


def fix_text(t):
    """Small repairs for typical OCR slips."""
    t = re.sub(r"(\d{2})[xX]{2}", r"\1XX", t)                    # 20xX -> 20XX
    t = re.sub(r"(?<=\s)[Il1|]\s?(?=20(?:\d\d|XX))", "| ", t)      # I20XX -> | 20XX
    t = re.sub(r"(20(?:\d\d|XX))\s*-\s*(?=20(?:\d\d|XX)|present)",
               "\\1 \u2013 ", t, flags=re.I)                      # 20XX-20XX
    t = re.sub(r"\s-\s?(?=present\b)", " \u2013 ", t, flags=re.I)  # -present
    def up(m):                                                    # COnTACT -> CONTACT
        w = m.group(0)
        low = sum(c.islower() for c in w)
        return w.upper() if len(w) >= 5 and low == 1 and w.isalpha() else w
    t = re.sub(r"Linkedl\s?n", "LinkedIn", t)
    return re.sub(r"[A-Za-z]+", up, t)


def ocr_items(img):
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()

    result, _ = _ENGINE(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    items = []
    for r in (result or []):
        box, text, conf = r[0], r[1], r[2]
        text = str(text).strip()
        if not text:
            continue
        text = fix_text(_restore_spaces(gray, box, text))
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        stroke_ = _stroke_ratio(gray, min(xs), max(xs), min(ys), max(ys))
        ink_ = _ink_height(gray, min(xs), max(xs), min(ys), max(ys))
        parts = split_cells(box, text, gray, float(conf), stroke_, ink_)
        if parts and len(parts) > 1:
            items.extend(parts)
            continue
        items.append({"text": text, "conf": float(conf),
                      "x0": min(xs), "x1": max(xs),
                      "y0": min(ys), "y1": max(ys),
                      "stroke": _stroke_ratio(gray, min(xs), max(xs), min(ys), max(ys)),
                      "ink_h": _ink_height(gray, min(xs), max(xs), min(ys), max(ys))})
    return items


def _ink_height(gray, x0, x1, y0, y1):
    """Real height of the printed letters (OCR boxes are padded)."""
    try:
        crop = gray[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rows = np.where((bw > 0).sum(axis=1) >= 1)[0]
        return float(rows[-1] - rows[0] + 1) if len(rows) else float(y1 - y0)
    except Exception:
        return float(y1 - y0)


def _stroke_ratio(gray, x0, x1, y0, y1):
    """Pen thickness / text height (bold text has a clearly higher value)."""
    try:
        crop = gray[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = float((bw > 0).sum())
        cnts, _h = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        per = sum(len(c) for c in cnts) / 2.0
        return (2.0 * ink / per) / max(1.0, y1 - y0) if per else 0.0
    except Exception:
        return 0.0


def to_source_coords(items, scale):
    out = []
    for it in items:
        q = dict(it)
        for k in ("x0", "x1", "y0", "y1", "ink_h"):
            if k in it:
                q[k] = it[k] / scale
        out.append(q)
    return out


def build_rows(items):
    if not items:
        return []
    h = statistics.median([max(1.0, i["y1"] - i["y0"]) for i in items])
    rows = []
    for it in sorted(items, key=lambda i: (i["y0"] + i["y1"]) / 2):
        cy = (it["y0"] + it["y1"]) / 2
        if rows and abs(cy - rows[-1]["cy"]) < max(8, h * 0.65):
            rows[-1]["cells"].append(it)
            n = len(rows[-1]["cells"])
            rows[-1]["cy"] = (rows[-1]["cy"] * (n - 1) + cy) / n
        else:
            rows.append({"cy": cy, "cells": [it]})
    for r in rows:
        r["cells"].sort(key=lambda c: c["x0"])
    return rows


def build_columns(rows, img_width):
    ranges = sorted((c["x0"], c["x1"]) for r in rows for c in r["cells"]
                    if c["x1"] - c["x0"] < img_width * 0.35)
    cols = []
    for a, b in ranges:
        if cols and a <= cols[-1][1] + 4:
            cols[-1][1] = max(cols[-1][1], b)
        else:
            cols.append([a, b])
    return cols or [[0, img_width]]


def col_index(cell, cols):
    mid = (cell["x0"] + cell["x1"]) / 2
    for i, (a, b) in enumerate(cols):
        if a <= mid <= b:
            return i + 1
    return min(range(len(cols)),
               key=lambda i: abs(((cols[i][0] + cols[i][1]) / 2) - mid)) + 1


# ====================================================================== COLORS
def _hex(bgr):
    if bgr is None:
        return None
    b, g, r = [int(round(v)) for v in bgr]
    return "%02X%02X%02X" % (r, g, b)


_PAGE_BG = None     # colour of the paper, set once per image
_PAGE_TOL = 60      # how close to the paper colour still counts as 'no colour'


def is_near_white(h):
    if not h:
        return True
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    if all(v >= 245 for v in rgb):
        return True
    if _PAGE_BG is not None:       # grey/yellow scanned paper is not a "cell colour"
        return sum(abs(a - b) for a, b in zip(rgb, _PAGE_BG)) <= _PAGE_TOL
    return False


def is_near_black(h):
    if not h:
        return True
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = max(r, g, b) - min(r, g, b)
    # dark grey text of a scan counts as ordinary black text
    return lum < 70 or (chroma < 30 and lum < 80)


def dominant_color(region):
    if region is None or region.size == 0:
        return None
    pixels = region.reshape(-1, 3).astype(np.uint8)
    if len(pixels) > 50000:
        idx = np.linspace(0, len(pixels) - 1, 50000).astype(int)
        pixels = pixels[idx]
    q = (pixels // 16).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    key = int(np.argmax(np.bincount(keys, minlength=4096)))
    chosen = pixels[keys == key]
    if len(chosen) == 0:
        return np.median(pixels, axis=0)
    return np.median(chosen, axis=0)


def sample_text_colors(img, it, scale=1.0, bg_hint=None):
    H, W = img.shape[:2]
    x0 = max(0, int(it["x0"] / scale))
    x1 = min(W, int(it["x1"] / scale) + 1)
    y0 = max(0, int(it["y0"] / scale))
    y1 = min(H, int(it["y1"] / scale) + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None

    pad = max(3, int((y1 - y0) * 0.75))
    ox0, ox1 = max(0, x0 - pad), min(W, x1 + pad)
    oy0, oy1 = max(0, y0 - pad), min(H, y1 + pad)
    outer = img[oy0:oy1, ox0:ox1]
    inner_mask = np.zeros(outer.shape[:2], np.uint8)
    inner_mask[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = 255
    ring = outer[inner_mask == 0]
    if len(ring) < 10:
        ring = outer.reshape(-1, 3)

    bg = dominant_color(ring)
    if bg_hint:
        bg = np.array([int(bg_hint[4:6], 16), int(bg_hint[2:4], 16), int(bg_hint[0:2], 16)])
    if bg is None:
        return None, None

    inside = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
    bg_i = np.asarray(bg, dtype=np.int16)
    dist = np.abs(inside - bg_i).sum(axis=1)
    fg_pixels = inside[dist > 110]
    fg = None
    if len(fg_pixels) >= 3:
        d = np.abs(fg_pixels - bg_i).sum(axis=1)
        fg = np.median(fg_pixels[d >= np.percentile(d, 75)], axis=0)

    bg_hex = _hex(bg)
    fg_hex = _hex(fg) if fg is not None else None
    if is_near_white(bg_hex):
        bg_hex = None
    if is_near_black(fg_hex):
        fg_hex = None
    return bg_hex, fg_hex


def sample_patch_color(img, x, y, half=5):
    H, W = img.shape[:2]
    x = int(min(max(x, 0), W - 1))
    y = int(min(max(y, 0), H - 1))
    patch = img[max(0, y - half):min(H, y + half + 1),
                max(0, x - half):min(W, x + half + 1)]
    return _hex(dominant_color(patch))


# ============================================================ TABLE DETECTION
def _cluster_positions(values, max_gap=4):
    values = sorted(int(v) for v in values)
    groups = []
    for v in values:
        if not groups or v - groups[-1][-1] > max_gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [int(round(sum(g) / len(g))) for g in groups]


def detect_tables(source_img):
    """
    Find EVERY real table in the image (each is a separate connected group of
    horizontal + vertical rules).  Returns a list of table dicts; an empty
    list means the image has no table.
    """
    H, W = source_img.shape[:2]
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    hm = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 15), 1)))
    vm = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, H // 30))))

    lines = cv2.dilate(cv2.bitwise_or(hm, vm), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    tables = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if w < 80 or h < 30:
            continue
        pad = 4
        rx0, ry0 = max(0, x - pad), max(0, y - pad)
        rx1, ry1 = min(W, x + w + pad), min(H, y + h + pad)

        hp = (hm[ry0:ry1, rx0:rx1] > 0).sum(axis=1)
        vp = (vm[ry0:ry1, rx0:rx1] > 0).sum(axis=0)
        if hp.max() == 0 or vp.max() == 0:
            continue

        ys = np.where(hp >= max(25, 0.6 * hp.max(), 0.35 * (rx1 - rx0)))[0]
        ys = [v + ry0 for v in _cluster_positions(ys)]
        if len(ys) < 2:
            continue
        # A long page-wide column line must not hide the short dividers of a
        # header box: count vertical rules only between first and last row rule.
        band = (vm[max(0, ys[0] - 3):ys[-1] + 4, rx0:rx1] > 0).sum(axis=0)
        span = ys[-1] - ys[0]
        xs = np.where(band >= max(20, 0.6 * min(span, band.max())))[0]
        xs = [v + rx0 for v in _cluster_positions(xs)]
        if len(xs) < 2:
            continue

        n_rows, n_cols = len(ys) - 1, len(xs) - 1
        if n_rows * n_cols < 2:          # a single box is not a table
            continue

        found = 0
        for yy in ys:
            for xx in xs:
                hs = hm[max(0, yy - 4):yy + 5, max(0, xx - 4):xx + 5]
                vs = vm[max(0, yy - 4):yy + 5, max(0, xx - 4):xx + 5]
                if hs.any() and vs.any():
                    found += 1
        if found < 0.6 * len(xs) * len(ys):
            continue

        cells = []
        for r in range(n_rows):
            for c in range(n_cols):
                cells.append({"row": r + 1, "col": c + 1,
                              "x0": xs[c] + 2, "y0": ys[r] + 2,
                              "x1": xs[c + 1] - 2, "y1": ys[r + 1] - 2})
        tables.append({"bbox": (xs[0], ys[0], xs[-1], ys[-1]),
                       "xs": xs, "ys": ys, "cells": cells,
                       "n_rows": n_rows, "n_cols": n_cols})

    tables.sort(key=lambda t: t["bbox"][1])
    return tables


# ====================================================================== ZONES
def detect_zones(source_img):
    """
    Side-by-side areas with a different background colour (e.g. the beige
    sidebar of a CV next to the white main column).  [] when the page has a
    single background.
    """
    H, W = source_img.shape[:2]
    col = np.median(source_img, axis=0).astype(np.float32)        # W x 3
    segs, start = [], 0
    for x in range(1, W + 1):
        if x == W or np.abs(col[x] - col[start:x].mean(axis=0)).sum() > 14:
            segs.append([start, x])
            start = x
    min_w = max(8, int(0.03 * W))
    segs = [sg for sg in segs if sg[1] - sg[0] >= min_w]       # drop thin rules
    merged = []
    for a, b in segs:
        c = np.median(col[a:b], axis=0)
        if merged and np.abs(merged[-1][2] - c).sum() <= 14:
            merged[-1][1], merged[-1][2] = b, (merged[-1][2] + c) / 2
        else:
            merged.append([a, b, c])
    if len(merged) < 2:
        return []
    zones = []
    for i, (a, b, c) in enumerate(merged):
        hx = _hex(c)
        zones.append({"x0": a if i else a, "x1": b,
                      "fill": None if min(c) >= 250 else hx})
    if all(z["fill"] is None for z in zones):
        return []
    zones[0]["x0"] = merged[0][0]
    zones[-1]["x1"] = max(zones[-1]["x1"], W)
    return zones


def find_bullet2(gray, zone_x0, row):
    """(bullet x, real text start x) -- a dot left of / at the start of a line."""
    bx = find_bullet(gray, zone_x0, row)
    if bx is not None:
        return bx, row["x0"]
    h = row["y1"] - row["y0"]
    xa, xb = int(row["x0"]), int(row["x0"] + 3.0 * h)
    ya, yb = int(row["y0"] + 0.12 * h), int(row["y1"] - 0.12 * h)
    if xb <= xa + 6 or yb <= ya + 3:
        return None, row["x0"]
    reg = gray[ya:yb, max(0, xa):xb]
    dark = reg < float(np.median(reg)) - 70
    cols = dark.any(axis=0)
    idx = np.where(cols)[0]
    if len(idx) == 0:
        return None, row["x0"]
    a = int(idx[0])
    b = a
    while b < len(cols) and cols[b]:
        b += 1
    rows_ = np.where(dark[:, a:b].any(axis=1))[0]
    nxt = np.where(cols[b:])[0]
    if len(nxt) == 0:
        return None, row["x0"]
    c = b + int(nxt[0])
    if (b - a) <= 0.55 * h and (rows_.max() - rows_.min() + 1) <= 0.55 * h and \
            (c - b) >= 0.3 * h and (b - a) >= 3:
        return float(max(0, xa) + a), float(max(0, xa) + c)
    return None, row["x0"]


def find_bullet(gray, zone_x0, row):
    """x of a bullet dot just left of a text line, or None."""
    h = row["y1"] - row["y0"]
    xa = int(max(zone_x0 + 2, row["x0"] - 2.4 * h))
    xb = int(row["x0"] - 0.35 * h)
    ya, yb = int(row["y0"] + 0.15 * h), int(row["y1"] - 0.15 * h)
    if xb - xa < 4 or yb - ya < 3:
        return None
    reg = gray[ya:yb, xa:xb]
    bg = float(np.median(reg))
    ys, xs = np.where(reg < bg - 70)
    if len(xs) < 5:
        return None
    # a bullet is a small dot close to the text (icons / pictures are bigger)
    if (xs.max() - xs.min() + 1) > 0.55 * h or (ys.max() - ys.min() + 1) > 0.55 * h:
        return None
    if row["x0"] - (xa + xs.max()) > 1.8 * h:
        return None
    return float(xa + xs.min())


def style_items(items):
    """Size (px), bold flag for every text piece; similar sizes are snapped together."""
    if not items:
        return
    med_h = statistics.median([i["y1"] - i["y0"] for i in items])
    med_s = statistics.median([i.get("stroke", 0) for i in items])
    bold_thr = max(0.2, 1.22 * med_s)
    for it in items:
        it["size"] = it["y1"] - it["y0"]
        it["bold"] = bool(it.get("stroke", 0) >= bold_thr or it["size"] >= 1.6 * med_h)
        t_ = it["text"]
        desc = bool(re.search(r"[gjpqy,;()@/|]", t_))
        asc = bool(re.search(r"[A-Zbdfhklt0-9]", t_))
        k_ = 0.93 if (desc and asc) else 0.73 if asc else 0.68 if desc else 0.55
        it["fpx"] = it.get("ink_h", it["size"]) / k_
    group = []
    for it in sorted(items, key=lambda i: i["fpx"]) + [None]:
        if it is not None and (not group or it["fpx"] <= group[-1]["fpx"] * 1.18):
            group.append(it)
            continue
        if group:
            med = statistics.median([g["fpx"] for g in group])
            for g in group:
                g["size"] = med
        group = [it] if it is not None else []


def digital_page(source_img):
    """
    Screenshots / exported charts have a few exact flat colours; photos and
    scans never do.  Returns (is_digital, background_hex_or_None_if_white).
    """
    sm = source_img[::3, ::3].reshape(-1, 3).astype(np.int32)
    key = (sm[:, 0] << 16) | (sm[:, 1] << 8) | sm[:, 2]
    u, c = np.unique(key, return_counts=True)
    k = int(u[int(np.argmax(c))])
    frac = float(c.max()) / len(key)
    if frac < 0.15:
        return False, None
    hx = _hex(((k >> 16) & 255, (k >> 8) & 255, k & 255))
    white = min(int(hx[i:i + 2], 16) for i in (0, 2, 4)) >= 245
    return True, (None if white else hx)


def flat_page_color(source_img):
    return digital_page(source_img)[1]


def detect_fill_rects(source_img, page_hex):
    """Solid coloured rectangles (cells, bars, boxes) of a digital image."""
    H, W = source_img.shape[:2]
    page = np.array([int(page_hex[4:6], 16), int(page_hex[2:4], 16), int(page_hex[0:2], 16)]) \
        if page_hex else np.array([255, 255, 255])
    sm = source_img[::2, ::2].reshape(-1, 3).astype(np.int32)
    key = (sm[:, 0] << 16) | (sm[:, 1] << 8) | sm[:, 2]
    u, c = np.unique(key, return_counts=True)
    order = np.argsort(-c)
    rects = []
    done = 0
    for idx in order:
        if c[idx] < 60 or done >= 14:
            break
        k = int(u[idx])
        col = np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255])
        lum = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2]
        if np.abs(col - page).sum() <= 12 or lum < 60:
            continue
        done += 1
        mask = (np.abs(source_img.astype(np.int32) - col).sum(axis=2) <= 14).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            if w < 18 or h < 8 or area / float(w * h) < 0.88:
                continue
            rects.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h,
                          "fill": _hex(col)})
    return rects


def detect_flat_pictures(source_img, page_hex, fills, items_src, tables, boxes):
    """Drawings / logos on a digital page: whatever is left after cells, boxes and text."""
    H, W = source_img.shape[:2]
    page = np.array([int(page_hex[4:6], 16), int(page_hex[2:4], 16), int(page_hex[0:2], 16)]) \
        if page_hex else np.array([255, 255, 255])
    diff = (np.abs(source_img.astype(np.int32) - page).sum(axis=2) > 40).astype(np.uint8) * 255
    for f in fills:
        diff[max(0, f["y0"] - 2):f["y1"] + 3, max(0, f["x0"] - 2):f["x1"] + 3] = 0
    for it in items_src:
        diff[max(0, int(it["y0"]) - 3):int(it["y1"]) + 4,
             max(0, int(it["x0"]) - 3):int(it["x1"]) + 4] = 0
    for t in tables:
        x0, y0, x1, y1 = t["bbox"]
        diff[max(0, y0 - 5):y1 + 6, max(0, x0 - 5):x1 + 6] = 0
    for b in boxes:
        diff[max(0, b["y0"] - 5):b["y1"] + 6, max(0, b["x0"] - 5):b["x1"] + 6] = 0
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    merged = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    objs = []
    for i in range(1, n):
        mask = (labels == i) & (diff > 0)
        if int(mask.sum()) < 250:
            continue
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        if x1 - x0 < 24 or y1 - y0 < 24 or (x1 - x0) > 0.9 * W and (y1 - y0) > 0.9 * H:
            continue
        if mask.sum() < 0.08 * (x1 - x0) * (y1 - y0):      # faint grid lines, not a picture
            continue
        if (x0 <= 1 or y0 <= 1 or x1 >= W - 1 or y1 >= H - 1) and \
                min(x1 - x0, y1 - y0) < 70:                     # cut-off screenshot edge
            continue
        # labels that belong to the drawing travel with it
        grow = True
        while grow:
            grow = False
            for it in items_src:
                if it.get("_pic"):
                    continue
                ix0, ix1, iy0, iy1 = it["x0"], it["x1"], it["y0"], it["y1"]
                ov = max(0, min(x1 + 14, ix1) - max(x0 - 14, ix0)) * \
                    max(0, min(y1 + 14, iy1) - max(y0 - 14, iy0))
                if ov >= 0.5 * max(1.0, (ix1 - ix0) * (iy1 - iy0)):
                    it["_pic"] = True
                    x0, x1 = int(min(x0, ix0)), int(max(x1, ix1))
                    y0, y1 = int(min(y0, iy0)), int(max(y1, iy1))
                    grow = True
        x0, y0, x1, y1 = max(0, x0 - 4), max(0, y0 - 4), min(W, x1 + 4), min(H, y1 + 4)
        objs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                     "crop": source_img[y0:y1, x0:x1].copy(), "row": None, "col": None,
                     "table": None, "kind": "picture", "keep_bg": True})
    objs.sort(key=lambda o: (o["y"], o["x"]))
    for k, o in enumerate(objs, 1):
        o["id"] = k
    return objs


# ============================================================ BOXES AND ARROWS
def detect_boxes(source_img, tables):
    """
    Closed outlines that are not part of a table: flow-chart boxes, framed
    notes, single bordered cells.  A box = the empty area enclosed by one
    connected outline (rounded corners are fine).
    """
    H, W = source_img.shape[:2]
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if w < 45 or h < 24 or w > 0.97 * W or h > 0.97 * H:
            continue
        if area > 0.45 * w * h:                   # a filled blob, not an outline
            continue
        mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        mask = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for k, c in enumerate(cnts):
            if hier[0][k][3] == -1:
                continue                          # outer contour; we want holes
            hx, hy, hw, hh = cv2.boundingRect(c)
            if hw < 36 or hh < 18:
                continue
            if cv2.contourArea(c) / float(hw * hh) < 0.80:
                continue
            gx, gy = x + hx - 2, y + hy - 2        # hole position in the image
            cx, cy = gx + hw / 2.0, gy + hh / 2.0
            if any(t["bbox"][0] - 3 <= cx <= t["bbox"][2] + 3 and
                   t["bbox"][1] - 3 <= cy <= t["bbox"][3] + 3 for t in tables):
                continue
            pad = 2
            bx0, by0 = max(0, gx - pad), max(0, gy - pad)
            bx1, by1 = min(W, gx + hw + pad), min(H, gy + hh + pad)
            patch = source_img[gy + 3:gy + min(hh - 3, 9), gx + 3:gx + min(hw - 3, 9)]
            fill = _hex(dominant_color(patch)) if patch.size else None
            if is_near_white(fill):
                fill = None
            boxes.append({"x0": bx0, "y0": by0, "x1": bx1, "y1": by1, "fill": fill})
    boxes.sort(key=lambda b: -(b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))
    keep = []
    for b in boxes:                               # drop double rules
        if not any(abs(b["x0"] - k["x0"]) < 8 and abs(b["x1"] - k["x1"]) < 8 and
                   abs(b["y0"] - k["y0"]) < 8 and abs(b["y1"] - k["y1"]) < 8 for k in keep):
            keep.append(b)
    keep.sort(key=lambda b: (b["y0"], b["x0"]))
    return keep


def detect_arrows(source_img, items_src, tables, boxes):
    """Arrows between boxes: thin elongated ink with a head that touches a box."""
    if len(boxes) < 2:
        return []
    H, W = source_img.shape[:2]
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 14)
    for it in items_src:                                    # text
        ink[max(0, int(it["y0"]) - 3):int(it["y1"]) + 4,
            max(0, int(it["x0"]) - 3):int(it["x1"]) + 4] = 0
    for b in boxes:                                         # the box outlines
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        for (a0, a1, c0, c1) in ((y0 - 4, y0 + 5, x0 - 4, x1 + 5), (y1 - 4, y1 + 5, x0 - 4, x1 + 5),
                                 (y0 - 4, y1 + 5, x0 - 4, x0 + 5), (y0 - 4, y1 + 5, x1 - 4, x1 + 5)):
            ink[max(0, a0):a1, max(0, c0):c1] = 0
    for t in tables:
        x0, y0, x1, y1 = t["bbox"]
        ink[max(0, y0 - 5):y1 + 6, max(0, x0 - 5):x1 + 6] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    def near_box(x, y, r=16):
        return any(b["x0"] - r <= x <= b["x1"] + r and b["y0"] - r <= y <= b["y1"] + r
                   for b in boxes)

    arrows = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        long_, short_ = max(w, h), min(w, h)
        if long_ < 14 or short_ > 24 or long_ < 1.5 * short_ or area < 30:
            continue
        mask = (labels[y:y + h, x:x + w] == i)
        if h >= w:                                           # vertical arrow
            prof = mask.sum(axis=1).astype(float)
            k = max(2, h // 3)
            head_top, head_bot = prof[:k].max(), prof[-k:].max()
            shaft = max(1.0, float(np.median(prof[prof > 0])))
            d = "down" if head_bot >= head_top else "up"
            head = max(head_top, head_bot)
            ends = [(x + w / 2.0, y), (x + w / 2.0, y + h)]
        else:
            prof = mask.sum(axis=0).astype(float)
            k = max(2, w // 3)
            head_l, head_r = prof[:k].max(), prof[-k:].max()
            shaft = max(1.0, float(np.median(prof[prof > 0])))
            d = "right" if head_r >= head_l else "left"
            head = max(head_l, head_r)
            ends = [(x, y + h / 2.0), (x + w, y + h / 2.0)]
        if head < 2.0 * shaft or head < 5:
            continue                                         # no arrow head
        if not any(near_box(*e) for e in ends):
            continue
        arrows.append({"x": x + w / 2.0, "y": y + h / 2.0, "x0": x, "x1": x + w,
                       "y0": y, "y1": y + h, "dir": d})
    return arrows


def recheck_empty_cells(source_img, tables, inside):
    """
    A table cell that contains ink but no recognised text (a lone digit, a tiny
    value) is read again on its own, enlarged.  Returns the new text pieces.
    """
    new_items = []
    if _ENGINE is None:
        return new_items
    for ti, t in enumerate(tables):
        have = set()
        for it in inside[ti]:
            cx, cy = (it["x0"] + it["x1"]) / 2, (it["y0"] + it["y1"]) / 2
            for c in t["cells"]:
                if c["x0"] - 3 <= cx <= c["x1"] + 3 and c["y0"] - 3 <= cy <= c["y1"] + 3:
                    have.add((c["row"], c["col"]))
                    break
        sizes = [i["size"] for i in inside[ti] if "size" in i] or [14]
        for c in t["cells"]:
            if (c["row"], c["col"]) in have:
                continue
            x0, y0, x1, y1 = c["x0"] + 2, c["y0"] + 2, c["x1"] - 2, c["y1"] - 2
            if x1 - x0 < 10 or y1 - y0 < 8:
                continue
            crop = source_img[y0:y1, x0:x1]
            med = np.median(crop.reshape(-1, 3), axis=0)
            if (np.abs(crop.astype(np.int32) - med).sum(axis=2) > 90).sum() < 10:
                continue
            bgc = [int(v) for v in med]
            pad = cv2.copyMakeBorder(crop, 16, 16, 24, 24, cv2.BORDER_CONSTANT, value=bgc)
            big = cv2.resize(pad, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            try:
                res, _ = _ENGINE(big)
            except Exception:
                continue
            for r in res or []:
                text, conf = str(r[1]).strip(), float(r[2])
                if not text or conf < 0.5:
                    continue
                xs = [p[0] / 3.0 - 24 + x0 for p in r[0]]
                ys = [p[1] / 3.0 - 16 + y0 for p in r[0]]
                h = max(ys) - min(ys)
                new_items.append({"text": fix_text(text), "conf": conf, "x0": min(xs),
                                  "x1": max(xs), "y0": min(ys), "y1": max(ys), "stroke": 0.0,
                                  "ink_h": h, "size": statistics.median(sizes),
                                  "bold": False, "fpx": h})
                inside[ti].append(new_items[-1])
    return new_items


# ==================================================================== GRID BUILD
def grid_from_table(items_src, cells, img, match_colors, skip_keys=()):
    """Put every OCR text into the REAL table cell that contains it."""
    buckets = {}
    for it in items_src:
        cx = (it["x0"] + it["x1"]) / 2
        cy = (it["y0"] + it["y1"]) / 2
        for cell in cells:
            if (cell["x0"] - 3 <= cx <= cell["x1"] + 3 and
                    cell["y0"] - 3 <= cy <= cell["y1"] + 3):
                buckets.setdefault((cell["row"], cell["col"]), []).append(it)
                break

    grid = {}
    for key, lst in buckets.items():
        lst.sort(key=lambda i: (i["y0"], i["x0"]))
        lines = []
        for it in lst:
            cy = (it["y0"] + it["y1"]) / 2
            if lines and abs(cy - lines[-1]["cy"]) < max(6, (it["y1"] - it["y0"]) * 0.6):
                lines[-1]["items"].append(it)
            else:
                lines.append({"cy": cy, "items": [it]})
        text = "\n".join(
            " ".join(i["text"] for i in sorted(l["items"], key=lambda i: i["x0"]))
            for l in lines)
        bg = fg = None
        if match_colors:
            fgs = []
            cell = next((c for c in cells if (c["row"], c["col"]) == key), None)
            cbg = None
            if cell is not None:
                crop_c = img[cell["y0"]:cell["y1"], cell["x0"]:cell["x1"]]
                if crop_c.size:
                    cbg = _hex(dominant_color(crop_c))
            for it in lst:
                b, f = sample_text_colors(img, it, 1.0, cbg)
                bg = bg or (None if is_near_white(cbg) else cbg) or b
                it["fg"] = f
                fgs.append(f)
            coloured = [f for f in fgs if f]
            if coloured:
                best = max(set(coloured), key=coloured.count)
                if coloured.count(best) > len(fgs) / 2.0:   # majority only
                    fg = best
        grid[key] = {"text": text,
                     "low": any(i["conf"] < LOW_CONF for i in lst),
                     "bg": bg, "fg": fg, "items": lst}

    if match_colors:       # coloured empty cells
        for cell in cells:
            key = (cell["row"], cell["col"])
            if key in grid or key in skip_keys:
                continue
            crop = img[cell["y0"]:cell["y1"], cell["x0"]:cell["x1"]]
            h = _hex(dominant_color(crop))
            if h and not is_near_white(h):
                grid[key] = {"text": "", "low": False, "bg": h, "fg": None}
    return grid


# ============================================================ OBJECT DETECTION
def detect_ink_objects(source_img, items_src, tables, boxes=(), arrows=()):
    """
    Find signatures, stamps, logos, drawings, QR codes ... = any ink that is
    NOT printed text and NOT a table rule.  Works even when the object sits
    in a cell that also contains text.
    """
    H, W = source_img.shape[:2]
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 14)

    # remove table rules
    for t in tables:
        for x in t["xs"]:
            ink[:, max(0, x - 4):x + 5][t["bbox"][1] - 4:t["bbox"][3] + 5] = 0
        for y in t["ys"]:
            ink[max(0, y - 4):y + 5, t["bbox"][0] - 4:t["bbox"][2] + 5] = 0

    for b in boxes:                       # outlines of boxes
        for (a0, a1, c0, c1) in ((b["y0"] - 5, b["y0"] + 6, b["x0"] - 5, b["x1"] + 6),
                                 (b["y1"] - 5, b["y1"] + 6, b["x0"] - 5, b["x1"] + 6),
                                 (b["y0"] - 5, b["y1"] + 6, b["x0"] - 5, b["x0"] + 6),
                                 (b["y0"] - 5, b["y1"] + 6, b["x1"] - 5, b["x1"] + 6)):
            ink[max(0, a0):a1, max(0, c0):c1] = 0
    for a in arrows:
        ink[max(0, a["y0"] - 4):a["y1"] + 5, max(0, a["x0"] - 4):a["x1"] + 5] = 0

    # remove OCR text boxes
    for it in items_src:
        x0, x1 = max(0, int(it["x0"]) - 4), min(W, int(it["x1"]) + 5)
        y0, y1 = max(0, int(it["y0"]) - 4), min(H, int(it["y1"]) + 5)
        ink[y0:y1, x0:x1] = 0

    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    objs = []
    for i in range(1, n):
        mask = (labels == i) & (ink > 0)
        count = int(mask.sum())
        if count < 150:
            continue
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < 22 or h < 22:
            continue
        if w > W * 0.9 and h > H * 0.9:
            continue
        # skip solid dots (e.g. punch-hole marks)
        if count / float(w * h) > 0.70 and 0.6 < w / float(h) < 1.6:
            continue
        m = 4
        x0, y0 = max(0, x0 - m), max(0, y0 - m)
        x1, y1 = min(W, x1 + m), min(H, y1 + m)
        objs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                     "crop": source_img[y0:y1, x0:x1].copy(),
                     "row": None, "col": None, "table": None,
                     "kind": "picture / signature / stamp"})

    objs.sort(key=lambda o: (o["y"], o["x"]))
    for k, o in enumerate(objs, 1):
        o["id"] = k
        cx, cy = o["x"] + o["w"] / 2, o["y"] + o["h"] / 2
        for ti, t in enumerate(tables):
            for c in t["cells"]:
                if c["x0"] - 2 <= cx <= c["x1"] + 2 and c["y0"] - 2 <= cy <= c["y1"] + 2:
                    o["table"], o["row"], o["col"] = ti, c["row"], c["col"]
                    break
            if o["table"] is not None:
                break
    return objs


# =============================================================== IMAGE HELPERS
def image_as_png_stream(image_path):
    raw = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                       cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot embed source image: {image_path}")
    ok, encoded = cv2.imencode(".png", raw)
    if not ok:
        raise ValueError(f"Cannot encode source image: {image_path}")
    return io.BytesIO(encoded.tobytes()), raw.shape[1], raw.shape[0]


def flatten_background(crop):
    """Turn the grey/yellow paper tone of a scan into clean white."""
    try:
        k = max(7, (min(crop.shape[:2]) // 3) | 1)
        bg = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        bg = cv2.GaussianBlur(bg, (k | 1, k | 1), 0).astype(np.float32)
        out = np.clip(crop.astype(np.float32) / np.maximum(bg, 1) * 255.0, 0, 255)
        return out.astype(np.uint8)
    except Exception:
        return crop


def crop_as_png_stream(crop, flatten=True):
    if flatten:
        crop = flatten_background(crop)
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Could not encode extracted image object.")
    return io.BytesIO(encoded.tobytes())


def _excel_col_letter(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _num_or_text(t):
    # keep phone numbers / codes with leading zeros as text
    if re.fullmatch(r"-?(0|[1-9]\d{0,14})(\.\d+)?", t or ""):
        return float(t) if "." in t else int(t)
    return t


def build_blocks(layout):
    """Everything in top-to-bottom order: free text lines, tables, free pictures."""
    blocks = []
    for r in layout["free_rows"]:
        blocks.append((r["y"], "text", r))
    for ti, t in enumerate(layout["tables"]):
        blocks.append((t["bbox"][1], "table", t))
    for o in layout["objects"]:
        if o["table"] is None:
            blocks.append((o["y"], "image", o))
    blocks.sort(key=lambda b: b[0])
    return blocks


# ============================================================ EXCEL EXPORT
def save_excel(layout, out_path, mark_low, original_image_path,
               keep_full_reference=False):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Border, Side, Alignment
    from openpyxl.drawing.image import Image as ExcelImage
    from openpyxl.drawing.spreadsheet_drawing import OneCellAnchor, AnchorMarker
    from openpyxl.drawing.xdr import XDRPositiveSize2D
    from openpyxl.utils.units import pixels_to_EMU

    tables = layout["tables"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted"
    ws.sheet_view.showGridLines = False
    thin = Side(style="thin", color="808080")
    red = PatternFill("solid", fgColor="FFFFC7CE")

    # Shared column grid built from ALL tables' vertical rules, so tables with
    # different column counts still line up like the original picture.
    free_cols = layout.get("free_cols") or []
    has_tab = any(r.get("tabular") for r in layout["free_rows"])
    xs_all = [x for t in tables for x in t["xs"]]
    if has_tab:                      # borderless columns of plain text
        xs_all += [a for a, _b in free_cols] + [free_cols[-1][1]]
    X = _cluster_positions(xs_all, 6) if xs_all else []

    def ucol(x):
        return min(range(len(X)), key=lambda i: abs(X[i] - x)) + 1

    zones = layout.get("zones") or []
    if zones:
        X = _cluster_positions([z["x0"] for z in zones] + [zones[-1]["x1"]], 2)
    n_ucols = max(1, len(X) - 1)
    if X:
        for i in range(len(X) - 1):
            ws.column_dimensions[_excel_col_letter(i + 1)].width = \
                max(4, min(45, (X[i + 1] - X[i]) / 7.0))
    else:
        ws.column_dimensions["A"].width = 100

    row = 1
    zrows = []                                  # zone mode: top of every Excel row, px
    if zones:
        frs = layout["free_rows"]
        tops = [min(i["y0"] for i in r["items"]) for r in frs]
        fsz = [max(7, min(40, 0.75 * max(i["size"] for i in r["items"]))) for r in frs]
        heights = []
        for k in range(len(frs)):
            nxt = tops[k + 1] if k + 1 < len(frs) else tops[k] + 24
            heights.append(max(fsz[k] * 1.45, (nxt - tops[k]) * 0.75))
        lead = max(1.0, (tops[0] if tops else 0) * 0.75)
        ws.row_dimensions[1].height = lead
        zrows = [(lead / 0.75, 1)]              # (top px, excel row)
        t_px = lead / 0.75
        for k, hgt in enumerate(heights):
            zrows.append((t_px, k + 2))
            ws.row_dimensions[k + 2].height = hgt
            t_px += hgt / 0.75
        total_px = t_px
        row = 2

    zk = 0
    for _y, kind, data in build_blocks(layout):
        if kind == "text" and data.get("tabular"):
            if data.get("gap"):
                row += 1                       # keep the blank line between groups
            cells = {}
            for it in data["items"]:
                ci = ucol(it["x0"] if it["col"] is None else free_cols[it["col"]][0])
                cells.setdefault(ci, []).append(it)
            for ci, its in cells.items():
                its.sort(key=lambda i: i["x0"])
                text = " ".join(i["text"] for i in its)
                if zones:
                    rz = next((r for z in zones for r in z["rows"]
                               if r["items"][0] is its[0]), None)
                    if rz is not None and rz.get("bullet") is not None:
                        text = "\u2022 " + text
                c = ws.cell(row, ci, _num_or_text(text) if not zones else text)
                if zones:
                    c.font = Font(name="Arial", size=round(0.75 * max(i["size"] for i in its) * 2) / 2,
                                  bold=any(i["bold"] for i in its),
                                  color="FF" + (its[0].get("fg") or "000000"))
                    c.alignment = Alignment(
                        vertical="top", wrap_text=False,
                        indent=int(max(0, its[0]["x0"] - zones[ci - 1]["x0"]) / 9))
                else:
                    c.font = Font(name="Arial", size=10,
                                  color="FF" + (its[0].get("fg") or "000000"))
                    c.alignment = Alignment(vertical="center", wrap_text=False)
                if mark_low and any(i["conf"] < LOW_CONF for i in its):
                    c.fill = red
            if not zones:
                ws.row_dimensions[row].height = 18
            row += 1

        elif kind == "text":
            c = ws.cell(row, ucol(data["items"][0]["x0"]) if X else 1, data["text"])
            c.font = Font(name="Arial", size=10,
                          color="FF" + (data.get("fg") or "000000"))
            c.alignment = Alignment(vertical="center", wrap_text=False)
            if mark_low and data.get("low"):
                c.fill = red
            ws.row_dimensions[row].height = 18
            row += 1

        elif kind == "table":
            t = data
            base = row
            cell_by = {(c["row"], c["col"]): c for c in t["cells"]}
            for r in range(1, t["n_rows"] + 1):
                px = max(c["y1"] - c["y0"] for c in t["cells"] if c["row"] == r)
                ws.row_dimensions[base + r - 1].height = max(15, min(220, px * 0.75))
                for cc in range(1, t["n_cols"] + 1):
                    cell = cell_by[(r, cc)]
                    d = t["grid"].get((r, cc),
                                      {"text": "", "low": False, "bg": None, "fg": None})
                    c0 = ucol(cell["x0"] - 2)
                    c1 = max(c0, ucol(cell["x1"] + 2) - 1)
                    xl = ws.cell(base + r - 1, c0)
                    xl.value = _num_or_text(d.get("text", ""))
                    xl.font = Font(name="Arial", size=10,
                                   color="FF" + (d.get("fg") or "000000"))
                    for k in range(c0, c1 + 1):
                        z = ws.cell(base + r - 1, k)
                        z.border = Border(top=thin, bottom=thin, left=thin, right=thin)
                        has_pic = any(o["table"] is not None and tables[o["table"]] is t
                                      and o["row"] == r and o["col"] == cc
                                      for o in layout["objects"])
                        z.alignment = Alignment(vertical="top" if has_pic else "center",
                                                wrap_text=True, horizontal="left")
                        if mark_low and d.get("low"):
                            z.fill = red
                        elif d.get("bg"):
                            z.fill = PatternFill("solid", fgColor="FF" + d["bg"])
                    if c1 > c0:
                        ws.merge_cells(start_row=base + r - 1, start_column=c0,
                                       end_row=base + r - 1, end_column=c1)

            for o in layout["objects"]:
                if o["table"] is None or tables[o["table"]] is not t:
                    continue
                cell = cell_by[(o["row"], o["col"])]
                c0 = ucol(cell["x0"] - 2)
                cw = cell["x1"] - cell["x0"]
                ch = cell["y1"] - cell["y0"]
                f = min(max(20, cw * 0.96) / max(1, o["w"]),
                        max(20, ch * 0.96) / max(1, o["h"]), 1.0)
                w_px = max(1, int(o["w"] * f))
                h_px = max(1, int(o["h"] * f))
                xl_img = ExcelImage(crop_as_png_stream(o["crop"]))
                xl_img.width, xl_img.height = w_px, h_px
                col_left = X[c0 - 1] if X else cell["x0"] - 2
                off_x = min(max(0, o["x"] - col_left), max(0, cw + 4 - w_px))
                off_y = max(0, o["y"] - (cell["y0"] - 2))
                txt = t["grid"].get((o["row"], o["col"]), {}).get("text", "")
                if txt:        # keep the picture clear of the cell's own text
                    off_y = max(off_y, (txt.count("\n") + 1) * 17 + 2)
                off_y = min(off_y, max(0, ch + 4 - h_px))
                marker = AnchorMarker(col=c0 - 1, row=base + o["row"] - 2,
                                      colOff=pixels_to_EMU(int(off_x)),
                                      rowOff=pixels_to_EMU(int(off_y)))
                xl_img.anchor = OneCellAnchor(
                    _from=marker,
                    ext=XDRPositiveSize2D(pixels_to_EMU(w_px), pixels_to_EMU(h_px)))
                ws.add_image(xl_img)

            row = base + t["n_rows"] + 1       # one blank row after a table

        elif zones and data.get("zone") is not None:     # picture inside a zone
            xl_img = ExcelImage(crop_as_png_stream(data["crop"], not data.get("keep_bg")))
            xl_img.width, xl_img.height = data["w"], data["h"]
            z = zones[data["zone"]]
            r_i = max(i for i, (tp, _r) in enumerate(zrows) if tp <= data["y"] + 0.5) \
                if any(tp <= data["y"] + 0.5 for tp, _r in zrows) else 0
            tp, r_xl = zrows[r_i]
            marker = AnchorMarker(col=ucol(z["x0"]) - 1, row=r_xl - 1,
                                  colOff=pixels_to_EMU(int(max(0, data["x"] - z["x0"]))),
                                  rowOff=pixels_to_EMU(int(max(0, data["y"] - tp))))
            xl_img.anchor = OneCellAnchor(
                _from=marker,
                ext=XDRPositiveSize2D(pixels_to_EMU(data["w"]), pixels_to_EMU(data["h"])))
            ws.add_image(xl_img)

        else:   # picture outside any table
            xl_img = ExcelImage(crop_as_png_stream(data["crop"]))
            f = min(500 / max(1, data["w"]), 300 / max(1, data["h"]), 1.0)
            xl_img.width = max(1, int(data["w"] * f))
            xl_img.height = max(1, int(data["h"] * f))
            xl_img.anchor = f"A{row}"
            ws.add_image(xl_img)
            row += int(xl_img.height / 20) + 2

    if zones:                                   # coloured areas (sidebar etc.)
        for zi, z in enumerate(zones):
            if z.get("fill"):
                fill = PatternFill("solid", fgColor="FF" + z["fill"])
                for r_ in range(1, len(zrows) + 3):
                    ws.cell(r_, ucol(z["x0"]) if X else zi + 1).fill = fill

    if keep_full_reference:
        visual = wb.create_sheet("Original Reference")
        visual.sheet_view.showGridLines = False
        stream, width, height = image_as_png_stream(original_image_path)
        original = ExcelImage(stream)
        if width > 1100:
            f = 1100 / width
            original.width, original.height = int(width * f), int(height * f)
        visual.add_image(original, "A1")

    _no_formulas(ws)
    wb.save(out_path)


# =============================================================== WORD EXPORT
_FONT_CACHE = {}


def _arial_metric_font(bold):
    """A font with Arial's letter widths (Liberation Sans) for measuring."""
    if bold in _FONT_CACHE:
        return _FONT_CACHE[bold]
    from PIL import ImageFont
    names = (["LiberationSans-Bold.ttf", "Arial_Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
             if bold else
             ["LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf", "DejaVuSans.ttf"])
    font = None
    for n in names:
        try:
            font = ImageFont.truetype(n, 100)
            break
        except Exception:
            continue
    _FONT_CACHE[bold] = font
    return font


def _track(run, e, text, size_pt, ppx, qn, OxmlElement):
    """Letter spacing so the line is as wide as in the picture (tracked headings)."""
    try:
        font = _arial_metric_font(bool(e.get("bold")))
        if font is None or len(text) < 3 or e.get("bullet") is not None:
            return
        natural = font.getlength(text) * (size_pt / ppx) / 100.0      # px
        target = e["x1"] - e["x0"]
        sp = (target - natural) / (len(text) - 1) * ppx               # pt per letter
        if sp < 0.9 or len(text) < 5 or text != text.upper():   # tracked CAPITAL headings only
            return
        sp = min(6.0, sp)
        el = OxmlElement("w:spacing")
        el.set(qn("w:val"), str(int(round(sp * 20))))
        run._r.get_or_add_rPr().append(el)
    except Exception:
        pass


def save_word(layout, out_path, mark_low, original_image_path, img_width,
              keep_full_reference=False):
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    def shade(cell, hex_fill):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)

    def style_run(run, fg, size=10):
        run.font.name = "Arial"
        run.font.size = Pt(size)
        if fg:
            run.font.color.rgb = RGBColor.from_string(fg)

    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.4)
        s.left_margin = s.right_margin = Inches(0.4)
    page_in = 7.5                      # usable width in inches
    inch_per_px = page_in / float(max(1, img_width))

    tables = layout["tables"]
    free_cols = layout.get("free_cols") or []
    wt = None                                   # open borderless text table
    zones = layout.get("zones") or []
    if zones:
        from docx.enum.text import WD_LINE_SPACING
        from docx.enum.table import WD_ROW_HEIGHT_RULE
        ppx = page_in * 72.0 / float(max(1, img_width))          # points per pixel
        zt = doc.add_table(rows=1, cols=len(zones))
        zt.autofit = False
        mar = OxmlElement("w:tblCellMar")                         # exact indents
        for side in ("top", "left", "bottom", "right"):
            el = OxmlElement("w:" + side)
            el.set(qn("w:w"), "0")
            el.set(qn("w:type"), "dxa")
            mar.append(el)
        zt._tbl.tblPr.append(mar)
        zt.rows[0].height = Pt(layout["size"][1] * ppx)
        zt.rows[0].height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for zi, z in enumerate(zones):
            w_in = (z["x1"] - z["x0"]) * inch_per_px
            zt.columns[zi].width = Inches(w_in)
            cell = zt.cell(0, zi)
            cell.width = Inches(w_in)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            if z.get("fill"):
                shade(cell, z["fill"])
            elems = [(r["y0"], "t", r) for r in z["rows"]] + \
                    [(o["y"], "i", o) for o in layout["objects"]
                     if o["table"] is None and o.get("zone") == zi]
            elems.sort(key=lambda e: e[0])
            cursor, first = 0.0, True
            for n_e, (y0, kind_e, e) in enumerate(elems):
                p = cell.paragraphs[0] if first else cell.add_paragraph()
                first = False
                pf = p.paragraph_format
                pf.space_after = Pt(0)
                pf.space_before = Pt(max(0.0, y0 - cursor) * ppx)
                if kind_e == "i":
                    pf.left_indent = Inches(max(0, e["x"] - z["x0"]) * inch_per_px)
                    p.add_run().add_picture(
                        crop_as_png_stream(e["crop"], not e.get("keep_bg")),
                        width=Inches(e["w"] * inch_per_px))
                    cursor = e["y"] + e["h"]
                    continue
                h_px = e["y1"] - e["y0"]
                cy = (e["y0"] + e["y1"]) / 2.0
                nxt = next((x for x in elems[n_e + 1:] if x[1] == "t"), None)
                dy = ((nxt[2]["y0"] + nxt[2]["y1"]) / 2.0 - cy) if nxt else 0
                size_pt = max(6.0, min(60.0, e["size"] * ppx * 0.97))
                line = dy if (nxt and 0 < dy < 2.2 * e["size"]) else 1.25 * e["size"]
                line = max(line, size_pt * 1.12 / ppx)
                pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
                pf.line_spacing = Pt(line * ppx)
                top = cy - line / 2.0
                pf.space_before = Pt(max(0.0, top - cursor) * ppx)
                left = max(0, e["x0"] - z["x0"]) * inch_per_px
                text = e["text"]
                if e.get("bullet") is not None:
                    hang = max(0.08, (e["x0"] - e["bullet"]) * inch_per_px)
                    pf.left_indent = Inches(left)
                    pf.first_line_indent = Inches(-hang)
                    text = "\u2022\t" + text
                else:
                    pf.left_indent = Inches(left)
                run = p.add_run(text)
                style_run(run, e.get("fg"), size_pt)
                run.bold = bool(e.get("bold"))
                _track(run, e, text, size_pt, ppx, qn, OxmlElement)
                cursor = max(cursor, top) + line
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    for _y, kind, data in ([] if zones else build_blocks(layout)):
        if not (kind == "text" and data.get("tabular")):
            wt = None
        if kind == "text" and data.get("tabular"):
            if wt is None:
                wt = doc.add_table(rows=0, cols=len(free_cols))
                wt.autofit = False
                widths = [(free_cols[i + 1][0] - free_cols[i][0]) * inch_per_px
                          for i in range(len(free_cols) - 1)]
                widths.append((free_cols[-1][1] - free_cols[-1][0] + 12) * inch_per_px)
                for i, w in enumerate(widths):
                    wt.columns[i].width = Inches(w)
            if data.get("gap"):
                gc = wt.add_row().cells
                for i, w in enumerate(widths):
                    gc[i].width = Inches(w)
            rc = wt.add_row().cells
            by_col = {}
            for it in data["items"]:
                by_col.setdefault(0 if it["col"] is None else it["col"], []).append(it)
            for i, w in enumerate(widths):
                rc[i].width = Inches(w)
                rc[i].paragraphs[0].paragraph_format.space_after = Pt(0)
            for ci, its in by_col.items():
                its.sort(key=lambda i: i["x0"])
                text = " ".join(i["text"] for i in its)
                p = rc[ci].paragraphs[0]
                if re.fullmatch(r"-?[\d.,]+", text):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                run = p.add_run(text)
                style_run(run, its[0].get("fg"), 10)
                if mark_low and any(i["conf"] < LOW_CONF for i in its):
                    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

        elif kind == "text":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(0)
            for idx, it in enumerate(data["items"]):
                if idx:
                    p.add_run("    ")
                run = p.add_run(it["text"])
                style_run(run, it.get("fg"), 11)
                if mark_low and it["conf"] < LOW_CONF:
                    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

        elif kind == "table":
            t = data
            table = doc.add_table(rows=t["n_rows"], cols=t["n_cols"])
            table.style = "Table Grid"
            table.autofit = False
            cell_by = {(c["row"], c["col"]): c for c in t["cells"]}
            for c in range(t["n_cols"]):
                sc_ = cell_by[(1, c + 1)]
                table.columns[c].width = Inches((sc_["x1"] - sc_["x0"]) * inch_per_px)
            for r in range(t["n_rows"]):
                for c in range(t["n_cols"]):
                    d = t["grid"].get((r + 1, c + 1),
                                      {"text": "", "low": False, "bg": None, "fg": None})
                    src_cell = cell_by[(r + 1, c + 1)]
                    cell = table.cell(r, c)
                    cell.width = Inches((src_cell["x1"] - src_cell["x0"]) * inch_per_px)
                    cell.text = ""
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_after = Pt(0)
                    if d.get("text"):
                        style_run(p.add_run(d["text"]), d.get("fg"), 9.5)
                    if mark_low and d.get("low"):
                        shade(cell, "FFC7CE")
                    elif d.get("bg"):
                        shade(cell, d["bg"])

            for o in layout["objects"]:
                if o["table"] is None or tables[o["table"]] is not t:
                    continue
                cell = table.cell(o["row"] - 1, o["col"] - 1)
                src_cell = cell_by[(o["row"], o["col"])]
                has_text = bool(t["grid"].get((o["row"], o["col"]), {}).get("text"))
                p = cell.add_paragraph() if has_text else cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                cw_in = (src_cell["x1"] - src_cell["x0"]) * inch_per_px
                width = min(o["w"] * inch_per_px * 0.9, max(0.5, cw_in * 0.5))
                p.add_run().add_picture(crop_as_png_stream(o["crop"]),
                                        width=Inches(max(0.4, width)))
            doc.add_paragraph().paragraph_format.space_after = Pt(0)

        else:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            width = max(0.4, min(5.0, data["w"] * inch_per_px))
            p.add_run().add_picture(crop_as_png_stream(data["crop"]),
                                    width=Inches(width))

    if keep_full_reference:
        doc.add_page_break()
        p = doc.add_paragraph()
        rr = p.add_run("Original Image Reference")
        rr.bold = True
        rr.font.size = Pt(14)
        stream, iw, ih = image_as_png_stream(original_image_path)
        sec = doc.sections[-1]
        mw = sec.page_width - sec.left_margin - sec.right_margin
        mh = sec.page_height - sec.top_margin - sec.bottom_margin - Inches(0.4)
        fit = min(mw / max(1, iw), mh / max(1, ih))
        p.add_run().add_picture(stream, width=Emu(int(iw * fit)),
                                height=Emu(int(ih * fit)))
    doc.save(out_path)


# ========================================================== POSITION GRID
def _align_in(x0, x1, tx0, tx1):
    """left / center / right of a text block inside a container."""
    l, r, w = tx0 - x0, x1 - tx1, max(1.0, x1 - x0)
    if abs(l - r) <= 0.05 * w:
        return "center"
    if r < 0.5 * l and r < 0.15 * w:
        return "right"
    return "left"


def _lines_of(items):
    """Group text pieces into visual lines (top to bottom, left to right)."""
    out = []
    for it in sorted(items, key=lambda i: ((i["y0"] + i["y1"]) / 2, i["x0"])):
        cy = (it["y0"] + it["y1"]) / 2
        if out and abs(cy - out[-1]["cy"]) < max(6, (it["y1"] - it["y0"]) * 0.6):
            out[-1]["items"].append(it)
        else:
            out.append({"cy": cy, "items": [it]})
    split = []                                  # far apart pieces are separate cells
    for ln in out:
        its = sorted(ln["items"], key=lambda i: i["x0"])
        cur = [its[0]]
        for it in its[1:]:
            if it["x0"] - cur[-1]["x1"] > 2.2 * max(it["size"], cur[-1]["size"]):
                split.append({"cy": ln["cy"], "items": cur})
                cur = [it]
            else:
                cur.append(it)
        split.append({"cy": ln["cy"], "items": cur})
    out = split
    for ln in out:
        ln["items"].sort(key=lambda i: i["x0"])
        its = ln["items"]
        ln.update(text=" ".join(i["text"] for i in its),
                  x0=min(i["x0"] for i in its), x1=max(i["x1"] for i in its),
                  y0=min(i["y0"] for i in its), y1=max(i["y1"] for i in its),
                  size=max(i["size"] for i in its), bold=any(i["bold"] for i in its),
                  fg=next((i.get("fg") for i in its if i.get("fg")), None),
                  low=any(i["conf"] < LOW_CONF for i in its))
    return out


def build_grid(source_img, tables, boxes, box_items, free_items, arrows, objects,
               match_colors, fills=(), page_hex=None):
    """
    One shared row/column grid for everything on the page, so tables, boxes,
    loose text, arrows and pictures keep their real positions (like a real
    spreadsheet laid over the picture).
    """
    H, W = source_img.shape[:2]
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

    def hint_for(it):
        cx, cy = (it["x0"] + it["x1"]) / 2, (it["y0"] + it["y1"]) / 2
        hit = [f for f in fills if f["x0"] <= cx <= f["x1"] and f["y0"] <= cy <= f["y1"]]
        return min(hit, key=lambda f: (f["x1"] - f["x0"]) * (f["y1"] - f["y0"]))["fill"] \
            if hit else None

    rects = []
    for ti, t in enumerate(tables):
        for c in t["cells"]:
            d = t["grid"].get((c["row"], c["col"]), {})
            rects.append({"kind": "cell",
                          "x0": t["xs"][c["col"] - 1], "x1": t["xs"][c["col"]],
                          "y0": t["ys"][c["row"] - 1], "y1": t["ys"][c["row"]],
                          "border": True, "fill": d.get("bg"),
                          "lines": _lines_of(d.get("items", []))})
    for k, b in enumerate(boxes):
        its = box_items[k]
        if match_colors:
            for it in its:
                it["fg"] = sample_text_colors(source_img, it, 1.0, hint_for(it))[1]
        lines = _lines_of(its)
        if len(lines) >= 2:                  # a heavier first line is the title
            def _stk(l):
                return statistics.mean([i.get("stroke", 0) for i in l["items"]])
            others = statistics.median([_stk(l) for l in lines[1:]])
            if _stk(lines[0]) >= 1.1 * others and lines[0]["size"] >= 0.95 * lines[1]["size"]:
                lines[0]["bold"] = True
        for ln in lines:                     # bullet dots
            ln["bullet"], ln["x0"] = find_bullet2(gray, b["x0"] + 6, ln)
        rects.append({"kind": "box", "x0": b["x0"], "x1": b["x1"], "y0": b["y0"],
                      "y1": b["y1"], "border": True, "fill": b["fill"], "lines": lines})
    free = []
    for it in free_items:
        if match_colors:
            it["fg"] = sample_text_colors(source_img, it, 1.0, hint_for(it))[1]
    for ln in _lines_of(free_items):
        free.append(ln)
    for f in fills:                          # plain colour areas, painted last
        rects.append({"kind": "fill", "x0": f["x0"], "x1": f["x1"], "y0": f["y0"],
                      "y1": f["y1"], "border": False, "fill": f["fill"], "lines": []})

    xv, yv = [], []
    for r in rects:
        xv += [r["x0"], r["x1"]]
        yv += [r["y0"], r["y1"]]
        if r["kind"] == "box":
            for ln in r["lines"]:
                yv += [ln["y0"], ln["y1"]]
    for ln in free:
        xv.append(ln["x0"])
        yv += [ln["y0"], ln["y1"]]
    for a in arrows:
        if a["dir"] in ("up", "down"):
            xv += [a["x"] - 9, a["x"] + 9]
            yv += [a["y0"], a["y1"]]
        else:
            xv += [a["x0"], a["x1"]]
            yv += [a["y"] - 9, a["y"] + 9]
    for o in objects:
        xv += [o["x"], o["x"] + o["w"]]
        yv += [o["y"], o["y"] + o["h"]]
    X = _cluster_positions(xv, 5)
    Y = _cluster_positions(yv, 4)
    return {"rects": rects, "free": free, "arrows": arrows, "objects": objects,
            "X": X, "Y": Y, "page_fill": page_hex if match_colors else None,
            "W": W, "H": H}


def _gx(G, x):
    X = G["X"]
    return min(range(len(X)), key=lambda i: abs(X[i] - x))


def _gy(G, y):
    Y = G["Y"]
    return min(range(len(Y)), key=lambda i: abs(Y[i] - y))


def _gx_floor(G, x):
    X = G["X"]
    return max([i for i in range(len(X)) if X[i] <= x + 0.5] or [0])


def _gy_floor(G, y):
    Y = G["Y"]
    return max([i for i in range(len(Y)) if Y[i] <= y + 0.5] or [0])


_ARROW = {"down": "\u2193", "up": "\u2191", "right": "\u2192", "left": "\u2190"}


def _no_formulas(ws):
    """Text such as '= INPUT' must stay text, never become an Excel formula."""
    for row in ws.iter_rows():
        for c in row:
            if c.data_type == "f" and isinstance(c.value, str):
                c.data_type = "s"


def save_excel_grid(layout, out_path, mark_low, original_image_path,
                    keep_full_reference=False):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Border, Side, Alignment
    from openpyxl.drawing.image import Image as ExcelImage
    from openpyxl.drawing.spreadsheet_drawing import OneCellAnchor, AnchorMarker
    from openpyxl.drawing.xdr import XDRPositiveSize2D
    from openpyxl.utils.units import pixels_to_EMU

    G = layout["grid"]
    X, Y = G["X"], G["Y"]
    nc, nr = len(X) - 1, len(Y) - 1
    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted"
    ws.sheet_view.showGridLines = False
    thin = Side(style="thin", color="000000")
    red = PatternFill("solid", fgColor="FFFFC7CE")

    for i in range(nc):
        ws.column_dimensions[_excel_col_letter(i + 1)].width = max(0.8, (X[i + 1] - X[i]) / 7.0)
    row_h = [max(2.0, (Y[j + 1] - Y[j]) * 0.75) for j in range(nr)]

    def pt(size_px):
        return max(7.0, min(40.0, round(0.75 * size_px * 2) / 2))

    if G.get("page_fill"):
        pf = PatternFill("solid", fgColor="FF" + G["page_fill"])
        for r in range(1, nr + 1):
            for c in range(1, nc + 1):
                ws.cell(r, c).fill = pf

    borders = {}
    merges, occupied = [], set()

    def can_merge(r0, c0, r1, c1):
        cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        return not (cells & occupied), cells

    def put(r, c0, c1, text, font_px, bold, fg, h_align, v_align="center",
            indent=0, wrap=False, r1=None, low=False):
        r1 = r if r1 is None else r1
        ok, cells = can_merge(r, c0, r1, c1)
        if not ok:
            c1, r1 = c0, r
            ok, cells = can_merge(r, c0, r, c0)
            if not ok:
                return
        if c1 > c0 or r1 > r:
            merges.append((r, c0, r1, c1))
        occupied.update(cells)
        cell = ws.cell(r, c0)
        cell.value = text
        cell.font = Font(name="Arial", size=pt(font_px), bold=bool(bold),
                         color="FF" + (fg or "000000"))
        cell.alignment = Alignment(horizontal=h_align, vertical=v_align, wrap_text=wrap,
                                   indent=indent)
        if mark_low and low:
            cell.fill = red
        need = pt(font_px) * 1.3
        for rr in range(r, r1 + 1):
            if r1 == r and row_h[rr - 1] < need:
                row_h[rr - 1] = need

    # fills, then borders
    for rc in sorted(G["rects"], key=lambda q: (q["kind"] == "fill",
                                                -(q["x1"] - q["x0"]) * (q["y1"] - q["y0"]))):
        c0, c1 = _gx(G, rc["x0"]), _gx(G, rc["x1"])
        r0, r1 = _gy(G, rc["y0"]), _gy(G, rc["y1"])
        if c1 <= c0 or r1 <= r0:
            continue
        rc["_g"] = (r0 + 1, c0 + 1, r1, c1)         # 1-based inclusive
        if rc.get("fill"):
            fl = PatternFill("solid", fgColor="FF" + rc["fill"])
            for r in range(r0 + 1, r1 + 1):
                for c in range(c0 + 1, c1 + 1):
                    ws.cell(r, c).fill = fl
        if rc.get("border"):
            for r in range(r0 + 1, r1 + 1):
                for c in range(c0 + 1, c1 + 1):
                    sd = borders.setdefault((r, c), {})
                    if r == r0 + 1:
                        sd["top"] = thin
                    if r == r1:
                        sd["bottom"] = thin
                    if c == c0 + 1:
                        sd["left"] = thin
                    if c == c1:
                        sd["right"] = thin

    # text of table cells: one merged cell, wrapped
    for rc in G["rects"]:
        if "_g" not in rc:
            continue
        r0, c0, r1, c1 = rc["_g"]
        lines = rc["lines"]
        same_row = any(abs(a["cy"] - b["cy"]) < 0.5 * max(a["size"], b["size"])
                       for i, a in enumerate(lines) for b in lines[i + 1:])
        if rc["kind"] == "cell" and same_row:
            rc["kind"] = "box"
        if rc["kind"] == "cell":
            if not lines:
                continue
            text = "\n".join(l["text"] for l in lines)
            first = lines[0]
            h_al = _align_in(rc["x0"], rc["x1"], min(l["x0"] for l in lines),
                             max(l["x1"] for l in lines))
            top = min(l["y0"] for l in lines) - rc["y0"]
            bot = rc["y1"] - max(l["y1"] for l in lines)
            h = max(1.0, rc["y1"] - rc["y0"])
            v_al = "center" if abs(top - bot) <= 0.25 * h else ("top" if top < bot else "bottom")
            val = _num_or_text(text) if len(lines) == 1 else text
            put(r0, c0, c1, val, first["size"], first["bold"], first["fg"], h_al, v_al,
                wrap=len(lines) > 1, r1=r1, low=any(l["low"] for l in lines))
        else:                                          # box: every line on its own row
            for ln in lines:
                r = _gy(G, ln["y0"]) + 1
                if not (r0 <= r <= r1):
                    r = min(max(r, r0), r1)
                text = ("\u2022 " if ln.get("bullet") is not None else "") + ln["text"]
                al = _align_in(rc["x0"], rc["x1"], ln["x0"], ln["x1"])
                if al == "left":
                    ind = int(max(0, ln["x0"] - rc["x0"] - 6) / 9)
                    put(r, c0, c0, text, ln["size"], ln["bold"], ln["fg"], "left",
                        indent=ind, low=ln["low"])
                else:
                    put(r, c0, c1, text, ln["size"], ln["bold"], ln["fg"], al, low=ln["low"])

    for ln in G["free"]:                              # loose text
        r = _gy(G, ln["y0"]) + 1
        c = _gx(G, ln["x0"]) + 1
        put(r, c, c, ln["text"], ln["size"], ln["bold"], ln["fg"], "left", low=ln["low"])

    for a in G["arrows"]:                             # arrows
        r = _gy_floor(G, a["y"]) + 1
        c = _gx_floor(G, a["x"]) + 1
        put(r, c, c, _ARROW[a["dir"]], 30, True, None, "center")

    from openpyxl.worksheet.properties import PageSetupProperties
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 1
    ws.page_setup.orientation = "landscape" if G["W"] > G["H"] else "portrait"
    for (r, c), sd in borders.items():
        ws.cell(r, c).border = Border(**sd)
    for (r0, c0, r1, c1) in merges:
        ws.merge_cells(start_row=r0, start_column=c0, end_row=r1, end_column=c1)
    for j, hgt in enumerate(row_h):
        ws.row_dimensions[j + 1].height = min(400, hgt)

    for o in G["objects"]:                            # pictures
        xl = ExcelImage(crop_as_png_stream(o["crop"], not G.get("page_fill")))
        xl.width, xl.height = o["w"], o["h"]
        ci, ri = _gx_floor(G, o["x"]), _gy_floor(G, o["y"])
        marker = AnchorMarker(col=ci, row=ri,
                              colOff=pixels_to_EMU(int(max(0, o["x"] - X[ci]))),
                              rowOff=pixels_to_EMU(int(max(0, o["y"] - Y[ri]))))
        xl.anchor = OneCellAnchor(
            _from=marker, ext=XDRPositiveSize2D(pixels_to_EMU(o["w"]), pixels_to_EMU(o["h"])))
        ws.add_image(xl)

    if keep_full_reference:
        visual = wb.create_sheet("Original Reference")
        visual.sheet_view.showGridLines = False
        stream, width, height = image_as_png_stream(original_image_path)
        original = ExcelImage(stream)
        if width > 1100:
            f = 1100 / width
            original.width, original.height = int(width * f), int(height * f)
        visual.add_image(original, "A1")
    _no_formulas(ws)
    wb.save(out_path)


# =============================================================== CONVERSION
def convert(image_path, want_excel, want_word, out_dir, log,
            match_colors=True, mark_low=False, extract_objects=True,
            keep_full_reference=False):
    image_path = Path(image_path)

    log("Reading image...")
    img, source_img, scale = read_image(image_path)

    global _PAGE_BG, _PAGE_TOL
    bgc = dominant_color(source_img)
    _PAGE_BG = [int(bgc[2]), int(bgc[1]), int(bgc[0])] if bgc is not None else None
    digital, page_hex = digital_page(source_img)
    _PAGE_TOL = 12 if digital else 60          # screenshots: even pale colours are real

    log("Running OCR...")
    items_src = to_source_coords(ocr_items(img), scale)

    style_items(items_src)

    log("Looking for tables and boxes...")
    tables = detect_tables(source_img)
    boxes = detect_boxes(source_img, tables)
    arrows = detect_arrows(source_img, items_src, tables, boxes) if not tables else []
    grid_mode = bool(boxes) or len(tables) >= 2
    fills, flat_objects = [], None
    if digital and grid_mode:
        fills = detect_fill_rects(source_img, page_hex)
        flat_objects = detect_flat_pictures(source_img, page_hex, fills, items_src,
                                            tables, boxes) if extract_objects else []
        items_src = [i for i in items_src if not i.get("_pic")]   # text inside drawings

    # Split OCR text: inside a table / inside a box / free text.
    inside = [[] for _ in tables]
    box_items = [[] for _ in boxes]
    free_items = []
    for it in items_src:
        cx, cy = (it["x0"] + it["x1"]) / 2, (it["y0"] + it["y1"]) / 2
        for ti, t in enumerate(tables):
            bx0, by0, bx1, by1 = t["bbox"]
            if bx0 - 3 <= cx <= bx1 + 3 and by0 - 3 <= cy <= by1 + 3:
                inside[ti].append(it)
                break
        else:
            hit = [k for k, b in enumerate(boxes)
                   if b["x0"] <= cx <= b["x1"] and b["y0"] <= cy <= b["y1"]]
            if hit:
                k = min(hit, key=lambda k: (boxes[k]["x1"] - boxes[k]["x0"]) *
                        (boxes[k]["y1"] - boxes[k]["y0"]))
                box_items[k].append(it)
            else:
                free_items.append(it)

    objects = []
    if flat_objects is not None:
        objects = flat_objects
    elif extract_objects:
        log("Extracting signatures / stamps / pictures...")
        objects = detect_ink_objects(source_img, items_src, tables, boxes, arrows)

    items_src.extend(recheck_empty_cells(source_img, tables, inside))

    for ti, t in enumerate(tables):
        skip = {(o["row"], o["col"]) for o in objects if o["table"] == ti}
        t["grid"] = grid_from_table(inside[ti], t["cells"], source_img,
                                    match_colors, skip)

    free_rows = []
    rows_ = build_rows(free_items)

    zones = [] if (tables or boxes) else detect_zones(source_img)
    if zones:
        free_cols = [[z["x0"], z["x1"]] for z in zones]
        gray_src = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        def zone_of(cx):
            for k, z in enumerate(zones):
                if z["x0"] <= cx < z["x1"]:
                    return k
            return 0 if cx < zones[0]["x0"] else len(zones) - 1

        for it in free_items:
            it["col"] = zone_of((it["x0"] + it["x1"]) / 2)
            if match_colors:
                it["fg"] = sample_text_colors(source_img, it, 1.0)[1]
        for k, z in enumerate(zones):
            z["rows"] = []
            for r in build_rows([i for i in free_items if i["col"] == k]):
                its = r["cells"]
                row = {"y0": min(i["y0"] for i in its), "y1": max(i["y1"] for i in its),
                       "x0": min(i["x0"] for i in its), "x1": max(i["x1"] for i in its),
                       "items": its, "text": " ".join(i["text"] for i in its),
                       "fg": next((i.get("fg") for i in its if i.get("fg")), None),
                       "bold": any(i["bold"] for i in its),
                       "size": max(i["size"] for i in its)}
                row["bullet"] = find_bullet(gray_src, z["x0"], row)
                z["rows"].append(row)
        for r in rows_:
            r["tabular"], r["gap"] = True, False
    else:
        free_cols = build_columns(rows_, source_img.shape[1])
    wide = source_img.shape[1] * 0.35
    for r in rows_:                       # which column does each text belong to
        for it in r["cells"]:
            if zones:
                continue
            it["col"] = (None if it["x1"] - it["x0"] >= wide
                         else col_index(it, free_cols) - 1)
    for r in rows_:
        if zones:
            break
        r["tabular"] = len({it["col"] for it in r["cells"]
                            if it["col"] is not None}) >= 2
    for k, r in enumerate(rows_):         # a lone cell between table rows stays in the table
        if zones:
            break
        if r["tabular"] or not all(it["col"] is not None for it in r["cells"]):
            continue
        if (k > 0 and rows_[k - 1]["tabular"]) or \
           (k + 1 < len(rows_) and rows_[k + 1]["tabular"]):
            r["tabular"] = True
    diffs = [b["cy"] - a["cy"] for a, b in zip(rows_, rows_[1:])]
    typical = statistics.median(diffs) if diffs else 0
    prev = None
    for r in rows_:                       # blank line only between two table-like rows
        r["gap"] = bool(prev is not None and prev["tabular"] and r["tabular"]
                        and typical and r["cy"] - prev["cy"] > 1.5 * typical)
        prev = r

    for r in rows_:
        its = r["cells"]
        if match_colors and not zones:
            for it in its:
                it["fg"] = sample_text_colors(source_img, it, 1.0)[1]
        free_rows.append({
            "y": r["cy"], "items": its,
            "tabular": r["tabular"], "gap": r["gap"],
            "text": "    ".join(i["text"] for i in its),
            "fg": next((i.get("fg") for i in its if i.get("fg")), None),
            "low": any(i["conf"] < LOW_CONF for i in its)})

    for o in objects:                    # which zone a picture belongs to
        o["zone"] = None
        if zones and o["table"] is None:
            cx = o["x"] + o["w"] / 2
            o["zone"] = next((k for k, z in enumerate(zones)
                              if z["x0"] <= cx < z["x1"]), 0)
            o["keep_bg"] = bool(zones[o["zone"]].get("fill"))
    layout = {"tables": tables, "free_rows": free_rows, "objects": objects,
              "free_cols": free_cols, "zones": zones,
              "size": (source_img.shape[1], source_img.shape[0])}
    if grid_mode:
        layout["grid"] = build_grid(source_img, tables, boxes, box_items, free_items,
                                    arrows, objects, match_colors, fills,
                                    page_hex if digital else None)

    saved = []
    if want_excel:
        log("Creating Excel...")
        p = Path(out_dir) / (image_path.stem + ".xlsx")
        (save_excel_grid if layout.get("grid") else save_excel)(
            layout, p, mark_low, image_path, keep_full_reference)
        saved.append(p)
    if want_word:
        log("Creating Word...")
        p = Path(out_dir) / (image_path.stem + ".docx")
        save_word(layout, p, mark_low, image_path, source_img.shape[1],
                  keep_full_reference)
        saved.append(p)

    if tables:
        desc = "; ".join(f"table {i + 1}: {t['n_rows']}x{t['n_cols']}"
                         for i, t in enumerate(tables))
        summary = f"{len(tables)} table(s) detected ({desc})."
    else:
        summary = "No table detected: plain text, no cells or borders."
    summary += f"\n{len(items_src)} text pieces, {len(objects)} pictures/signatures."
    return saved, summary


