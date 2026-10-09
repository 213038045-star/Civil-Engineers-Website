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
def _restore_spaces(gray, box, chars, wboxes):
    """
    OCR often glues words together ("AIRCONDITIONINGSYSTEM").  Use the real
    white gaps in the image plus the per-character positions to put spaces back.
    """
    try:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = max(0, int(min(xs))), min(gray.shape[1], int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(gray.shape[0], int(max(ys)) + 1)
        if x1 - x0 < 8 or y1 - y0 < 6:
            return None
        crop = gray[y0:y1, x0:x1]
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        col_ink = bw.any(axis=0)
        thr = max(4, int(0.20 * (y1 - y0)))

        runs, start = [], None
        for i, v in enumerate(col_ink):
            if not v and start is None:
                start = i
            elif v and start is not None:
                if i - start >= thr and start > 0:
                    runs.append((start + x0 + i + x0) / 2.0)
                start = None

        letters = []   # (char, center_x, had_space_before)
        pending = False
        for ch, wb in zip(chars, wboxes):
            if ch.isspace():
                pending = True
                continue
            cx = (min(p[0] for p in wb) + max(p[0] for p in wb)) / 2.0
            letters.append((ch, cx, pending))
            pending = False
        if not letters:
            return None

        out = [letters[0][0]]
        for k in range(1, len(letters)):
            ch, cx, had = letters[k]
            prev_cx = letters[k - 1][1]
            gap = any(prev_cx < rc < cx for rc in runs)
            out.append((" " if (had or gap) else "") + ch)
        return "".join(out)
    except Exception:
        return None


def ocr_items(img):
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()

    try:
        result, _ = _ENGINE(img, return_word_box=True)
    except TypeError:
        result, _ = _ENGINE(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    items = []
    for r in (result or []):
        box, text, conf = r[0], r[1], r[2]
        text = str(text).strip()
        if not text:
            continue
        if len(r) >= 5 and r[3] and r[4] and len(r[3]) == len(r[4]):
            fixed = _restore_spaces(gray, box, r[4], r[3])
            if fixed:
                text = fixed
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append({"text": text, "conf": float(conf),
                      "x0": min(xs), "x1": max(xs),
                      "y0": min(ys), "y1": max(ys)})
    return items


def to_source_coords(items, scale):
    out = []
    for it in items:
        q = dict(it)
        for k in ("x0", "x1", "y0", "y1"):
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


def is_near_white(h):
    if not h:
        return True
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    if all(v >= 245 for v in rgb):
        return True
    if _PAGE_BG is not None:       # grey/yellow scanned paper is not a "cell colour"
        return sum(abs(a - b) for a, b in zip(rgb, _PAGE_BG)) <= 60
    return False


def is_near_black(h):
    if not h:
        return True
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = max(r, g, b) - min(r, g, b)
    # dark grey text of a scan counts as ordinary black text
    return lum < 55 or chroma < 45


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


def sample_text_colors(img, it, scale=1.0):
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
    if bg is None:
        return None, None

    inside = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
    bg_i = np.asarray(bg, dtype=np.int16)
    dist = np.abs(inside - bg_i).sum(axis=1)
    fg_pixels = inside[dist > 110]
    fg = None
    if len(fg_pixels) >= 3:
        d = np.abs(fg_pixels - bg_i).sum(axis=1)
        fg = np.median(fg_pixels[d >= np.percentile(d, 55)], axis=0)

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
        xs = np.where(vp >= max(20, 0.6 * vp.max(), 0.35 * (ry1 - ry0)))[0]
        ys = [v + ry0 for v in _cluster_positions(ys)]
        xs = [v + rx0 for v in _cluster_positions(xs)]
        if len(xs) < 2 or len(ys) < 2:
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
            for it in lst:
                b, f = sample_text_colors(img, it, 1.0)
                bg = bg or b
                fgs.append(f)
            coloured = [f for f in fgs if f]
            if coloured:
                best = max(set(coloured), key=coloured.count)
                if coloured.count(best) > len(fgs) / 2.0:   # majority only
                    fg = best
        grid[key] = {"text": text,
                     "low": any(i["conf"] < LOW_CONF for i in lst),
                     "bg": bg, "fg": fg}

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
def detect_ink_objects(source_img, items_src, tables):
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


def crop_as_png_stream(crop):
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
    if tables:
        X = _cluster_positions([x for t in tables for x in t["xs"]], 6)
    else:
        X = []

    def ucol(x):
        return min(range(len(X)), key=lambda i: abs(X[i] - x)) + 1

    n_ucols = max(1, len(X) - 1)
    if X:
        for i in range(len(X) - 1):
            ws.column_dimensions[_excel_col_letter(i + 1)].width = \
                max(4, min(45, (X[i + 1] - X[i]) / 7.0))
    else:
        ws.column_dimensions["A"].width = 100

    row = 1
    for _y, kind, data in build_blocks(layout):
        if kind == "text":
            c = ws.cell(row, 1, data["text"])
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

        else:   # picture outside any table
            xl_img = ExcelImage(crop_as_png_stream(data["crop"]))
            f = min(500 / max(1, data["w"]), 300 / max(1, data["h"]), 1.0)
            xl_img.width = max(1, int(data["w"] * f))
            xl_img.height = max(1, int(data["h"] * f))
            xl_img.anchor = f"A{row}"
            ws.add_image(xl_img)
            row += int(xl_img.height / 20) + 2

    if keep_full_reference:
        visual = wb.create_sheet("Original Reference")
        visual.sheet_view.showGridLines = False
        stream, width, height = image_as_png_stream(original_image_path)
        original = ExcelImage(stream)
        if width > 1100:
            f = 1100 / width
            original.width, original.height = int(width * f), int(height * f)
        visual.add_image(original, "A1")

    wb.save(out_path)


# =============================================================== WORD EXPORT
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
    for _y, kind, data in build_blocks(layout):
        if kind == "text":
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


# =============================================================== CONVERSION
def convert(image_path, want_excel, want_word, out_dir, log,
            match_colors=True, mark_low=False, extract_objects=True,
            keep_full_reference=False):
    image_path = Path(image_path)

    log("Reading image...")
    img, source_img, scale = read_image(image_path)

    global _PAGE_BG
    bgc = dominant_color(source_img)
    _PAGE_BG = [int(bgc[2]), int(bgc[1]), int(bgc[0])] if bgc is not None else None

    log("Running OCR...")
    items_src = to_source_coords(ocr_items(img), scale)

    log("Looking for tables...")
    tables = detect_tables(source_img)

    # Split OCR text: inside a table vs. free text.
    inside = [[] for _ in tables]
    free_items = []
    for it in items_src:
        cx, cy = (it["x0"] + it["x1"]) / 2, (it["y0"] + it["y1"]) / 2
        for ti, t in enumerate(tables):
            bx0, by0, bx1, by1 = t["bbox"]
            if bx0 - 3 <= cx <= bx1 + 3 and by0 - 3 <= cy <= by1 + 3:
                inside[ti].append(it)
                break
        else:
            free_items.append(it)

    objects = []
    if extract_objects:
        log("Extracting signatures / stamps / pictures...")
        objects = detect_ink_objects(source_img, items_src, tables)

    for ti, t in enumerate(tables):
        skip = {(o["row"], o["col"]) for o in objects if o["table"] == ti}
        t["grid"] = grid_from_table(inside[ti], t["cells"], source_img,
                                    match_colors, skip)

    free_rows = []
    for r in build_rows(free_items):
        its = r["cells"]
        if match_colors:
            for it in its:
                it["fg"] = sample_text_colors(source_img, it, 1.0)[1]
        free_rows.append({
            "y": r["cy"], "items": its,
            "text": "    ".join(i["text"] for i in its),
            "fg": next((i.get("fg") for i in its if i.get("fg")), None),
            "low": any(i["conf"] < LOW_CONF for i in its)})

    layout = {"tables": tables, "free_rows": free_rows, "objects": objects}

    saved = []
    if want_excel:
        log("Creating Excel...")
        p = Path(out_dir) / (image_path.stem + ".xlsx")
        save_excel(layout, p, mark_low, image_path, keep_full_reference)
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


