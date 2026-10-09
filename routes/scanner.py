"""
Document scanner blueprint (CamScanner-style).

Drop-in replacement: keeps the same blueprint name (`scanner`), the same
url_prefix (/scanner) and the same template (scanner.html).

Pipeline
  1. /upload        save photo, auto-detect the page corners
  2. (browser)      user can drag the 4 corners to fix the crop
  3. /process       perspective-correct + enhance (Magic color, B&W, ...)
  4. /pdf           multi-page PDF   |  /result/<id>_out.jpg   single JPG
  Extras: /detect-frame (live camera outline), /ocr (extract text)

Needs: flask, opencv-python, numpy, pillow, reportlab
Optional: rapidocr-onnxruntime  (only for "Extract text")
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
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Feature-Policy"] = "camera 'self'"
    return response


@scanner.route("/")
def scanner_page():
    return render_template("scanner.html")


# ---------------------------------------------------------
# PAGE DETECTION
# ---------------------------------------------------------

def order_points(points):
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left.

    Sorting by angle around the centre also works for pages that are rotated
    or that the user dragged into an odd order.
    """
    pts = np.array(points, dtype="float32").reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]            # clockwise on screen
    start = int(np.argmin(pts.sum(axis=1)))  # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)


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


def _score_quad(quad, hull_area, width, height):
    """0 = reject. Higher = looks more like a photographed page."""
    ordered = order_points(quad)
    area = cv2.contourArea(ordered)
    fraction = area / float(width * height)

    if fraction < 0.10 or fraction > 0.995:
        return 0.0

    # A mask that just traces the picture frame is not a page.
    margin = 3
    on_edge = sum(
        1 for x, y in ordered
        if x <= margin or y <= margin or x >= width - 1 - margin or y >= height - 1 - margin
    )
    if on_edge >= 3:
        return 0.0

    sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 0.12 * min(width, height):
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

    fit = min(hull_area, area) / max(hull_area, area, 1.0)
    return fraction * fit


def _masks(small):
    """Several independent ways to separate the page from the background."""
    height, width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    close_kernel = np.ones((9, 9), np.uint8)
    edge_kernel = np.ones((5, 5), np.uint8)

    masks = []

    # 1) Canny edges (auto + fixed thresholds), closed into outlines
    median = float(np.median(blur))
    for low, high in ((max(10, 0.66 * median), min(255, 1.33 * median)), (30, 90)):
        edges = cv2.Canny(blur, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
        masks.append(edges)

    # 2) Bright page on darker surface (and the reverse)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (otsu, 255 - otsu):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, edge_kernel)
        masks.append(mask)

    # 3) Anything that differs in colour from the picture border
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.03 * min(height, width)))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)
    ])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance = np.clip(distance * 255.0 / max(float(distance.max()), 1.0), 0, 255).astype(np.uint8)
    distance = cv2.GaussianBlur(distance, (7, 7), 0)
    _, diff = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, edge_kernel)
    masks.append(diff)

    return masks


def detect_corners(image):
    """Return 4 normalised (0..1) corners [TL, TR, BR, BL], or None."""
    height, width = image.shape[:2]
    scale = 640.0 / max(height, width)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    sh, sw = small.shape[:2]

    best_quad, best_score = None, 0.0

    for mask in _masks(small):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            if cv2.contourArea(contour) < 0.08 * sw * sh:
                continue
            quad, hull_area = _quad_from_contour(contour)
            score = _score_quad(quad, hull_area, sw, sh)
            if score > best_score:
                best_quad, best_score = quad, score

    if best_quad is None:
        return None

    ordered = order_points(best_quad)
    ordered[:, 0] = np.clip(ordered[:, 0] / float(sw), 0, 1)
    ordered[:, 1] = np.clip(ordered[:, 1] / float(sh), 0, 1)
    return [[round(float(x), 5), round(float(y), 5)] for x, y in ordered]


def _parse_corners(raw):
    """Validate corners sent by the browser. None -> use the whole photo."""
    if raw is None:
        return None
    try:
        pts = np.array(raw, dtype="float32")
    except (TypeError, ValueError):
        raise ValueError("Corners are not valid.")
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Corners are not valid.")
    return np.clip(pts, 0.0, 1.0)


# ---------------------------------------------------------
# PERSPECTIVE CORRECTION
# ---------------------------------------------------------

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
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    longest = max(warped.shape[:2])
    if longest > OUT_MAX_SIDE:
        ratio = OUT_MAX_SIDE / float(longest)
        warped = cv2.resize(warped, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    return warped


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


def apply_filter(image, filter_name, brightness=0.0, contrast=1.0):
    """Apply scanner-style enhancement presets implemented on the server."""
    original = image.copy()

    if filter_name == "original":
        out = original

    elif filter_name == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = _flatten(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        block = _odd(max(31, max(gray.shape) // 30))
        block = min(block, min(gray.shape) if min(gray.shape) % 2 else min(gray.shape) - 1)
        block = max(3, block)
        offset = _clamp(12 + brightness / 8.0, 2, 28)
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, offset
        )
        out = cv2.medianBlur(out, 3)
        brightness = 0

    elif filter_name == "gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out = _levels(_flatten(gray), 30, 238)
        out = _sharpen(out, 0.45)

    elif filter_name == "lighten":
        flat = _flatten(image)
        out = cv2.addWeighted(image, 0.30, flat, 0.70, 0)
        out = _gamma(out, 0.82)
        out = _sharpen(out, 0.35)

    elif filter_name == "document":
        # Flatten uneven paper lighting, then enhance local text contrast.
        flat = _flatten(image)
        lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        out = _levels(out, 12, 246)
        out = _sharpen(out, 0.45)

    elif filter_name in ("magic", "magicpro"):
        flat = _flatten(image)
        lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 2.0 if filter_name == "magicpro" else 1.35
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.08 if filter_name == "magicpro" else 1.04), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if filter_name == "magicpro":
            out = cv2.bilateralFilter(out, 5, 24, 24)
            out = _sharpen(out, 0.55)
        else:
            out = _sharpen(out, 0.35)

    else:
        out = original

    if brightness or contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness + 128 * (1 - contrast))
    return out


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

    corners = None
    try:
        corners = detect_corners(image)
    except Exception:
        current_app.logger.exception("Page detection failed")

    detected = corners is not None

    if not detected:
        # start with a slightly inset frame so the handles are easy to grab
        corners = [[0.03, 0.03], [0.97, 0.03], [0.97, 0.97], [0.03, 0.97]]

    height, width = image.shape[:2]

    return jsonify({
        "success": True,
        "id": pid,
        "width": width,
        "height": height,
        "src": url_for("scanner.result", filename=pid + "_src.jpg"),
        "corners": corners,
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

    filter_name = data.get("filter", "magic")
    if filter_name not in FILTERS:
        filter_name = "magic"

    try:
        brightness = _clamp(float(data.get("brightness", 0)), -100.0, 100.0)
        contrast = _clamp(float(data.get("contrast", 1)), 0.5, 2.0)
        rotate = int(data.get("rotate", 0)) % 360
        corners = _parse_corners(data.get("corners"))
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
            points = corners * np.array([width, height], dtype="float32")
            polygon_area = cv2.contourArea(order_points(points))
            if polygon_area < 0.01 * width * height:
                return _fail("The crop area is too small. Drag the corners further apart.")
            scanned = four_point_transform(image, points)

        if rotate == 90:
            scanned = cv2.rotate(scanned, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            scanned = cv2.rotate(scanned, cv2.ROTATE_180)
        elif rotate == 270:
            scanned = cv2.rotate(scanned, cv2.ROTATE_90_COUNTERCLOCKWISE)

        scanned = apply_filter(scanned, filter_name, brightness, contrast)

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
    # Fill the selected paper size by default, like CamScanner. This can
    # trim a little from the image edges when aspect ratios differ.
    # fill=0 keeps the complete crop visible and may leave white margins.
    fill_page = request.args.get("fill", "1") == "1"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setTitle("Scanned Document")
    pdf.setCreator("RAUF Document Scanner")

    for path in paths:

        with Image.open(path) as picture:
            img_w, img_h = picture.size

        if size in PDF_SIZES:
            short, long_ = sorted(PDF_SIZES[size])
            page_w, page_h = (long_, short) if img_w > img_h else (short, long_)
        else:
            # "Original size": 150 dpi, so a photo of an A4 sheet becomes ~A4
            page_w, page_h = img_w * 72.0 / 150.0, img_h * 72.0 / 150.0

        if fill_page:
            # Scale until the image covers the whole paper; center it so the
            # excess is clipped evenly from opposite edges.
            ratio = max(page_w / img_w, page_h / img_h)
        else:
            # Keep the entire cropped image inside the paper without clipping.
            ratio = min(page_w / img_w, page_h / img_h)
        draw_w, draw_h = img_w * ratio, img_h * ratio

        pdf.setPageSize((page_w, page_h))
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


# ---------------------------------------------------------
# EXTRACT TEXT (optional: pip install rapidocr-onnxruntime)
# ---------------------------------------------------------

_ocr_engine = None
_ocr_lock = threading.Lock()


@scanner.route("/ocr", methods=["POST"])
def ocr():

    data = request.get_json(silent=True) or {}
    pid = data.get("id")

    if not _valid_id(pid):
        return _fail("Unknown page.")

    path = _out_path(pid) if os.path.exists(_out_path(pid)) else _src_path(pid)

    if not os.path.exists(path):
        return _fail("This photo expired. Capture or upload it again.", 404)

    global _ocr_engine

    with _ocr_lock:
        try:
            if _ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _ocr_engine = RapidOCR()
        except ImportError:
            return _fail(
                "Text extraction needs an extra package. Run: pip install rapidocr-onnxruntime",
                501
            )

        try:
            found, _ = _ocr_engine(path)
        except Exception:
            current_app.logger.exception("OCR failed")
            return _fail("Text extraction failed on this page.", 500)

    lines = [item[1] for item in (found or [])]

    return jsonify({"success": True, "text": "\n".join(lines)})