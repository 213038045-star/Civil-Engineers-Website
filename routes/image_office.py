"""
Image -> Word / Excel tool.
Page : /image_to_word_excel
API  : POST /image_to_word_excel/convert
"""
import io
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

try:
    from image_to_office import convert
except ImportError:          # keep the whole site running if the converter is missing
    convert = None

image_office_bp = Blueprint("image_office", __name__)

MAX_BYTES = 10 * 1024 * 1024                      # 10 MB per image
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# OCR is heavy: process one image at a time so the rest of the site stays fast.
_busy = threading.Semaphore(1)


def _flag(name, default=True):
    value = request.form.get(name)
    return default if value is None else value.lower() in ("1", "true", "on", "yes")


@image_office_bp.route("/image_to_word_excel")
def image_to_word_excel():
    return render_template("others/image_to_word_excel.html")


@image_office_bp.route("/image_to_word_excel/convert", methods=["POST"])
def image_to_word_excel_convert():
    if convert is None:
        return jsonify(error="Converter is not installed on the server."), 503

    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="Please choose an image."), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error="Please upload an image (png, jpg, bmp, tif, webp)."), 400

    want_excel, want_word = _flag("excel"), _flag("word")
    if not (want_excel or want_word):
        return jsonify(error="Choose Excel, Word, or both."), 400

    data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return jsonify(error="File too large (max 10 MB)."), 413

    if not _busy.acquire(blocking=False):
        return jsonify(error="The converter is busy. Please try again in a moment."), 503

    work = Path(tempfile.mkdtemp())
    try:
        img = work / ("upload" + ext)
        img.write_bytes(data)
        saved, _summary = convert(
            img, want_excel, want_word, work, lambda m: None,
            match_colors=_flag("colors"), extract_objects=_flag("objects"))

        stem = secure_filename(Path(f.filename).stem) or "converted"
        if len(saved) == 1:
            out = io.BytesIO(saved[0].read_bytes())
            name, mime = f"{stem}{saved[0].suffix}", None
        else:
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for p in saved:
                    z.write(p, f"{stem}{p.suffix}")
            name, mime = f"{stem}.zip", "application/zip"
        out.seek(0)
    except Exception as e:
        return jsonify(error=f"Conversion failed: {e}"), 500
    finally:
        shutil.rmtree(work, ignore_errors=True)   # uploaded image is always deleted
        _busy.release()

    return send_file(out, as_attachment=True, download_name=name, mimetype=mime)
