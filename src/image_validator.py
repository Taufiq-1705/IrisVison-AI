"""Domain image validation, magic-byte inspection, EXIF stripping, and channel normalization.

Implements the `FundusImage` class per IrisVision AI SRS & QA specifications:
- Attributes: `format`, `lebarPx`, `tinggiPx`, `channels`, `sha256`, `raw_bytes`
- Methods: `validateFormat()`, `computeHash()`, `stripExif()`, `to_bgr_array()`
"""

import hashlib
import io
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from src.config import ALLOWED_API_FORMATS, MIN_IMAGE_DIMENSION

# Enforce strict decoding so truncated image streams raise an error
ImageFile.LOAD_TRUNCATED_IMAGES = False

FORMAT_TO_EXTENSIONS: dict[str, set[str]] = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
}


class ImageValidationError(ValueError):
    """Raised when an uploaded ocular image fails format, magic-byte, integrity, or resolution checks."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def detect_magic_format(data: bytes) -> str:
    """Detect true image container format strictly from magic-byte signatures."""
    if not data or len(data) < 4:
        return "UNKNOWN"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(b"BM"):
        return "BMP"
    return "UNKNOWN"


class FundusImage:
    """Represents and validates an ocular fundus image in memory."""

    def __init__(
        self,
        raw_bytes: bytes,
        filename: Optional[str] = None,
    ) -> None:
        self.raw_bytes: bytes = raw_bytes if raw_bytes is not None else b""
        self.filename: Optional[str] = filename
        self.format: str = detect_magic_format(self.raw_bytes)
        self.lebarPx: int = 0
        self.tinggiPx: int = 0
        self.channels: int = 0
        self.sha256: str = self.computeHash() if self.raw_bytes else ""
        self._decoded_raw: Optional[np.ndarray] = None

        if self.format != "UNKNOWN":
            self._try_inspect_dimensions()

    def _try_inspect_dimensions(self) -> None:
        """Populate lebarPx, tinggiPx, and channels if bytes are decodable."""
        try:
            nparr = np.frombuffer(self.raw_bytes, np.uint8)
            arr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if arr is not None and arr.size > 0:
                h, w = arr.shape[:2]
                ch = 1 if arr.ndim == 2 else int(arr.shape[2])
                self.tinggiPx = int(h)
                self.lebarPx = int(w)
                self.channels = ch
                self._decoded_raw = arr
        except Exception:
            pass

    def computeHash(self) -> str:
        """Compute SHA-256 hex digest of the current image bytes."""
        if not self.raw_bytes:
            self.sha256 = ""
            return ""
        self.sha256 = hashlib.sha256(self.raw_bytes).hexdigest()
        return self.sha256

    def validateFormat(
        self,
        allowed_formats: Optional[Set[str]] = None,
        min_width: int = MIN_IMAGE_DIMENSION,
        min_height: int = MIN_IMAGE_DIMENSION,
        check_extension: bool = True,
    ) -> bool:
        """Validate non-empty bytes, extension, magic bytes, stream integrity, and minimum resolution.

        Default `allowed_formats` is `{"JPEG", "PNG"}` per production API SRS.
        Pass `allowed_formats={"JPEG", "PNG", "WEBP", "BMP"}` in robustness benchmarks.
        """
        if allowed_formats is None:
            allowed_formats = set(ALLOWED_API_FORMATS)
        else:
            allowed_formats = {f.upper() for f in allowed_formats}

        # 1. Check empty payload
        if not self.raw_bytes:
            raise ImageValidationError("Uploaded file is empty (0 bytes).", status_code=400)

        # 2. Check filename extension if provided
        if check_extension and self.filename:
            ext = Path(self.filename).suffix.lower()
            allowed_exts: set[str] = set()
            for fmt in allowed_formats:
                allowed_exts.update(FORMAT_TO_EXTENSIONS.get(fmt, set()))
            if ext and ext not in allowed_exts:
                raise ImageValidationError(
                    f"Unsupported file extension '{ext}'. Allowed formats: {sorted(allowed_formats)} "
                    f"({sorted(allowed_exts)}).",
                    status_code=400,
                )

        # 3. Check magic-byte signature (prevents fake extension spoofing e.g. PDF/TXT renamed to .jpg)
        if self.format not in allowed_formats:
            raise ImageValidationError(
                f"Invalid or spoofed image format (detected magic bytes: '{self.format}'). "
                f"Allowed formats: {sorted(allowed_formats)}.",
                status_code=400,
            )

        # 4. Cross-check extension vs magic bytes
        if check_extension and self.filename:
            ext = Path(self.filename).suffix.lower()
            expected_exts = FORMAT_TO_EXTENSIONS.get(self.format, set())
            if ext and expected_exts and ext not in expected_exts:
                raise ImageValidationError(
                    f"File extension '{ext}' does not match actual magic-byte format '{self.format}'.",
                    status_code=400,
                )

        # 5. Check stream integrity & truncated bytes
        if self.format == "JPEG":
            trimmed = self.raw_bytes.rstrip(b"\x00\r\n\t ")
            if not trimmed.endswith(b"\xff\xd9"):
                raise ImageValidationError(
                    "Truncated or incomplete JPEG byte stream (missing EOI marker).",
                    status_code=400,
                )

        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(io.BytesIO(self.raw_bytes)) as pil_check:
                pil_check.verify()
            with Image.open(io.BytesIO(self.raw_bytes)) as pil_load:
                pil_load.load()
                w, h = pil_load.size
        except Exception as exc:
            raise ImageValidationError(
                f"Corrupt or truncated image data: {exc}",
                status_code=400,
            ) from exc

        if self._decoded_raw is None:
            self._try_inspect_dimensions()
        if self._decoded_raw is None or self.lebarPx <= 0 or self.tinggiPx <= 0:
            raise ImageValidationError(
                "Unable to decode image pixel array.",
                status_code=400,
            )

        # 6. Check minimum resolution requirement
        if self.lebarPx < min_width or self.tinggiPx < min_height:
            raise ImageValidationError(
                f"Image resolution {self.lebarPx}x{self.tinggiPx}px is below the minimum "
                f"required resolution of {min_width}x{min_height}px.",
                status_code=400,
            )

        return True

    def has_exif(self) -> bool:
        """Return True if the image currently contains any EXIF metadata tags."""
        try:
            with Image.open(io.BytesIO(self.raw_bytes)) as img:
                exif = img.getexif()
                if exif and len(exif) > 0:
                    return True
                if img.info.get("exif"):
                    return True
        except Exception:
            pass
        return False

    def stripExif(self, jpeg_quality: int = 95) -> bytes:
        """Remove all EXIF/GPS/camera metadata for patient privacy (de-identification).

        Reconstructs a clean pixel-only image buffer in the same container format,
        updates `self.raw_bytes` and `self.sha256`, and returns the sanitized bytes.
        """
        if not self.raw_bytes or self.format == "UNKNOWN":
            raise ImageValidationError("Cannot strip EXIF from invalid or empty image bytes.")

        with Image.open(io.BytesIO(self.raw_bytes)) as img:
            # Apply EXIF orientation before stripping tags so image isn't rotated wrong
            img_oriented = ImageOps.exif_transpose(img)
            clean_img = Image.fromarray(np.asarray(img_oriented))

            buf = io.BytesIO()
            save_fmt = self.format if self.format in {"JPEG", "PNG", "WEBP", "BMP"} else "PNG"
            if save_fmt == "JPEG":
                if clean_img.mode not in ("RGB", "L"):
                    clean_img = clean_img.convert("RGB")
                clean_img.save(buf, format="JPEG", quality=jpeg_quality)
            else:
                clean_img.save(buf, format=save_fmt)

        self.raw_bytes = buf.getvalue()
        self.computeHash()
        self._try_inspect_dimensions()
        return self.raw_bytes

    def to_bgr_array(self) -> np.ndarray:
        """Return a 3-channel BGR uint8 OpenCV image array (handling 1-ch Grayscale and 4-ch RGBA)."""
        if self._decoded_raw is None:
            self._try_inspect_dimensions()
        if self._decoded_raw is None:
            raise ImageValidationError("Image cannot be decoded to BGR array.")

        arr = self._decoded_raw
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.shape[2] == 3:
            return arr.copy()
        raise ImageValidationError(f"Unsupported channel count: {arr.shape}")
