# src/tiletorch/codecs.py
from __future__ import annotations
from io import BytesIO
from PIL import Image

def encode_jpeg(img: Image.Image, quality: int = 92, subsampling: int = 0, optimize: bool = True) -> bytes:
    """
    JPEG 4:4:4 by default (subsampling=0) — recommended for H&E.
    """
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=optimize)
    return buf.getvalue()

def encode_webp(img: Image.Image, quality: int = 90, method: int = 6) -> bytes:
    """
    WebP lossy (smaller than JPEG at similar perceptual quality).
    """
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=method)  # method 0..6, slower→better
    return buf.getvalue()

def encode_image(img: Image.Image, fmt: str = "jpeg", **kwargs) -> bytes:
    """
    Generic gateway so sinks can call one function.
    fmt: 'jpeg' or 'webp'
    """
    fmt = fmt.lower()
    if fmt == "jpeg":
        return encode_jpeg(img, **kwargs)
    if fmt == "webp":
        return encode_webp(img, **kwargs)
    raise ValueError(f"Unsupported format: {fmt}")