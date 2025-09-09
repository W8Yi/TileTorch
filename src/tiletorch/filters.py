# src/tiletorch/filters.py
from __future__ import annotations
from typing import Callable, Dict, Iterable
import numpy as np
from PIL import Image
import cv2  # opencv-python-headless

# ---------- Metrics ----------

def compute_metrics(img: Image.Image) -> Dict[str, float]:
    """
    Compute inexpensive per-tile metrics.
    Returns keys used by default_tile_filter in tiling.py:
      - "tissue_frac" : fraction of non-white, sufficiently saturated pixels
      - "entropy"     : Shannon entropy of grayscale histogram
      - "lapvar"      : variance of Laplacian (focus proxy)
    """
    # Tissue fraction (HSV heuristic)
    small = img.resize((64, 64), Image.BILINEAR)
    hsv = np.array(small.convert("HSV"), dtype=np.uint8)
    s = hsv[..., 1]; v = hsv[..., 2]
    tissue_frac = float(((s > 20) & (v < 245)).mean())

    # Entropy
    g = np.array(img.convert("L"))
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    # Focus
    lap = cv2.Laplacian(g, cv2.CV_64F)
    lapvar = float(lap.var())

    return {"tissue_frac": tissue_frac, "entropy": entropy, "lapvar": lapvar}

# ---------- Predicate builders ----------

from PIL import Image
def keep_tissue(min_tissue: float = 0.10) -> Callable[[Image.Image, Dict[str, float]], bool]:
    def pred(img, m):  # noqa: ARG001
        return m.get("tissue_frac", 0.0) >= min_tissue
    return pred

def keep_entropy(min_entropy: float = 3.5) -> Callable[[Image.Image, Dict[str, float]], bool]:
    def pred(img, m):  # noqa: ARG001
        return m.get("entropy", 0.0) >= min_entropy
    return pred

def keep_focus(min_lapvar: float = 60.0) -> Callable[[Image.Image, Dict[str, float]], bool]:
    def pred(img, m):  # noqa: ARG001
        return m.get("lapvar", 0.0) >= min_lapvar
    return pred

def compose_all(preds: Iterable[Callable[[Image.Image, Dict[str, float]], bool]]):
    preds = list(preds)
    def f(img, m): return all(p(img, m) for p in preds)
    return f

def compose_any(preds: Iterable[Callable[[Image.Image, Dict[str, float]], bool]]):
    preds = list(preds)
    def f(img, m): return any(p(img, m) for p in preds)
    return f

def make_filter(
    min_tissue: float = 0.10,
    min_entropy: float = 3.5,
    min_lapvar: float = 0.0,  # set >0 to enforce focus
    mode: str = "all",        # "all" (AND) or "any" (OR)
):
    preds = [keep_tissue(min_tissue), keep_entropy(min_entropy)]
    if min_lapvar > 0:
        preds.append(keep_focus(min_lapvar))
    comb = compose_all if mode == "all" else compose_any
    return comb(preds)

# ---------- Mask-aware predicate (optional) ----------

def keep_in_mask(mask: np.ndarray, min_frac: float = 0.5):
    """
    Returns a predicate that keeps tiles overlapping a level-0 ROI mask by >= min_frac.
    To use with tiling.extract_tiles, wrap it with a closure that supplies x,y,level,tile_size
    (or extend your filter signature).
    """
    def pred_with_coords(img, m, *, x: int, y: int, level: int, tile_size: int):
        scale = 2 ** level
        x0, y0 = x * scale, y * scale
        x1, y1 = x0 + tile_size * scale, y0 + tile_size * scale
        roi = mask[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        return float(roi.mean()) >= min_frac
    return pred_with_coords