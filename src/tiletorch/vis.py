# src/tiletorch/vis.py
from __future__ import annotations
from typing import Iterable, Tuple, Sequence
from pathlib import Path
from PIL import Image, ImageDraw
import openslide

def _level_scale(level_dims: Sequence[Tuple[int, int]], level: int, thumb_size: Tuple[int, int]):
    """Return scale factors from level pixels → thumbnail pixels."""
    lw, lh = level_dims[level]
    tw, th = thumb_size
    sx = tw / lw
    sy = th / lh
    return sx, sy

def make_tile_thumbnail(
    slide_path: str,
    level: int,
    tile_size: int,
    coords: Iterable[Tuple[int, int]],
    kept_mask: Iterable[bool],
    out_path: str | Path,
    thumb_max: int = 2048,
    kept_color: Tuple[int, int, int] = (0, 200, 0),
    drop_color: Tuple[int, int, int] = (220, 0, 0),
    line_w: int = 2,
) -> str:
    """
    Create a thumbnail and overlay rectangles for each tile.
    kept_mask aligns with coords (True=kept, False=dropped).
    """
    slide = openslide.OpenSlide(slide_path)
    try:
        level_dims = [tuple(map(int, d)) for d in slide.level_dimensions]
        lw, lh = level_dims[level]
        # Choose thumbnail size constrained by thumb_max while keeping aspect
        ratio = max(lw, lh) / float(thumb_max)
        tw, th = (max(1, int(lw / ratio)), max(1, int(lh / ratio)))
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")

        sx, sy = _level_scale(level_dims, level, (tw, th))
        draw = ImageDraw.Draw(thumb)

        for (x, y), keep in zip(coords, kept_mask):
            x0 = int(x * sx)
            y0 = int(y * sy)
            x1 = int((x + tile_size) * sx)
            y1 = int((y + tile_size) * sy)
            color = kept_color if keep else drop_color
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)

        out_path = str(Path(out_path))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        thumb.save(out_path, format="PNG")
        return out_path
    finally:
        slide.close()