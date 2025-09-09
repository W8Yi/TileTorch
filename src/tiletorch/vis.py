# src/tiletorch/vis.py
from PIL import Image, ImageDraw
import openslide
from pathlib import Path
from typing import Iterable, Tuple, Sequence

def _level_scale(level_dims: Sequence[Tuple[int, int]], level: int, thumb_size: Tuple[int, int]):
    lw, lh = level_dims[level]
    tw, th = thumb_size
    return tw / lw, th / lh

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
    slide = openslide.OpenSlide(slide_path)
    try:
        level_dims = [tuple(map(int, d)) for d in slide.level_dimensions]
        lw, lh = level_dims[level]

        # *** ALWAYS DOWNSCALE & CAP TOTAL PIXELS ***
        if max(lw, lh) > thumb_max:
            ratio = max(lw, lh) / float(thumb_max)
        else:
            ratio = 1.0

        tw = max(1, int(lw / ratio))
        th = max(1, int(lh / ratio))

        MAX_PIXELS = 3_000_000  # ~3MP safety ceiling
        if tw * th > MAX_PIXELS:
            scale = (tw * th / MAX_PIXELS) ** 0.5
            tw = max(1, int(tw / scale))
            th = max(1, int(th / scale))

        thumb = slide.get_thumbnail((tw, th)).convert("RGB")

        sx, sy = _level_scale(level_dims, level, (tw, th))
        draw = ImageDraw.Draw(thumb)

        for (x, y), keep in zip(coords, kept_mask):
            x0 = int(x * sx); y0 = int(y * sy)
            x1 = int((x + tile_size) * sx); y1 = int((y + tile_size) * sy)
            draw.rectangle([x0, y0, x1, y1], outline=(kept_color if keep else drop_color), width=line_w)

        out_path = str(Path(out_path))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        thumb.save(out_path, format="PNG")
        return out_path
    finally:
        slide.close()