# src/tiletorch/tiling.py
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import openslide
from PIL import Image
from tqdm import tqdm

from .filters import compute_metrics  # metric impl lives in filters.py
from . import vis  # for preview thumbnail
from .sinks import TileSink

# ---------------------------
# Data container
# ---------------------------

@dataclass
class TileRecord:
    slide_id: str
    tile_path: str
    x: int
    y: int
    level: int
    tile_size: int
    tissue_frac: float
    entropy: float


# ---------------------------
# Grid generation
# ---------------------------

def _generate_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int = 0,
    keep_edge: bool = False,
) -> Iterable[Tuple[int, int]]:
    """Yield (x, y) top-left coordinates for a regular grid at the given level."""
    assert tile_size > 0, "tile_size must be positive"
    assert 0 <= overlap < tile_size, "overlap must be in [0, tile_size)"
    stride = tile_size - overlap

    if not keep_edge:
        xmax = width - tile_size
        ymax = height - tile_size
        for x in range(0, xmax + 1, stride):
            for y in range(0, ymax + 1, stride):
                yield x, y
        return

    # include edges
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    for x in xs:
        for y in ys:
            yield x, y


# ---------------------------
# Default filter (simple, threshold-based)
# ---------------------------

def default_tile_filter(
    tile_img: Image.Image,
    metrics: Dict[str, float],
    *,
    min_tissue: float = 0.10,
    min_entropy: float = 3.5,
) -> bool:
    """
    Keep the tile if it has enough tissue and texture.
    Metrics must include keys: "tissue_frac", "entropy".
    """
    return (metrics.get("tissue_frac", 0.0) >= min_tissue) and (metrics.get("entropy", 0.0) >= min_entropy)


# ---------------------------
# Public API: tiling
# ---------------------------

def extract_tiles(
    slide_path: str,
    out_dir: str | Path,
    tile_size: int = 256,
    level: int = 0,
    overlap: int = 0,
    keep_edge: bool = False,
    # filtering controls
    filter_fn: Optional[Callable[[Image.Image, Dict[str, float]], bool]] = None,
    filter_kwargs: Optional[Dict] = None,
    # outputs
    write_index_csv: Optional[str | Path] = None,
    jpeg_quality: int = 95,
    progress: bool = True,
) -> List[TileRecord]:
    """
    Read a WSI via OpenSlide, save tiles to disk, and return a tile index.

    Filtering:
      - Provide `filter_fn(tile_img, metrics)->bool` or it uses `default_tile_filter`.
      - `compute_metrics` (from filters.py) supplies: {"tissue_frac", "entropy", ... if extended}.
    """
    slide_path = str(slide_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slide_id = Path(slide_path).stem
    slide = openslide.OpenSlide(slide_path)

    try:
        if level < 0 or level >= len(slide.level_dimensions):
            raise ValueError(f"Requested level={level}, but slide has {len(slide.level_dimensions)} levels.")
        width, height = slide.level_dimensions[level]

        coords = list(_generate_grid(width, height, tile_size, overlap, keep_edge=keep_edge))
        iterator = tqdm(coords, desc=f"Tiling {slide_id} (L{level})") if progress else coords

        index: List[TileRecord] = []

        # choose filter
        filt = filter_fn or default_tile_filter
        fkw = filter_kwargs or {}

        for x, y in iterator:
            region = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            if region.size != (tile_size, tile_size):
                continue  # safety near edges on some pyramids

            metrics = compute_metrics(region)
            if not filt(region, metrics, **fkw):
                continue

            # Save tile & record
            fname = f"{slide_id}_x{x}_y{y}_lv{level}_sz{tile_size}.jpg"
            fpath = out_dir / fname
            region.save(fpath, format="JPEG", quality=jpeg_quality, subsampling=0)

            index.append(
                TileRecord(
                    slide_id=slide_id,
                    tile_path=str(fpath),
                    x=x,
                    y=y,
                    level=level,
                    tile_size=tile_size,
                    tissue_frac=float(metrics.get("tissue_frac", 0.0)),
                    entropy=float(metrics.get("entropy", 0.0)),
                )
            )

        if write_index_csv:
            _write_index_csv(write_index_csv, index)

        return index

    finally:
        slide.close()


# ---------------------------
# Public API: preview-only (thumbnail overlay)
# ---------------------------

def preview_tiling(
    slide_path: str,
    out_path: str | Path,
    tile_size: int = 256,
    level: int = 0,
    overlap: int = 0,
    keep_edge: bool = False,
    filter_fn: Optional[Callable[[Image.Image, Dict[str, float]], bool]] = None,
    filter_kwargs: Optional[Dict] = None,
    thumb_max: int = 2048,
    progress: bool = True,
) -> Dict[str, float | int | str]:
    """
    Generate a tiling preview thumbnail only (no tiles written).
    Returns stats:
      {"preview_path", "total_tiles", "kept_tiles", "kept_ratio"}
    """
    slide_path = str(slide_path)
    out_path = Path(out_path)
    slide_id = Path(slide_path).stem

    slide = openslide.OpenSlide(slide_path)
    try:
        if level < 0 or level >= len(slide.level_dimensions):
            raise ValueError(f"Requested level={level}, but slide has {len(slide.level_dimensions)} levels.")
        width, height = slide.level_dimensions[level]

        coords = list(_generate_grid(width, height, tile_size, overlap, keep_edge=keep_edge))
        iterator = tqdm(coords, desc=f"Preview {slide_id} (L{level})") if progress else coords

        kept_mask: List[bool] = []
        pred = filter_fn or default_tile_filter
        fkw = filter_kwargs or {}

        for x, y in iterator:
            region = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            if region.size != (tile_size, tile_size):
                kept_mask.append(False)
                continue
            metrics = compute_metrics(region)
            kept_mask.append(bool(pred(region, metrics, **fkw)))

        preview_path = vis.make_tile_thumbnail(
            slide_path=slide_path,
            level=level,
            tile_size=tile_size,
            coords=coords,
            kept_mask=kept_mask,
            out_path=out_path,
            thumb_max=thumb_max,
        )

        total = len(coords)
        kept = int(sum(1 for k in kept_mask if k))
        kept_ratio = (kept / total) if total else 0.0

        return {
            "preview_path": str(preview_path),
            "total_tiles": total,
            "kept_tiles": kept,
            "kept_ratio": kept_ratio,
        }
    finally:
        slide.close()


# ---------------------------
# Helpers
# ---------------------------

def summarize_slide(slide_path: str) -> dict:
    slide = openslide.OpenSlide(slide_path)
    try:
        dims = [tuple(map(int, d)) for d in slide.level_dimensions]
        return {"slide_path": slide_path, "levels": len(dims), "level_dimensions": dims}
    finally:
        slide.close()


def _write_index_csv(path: str | Path, records: List[TileRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slide_id", "tile_path", "x", "y", "level", "tile_size", "tissue_frac", "entropy"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

# ---------------------------
# Public API: tiling to sink (e.g. WebDataset shards)
# ---------------------------

def extract_tiles_to_sink(
    slide_path: str,
    sink: TileSink,
    tile_size: int = 256,
    level: int = 0,
    overlap: int = 0,
    keep_edge: bool = False,
    filter_fn: Optional[Callable[[Image.Image, Dict[str, float]], bool]] = None,
    filter_kwargs: Optional[Dict] = None,
    progress: bool = True,
) -> List[TileRecord]:
    """
    Same as extract_tiles, but writes each kept tile to a user-provided sink (FileSink, WebDatasetSink, ...).
    """
    slide_path = str(slide_path)
    slide_id = Path(slide_path).stem
    slide = openslide.OpenSlide(slide_path)
    try:
        if level < 0 or level >= len(slide.level_dimensions):
            raise ValueError(f"Requested level={level}, but slide has {len(slide.level_dimensions)} levels.")
        width, height = slide.level_dimensions[level]
        coords = list(_generate_grid(width, height, tile_size, overlap, keep_edge=keep_edge))
        iterator = tqdm(coords, desc=f"Tiling {slide_id} (L{level})") if progress else coords

        filt = filter_fn or default_tile_filter
        fkw = filter_kwargs or {}
        index: List[TileRecord] = []

        for x, y in iterator:
            region = slide.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            if region.size != (tile_size, tile_size):
                continue
            metrics = compute_metrics(region)
            if not filt(region, metrics, **fkw):
                continue

            key = f"{slide_id}_x{x}_y{y}_lv{level}_sz{tile_size}"
            meta = {
                "slide_id": slide_id, "x": x, "y": y, "level": level, "tile_size": tile_size,
                **metrics,
            }
            sink_uri = sink.append(region, meta, key)  # file path or shard:key

            index.append(
                TileRecord(
                    slide_id=slide_id,
                    tile_path=str(sink_uri),
                    x=x, y=y,
                    level=level, tile_size=tile_size,
                    tissue_frac=float(metrics.get("tissue_frac", 0.0)),
                    entropy=float(metrics.get("entropy", 0.0)),
                )
            )
        return index
    finally:
        slide.close()
        try:
            sink.close()
        except Exception:
            pass