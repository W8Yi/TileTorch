"""
TileTorch: Whole Slide Image tiling, dataset building, and training utilities.
"""

# Package version (kept in sync with pyproject.toml)
__version__ = "0.1.0"

# Expose key public functions
from .tiling import extract_tiles, extract_tiles_to_sink
# from .dataset import TileDataset
# from .training import train_model

__all__ = [
    "extract_tiles",
    "extract_tiles_to_sink",
    "TileDataset",
    "train_model",
    "__version__",
]
