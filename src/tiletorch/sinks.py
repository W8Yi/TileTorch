# src/tiletorch/sinks.py
from __future__ import annotations
import io, json, tarfile, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
from .codecs import encode_image

class TileSink:
    """
    Abstract tile sink: receives (pil_image, meta_dict, key_str) and writes somewhere.
    """
    def append(self, img: Image.Image, meta: Dict, key: str) -> str:
        raise NotImplementedError
    def close(self) -> None:
        pass

class FileSink(TileSink):
    """
    Writes each tile as an individual file under out_dir using requested format (jpeg/webp).
    The returned string is the file path; also writes optional JSON sidecar if wanted.
    """
    def __init__(self, out_dir: str | Path, fmt: str = "jpeg", codec_kwargs: Optional[Dict] = None, write_json: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ext = ".jpg" if fmt.lower() == "jpeg" else ".webp"
        self.fmt = fmt.lower()
        self.codec_kwargs = codec_kwargs or {}
        self.write_json = write_json

    def append(self, img: Image.Image, meta: Dict, key: str) -> str:
        data = encode_image(img, self.fmt, **self.codec_kwargs)
        fpath = self.out_dir / f"{key}{self.ext}"
        with open(fpath, "wb") as f:
            f.write(data)
        if self.write_json:
            (self.out_dir / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")
        return str(fpath)

class WebDatasetSink(TileSink):
    """
    Writes rolling WebDataset shards: tiles-000000.tar, tiles-000001.tar, ...
    Limits by max_count or max_bytes per shard. Adds a JSON sidecar per sample.
    """
    def __init__(
        self,
        shard_pattern: str | Path,              # e.g. "data/shards/tiles-%06d.tar"
        fmt: str = "jpeg",
        codec_kwargs: Optional[Dict] = None,
        max_count: int = 10000,                 # tiles per shard limit
        max_bytes: int = 750 * 1024 * 1024,     # ~750MB per shard limit
    ):
        self.shard_pattern = str(shard_pattern)
        Path(self.shard_pattern).parent.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt.lower()
        self.codec_kwargs = codec_kwargs or {}
        self.max_count = max_count
        self.max_bytes = max_bytes

        self._shard_idx = 0
        self._tar: Optional[tarfile.TarFile] = None
        self._count = 0
        self._bytes = 0
        self._open_new_shard()

    def _open_new_shard(self):
        if self._tar is not None:
            self._tar.close()
        shard_path = self.shard_pattern % self._shard_idx if "%d" in self.shard_pattern else f"{self.shard_pattern}-{self._shard_idx:06d}.tar"
        self._tar = tarfile.open(shard_path, "w")
        self._count = 0
        self._bytes = 0

    def _need_roll(self, nbytes: int) -> bool:
        return (self._count >= self.max_count) or (self._bytes + nbytes > self.max_bytes)

    def append(self, img: Image.Image, meta: Dict, key: str) -> str:
        assert self._tar is not None
        # encode
        img_bytes = encode_image(img, self.fmt, **self.codec_kwargs)
        img_name = f"{key}.{ 'jpg' if self.fmt == 'jpeg' else 'webp' }"
        meta_json = json.dumps(meta).encode("utf-8")
        img_info = tarfile.TarInfo(img_name); img_info.size = len(img_bytes); img_info.mtime = int(time.time())
        json_info = tarfile.TarInfo(f"{key}.json"); json_info.size = len(meta_json); json_info.mtime = img_info.mtime

        # roll shard if needed
        if self._need_roll(len(img_bytes) + len(meta_json)):
            self._shard_idx += 1
            self._open_new_shard()

        # write both
        self._tar.addfile(img_info, io.BytesIO(img_bytes))
        self._tar.addfile(json_info, io.BytesIO(meta_json))

        self._count += 1
        self._bytes += len(img_bytes) + len(meta_json)
        # return a logical URI for the sample (shard path + key)
        # (webdataset readers usually need just the shard path; return name for logging)
        return f"{self._shard_idx}:{img_name}"

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None