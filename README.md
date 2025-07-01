# TileTorch

**Blazing-fast WSI tiling & dataloaders for PyTorch.**  
Turn gigapixel pathology slides into ready-to-train tensors for Multi-Instance Learning, Vision-Transformers, and latent-diffusion models—all with one import.

<p align="center">
  <img src="docs/_static/tiletorch_logo.svg" width="220" alt="TileTorch logo">
</p>

---

## Key features

| Feature | What it gives you |
|---------|------------------|
| **GPU-accelerated slide I/O** *(cuCIM → CPU fallback)* | 700 + 256 px crops / s on a single NVMe |
| **Typed coordinate catalogue** *(Parquet/Arrow)* | Loss-less metadata with instant filtering & joins |
| **Streaming tar shards** *(WebDataset)* | Append-only, S3-friendly, no “10⁵ small files” tax |
| **Zarr feature/latent cache** | Store ViT embeddings or VAE latents in ~5 MB/slide |
| **PyTorch `IterableDataset`s** | One-liner loaders for RGB, features, or latents |
| **Scale tokens & stain jitter** | Built-in multi-magnification & colour augmentations |
| **CLI pipeline** | `tiletorch mask → tile → shard → feats` in four commands |

---

## Installation

```bash
# CUDA 12 example
pip install tiletorch[cucim]   # core + GPU back-end
# or CPU-only
pip install tiletorch

