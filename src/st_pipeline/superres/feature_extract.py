from __future__ import annotations

from pathlib import Path
import pickle
import warnings

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from PIL import Image

from .hipt.hipt_4k import HIPT_4K
from .hipt.hipt_model_utils import eval_transforms

Image.MAX_IMAGE_PIXELS = None


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pad_to_multiple(x: np.ndarray, multiple: int, pad_value: int = 255) -> np.ndarray:
    h, w = x.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return x
    if x.ndim == 2:
        pad_spec = ((0, pad_h), (0, pad_w))
    else:
        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(x, pad_spec, mode="constant", constant_values=pad_value)


def _patchify(x: np.ndarray, patch_size: int):
    shape_ori = np.array(x.shape[:2])
    shape_ext = ((shape_ori + patch_size - 1) // patch_size) * patch_size
    x = np.pad(
        x,
        (
            (0, shape_ext[0] - x.shape[0]),
            (0, shape_ext[1] - x.shape[1]),
            (0, 0),
        ),
        mode="edge",
    )
    tiles_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size
        b0 = a0 + patch_size
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size
            b1 = a1 + patch_size
            tiles.append(x[a0:b0, a1:b1])
    shapes = dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)
    return tiles, shapes


def _get_embeddings_sub(model: HIPT_4K, x: np.ndarray):
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def _get_embeddings_cls(model: HIPT_4K, x: np.ndarray):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    x_sub4k = x_sub4k[0].transpose(1, 2, 0)
    return x_sub4k


def _downsample_rgb(img: np.ndarray, factor: int = 16) -> np.ndarray:
    img = _pad_to_multiple(img, factor, pad_value=255)
    out = np.stack(
        [
            reduce(
                img[..., i].astype(np.float32) / 255.0,
                "(h1 h) (w1 w) -> h1 w1",
                "mean",
                h=factor,
                w=factor,
            ).astype(np.float32)
            for i in range(3)
        ],
        axis=0,
    )
    return out


def _smoothen_channels(channels, size: int = 4):
    try:
        from scipy.ndimage import uniform_filter
    except Exception:
        warnings.warn("scipy not available; skip smoothing")
        return channels
    out = []
    for c in channels:
        out.append(uniform_filter(c, size=size))
    return out


def extract_embeddings(
    he_path: Path,
    out_path: Path,
    device: str | torch.device = "cuda",
    model_dir: Path | None = None,
    smoothen: bool = True,
    use_shift: bool = False,
    force: bool = False,
) -> Path:
    he_path = Path(he_path)
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path

    img = _load_image(he_path)
    img = _pad_to_multiple(img, 256, pad_value=255)

    model256_path = None
    model4k_path = None
    if model_dir is not None:
        model256_path = str(Path(model_dir) / "vit256_small_dino.pth")
        model4k_path = str(Path(model_dir) / "vit4k_xs_dino.pth")

    model = HIPT_4K(
        model256_path=model256_path,
        model4k_path=model4k_path,
        device256=device,
        device4k=device,
    )
    model.eval()

    tile_size = 4096
    tiles, shapes = _patchify(img, patch_size=tile_size)

    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print("tile", i, "/", len(tiles))
        x_mid, x_sub = _get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)

    emb_mid = rearrange(
        emb_mid,
        "(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k",
        h1=shapes["tiles"][0],
        w1=shapes["tiles"][1],
    )

    emb_cls = _get_embeddings_cls(model, emb_mid)

    shape_orig = np.array(shapes["original"]) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
            np.array([e[..., i] for e in emb_sub]),
            "(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)",
            h1=shapes["tiles"][0],
            w1=shapes["tiles"][1],
        )
        chan = chan[: shape_orig[0], : shape_orig[1]]
        chans_sub.append(chan)

    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
            np.array([e[..., i] for e in emb_cls]),
            "h12 w12 -> (h12 h3) (w12 w3)",
            h3=n_subpatches[0],
            w3=n_subpatches[1],
        )
        chan = chan[: shape_orig[0], : shape_orig[1]]
        chans_cls.append(chan)

    embs = {"cls": chans_cls, "sub": chans_sub}
    embs["rgb"] = _downsample_rgb(img, factor=16)

    if smoothen:
        embs["cls"] = _smoothen_channels(embs["cls"], size=16)
        embs["sub"] = _smoothen_channels(embs["sub"], size=4)

    _save_pickle(embs, out_path)
    print("Saved embeddings:", out_path)
    return out_path
