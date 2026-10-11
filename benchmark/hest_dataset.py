import os
import sys
from pathlib import Path
import math
import numpy as np
import scipy.sparse

# Torch is imported tolerantly because one environment this benchmark runs in cannot
# import it at all. `stimage_bench` on the cluster ships torch 2.4.1+cu118 next to
# NCCL 2.16.5+cuda12.0, and `libtorch_cuda.so` wants `ncclCommRegister`, which NCCL
# only added in 2.19 -- so `import torch` raises
# `undefined symbol: ncclCommRegister` no matter how LD_LIBRARY_PATH or LD_PRELOAD
# are arranged (both were tried and recorded). STimage is a TensorFlow method and
# needs no torch, but this module is shared, so the import is tolerated here and the
# torch-backed datasets below fail loudly if they are actually used without it.
# This mirrors the reference HEST adaptation of STimage, which decoupled the STimage
# loader from the PyTorch loaders for the same reason.
try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
    _TORCH_IMPORT_ERROR = None
except Exception as _torch_exc:  # noqa: BLE001 - any failure here is the documented one
    torch = None
    HAS_TORCH = False
    _TORCH_IMPORT_ERROR = _torch_exc

    class Dataset:  # type: ignore[no-redef]
        """Stand-in for `torch.utils.data.Dataset` when torch is unimportable.

        Every dataset class in this module calls `super().__init__()` first, so
        raising here turns "silently missing torch" into a clear message at the
        point of use and leaves the torch-free STimage path unaffected.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "This dataset needs torch, which failed to import in this environment "
                f"({type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}). "
                "Only the TensorFlow-based STimage row is expected to run without torch."
            )


import anndata as ad
from PIL import Image
from scipy.spatial import distance

# 尝试修复 Windows 下 OpenSlide 找不到 DLL 的问题
if os.name == 'nt':
    # 尝试寻找当前 Conda 环境下的 Library/bin
    conda_prefix = os.environ.get('CONDA_PREFIX')
    curr_python = sys.executable
    # 简单的推断策略
    possible_paths = []
    if conda_prefix:
        possible_paths.append(Path(conda_prefix) / 'Library' / 'bin')
    if curr_python:
        possible_paths.append(Path(curr_python).parent / 'Library' / 'bin')
        possible_paths.append(Path(curr_python).parent / '..' / 'Library' / 'bin')
    
    # 显式添加找到的路径
    for p in possible_paths:
        # Check for either v1 or v0 DLLs
        if p.exists() and ((p / 'libopenslide-1.dll').exists() or (p / 'libopenslide-0.dll').exists()):
            try:
                os.add_dll_directory(str(p))
            except AttributeError:
                pass # Python < 3.8
            os.environ['PATH'] = str(p) + os.pathsep + os.environ['PATH']
            break

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


def _read_h5ad(path):
    return ad.read_h5ad(path)


def _normalize_total(adata, target_sum=1e4):
    x = adata.X
    if scipy.sparse.issparse(x):
        sums = np.asarray(x.sum(axis=1)).ravel()
        scale = np.divide(target_sum, sums, out=np.zeros_like(sums, dtype=np.float64), where=sums != 0)
        adata.X = scipy.sparse.diags(scale).dot(x)
    else:
        x = np.asarray(x, dtype=np.float32)
        sums = x.sum(axis=1, keepdims=True)
        scale = np.divide(target_sum, sums, out=np.zeros_like(sums, dtype=np.float32), where=sums != 0)
        adata.X = x * scale


def _log1p(adata):
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.copy()
        adata.X.data = np.log1p(adata.X.data)
    else:
        adata.X = np.log1p(adata.X)

class HESTHisToGeneDataset(Dataset):
    def __init__(self, slide_ids, hest_dir, spatial_dir, genes, patch_size=112, sr=False, n_pos=64, train=True):
        super().__init__()
        self.slide_ids = slide_ids
        self.hest_dir = Path(hest_dir)
        self.spatial_dir = Path(spatial_dir)
        self.genes = list(genes)
        self.patch_size = patch_size
        self.r = patch_size // 2
        self.sr = sr
        self.n_pos = n_pos
        self.train = train

        self.data_list = []  # Store (slide_id, index_in_adata) tuples
        self.adatas = {}
        # Lazy loading: store paths instead of open objects to avoid pickling issues
        self.wsi_paths = {} 
        self.wsis = {} # Cache for opened WSIs (process-local)

        print(f"Loading {len(slide_ids)} slides...")
        for sid in slide_ids:
            # Load h5ad
            h5ad_path = self.spatial_dir / f'{sid}.h5ad'
            adata = _read_h5ad(h5ad_path)
            
            # Filter genes
            if self.genes:
                adata = adata[:, self.genes].copy()
            
            # Preprocess
            if 'counts' in adata.layers:
                adata.X = adata.layers['counts'].copy()
            elif scipy.sparse.issparse(adata.X):
                adata.X = adata.X.toarray()
                
            # Standard ST preprocessing
            _normalize_total(adata, target_sum=1e4)
            _log1p(adata)
            
            # Identify WSI path
            wsi_path = list((self.hest_dir / 'wsis').glob(f"{sid}.*"))[0]
            self.wsi_paths[sid] = str(wsi_path)

            self.adatas[sid] = adata
            
            # Compute grid coordinates for this slide
            coords = adata.obsm['spatial'] # (N, 2) [x, y] in pixels
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            
            span_x = max_x - min_x + 1e-6
            span_y = max_y - min_y + 1e-6
            
            grid_x = np.floor((coords[:, 0] - min_x) / span_x * (self.n_pos - 1)).astype(int)
            grid_y = np.floor((coords[:, 1] - min_y) / span_y * (self.n_pos - 1)).astype(int)
            
            self.adatas[sid].obsm['grid_coords'] = np.stack([grid_x, grid_y], axis=1)

            for i in range(adata.shape[0]):
                self.data_list.append((sid, i))
                
        print(f"Total spots: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sid, i = self.data_list[idx]
        adata = self.adatas[sid]
        
        # Lazy open WSI if not present in this process/thread
        if sid not in self.wsis:
            wsi_path = self.wsi_paths[sid]
            try:
                if HAS_OPENSLIDE:
                    self.wsis[sid] = openslide.OpenSlide(wsi_path)
                else:
                    self.wsis[sid] = Image.open(wsi_path)
            except Exception as e:
                print(f"Error opening WSI for {sid}: {e}, using PIL fallback")
                self.wsis[sid] = Image.open(wsi_path)
        
        wsi = self.wsis[sid]
        
        center = adata.obsm['spatial'][i].astype(int)
        grid_pos = adata.obsm['grid_coords'][i]
        exp = adata.X[i]
        if hasattr(exp, 'toarray'):
            exp = exp.toarray().flatten()
        
        cx, cy = center
        # Crop patch
        x_tl = cx - self.r
        y_tl = cy - self.r
        
        try:
            if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
                patch = wsi.read_region((int(x_tl), int(y_tl)), 0, (self.patch_size, self.patch_size)).convert('RGB')
            else:
                # PIL Image
                patch = wsi.crop((x_tl, y_tl, x_tl + self.patch_size, y_tl + self.patch_size)).convert('RGB')
        except Exception as e:
            # Fill with zeros if out of bounds or error
            # print(f"Error reading patch for {sid} at {center}: {e}")
            patch = Image.new('RGB', (self.patch_size, self.patch_size))

        # Transform to tensor and normalize
        patch_tensor = torch.from_numpy(np.array(patch)).float() / 255.0
        if patch_tensor.shape[0] != self.patch_size or patch_tensor.shape[1] != self.patch_size:
             patch_tensor = torch.zeros((self.patch_size, self.patch_size, 3))
        
        patch_tensor = patch_tensor.permute(2, 0, 1) # [C, H, W]
        patch_flat = patch_tensor.flatten()
        # Transform to [1, ...] sequence format for HisToGene (ViT expects inputs as sequences)
        patch_flat = patch_flat.unsqueeze(0) # [1, 3*H*W]
        grid_pos = torch.LongTensor(grid_pos).unsqueeze(0) # [1, 2]
        exp = torch.Tensor(exp).unsqueeze(0) # [1, n_genes]

        return patch_flat, grid_pos, exp


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    spatialMatrix = coord
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


def _safe_corr(a, b):
    a = a.astype(float)
    b = b.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return -1e9
    return float((a * b).sum() / denom)


def _align_array_to_spatial(array_col, array_row, spatial):
    # Choose swap/flip that best aligns array indices with spatial x/y axes.
    x = spatial[:, 0]
    y = spatial[:, 1]
    candidates = []
    for swap in (False, True):
        ax = array_row if swap else array_col
        ay = array_col if swap else array_row
        for flip_x in (1, -1):
            for flip_y in (1, -1):
                gx = ax * flip_x
                gy = ay * flip_y
                score = _safe_corr(gx, x) + _safe_corr(gy, y)
                candidates.append((score, gx, gy))
    best = max(candidates, key=lambda item: item[0])
    return best[1], best[2]


def _get_grid_coords(adata, n_pos):
    # Prefer array coords and shift to zero-based grid for each slide.
    if "array_col" in adata.obs and "array_row" in adata.obs:
        array_col = adata.obs["array_col"].to_numpy().astype(int)
        array_row = adata.obs["array_row"].to_numpy().astype(int)
        if "spatial" in adata.obsm:
            gx, gy = _align_array_to_spatial(array_col, array_row, adata.obsm["spatial"])
            grid = np.stack([gx, gy], axis=1)
        else:
            grid = np.stack([array_col, array_row], axis=1)
        grid = grid - grid.min(axis=0, keepdims=True)
        max_pos = grid.max(axis=0)
        if n_pos is not None and (max_pos[0] >= n_pos or max_pos[1] >= n_pos):
            raise ValueError(
                f"n_pos={n_pos} is too small for grid coords "
                f"(max col={max_pos[0]}, max row={max_pos[1]})."
            )
        return grid

    # Fallback to min-max normalized spatial coords when array coords are missing.
    if n_pos is None:
        raise ValueError("n_pos must be provided when array coords are unavailable.")
    coords = adata.obsm["spatial"]
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    span_x = max_x - min_x + 1e-6
    span_y = max_y - min_y + 1e-6
    grid_x = np.floor((coords[:, 0] - min_x) / span_x * (n_pos - 1)).astype(int)
    grid_y = np.floor((coords[:, 1] - min_y) / span_y * (n_pos - 1)).astype(int)
    return np.stack([grid_x, grid_y], axis=1)



class HESTTHItoGeneDataset(Dataset):
    def __init__(self, slide_ids, hest_dir, spatial_dir, genes, patch_size=112, n_pos=64, train=True):
        super().__init__()
        self.slide_ids = slide_ids
        self.hest_dir = Path(hest_dir)
        self.spatial_dir = Path(spatial_dir)
        self.genes = list(genes)
        self.patch_size = patch_size
        self.r = patch_size // 2
        self.n_pos = n_pos
        self.train = train

        self.adatas = {}
        self.wsi_paths = {}
        self.wsis = {}

        print(f"Loading {len(slide_ids)} slides for THItoGene...")
        for sid in slide_ids:
            h5ad_path = self.spatial_dir / f'{sid}.h5ad'
            adata = _read_h5ad(h5ad_path)
            
            if self.genes:
                adata = adata[:, self.genes].copy()
            
            if 'counts' in adata.layers:
                adata.X = adata.layers['counts'].copy()
            elif scipy.sparse.issparse(adata.X):
                adata.X = adata.X.toarray()
            
            _normalize_total(adata, target_sum=1e4)
            _log1p(adata)
            
            wsi_path = list((self.hest_dir / 'wsis').glob(f"{sid}.*"))[0]
            self.wsi_paths[sid] = str(wsi_path)
            
            # Grid coords
            adata.obsm['grid_coords'] = _get_grid_coords(adata, self.n_pos)  # (N, 2)
            
            self.adatas[sid] = adata

    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        adata = self.adatas[sid]
        
        # Open WSI
        if sid not in self.wsis:
            wsi_path = self.wsi_paths[sid]
            try:
                if HAS_OPENSLIDE:
                    self.wsis[sid] = openslide.OpenSlide(wsi_path)
                else:
                    self.wsis[sid] = Image.open(wsi_path)
            except Exception as e:
                print(f"Error opening WSI for {sid}: {e}, using PIL fallback")
                self.wsis[sid] = Image.open(wsi_path)
        wsi = self.wsis[sid]
        
        # Prepare batch data (Whole Slide)
        N = adata.shape[0]
        
        # 1. Patches
        # Note: Loading all patches at once might be slow, but necessary for THItoGene structure if batch_size=1
        patches = torch.zeros((N, 3, self.patch_size, self.patch_size))
        
        spatial_coords = adata.obsm['spatial'].astype(int)
        
        for i in range(N):
            cx, cy = spatial_coords[i]
            x_tl = cx - self.r
            y_tl = cy - self.r
            
            try:
                if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
                    patch = wsi.read_region((int(x_tl), int(y_tl)), 0, (self.patch_size, self.patch_size)).convert('RGB')
                else:
                    patch = wsi.crop((x_tl, y_tl, x_tl + self.patch_size, y_tl + self.patch_size)).convert('RGB')
            except:
                patch = Image.new('RGB', (self.patch_size, self.patch_size))
            
            p_tensor = torch.from_numpy(np.array(patch)).float() / 255.0
            if p_tensor.shape[0] != self.patch_size or p_tensor.shape[1] != self.patch_size:
                 p_tensor = torch.zeros((self.patch_size, self.patch_size, 3))
            
            patches[i] = p_tensor.permute(2, 0, 1) # [C, H, W]
            
        # 2. Positions (Grid coords)
        grid_pos = torch.LongTensor(adata.obsm['grid_coords']) # (N, 2)
        
        # 3. Expressions
        exp_np = adata.X
        if hasattr(exp_np, 'toarray'):
            exp_np = exp_np.toarray()
        exp = torch.Tensor(exp_np) # (N, n_genes)

        # 4. Adjacency
        # Build adjacency on grid coords to match positional embedding space.
        adj = calcADJ(adata.obsm["grid_coords"], k=4)
        adj = torch.Tensor(adj)
        
        if self.train:
             return patches, grid_pos, exp, adj
        else:
             # Return center coords (pixel) as well for prediction saving
             centers = torch.Tensor(spatial_coords)
             return patches, grid_pos, exp, centers, adj


# ============================================================================
# 2026-09 extension: Path2Space
# ----------------------------------------------------------------------------
# Source-faithful Path2Space ST-prediction core on HEST.
#
# Kept from upstream (must NOT be replaced):
#   scripts/1.ST_prediction/1.1.Feature_extraction/func/ctrans_model/ctranspath.py
#       ::CTransPath            -> 768-d feature extractor, weight frozen
#   scripts/1.ST_prediction/1.2.Regression/model_MLP.py
#       ::MLP_regression_relu_two -> Linear(768,768)->ReLU->Dropout(0.2)
#                                    ->Linear(768,G)->ReLU
#   loss MSE, Adam lr=1e-4, output bias initialised to the training-label mean
#   (see 1main_regression.py; the training loop lives in the notebook)
#
# Adapted surface only (data interface):
#   HEST whole-slide reading, spot barcode alignment, gene-panel alignment to
#   data/spatial_data/common_genes.txt, fixed slide-level splits, prediction
#   export, spatial-map rendering.
#
# Structural note: upstream Path2Space treats ONE SPOT as ONE 224x224 TILE
# (`TILE_SIZE = 224` in 1main_feature_extraction.py, start = center-112). The
# upstream model's `torch.mean(x, dim=0)` is a no-op for a single tile. This
# adapter follows that literally; `tile_size` is exposed only so the tiling
# convention can be recorded per run.
# ============================================================================

P2S_TILE_SIZE = 224
P2S_IMAGENET_MEAN = (0.485, 0.456, 0.406)
P2S_IMAGENET_STD = (0.229, 0.224, 0.225)


def _p2s_feature_extraction_root(upstream_root):
    """Directory to put on sys.path so that `from func.ctrans_model import ...` works."""
    return Path(upstream_root) / "scripts" / "1.ST_prediction" / "1.1.Feature_extraction"


def _p2s_import_ctranspath(upstream_root):
    """Import upstream `func.ctrans_model.CTransPath` from the Path2Space source tree."""
    root = _p2s_feature_extraction_root(upstream_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Path2Space feature-extraction root not found: {root}. "
            "Check `upstream_root` (expects benchmark/Path2Space)."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from func.ctrans_model import CTransPath  # noqa: E402  (path inserted above)
    return CTransPath


def build_path2space_ctranspath(upstream_root, ctranspath_weights, device="cpu"):
    """Build upstream CTransPath (num_classes=0) and load the official weights.

    The weight file is the same `ctranspath.pth` used by the HiST benchmark
    (sha-checked separately); upstream Path2Space ships no weights of its own.
    """
    CTransPath = _p2s_import_ctranspath(upstream_root)
    model = CTransPath(num_classes=0)
    payload = torch.load(str(ctranspath_weights), map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    model.load_state_dict(payload, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _p2s_macenko_normalizer(use_macenko):
    """Return a callable `rgb_uint8 -> rgb_uint8`, or None when disabled.

    Upstream `1main_feature_extraction.py` applies Macenko normalization to every
    tile, so it is part of the official feature-extraction path. It is disabled
    by default here for two reasons, both recorded in
    `benchmark/Path2Space/README_MORPHOFM_ADAPTATION.md`:

    1. **Comparability.** None of the five existing benchmark baselines
       (HisToGene / iStar / mclSTExp / sCellST / HiST) apply stain
       normalization -- they read raw HEST tiles. Enabling it here would give
       Path2Space a preprocessing advantage the rest of the table does not have.
    2. **Throughput.** `spams.lasso` inside `get_concentrations` costs ~575 ms
       per 224x224 tile on this machine (measured 2026-09-23), i.e. ~0.5 min per
       slide without it versus ~34 min per slide with it.

    The reference HEST adaptation of this method used the same setting
    (`--use-macenko` is an opt-in flag defaulting to off).
    """
    if not use_macenko:
        return None
    try:
        root = None
        for candidate in sys.path:
            if (Path(candidate) / "func" / "utils_color_norm.py").exists():
                root = candidate
                break
        if root is None:
            raise ImportError("Path2Space func/ not on sys.path")
        from func.utils_color_norm import macenko_normalizer  # noqa: E402

        normalizer = macenko_normalizer()

        def _apply(rgb):
            # Degenerate (near-uniform) tiles can break the stain-matrix estimate.
            try:
                return normalizer.transform(rgb)
            except Exception:
                return rgb

        return _apply
    except Exception as exc:  # pragma: no cover - depends on optional spams
        print(f"[Path2Space] Macenko unavailable ({exc}); continuing without it.")
        return None


def _p2s_read_tile(wsi, cx, cy, crop_size, resize_to):
    """Crop a `crop_size` square centred on (cx, cy) at full resolution.

    `resize_to` (if set) rescales the crop, matching the resize-to-224 step the
    other baselines apply to their spot-diameter crops.
    """
    half = crop_size // 2
    left, top = int(round(cx)) - half, int(round(cy)) - half
    try:
        if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
            tile = wsi.read_region((left, top), 0, (crop_size, crop_size)).convert("RGB")
        else:
            tile = wsi.crop((left, top, left + crop_size, top + crop_size)).convert("RGB")
        if resize_to is not None and resize_to != crop_size:
            tile = tile.resize((resize_to, resize_to), Image.BICUBIC)
        arr = np.array(tile)
    except Exception:
        arr = None
    if arr is None or arr.shape[0] != resize_to or arr.shape[1] != resize_to:
        arr = np.zeros((resize_to, resize_to, 3), dtype=np.uint8)
    return arr


def extract_path2space_features(
    slide_ids,
    hest_dir,
    spatial_dir,
    upstream_root,
    ctranspath_weights,
    tile_size=P2S_TILE_SIZE,
    window=None,
    resize_to=P2S_TILE_SIZE,
    use_macenko=False,
    feature_cache_dir=None,
    batch_size=128,
    device="cpu",
    verbose=True,
):
    """One 768-d CTransPath feature per in-tissue spot, for each requested slide.

    Parameters
    ----------
    window : {"spot_diameter", "tile_px"}
        Spatial footprint of the crop taken around each spot centre.

        ``"spot_diameter"`` (default)
            Crop ``spot_diameter_fullres`` pixels (55 um on HEST, i.e. exactly one
            Visium spot) and rescale to ``resize_to``. This matches the spatial
            footprint every existing baseline in this benchmark uses (HiST crops
            the spot diameter and resizes to 224) and the patch-based reference
            adaptation of this method, so the comparison is like-for-like.

        ``"tile_px"``
            Crop ``tile_size`` pixels at full resolution with no rescale -- the
            literal reading of upstream `TILE_SIZE = 224` in
            `1main_feature_extraction.py`. Note that on HEST this covers about
            2.5 spot diameters, because upstream's own demo slides are at a
            different resolution.

    Returns
    -------
    dict
        ``{slide_id: {"spot_ids": np.ndarray[str], "features": np.ndarray (n,768)}}``
        Rows follow the h5ad file order restricted to ``in_tissue == 1`` -- the
        same ordering the other benchmark baselines use, so `true_bag` lines up.

    Caching
    -------
    When ``feature_cache_dir`` is given, each slide is cached as
    ``<sid>.npz`` holding ``spot_ids`` and ``features``; an existing cache is
    reused. Delete the file to force recomputation.
    """
    from PIL import Image  # local import: module level already imports PIL
    import torchvision  # local import: keep the module header untouched

    # `window` is deliberately required. It changes the spatial footprint of every
    # extracted feature, so a silent default here would let two call sites in the
    # same notebook use different crops and produce mutually inconsistent numbers.
    if window is None:
        raise ValueError(
            "window must be given explicitly: 'tile_px' (upstream literal 224 px crop, "
            "primary benchmark setting) or 'spot_diameter' (one Visium spot diameter "
            "rescaled to resize_to, the footprint the other baselines use)."
        )

    if feature_cache_dir is not None:
        feature_cache_dir = Path(feature_cache_dir)
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    model = build_path2space_ctranspath(upstream_root, ctranspath_weights, device=device)
    normalizer = _p2s_macenko_normalizer(use_macenko)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=P2S_IMAGENET_MEAN, std=P2S_IMAGENET_STD),
        ]
    )

    hest_dir = Path(hest_dir)
    spatial_dir = Path(spatial_dir)
    bundle = {}

    for sid in slide_ids:
        cache_path = None
        if feature_cache_dir is not None:
            cache_path = feature_cache_dir / f"{sid}__{window}__{tile_size if window == 'tile_px' else resize_to}.npz"
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            bundle[sid] = {
                "spot_ids": cached["spot_ids"].astype(str),
                "features": cached["features"].astype(np.float32),
            }
            if verbose:
                print(f"[Path2Space] {sid}: cache hit {cache_path.name} {bundle[sid]['features'].shape}")
            continue

        adata = _read_h5ad(spatial_dir / f"{sid}.h5ad")
        in_tissue = adata.obs["in_tissue"].to_numpy().astype(bool)
        spot_ids = adata.obs_names.to_numpy()[in_tissue]
        coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()[in_tissue]

        wsi_files = list((hest_dir / "wsis").glob(f"{sid}.*"))
        if not wsi_files:
            raise FileNotFoundError(f"WSI not found for {sid} in {hest_dir / 'wsis'}")
        if HAS_OPENSLIDE:
            wsi = openslide.OpenSlide(str(wsi_files[0]))
        else:
            wsi = Image.open(wsi_files[0])

        feats = np.zeros((len(spot_ids), 768), dtype=np.float32)

        if window == "spot_diameter":
            scalefactors = {}
            if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict) and adata.uns["spatial"]:
                entry = next(iter(adata.uns["spatial"].values()))
                scalefactors = entry.get("scalefactors", {}) or {}
            spot_diameter = scalefactors.get("spot_diameter_fullres")
            if spot_diameter is None:
                raise KeyError(
                    f"{sid}: spot_diameter_fullres missing; use window='tile_px' for this slide"
                )
            crop_size = int(round(float(spot_diameter)))
        elif window == "tile_px":
            crop_size = int(tile_size)
        else:
            raise ValueError(f"unknown window mode: {window!r}")

        if verbose:
            print(f"[Path2Space] {sid}: window={window} crop={crop_size}px -> {resize_to}px")

        try:
            with torch.no_grad():
                for start in range(0, len(spot_ids), batch_size):
                    stop = min(start + batch_size, len(spot_ids))
                    batch = []
                    for i in range(start, stop):
                        arr = _p2s_read_tile(wsi, coords[i, 0], coords[i, 1], crop_size, resize_to)
                        if normalizer is not None:
                            arr = normalizer(arr)
                        batch.append(transform(Image.fromarray(arr)))
                    x = torch.stack(batch).to(device)
                    feats[start:stop] = model(x).detach().cpu().numpy().astype(np.float32)
                    if verbose:
                        print(f"[Path2Space] {sid}: {stop}/{len(spot_ids)} spots", end="\r")
        finally:
            try:
                wsi.close()
            except Exception:
                pass
        if verbose:
            print()

        bundle[sid] = {"spot_ids": spot_ids, "features": feats}
        if cache_path is not None:
            np.savez_compressed(cache_path, spot_ids=spot_ids, features=feats)

    return bundle


def path2space_target_matrix(adata, genes):
    """Target matrix aligned to `genes`, using the benchmark's log1p(CPM-1e4) target.

    Identical preprocessing to `HESTHisToGeneDataset`, so Path2Space and the other
    baselines share one `true_bag`.
    """
    ad = adata[:, list(genes)].copy()
    if "counts" in ad.layers:
        ad.X = ad.layers["counts"].copy()
    elif scipy.sparse.issparse(ad.X):
        ad.X = ad.X.toarray()
    _normalize_total(ad, target_sum=1e4)
    _log1p(ad)
    x = ad.X.toarray() if scipy.sparse.issparse(ad.X) else np.asarray(ad.X)
    return x.astype(np.float32)


class HESTPath2SpaceDataset(Dataset):
    """Path2Space training/eval set: one spot -> (feature, target).

    Mirrors upstream `SlideRNADataset`: a sample is
    ``(features[i].unsqueeze(0), targets[i])`` -- a 1x768 tensor and a 1-D
    expression vector, which is what `MLP_regression_relu_two` expects.

    Parameters
    ----------
    feature_bundle
        Output of :func:`extract_path2space_features`, restricted to the slides
        in this split.
    targets
        ``{slide_id: np.ndarray (n_spots, n_genes)}``; rows must match
        ``feature_bundle[slide_id]["features"]``.
    """

    def __init__(self, feature_bundle, targets, genes):
        super().__init__()
        self.genes = list(genes)
        self.samples = []  # (slide_id, row_index)
        self.features = {}
        self.targets = {}
        self.spot_ids_by_slide = {}
        for sid, payload in feature_bundle.items():
            feats = np.asarray(payload["features"], dtype=np.float32)
            tgt = np.asarray(targets[sid], dtype=np.float32)
            if feats.shape[0] != tgt.shape[0]:
                raise ValueError(
                    f"{sid}: feature rows ({feats.shape[0]}) != target rows ({tgt.shape[0]})"
                )
            if tgt.shape[1] != len(self.genes):
                raise ValueError(
                    f"{sid}: target has {tgt.shape[1]} genes, expected {len(self.genes)}"
                )
            self.features[sid] = torch.from_numpy(feats)
            self.targets[sid] = torch.from_numpy(tgt)
            self.spot_ids_by_slide[sid] = np.asarray(payload["spot_ids"]).astype(str)
            for i in range(feats.shape[0]):
                self.samples.append((sid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        sample = self.features[sid][i].unsqueeze(0)  # [1, 768]
        target = self.targets[sid][i]
        return sample, target

    def spot_ids(self):
        """Spot barcodes in row order (parallel to the order `__getitem__` returns)."""
        ids = []
        for sid in self.spot_ids_by_slide:
            ids.extend(list(self.spot_ids_by_slide[sid]))
        return np.asarray(ids)



# ============================================================================
# 2026-09 extension: ST-Net
# ----------------------------------------------------------------------------
# Source-faithful ST-Net ST-prediction core on HEST.
#
# Kept from upstream (must NOT be replaced):
#   torchvision.models.densenet121(pretrained=True)
#       -> ImageNet-1k DenseNet-121 encoder (densenet121-a639ec97.pth)
#   stnet/utils/nn.py::set_out_features
#       -> classifier replaced by Linear(1024, G) with zeroed weight and bias
#   stnet/transforms/eight_symmetry.py::EightSymmetry
#       -> the `--average` test-time augmentation over the 8 rotations/reflections
#   MSE gene loss `sum((pred - gene)**2) / n_genes`, SGD lr=1e-6 momentum=0.9,
#   batch 32, 50 epochs, 224 px window at downsample 1
#   (see stnet/cmd/run_spatial.py and the README "Training models" main command)
#
# Adapted surface only (data interface):
#   HEST whole-slide reading, spot-window caching, barcode alignment, gene-panel
#   alignment to data/spatial_data/common_genes.txt, fixed train/validation
#   splits, prediction export, spatial-map rendering.
#
# Import note: `import stnet` cannot be used. `stnet/__init__.py` builds the CLI
# parser at import time and `stnet/utils/ensembl.py` unconditionally loads
# `ensembl.pkl` / `ensembl.tsv`, which upstream gitignores and which are absent
# from the distributed source. Neither is on the ST-prediction path: this
# benchmark supplies approved gene symbols from `common_genes.txt` and never maps
# Ensembl IDs. The two upstream modules the training path actually uses are
# therefore loaded by file path, so the executed code stays upstream's own.
# ============================================================================

STNET_WINDOW = 224
STNET_IMAGENET_MEAN = (0.485, 0.456, 0.406)
STNET_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_stnet_module(stnet_root, submodule, alias):
    """Load one upstream ST-Net module by file path, bypassing `stnet/__init__.py`.

    See the import note above for why the package cannot be imported normally.
    `alias` must be unique per module so repeated loads stay independent.
    """
    import importlib.util
    import types

    stnet_root = Path(stnet_root)
    if not (stnet_root / "stnet").is_dir():
        raise FileNotFoundError(
            f"ST-Net source not found under {stnet_root}. Expects benchmark/ST-Net."
        )

    # `stnet/utils/nn.py` does `import stnet`; give it a namespace package to
    # bind to so that import succeeds without running the real __init__.
    if "stnet" not in sys.modules:
        stub = types.ModuleType("stnet")
        stub.__path__ = [str(stnet_root / "stnet")]
        sys.modules["stnet"] = stub

    module_path = stnet_root / "stnet" / Path(*submodule.split(".")).with_suffix(".py")
    if not module_path.is_file():
        raise FileNotFoundError(f"Upstream ST-Net module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stnet_nn(stnet_root):
    """Upstream `stnet.utils.nn` (`set_out_features`, `get_finetune_parameters`)."""
    return load_stnet_module(stnet_root, "utils.nn", "stnet_upstream_nn")


def load_stnet_eight_symmetry(stnet_root):
    """Upstream `stnet.transforms.eight_symmetry` (`EightSymmetry`)."""
    return load_stnet_module(
        stnet_root, "transforms.eight_symmetry", "stnet_upstream_eight_symmetry"
    ).EightSymmetry


def build_stnet_model(stnet_root, n_genes, pretrained=True, device="cpu"):
    """Build upstream ST-Net: torchvision DenseNet-121 + `set_out_features`.

    The ImageNet initialization is torchvision's own `densenet121-a639ec97.pth`
    (SHA256 a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395),
    which is what upstream `--pretrain` loads.
    """
    import torchvision  # local import: keep the module header untouched

    nn_module = load_stnet_nn(stnet_root)
    if pretrained:
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = torchvision.models.densenet121(weights=weights)
    nn_module.set_out_features(model, n_genes)  # zeroes the new weight and bias
    return model.to(device)


def _stnet_read_window(wsi, cx, cy, crop_size, resize_to):
    """Crop a `crop_size` square centred on (cx, cy) at full resolution."""
    half = crop_size // 2
    left, top = int(round(cx)) - half, int(round(cy)) - half
    try:
        if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
            tile = wsi.read_region((left, top), 0, (crop_size, crop_size)).convert("RGB")
        else:
            tile = wsi.crop((left, top, left + crop_size, top + crop_size)).convert("RGB")
        if resize_to is not None and resize_to != crop_size:
            tile = tile.resize((resize_to, resize_to), Image.BICUBIC)
        arr = np.array(tile)
    except Exception:
        arr = None
    if arr is None or arr.shape[0] != resize_to or arr.shape[1] != resize_to:
        arr = np.zeros((resize_to, resize_to, 3), dtype=np.uint8)
    return arr


def build_stnet_window_cache(
    slide_ids,
    hest_dir,
    spatial_dir,
    window=STNET_WINDOW,
    crop_mode="tile_px",
    cache_dir=None,
    verbose=True,
):
    """Cache one RGB window per in-tissue spot, for each requested slide.

    Parameters
    ----------
    window : int
        Side length of the window handed to DenseNet-121 (upstream `--window`,
        default 224).
    crop_mode : {"tile_px", "spot_diameter"}
        ``"tile_px"`` (primary)
            Crop ``window`` pixels at full resolution with no rescale. Upstream
            `stnet/datasets/spatial.py` reads
            ``(pixel - window*downsample//2, window*downsample, window*downsample)``
            at ``--downsample 1``, i.e. a `window`-pixel full-resolution crop.
            On HEST the Visium spot is ~89 px across, so this covers ~2.5 spot
            diameters. Same convention the Path2Space row uses, for the same
            reason: run each method as published.
        ``"spot_diameter"`` (sensitivity)
            Crop one Visium spot diameter and rescale to ``window``, matching the
            spatial footprint the other baselines use.

    Returns
    -------
    dict
        ``{slide_id: {"spot_ids": np.ndarray[str], "cache_path": Path,
        "shape": (n, window, window, 3)}}``

        Rows follow the h5ad file order restricted to ``in_tissue == 1`` -- the
        ordering every baseline in this benchmark uses, so `true_bag` lines up.
        The cache holds uint8 pixels; `HESTSTNetDataset` memory-maps it.

    Caching
    -------
    ``<cache_dir>/<sid>__<crop_mode><crop_size>_<window>.npy`` holds the pixels
    and ``<sid>__spot_ids.npy`` the barcodes. Existing files are reused; delete
    them to force recomputation.
    """
    hest_dir = Path(hest_dir)
    spatial_dir = Path(spatial_dir)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    bundle = {}
    for sid in slide_ids:
        adata = _read_h5ad(spatial_dir / f"{sid}.h5ad")
        in_tissue = adata.obs["in_tissue"].to_numpy().astype(bool)
        spot_ids = adata.obs_names.to_numpy()[in_tissue]
        coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()[in_tissue]

        if crop_mode == "spot_diameter":
            scalefactors = {}
            if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict) and adata.uns["spatial"]:
                entry = next(iter(adata.uns["spatial"].values()))
                scalefactors = entry.get("scalefactors", {}) or {}
            spot_diameter = scalefactors.get("spot_diameter_fullres")
            if spot_diameter is None:
                raise KeyError(
                    f"{sid}: spot_diameter_fullres missing; use crop_mode='tile_px' for this slide"
                )
            crop_size = int(round(float(spot_diameter)))
        elif crop_mode == "tile_px":
            crop_size = int(window)
        else:
            raise ValueError(f"unknown crop_mode: {crop_mode!r}")

        cache_path = None
        spot_ids_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"{sid}__{crop_mode}{crop_size}_{int(window)}.npy"
            spot_ids_path = cache_dir / f"{sid}__spot_ids.npy"

        if cache_path is not None and cache_path.exists() and spot_ids_path.exists():
            cached_ids = np.load(spot_ids_path, allow_pickle=True).astype(str)
            shape = np.load(cache_path, mmap_mode="r").shape
            if len(cached_ids) != shape[0]:
                raise ValueError(
                    f"{sid}: window cache row count ({shape[0]}) != cached barcodes "
                    f"({len(cached_ids)}). Delete {cache_path} and re-run."
                )
            bundle[sid] = {"spot_ids": cached_ids, "cache_path": cache_path, "shape": shape}
            if verbose:
                print(f"[ST-Net] {sid}: cache hit {cache_path.name} {shape}")
            continue

        wsi_files = list((hest_dir / "wsis").glob(f"{sid}.*"))
        if not wsi_files:
            raise FileNotFoundError(f"WSI not found for {sid} in {hest_dir / 'wsis'}")
        if HAS_OPENSLIDE:
            wsi = openslide.OpenSlide(str(wsi_files[0]))
        else:
            wsi = Image.open(wsi_files[0])

        if verbose:
            print(
                f"[ST-Net] {sid}: crop_mode={crop_mode} crop={crop_size}px -> {window}px "
                f"for {len(spot_ids)} spots"
            )

        out = np.zeros((len(spot_ids), int(window), int(window), 3), dtype=np.uint8)
        try:
            for i in range(len(spot_ids)):
                out[i] = _stnet_read_window(wsi, coords[i, 0], coords[i, 1], crop_size, int(window))
                if verbose and (i + 1) % 500 == 0:
                    print(f"[ST-Net] {sid}: {i + 1}/{len(spot_ids)} spots", end="\r")
        finally:
            try:
                wsi.close()
            except Exception:
                pass
        if verbose:
            print()

        if cache_path is not None:
            np.save(cache_path, out)
            np.save(spot_ids_path, spot_ids)
        bundle[sid] = {
            "spot_ids": spot_ids,
            "cache_path": cache_path,
            "shape": out.shape,
            "images": out if cache_path is None else None,
        }

    return bundle


def stnet_target_matrix(adata, genes):
    """Target matrix aligned to `genes`, using the benchmark's log1p(CPM-1e4) target.

    Delegates to `path2space_target_matrix` so that every method in this benchmark
    scores against one identical `true_bag`. This is the documented ST-Net target
    adaptation: upstream `--norm --gene_transform log` would give
    `log((1+y)/(n_genes + total_counts))`, which is not comparable to the other
    rows and is explicitly listed as an allowed substitution in the upstream
    `source_provenance.json` ("coverage95 log1p_rate target construction").
    """
    return path2space_target_matrix(adata, genes)


class HESTSTNetDataset(Dataset):
    """ST-Net training/eval set: one spot -> (image window, target).

    Mirrors upstream `stnet.datasets.Spatial`: a sample is
    ``(X, y)`` where ``X`` is a PIL image that the DataLoader-attached transform
    turns into a tensor, exactly as `run_spatial.py` wires it (augmentation +
    ``ToTensor`` + ``Normalize`` for training, ``EightSymmetry`` +
    ``ToTensor`` + ``Normalize`` for the ``--average`` test pass).

    Windows are memory-mapped from the cache produced by
    :func:`build_stnet_window_cache`, so DataLoader workers share one copy
    instead of each holding the whole slide.

    Parameters
    ----------
    window_bundle
        Output of :func:`build_stnet_window_cache`, restricted to this split.
    targets
        ``{slide_id: np.ndarray (n_spots, n_genes)}``; rows must match
        ``window_bundle[slide_id]``.
    transform
        torchvision transform applied to the PIL window. Assigned as a plain
        attribute by the caller for the train pass, matching upstream's
        ``train_dataset.transform = transform``.
    """

    def __init__(self, window_bundle, targets, genes, transform=None):
        super().__init__()
        self.genes = list(genes)
        self.transform = transform
        self.window_bundle = dict(window_bundle)
        self.samples = []  # (slide_id, row_index)
        self._images_cache = {}
        self.targets = {}
        self.spot_ids_by_slide = {}
        for sid, payload in window_bundle.items():
            n_rows = int(payload["shape"][0])
            tgt = np.asarray(targets[sid], dtype=np.float32)
            if tgt.shape[0] != n_rows:
                raise ValueError(
                    f"{sid}: window rows ({n_rows}) != target rows ({tgt.shape[0]})"
                )
            if tgt.shape[1] != len(self.genes):
                raise ValueError(
                    f"{sid}: target has {tgt.shape[1]} genes, expected {len(self.genes)}"
                )
            self.targets[sid] = torch.from_numpy(tgt)
            self.spot_ids_by_slide[sid] = np.asarray(payload["spot_ids"]).astype(str)
            for i in range(n_rows):
                self.samples.append((sid, i))

    def _images(self, sid):
        """mmap (or in-memory array) holding this slide's cached windows."""
        if sid not in self._images_cache:
            payload = self.window_bundle[sid]
            if payload.get("cache_path") is not None:
                # Memory-mapped so DataLoader workers share one copy of the slide.
                self._images_cache[sid] = np.load(str(payload["cache_path"]), mmap_mode="r")
            else:
                self._images_cache[sid] = payload["images"]
        return self._images_cache[sid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        arr = self._images(sid)[i]
        img = Image.fromarray(np.asarray(arr))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[sid][i]

    def spot_ids(self):
        """Spot barcodes in row order (parallel to the order `__getitem__` returns)."""
        ids = []
        for sid in self.spot_ids_by_slide:
            ids.extend(list(self.spot_ids_by_slide[sid]))
        return np.asarray(ids)


# ============================================================================
# STimage  (https://github.com/BiomedicalMachineLearning/STimage)
# ============================================================================
#
# Upstream `stimage/_model.py::CNN_NB_multiple_genes` builds
#
#     cnn -> GlobalAveragePooling2D -> [Dense(2) -> Lambda(negative_binomial_layer)] x n_genes
#
# i.e. one Keras output head per gene. Upstream only ever runs this with a handful
# of genes (the 8-gene and 30-gene lists hard-coded in `stimage/02_Training.py`).
# This repository's panel is `data/spatial_data/common_genes.txt` (17,511 genes),
# and at that width the literal multi-output Keras route does not complete:
#
#   * measured here (scripts/probe_stimage_model_scale.py, A800, TF 2.15):
#       n_genes=8       build 9.9 s,   192 layers,  forward(2) 44 s,  747 MiB
#       n_genes=1024    build 19.0 s, 2224 layers,  forward(2) 35 s,  764 MiB
#       n_genes=17511   build 191.3 s, 35198 layers, forward(2) 229 s, 1025 MiB
#     Cost scales with output-head count, not with GPU work: a 2-sample forward
#     pass spends ~229 s in Keras bookkeeping against ~1 GB of GPU memory.
#
#   * the reference project that first adapted STimage to HEST hit the same wall
#     and recorded it: their stress job 781617 ran 4 h 35 min at 16,942 genes on
#     one epoch with no output, ~29 GB RSS and 1.4 GB GPU, and `py-spy` placed it
#     in `keras/src/optimizers/optimizer.py::apply_grad_to_update_var` during
#     train-function tracing of the per-gene output-head variables. They cancelled
#     it and shipped the "source-equivalent feature-cached" row instead, with the
#     explicit instruction that it be reported as `STimage (feature-cached)` and
#     *not* as a literal official `model.fit` run.
#
# So this row takes the same documented route. It is an exact reformulation, not an
# approximation, because upstream's CNN trunk is frozen (`ft=False`) and the head is
# a linear map per gene:
#
#   * `build_stimage_resnet50_gap` is upstream's trunk verbatim -- Keras ResNet50
#     ImageNet no-top + GlobalAveragePooling2D -> 2048-d -- run once per spot and
#     cached, so the frozen trunk is never re-executed inside the training loop.
#   * `build_stimage_vectorized_head` holds kernel (2048, n_genes, 2) and bias
#     (n_genes, 2). For gene g this is exactly one `Dense(2)`, so n_genes separate
#     heads are one einsum. Parameterization reuses upstream's own
#     `negative_binomial_layer` unchanged, since that layer operates on the last
#     axis and a `(batch, n_genes, 2)` tensor satisfies it.
#   * `stimage_nb_loss` is upstream `negative_binomial_loss` restated for that
#     layout. Upstream's version cannot be reused verbatim: it expands `n` to
#     `(batch, 1, 1)` against a per-head scalar `y_true` of `(batch, 1)`, whereas
#     the vectorized `y_true` is `(batch, n_genes)` and would broadcast to
#     `(batch, n_genes, n_genes)`. `verify_stimage_vectorization` below checks the
#     restatement against upstream's function numerically rather than asserting it.
#
# Everything else follows this repository's benchmark surface: HEST whole-slide
# reading, the `in_tissue == 1` row order every other row uses, and the shared
# `log1p(CPM-1e4)` target (see `stimage_target_matrix`).

STIMAGE_TILE_SIZE = 299
STIMAGE_FEATURE_DIM = 2048

# Set by `bind_stimage_upstream` before a head is built; keeps the upstream import
# explicit at the call site instead of an implicit module-level import of TensorFlow.
_STIMAGE_MODEL_HANDLE = [None]


def load_stimage_upstream(upstream_root):
    """Import upstream `stimage._model` by path.

    `stimage/__init__.py` is empty, so a plain `sys.path` entry is enough and the
    executed code stays upstream's own.
    """
    upstream_root = Path(upstream_root)
    if not (upstream_root / "stimage" / "_model.py").is_file():
        raise FileNotFoundError(
            f"STimage source not found under {upstream_root}. Expects benchmark/STimage."
        )
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    import stimage._model as stimage_model  # noqa: PLC0415 - deliberately lazy

    return stimage_model


def bind_stimage_upstream(upstream_root):
    """Import upstream `stimage._model` and make it available to the head factory."""
    stimage_model = load_stimage_upstream(upstream_root)
    _STIMAGE_MODEL_HANDLE[0] = stimage_model
    return stimage_model


def build_stimage_resnet50_gap(tile_size=STIMAGE_TILE_SIZE):
    """Upstream STimage's frozen CNN trunk: ResNet50 ImageNet no-top + GAP.

    Mirrors the first two stages of `CNN_NB_multiple_genes` for the default
    ``cnnbase="resnet50"`` with ``ft=False``. Keras applies its own ResNet50
    preprocessing inside the graph, so callers pass raw 0-255 RGB.
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(int(tile_size), int(tile_size), 3), name="tile_input")
    base = tf.keras.applications.ResNet50(
        input_tensor=inputs, weights="imagenet", include_top=False
    )
    base.trainable = False  # upstream `ft=False` freezes every trunk layer
    outputs = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _stimage_read_window(wsi, cx, cy, crop_size, resize_to):
    """Crop a `crop_size` square centred on (cx, cy) at full resolution."""
    half = crop_size // 2
    left, top = int(round(cx)) - half, int(round(cy)) - half
    try:
        if HAS_OPENSLIDE and isinstance(wsi, openslide.OpenSlide):
            tile = wsi.read_region((left, top), 0, (crop_size, crop_size)).convert("RGB")
        else:
            tile = wsi.crop((left, top, left + crop_size, top + crop_size)).convert("RGB")
        if resize_to is not None and resize_to != crop_size:
            tile = tile.resize((resize_to, resize_to), Image.BICUBIC)
        arr = np.array(tile)
    except Exception:
        arr = None
    if arr is None or arr.shape[0] != resize_to or arr.shape[1] != resize_to:
        arr = np.zeros((resize_to, resize_to, 3), dtype=np.uint8)
    return arr


def extract_stimage_features(
    slide_ids,
    hest_dir,
    spatial_dir,
    window=STIMAGE_TILE_SIZE,
    crop_mode="tile_px",
    resize_to=STIMAGE_TILE_SIZE,
    feature_cache_dir=None,
    batch_size=256,
    verbose=True,
):
    """Cache one 2048-d ResNet50-GAP trunk feature per in-tissue spot, per slide.

    This is the feature-cached route described in the section header above: it runs
    upstream's frozen trunk once per spot instead of once per spot per epoch.

    Parameters
    ----------
    window : int
        Side of the full-resolution crop taken around each spot.
    crop_mode : {"tile_px", "spot_diameter"}
        ``"tile_px"`` (primary)
            Crop ``window`` pixels at full resolution, no rescale. Same convention
            and same reasoning as `build_stnet_window_cache`: run each method as
            published. Upstream's default input tensor is 299x299, and 299 px
            covers ~3.3 Visium spot diameters on HEST.
        ``"spot_diameter"`` (sensitivity)
            Crop one Visium spot diameter and rescale to ``resize_to``, matching the
            spatial footprint the other baselines use.
    resize_to : int
        Side of the tensor handed to ResNet50. Upstream `tiling(target_size=299)`.
        Note upstream's own `tiling(crop_size=40, target_size=299)` crops a much
        smaller region than either mode here and upsamples; HEST's DPI differs from
        theirs, so no direct pixel correspondence exists and the choice is recorded
        rather than inferred.

    Returns
    -------
    dict
        ``{slide_id: {"spot_ids", "feature_path", "shape" (n, 2048)}}`` in h5ad
        row order restricted to ``in_tissue == 1``.

    Caching
    -------
    ``<feature_cache_dir>/<sid>__<crop_mode><crop_size>_<resize_to>.npy`` holds the
    float32 features, ``<sid>__spot_ids.npy`` the barcodes. Existing files are
    reused; delete them to force recomputation.
    """
    hest_dir = Path(hest_dir)
    spatial_dir = Path(spatial_dir)
    if feature_cache_dir is not None:
        feature_cache_dir = Path(feature_cache_dir)
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = None
    bundle = {}
    for sid in slide_ids:
        adata = _read_h5ad(spatial_dir / f"{sid}.h5ad")
        in_tissue = adata.obs["in_tissue"].to_numpy().astype(bool)
        spot_ids = adata.obs_names.to_numpy()[in_tissue]
        coords = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()[in_tissue]

        if crop_mode == "spot_diameter":
            scalefactors = {}
            if "spatial" in adata.uns and isinstance(adata.uns["spatial"], dict) and adata.uns["spatial"]:
                entry = next(iter(adata.uns["spatial"].values()))
                scalefactors = entry.get("scalefactors", {}) or {}
            spot_diameter = scalefactors.get("spot_diameter_fullres")
            if spot_diameter is None:
                raise KeyError(
                    f"{sid}: spot_diameter_fullres missing; use crop_mode='tile_px' for this slide"
                )
            crop_size = int(round(float(spot_diameter)))
        elif crop_mode == "tile_px":
            crop_size = int(window)
        else:
            raise ValueError(f"unknown crop_mode: {crop_mode!r}")

        feature_path = None
        spot_ids_path = None
        if feature_cache_dir is not None:
            feature_path = feature_cache_dir / f"{sid}__{crop_mode}{crop_size}_{int(resize_to)}.npy"
            spot_ids_path = feature_cache_dir / f"{sid}__spot_ids.npy"

        if feature_path is not None and feature_path.exists() and spot_ids_path.exists():
            cached_ids = np.load(spot_ids_path, allow_pickle=True).astype(str)
            shape = np.load(feature_path, mmap_mode="r").shape
            if len(cached_ids) != shape[0]:
                raise ValueError(
                    f"{sid}: feature cache rows ({shape[0]}) != cached barcodes "
                    f"({len(cached_ids)}). Delete {feature_path} and re-run."
                )
            bundle[sid] = {"spot_ids": cached_ids, "feature_path": feature_path, "shape": shape}
            if verbose:
                print(f"[STimage] {sid}: cache hit {feature_path.name} {shape}")
            continue

        wsi_files = list((hest_dir / "wsis").glob(f"{sid}.*"))
        if not wsi_files:
            raise FileNotFoundError(f"WSI not found for {sid} in {hest_dir / 'wsis'}")
        if extractor is None:
            extractor = build_stimage_resnet50_gap(tile_size=int(resize_to))
        if HAS_OPENSLIDE:
            wsi = openslide.OpenSlide(str(wsi_files[0]))
        else:
            wsi = Image.open(wsi_files[0])

        if verbose:
            print(
                f"[STimage] {sid}: crop_mode={crop_mode} crop={crop_size}px -> "
                f"{int(resize_to)}px, features {STIMAGE_FEATURE_DIM}-d for {len(spot_ids)} spots"
            )

        features = np.zeros((len(spot_ids), STIMAGE_FEATURE_DIM), dtype=np.float32)
        try:
            for start in range(0, len(spot_ids), int(batch_size)):
                stop = min(start + int(batch_size), len(spot_ids))
                batch = np.zeros((stop - start, int(resize_to), int(resize_to), 3), dtype=np.float32)
                for j in range(start, stop):
                    batch[j - start] = _stimage_read_window(
                        wsi, coords[j, 0], coords[j, 1], crop_size, int(resize_to)
                    ).astype(np.float32)
                features[start:stop] = extractor.predict_on_batch(batch)
                if verbose:
                    print(f"[STimage] {sid}: {stop}/{len(spot_ids)} spots", end="\r")
        finally:
            try:
                wsi.close()
            except Exception:
                pass
        if verbose:
            print()

        if feature_path is not None:
            np.save(feature_path, features)
            np.save(spot_ids_path, spot_ids)
        bundle[sid] = {
            "spot_ids": spot_ids,
            "feature_path": feature_path,
            "shape": features.shape,
            "features": features if feature_path is None else None,
        }

    return bundle


def stimage_target_matrix(adata, genes):
    """Target matrix aligned to `genes`, using the benchmark's log1p(CPM-1e4) target.

    Delegates to `path2space_target_matrix` so this row scores against the same
    `true_bag` as every other row. Upstream STimage fits its NB likelihood to raw
    counts (`_data_generator.py::_load_label` returns `adata.to_df()[gene]`), and
    the reference HEST adaptation used `log1p(counts / size_factor)` with mean-one
    size factors. Neither is comparable to the other rows here, so the target is
    substituted and the substitution is recorded instead of hidden -- the same
    treatment `stnet_target_matrix` documents.
    """
    return path2space_target_matrix(adata, genes)


def build_stimage_vectorized_head(feature_dim, n_genes, seed=42):
    """Vectorized form of upstream's n_genes separate `Dense(2)` NB heads.

    Holds one kernel of shape ``(feature_dim, n_genes, 2)`` and one bias of
    ``(n_genes, 2)``. Slice ``[:, g, :]`` is exactly upstream's ``Dense(2)`` for
    gene ``g`` applied to the GAP features, so this is the same function class with
    the per-gene layers stacked into one einsum. Kernel init uses the Glorot-uniform
    limit ``sqrt(6 / (feature_dim + 2))``, matching Keras' default `Dense` initializer.
    """
    import tensorflow as tf

    stimage_model = _STIMAGE_MODEL_HANDLE[0]
    if stimage_model is None:
        raise RuntimeError(
            "Call bind_stimage_upstream(upstream_root) before building a vectorized head."
        )

    class _VectorizedSTimageNBHead(tf.Module):
        def __init__(self):
            super().__init__()
            limit = math.sqrt(6.0 / float(int(feature_dim) + 2))
            rng = tf.random.Generator.from_seed(int(seed))
            self.kernel = tf.Variable(
                rng.uniform(
                    shape=(int(feature_dim), int(n_genes), 2),
                    minval=-limit,
                    maxval=limit,
                    dtype=tf.float32,
                ),
                name="kernel",
            )
            self.bias = tf.Variable(
                tf.zeros((int(n_genes), 2), dtype=tf.float32), name="bias"
            )

        @property
        def trainable_variables(self):
            return [self.kernel, self.bias]

        def logits(self, features):
            return tf.einsum("bf,fgp->bgp", features, self.kernel) + self.bias

        def __call__(self, features):
            # Upstream's own parameterization: unstack -> softplus(n) / sigmoid(p)
            # -> concat on the last axis. Valid here because it acts on axis -1.
            return stimage_model.negative_binomial_layer(self.logits(features))

        def nb_mean(self, features):
            """Predicted NB mean per gene: n * (1 - p) / p, as upstream predicts."""
            logits = self.logits(features)
            n = tf.nn.softplus(logits[..., 0])
            p = tf.clip_by_value(tf.nn.sigmoid(logits[..., 1]), 1.0e-6, 1.0 - 1.0e-6)
            return n * (1.0 - p) / p

    return _VectorizedSTimageNBHead()


def stimage_nb_loss(y_true, y_pred, weights=None):
    """Upstream `negative_binomial_loss` restated for the ``(batch, n_genes, 2)`` layout.

    Term for term identical to upstream's function; see the section header for why
    upstream's own version cannot be reused at this layout. `verify_stimage_vectorization`
    checks the two agree numerically.

    Declared deviation: `p` is clipped to ``[1e-6, 1 - 1e-6]`` before the logarithms.
    Upstream does not clip, and since `negative_binomial_layer` parameterizes `p` with a
    sigmoid that saturates in float32, upstream's own loss can evaluate ``log(0)``. The
    reference HEST adaptation of STimage clips the same way. The clip is applied here
    rather than inside the head so that the head itself stays upstream's exact
    parameterization.

    Parameters
    ----------
    y_true : tf.Tensor (batch, n_genes)
    y_pred : tf.Tensor (batch, n_genes, 2) -- n and p from `negative_binomial_layer`
    weights : tf.Tensor (batch, n_genes) or None
        Per-gene mask. Reduces genes the way Keras reduces a list of per-gene
        outputs (mean over spots) and then sums over genes, matching upstream's
        `model.compile(loss=negative_binomial_loss)` with one output per gene.
        ``None`` weights every gene equally.
    """
    import tensorflow as tf

    n = y_pred[..., 0]
    p = tf.clip_by_value(y_pred[..., 1], 1.0e-6, 1.0 - 1.0e-6)
    nll = (
        tf.math.lgamma(n)
        + tf.math.lgamma(y_true + 1.0)
        - tf.math.lgamma(n + y_true)
        - n * tf.math.log(p)
        - y_true * tf.math.log(1.0 - p)
    )
    if weights is None:
        return tf.reduce_sum(tf.reduce_mean(nll, axis=0))
    w = tf.cast(weights, nll.dtype)
    numerator = tf.reduce_sum(nll * w, axis=0)
    denominator = tf.reduce_sum(w, axis=0)
    per_gene = tf.where(denominator > 0.0, numerator / tf.maximum(denominator, 1.0e-6), 0.0)
    return tf.reduce_sum(per_gene)


def verify_stimage_vectorization(upstream_root, feature_dim=2048, n_genes=64, seed=0, rel_atol=1e-5):
    """Check the vectorized head and loss against upstream's own code.

    Two independent checks, so a failure points at exactly one of them:

    1. Head. Upstream's per-gene `negative_binomial_layer` applied to `n_genes`
       separate `Dense(2)` outputs, against `build_stimage_vectorized_head` loaded
       with the same weights. Compared with a *relative* tolerance, because the two
       sum the same products in different orders and float32 rounding differs in the
       last bits.

    2. Loss. Upstream's `negative_binomial_loss` applied head by head against
       `stimage_nb_loss` on the vectorized tensor, with **both fed the identical
       `y_pred`**. Feeding the same tensor is what makes this a test of the
       restatement rather than of check 1: the negative-binomial NLL is unbounded as
       `n -> 0` (its `lgamma(n)` term diverges), so a last-bit head difference can be
       amplified into an arbitrarily large loss difference. An earlier version of
       this check fed each side its own head output and duly reported a spurious
       `loss_diff=inf` at 4096 genes.

    Declared deviation from upstream, applied on both sides here so it cannot mask a
    restatement bug: `p` is clipped to `[1e-6, 1 - 1e-6]` before the logarithms.
    Upstream does not clip, and `negative_binomial_layer`'s sigmoid saturates, so
    upstream's own loss can evaluate `log(0)`. The reference HEST adaptation of
    STimage clips the same way.

    Returns the observed differences. Raises if either check exceeds tolerance, so a
    mismatch fails the run instead of being reported as a pass.
    """
    import tensorflow as tf

    stimage_model = bind_stimage_upstream(upstream_root)
    rng = np.random.default_rng(int(seed))
    features = rng.normal(size=(2, int(feature_dim))).astype(np.float32)
    y_true_np = rng.uniform(0.0, 4.0, size=(2, int(n_genes))).astype(np.float32)
    kernels = rng.normal(scale=0.1, size=(int(n_genes), int(feature_dim), 2)).astype(np.float32)
    biases = rng.normal(scale=0.1, size=(int(n_genes), 2)).astype(np.float32)

    # --- check 1: head --------------------------------------------------------
    f = tf.constant(features)
    upstream_heads = []  # one (batch, 2) tensor per gene, upstream's real head shape
    for g in range(int(n_genes)):
        logits_g = tf.einsum("bf,fo->bo", f, tf.constant(kernels[g])) + tf.constant(biases[g])
        upstream_heads.append(stimage_model.negative_binomial_layer(logits_g))
    upstream_stacked = tf.stack(upstream_heads, axis=1)

    head = build_stimage_vectorized_head(int(feature_dim), int(n_genes), seed=int(seed))
    head.kernel.assign(tf.constant(np.transpose(kernels, (1, 0, 2))))
    head.bias.assign(tf.constant(biases))
    vectorized = head(f)

    head_diff = float(tf.reduce_max(tf.abs(upstream_stacked - vectorized)).numpy())
    head_scale = float(tf.reduce_max(tf.abs(upstream_stacked)).numpy())
    head_rel = head_diff / max(head_scale, 1e-12)

    # --- check 2: loss, on identical inputs ------------------------------------
    y_pred = tf.stack(
        [
            vectorized[..., 0],
            tf.clip_by_value(vectorized[..., 1], 1.0e-6, 1.0 - 1.0e-6),
        ],
        axis=-1,
    )
    y_true = tf.constant(y_true_np)

    def _upstream_loss(yt, yp):
        # Upstream's `negative_binomial_loss` does not reduce over the batch; Keras
        # averages each per-gene output afterwards, so the mean over spots is the
        # comparable per-gene quantity.
        #
        # `yp[:, g, :]` is `(batch, 2)` -- the shape upstream's heads actually produce.
        # That shape matters: handing the function a `(batch, 1, 2)` slice instead makes
        # `n` become `(batch, 1, 1)`, and `tf.math.lgamma(n + y_true)` then broadcasts
        # against `y_true` of `(batch, 1)` into `(batch, batch, 1)`, silently mixing
        # spots. An earlier version of this check did exactly that and reported a
        # spurious loss mismatch at every gene count, including one gene.
        return tf.add_n(
            [
                tf.reduce_mean(stimage_model.negative_binomial_loss(yt[:, g : g + 1], yp[:, g, :]))
                for g in range(int(n_genes))
            ]
        )

    # Correctness is judged in float64. The negative-binomial NLL carries `lgamma(n)`,
    # which diverges as n -> 0, so in float32 the large cancelling terms lose most of
    # their significant digits and two evaluations of the *same* formula diverge by
    # ~1e-3 relative. That is a property of upstream's loss, not of the restatement,
    # and float32 cannot separate the two -- float64 can.
    y_pred64 = tf.cast(y_pred, tf.float64)
    y_true64 = tf.cast(y_true, tf.float64)
    upstream_loss64 = float(_upstream_loss(y_true64, y_pred64).numpy())
    vectorized_loss64 = float(stimage_nb_loss(y_true64, y_pred64).numpy())
    loss_diff64 = abs(upstream_loss64 - vectorized_loss64)
    loss_rel64 = loss_diff64 / max(abs(upstream_loss64), 1.0)

    upstream_loss = float(_upstream_loss(y_true, y_pred).numpy())
    vectorized_loss = float(stimage_nb_loss(y_true, y_pred).numpy())
    loss_diff = abs(upstream_loss - vectorized_loss)
    loss_rel = loss_diff / max(abs(upstream_loss), 1.0)

    # Per-gene breakdown. If the two losses disagree, this says which gene and by how
    # much, which is what distinguishes a restatement bug from an accumulation effect.
    # `stimage_nb_loss` with a one-hot weight mask reduces to a single gene's mean over
    # spots -- the same quantity upstream's per-head loss is reduced to -- so the two
    # are directly comparable gene by gene.
    per_gene_diffs = np.zeros(int(n_genes), dtype=np.float64)
    for g in range(int(n_genes)):
        mask = np.zeros((y_true_np.shape[0], int(n_genes)), dtype=np.float32)
        mask[:, g] = 1.0
        mine_g = float(stimage_nb_loss(y_true, y_pred, weights=tf.constant(mask)).numpy())
        up_g = float(
            tf.reduce_mean(
                stimage_model.negative_binomial_loss(y_true[:, g : g + 1], y_pred[:, g, :])
            ).numpy()
        )
        per_gene_diffs[g] = abs(mine_g - up_g)
    worst_gene = int(np.argmax(per_gene_diffs))

    # Raw per-spot terms for gene 0, plus a numpy restatement of upstream's formula.
    # Included so that a mismatch points at what actually differs instead of being
    # lost to the exception: an earlier version of this check raised before the report
    # was attached anywhere, and the diagnostics were discarded.
    from scipy.special import gammaln

    up_nll0 = stimage_model.negative_binomial_loss(
        y_true64[:, 0:1], y_pred64[:, 0, :]
    ).numpy().reshape(-1)
    n0 = y_pred64[:, 0, 0].numpy()
    p0 = y_pred64[:, 0, 1].numpy()
    y0 = y_true64[:, 0].numpy()
    np_nll0 = (
        gammaln(n0) + gammaln(y0 + 1.0) - gammaln(n0 + y0) - n0 * np.log(p0) - y0 * np.log(1.0 - p0)
    )

    report = {
        "n_genes": int(n_genes),
        "feature_dim": int(feature_dim),
        "max_abs_head_difference": head_diff,
        "relative_head_difference": head_rel,
        "relative_loss_difference_float64": loss_rel64,
        "abs_loss_difference_float64": loss_diff64,
        "relative_loss_difference_float32": loss_rel,
        "abs_loss_difference_float32": loss_diff,
        "upstream_loss_float64": upstream_loss64,
        "vectorized_loss_float64": vectorized_loss64,
        "max_per_gene_loss_difference_float32": float(per_gene_diffs.max()),
        "worst_gene_index": worst_gene,
        "gene0_n": [float(v) for v in n0],
        "gene0_p": [float(v) for v in p0],
        "gene0_y_true": [float(v) for v in y0],
        "gene0_upstream_nll": [float(v) for v in up_nll0],
        "gene0_numpy_reference_nll": [float(v) for v in np_nll0],
        "gene0_upstream_vs_numpy_max_abs": float(np.max(np.abs(up_nll0 - np_nll0))),
        "rel_atol": float(rel_atol),
        "head_matches": bool(head_rel <= rel_atol),
        "loss_matches": bool(loss_rel64 <= rel_atol),
    }
    if not (report["head_matches"] and report["loss_matches"]):
        error = AssertionError(
            "STimage vectorization does not match upstream: "
            f"head_rel={head_rel:.3e} loss_rel_float64={loss_rel64:.3e} rel_atol={rel_atol:.1e}"
        )
        error.report = report  # carried so the caller can still write the diagnostics
        raise error
    return report


def save_stimage_head(head, path):
    """Persist a vectorized head's kernel/bias as a single ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kernel=head.kernel.numpy(), bias=head.bias.numpy())
    return path


def load_stimage_head_weights(path, feature_dim, n_genes, seed=42, upstream_root=None):
    """Rebuild a vectorized head and load kernel/bias saved by `save_stimage_head`."""
    if _STIMAGE_MODEL_HANDLE[0] is None:
        if upstream_root is None:
            raise ValueError("upstream_root is required before the upstream module is bound")
        bind_stimage_upstream(upstream_root)
    head = build_stimage_vectorized_head(int(feature_dim), int(n_genes), seed=int(seed))
    payload = np.load(str(path))
    head.kernel.assign(payload["kernel"])
    head.bias.assign(payload["bias"])
    return head


def _stimage_open_features(feature_bundle, slide_ids):
    """Return per-slide feature arrays plus the cumulative row count."""
    arrays = {}
    lengths = []
    for sid in slide_ids:
        payload = feature_bundle[sid]
        if payload.get("feature_path") is not None:
            arrays[sid] = np.load(str(payload["feature_path"]), mmap_mode="r")
        else:
            arrays[sid] = payload["features"]
        lengths.append(int(payload["shape"][0]))
    return arrays, np.cumsum(lengths)


def stimage_feature_batches(feature_bundle, slide_ids, targets=None, batch_size=32, shuffle=False, seed=42):
    """Yield ``(features, targets)`` batches over the given slides.

    Rows are concatenated in ``slide_ids`` order and, within a slide, follow the
    h5ad ``in_tissue == 1`` ordering every other row in this benchmark uses, so
    `targets` index the same way the rest of the pipeline does. ``targets`` is a
    ``{slide_id: (n_spots, n_genes)}`` mapping or ``None`` to yield features only.

    This row has no torch `Dataset` counterpart on purpose: it runs in
    `stimage_bench`, where torch cannot be imported (see the import note at the top
    of this module), and the head is a TensorFlow module. Training and evaluation
    therefore consume the cached features as plain arrays.
    """
    arrays, cumlen = _stimage_open_features(feature_bundle, slide_ids)
    order = np.arange(int(cumlen[-1]), dtype=np.int64)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        idx = order[start : start + int(batch_size)]
        slide_idx = np.searchsorted(cumlen, idx, side="right")
        prev = np.where(slide_idx == 0, 0, cumlen[np.maximum(slide_idx - 1, 0)])
        local = idx - prev
        feat_pieces = []
        tgt_pieces = []
        for k, sid in enumerate(slide_ids):
            mask = slide_idx == k
            if not np.any(mask):
                continue
            feat_pieces.append(np.asarray(arrays[sid][local[mask]], dtype=np.float32))
            if targets is not None:
                tgt_pieces.append(np.asarray(targets[sid], dtype=np.float32)[local[mask]])
        x = np.concatenate(feat_pieces, axis=0)
        y = np.concatenate(tgt_pieces, axis=0) if tgt_pieces else None
        yield x, y


def stimage_spot_matrix(feature_bundle, slide_ids, targets=None):
    """Concatenate cached features (and targets) in spot order, for whole-split passes.

    Returns ``(features, targets_or_None, spot_ids)`` with rows in the same order
    `stimage_feature_batches` iterates when ``shuffle=False``.
    """
    arrays, cumlen = _stimage_open_features(feature_bundle, slide_ids)
    features = np.concatenate(
        [np.asarray(arrays[sid], dtype=np.float32) for sid in slide_ids], axis=0
    )
    out_targets = None
    if targets is not None:
        out_targets = np.concatenate(
            [np.asarray(targets[sid], dtype=np.float32) for sid in slide_ids], axis=0
        )
    ids = np.concatenate(
        [np.asarray(feature_bundle[sid]["spot_ids"]).astype(str) for sid in slide_ids], axis=0
    )
    return features, out_targets, ids
