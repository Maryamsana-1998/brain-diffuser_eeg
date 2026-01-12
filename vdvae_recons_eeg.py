#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

# -----------------------------
# VDVAE imports
# -----------------------------
VDVAE_ROOT = "vdvae"  # change if needed
sys.path.append(VDVAE_ROOT)

from model_utils import set_up_data, load_vaes  # from your vdvae codebase


# -----------------------------
# Helpers
# -----------------------------
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FolderImageDataset(Dataset):
    """
    Dataset for structure like:
      root/
        00001_classA/img1.png
        00002_classB/img2.jpg
        ...

    It sorts subfolders, then picks (by default) the first image inside each folder.
    """

    def __init__(
        self,
        folder_path: str,
        resize: Tuple[int, int] = (64, 64),
        max_images: Optional[int] = None,
        pick: str = "first",  # "first" or "all"
    ):
        self.root = Path(folder_path)
        self.resize = resize
        self.pick = pick

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        # 1) sorted subfolders
        subdirs = sorted([d for d in self.root.iterdir() if d.is_dir()])

        if max_images is not None:
            subdirs = subdirs[:max_images]

        paths: List[Path] = []

        if pick == "first":
            # pick 1 image per folder (the first sorted filename)
            for d in subdirs:
                imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])
                if len(imgs) > 0:
                    paths.append(imgs[0])

        elif pick == "all":
            # include all images from each folder
            for d in subdirs:
                imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])
                paths.extend(imgs)

        else:
            raise ValueError("pick must be 'first' or 'all', got: {}".format(pick))

        if len(paths) == 0:
            raise FileNotFoundError("No images found under: {} (searched subfolders)".format(folder_path))

        self.paths = paths

        # Convert to a stable tensor transform: PIL -> resized -> tensor in [0,1], CHW
        self.to_tensor = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.to_tensor(img)  # float32, CHW, [0,1]
        print('image shape', img.shape)
        return img


# -----------------------------
# VDVAE Latent utils
# -----------------------------
LAYER_DIMS = np.array([
    2**4, 2**4,
    2**8, 2**8, 2**8, 2**8,
    2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10,
    2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12,
    2**14
], dtype=np.int64)


StatsType = List[Dict[str, Any]]


def flatten_stats_to_latents(stats: StatsType) -> np.ndarray:
    """
    stats[i]['z'] is a tensor [B, C, H, W].
    Returns flattened latent per sample: [B, sum(layer_dims)].
    """
    batch_latent: List[np.ndarray] = []
    for i in range(31):
        z = stats[i]["z"].detach().cpu().numpy()
        batch_latent.append(z.reshape(z.shape[0], -1))
    return np.hstack(batch_latent)


def latents_flat_to_hier(latents_flat: np.ndarray, ref_stats: StatsType) -> List[np.ndarray]:
    """
    Convert flattened latents [N, sum] -> list of 31 arrays shaped like ref_stats[i]['z'].
    """
    transformed: List[np.ndarray] = []
    start = 0
    for i in range(31):
        dim = int(LAYER_DIMS[i])
        chunk = latents_flat[:, start:start + dim]
        start += dim

        # ref_stats[i]['z'] is [B, C, H, W]
        _, c, h, w = ref_stats[i]["z"].shape
        transformed.append(chunk.reshape(len(latents_flat), c, h, w))
    return transformed


def sample_from_hier_latents(
    latents_hier: List[np.ndarray],
    sample_ids: Sequence[int],
    device: torch.device
) -> List[torch.Tensor]:
    """
    latents_hier: list of 31 arrays [N, C, H, W]
    returns: list of 31 tensors [B, C, H, W] on GPU/CPU
    """
    if len(latents_hier) == 0:
        raise ValueError("latents_hier is empty")

    n_total = latents_hier[0].shape[0]
    valid_ids = [i for i in sample_ids if i < n_total]

    out: List[torch.Tensor] = []
    for layer in latents_hier:
        out.append(torch.from_numpy(layer[valid_ids]).float().to(device))
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def build_hparams() -> DotDict:
    H = {
        "image_size": 64,
        "image_channels": 3,
        "seed": 0,
        "port": 29500,
        "save_dir": "./saved_models/test",
        "data_root": "./",
        "desc": "test",
        "hparam_sets": "imagenet64",
        "restore_path": "imagenet64-iter-1600000-model.th",
        "restore_ema_path": "vdvae/model/imagenet64-iter-1600000-model-ema.th",
        "restore_log_path": "imagenet64-iter-1600000-log.jsonl",
        "restore_optimizer_path": "imagenet64-iter-1600000-opt.th",
        "dataset": "imagenet64",
        "ema_rate": 0.999,
        "enc_blocks": "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
        "dec_blocks": "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
        "zdim": 16,
        "width": 512,
        "custom_width_str": "",
        "bottleneck_multiple": 0.25,
        "no_bias_above": 64,
        "scale_encblock": False,
        "test_eval": True,
        "warmup_iters": 100,
        "num_mixtures": 10,
        "grad_clip": 220.0,
        "skip_threshold": 380.0,
        "lr": 0.00015,
        "lr_prior": 0.00015,
        "wd": 0.01,
        "wd_prior": 0.0,
        "num_epochs": 10000,
        "n_batch": 4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.9,
        "temperature": 1.0,
        "iters_per_ckpt": 25000,
        "iters_per_print": 1000,
        "iters_per_save": 10000,
        "iters_per_images": 10000,
        "epochs_per_eval": 1,
        "epochs_per_probe": None,
        "epochs_per_eval_save": 1,
        "num_images_visualize": 8,
        "num_variables_visualize": 6,
        "num_temperatures_visualize": 3,
        "mpi_size": 1,
        "local_rank": 0,
        "rank": 0,
        "logdir": "./saved_models/test/log",
    }
    return DotDict(H)


def compute_true_latents(
    ema_vae: Any,
    preprocess_fn: Any,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, StatsType]:
    """
    Encodes actual images through VDVAE to get:
      - flattened latents per image
      - one 'stats' object to use as reference shapes
    """
    all_flat: List[np.ndarray] = []
    ref_stats: Optional[StatsType] = None

    ema_vae.eval()
    for step, x in enumerate(loader):
        # x is already CHW float in [0,1] from dataset
        # most vdvae preprocess expects BCHW and maybe returns (data_input, target)
        data_input, _target = preprocess_fn(x)
        print('data shapes', data_input.shape)

        with torch.no_grad():
            activations = ema_vae.encoder.forward(data_input)
            _px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)

        if ref_stats is None:
            ref_stats = stats  # keep shapes

        all_flat.append(flatten_stats_to_latents(stats))
        print("[encode] batch {} -> {} latents".format(step, x.shape[0]))

    if ref_stats is None:
        raise RuntimeError("Failed to compute ref_stats (no batches produced).")

    return np.concatenate(all_flat, axis=0), ref_stats


def decode_pred_latents_to_images(
    ema_vae: Any,
    pred_latents_flat: np.ndarray,
    ref_stats: StatsType,
    out_dir: str,
    batch_size: int,
    upscale_to: Optional[Tuple[int, int]] = (512, 512),
) -> None:
    device = get_device()
    ema_vae.eval()

    ensure_dir(out_dir)

    # Convert flat -> hierarchical
    pred_hier = latents_flat_to_hier(pred_latents_flat, ref_stats)

    n = pred_latents_flat.shape[0]
    n_batches = int(np.ceil(n / float(batch_size)))

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n)
        sample_ids = list(range(start, end))

        print("[decode] samples {}..{}".format(start, end - 1))

        samp = sample_from_hier_latents(pred_hier, sample_ids, device=device)

        with torch.no_grad():
            px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
            imgs = ema_vae.decoder.out_net.sample(px_z)  # often uint8 HWC

        for j, im in enumerate(imgs):
            img = Image.fromarray(im).convert("RGB")
            if upscale_to is not None:
                img = img.resize(upscale_to, resample=Image.BICUBIC)
            img.save(os.path.join(out_dir, "{:06d}.png".format(start + j)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_latents_path", type=str, required=True, help="Path to predicted latents .npy")
    parser.add_argument("--out_dir", type=str, default="results/vdvae_out")
    parser.add_argument("--image_root", type=str, default="EEG_dataset/things-eeg/Image_set/test_images/",
                        help="Folder with subfolders (00001_*, 00002_*, ...)")
    parser.add_argument("--num_images", type=int, default=10, help="How many subfolders/images to use")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--compute_true_latents", action="store_true",
                        help="Encode images through VDVAE to get ref stats (required for correct reshaping).")
    args = parser.parse_args()

    device = get_device()
    print("[info] device:", device)
    print("[info] loading VDVAE...")

    H = build_hparams()
    H, preprocess_fn = set_up_data(H)

    ema_vae = load_vaes(H)

    # Dataset
    dataset = FolderImageDataset(
        args.image_root,
        resize=(64, 64),
        max_images=args.num_images,
        pick="first",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Predicted latents
    pred_latents_flat = np.load(args.pred_latents_path)
    pred_latents_flat = np.asarray(pred_latents_flat)
    print("[info] pred_latents_flat shape:", pred_latents_flat.shape)

    if not args.compute_true_latents:
        raise RuntimeError(
            "You need ref_stats to reshape latents correctly.\n"
            "Run with --compute_true_latents so we obtain latent shapes from the model."
        )

    _true_latents_flat, ref_stats = compute_true_latents(ema_vae, preprocess_fn, loader, device)
    print("[info] true_latents_flat shape:", _true_latents_flat.shape)

    decode_pred_latents_to_images(
        ema_vae=ema_vae,
        pred_latents_flat=pred_latents_flat,
        ref_stats=ref_stats,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        upscale_to=(512, 512),
    )

    print("[done] wrote images to:", args.out_dir)


if __name__ == "__main__":
    main()
