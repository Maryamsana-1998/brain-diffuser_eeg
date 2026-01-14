import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import numpy as np
from sklearn.linear_model import Ridge


# ----------------------------
# EEG loading + normalization
# ----------------------------
@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def load_pt_eeg(path: str, verbose: bool = True) -> Dict:
    """Load a .pt file (CPU) and return its dict."""
    data = torch.load(path, map_location="cpu")
    if verbose:
        print("Loaded file:", path)
        print("Keys in file:", data.keys())
    return data


def inspect_eeg_tensor(eeg: torch.Tensor, verbose: bool = True) -> np.ndarray:
    """Print diagnostics and return EEG as numpy."""
    if verbose:
        print("EEG tensor shape:", tuple(eeg.shape))
        print("EEG dtype:", eeg.dtype)
    # eeg_np = eeg.detach().cpu().numpy()
    eeg_np = eeg

    # Global statistics (kept from your code)
    mean_per_trial = np.mean(eeg_np, axis=1)
    if verbose:
        print("Mean EEG per trial shape:", mean_per_trial.shape)
        print("Global EEG mean/std:", float(eeg_np.mean()), float(eeg_np.std()))
    return eeg_np


def compute_norm_stats(x: np.ndarray, eps: float = 1e-8) -> NormStats:
    """Compute per-feature mean/std over samples axis=0 with ddof=1."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1)
    std = np.where(std < eps, 1.0, std)  # avoid division by zero
    return NormStats(mean=mean, std=std)


def apply_normalization(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply z-score normalization using provided stats."""
    return (x - stats.mean) / stats.std


def load_and_inspect_eeg(
    path: str,
    num_samples: int = 10,
    eps: float = 1e-8,
    verbose: bool = True
) -> Tuple[np.ndarray, NormStats]:
    """
    Load EEG data from a .pt file, take first num_samples, normalize, and print diagnostics.
    Returns (normalized_eeg_subset, stats_used).
    """
    data = load_pt_eeg(path, verbose=verbose)

    if "eeg" not in data:
        raise KeyError(f"'eeg' key not found in {path}. Available keys: {list(data.keys())}")

    eeg_np = inspect_eeg_tensor(data["eeg"], verbose=verbose)
    # print("eeg np ", eeg_np)

    eeg_subset = eeg_np[:num_samples]
    if verbose:
        print(f"Using first {num_samples} trials")
        print("Subset shape:", eeg_subset.shape)

    stats = compute_norm_stats(eeg_subset, eps=eps)
    eeg_norm = apply_normalization(eeg_subset, stats)

    if verbose:
        print("Normalized EEG shape:", eeg_norm.shape)
        print("Normalized max/min:", float(np.max(eeg_norm)), float(np.min(eeg_norm)))
        print("Any NaNs?:", bool(np.isnan(eeg_norm).any()))
        print("Any Infs?:", bool(np.isinf(eeg_norm).any()))

    return eeg_norm, stats


# ----------------------------
# Latents loading
# ----------------------------
def load_latents_from_npz(npz_path: str, verbose: bool = True) -> np.ndarray:
    """
    Load latents from an NPZ.
    Preference order:
      1) 'train_latents' if present
      2) 'latents' if present
      3) first array in file
    """
    lat_npz = np.load(npz_path, allow_pickle=True)
    if verbose:
        print("Latents NPZ keys:", lat_npz.files)

    if "train_latents" in lat_npz.files:
        arr = lat_npz["train_latents"]
    elif "latents" in lat_npz.files:
        arr = lat_npz["latents"]
    else:
        arr = lat_npz[lat_npz.files[0]]

    arr = np.asarray(arr)
    if verbose:
        print("Loaded latents shape:", arr.shape, "dtype:", arr.dtype)
    return arr


# ----------------------------
# Feature shaping helpers
# ----------------------------
def ensure_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (n_samples, n_features/targets)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def make_features(
    X: np.ndarray,
    use_time_mean: bool = False
) -> np.ndarray:
    """
    Convert EEG tensor X into 2D features for sklearn.

    - If use_time_mean=False: flatten everything except batch: (n, -1)
    - If use_time_mean=True: mean over last axis (time), then flatten: (n, -1)
    """
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError(f"Expected X to have at least 2 dims; got shape {X.shape}")

    if use_time_mean:
        X_time_mean = X.mean(axis=-1)
        X_feat = X_time_mean.reshape(X_time_mean.shape[0], -1)
    else:
        X_feat = X.reshape(X.shape[0], -1)

    return X_feat


def align_samples(X: np.ndarray, Y: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate X and Y to the same number of samples if needed."""
    n_x, n_y = X.shape[0], Y.shape[0]
    if verbose:
        print("n_samples X:", n_x, "n_samples Y:", n_y)

    if n_x != n_y:
        n = min(n_x, n_y)
        if verbose:
            print(f"[WARN] sample mismatch; truncating both to n={n}")
        X = X[:n]
        Y = Y[:n]
    return X, Y


# ----------------------------
# Regression + prediction
# ----------------------------
def fit_ridge(
    X_feat: np.ndarray,
    Y: np.ndarray,
    alpha: float = 50000,
    max_iter: int = 10000,
    fit_intercept: bool = True,
    verbose: bool = True
) -> Ridge:
    """Fit ridge regression."""
    X_feat = ensure_2d(X_feat)
    Y = ensure_2d(Y)

    reg = Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept)
    reg.fit(X_feat, Y)

    if verbose:
        print("Fitted Ridge.")
        print("coef_.shape:", reg.coef_.shape)
        print("intercept_.shape:", np.shape(reg.intercept_))

    return reg


def standardize_then_match_train(pred: np.ndarray, train_Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Your original post-processing:
      std_norm_pred = (pred - mean(pred)) / std(pred)
      pred_latents  = std_norm_pred * std(train_Y) + mean(train_Y)
    """
    pred = np.asarray(pred)
    train_Y = np.asarray(train_Y)

    pred_mean = np.mean(pred, axis=0)
    pred_std = np.std(pred, axis=0)
    pred_std = np.where(pred_std < eps, 1.0, pred_std)

    std_norm_pred = (pred - pred_mean) / pred_std

    train_mean = np.mean(train_Y, axis=0)
    train_std = np.std(train_Y, axis=0)
    train_std = np.where(train_std < eps, 1.0, train_std)

    return std_norm_pred * train_std + train_mean


def save_outputs(
    pred_latents: np.ndarray,
    latents_out_path: str,
    weights_out_path: str,
    weights_payload: Dict,
    verbose: bool = True
) -> None:
    os.makedirs(os.path.dirname(latents_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(weights_out_path), exist_ok=True)

    np.save(latents_out_path, pred_latents)
    if verbose:
        print("Saved predicted latents to:", latents_out_path)

    with open(weights_out_path, "wb") as f:
        pickle.dump(weights_payload, f)

    if verbose:
        print("Saved regression payload to:", weights_out_path)


# ----------------------------
# End-to-end pipeline (same logic, cleaned)
# ----------------------------
def run_eeg_to_latents_ridge(
    train_pt_path: str,
    test_pt_path: str,
    latents_npz_path: str,
    out_latents_path: str = "data/predicted_features/eeg_10test.npy",
    out_weights_path: str = "data/regression_weights/eeg_vdvae_regression_weights.pkl",
    num_samples: int = 10,
    use_time_mean_features: bool = False,
    alpha: float = 50000,
    max_iter: int = 10000,
    eps: float = 1e-8,
    verbose: bool = True
) -> None:
    # 1) Load + normalize TRAIN EEG subset
    train_eeg, train_stats = load_and_inspect_eeg(
        train_pt_path, num_samples=num_samples, eps=eps, verbose=verbose
    )

    # 2) Load latents
    train_latents = load_latents_from_npz(latents_npz_path, verbose=verbose)

    # 3) Align sample counts (same behavior you had)
    train_eeg, train_latents = align_samples(train_eeg, train_latents, verbose=verbose)

    # 4) Prepare features/targets for ridge
    X = train_eeg
    Y = ensure_2d(train_latents)

    if verbose:
        print("Raw X shape:", X.shape, "ndim:", X.ndim, "dtype:", X.dtype)
        print("Raw Y shape:", Y.shape, "ndim:", Y.ndim, "dtype:", Y.dtype)

    X_feat = make_features(X, use_time_mean=use_time_mean_features)
    if verbose:
        print("X_feat shape:", X_feat.shape)
        print("Final Y shape:", Y.shape)

    # 5) Fit ridge
    reg = fit_ridge(X_feat, Y, alpha=alpha, max_iter=max_iter, verbose=verbose)

    # NOTE: your original code did reg.score(train_fmri, train_latents)
    # but train_fmri was not the same feature matrix you trained on if you flattened.
    # So we score on the same X_feat used for training.
    if verbose:
        print("train R^2 (on X_feat):", float(reg.score(X_feat, Y)))

    # 6) Load + normalize TEST EEG subset (kept same behavior: normalize test subset independently)
    test_eeg, test_stats = load_and_inspect_eeg(
        test_pt_path, num_samples=num_samples, eps=eps, verbose=verbose
    )
    if verbose:
        print("Test EEG shape:", test_eeg.shape)

    # 7) Predict latents from TEST EEG
    X_test_feat = make_features(test_eeg, use_time_mean=use_time_mean_features)
    if verbose:
        print("Test features shape:", X_test_feat.shape)

    pred_test_latent = reg.predict(X_test_feat)

    # 8) Post-process preds (your exact logic)
    pred_latents = standardize_then_match_train(pred_test_latent, train_latents, eps=eps)

    # 9) Save outputs
    # Your old code tried to pickle `datadict` but it was undefined.
    # Here we create a sensible payload that includes everything needed to reuse the model.
    weights_payload = {
        "ridge": reg,
        "alpha": alpha,
        "max_iter": max_iter,
        "use_time_mean_features": use_time_mean_features,
        "num_samples": num_samples,
        "train_norm_stats": train_stats,
        "test_norm_stats": test_stats,
        "train_latents_mean": np.mean(train_latents, axis=0),
        "train_latents_std": np.std(train_latents, axis=0),
    }

    save_outputs(pred_latents, out_latents_path, out_weights_path, weights_payload, verbose=verbose)


# ----------------------------
# Example usage (matches your paths)
# ----------------------------
if __name__ == "__main__":
    train_path = "/home/sanama/EEG_dataset/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/train.pt"
    test_path = "/home/sanama/EEG_dataset/things-eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt"
    latents_path = "/home/sanama/brain-diffuser/aardvak.npz"

    run_eeg_to_latents_ridge(
        train_pt_path=train_path,
        test_pt_path=test_path,
        latents_npz_path=latents_path,
        out_latents_path="data/predicted_features/eeg_10test.npy",
        out_weights_path="data/regression_weights/eeg_vdvae_regression_weights.pkl",
        num_samples=10,
        use_time_mean_features=False,  # set True to use time-averaged features
        alpha=50000,
        max_iter=10000,
        verbose=True
    )
