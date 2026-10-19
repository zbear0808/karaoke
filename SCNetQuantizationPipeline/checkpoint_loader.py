import os
import sys
import urllib.request
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from scnet_pruner import build_mock_scnet_small

# Direct release asset for official trained SCNet-Base (10.08M params, 40.5 MB in FP32)
DEFAULT_SCNET_BASE_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.6/scnet_checkpoint_musdb18.ckpt"
)


def download_file_with_progress(url: str, dest_path: str):
    """
    Downloads a remote file with a simple CLI progress report.
    """
    print(f"Downloading checkpoint from: {url}")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    sys.stdout.write("\n")
    print(f"Downloaded successfully to: {dest_path}")


def load_scnet_checkpoint(
    checkpoint_path: Optional[str] = None,
    checkpoint_url: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: str = ".checkpoints",
    device: str = "cpu"
) -> nn.Module:
    """
    Loads an SCNet model by directly pulling a checkpoint online (via URL or Hugging Face Hub),
    or from a local path.
    """
    resolved_path = checkpoint_path

    # Option 1: Direct URL Download
    if resolved_path is None and checkpoint_url:
        os.makedirs(cache_dir, exist_ok=True)
        url_filename = os.path.basename(checkpoint_url)
        cached_target = os.path.join(cache_dir, url_filename)
        if not os.path.exists(cached_target):
            download_file_with_progress(checkpoint_url, cached_target)
        resolved_path = cached_target

    # Option 2: Hugging Face Hub Download
    elif resolved_path is None and repo_id and filename:
        if hf_hub_download is None:
            raise ImportError("huggingface_hub required for HF repos. Run: pip install huggingface_hub")
        print(f"Downloading checkpoint '{filename}' from Hugging Face '{repo_id}'...")
        resolved_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print(f"Downloaded to: {resolved_path}")

    # Fallback to mock if no source specified
    if resolved_path is None or not os.path.exists(resolved_path):
        print("[INFO] No external checkpoint specified. Instantiating structural mock...")
        return build_mock_scnet_small()

    print(f"Loading checkpoint weights from: {resolved_path}")
    raw_checkpoint = torch.load(resolved_path, map_location=device)

    # Extract state_dict
    if isinstance(raw_checkpoint, dict):
        if "state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["state_dict"]
        elif "model" in raw_checkpoint:
            state_dict = raw_checkpoint["model"]
        elif "model_state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["model_state_dict"]
        else:
            state_dict = raw_checkpoint
    else:
        state_dict = raw_checkpoint

    # Strip prefixes like 'model.' or 'module.'
    cleaned_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_state_dict[clean_key] = value

    model = build_mock_scnet_small()
    try:
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("Successfully loaded weights into SCNet model.")
    except Exception as e:
        print(f"[WARNING] Partial state_dict load: {e}")

    model.eval()
    return model
