import os
import sys
import urllib.request

CHECKPOINTS = [
    {
        "name": "SCNet Base",
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt",
        "filename": "scnet_base.ckpt",
    },
    {
        "name": "SCNet Base Config",
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml",
        "filename": "config_scnet_base.yaml",
    },
    {
        "name": "SCNet Small",
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.16/model_scnet_masked_ep_156_sdr_8.8149.ckpt",
        "filename": "scnet_small.ckpt",
    },
    {
        "name": "SCNet Small Config",
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.16/config_musdb18_scnet_small.yaml",
        "filename": "config_scnet_small.yaml",
    },
]


def reporthook(block_num, block_size, total_size):
    dl = block_num * block_size
    if total_size > 0:
        p = min(100.0, dl * 100.0 / total_size)
        mb_dl = dl / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        sys.stdout.write(f"\r   Progress: {p:.1f}% ({mb_dl:.1f}/{mb_tot:.1f} MB)")
        sys.stdout.flush()


def download_all(dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Target Checkpoints Directory: {dest_dir}\n")

    for item in CHECKPOINTS:
        target = os.path.join(dest_dir, item["filename"])
        if os.path.exists(target) and os.path.getsize(target) > 5000:
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"[FOUND] {item['filename']} already downloaded ({size_mb:.2f} MB). Skipping.")
            continue

        print(f"[DOWNLOADING] {item['name']} from {item['url']}...")
        urllib.request.urlretrieve(item["url"], target, reporthook=reporthook)
        sys.stdout.write("\n")
        size_mb = os.path.getsize(target) / (1024 * 1024)
        print(f"[SAVED] {item['filename']} ({size_mb:.2f} MB)\n")

    print("All checkpoints downloaded successfully!")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    download_all(base_dir)

