"""
Upload/download data and checkpoints to HuggingFace Hub.

Setup:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    # Upload processed data (instructions, labels, images, matched_shas.txt)
    python hf_storage.py upload-data --repo your-username/grasp-anything-data

    # Download data to local
    python hf_storage.py download-data --repo your-username/grasp-anything-data

    # Upload checkpoints
    python hf_storage.py upload-checkpoints --repo your-username/grasp-anything-checkpoints

    # Download checkpoints
    python hf_storage.py download-checkpoints --repo your-username/grasp-anything-checkpoints
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo


def upload_data(repo_id: str, data_dir: str = "./data"):
    """
    Pack data into zip files then upload to HF Hub.
    This avoids rate limits from uploading 30K+ individual files.
    """
    import zipfile

    api = HfApi()
    data_path = Path(data_dir)
    pack_dir = data_path / "_packed"
    pack_dir.mkdir(exist_ok=True)

    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    print(f"Uploading data to {repo_id}...")

    # Pack each folder into a zip
    folders = ["grasp_instructions", "grasp_label_positive", "images"]
    for folder in folders:
        folder_path = data_path / folder
        if not folder_path.exists():
            print(f"  [SKIP] {folder} not found")
            continue
        zip_path = pack_dir / f"{folder}.zip"
        if not zip_path.exists():
            print(f"  Packing {folder}/ into zip...")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for f in sorted(folder_path.iterdir()):
                    if f.is_file():
                        zf.write(f, f"{folder}/{f.name}")
            print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  [SKIP] {zip_path} already packed")

    # Copy txt files to pack dir
    for txt_file in ["matched_shas.txt", "selected_shas.txt"]:
        src = data_path / txt_file
        if src.exists():
            import shutil
            shutil.copy2(src, pack_dir / txt_file)

    # Upload the pack dir (only a few files)
    print(f"\n  Uploading packed files to HF Hub...")
    api.upload_folder(
        folder_path=str(pack_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nData uploaded to: https://huggingface.co/datasets/{repo_id}")


def download_data(repo_id: str, data_dir: str = "./data"):
    """Download packed data from HF Hub and extract."""
    import zipfile

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_path),
    )

    # Extract zips
    for zip_name in ["grasp_instructions.zip", "grasp_label_positive.zip", "images.zip"]:
        zip_path = data_path / zip_name
        if not zip_path.exists():
            continue
        print(f"  Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(data_path))
        print(f"  Done: {zip_name}")

    print(f"Data ready at {data_dir}")


def upload_checkpoints(repo_id: str, checkpoint_dir: str = "./checkpoints"):
    """Upload checkpoints to HF Hub."""
    api = HfApi()
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        print("No checkpoints directory found")
        return

    create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
    print(f"Uploading checkpoints to {repo_id}...")

    api.upload_large_folder(
        folder_path=str(ckpt_path),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Checkpoints uploaded to: https://huggingface.co/{repo_id}")


def download_checkpoints(repo_id: str, checkpoint_dir: str = "./checkpoints"):
    """Download checkpoints from HF Hub."""
    print(f"Downloading checkpoints from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=checkpoint_dir,
    )
    print(f"Checkpoints downloaded to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Hub storage for data & checkpoints")
    subparsers = parser.add_subparsers(dest="command")

    # Upload data
    p = subparsers.add_parser("upload-data")
    p.add_argument("--repo", type=str, required=True, help="HF repo ID (e.g., username/grasp-data)")
    p.add_argument("--data_dir", type=str, default="./data")

    # Download data
    p = subparsers.add_parser("download-data")
    p.add_argument("--repo", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")

    # Upload checkpoints
    p = subparsers.add_parser("upload-checkpoints")
    p.add_argument("--repo", type=str, required=True, help="HF repo ID (e.g., username/grasp-checkpoints)")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    # Download checkpoints
    p = subparsers.add_parser("download-checkpoints")
    p.add_argument("--repo", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    if args.command == "upload-data":
        upload_data(args.repo, args.data_dir)
    elif args.command == "download-data":
        download_data(args.repo, args.data_dir)
    elif args.command == "upload-checkpoints":
        upload_checkpoints(args.repo, args.checkpoint_dir)
    elif args.command == "download-checkpoints":
        download_checkpoints(args.repo, args.checkpoint_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
