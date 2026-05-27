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
    """Upload processed data (no raw zips) to HF Hub using large folder upload."""
    api = HfApi()
    data_path = Path(data_dir)

    # Create repo if not exists
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    print(f"Uploading data to {repo_id}...")
    print(f"  Using upload_large_folder for reliable upload of many files...")

    # Upload entire data dir, ignoring raw/ folder
    api.upload_large_folder(
        folder_path=str(data_path),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["raw/*", "raw/**"],
    )

    print(f"\nData uploaded to: https://huggingface.co/datasets/{repo_id}")


def download_data(repo_id: str, data_dir: str = "./data"):
    """Download processed data from HF Hub."""
    print(f"Downloading data from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_dir,
    )
    print(f"Data downloaded to {data_dir}")


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
