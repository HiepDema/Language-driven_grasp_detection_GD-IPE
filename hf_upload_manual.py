"""
Upload packed data to HuggingFace using git LFS (bypasses API rate limit).

Usage:
    python hf_upload_manual.py --repo Hiep1234/grasp-anything-data --data_dir ./data
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def pack_data(data_dir: str):
    """Pack 3 data folders into zip files."""
    data_path = Path(data_dir)
    folders = ["grasp_instructions", "grasp_label_positive", "images"]

    for folder in folders:
        folder_path = data_path / folder
        zip_path = data_path / f"{folder}.zip"

        if zip_path.exists():
            print(f"  [SKIP] {zip_path} already exists")
            continue
        if not folder_path.exists():
            print(f"  [SKIP] {folder}/ not found")
            continue

        print(f"  Packing {folder}/...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            files = sorted(folder_path.iterdir())
            for i, f in enumerate(files):
                if f.is_file():
                    zf.write(f, f"{folder}/{f.name}")
                if (i + 1) % 5000 == 0:
                    print(f"    {i+1}/{len(files)} files packed")
        size_gb = zip_path.stat().st_size / 1e9
        print(f"  -> {zip_path.name} ({size_gb:.2f} GB)")


def upload_via_git(repo_id: str, data_dir: str):
    """Upload using git LFS — no API rate limit."""
    data_path = Path(data_dir)
    tmp_repo = Path("/tmp/hf_upload_repo")

    # Files to upload
    upload_files = []
    for name in ["grasp_instructions.zip", "grasp_label_positive.zip", "images.zip", "matched_shas.txt", "selected_shas.txt"]:
        fpath = data_path / name
        if fpath.exists():
            upload_files.append(fpath)

    if not upload_files:
        print("No files to upload!")
        return

    print(f"\nFiles to upload:")
    for f in upload_files:
        size = f.stat().st_size / 1e9
        print(f"  {f.name} ({size:.2f} GB)")

    # Clone the HF repo
    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nCloning {hf_url}...")

    if tmp_repo.exists():
        subprocess.run(["rm", "-rf", str(tmp_repo)], check=True)

    result = subprocess.run(
        ["git", "clone", hf_url, str(tmp_repo)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Repo might not exist, create it first
        print("  Repo not found, creating...")
        subprocess.run(
            ["huggingface-cli", "repo", "create", repo_id.split("/")[1],
             "--type", "dataset", "--organization", repo_id.split("/")[0]],
            capture_output=True, text=True
        )
        subprocess.run(["git", "clone", hf_url, str(tmp_repo)], check=True)

    os.chdir(str(tmp_repo))

    # Setup git LFS
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "lfs", "track", "*.zip"], check=True)

    # Copy files
    print("\nCopying files to repo...")
    for f in upload_files:
        dest = tmp_repo / f.name
        subprocess.run(["cp", str(f), str(dest)], check=True)
        print(f"  Copied {f.name}")

    # Commit and push
    subprocess.run(["git", "add", ".gitattributes"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Upload packed dataset (30K samples)"], check=True)

    print("\nPushing to HuggingFace (this may take a while for large files)...")
    subprocess.run(["git", "push"], check=True)

    print(f"\nDone! Data uploaded to: https://huggingface.co/datasets/{repo_id}")

    # Cleanup
    os.chdir("/")
    subprocess.run(["rm", "-rf", str(tmp_repo)])


def main():
    parser = argparse.ArgumentParser(description="Upload data to HF via git LFS")
    parser.add_argument("--repo", type=str, required=True, help="HF repo (e.g., Hiep1234/grasp-anything-data)")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--pack_only", action="store_true", help="Only pack zips, don't upload")
    args = parser.parse_args()

    print("=" * 50)
    print("Step 1: Pack data into zip files")
    print("=" * 50)
    pack_data(args.data_dir)

    if args.pack_only:
        print("\nPacking done. Use --repo to upload.")
        return

    print("\n" + "=" * 50)
    print("Step 2: Upload via git LFS")
    print("=" * 50)
    upload_via_git(args.repo, args.data_dir)


if __name__ == "__main__":
    main()
