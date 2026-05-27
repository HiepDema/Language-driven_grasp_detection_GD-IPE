"""
Download a 30,000-sample subset from Grasp-Anything + Grasp-Anything-pp datasets.
Files are matched by SHA-256 filename across both datasets.

Grasp-Anything (images): https://huggingface.co/datasets/airvlab/Grasp-Anything
Grasp-Anything-pp (instructions + labels): https://huggingface.co/datasets/airvlab/Grasp-Anything-pp

Usage:
    python download_subset.py --num_samples 30000 --data_dir ./data
"""

import os
import io
import zipfile
import pickle
import argparse
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import hf_hub_url, hf_hub_download
from tqdm import tqdm


HF_REPO_IMAGES = "airvlab/Grasp-Anything"
HF_REPO_PP = "airvlab/Grasp-Anything-pp"

INSTRUCTIONS_ZIP = "grasp_instructions.zip"
POSITIVE_LABEL_ZIP = "grasp_label_positive.zip"
IMAGE_PART_AA = "image_part_aa"
IMAGE_PART_AB = "image_part_ab"


def get_hf_download_url(repo_id, filename, repo_type="dataset"):
    return hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] {dest_path} already exists")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dest_path)
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
            pbar.update(len(chunk))


def download_with_hf_hub(repo_id, filename, local_dir, repo_type="dataset"):
    """Download using huggingface_hub (handles caching, resume)."""
    print(f"  Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=local_dir,
    )
    print(f"  -> Saved to {path}")
    return path


def extract_subset_from_zip(zip_path, output_dir, target_shas=None, max_files=None):
    """Extract files from zip, optionally filtering by SHA names."""
    os.makedirs(output_dir, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for name in tqdm(names, desc=f"Extracting {os.path.basename(zip_path)}"):
            if name.endswith("/"):
                continue
            basename = os.path.basename(name)
            sha = os.path.splitext(basename)[0]
            if target_shas is not None and sha not in target_shas:
                continue
            out_path = os.path.join(output_dir, basename)
            if not os.path.exists(out_path):
                data = zf.read(name)
                with open(out_path, "wb") as f:
                    f.write(data)
            extracted.append(sha)
            if max_files and len(extracted) >= max_files:
                break
    return extracted


def get_shas_from_zip(zip_path, extension=None):
    """List all SHA filenames inside a zip."""
    shas = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            basename = os.path.basename(name)
            sha = os.path.splitext(basename)[0]
            if extension is None or basename.endswith(extension):
                shas.append(sha)
    return shas


def main():
    parser = argparse.ArgumentParser(description="Download Grasp-Anything subset")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--skip_images", action="store_true",
                        help="Skip image download (useful if you already have images)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Step 1: Download grasp_instructions.zip from Grasp-Anything-pp
    # This is the smallest file (1.54 GB) and gives us valid SHAs
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 1: Download grasp_instructions.zip (1.54 GB)")
    print("=" * 60)
    instructions_zip_path = str(raw_dir / INSTRUCTIONS_ZIP)
    if not os.path.exists(instructions_zip_path):
        download_with_hf_hub(HF_REPO_PP, INSTRUCTIONS_ZIP, str(raw_dir))
    else:
        print(f"  [SKIP] {instructions_zip_path} already exists")

    # =========================================================
    # Step 2: Get list of SHAs and select subset
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 2: Select SHA subset")
    print("=" * 60)
    all_shas = get_shas_from_zip(instructions_zip_path, extension=".pkl")
    print(f"  Total samples available: {len(all_shas)}")

    subset_shas = set(all_shas[: args.num_samples])
    print(f"  Selected subset: {len(subset_shas)} samples")

    # Save the selected SHAs for reference
    sha_list_path = data_dir / "selected_shas.txt"
    with open(sha_list_path, "w") as f:
        for sha in sorted(subset_shas):
            f.write(sha + "\n")
    print(f"  SHA list saved to {sha_list_path}")

    # =========================================================
    # Step 3: Extract instructions for subset
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Extract grasp instructions")
    print("=" * 60)
    instructions_dir = data_dir / "grasp_instructions"
    extracted = extract_subset_from_zip(
        instructions_zip_path, str(instructions_dir), target_shas=subset_shas
    )
    print(f"  Extracted {len(extracted)} instruction files")

    # =========================================================
    # Step 4: Download and extract positive labels from Grasp-Anything-pp
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: Download grasp_label_positive.zip from pp (3.95 GB)")
    print("=" * 60)
    positive_zip_path = str(raw_dir / POSITIVE_LABEL_ZIP)
    if not os.path.exists(positive_zip_path):
        download_with_hf_hub(HF_REPO_PP, POSITIVE_LABEL_ZIP, str(raw_dir))
    else:
        print(f"  [SKIP] {positive_zip_path} already exists")

    labels_dir = data_dir / "grasp_label_positive"
    extracted_labels = extract_subset_from_zip(
        positive_zip_path, str(labels_dir), target_shas=subset_shas
    )
    print(f"  Extracted {len(extracted_labels)} label files")

    # =========================================================
    # Step 5: Download and extract images from Grasp-Anything
    # =========================================================
    if not args.skip_images:
        print("\n" + "=" * 60)
        print("STEP 5: Download images (requires ~65 GB download)")
        print("  image_part_aa (34.4 GB) + image_part_ab (30.7 GB)")
        print("=" * 60)

        part_aa_path = str(raw_dir / IMAGE_PART_AA)
        part_ab_path = str(raw_dir / IMAGE_PART_AB)
        image_zip_path = str(raw_dir / "image.zip")

        # Download image parts
        if not os.path.exists(part_aa_path):
            download_with_hf_hub(HF_REPO_IMAGES, IMAGE_PART_AA, str(raw_dir))
        else:
            print(f"  [SKIP] {part_aa_path} already exists")

        if not os.path.exists(part_ab_path):
            download_with_hf_hub(HF_REPO_IMAGES, IMAGE_PART_AB, str(raw_dir))
        else:
            print(f"  [SKIP] {part_ab_path} already exists")

        # Concatenate parts into image.zip
        if not os.path.exists(image_zip_path):
            print("  Concatenating image parts...")
            with open(image_zip_path, "wb") as outf:
                for part in [part_aa_path, part_ab_path]:
                    with open(part, "rb") as inf:
                        while True:
                            chunk = inf.read(8192 * 1024)
                            if not chunk:
                                break
                            outf.write(chunk)
            print(f"  -> Created {image_zip_path}")
        else:
            print(f"  [SKIP] {image_zip_path} already exists")

        # Extract subset of images
        images_dir = data_dir / "images"
        extracted_imgs = extract_subset_from_zip(
            image_zip_path, str(images_dir), target_shas=subset_shas
        )
        print(f"  Extracted {len(extracted_imgs)} images")
    else:
        print("\n[SKIP] Image download skipped (--skip_images)")

    # =========================================================
    # Step 6: Verify alignment
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 6: Verify data alignment")
    print("=" * 60)

    instructions_dir = data_dir / "grasp_instructions"
    labels_dir = data_dir / "grasp_label_positive"
    images_dir = data_dir / "images"

    instr_shas = {f.stem for f in instructions_dir.glob("*.pkl")} if instructions_dir.exists() else set()
    label_shas = {f.stem for f in labels_dir.glob("*.pt")} if labels_dir.exists() else set()
    image_shas = {f.stem for f in images_dir.glob("*.jpg")} if images_dir.exists() else set()

    if image_shas:
        common = instr_shas & label_shas & image_shas
    else:
        common = instr_shas & label_shas

    print(f"  Instructions: {len(instr_shas)}")
    print(f"  Labels: {len(label_shas)}")
    print(f"  Images: {len(image_shas)}")
    print(f"  Common (matched): {len(common)}")

    # Save final matched list
    matched_path = data_dir / "matched_shas.txt"
    with open(matched_path, "w") as f:
        for sha in sorted(common):
            f.write(sha + "\n")
    print(f"  Matched SHA list saved to {matched_path}")
    print(f"\n{'=' * 60}")
    print(f"DONE! Dataset ready at: {data_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
