"""
Convenience script: Download 30K subset and verify the DataLoader works.

Usage:
    # Full download (needs ~70 GB free space):
    python run_download.py

    # Without images (needs ~6 GB free space):
    python run_download.py --skip_images

    # After downloading images separately:
    python run_download.py --images_dir /path/to/images
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_subset import main as download_main
from dataloader import get_grasp_dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--test_loader", action="store_true", default=True)
    args = parser.parse_args()

    # Inject args into sys.argv for download_subset
    sys.argv = [
        "download_subset.py",
        "--num_samples", str(args.num_samples),
        "--data_dir", args.data_dir,
    ]
    if args.skip_images:
        sys.argv.append("--skip_images")

    # Run download
    download_main()

    # Test DataLoader
    if args.test_loader:
        print("\n" + "=" * 60)
        print("TESTING DATALOADER")
        print("=" * 60)
        try:
            train_loader, val_loader, test_loader = get_grasp_dataloader(
                data_dir=args.data_dir,
                batch_size=4,
                num_workers=0,
                load_images=not args.skip_images,
            )
            batch = next(iter(train_loader))
            print(f"\n  Batch keys: {list(batch.keys())}")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Instructions: {batch['instruction'][:2]}")
            if isinstance(batch["positive_label"], list):
                print(f"  Label shapes: {[l.shape for l in batch['positive_label'][:2]]}")
            else:
                print(f"  Label shape: {batch['positive_label'].shape}")
            print("\n  DataLoader is working correctly!")
        except Exception as e:
            print(f"\n  DataLoader test failed: {e}")
            raise
