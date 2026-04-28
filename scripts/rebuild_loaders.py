# scripts/rebuild_loaders.py
import os
import argparse

import torch  # not strictly required, but fine
from rwkv_medsam2.train_sam2 import (
    load_config,
    get_data_loaders,
    save_data_loaders,
)


def main():
    parser = argparse.ArgumentParser(description="Build and save DataLoaders for RWKV-MedSAM2.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. ext/sam2/configs/sam2.1/sam2.1_vcr.yaml)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .pkl path for saved loaders (e.g. /data/loaders32.pkl)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Build DataLoaders (this is the long bit)
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"Built loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # Save to disk
    save_data_loaders(args.out, train_loader, val_loader, test_loader)
    print(f"Saved DataLoaders to: {args.out}")


if __name__ == "__main__":
    main()
