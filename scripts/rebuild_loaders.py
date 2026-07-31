# scripts/rebuild_loaders.py
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rwkv_medsam2.train_sam2 import (
    get_data_loaders,
    load_config,
    save_data_loaders,
)
from rwkv_medsam2.utils.teacher_refinement import (
    assert_disjoint_sources,
    source_paths,
)


def _validate_loader_sources(config, train_loader, val_loader, test_loader):
    expected_root = os.path.normcase(os.path.abspath(str(config.dripp.output_dir))) + os.sep
    splits = {
        "train": train_loader.dataset.sequences,
        "validation": val_loader.dataset.sequences,
        "test": test_loader.dataset.sequences,
    }
    for split_name, sequences in splits.items():
        wrong = sorted(path for path in source_paths(sequences) if not path.startswith(expected_root))
        if wrong:
            raise RuntimeError(
                f"{split_name} contains {len(wrong)} sources outside "
                f"{config.dripp.output_dir}: {wrong[:3]}"
            )
    assert_disjoint_sources(splits["train"], splits["validation"])
    print("Validated loader source root and zero train/validation image overlap.")


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

    config = load_config(args.config)
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(
        f"Built loaders: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader)}"
    )
    _validate_loader_sources(config, train_loader, val_loader, test_loader)

    output_path = os.path.abspath(args.out)
    temporary_path = output_path + ".tmp"
    try:
        save_data_loaders(temporary_path, train_loader, val_loader, test_loader)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"Saved validated DataLoaders atomically to: {output_path}")


if __name__ == "__main__":
    main()
