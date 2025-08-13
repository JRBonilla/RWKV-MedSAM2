# scripts/build_loaders.py
# Helper script to build data loaders
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # /RWKV-MedSAM2
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
import numpy as np
    
from rwkv_medsam2.train_sam2 import load_config, setup_logger, get_data_loaders, save_data_loaders

from tqdm.auto import tqdm

# Load and override configuration for quick sanity check
config_path = "/RWKV-MedSAM2/ext/sam2/configs/sam2.1/sam2.1_vcr.yaml"
config = load_config(config_path)

# Set up logger
log_cfg = config.logging
logger = setup_logger(log_cfg)

# Build data loaders
train_loader, val_loader, test_loader = get_data_loaders(config)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# Print some stats
for split_name, loader in [
    ("TRAIN", train_loader),
    ("VAL",   val_loader),
    ("TEST",  test_loader)
]:
    n_pairs   = len(loader.dataset)  # number of pairings
    n_batches = len(loader)          # number of batches
    print(f"{split_name:5s} -> {n_pairs:5d} pairings, {n_batches:4d} batches")
    
# Save data loaders
save_data_loaders("/data/loaders32.pkl", train_loader, val_loader, test_loader)