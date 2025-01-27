import os
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

# Clear any existing Hydra config
GlobalHydra.instance().clear()

@hydra.main(config_path="ext/sam2/configs/sam2.1", config_name="sam2.1_vcr.yaml")
def train(cfg):
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model
    from ext.sam2.build_sam import build_sam2
    model = build_sam2(
        config_file=cfg.model_config,
        ckpt_path=None,
        device="cuda"
    )

    # Save initial checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'config_file': cfg.model_config,
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/sam2_vcr.pt"
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()