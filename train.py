import os
import csv
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from model import ConditionalTemporalUnet1d
from dataset import get_dataloader


def train():
    parser = argparse.ArgumentParser(description="Train Action-Diffusion Policy")
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable Exponential Moving Average (EMA) training",
    )
    args = parser.parse_args()

    # Set paths and intervals based on EMA flag
    ckpt_dir = "checkpoints_ema" if args.use_ema else "checkpoints"
    log_file = "training_progress_ema.csv" if args.use_ema else "training_progress.csv"
    latest_ckpt_name = (
        "diffusion_ema_ckpt_latest.pth" if args.use_ema else "diffusion_ckpt_latest.pth"
    )
    save_interval = 10 if args.use_ema else 5

    os.makedirs(ckpt_dir, exist_ok=True)

    # Logging initialization
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "lr"])

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(">>> Using Nvidia cuda")
    else:
        device = torch.device("cpu")
        print(">>> Using CPU")

    epochs = 200
    horizon = 16
    batch_size = 32

    # Initialize main model and optimizer
    model = ConditionalTemporalUnet1d(action_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Checkpoint loading for the main model and optimizer
    start_epoch = 0
    checkpoint = None
    if args.resume_path:
        if os.path.exists(args.resume_path):
            print(f"Loading checkpoint: {args.resume_path}")
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(
                f"Warning: Checkpoint {args.resume_path} not found. Starting from scratch."
            )

    # Initialize EMA wrapper if requested
    ema_model = None
    if args.use_ema:
        print(">>> EMA Training Enabled")
        ema_model = EMAModel(
            model.parameters(),
            decay=0.9999,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=3 / 4,
        )
        ema_model.to(device)

        # Load EMA state dict if resuming
        if args.resume_path and checkpoint is not None:
            if "ema_model_state_dict" in checkpoint:
                ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
                print("Restored EMA weights from checkpoint.")
            else:
                print(
                    ">>> Warning: No EMA weights found in checkpoint. Initializing from current model."
                )

    dataloader = get_dataloader(batch_size=batch_size, horizon=horizon)
    noise_scheduler = DDPMScheduler(num_train_timesteps=100)

    print(">>> Start Training...")
    model.train()

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        epoch_loss = 0

        for batch in pbar:
            image = batch["image"].to(device)
            state = batch["agent_pos"].to(device)
            action_0 = batch["action"].to(device)

            obs_dict = {"image": image, "agent_pos": state}
            B = action_0.shape[0]

            # Forward process: noise sampling & injection
            noise = torch.randn_like(action_0).to(device)
            timesteps = torch.randint(0, 100, (B,), device=device).long()

            # A^k generation by reparameterization trick
            noisy_actions = noise_scheduler.add_noise(action_0, noise, timesteps)

            # Prediction & loss
            noise_pred = model(noisy_actions, timesteps, obs_dict)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA Update Step
            if args.use_ema:
                ema_model.step(model.parameters())

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Avg Loss {avg_loss:.4f}")

        # Save checkpoint securely
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }

        if args.use_ema:
            checkpoint_dict["ema_model_state_dict"] = ema_model.state_dict()

        torch.save(checkpoint_dict, latest_ckpt_name)

        if epoch % save_interval == 0:
            ckpt_prefix = "diffusion_ema" if args.use_ema else "diffusion"
            torch.save(checkpoint_dict, f"{ckpt_dir}/{ckpt_prefix}_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, optimizer.param_groups[0]["lr"]])

    print(">>> Training Finished!")


if __name__ == "__main__":
    train()
