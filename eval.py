import os
import argparse
import torch
import gymnasium as gym
import gym_pusht
import numpy as np
import cv2
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from model import ConditionalTemporalUnet1d
from dataset import PushTDiffusionDataset


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Action-Diffusion Policy")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Evaluate using EMA weights"
    )
    parser.add_argument(
        "--horizon", type=int, default=8, help="Action execution horizon (H)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_tests", type=int, default=50, help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    dataset = PushTDiffusionDataset()
    stats = dataset.stats

    # Load Model
    model = ConditionalTemporalUnet1d(action_dim=2).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.use_ema:
        print(f">>> Loading EMA weights from {args.checkpoint}...")
        model.load_state_dict(checkpoint["model_state_dict"])
        ema_model = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True)
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        ema_model.copy_to(model.parameters())
    else:
        print(f">>> Loading Standard weights from {args.checkpoint}...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    noise_scheduler = DDPMScheduler(num_train_timesteps=100)

    success_count = 0
    success_steps = []
    video_saved = False

    res_dir = "eval_results"
    os.makedirs(res_dir, exist_ok=True)

    model_type = "EMA" if args.use_ema else "BASE"
    print(f"\n========================================")
    print(f" Evaluating {model_type} Model")
    print(f" Horizon: {args.horizon} | Seed: {args.seed} | Episodes: {args.num_tests}")
    print(f"========================================")

    for i in tqdm(range(args.num_tests), desc="Evaluating"):
        obs, _ = env.reset(seed=args.seed * 100 + i)
        done = False
        step_count = 0
        frames = []
        max_reward = 0.0
        first_success_step = -1

        while not done and step_count < 300:
            img = torch.from_numpy(obs["pixels"]).moveaxis(-1, 0).float() / 255.0
            img = dataset.normalize(img).unsqueeze(0).to(device)

            s_min = torch.tensor(stats["observation.state"]["min"], device=device)
            s_max = torch.tensor(stats["observation.state"]["max"], device=device)
            agent_pos_tensor = torch.from_numpy(obs["agent_pos"]).to(device)

            state = (agent_pos_tensor - s_min) / (s_max - s_min + 1e-8) * 2 - 1
            state = state.unsqueeze(0).float()

            obs_dict = {"image": img, "agent_pos": state}

            noisy_action = torch.randn((1, 16, 2), device=device)
            noise_scheduler.set_timesteps(100)

            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    noise_pred = model(
                        noisy_action, t.unsqueeze(0).to(device), obs_dict
                    )
                    noisy_action = noise_scheduler.step(
                        noise_pred, t, noisy_action
                    ).prev_sample

            pred_actions = noisy_action.squeeze(0).cpu().numpy()
            a_min = np.array(stats["action"]["min"])
            a_max = np.array(stats["action"]["max"])
            unnorm_actions = (pred_actions + 1) / 2 * (a_max - a_min + 1e-8) + a_min

            for k in range(args.horizon):
                obs, reward, terminated, truncated, _ = env.step(unnorm_actions[k])

                if not video_saved:
                    frames.append(obs["pixels"])

                step_count += 1

                if reward > max_reward:
                    max_reward = reward

                if reward > 0.9 and first_success_step == -1:
                    first_success_step = step_count
                    done = True
                    break

                if terminated or truncated:
                    done = True
                    break

        is_success = max_reward > 0.9

        if is_success:
            success_count += 1
            success_steps.append(first_success_step)

            if not video_saved:
                filename = f"success_video_{model_type.lower()}_H{args.horizon}_seed{args.seed}.mp4"
                filepath = os.path.join(res_dir, filename)
                out = cv2.VideoWriter(
                    filepath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96)
                )
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                video_saved = True

    sr = (success_count / args.num_tests) * 100
    avg_s = np.mean(success_steps) if success_steps else 0.0

    print(f"\n>>> FINAL RESULTS: {model_type} Model | Horizon={args.horizon}")
    print(f">>> Success Rate: {sr:.2f}% ({success_count}/{args.num_tests})")
    print(f">>> Avg Steps to Success: {avg_s:.1f}")
    if video_saved:
        print(f">>> Best video saved in '{res_dir}' folder.")


if __name__ == "__main__":
    evaluate()
