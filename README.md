# Action-Diffusion Policy with EMA for Multimodal Manipulation

<p align="center">
  <video src="eval_results/eval_ep_17_SUCCESS_reward_1.00.mp4" width="100%" autoplay loop muted></video>
  <br>
  <em>EMA Diffusion Policy ($H=8$) successfully completing the Push-T task (IoU > 0.9).</em>
</p>

Implementation of an Action-Diffusion Policy with Exponential Moving Average (EMA) and Action Chunking for multimodal manipulation on the Push-T task. 

**Course Project for ECE 285 at UC San Diego.**

## 📌 Project Overview
Explicit Behavior Cloning often fails in complex, contact-rich environments due to "mode-averaging" when faced with multimodal expert demonstrations. This project implements a Conditional Denoising Diffusion Policy combined with **Action Chunking** (Receding Horizon Control) and an **EMA** pipeline to generate temporally consistent, high-fidelity trajectories.

* **Task:** `Push-T` (Vision-based planar manipulation)
* **Key Features:** 1D Temporal U-Net, DDPM Scheduler, EMA weight smoothing, Receding Horizon Evaluation.
* **Performance:** Achieved a peak success rate of **81.33%** with the EMA Diffusion Policy (compared to complete failure in explicit BC baselines).

## ⚙️ Installation

1. Clone this repository:
```bash
git clone https://github.com/hyomuk-kim/action-diffusion-pusht.git
cd action-diffusion-pusht
```

2. Create a conda environment and install dependencies:
```bash
conda create -n diffusion python=3.10
conda activate diffusion
pip install -r requirements.txt
```
*(Note: If you are using `lerobot` datasets or utilities, ensure you install it via `pip install git+https://github.com/huggingface/lerobot.git`)*

## 🚀 Usage

### 1. Training
The training script is unified. You can train the base Diffusion Policy or enable the EMA pipeline using the `--use_ema` flag.

```bash
# Train the Base Diffusion Model
python train.py

# Train the Diffusion Model with EMA enabled
python train.py --use_ema

# Resume training from a specific checkpoint
python train.py --use_ema --resume_path checkpoints_ema/diffusion_ema_epoch_50.pth
```

### 2. Evaluation
The evaluation script runs 50 episodes and automatically saves the first successful rollout as an MP4 video in the `eval_results/` directory.

```bash
# Evaluate the Base Model (Horizon = 8)
python eval.py --checkpoint checkpoints/diffusion_ckpt_latest.pth --horizon 8

# Evaluate the EMA Model (Horizon = 8)
python eval.py --checkpoint checkpoints_ema/diffusion_ema_ckpt_latest.pth --use_ema --horizon 8
```

## 📊 Results & Artifacts
* **Pre-trained Weights:** Due to GitHub's file size limits, model weights are hosted on Google Drive. 
  * [Download checkpoints here](https://drive.google.com/drive/folders/1eHrKflhXoysesMHpm0mSaNnwx8EtsPkr?usp=sharing)
* **Logs:** Training progression (Loss, LR) is tracked in the `logs/` directory.
* **Videos:** Evaluation videos demonstrating the agent's performance can be found in `eval_results/`.

## 👨‍💻 Author
**Hyomuk Kim**
M.S. Student in Electrical and Computer Engineering (Intelligent Systems, Robotics, and Control)
UC San Diego
