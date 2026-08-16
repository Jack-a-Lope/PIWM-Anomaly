# Diffusion-based Physical Constraint Correction for PIWM

## Required Checkpoints (download from TEA Lab OneDrive - PIWM folder)
- `checkpoints/intrinsic_vae_200ep/best.pt` — Intrinsic VAE
- `checkpoints/score_net_intrinsic_guided_v11/best.pt` — Score network
- `checkpoints/score_net_intrinsic_guided_v11/z_unint_mean.npy`
- `checkpoints/score_net_intrinsic_guided_v11/z_unint_std.npy`
- `checkpoints/dynamics_discrete/best.pt` — Discrete dynamics model

## Required Data (download from TEA Lab OneDrive - PIWM folder)
- `data/lunar/lunar_discrete/train/` — 30k discrete trajectories (96x128 RGB)
- `data/lunar/lunar_test_intrinsic.npz` — Test data for ground truth inference

## Required Repos
- LSGM repo cloned at `../LSGM`
- PIWM/in-conti for `train_lunar.py` (zj's IntrinsicVAE)

## Scripts
- `train_score_net.py` — Train score network
- `solution1_intrinsic_gt_guided.py` — Inference with ground truth (99.2%)
- `solution1_intrinsic_dynamics_guided_v1.py` — Inference with dynamics (93.8%)
- `constraint_checker.py` — Physical constraint checker
- `differentiable_checker.py` — Differentiable constraint for guidance

## Dynamic folder
- `lunar_discrete.py` — Discrete dynamics model
- `train_dynamics_discrete.py` — Train dynamics model
- `collect_discrete.py` — Collect discrete trajectories
