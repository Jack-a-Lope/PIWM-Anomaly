import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
sys.path.insert(0, './')
sys.path.insert(0, './LSGM')
sys.path.insert(0, './PIWM/in-conti')
from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess
from differentiable_checker import soft_purple_mask, soft_centroid, differentiable_constraint_loss
from train_lunar import IntrinsicVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Zhenjiang's 200ep VAE (frozen)
vae = IntrinsicVAE(latent_dim=128, state_dim=3).to(device)
ckpt_vae = torch.load('./checkpoints/intrinsic_vae_200ep/best.pt',
                      map_location=device, weights_only=False)
vae.load_state_dict(ckpt_vae['model_state_dict'])
vae.eval()
for param in vae.parameters():
    param.requires_grad_(False)
print(f"✅ Zhenjiang VAE loaded and frozen!")

# Calibration constants
CALIB_X_SLOPE = 0.01329
CALIB_X_INTERCEPT = -0.9906
CALIB_Y_SLOPE = -0.02001
CALIB_Y_INTERCEPT = 1.3958

class IntrinsicLatentDataset(Dataset):
    def __init__(self, z_unint_path, z_fixed_path, imgs_path):
        z_unint = np.load(z_unint_path)
        z_fixed = np.load(z_fixed_path)
        imgs = np.load(imgs_path)

        # Filter valid frames (non-zero z_unint)
        valid = np.any(z_unint != 0, axis=1)
        self.z_unint = z_unint[valid]
        self.z_fixed = z_fixed[valid]
        self.imgs = imgs[valid]
        print(f"Loaded {len(self.z_unint)} valid frames")
        print(f"z_unint: {self.z_unint.shape}")
        print(f"z_fixed: {self.z_fixed.shape}")
        print(f"imgs: {self.imgs.shape}")

    def __len__(self):
        return len(self.z_unint)

    def __getitem__(self, idx):
        # Resize image to 96x128 (Zhenjiang's VAE input size)
        from PIL import Image
        img = self.imgs[idx]  # (100, 150, 3) uint8
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((128, 96))
        img_resized = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img_resized).permute(2, 0, 1)  # (3, 96, 128)

        return (torch.FloatTensor(self.z_unint[idx]),
                torch.FloatTensor(self.z_fixed[idx]),
                img_tensor)

# Load dataset — includes original images for R(a_i)
dataset = IntrinsicLatentDataset(
    './data/lunar/intrinsic_z_unint_train_200ep.npy',
    './data/lunar/intrinsic_z_fixed_train_200ep.npy',
    './data/lunar/train_imgs.npy'
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Normalization stats
print("\nComputing normalization stats...")
z_unint_mean = dataset.z_unint.mean(axis=0)
z_unint_std = dataset.z_unint.std(axis=0)
z_unint_std = np.where(z_unint_std < 1e-8, 1.0, z_unint_std)

save_dir = './checkpoints/score_net_intrinsic_guided_v11'
os.makedirs(save_dir, exist_ok=True)
np.save(f'{save_dir}/z_unint_mean.npy', z_unint_mean)
np.save(f'{save_dir}/z_unint_std.npy', z_unint_std)

z_unint_mean_t = torch.FloatTensor(z_unint_mean).to(device)
z_unint_std_t = torch.FloatTensor(z_unint_std).to(device)

# Score network — latent_dim=125, physical_dim=3
score_net = NCSNppLinear(
    latent_dim=125,
    physical_dim=3,
    nf=128,
    ch_mult=(1, 2, 2),
    num_res_blocks=2,
    temb_dim=128
).to(device)

print(f"\nScore network parameters: {sum(p.numel() for p in score_net.parameters()):,}")

diffusion = DiffusionProcess(T=1000, device=device)
optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-4)

# Training parameters
num_epochs = 1000
lambda_c = 15.0
eps_x = 0.06
eps_y = 0.10
best_loss = float('inf')

print(f"\nTraining with constraint-guided loss (using R(a_i) — no simulator labels)...")
print(f"lambda_c={lambda_c}, eps_x={eps_x}, eps_y={eps_y}\n")

for epoch in range(1, num_epochs + 1):
    score_net.train()
    epoch_score_loss = 0
    epoch_constraint_loss = 0
    num_batches = 0

    for z_unint, z_fixed, img_orig in dataloader:
        z_unint = z_unint.to(device)
        z_fixed = z_fixed.to(device)
        img_orig = img_orig.to(device)  # (B, 3, 96, 128) original image

        # Compute R(a_i) — centroid from original image (label-free, no grad)
        with torch.no_grad():
            mask_orig = soft_purple_mask(img_orig)       # (B, H, W)
            cx_orig, cy_orig = soft_centroid(mask_orig)  # (B,), (B,)
            x_orig = CALIB_X_SLOPE * cx_orig + CALIB_X_INTERCEPT  # (B,)
            y_orig = CALIB_Y_SLOPE * cy_orig + CALIB_Y_INTERCEPT  # (B,)

        # Normalize z_unint
        z_unint_norm = (z_unint - z_unint_mean_t) / z_unint_std_t

        # Random timestep
        t = torch.randint(0, 1000, (z_unint.shape[0],)).to(device)

        # Add noise
        z_noisy, noise = diffusion.add_noise(z_unint_norm, t)

        # Score network predicts noise conditioned on z_fixed
        predicted_noise = score_net(z_noisy, t, z_fixed)

        # Score matching loss
        score_loss = nn.functional.mse_loss(predicted_noise, noise)

        # One-step denoised latent
        alpha_bar = diffusion.alpha_bars[t].to(device).view(-1, 1)
        z_denoised_norm = (z_noisy - torch.sqrt(1 - alpha_bar) * predicted_noise) / \
                          (torch.sqrt(alpha_bar) + 1e-8)

        # Unnormalize
        z_denoised = z_denoised_norm * z_unint_std_t + z_unint_mean_t

        # Decode: concat(z_fixed, z_denoised) → image
        z_combined = torch.cat([z_fixed, z_denoised], dim=1)  # (B, 128)
        img_decoded = vae.decode(z_combined)  # (B, 3, 96, 128)

        # Constraint penalty using R(a_i) — label-free!
        constraint_loss, x_err, y_err = differentiable_constraint_loss(
            img_decoded, x_orig, y_orig, eps_x=eps_x, eps_y=eps_y
        )

        # Total loss = score matching + constraint penalty
        total_loss = score_loss + lambda_c * constraint_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()

        epoch_score_loss += score_loss.item()
        epoch_constraint_loss += constraint_loss.item()
        num_batches += 1

    avg_score = epoch_score_loss / num_batches
    avg_constraint = epoch_constraint_loss / num_batches
    avg_total = avg_score + lambda_c * avg_constraint

    if avg_total < best_loss:
        best_loss = avg_total
        torch.save({
            'epoch': epoch,
            'score_net': score_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': best_loss,
            'score_loss': avg_score,
            'constraint_loss': avg_constraint,
        }, f'{save_dir}/best.pt')

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d}/{num_epochs} | "
              f"Score: {avg_score:.6f} | "
              f"Constraint: {avg_constraint:.6f} | "
              f"Total: {avg_total:.6f} | "
              f"Best: {best_loss:.6f}")

print(f"\nTraining complete! Best loss: {best_loss:.6f}")
