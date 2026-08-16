
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab')
sys.path.insert(0, '/blue/iruchkin/aj.byreddy/tea_lab/LSGM')

from lunar_vae import load_vae
from score_sde.ncsnpp_linear import NCSNppLinear

class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, l_0, t):
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        noise = torch.randn_like(l_0)
        l_t = torch.sqrt(alpha_bar_t) * l_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return l_t, noise

def reverse_diffusion(score_net, decoder, l_noisy, f_i, diffusion, num_steps=100):
    device = diffusion.device
    score_net.eval()
    l_t = l_noisy.clone()
    timesteps = torch.linspace(999, 0, num_steps).long()
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.expand(l_t.shape[0]).to(device)
            alpha_bar_t = diffusion.alpha_bars[t_val].to(device)
            if t_val > 0:
                alpha_bar_prev = diffusion.alpha_bars[t_val - 1].to(device)
            else:
                alpha_bar_prev = torch.tensor(1.0).to(device)
            predicted_noise = score_net(l_t, t, f_i)
            l_0_pred = (l_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_val > 0:
                noise = torch.randn_like(l_t)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                l_t = torch.sqrt(alpha_bar_prev) * l_0_pred + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise + sigma_t * noise
            else:
                l_t = l_0_pred
    return l_t

def load_everything():
    encoder, decoder, device = load_vae()

    score_net = NCSNppLinear(
        latent_dim=3, physical_dim=3,
        nf=128, ch_mult=(1, 2, 2),
        num_res_blocks=2, temb_dim=128,
    ).to(device)

    ckpt = torch.load(
        "/blue/iruchkin/aj.byreddy/tea_lab/checkpoints/score_net/best.pt",
        map_location=device, weights_only=False
    )
    score_net.load_state_dict(ckpt["score_net"])
    score_net.eval()
    print(f"✅ Score network loaded! Loss: {ckpt['loss']:.6f}")

    diffusion = DiffusionProcess(T=1000, device=device)

    imgs = np.load("/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy")
    latents = np.load("/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_latents.npy")
    states = np.load("/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy")

    indices = [100, 500, 1000, 2000, 4000]
    z_clean = torch.FloatTensor(latents[indices]).to(device)
    f_i = torch.FloatTensor(states[indices]).to(device)

    print("✅ Everything loaded!")
    return encoder, decoder, device, score_net, diffusion, imgs, latents, states, indices, z_clean, f_i
