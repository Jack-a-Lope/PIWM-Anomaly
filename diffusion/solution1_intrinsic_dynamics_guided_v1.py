import sys
import torch
import numpy as np
import glob
from tqdm import tqdm

sys.path.insert(0, './')
sys.path.insert(0, './LSGM')
sys.path.insert(0, './PIWM/in-conti')
sys.path.insert(0, './PIWM/dynamic')

from score_sde.ncsnpp_linear import NCSNppLinear
from setup import DiffusionProcess
from constraint_checker import constraint_checker
from differentiable_checker import differentiable_constraint_loss
from train_lunar import IntrinsicVAE
from lunar_discrete import LunarLanderDynamicsDiscrete

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------
# v29 DESIGN
#   Same as v28 but uses TWO consecutive states for dynamics:
#   lander_raw[i] + lander_raw[i+1] + action[i+1] → predicted i+2
#   Velocity computed from position difference (PIWM paper approach)
# ---------------------------------------------------------------
GUIDANCE_EPS_X = 0.00
GUIDANCE_EPS_Y = 0.00

# Conversion constants raw → normalized
X_SCALE = 0.1
X_OFFSET = -1.0
Y_SCALE = 0.15
Y_OFFSET = -0.59

def raw_to_normalized(x_raw, y_raw):
    return X_SCALE * x_raw + X_OFFSET, Y_SCALE * y_raw + Y_OFFSET

# Load VAE
vae = IntrinsicVAE(latent_dim=128, state_dim=3).to(device)
ckpt_vae = torch.load('./checkpoints/intrinsic_vae_200ep/best.pt',
                      map_location=device, weights_only=False)
vae.load_state_dict(ckpt_vae['model_state_dict'])
vae.eval()
print("VAE loaded!")

# Load score network
score_net = NCSNppLinear(
    latent_dim=125, physical_dim=3,
    nf=128, ch_mult=(1, 2, 2),
    num_res_blocks=2, temb_dim=128
).to(device)
ckpt = torch.load('./checkpoints/score_net_intrinsic_guided_v11/best.pt',
                  map_location=device, weights_only=False)
score_net.load_state_dict(ckpt['score_net'])
score_net.eval()
print(f"Score network loaded! Epoch {ckpt['epoch']}, Loss {ckpt['loss']:.6f}")

# Load normalization stats
z_unint_mean = np.load('./checkpoints/score_net_intrinsic_guided_v11/z_unint_mean.npy')
z_unint_std  = np.load('./checkpoints/score_net_intrinsic_guided_v11/z_unint_std.npy')
z_unint_mean_t = torch.FloatTensor(z_unint_mean).to(device)
z_unint_std_t  = torch.FloatTensor(z_unint_std).to(device)

# Load dynamics v2 (two state input)
dynamics = LunarLanderDynamicsDiscrete(learnable=False).to(device)
ckpt_dyn = torch.load('./checkpoints/dynamics_discrete/best.pt',
                      map_location=device)
dynamics.load_state_dict(ckpt_dyn['model_state_dict'])
dynamics.eval()
print(f"Dynamics v2 loaded! main_power={ckpt_dyn['main_engine_power']:.4f}, side_power={ckpt_dyn['side_engine_power']:.4f}")

diffusion = DiffusionProcess(T=1000, device=device)

# Load test data from shard_0029
test_dir = './data/lunar/lunar_discrete/train/shard_0029'
test_files = sorted(glob.glob(f"{test_dir}/*.npz"))[:200]
print(f"Loaded {len(test_files)} test trajectories from shard_0029")

def reverse_diffusion_guided(score_net, z_unint_noisy_norm, z_fixed,
                             diffusion, vae, z_unint_mean_t, z_unint_std_t,
                             true_x, true_y, start_t=20, num_steps=100,
                             guidance_scale=1.2,
                             guidance_eps_x=0.00, guidance_eps_y=0.00):
    score_net.eval()
    l_t = z_unint_noisy_norm.clone()
    timesteps = torch.linspace(start_t, 0, num_steps).long()

    for t_val in timesteps:
        t = t_val.expand(l_t.shape[0]).to(device)
        alpha_bar_t    = diffusion.alpha_bars[t_val].to(device)
        alpha_bar_prev = diffusion.alpha_bars[t_val - 1].to(device) if t_val > 0 else torch.tensor(1.0).to(device)

        with torch.no_grad():
            predicted_noise = score_net(l_t, t, z_fixed)
            l_0_pred = (l_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_val > 0:
                noise   = torch.randn_like(l_t)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                l_t_prev = torch.sqrt(alpha_bar_prev) * l_0_pred + \
                           torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * predicted_noise + \
                           sigma_t * noise
            else:
                l_t_prev = l_0_pred

        l_t_prev = l_t_prev.detach().requires_grad_(True)
        z_denoised = l_t_prev * z_unint_std_t + z_unint_mean_t
        z_combined = torch.cat([z_fixed, z_denoised], dim=1)
        img_decoded = vae.decode(z_combined)

        constraint_loss, x_err, y_err = differentiable_constraint_loss(
            img_decoded, true_x, true_y,
            eps_x=guidance_eps_x, eps_y=guidance_eps_y
        )

        if constraint_loss > 0:
            grad = torch.autograd.grad(constraint_loss, l_t_prev)[0]
            l_t = (l_t_prev - guidance_scale * grad).detach()
        else:
            l_t = l_t_prev.detach()

    return l_t

# EVAL tolerances
eps_x = 0.06
eps_y = 0.10
t_level = 20
max_iterations = 10
guidance_scale = 1.2
BAD_X_THRESH = 0.20
BAD_Y_THRESH = 0.30

results = []
skipped = 0

print(f"\nRunning v29 (two-state dynamics guidance)...")
print(f"guidance_scale={guidance_scale}, t_level={t_level}, max_iterations={max_iterations}")

for traj_file in tqdm(test_files):
    d = np.load(traj_file, allow_pickle=False)
    images      = d['image']
    lander_raws = d['lander_raw']
    actions     = d['action']

    # need i, i+1, i+2 so iterate up to len(actions)-1
    for i in range(len(actions) - 1):

        # Step 1 — dynamics with two consecutive states
        state_i   = torch.tensor(lander_raws[i],   dtype=torch.float32).unsqueeze(0).to(device)
        state_i1  = torch.tensor(lander_raws[i+1], dtype=torch.float32).unsqueeze(0).to(device)
        action_i1 = torch.tensor([actions[i+1]],   dtype=torch.long).to(device)

        with torch.no_grad():
            pred_next = dynamics(state_i, state_i1, action_i1)

        pred_x_norm, pred_y_norm = raw_to_normalized(
            pred_next[0, 0].item(), pred_next[0, 1].item()
        )

        true_x_t = torch.tensor([pred_x_norm], dtype=torch.float32, device=device)
        true_y_t = torch.tensor([pred_y_norm], dtype=torch.float32, device=device)

        # Step 2 — encode image[i] with VAE
        img = images[i].astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            h  = vae.encoder(img_tensor)
            h  = h.reshape(h.size(0), -1)
            mu = vae.fc_mu(h)

        z_fixed = mu[:, :3]
        z_unint = mu[:, 3:]

        with torch.no_grad():
            z_combined = torch.cat([z_fixed, z_unint], dim=1)
            recon = vae.decode(z_combined)
        recon_img = recon[0].permute(1, 2, 0).cpu().numpy()

        result_init = constraint_checker(recon_img, pred_x_norm, pred_y_norm, 0.0)
        if not result_init['visible']:
            skipped += 1
            continue
        if result_init['x_err'] > BAD_X_THRESH or result_init['y_err'] > BAD_Y_THRESH:
            skipped += 1
            continue

        frame_history = []

        for iteration in range(max_iterations + 1):
            with torch.no_grad():
                z_combined = torch.cat([z_fixed, z_unint], dim=1)
                recon = vae.decode(z_combined)
            recon_img = recon[0].permute(1, 2, 0).cpu().numpy()

            result = constraint_checker(recon_img, pred_x_norm, pred_y_norm, 0.0)

            x_err = result['x_err'] if result['visible'] else None
            y_err = result['y_err'] if result['visible'] else None

            x_ok = x_err is not None and x_err <= eps_x
            y_ok = y_err is not None and y_err <= eps_y

            frame_history.append({
                'iteration': iteration,
                'x_err': x_err, 'y_err': y_err,
                'x_ok': x_ok, 'y_ok': y_ok,
                'xy_ok': x_ok and y_ok,
            })

            if iteration == max_iterations:
                break
            if x_ok and y_ok:
                break

            z_unint_norm = (z_unint - z_unint_mean_t) / z_unint_std_t
            t_batch = torch.tensor([t_level]).to(device)
            l_noisy_norm, _ = diffusion.add_noise(z_unint_norm, t_batch)

            l_corr_norm = reverse_diffusion_guided(
                score_net, l_noisy_norm, z_fixed,
                diffusion, vae, z_unint_mean_t, z_unint_std_t,
                true_x_t, true_y_t,
                start_t=t_level, num_steps=100,
                guidance_scale=guidance_scale,
                guidance_eps_x=GUIDANCE_EPS_X, guidance_eps_y=GUIDANCE_EPS_Y
            )
            z_unint = l_corr_norm * z_unint_std_t + z_unint_mean_t

        while len(frame_history) < max_iterations + 1:
            padded = dict(frame_history[-1])
            padded['iteration'] = len(frame_history)
            frame_history.append(padded)

        results.append({'history': frame_history})

print(f"\nSkipped: {skipped}, Evaluated: {len(results)}")
np.save('./results_solution1_intrinsic_guided_v29.npy', results)
print("Results saved!")

print("\n" + "=" * 65)
print("v29 — two-state dynamics guidance")
print("=" * 65)
for it in range(max_iterations + 1):
    x_errs = [r['history'][it]['x_err'] for r in results if r['history'][it]['x_err'] is not None]
    y_errs = [r['history'][it]['y_err'] for r in results if r['history'][it]['y_err'] is not None]
    xy_ok  = sum(r['history'][it]['xy_ok'] for r in results)
    print(f"\nIteration {it} ({len(results)} frames):")
    if x_errs: print(f"  x error: mean={np.mean(x_errs):.4f} median={np.median(x_errs):.4f}")
    if y_errs: print(f"  y error: mean={np.mean(y_errs):.4f} median={np.median(y_errs):.4f}")
    print(f"  x,y satisfied: {xy_ok}/{len(results)} ({100*xy_ok/len(results):.1f}%)")
