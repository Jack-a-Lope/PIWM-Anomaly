"""
Differentiable constraint checker for training the score network.
Uses soft sigmoid mask instead of hard boolean threshold.
Only x,y constraints (θ skipped — PCA/skewness not differentiable).
"""
import torch
import torch.nn.functional as F

# Calibration constants (same as our constraint_checker.py)
CALIB_X_SLOPE = 0.01329
CALIB_X_INTERCEPT = -0.9906
CALIB_Y_SLOPE = -0.02001
CALIB_Y_INTERCEPT = 1.3958

# Soft mask temperature — lower = sharper (closer to hard mask)
# Higher = smoother gradients but less accurate
TEMPERATURE = 0.01

def soft_purple_mask(img_tensor):
    """
    Differentiable soft purple mask.
    
    img_tensor: (B, 3, H, W) float tensor in [0, 1]
    returns: (B, H, W) soft mask values in [0, 1]
    
    Hard version: mask = (b > r + 0.051) & (b > g + 0.051) & (b > 0.10)
    Soft version: sigmoid approximation of above conditions
    """
    r = img_tensor[:, 0, :, :]  # (B, H, W)
    g = img_tensor[:, 1, :, :]
    b = img_tensor[:, 2, :, :]
    
    PURPLE_BIAS = 0.051
    PURPLE_MIN = 0.10
    
    # Soft approximation of each condition
    cond1 = torch.sigmoid((b - r - PURPLE_BIAS) / TEMPERATURE)  # b > r + bias
    cond2 = torch.sigmoid((b - g - PURPLE_BIAS) / TEMPERATURE)  # b > g + bias
    cond3 = torch.sigmoid((b - PURPLE_MIN) / TEMPERATURE)        # b > min
    
    # Combine conditions (soft AND = multiply)
    mask = cond1 * cond2 * cond3  # (B, H, W)
    return mask


def soft_centroid(mask):
    """
    Differentiable centroid from soft mask.
    
    mask: (B, H, W) soft mask
    returns: cx (B,), cy (B,) pixel coordinates
    """
    B, H, W = mask.shape
    device = mask.device
    
    # Create coordinate grids
    x_coords = torch.arange(W, dtype=torch.float32, device=device)  # (W,)
    y_coords = torch.arange(H, dtype=torch.float32, device=device)  # (H,)
    
    # Expand for batch
    x_grid = x_coords.view(1, 1, W).expand(B, H, W)  # (B, H, W)
    y_grid = y_coords.view(1, H, 1).expand(B, H, W)  # (B, H, W)
    
    # Weighted average (differentiable centroid)
    total = mask.sum(dim=[1, 2]) + 1e-8  # (B,) avoid division by zero
    cx = (mask * x_grid).sum(dim=[1, 2]) / total  # (B,)
    cy = (mask * y_grid).sum(dim=[1, 2]) / total  # (B,)
    
    return cx, cy


def differentiable_constraint_loss(img_tensor, true_x, true_y, eps_x=0.05, eps_y=0.10):
    """
    Differentiable constraint violation loss for x and y.
    
    img_tensor: (B, 3, H, W) float tensor in [0, 1] — decoded image
    true_x: (B,) true x physical coordinates
    true_y: (B,) true y physical coordinates
    eps_x: x tolerance
    eps_y: y tolerance
    
    returns: scalar constraint violation loss
    """
    # Soft purple mask
    mask = soft_purple_mask(img_tensor)  # (B, H, W)
    
    # Differentiable centroid in pixels
    cx, cy = soft_centroid(mask)  # (B,), (B,)
    
    # Convert pixels to physical coordinates (our calibration)
    x_hat = CALIB_X_SLOPE * cx + CALIB_X_INTERCEPT   # (B,)
    y_hat = CALIB_Y_SLOPE * cy + CALIB_Y_INTERCEPT    # (B,)
    
    # Constraint violations
    x_err = torch.abs(x_hat - true_x)  # (B,)
    y_err = torch.abs(y_hat - true_y)  # (B,)
    
    # Hinge loss — only penalize when violation exceeds epsilon
    # (violation - epsilon)^2_+ means max(0, violation - epsilon)^2
    x_violation = F.relu(x_err - eps_x) ** 2  # (B,)
    y_violation = F.relu(y_err - eps_y) ** 2  # (B,)
    
    # Combined constraint loss
    constraint_loss = (x_violation + y_violation).mean()
    
    return constraint_loss, x_err.mean().item(), y_err.mean().item()


def test_differentiable_checker():
    """Quick test to verify gradients flow correctly."""
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    # Load a real test image
    imgs = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_imgs_100x150.npy')
    states = np.load('/blue/iruchkin/aj.byreddy/tea_lab/data/lunar/lunar_states_xy_angle.npy')
    
    # Find a good frame
    good = [i for i in range(len(states)) if states[i,1] < 1.0]
    idx = good[10]
    
    img = imgs[idx].astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)
    
    true_x = torch.tensor([states[idx, 0]], dtype=torch.float32, device=device)
    true_y = torch.tensor([states[idx, 1]], dtype=torch.float32, device=device)
    
    # Compute constraint loss
    loss, x_err, y_err = differentiable_constraint_loss(
        img_tensor, true_x, true_y, eps_x=0.05, eps_y=0.10
    )
    
    print(f"Constraint loss: {loss.item():.6f}")
    print(f"x error: {x_err:.4f}, y error: {y_err:.4f}")
    
    # Test gradient flow
    loss.backward()
    
    if img_tensor.grad is not None:
        print(f"✅ Gradients flow! Grad norm: {img_tensor.grad.norm().item():.6f}")
    else:
        print("❌ No gradients!")
    
    # Compare with hard mask
    from constraint_checker import constraint_checker
    result = constraint_checker(imgs[idx], states[idx,0], states[idx,1], states[idx,2])
    print(f"\nComparison with hard mask checker:")
    print(f"  Hard mask x_err: {result['x_err']:.4f}")
    print(f"  Soft mask x_err: {x_err:.4f}")
    print(f"  Hard mask y_err: {result['y_err']:.4f}")
    print(f"  Soft mask y_err: {y_err:.4f}")

if __name__ == "__main__":
    test_differentiable_checker()
