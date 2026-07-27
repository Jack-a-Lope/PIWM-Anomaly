import torch
import torch.nn as nn
import numpy as np

# Defaults
GRAVITY = 9.8
TAU = 0.02
# these 4 need to be learned
FORCE_MAG = 10.0
MASS_CART = 1.0
MASS_POLE = 0.1
LENGTH = 0.5 


class CartPoleDynamics(nn.Module):
    """
    Known cartpole model, one-step update.
    force_mag, mass_cart, mass_pole, length are learned/settable constants.

    state z: (B,4) = [x, x_dot, theta, theta_dot]
    action a: (B,1) or (B,) in {-1, +1} (or continuous force if you want)
    """
    def __init__(self, tau: float = TAU,
                 force_mag: float = FORCE_MAG,
                 mass_cart: float = MASS_CART,
                 mass_pole: float = MASS_POLE,
                 length: float = LENGTH,
                 learnable: bool = False):
        super().__init__()
        self.tau = tau
        self.gravity = GRAVITY

        params = dict(force_mag=force_mag, mass_cart=mass_cart,
                      mass_pole=mass_pole, length=length)
        for name, val in params.items():
            t = torch.tensor(float(val))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B,4) = [x, x_dot, theta, theta_dot]
        a: (B,) or (B,1), sign gives direction of force (or pass continuous force directly)
        """
        x, x_dot, theta, theta_dot = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        a = a.squeeze(-1) if a.dim() > 1 else a

        total_mass = self.mass_cart + self.mass_pole
        polemass_length = self.mass_pole * self.length

        force = self.force_mag * torch.sign(a)  # or: force = a  (continuous)

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x_next = x + self.tau * x_dot
        x_dot_next = x_dot + self.tau * xacc
        theta_next = theta + self.tau * theta_dot
        theta_dot_next = theta_dot + self.tau * thetaacc

        return torch.stack([x_next, x_dot_next, theta_next, theta_dot_next], dim=1)
