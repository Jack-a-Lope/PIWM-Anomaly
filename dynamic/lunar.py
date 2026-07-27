import torch
import torch.nn as nn
import numpy as np

# Defaults (rough values inspired by the real Gym env's constants)
TAU = 1.0 / 50.0          # FPS = 50
GRAVITY = 10.0             # magnitude, acts in -y
MOMENT_OF_INERTIA = 1.0    # lumps mass + lander shape into one rotational constant

#need to be learned
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6


class LunarLanderDynamics(nn.Module):
    def __init__(self, tau: float = TAU,
                 gravity: float = GRAVITY,
                 main_engine_power: float = MAIN_ENGINE_POWER,
                 side_engine_power: float = SIDE_ENGINE_POWER,
                 moment_of_inertia: float = MOMENT_OF_INERTIA,
                 learnable: bool = False):
        super().__init__()
        self.tau = tau

        params = dict(gravity=gravity,
                      main_engine_power=main_engine_power,
                      side_engine_power=side_engine_power,
                      moment_of_inertia=moment_of_inertia)
        for name, val in params.items():
            t = torch.tensor(float(val))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x, y, x_dot, y_dot, theta, theta_dot = z.unbind(dim=1)
        main_throttle, side_throttle = a[:, 0], a[:, 1]

        # Main engine thrust acts along the body's -y axis (already mass-normalized,
        # i.e. this is an acceleration, mirroring how CartPole's force_mag/mass
        # combination ultimately produces an acceleration term)
        thrust = self.main_engine_power * torch.clamp(main_throttle, 0.0, 1.0)
        ax_thrust = -thrust * torch.sin(theta)
        ay_thrust = thrust * torch.cos(theta)

        # Side engines: lateral push + torque
        side_force = self.side_engine_power * side_throttle
        ax_side = side_force * torch.cos(theta)
        ay_side = side_force * torch.sin(theta)
        torque = side_force  # simplified: torque proportional to side force

        x_acc = ax_thrust + ax_side
        y_acc = ay_thrust + ay_side - self.gravity
        theta_acc = torque / self.moment_of_inertia

        x_next = x + self.tau * x_dot
        y_next = y + self.tau * y_dot
        x_dot_next = x_dot + self.tau * x_acc
        y_dot_next = y_dot + self.tau * y_acc
        theta_next = theta + self.tau * theta_dot
        theta_dot_next = theta_dot + self.tau * theta_acc

        return torch.stack([x_next, y_next, x_dot_next, y_dot_next,
                             theta_next, theta_dot_next], dim=1)
