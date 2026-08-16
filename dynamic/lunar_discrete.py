import torch
import torch.nn as nn

FPS = 50
ACTION_REPEAT = 4
TAU = ACTION_REPEAT / FPS  # 0.08

class LunarLanderDynamicsDiscrete(nn.Module):
    def __init__(self,
                 main_engine_power: float = 13.0,
                 side_engine_power: float = 0.6,
                 gravity: float = 10.0,
                 learnable: bool = False):
        super().__init__()
        
        params = dict(
            main_engine_power=main_engine_power,
            side_engine_power=side_engine_power,
            gravity=gravity
        )
        for name, val in params.items():
            t = torch.tensor(float(val))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z1: (B, 6) - lander_raw[i]   = [x, y, x_dot, y_dot, theta, theta_dot]
        z2: (B, 6) - lander_raw[i+1] = [x, y, x_dot, y_dot, theta, theta_dot]
        a:  (B,)   - action[i+1]
        Returns: predicted state at i+2 (raw physics units)
        """
        # Use z2 position and theta
        x     = z2[:, 0]
        y     = z2[:, 1]
        theta = z2[:, 4]

        # Compute velocity from position difference between z1 and z2
        x_dot     = (z2[:, 0] - z1[:, 0]) / TAU
        y_dot     = (z2[:, 1] - z1[:, 1]) / TAU
        theta_dot = (z2[:, 4] - z1[:, 4]) / TAU

        tip_x  = torch.sin(theta)
        tip_y  = torch.cos(theta)
        side_x = -torch.cos(theta)

        fire_main  = (a == 2).float()
        fire_left  = (a == 1).float()
        fire_right = (a == 3).float()

        # Main engine
        mpower = fire_main
        x_dot_new = x_dot - tip_x * self.main_engine_power * mpower * TAU
        y_dot_new = y_dot + tip_y * self.main_engine_power * mpower * TAU

        # Side engines
        spower    = fire_left + fire_right
        direction = fire_right - fire_left
        x_dot_new     = x_dot_new + side_x * self.side_engine_power * spower * direction * TAU
        theta_dot_new = theta_dot - self.side_engine_power * spower * direction * TAU

        # Gravity
        y_dot_new = y_dot_new - self.gravity * TAU

        # Position update
        x_new     = x     + x_dot_new * TAU
        y_new     = y     + y_dot_new * TAU
        theta_new = theta + theta_dot_new * TAU

        return torch.stack([x_new, y_new, x_dot_new, y_dot_new,
                            theta_new, theta_dot_new], dim=1)
