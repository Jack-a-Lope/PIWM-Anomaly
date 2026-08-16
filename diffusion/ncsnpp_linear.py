
# NCSN++ adapted for 3-dim flat latent space
# Replaces Conv2d with Linear layers
# Keeps ResNet block structure and timestep embedding

import torch
import torch.nn as nn
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding — same as NCSN++"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResnetBlockLinear(nn.Module):
    """
    ResNet block with Linear layers instead of Conv2d.
    Keeps the skip connection structure from NCSN++.
    """
    def __init__(self, in_dim, out_dim, temb_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.temb_proj = nn.Linear(temb_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.linear1(h)
        h = h + self.temb_proj(self.act(temb))
        h = self.act(self.norm2(h))
        h = self.linear2(h)
        return h + self.skip(x)


class NCSNppLinear(nn.Module):
    """
    NCSN++ adapted for 3-dim flat latent space.
    Replaces Conv2d with Linear, keeps ResNet+skip connection structure.

    Input:  l_t (batch, 3) + t (batch,) + f_i (batch, 3)
    Output: predicted noise (batch, 3)
    """
    def __init__(
        self,
        latent_dim=3,
        physical_dim=3,
        nf=128,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        temb_dim=128,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        temb_hidden = nf * 4

        # Timestep embedding
        self.temb = TimestepEmbedding(temb_dim)
        self.temb_proj = nn.Sequential(
            nn.Linear(temb_dim, temb_hidden),
            nn.SiLU(),
            nn.Linear(temb_hidden, temb_hidden),
        )

        # Physical vector embedding
        self.f_proj = nn.Sequential(
            nn.Linear(physical_dim, temb_hidden),
            nn.SiLU(),
            nn.Linear(temb_hidden, temb_hidden),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim, nf)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_dims = []

        in_dim = nf
        for i, mult in enumerate(ch_mult):
            out_dim = nf * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlockLinear(in_dim, out_dim, temb_hidden))
                in_dim = out_dim
            self.down_dims.append(in_dim)

        # Middle block
        self.mid_block1 = ResnetBlockLinear(in_dim, in_dim, temb_hidden)
        self.mid_block2 = ResnetBlockLinear(in_dim, in_dim, temb_hidden)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_dim = nf * mult
            skip_dim = self.down_dims[i]
            for j in range(num_res_blocks):
                self.up_blocks.append(
                    ResnetBlockLinear(in_dim + (skip_dim if j == 0 else 0), out_dim, temb_hidden)
                )
                in_dim = out_dim

        # Output
        self.output_norm = nn.LayerNorm(in_dim)
        self.output_proj = nn.Linear(in_dim, latent_dim)
        self.act = nn.SiLU()

    def forward(self, l_t, t, f_i):
        # Timestep embedding
        temb = self.temb(t)
        temb = self.temb_proj(temb)

        # Physical conditioning — add to timestep embedding
        f_emb = self.f_proj(f_i)
        temb = temb + f_emb

        # Input projection
        h = self.input_proj(l_t)

        # Downsampling — save skips
        skips = []
        block_idx = 0
        for i, mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, temb)
                block_idx += 1
            skips.append(h)

        # Middle
        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)

        # Upsampling — use skips
        block_idx = 0
        for i, mult in reversed(list(enumerate(self.ch_mult))):
            skip = skips.pop()
            for j in range(self.num_res_blocks):
                if j == 0:
                    h = torch.cat([h, skip], dim=-1)
                h = self.up_blocks[block_idx](h, temb)
                block_idx += 1

        # Output
        h = self.act(self.output_norm(h))
        return self.output_proj(h)
