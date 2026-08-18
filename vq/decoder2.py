"""
Stage-2 decoder for a two-stage (VQ-)VAE world model.

Pipeline
--------
Stage 1 (frozen after pretraining):
    image  --VQ encoder-->  z  --VQ decoder-->  image

Stage 2:
    z  --extractor (your ImprovedStateTransformer)-->  s  (physical state)
    s  --StateToLatentDecoder (this file)-->           z_hat
    z_hat --stage-1 decoder-->                         image_hat

Since s has far fewer dimensions than z, the physical state alone cannot
reconstruct the image. The decoder input is therefore split into
[physical state | residual code], where the residual comes from a small
unsupervised encoder over the stage-1 latent and carries the physically
irrelevant content (background, texture, lighting).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Codebook (only needed if stage 1 is a VQ-VAE: the stage-2 output must land
# on the same codebook the stage-1 decoder was trained against)
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.beta = beta

    def forward(self, z):
        """z: (B, N, D) -> z_q: (B, N, D), vq_loss, indices: (B, N)"""
        flat = z.reshape(-1, z.shape[-1])
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        z_q = self.embedding(idx).view_as(z)

        vq_loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()          # straight-through estimator
        return z_q, vq_loss, idx.view(z.shape[:-1])


# ---------------------------------------------------------------------------
# Residual encoder: squeeze out whatever in the stage-1 latent is not
# explained by the physical state
# ---------------------------------------------------------------------------
class ResidualEncoder(nn.Module):
    def __init__(self, latent_dim=128, residual_dim=16, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, residual_dim),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Variant A -- flat latent (matches your current VAE, latent_dim=128)
# ---------------------------------------------------------------------------
class StateToLatentDecoder(nn.Module):
    """
    s (+ residual) -> z_hat, structurally mirroring ImprovedStateTransformer.

    Each physical variable becomes its own token, tagged with a learnable
    variable embedding. This keeps the attention weights readable: you can
    see which physical quantity the model leans on when reconstructing the
    latent code.
    """

    def __init__(
        self,
        state_dim=2,
        residual_dim=16,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        latent_dim=128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.residual_dim = residual_dim

        # one token per scalar physical variable
        self.value_proj = nn.Linear(1, d_model)
        self.var_embed = nn.Parameter(torch.randn(state_dim, d_model) * 0.02)

        # residual code -> a single extra token
        self.residual_proj = (
            nn.Linear(residual_dim, d_model) if residual_dim > 0 else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, latent_dim),
        )

    def forward(self, state, residual=None):
        """state: (B, state_dim), residual: (B, residual_dim) or None"""
        tokens = self.value_proj(state.unsqueeze(-1)) + self.var_embed  # (B, S, d)

        if self.residual_proj is not None and residual is not None:
            tokens = torch.cat([tokens, self.residual_proj(residual).unsqueeze(1)], dim=1)

        h = self.transformer(tokens)
        return self.readout(h.mean(dim=1))       # (B, latent_dim)


# ---------------------------------------------------------------------------
# Variant B -- spatial VQ grid (stage-1 latent shaped (B, D, H, W))
# ---------------------------------------------------------------------------
class StateToGridDecoder(nn.Module):
    """
    H*W learnable query tokens cross-attend to the state tokens, emit the
    full code map, then snap it to the codebook.
    """

    def __init__(
        self,
        state_dim=2,
        residual_dim=16,
        grid_hw=(14, 14),
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        embedding_dim=128,
        quantizer=None,
    ):
        super().__init__()
        self.grid_hw = grid_hw
        h, w = grid_hw

        self.value_proj = nn.Linear(1, d_model)
        self.var_embed = nn.Parameter(torch.randn(state_dim, d_model) * 0.02)
        self.residual_proj = (
            nn.Linear(residual_dim, d_model) if residual_dim > 0 else None
        )

        self.queries = nn.Parameter(torch.randn(h * w, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.to_code = nn.Linear(d_model, embedding_dim)
        self.quantizer = quantizer

    def forward(self, state, residual=None):
        B = state.shape[0]
        memory = self.value_proj(state.unsqueeze(-1)) + self.var_embed
        if self.residual_proj is not None and residual is not None:
            memory = torch.cat(
                [memory, self.residual_proj(residual).unsqueeze(1)], dim=1
            )

        tgt = self.queries.unsqueeze(0).expand(B, -1, -1)
        h = self.transformer(tgt, memory)
        codes = self.to_code(h)                                  # (B, HW, D)

        vq_loss = codes.new_zeros(())
        if self.quantizer is not None:
            codes, vq_loss, _ = self.quantizer(codes)

        hgt, wid = self.grid_hw
        codes = codes.transpose(1, 2).reshape(B, -1, hgt, wid)   # (B, D, H, W)
        return codes, vq_loss


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------
def stage2_decoder_loss(
    z_true,
    state_pred,
    residual,
    decoder,
    stage1_decoder=None,
    image_true=None,
    lambda_image=1.0,
    lambda_vq=1.0,
):
    """
    Latent-level reconstruction plus an optional image-level term routed
    through the frozen stage-1 decoder.

    Latent MSE alone is usually not enough: equal distances in latent space
    are not equally visible in pixel space, and the image term reweights the
    error by what actually shows up in the render.
    """
    out = decoder(state_pred, residual)
    if isinstance(out, tuple):
        z_hat, vq_loss = out
    else:
        z_hat, vq_loss = out, out.new_zeros(())

    loss = F.mse_loss(z_hat, z_true) + lambda_vq * vq_loss

    if stage1_decoder is not None and image_true is not None:
        stage1_decoder.eval()
        # gradients flow back through z_hat; stage-1 params stay frozen
        image_hat = stage1_decoder(z_hat)
        loss = loss + lambda_image * F.mse_loss(image_hat, image_true)

    return loss


if __name__ == "__main__":
    B = 8
    state = torch.randn(B, 2)
    residual = torch.randn(B, 16)

    dec_a = StateToLatentDecoder(state_dim=2, residual_dim=16, latent_dim=128)
    print("A:", dec_a(state, residual).shape)                    # (8, 128)

    vq = VectorQuantizer(num_embeddings=512, embedding_dim=128)
    dec_b = StateToGridDecoder(
        state_dim=2, residual_dim=16, grid_hw=(14, 14),
        embedding_dim=128, quantizer=vq,
    )
    codes, vq_loss = dec_b(state, residual)
    print("B:", codes.shape, vq_loss.item())                     # (8, 128, 14, 14)
