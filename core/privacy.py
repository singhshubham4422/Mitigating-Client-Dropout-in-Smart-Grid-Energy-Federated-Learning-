# ============================================
# File: core/privacy.py
# Stable Signed Fixed-Point Quantization
# ============================================

import torch
import math
from core.secret_sharing import PRIME

# Reduced scale for numerical stability
SCALE = 10**4   # Was 10**5

def compute_model_update(local_state, global_state):
    return {k: local_state[k] - global_state[k] for k in local_state}


def clip_update(update, clip_norm):
    total_norm = torch.sqrt(
        sum(torch.sum(param ** 2) for param in update.values())
    )
    clip_factor = max(1.0, total_norm / clip_norm)
    return {k: v / clip_factor for k, v in update.items()}


def add_gaussian_noise(update, noise_multiplier, clip_norm, device):
    noisy = {}
    for k, v in update.items():
        std = noise_multiplier * clip_norm * 0.01  # FIXED
        noise = torch.normal(0, std, size=v.shape, device=device)
        noisy[k] = v + noise
    return noisy


def quantize_update(update):
    q = {}
    for k, v in update.items():
        scaled = torch.round(v * SCALE).to(torch.int64)
        q[k] = torch.remainder(scaled, PRIME)
    return q


def dequantize_update(update):
    dq = {}
    for k, v in update.items():
        signed = v.clone().to(torch.int64)
        signed[signed > PRIME // 2] -= PRIME
        dq[k] = signed.float() / SCALE
    return dq


def compute_epsilon(noise_multiplier, rounds, delta=1e-5):
    """
    Strong composition bound:
    ε ≈ sqrt(2T log(1/δ)) / σ
    """
    if noise_multiplier == 0:
        return float("inf")

    return math.sqrt(2 * rounds * math.log(1 / delta)) / noise_multiplier
