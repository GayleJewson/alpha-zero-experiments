"""
AlphaZero neural network.

Architecture (ResNet with dual heads):
  Input:  (C, H, W) board encoding — shape determined by the game.
  Stem:   Conv(C→64, 3×3, pad=1) → BatchNorm → ReLU
  Trunk:  N × ResBlock(64→64)
  Heads:
    Policy: Conv(64→2, 1×1) → BN → ReLU → Flatten → FC → action_size → Softmax
    Value:  Conv(64→1, 1×1) → BN → ReLU → Flatten → FC(64) → ReLU → FC(1) → Tanh

The network is game-agnostic: it adapts to any input shape and action-space
size via the constructor arguments derived from the game interface.

Key design choices:
  - Separate policy and value heads (standard AlphaZero).
  - BatchNorm throughout for stable training.
  - Value head outputs a scalar in [-1, 1] (Tanh), matching the ±1 game outcomes.
  - Policy head outputs logits (training uses cross-entropy; predict() softmaxes).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResBlock(nn.Module):
    """Standard pre-activation residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNetwork(nn.Module):
    """
    Game-agnostic AlphaZero network.

    Construct once per game:
        net = AlphaZeroNetwork(
            input_shape=game.get_state_representation(game.get_init_state()).shape,
            action_size=game.get_action_size(),
            num_res_blocks=6,
            num_channels=128,
        )
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        action_size: int,
        num_res_blocks: int = 6,
        num_channels: int = 128,
    ):
        """
        Args:
            input_shape:    Shape of the state tensor, e.g. (6, 8, 8).
                            Must be 3-D (C, H, W) for convolutional trunk.
            action_size:    Size of the flat action space.
            num_res_blocks: Depth of the residual tower.
            num_channels:   Width of the hidden representation.
        """
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be (C, H, W); got {input_shape}. "
                "If your game uses a flat state, reshape it to (1, 1, N)."
            )

        in_channels, h, w = input_shape
        flat_size = num_channels * h * w

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_tower = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * h * w, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(h * w, 64)
        self.value_fc2  = nn.Linear(64, 1)

        # Store metadata
        self.input_shape = input_shape
        self.action_size = action_size

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, C, H, W) tensor.
        Returns:
            policy_logits: (batch, action_size)
            value:         (batch, 1)  in range [-1, 1]
        """
        # Stem
        out = F.relu(self.stem_bn(self.stem_conv(x)))

        # Trunk
        for block in self.res_tower:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, state_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Single-state inference for use in MCTS.

        Args:
            state_array: numpy array of shape (C, H, W).
        Returns:
            (policy_probs, value_scalar)
            policy_probs: np.ndarray of shape (action_size,), sums to 1.
            value_scalar: float in [-1, 1].
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(device)
        policy_logits, value = self(x)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value_scalar = float(value.squeeze().cpu())
        return policy, value_scalar

    def predict_batch(
        self, state_arrays: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch inference.

        Args:
            state_arrays: numpy array of shape (batch, C, H, W).
        Returns:
            (policy_probs, values)
            policy_probs: (batch, action_size)
            values:       (batch,)
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.tensor(state_arrays, dtype=torch.float32).to(device)
        with torch.no_grad():
            policy_logits, values = self(x)
        policy = F.softmax(policy_logits, dim=1).cpu().numpy()
        values_np = values.squeeze(1).cpu().numpy()
        return policy, values_np

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
