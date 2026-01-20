"""Sidon RL helpers (Gymnasium env + incremental action masking).

This package is designed to integrate with an existing codebase that already
contains `sidon.py` defining `SidonSetDataPoint`. If that import fails (e.g.
when running standalone tests), we fall back to a lightweight internal
implementation.

Primary exports:
- SidonAddOnlyState: add-only Sidon state with incremental legal-action mask updates
- SidonAddEnv:   Gymnasium environment exposing `action_masks()` for MaskablePPO
"""

from sidon_RL.sidon_rl_state import SidonAddOnlyState
from sidon_RL.sidon_gym_env import SidonAddEnv

__all__ = [
    "SidonAddOnlyState",
    "SidonAddEnv",
]