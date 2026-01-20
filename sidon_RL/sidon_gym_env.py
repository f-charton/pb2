"""
sidon_gym_env.py

Gymnasium environment for add-only Sidon sets using SidonAddOnlyState with incremental masks.
Compatible with sb3-contrib MaskablePPO (invalid action masking via action_masks()).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict as DictSpace

from sidon_RL.sidon_rl_state import SidonAddOnlyState


# def _as_start_set(x, N: int):
#     """Accept a list[int] or a SidonSetDataPoint-like object with `.val`."""
#     if x is None:
#         return [0]
#     # SidonSetDataPoint (or any object with .val)
#     if hasattr(x, "val"):
#         x = getattr(x, "val")
#     # list/iterable of ints
#     start = list(x)
#     start = sorted(set(int(t) for t in start if 0 <= int(t) <= N))
#     return start if len(start) > 0 else [0]


class SidonAddEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        N: int,
        start_set=None,
        with_stop: bool = True,
        obs_include_diffs: bool = True,
        prefer_sidon_datapoint: bool = True,
        invalid_action_terminates: bool = True,
    ):
        super().__init__()
        self.N = int(N)
        self.with_stop = bool(with_stop)
        self.obs_include_diffs = bool(obs_include_diffs)
        self.prefer_sidon_datapoint = bool(prefer_sidon_datapoint)
        self.invalid_action_terminates = bool(invalid_action_terminates)
        self._start_set = [0] if start_set is None else list(start_set)

        # State
        self.state = SidonAddOnlyState(
            N=self.N,
            with_stop=self.with_stop,
            obs_include_diffs=self.obs_include_diffs,
            prefer_sidon_datapoint=self.prefer_sidon_datapoint,
        )
        self.state.reset(self._start_set)

        # Spaces
        self.action_space = Discrete(self.state.action_space_n)
        if not self.obs_include_diffs:
            self.observation_space = Box(low=0, high=1, shape=(self.N + 1,), dtype=np.int8)
        else:
            self.observation_space = DictSpace(
                {
                    "present": Box(low=0, high=1, shape=(self.N + 1,), dtype=np.int8),
                    "diffs_used": Box(low=0, high=1, shape=(self.N + 1,), dtype=np.int8),
                }
            )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        start_set = self._start_set
        if options is not None and "start_set" in options and options["start_set"] is not None:
            start_set = options["start_set"]
        self.state.reset(start_set)
        self._episode_start_size = len(self.state.vals)
        obs = self.state.observation()
        info = {"size": len(self.state.vals), "start_size": self._episode_start_size}
        return obs, info

    def action_masks(self) -> np.ndarray:
        mask = np.asarray(self.state.action_mask(), dtype=np.bool_).copy()
        # Safety: never return all-False masks (would yield NaNs in MaskableCategorical)
        if not mask.any():
            if self.with_stop:
                mask[self.state.stop_action] = True
            else:
                mask[0] = True
        return mask
    def step(self, action: int):
        # STOP action (optional)
        if self.with_stop and action == self.state.stop_action:
            obs = self.state.observation()
            final_size = len(self.state.vals)
            info = {
                "size": final_size,
                "start_size": getattr(self, "_episode_start_size", final_size),
                "final_size": final_size,
                "added": final_size - getattr(self, "_episode_start_size", final_size),
                "stopped": True,
            }
            return obs, 0.0, True, False, info

        # Invalid action safety
        mask = self.state.action_mask()
        if action < 0 or action >= mask.shape[0] or not bool(mask[action]):
            obs = self.state.observation()
            final_size = len(self.state.vals)
            info = {
                "size": final_size,
                "start_size": getattr(self, "_episode_start_size", final_size),
                "final_size": final_size,
                "added": final_size - getattr(self, "_episode_start_size", final_size),
                "invalid": True,
                "action": int(action),
            }
            if self.invalid_action_terminates:
                return obs, -1.0, True, False, info
            return obs, -1.0, False, False, info

        # Legal add
        self.state.add(int(action))
        reward = (len(self.state.vals)-1)**2-len(self.state.vals)**2

        # Terminal if no add moves remain (agent can also STOP earlier)
        terminated = not self.state.has_any_add_move()
        obs = self.state.observation()
        final_size = len(self.state.vals)
        info = {"size": final_size}
        if terminated:
            info.update(
                {
                    "start_size": getattr(self, "_episode_start_size", final_size),
                    "final_size": final_size,
                    "added": final_size - getattr(self, "_episode_start_size", final_size),
                }
            )
        return obs, reward, terminated, False, info
