"""
sidon_rl_state.py

Add-only Sidon-set state with *incremental* valid-action masking.

Design goals:
- Integrate with your existing sidon.py as much as possible:
  - If sidon.SidonSetDataPoint is importable, we wrap it and use its incremental diff updates.
  - Otherwise (e.g., in minimal environments), we fall back to a lightweight internal implementation.
- Fast action masking for N up to ~5000 (typical Sidon size ~O(sqrt(N)) ~ 70):
  - One O(N*k) mask computation at reset.
  - Then per-add updates are ~O(k^2), without rescanning 0..N.
"""

from __future__ import annotations
from sidon_RL.rl_state import RLDataclass
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple
import bisect
import numpy as np

from sidon_RL.gym_env import DataClassGymEnv


def is_sidon(vals: List[int]) -> bool:
    """Helper to check sidon property: all positive pairwise differences are distinct."""
    #Sidon specific out of main file
    v = sorted(vals)
    diffs: Set[int] = set()
    for i, a in enumerate(v):
        for b in v[i + 1 :]:
            d = b - a
            if d in diffs:
                return False
            diffs.add(d)
    return True


@dataclass
class SidonAddOnlyState(RLDataclass):
    """
    Add-only Sidon-set search state with cached + incrementally-updated action mask.

    Actions:
      - 0..N: add that element (if legal)
      - optional STOP action at index N+1 (always legal) if with_stop=True
    """

    N: int
    with_stop: bool = True
    obs_include_diffs: bool = False


    def _slow_mask(self,N: int, vals: List[int], with_stop: bool) -> np.ndarray:
        """
        Slow O(N*k) recomputation of the legal-add mask (plus optional STOP).
        Used for correctness checks / debugging.
        """
        #Sidon specific
        vset = set(vals)
        # used diffs
        used = [False] * (N + 1)
        v = sorted(vals)
        for i, a in enumerate(v):
            for b in v[i + 1 :]:
                used[b - a] = True

        mask = np.zeros((N + 2) if with_stop else (N + 1), dtype=bool)
        for x in range(N + 1):
            if x in vset:
                continue
            local = set()
            ok = True
            for a in v:
                d = abs(x - a)
                if d == 0 or d in local or used[d]:
                    ok = False
                    break
                local.add(d)
            mask[x] = ok

        if with_stop:
            mask[N + 1] = True
        return mask

    def observation(self) -> object:
        present = np.zeros(self.N + 1, dtype=np.int8)
        present[self._vals] = 1
        if not self.obs_include_diffs:
            return present
        diffs_used = np.asarray(self._used, dtype=np.int8)  # length N+1
        return {"present": present, "diffs_used": diffs_used}

    def reset(self, start_set: Optional[List[int]] = None) -> None:
        """
        Initialize state and compute initial mask in O(N*k).
        """
        start = [] if start_set is None else list(start_set)
        start = sorted(set(int(x) for x in start if 0 <= int(x) <= self.N))

        if len(start) == 0:
            start = [0]
        self._vals = start[:]  # plain python list (sorted)
        self._valset: Set[int] = set(self._vals)

        # Maintain used diffs as both:
        # - boolean array for O(1) membership (used[d])
        # - set for iteration in incremental updates (diff_set)
        self._used = [False] * (self.N + 1)
        self._diff_set: Set[int] = set()
        v = self._vals
        for i, a in enumerate(v):
            for b in v[i + 1 :]:
                d = b - a
                self._used[d] = True
                self._diff_set.add(d)

        # Cached mask
        self._mask = self._slow_mask(self.N, self._vals, self.with_stop)

    def add(self, x: int) -> None:
        """
        Apply a legal add move and update mask incrementally.
        Raises ValueError if x is not currently legal.
        """
        # Sidon specific
        if x < 0 or x > self.N:
            raise ValueError(f"Action {x} out of range 0..{self.N}")
        if x in self._valset:
            raise ValueError(f"Action {x} is already in the set.")
        if not bool(self._mask[x]):
            raise ValueError(f"Action {x} is not legal under current mask.")

        old_vals = list(self._vals)
        old_diff_list = list(self._diff_set)

        # Compute diffs from x to old set; must be all new and distinct for Sidon add-only legality.
        diffs_added: List[int] = []
        local = set()
        for a in old_vals:
            d = abs(x - a)
            if d == 0 or d in local or self._used[d]:
                # Should not happen if mask is correct.
                raise ValueError(f"Internal inconsistency: adding {x} would violate Sidon.")
            local.add(d)
            diffs_added.append(d)

        # Update underlying datapoint diffs_count incrementally (and val list).
        bisect.insort(self._vals, x)

        # Update our own bookkeeping for diffs
        for d in diffs_added:
            self._used[d] = True
            self._diff_set.add(d)

        # Update valset
        self._valset.add(x)

        # Invalidate x itself
        self._invalidate(x)

        # 1) Midpoints: adding x forbids y=(x+a)/2 for each a in old set when integer.
        for a in old_vals:
            s = x + a
            if (s & 1) == 0:
                m = s >> 1
                self._invalidate(m)

        # 2) New diffs: for each new diff d, forbid y = b ± d for all b in new set.
        #    (Collision with newly-used differences.)
        new_vals = self._vals  # already includes x
        for d in diffs_added:
            for b in new_vals:
                self._invalidate(b - d)
                self._invalidate(b + d)

        # 3) Old diffs with the new element: forbid y such that |y - x| is an *old* used difference.
        #    That is y = x ± d for d in D_old.
        for d in old_diff_list:
            self._invalidate(x - d)
            self._invalidate(x + d)

        # STOP is always valid if enabled
        if self.with_stop:
            self._mask[self._stop_action] = True
        if not self._mask.any():
            if self.with_stop:
                self._mask[self._stop_action] = True
            else:
                self._mask[0] = True



class SidonAddEnv(DataClassGymEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        params,
    ):
        start_set = params.start_set
        with_stop = params.with_stop
        invalid_action_terminates = params.invalid_action_terminates
        self.N = int(params.N)
        super().__init__(start_set=start_set,with_stop=with_stop,invalid_action_terminates=invalid_action_terminates)
        # State
        self.state = SidonAddOnlyState(
            N=self.N,
            with_stop=self.with_stop,
            obs_include_diffs=params.obs_include_diffs,
        )
        self.state.reset(self._start_set)
        self.build_action_state()

    def compute_reward(self) -> int:
        """
        Returns a reward when one element is added to the set successfully. Set to 1 for uniform reward
        """
        return max(1,len(self.state.vals)-int(np.sqrt(self.N)/2))
