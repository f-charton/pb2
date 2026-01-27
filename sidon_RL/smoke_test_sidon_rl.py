"""
smoke_test_sidon_rl.py

Correctness tests for SidonAddOnlyState incremental masking (no gym/SB3 required).
"""

from __future__ import annotations

import random
import numpy as np

from sidon_RL.sidon_rl_state import SidonAddOnlyState, is_sidon


def run_one(N: int, seed: int) -> None:
    rng = random.Random(seed)
    st = SidonAddOnlyState(N=N, with_stop=True, obs_include_diffs=True)
    st.reset([0])

    # Check initial correctness
    st.assert_mask_correct()
    assert is_sidon(st.vals)

    steps = 0
    while True:
        mask = st.action_mask()
        valid_adds = [i for i in range(N + 1) if mask[i]]
        if not valid_adds:
            break

        # Random valid add
        x = rng.choice(valid_adds)
        st.add(x)

        assert is_sidon(st.vals), f"Not Sidon after adding {x}, vals={st.vals}"

        # Periodically cross-check mask vs slow recomputation
        if steps % 5 == 0:
            st.assert_mask_correct()

        steps += 1

    # terminal should have no adds
    assert not st.has_any_add_move()
    st.assert_mask_correct()


def main():
    for N in [50, 100, 200, 500]:
        for seed in range(5):
            run_one(N, seed)
    print("OK: smoke tests passed.")


if __name__ == "__main__":
    main()
