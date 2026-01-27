"""
train_maskableppo.py

Training script for SidonAddEnv using sb3-contrib MaskablePPO.

Note:
- Imports of sb3/sb3_contrib are inside main() so the file can be imported/compiled
  even if those deps are not installed.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
import re
from typing import List, Optional

from sidon_RL import build_env
from utils import bool_flag




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500, help="Maximum element n (actions 0..n).")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-diffs-obs", action="store_true", help="Use membership-only observation.")
    parser.add_argument("--no-stop", action="store_true", help="Remove STOP action (not recommended).")
    parser.add_argument("--logdir", type=str, default="runs_sidon")
    parser.add_argument("--start_set", type=str, default="")
    parser.add_argument("--with_stop", type=bool_flag, default="true")
    parser.add_argument("--invalid_action_terminates", type=bool_flag, default="false")
    parser.add_argument("--obs_include_diffs", type=bool_flag, default="false")
    parser.add_argument(
        "--torch-validate-distributions",
        action="store_true",
        help=(
            "Garde la validation PyTorch des distributions activée. "
            "ATTENTION: MaskablePPO peut crasher avec l'erreur `Simplex()`."
        ),
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--env_name", type=str, default="sidon")
    parser.add_argument("--start-k-max", type=int, default=0,
                         help="If >0, sample random Sidon start sets up to this size for evaluation.")

    args = parser.parse_args()

    # Workaround: avoid sporadic `constraint Simplex()` crashes in PyTorch Categorical
    # with masked distributions (MaskablePPO). Disable distribution validation unless
    # the user explicitly requests it.
    if not args.torch_validate_distributions:
        import torch as th
        # Newer torch:
        if hasattr(th.distributions.Distribution, "set_default_validate_args"):
            th.distributions.Distribution.set_default_validate_args(False)
        # Fallback for older torch:
        try:
            import torch.distributions.distribution as _dist
            _dist.Distribution._validate_args = False
            th.distributions.Distribution._validate_args = False
        except Exception:
            pass

    def _parse_start_set(s: str) -> Optional[List[int]]:
        """Parse --start_set like '0,1,4' or '0 1 4'. Empty string -> None."""
        s = (s or "").strip()
        if not s:
            return None
        parts = [p for p in re.split(r"[\s,]+", s) if p]
        return [int(p) for p in parts]


    if args.no_stop:
        args.with_stop = False
    if args.no_diffs_obs:
        args.obs_include_diffs = False
    args.start_set = _parse_start_set(args.start_set)

    env = build_env(args)
    #restart from here
    # env = SidonAddEnv(
    #     N=args.n,
    #     with_stop=with_stop,
    # )

    # Import here to keep the module importable without RL deps
    from sb3_contrib import MaskablePPO
    # from sb3_contrib.common.callbacks import BaseCallback

    policy = "MultiInputPolicy" if args.obs_include_diffs else "MlpPolicy"

    run_name = f"{args.env_name}_n{args.N}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = os.path.join(args.logdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    model = MaskablePPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=outdir,
        n_steps=2048,
        batch_size=256,
        gamma=1.0,  # episodic length maximization; reward is +1 per add
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(outdir, "model"))

    # Quick greedy rollout
    # obs, _ = env.reset()
    # done = False
    # total = 0
    # while not done:
    #     masks = env.action_masks()
    #     action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    #     obs, r, done, _, info = env.step(int(action))
    #     total += r
    # print("Final size:", info.get("size"), "total reward:", total)


    final_sizes = []
    rewards = []
    for _ in range(int(args.eval_episodes)):
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, r, done, _, info = env.step(int(action))
            total += float(r)

        final_sizes.append(int(info.get("final_size", info.get("size", len(env.state.vals)))))
        rewards.append(total)

    avg_size = sum(final_sizes) / len(final_sizes)
    max_size = max(final_sizes)
    avg_rew = sum(rewards) / len(rewards)
    print(f"[EVAL] episodes={args.eval_episodes} avg_final_size={avg_size:.3f} max_final_size={max_size} avg_reward={avg_rew:.3f}")

if __name__ == "__main__":
    main()
