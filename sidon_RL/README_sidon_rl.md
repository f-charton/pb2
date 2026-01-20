# Sidon RL (add-only) — Gymnasium + MaskablePPO with incremental masks

This adds an add-only Sidon-set environment that is compatible with **sb3-contrib MaskablePPO**.

## Files
- `sidon_rl_state.py` — core incremental mask logic (optionally wraps your `sidon.py` `SidonSetDataPoint`)
- `sidon_gym_env.py` — Gymnasium environment (implements `action_masks()`)
- `train_maskableppo.py` — training script
- `smoke_test_sidon_rl.py` — correctness tests (no gym/SB3 required)
- `requirements_sidon_rl.txt` — dependencies

## Quick test (no gym required)
```bash
python smoke_test_sidon_rl.py
```

## Install deps
```bash
pip install -r requirements_sidon_rl.txt
```

## Train
```bash
python train_maskableppo.py --n 500 --timesteps 300000
python train_maskableppo.py --n 5000 --timesteps 1000000
```

### Notes on integration with your existing `sidon.py`
`sidon_rl_state.py` *tries* to import `SidonSetDataPoint` from `sidon.py` and use its incremental `_add_element()` updates.

If your broader codebase is available (so `sidon.py` imports cleanly), it will use the real class.
Otherwise it falls back to a lightweight internal datapoint, so the core logic remains runnable in isolation.
