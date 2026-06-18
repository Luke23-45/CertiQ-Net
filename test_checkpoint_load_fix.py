"""
Fast verification script for the best_checkpoint_load_failed fix.

Reproduces the exact scenario:
  - Lightning checkpoint containing model.* keys + extra buffer keys
    (dual_lambda, _smoothed_residual, _dual_update_count)
  - Raw model that only has the model.* keys
  - Tries loading with the OLD (broken) logic  -> expects RuntimeError
  - Tries loading with the NEW (fixed) logic  -> expects success

Runs in ~seconds, no GPU needed, no disk I/O.
"""
import copy
import io
import torch
import torch.nn as nn


class RawModel(nn.Module):
    """Simulates the inner raw model (what model.state_dict() returns)."""
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.value_head(self.encoder(x))


class LightningWrapper(nn.Module):
    """
    Simulates the LightningModule state_dict structure:
      - model.encoder.weight, model.value_head.weight  (from the raw model)
      - dual_lambda, _smoothed_residual, _dual_update_count  (from register_buffer)
    """
    def __init__(self, raw: RawModel):
        super().__init__()
        self.model = raw
        self.register_buffer("dual_lambda", torch.tensor(1.0))
        self.register_buffer("_smoothed_residual", torch.tensor(0.0))
        self.register_buffer("_dual_update_count", torch.tensor(0, dtype=torch.long))


def simulate_old_broken_logic(raw_sd: dict) -> bool:
    """Replicates the broken pipeline.py:480-483 logic."""
    try:
        cleaned = {k.removeprefix("model."): v for k, v in raw_sd.items()}
        raw_model = RawModel()
        raw_model.load_state_dict(cleaned)
        return True  # should NOT reach here
    except Exception:
        return False


def simulate_new_fixed_logic(raw_sd: dict) -> bool:
    """Replicates the fix: only process keys that start with 'model.'."""
    try:
        cleaned = {k.removeprefix("model."): v for k, v in raw_sd.items()
                   if k.startswith("model.")}
        raw_model = RawModel()
        raw_model.load_state_dict(cleaned)
        return True
    except Exception:
        return False


def main():
    # Build a LightningWrapper and grab its state_dict (simulates a checkpoint)
    raw = RawModel()
    wrapper = LightningWrapper(raw)
    lightning_sd = wrapper.state_dict()  # This is what goes into checkpoint["state_dict"]

    print("=== Lightning state_dict keys ===")
    for k in sorted(lightning_sd.keys()):
        print(f"  {k}: {tuple(lightning_sd[k].shape)}")

    # --- Test OLD broken logic ---
    old_ok = simulate_old_broken_logic(copy.deepcopy(dict(lightning_sd)))
    if old_ok:
        print("\n[FAIL] OLD logic unexpectedly succeeded — something changed.")
    else:
        print("\n[PASS] OLD logic correctly fails on non-model keys.")

    # --- Test NEW fixed logic ---
    new_ok = simulate_new_fixed_logic(copy.deepcopy(dict(lightning_sd)))
    if new_ok:
        print("[PASS] NEW logic successfully loads model weights (ignores buffer keys).")
    else:
        print("[FAIL] NEW logic still fails — unexpected.")

    # --- Verify weights match ---
    raw_ref = RawModel()
    raw_ref.load_state_dict(raw.state_dict())  # reference copy before load
    cleaned = {k.removeprefix("model."): v for k, v in lightning_sd.items()
               if k.startswith("model.")}
    raw_loaded = RawModel()
    raw_loaded.load_state_dict(cleaned)
    ok = all(torch.equal(p_ref, p_loaded)
             for (_, p_ref), (_, p_loaded) in zip(raw_ref.named_parameters(),
                                                   raw_loaded.named_parameters()))
    print(f"[{'PASS' if ok else 'FAIL'}] Loaded weights match reference weights.")

    # Summary
    print("\n=== RESULT ===")
    if not old_ok and new_ok and ok:
        print("ALL CHECKS PASSED. The fix is correct and ready for deployment.")
    else:
        print("SOME CHECKS FAILED — review the output above.")


if __name__ == "__main__":
    main()
