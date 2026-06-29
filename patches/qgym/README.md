# QGym Patch Set

This directory contains the patch files that are applied to `extern/QGym`
before training, evaluation, or dataset collection.

Apply them with:

```bash
python scripts/apply_patches_submodule.py
```

The apply step always resets the submodule worktree to `HEAD` first, then
applies the patches in filename order.

This active patch set is intentionally curated for the CertiQ rollout path.
It now includes the batched rollout path used by training and validation.
Historical experiments in `studies/patches` are archival and include older
batch-oriented or trainer-specific changes that are not applied here.
