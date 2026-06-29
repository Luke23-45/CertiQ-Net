# QGym Patch Set

This directory contains the patch files that are applied to `extern/QGym`
before training, evaluation, or dataset collection.

Apply them with:

```bash
python scripts/apply_patches_submodule.py
```

The apply step always resets the submodule worktree to `HEAD` first, then
applies the patches in filename order.

