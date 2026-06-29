"""Runtime patching for QGym submodule (rl_env.py).

QGym is a vendored git submodule; we must NOT modify it directly.
Instead, this module restores the committed original at every run
and then applies three targeted source-level patches to
``extern/QGym/RL/utils/rl_env.py``.

Patches applied
---------------
1. ``lam(t)`` → ``lam(t, rng=None, batch=None)``, adding ``elif lam_type == 'hyper':``
2. ``draw_inter_arrivals`` — pass ``rng`` / ``batch`` to ``lam()``
3. ``draw_service`` — add ``service_type == 'hyper'`` branch
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── Project root discovery ────────────────────────────────────────────────


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains ``configs/``)."""
    here = Path(__file__).resolve().parent  # .../certiqnet/utils
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "extern").is_dir():
            return parent
    raise RuntimeError(
        "Cannot locate project root from "
        f"{here} — expected parent with configs/ and extern/ directories."
    )


def _rl_env_path() -> Path:
    """Return the absolute path to ``rl_env.py``."""
    return _find_project_root() / "extern" / "QGym" / "RL" / "utils" / "rl_env.py"


# ── Restore original ──────────────────────────────────────────────────────


def _restore_original(rl_env_path: Path) -> None:
    """Revert ``rl_env.py`` to its committed state via ``git checkout``."""
    qgym_root = rl_env_path.parents[2]  # extern/QGym
    relative = rl_env_path.relative_to(qgym_root)
    result = subprocess.run(
        ["git", "-C", str(qgym_root), "checkout", "--", str(relative)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    # Fallback: restore from pristine backup.
    backup = rl_env_path.with_suffix(".py.pristine")
    if backup.exists():
        log.warning("git checkout failed, falling back to .pristine backup:\n%s", result.stderr)
        shutil.copy2(str(backup), str(rl_env_path))
        return

    raise RuntimeError(
        f"Failed to restore {rl_env_path} to its original state.\n"
        f"git checkout error:\n{result.stderr}\n\n"
        f"To recover, manually run:\n"
        f"    git -C {qgym_root} checkout -- {relative}\n"
        f"Then re-run the experiment."
    )


def _ensure_pristine_backup(rl_env_path: Path) -> None:
    """Create a ``.pristine`` backup of the committed file on first encounter."""
    backup = rl_env_path.with_suffix(".py.pristine")
    if not backup.exists():
        try:
            shutil.copy2(str(rl_env_path), str(backup))
            log.debug("Created pristine backup → %s", backup)
        except OSError as exc:
            log.warning("Could not create pristine backup: %s", exc)


# ── Hunk definitions ──────────────────────────────────────────────────────
# Each hunk is (old_string, new_string).
# The old_string MUST appear exactly once in the file.

_HUNKS: list[tuple[str, str]] = []
_PATCHED = False

# -- Hunk 1: lam() signature + hyper branch ---------------------------------

_HUNKS.append(
    (
        # old
        "    def lam(t):\n"
        '            if lam_type == \'constant\':\n'
        "                lam = lam_r\n"
        "            elif lam_type == 'step':\n"
        "                is_surge = 1*(t.data.cpu().numpy() <= lam_params['t_step'])\n"
        "                lam = is_surge * np.array(lam_params['val1']) + (1 - is_surge) * np.array(lam_params['val2'])\n"
        "            else:\n"
        "                return 'Nonvalid arrival rate'\n"
        "            \n"
        "            return lam\n",
        # new
        "    def lam(t, rng=None, batch=None):\n"
        "            if lam_type == 'constant':\n"
        "                lam = lam_r\n"
        "            elif lam_type == 'step':\n"
        "                is_surge = 1*(t.data.cpu().numpy() <= lam_params['t_step'])\n"
        "                lam = is_surge * np.array(lam_params['val1']) + (1 - is_surge) * np.array(lam_params['val2'])\n"
        "            elif lam_type == 'hyper':\n"
        "                scale = lam_params.get('scale', 0.8)\n"
        "                lam_r_arr = np.asarray(lam_r, dtype=float)\n"
        "                if rng is None or batch is None:\n"
        "                    lam = lam_r_arr\n"
        "                else:\n"
        "                    lam_r_2d = lam_r_arr.reshape((1, len(lam_r_arr))).repeat(batch, axis=0)\n"
        "                    switch = rng.binomial(1, 0.5, (batch, 1))\n"
        "                    lam = switch * (lam_r_2d / (1 + scale)) + (1 - switch) * (lam_r_2d / (1 - scale))\n"
        "            else:\n"
        "                return 'Nonvalid arrival rate'\n"
        "            \n"
        "            return lam\n",
    )
)

# -- Hunk 2: pass rng/batch to lam() in draw_inter_arrivals -----------------

_HUNKS.append(
    (
        # old
        "            lam_rate = lam(t)\n",
        # new
        "            lam_rate = lam(t, rng=state, batch=batch)\n",
    )
)

# -- Hunk 3: hyper service_type branch in draw_service ----------------------

_HUNKS.append(
    (
        # old
        "    def draw_service(self, time):\n"
        "        def service_dists(state, batch, t):\n"
        "            return state.exponential(1, (batch, orig_q))\n"
        "        service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)\n"
        "        return service\n",
        # new
        "    def draw_service(self, time):\n"
        "        def service_dists(state, batch, t):\n"
        '            if "service_type" in env_config.keys() and env_config["service_type"] == "hyper":\n'
        "                scale = lam_params.get('scale', 0.8)\n"
        "                coins = state.binomial(1, 0.5, size=(batch, orig_q))\n"
        "                a = state.exponential((1 + scale), (batch, orig_q))\n"
        "                b = state.exponential((1 - scale), (batch, orig_q))\n"
        "                return coins * a + (1 - coins) * b\n"
        "            return state.exponential(1, (batch, orig_q))\n"
        "        service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)\n"
        "        return service\n",
    )
)


# ── Patch application ─────────────────────────────────────────────────────


# -- Hunk 4: batch-aware env import ------------------------------------------

_HUNKS.append(
    (
        # old
        "from main.env import DiffDiscreteEventSystem\n",
        # new
        "from main.env import BatchedEnv, DiffDiscreteEventSystem\n",
    )
)

# -- Hunk 5: choose batched env for batch > 1 --------------------------------

_HUNKS.append(
    (
        # old
        "    dq = RL_Wrapper_P_DiffDiscreteEventSystem(network, mu, h, \n"
        "                                       draw_service= draw_service, draw_inter_arrivals = draw_inter_arrivals, init_time = 0, \n"
        "                                    queue_event_options= queue_event_options,\n"
        "                                    batch = batch, \n"
        "                                    temp = temp, seed = seed,\n"
        "                                    time_f = False,\n"
        "                                    reward_scale = 1.0,\n"
        "                                    policy_name= policy_name,\n"
        "                                    action_map = None,\n"
        "                                    device = torch.device(device))\n"
        "\n"
        "    return dq\n",
        # new
        "    env_cls = BatchedEnv if batch > 1 else RL_Wrapper_P_DiffDiscreteEventSystem\n"
        "    dq = env_cls(network, mu, h,\n"
        "                 draw_service=draw_service, draw_inter_arrivals=draw_inter_arrivals, init_time=0,\n"
        "                 queue_event_options=queue_event_options,\n"
        "                 batch=batch,\n"
        "                 temp=temp, seed=seed,\n"
        "                 time_f=False,\n"
        "                 reward_scale=1.0,\n"
        "                 policy_name=policy_name,\n"
        "                 action_map=None,\n"
        "                 device=torch.device(device))\n"
        "\n"
        "    return dq\n",
    )
)

def apply_qgym_patches() -> None:
    """Restore the original ``rl_env.py``, then apply all tracked hunks.

    Raises
    ------
    RuntimeError
        If the file is missing or a hunk's old string is not found
        or matches multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    rl_env = _rl_env_path()
    if not rl_env.is_file():
        raise FileNotFoundError(f"QGym rl_env.py not found at: {rl_env}")

    # 1. Restore committed original.
    _ensure_pristine_backup(rl_env)
    _restore_original(rl_env)

    # 2. Read the clean original.
    source = rl_env.read_text(encoding="utf-8")

    # 3. Apply every hunk.
    for idx, (old, new) in enumerate(_HUNKS, start=1):
        count = source.count(old)
        if count == 0:
            raise RuntimeError(
                f"[QGym patch hunk {idx}] Old string not found in\n"
                f"    {rl_env}\n\n"
                f"Expected to find:\n{old}\n"
                f"The QGym submodule may have been updated — verify and update the patch."
            )
        if count > 1:
            raise RuntimeError(
                f"[QGym patch hunk {idx}] Old string found {count} times (expected 1).\n"
                f"    {rl_env}\n\n"
                f"The hunk is ambiguous — provide more surrounding context."
            )
        source = source.replace(old, new, 1)

    # 4. Write the patched version back.
    rl_env.write_text(source, encoding="utf-8")
    _PATCHED = True
    log.info("QGym rl_env.py patched successfully (%d hunks).", len(_HUNKS))


# ── CLI convenience --------------------------------------------------------


def main() -> None:
    """Apply patches from the command line."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        apply_qgym_patches()
        print("QGym patches applied.", file=sys.stderr)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
