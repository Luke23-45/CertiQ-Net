"""Weights & Biases sweep launcher."""

import subprocess


def main() -> None:
    """Launch the documented Hydra multirun sweep."""
    cmd = [
        "python",
        "scripts/train.py",
        "--multirun",
        "model=certiqnet_s,certiqnet_p,backbone_only,certiqnet_x_ablation",
        "env=family_a,family_b,family_c,family_e",
        "project.seed=0,1,2",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
