"""
LightningCLI entrypoint for CertiQ‑Net.

Instantiates all classes directly from YAML using jsonargparse.
Uses subclass mode to inject backbones, loss functions, and data modules.

Usage
-----
Training:
    certiqnet-train fit --config configs/experiments/queueing/certiq_index.yaml

Testing:
    certiqnet-train test --config <config> --ckpt_path best

Resume from checkpoint:
    certiqnet-train fit --config <config> --ckpt_path <path>
"""

from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl

from certiqnet.train.common.module import BaseCertiQLightningModule


class CertiQNetCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass


def main():
    cli = CertiQNetCLI(
        model_class=BaseCertiQLightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=True,
    )


if __name__ == "__main__":
    main()
