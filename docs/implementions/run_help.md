python -m certiqnet.data.qgym.collect_qgym collect re-reentrant_3_hyper --n-steps 10000


python -m certiqnet.data.qgym.audit_qgym re-reentrant_3_hyper --output audit_report.json


python -m certiqnet.scripts.train --config-name experiments/main_queueing

python -m certiqnet.experiments.runner --study main_queueing --variant re-reentrant_3_hyper


python -m studies.runner.families.main_queueing --dataset reentrant_2 --stages train,baselines --seeds 42 --baselines-include sed,max_weight,max_pressure,c_mu


======

python -m certiqnet.data.qgym.collect_qgym collect re-reentrant_3_hyper --n-steps 10000 --trust-override --force

===
python -m certiqnet.experiments.runner --study main_queueing --variant re-reentrant_3_hyper --seed 42 --set studies.runner.baselines.include=[sed,max_weight,max_pressure,c_mu]
==============
========================






==================

The corrected command is:

```powershell
python -m certiqnet.experiments.runner --study main_queueing --variant reentrant_2 --seed 42 --set studies.runner.baselines.include=[sed,max_weight,max_pressure,c_mu]
```

Differences from the old command:

| Old | New | Reason |
|---|---|---|
| `studies.runner.families.main_queueing` | `certiqnet.experiments.runner` | That file is now a deprecated shim forwarding to the new runner |
| `--dataset reentrant_2` | `--variant reentrant_2` | Dataset is baked into the study's `TrainingVariant` definition |
| `--stages train,baselines` | *(omitted)* | New runner always runs all spec stages (train+audit+baselines). Audit is fast (reads saved config only). To skip it you'd run training and baselines scripts manually |
| `--seeds 42` | `--seed 42` | Repeatable (`--seed 42 --seed 43`) |
| `--baselines-include sed,max_weight,max_pressure,c_mu` | `--set studies.runner.baselines.include=[sed,max_weight,max_pressure,c_mu]` | Hydra override syntax via `--set` |

Or equivalently using `run.py` at the project root:
```powershell
python run.py --study main_queueing --variant reentrant_2 --seed 42 --set studies.runner.baselines.include=[sed,max_weight,max_pressure,c_mu]
```