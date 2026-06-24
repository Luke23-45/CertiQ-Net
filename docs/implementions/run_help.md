python -m certiqnet.data.qgym.collect_qgym collect re-reentrant_3_hyper --n-steps 10000


python -m certiqnet.data.qgym.audit_qgym re-reentrant_3_hyper --output audit_report.json


python -m certiqnet.scripts.train --config-name experiments/main_queueing

python -m certiqnet.experiments.runner --study main_queueing --variant re-reentrant_3_hyper


python -m studies.runner.families.main_queueing --dataset reentrant_2 --stages train,baselines --seeds 42 --baselines-include sed,max_weight,max_pressure,c_mu