from certiqnet.experiments.catalog.specs import StudySpec, TrainingVariant

main_queueing_study = StudySpec(
    name="main_queueing",
    description="Exact-certified queueing with baseline comparison",
    config_name="experiments/main_queueing",
    stages=("train", "audit", "baselines"),
    variants=[
        TrainingVariant(
            label="reentrant_3_hyper",
            dataset="reentrant_3_hyper",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
        TrainingVariant(
            label="reentrant_2",
            dataset="reentrant_2",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
        TrainingVariant(
            label="re-reentrant_3_hyper",
            dataset="re-reentrant_3_hyper",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
        TrainingVariant(
            label="n_model_large",
            dataset="n_model_large",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
        TrainingVariant(
            label="hospital",
            dataset="hospital",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
    ],
)
