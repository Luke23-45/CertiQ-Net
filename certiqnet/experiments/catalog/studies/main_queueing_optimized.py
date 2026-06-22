from certiqnet.experiments.catalog.specs import StudySpec, TrainingVariant

main_queueing_optimized_study = StudySpec(
    name="main_queueing_optimized",
    description="Exact-certified queueing (optimized, no baselines)",
    config_name="experiments/main_queueing_optimized",
    stages=("train", "audit"),
    variants=[
        TrainingVariant(
            label="reentrant_3_hyper",
            dataset="reentrant_3_hyper",
            model="certiq_index",
            adapter="queueing",
            seeds=[42],
        ),
    ],
)
