from certiqnet.experiments.catalog.studies.main_queueing import main_queueing_study
from certiqnet.experiments.catalog.studies.main_queueing_optimized import main_queueing_optimized_study


def _collect_specs():
    return [
        main_queueing_study,
        main_queueing_optimized_study,
    ]
