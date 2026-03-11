"""Active learning module for DPGen2-style iterative training."""

__all__ = [
    "evaluate_pes_coverage",
    "ModelDeviCalculator",
    "ConfSelector",
    "DiversitySelector",
    "merge_training_data",
    "ExplorationStage",
    "StageScheduler",
    "make_single_stage_scheduler",
    "generate_perturbed_structures",
    "generate_init_dataset",
]

# Lazy imports to avoid loading heavy deps (dscribe, etc.) at package load
def __getattr__(name):
    if name == "evaluate_pes_coverage":
        from molecular_force_field.active_learning.pes_coverage import evaluate_pes_coverage
        return evaluate_pes_coverage
    if name == "ModelDeviCalculator":
        from molecular_force_field.active_learning.model_devi import ModelDeviCalculator
        return ModelDeviCalculator
    if name == "ConfSelector":
        from molecular_force_field.active_learning.conf_selector import ConfSelector
        return ConfSelector
    if name == "DiversitySelector":
        from molecular_force_field.active_learning.diversity_selector import DiversitySelector
        return DiversitySelector
    if name == "merge_training_data":
        from molecular_force_field.active_learning.data_merge import merge_training_data
        return merge_training_data
    if name == "ExplorationStage":
        from molecular_force_field.active_learning.stage_scheduler import ExplorationStage
        return ExplorationStage
    if name == "StageScheduler":
        from molecular_force_field.active_learning.stage_scheduler import StageScheduler
        return StageScheduler
    if name == "make_single_stage_scheduler":
        from molecular_force_field.active_learning.stage_scheduler import make_single_stage_scheduler
        return make_single_stage_scheduler
    if name == "generate_perturbed_structures":
        from molecular_force_field.active_learning.init_data import generate_perturbed_structures
        return generate_perturbed_structures
    if name == "generate_init_dataset":
        from molecular_force_field.active_learning.init_data import generate_init_dataset
        return generate_init_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
