
from dataclasses import dataclass, field
from typing import Type, Callable, List, Tuple, Any, Optional
from copy import deepcopy
import numpy as np

from datasets import BiasDataset, CelebA, MuraliChestXRay14
from models import BiasModel, CelebAModel, ChestXRay14Model
from subgroup_identification import BaseSubgroupIdentifier, GTSubgroups

# ==============================================================================
#  Helper Functions
# ==============================================================================

def _cvgisic_subgroup_postprocessor(group_ids_train_dfr: np.ndarray, group_ids_train_score_mapping: np.ndarray, group_ids_test: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    n_subgroups = len(np.unique(group_ids_test))
    group_id_mapping = {old_id: new_id for new_id, old_id in enumerate(np.unique(group_ids_test))}
    group_ids_train_dfr, group_ids_train_score_mapping, group_ids_test = [np.array([group_id_mapping[g] for g in group_ids]) for group_ids in (group_ids_train_dfr, group_ids_train_score_mapping, group_ids_test)]

    return n_subgroups, group_ids_train_dfr, group_ids_train_score_mapping, group_ids_test

# ==============================================================================
#  Dataset Configuration Structure
# ==============================================================================

@dataclass
class DatasetConfig:
    """A dataclass to hold all dataset-specific configurations."""

    # Dataset specific fields.
    root_dir: str
    DatasetClass: Type[BiasDataset]
    ModelClass: Type[BiasModel]
    transform: Callable
    get_base_models: Callable[[int], List[str]]
    regularization_parameter_c: float

    # Fields with defaults.
    balancing_factor_min: float = 0.0
    balancing_factor_max: float = 1.0
    SubgroupIdentifierClass: Type[BaseSubgroupIdentifier] = GTSubgroups
    subgroup_postprocessor: Optional[Callable] = None

    def get_datasets(self) -> Tuple[BiasDataset, BiasDataset]:
        # Validation dataset.
        dataset_config_val = {
            "root": self.root_dir,
            "split": "val",
            "bias": "all",
            "return_bias": False,
            "download": True,
            "transform": self.transform,
        }
        dataset_val = self.DatasetClass(
            **dataset_config_val,
        )

        # Test dataset.
        dataset_config_test = deepcopy(dataset_config_val)
        dataset_config_test["split"] = "test"

        dataset_test = self.DatasetClass(
            **dataset_config_test
        )
        
        return (dataset_val, dataset_test)

# ==============================================================================
#  Specific Dataset Configurations
# ==============================================================================

DATASET_CONFIGS = {
    "celeba": {
        "DatasetClass": CelebA,
        "ModelClass": CelebAModel,
        "transform": CelebA.default_transform(normalize=True, augmentation=False),
        "get_base_models": lambda: [f"izmailov_resnet50_erm_seed{i}" for i in range(1, 6)],
        "regularization_parameter_c": 0.7,
    },
    "chestx-ray14": {
        "DatasetClass": MuraliChestXRay14,
        "ModelClass": ChestXRay14Model,
        "transform": MuraliChestXRay14.default_transform(normalize=True),
        "get_base_models": lambda: [f"murali_dense121_erm_seed{i}" for i in range(1, 6)],
        "regularization_parameter_c": 0.7,
    },
}


def get_config(dataset_name: str, root_dir: str) -> DatasetConfig:
    """
    Retrieves the configuration for a given dataset name.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Configuration for dataset '{dataset_name}' not found!")
    
    # Load config and add root directory.
    config = DATASET_CONFIGS[dataset_name]
    config["root_dir"] = root_dir
    
    return DatasetConfig(**config)