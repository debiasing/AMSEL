from tqdm import tqdm
import torch
from subgroup_identification.base import BaseSubgroupIdentifier
from datasets import BiasDataset
from models import BiasModel

class GTSubgroups(BaseSubgroupIdentifier):
    def __init__(self):
        self._name = "Ground Truth Subgroups"
    
    def identify_subgroups(self, model: BiasModel, dataset: BiasDataset, device: str="cpu") -> tuple[torch.Tensor, int]:
        # Hidden subgroups are combination of labels + bias attributes.
        class_labels = dataset.class_labels
        bias_levels = dataset.bias_levels
        n_subgroups = len(class_labels)*len(bias_levels)
        
        # Group enumeration:
        #   label_0+bias_0, label_0+bias_1, ..., label_n+bias_n
        group_ids = torch.tensor([dataset.label(idx)*len(bias_levels) + dataset.bias_attr(idx) for idx in tqdm(range(len(dataset)), desc="Extract subgroup ids")])
        
        return group_ids.to(device), n_subgroups
    
class GTBiasAttributes(BaseSubgroupIdentifier):
    def __init__(self):
        self._name = "Ground Truth Bias Attribute"
    
    def identify_subgroups(self, model: BiasModel, dataset: BiasDataset, device: str="cpu") -> tuple[torch.Tensor, int]:
        # Hidden subgroups are the bias attributes.
        bias_levels = dataset.bias_levels
        n_subgroups = len(bias_levels)
        
        # Group enumeration:
        #   bias_0, bias_1, ..., bias_n
        group_ids = torch.tensor([dataset.bias_attr(idx) for idx in tqdm(range(len(dataset)), desc="Extract subgroup ids")])
        
        return group_ids.to(device), n_subgroups
    

class GTAlignVSConflict(BaseSubgroupIdentifier):
    def __init__(self):
        self._name = "Ground Truth Align vs. Conflict Subgroups"
    
    def identify_subgroups(self, model: BiasModel, dataset: BiasDataset, device: str="cpu") -> tuple[torch.Tensor, int]:
        # Hidden subgroups are combination of labels + bias alignment.
        class_labels = dataset.class_labels
        n_subgroups = 2* len(class_labels)

        # Group enumeration:
        #   label_0+bias_conflict, label_0+bias_align, ..., label_n+bias_align
        group_ids = torch.tensor([dataset.label(idx)*2 + int(dataset.bias_aligned(idx)) for idx in tqdm(range(len(dataset)), desc="Extract subgroup ids")])
        
        return group_ids.to(device), n_subgroups