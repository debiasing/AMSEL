import torch
from datasets import BiasDataset
from models import BiasModel

class BaseShortcutRemover():
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self._name
    
    def remove_shortcut(self, model: BiasModel, dataset: BiasDataset, group_ids: torch.Tensor, device: str="cpu") -> BiasModel:
        raise NotImplementedError("Method has to be implemented by subclass!")