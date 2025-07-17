import torch
from datasets import BiasDataset
from models import BiasModel

class BaseSubgroupIdentifier():
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self._name

    def identify_subgroups(self, model: BiasModel, dataset: BiasDataset, device: str="cpu") -> tuple[torch.Tensor, int]:
        raise NotImplementedError("Method has to be implemented by subclass!")