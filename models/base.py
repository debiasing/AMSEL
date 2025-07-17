from typing import Iterator
from os import listdir
from os.path import join, isfile, isdir
from copy import deepcopy
from natsort import natsorted
import torch
import torch.nn as nn

class BiasModel(nn.Module):
    def __init__(self, root, dataset_name, model_name):
        super().__init__()
        if not isdir(root):
            raise FileNotFoundError("The specified root folder could not be located at " + str(root) + ". Please create the root folder before initializing the model!")

        self.root = root
        self.model_path = join(root, dataset_name, "models", model_name)
        self.weights_path = join(self.model_path, "best_model.th")

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def checkpoints(self) -> list[str]:
        """Returns names of the model checkpoints.

        Returns the names of the model checkpoints sorted in asceding order with
        respect to their creation in the training process. That is, the first
        model checkpoint is returned first and the latest model checkpoint is
        returned last.
        The names of the model checkpoints can be passed to
        BiasModel.load_checkpoint(...) to load the corresponding checkpoint.

        Returns:
            list[str]: List of model checkpoints.
        """

        # Checkpoints are those files in the model directory that end in '.th'
        # and are not named 'best_model.th'. We attempt to use a sorting that
        # also works if enumerations in the filenames don't have leading zeros
        # (by applying a natural sorting).
        return natsorted([f for f in listdir(self.model_path) if f[-3:] == ".th" and f != "best_model.th"])
    
    def load_checkpoint(self, checkpoint: str):
        """Loads the weighhts of a given model checkpoint.

        Args:
            checkpoint (str): Name of the model checkpoint to load as provided
                by BiasModel.checkpoints.

        Raises:
            FileNotFoundError: Raised if no weights file could be located for
                the given checkpoint.
        """

        self.weights_path = join(self.model_path, checkpoint)
        if not isfile(self.weights_path):
            raise FileNotFoundError("The specified checkpoint does not exist: " + str(self.weights_path))
        self.model.load_state_dict(torch.load(self.weights_path)['state_dict'])

    def checkpoints_generator(self) -> Iterator[nn.Module]:
        """Generator for checkpointed models.

        Returns all checkpointed models for this model. The checkpointed models
        are returned in the order of creation during the training process.

        Yields:
            Iterator[nn.Module]: Generator that returns each checkpointed model.
        """
        
        # Construct generator that yields all checkpointed models from the
        # training. Since checkpoints is already ordered, we don't need to
        # re-order the checkpoints.
        for checkpoint in self.checkpoints:
            model = deepcopy(self)
            model.load_checkpoint(checkpoint)
            model = model.eval()

            yield model

    def get_feature_extractor(self) -> nn.Module:
        # We need to define a nn.Module for the feature extractor so that
        # .to(device) can be applied.
        # Return should be in eval mode.
        raise NotImplementedError("Method has to be implemented by subclass!")
    
    def get_auxiliary_feature_extractors(self) -> tuple[Iterator[nn.Module], list]:
        raise NotImplementedError("Method has to be implemented by subclass!")