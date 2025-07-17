from typing import Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

@torch.no_grad()
def extract_features(feature_extractor: nn.Module,
                      dataloader: DataLoader,
                      extract_labels: bool = False,
                      device: str = "cpu") -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Extract feature space representation for a given dataloader.

    Note:
        Even though the features are extracted on the specified cuda device,
        they are transferred to the cpu after extraction (of the respective
        batch) and detached to save GPU memory. Thus, this function is not
        differentiable.

    Args:
        feature_extractor (nn.Module): Module that returns the representation of
            a given input in the feature space.
        dataloader (DataLoader): Dataloader to extract features for.
        extract_labels (bool, optional): Whether not only the feature
            representation but also the corresponding labels should be extracted
            Defaults to False.
        device (str, optional): Specifies which cuda device is used for
            computation. Make sure that the feature_extractor is on the same
            device. Defaults to "cpu".

    Returns:
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: If labels are
            not extracted, only the features are returned. Otherwise a tuple of
            (features, labels) is returned. The features are a matrix of shape
            [n_samples, <flattened feature shape>], while the labels are a
            vector of shape [n_samples]. All resulting tensors are stored on the
            device "cpu".
    """

    # The features can be calculated without keeping track of gradients
    # (decorated with @torch.no_grad()). The resulting tensors are directly
    # moved to CPU to free the GPU memory space.
    features = [(feature_extractor(x[0].to(device)).cpu(), x[1].cpu())
                if extract_labels else feature_extractor(x[0].to(device)).cpu()
                for x in tqdm(dataloader, desc="Extract features")]
    if extract_labels:
        features, labels = zip(*features)

    features = torch.row_stack(features).flatten(start_dim=1)
    
    if extract_labels:
        return features, torch.cat(labels)
    else:
        return features

# The feature extractor has to be wrapped into a class to:
#   (1) Create a nn.Module,
#   (2) Abstract away from having to specify the feature layer after execution.
class FeatureExtractor(nn.Module):
    def __init__(
        self, model: nn.Module, feature_layer: str, downsampling_size: tuple = None, allow_upsampling=True,
    ):
        # Only performs downsampling if original is larger than downsampling
        # result or allow_upsampling=True.
        # Model is in eval mode.
        
        assert downsampling_size is None or len(downsampling_size) in [1, 2], \
            ("Downsampling size has to be provided as a tuple of (height, width)"
            " for two-dimensional downsampling and as tuple of (len,) for one-"
            "dimensional downsampling.")
        super().__init__()
        
        # Create feature extractor and set it to eval mode.
        self.feature_extractor = create_feature_extractor(
            model, return_nodes=[feature_layer]
        )
        self.feature_extractor = self.feature_extractor.eval()

        self.feature_layer = feature_layer
        self.downsampling_size = downsampling_size
        self.allow_upsampling = allow_upsampling

    def forward(self, x):
        x = self.feature_extractor(x)[self.feature_layer]
        # Only perform downsampling if it is desired.
        if self.downsampling_size is not None:
            # If not self.downsampling_if_larger only perform downsampling if it
            # actually results in a lower resolution. That is, in this case,
            # we avoid accidental upsampling.
            if self.allow_upsampling or (x.shape[-2]*x.shape[-1] > self.downsampling_size[0]*self.downsampling_size[1]):
                assert len(x.shape) >= 3, ("The feature dimensions before"
                    "downsampling is expected to be at least 2D (+batch dimension)."
                    " For one-dimensional feature vectors, deactivate downsampling.")
                x = F.interpolate(x, self.downsampling_size) 
        return x

class SoftmaxModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x, dim=1)
        return x