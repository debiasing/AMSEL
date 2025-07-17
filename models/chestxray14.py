from os.path import isfile, join
from pathlib import Path
from typing import Iterator
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet

from models.base import BiasModel
from models.utils import FeatureExtractor
from models.download import download_and_verify

class ChestXRay14Model(BiasModel):
    """Loads a pre-trained ERM model for the NIH ChestX-ray14 dataset.

    This model is a DenseNet121 trained with Empirical Risk Minimization (ERM)
    to serve as a baseline for debiasing experiments. It is trained on the
    ChestX-ray14 dataset to perform binary classification for the detection of
    pneumothorax.

    In this task, the presence of a chest drain, a treatment artifact, acts as a
    spurious feature. The training procedure and chest drain labels are adopted
    from Murali et al. (2023).

    Note that the model is not trained on the standard train-test split for
    ChestX-ray14, but rather follows the split performed by Murali et al.
    (2023).

    Encoding:
        Label (pneumothorax): 0 -> not diagnosed, 1 -> diagnosed
        Bias (chest tubes):   0 -> absent, 1 -> present

    Note:
        Following Murali et al. (2023), this model implementation includes
        specific modifications from a standard DenseNet121 architecture. It uses
        a single input channel for grayscale images and applies a final sigmoid
        activation for binary output.

    References:
        X. Wang, Y. Peng, Le Lu, Z. Lu, M. Bagheri, and R. M. Summers,
            "ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on
            Weakly-Supervised Classification and Localization of Common Thorax
            Diseases," in Proceedings of the IEEE Conference on Computer Vision
            and Pattern Recognition (CVPR), 2017.
        N. Murali, A. Puli, K. Yu, R. Ranganath, and K. Batmanghelich,
            "Beyond Distribution Shift: Spurious Features Through the Lens of
            Training Dynamics," Transactions on Machine Learning Research, vol.
            2023, 2023.
    """
    def __init__(
        self,
        root: str,
        model: str = "murali_dense121_erm_seed1",
        download: bool = False,
    ) -> None:
        """Initializes and loads the ChestX-ray14 ERM model.

        Args:
            root (str): The root directory where datasets are stored and model
                        weights will be downloaded.
            model (str, optional): The specific pre-trained model checkpoint to
                load. Defaults to 'murali_dense121_erm_seed1'.
                Available options are:
                    - 'murali_dense121_erm_seed1'
                    - 'murali_dense121_erm_seed2'
                    - 'murali_dense121_erm_seed3'
                    - 'murali_dense121_erm_seed4'
                    - 'murali_dense121_erm_seed5'
            download (bool, optional): If True, automatically downloads the
                                       model weights if they are not found
                                       locally. Defaults to False.

        Raises:
            ValueError: If an unsupported `model` name is specified.
            FileNotFoundError: If the model weights are not found at the
                               expected path and `download` is set to False.
            Exception: Re-raises exceptions from the download or file
                       extraction process on failure.
        """
        # MC-Dropout
        # Only one input channel (grayscale images)
        # Different forward pass to standard DenseNet.
        # Result is sigmoided, i.e., no pneumothorax is 0 and pneumothorax is 1.
        # For class weights, we follow Murali et al. because we also use their
        # train-test split. For the standard split, we would have to adapt the
        # class weights!
        super().__init__(root=root, dataset_name="chestx-ray14", model_name=model)
        
        model_names = {
            "murali_dense121_erm_seed1": "DenseNet121 - NIH ChestX-ray14 (retrained following Murali et al.)",
            "murali_dense121_erm_seed2": "DenseNet121 - NIH ChestX-ray14 (retrained following Murali et al.)",
            "murali_dense121_erm_seed3": "DenseNet121 - NIH ChestX-ray14 (retrained following Murali et al.)",
            "murali_dense121_erm_seed4": "DenseNet121 - NIH ChestX-ray14 (retrained following Murali et al.)",
            "murali_dense121_erm_seed5": "DenseNet121 - NIH ChestX-ray14 (retrained following Murali et al.)",
        }
        if not model in model_names.keys():
            raise ValueError("Invalid model specified. Please specify one of the following options: " + str(", ".join(list(model_names.keys()))) + ".")
        
        self._name = model_names[model]
        self._version = None

        if not isfile(self.weights_path) and download:
            # Create .../dataset/models/ folder.
            models_dir = Path(self.model_path).parent.absolute()
            models_dir.mkdir(parents=True, exist_ok=True)

            # Download and verify.
            files_to_download = {
                "chestx-ray14-1.zip": ("https://github.com/debiasing/AMSEL/releases/download/v1.0.0/chestx-ray14-1.zip", "A5185AF055475153177948A931791AC379A470398A0DB7A00CD5E804AE98B72C"), 
                "chestx-ray14-2.zip": ("https://github.com/debiasing/AMSEL/releases/download/v1.0.0/chestx-ray14-2.zip", "4067B64038ECE27D6C4C0A4DD93BF8FBB8E8226E03E518080A9C2752BC96409B"), 
            }
            for filename, (url, checksum) in files_to_download.items():
                download_and_verify(
                    url=url,
                    filename=join(models_dir, filename),
                    expected_checksum=checksum,
                    hash_algorithm="sha256"
                )
        
            # Unzip the zip files.
            for filename in files_to_download.keys():
                gdown.extractall(join(models_dir, filename))

        if not isfile(self.weights_path):
            raise FileNotFoundError("The ChestX-ray14 model weights for \'" + model + "\' could not be located in " + str(self.root) + ". You can specify download=True to download the model weights.")
        
        # Load specified model.
        # DenseNet 121
        self.config = {
            "num_classes": 1,
            "drop_rate": 0.00,
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
            "num_init_features": 64,
            "in_channels": 1,
        }
        self.model = DenseNet(**{k:v for k, v in self.config.items() if not k =="in_channels"})
        self.model.features[0] = nn.Conv2d(self.config["in_channels"], self.config["num_init_features"], kernel_size=7, stride=2, padding=3, bias=False)
        self.model.load_state_dict(torch.load(self.weights_path)['state_dict'], strict=True)

    def forward(self, x):  
        # Murali et al. mention that for some experiments, they perform
        # Monte-Carlo dropout (training explicitly set to True and drop rate
        # always != 0 in F.dropout(...). However, for the ChestXRay14
        # experiment, they set thé drop rate to zero. Thus, we can simply
        # perform a normal forward pass through all layers of the wrapped
        # DenseNet121.
        x = self.model(x)

        # Implementation following Murali et al.:
        # x = self.model.features(x)
        # x = F.relu(x, inplace=True)
        # x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        # # Murali et al. explicitly set training to True. That is because the
        # # drop_rate is always != 0, this is Monte-Carlo dropout.
        # x = F.dropout(x, p=self.config["drop_rate"], training=True) 
        # x = self.model.classifier(x)

        # When applying the sigmoid function, we don't need to worry about
        # dimensions because the final layers has only a single neuron. Thus, it
        # is also only reasonable for the output to have shape [batch_size].
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        
        return x
    
    def get_feature_extractor(self) -> nn.Module:
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                x = self.model.features(x)
                x = F.relu(x, inplace=True)
                x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
            
                return x    
        
        return FeatureExtractor(self.model).eval()
    
    def get_auxiliary_feature_extractors(self, downsampling_size: tuple=None) -> tuple[Iterator[nn.Module], list]:
        # Following Murali et al., we extract the features from the final layer
        # of every second DenseLayer. However, Murali et al. consider the
        # torch.cat(block_input, block_output) at the end of a block to belong
        # to the block itself. Pytorch, on the other hand, considers this layer
        # to belong to the next block. Thus, we have to adapt to this.

        # Define layers of interest.
        features = self.model.features
        pd_layers = []
        for block, block_name in zip([features.denseblock1, features.denseblock2, features.denseblock3, features.denseblock4], ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]):
            for idx, layer in enumerate(block):
                if idx%2 == 1:
                    pd_layers.append("features." + block_name + "." + layer + ".cat")
            # For unkown reasons, Murali et al. don't extract the final
            # torch.cat operation of the DenseBlock.
            # if block_name != "denseblock1":
            #     pd_layers.append("features." + block_name + ".cat")
        
        # Construct generator that yields the feature extractors.
        def generate() -> Iterator[nn.Module]:
            for pd_layer in pd_layers:
                # The feature extractor has to be wrapped into a class to:
                #   (1) Create a nn.Module,
                #   (2) Abstract away from having to specify the pd_layer.
                yield FeatureExtractor(self.model, pd_layer, downsampling_size)
        return generate(), pd_layers
    
    """
    def pd_layers(self) -> list[str]:
        features = self.model.features
        pd_layers = []
        for block, block_name in zip([features.denseblock1, features.denseblock2, features.denseblock3, features.denseblock4], ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]):
            for idx, layer in enumerate(block):
                if idx%2==0:
                    pd_layers.append("features." + block_name + "." + layer)
        
        return pd_layers
    
    def get_pd_feature_extractor(self, pd_layer: str, downsampling_size: tuple=None) -> nn.Module:
        feature_extractor = create_feature_extractor(self.model, return_nodes=[pd_layer])

        class FeatureExtractor(nn.Module):
            def __init__(self, feature_extractor):
                super().__init__()
                self.feature_extractor = feature_extractor

            def forward(self, x):
                x = self.feature_extractor(x)[pd_layer]
                if downsampling_size is not None:
                    x = F.interpolate(x, downsampling_size)
                return x
            
        return FeatureExtractor(feature_extractor)
    """