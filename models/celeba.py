from os.path import isfile, join
from pathlib import Path
from typing import Iterator
import gdown
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names

from models.base import BiasModel
from models.utils import FeatureExtractor, SoftmaxModel
from models.download import download_and_verify

class CelebAModel(BiasModel):
    """Loads a pre-trained ERM model for the CelebA dataset.

    This model is a ResNet50 trained with Empirical Risk Minimization (ERM) to
    serve as a baseline for debiasing experiments. Following the setup from
    Sagawa et al. (2020), it is trained on the CelebA dataset to predict the
    binary target attribute 'blond hair', which is known to be spuriously
    correlated with gender.

    The training procedure follows Izmailov et al. (2022), and the resulting
    model can be used as a fixed feature extractor for downstream debiasing
    methods.

    Encoding:
        Label (blond):      0 -> dark hair, 1 -> blond
        Bias (male):        0 -> female, 1 -> male  
    
    References:
        Z. Liu, P. Luo, X. Wang, and X. Tang, "Deep Learning Face Attributes
            in the Wild," in Proceedings of the IEEE International Conference
            on Computer Vision (ICCV), 2015. URL:
            http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
        S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang,
            "Distributionally Robust Neural Networks," in International
            Conference on Learning Representations, 2020.
        P. Izmailov, P. Kirichenko, N. Gruver, and A. G. Wilson, "On Feature
           Learning in the Presence of Spurious Correlations," in Advances in
           Neural Information Processing Systems, 2022.
    """
    def __init__(
        self,
        root: str,
        model: str = "izmailov_resnet50_erm_seed1",
        download: bool = False,
    ) -> None:
        """Initializes and loads the CelebA ERM model.

        Args:
            root (str): The root directory where datasets are stored and model
                        weights will be downloaded.
            model (str, optional): The specific pre-trained model checkpoint to
                load. Defaults to 'izmailov_resnet50_erm_seed1'.
                Available options are:
                    - 'izmailov_resnet50_erm_seed1'
                    - 'izmailov_resnet50_erm_seed2'
                    - 'izmailov_resnet50_erm_seed3'
                    - 'izmailov_resnet50_erm_seed4'
                    - 'izmailov_resnet50_erm_seed5'
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
        
        super().__init__(root=root, dataset_name="celeba", model_name=model)
        
        model_names = {
            "izmailov_resnet50_erm_seed1": "ResNet50 - CelebA ERM Seed 1 (retrained following Izmailov et al.)",
            "izmailov_resnet50_erm_seed2": "ResNet50 - CelebA ERM Seed 2 (retrained following Izmailov et al.)",
            "izmailov_resnet50_erm_seed3": "ResNet50 - CelebA ERM Seed 3 (retrained following Izmailov et al.)",
            "izmailov_resnet50_erm_seed4": "ResNet50 - CelebA ERM Seed 4 (retrained following Izmailov et al.)",
            "izmailov_resnet50_erm_seed5": "ResNet50 - CelebA ERM Seed 5 (retrained following Izmailov et al.)",
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
                "celeba.zip": ("https://github.com/debiasing/AMSEL/releases/download/v1.0.0/celeba.zip", "C689F85B8C5B08C8FC8B4762ECA1BA114068A308AC94DDBFB17B9E4E65F06153")
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
            raise FileNotFoundError("The CelebA model weights could not be located in " + str(self.root) + ". You can specify download=True to download the model weights.")
        
        # Load specified model.
        num_classes = 2
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(self.weights_path)['state_dict'])
    
    def forward(self, x):
        return self.model(x)
    
    def get_feature_extractor(self, downsampling_size: tuple = None, allow_upsampling=True) -> nn.Module:
        return FeatureExtractor(
            model=self.model,
            feature_layer="flatten",
            downsampling_size=downsampling_size,
            allow_upsampling=allow_upsampling
        ).eval()
    
    def get_auxiliary_feature_extractors(self, downsampling_size: tuple=None) -> tuple[Iterator[nn.Module], list]:
        """Returns auxiliary feature extractors.

        This function returns auxiliary feature extractors as needed for example
        to calculate prediction depth (PD). The placement of the feature
        extractors imitates Baldock et al. in adding a feature extractor after
        every summation at the end of a block (i.e., not the additions within a
        block) and the softmax layer. Baldock et al. also extract features after
        the first Group Norm operation. Since we have batch normalization
        instead of Group Norm, we place the feature extraction after the first
        batch normalization layer.

        References:
            R. J. N. Baldock, H. Maennel, and B. Neyshabur, "Deep Learning
                Through the Lens of Example Difficulty," in Advances in Neural
                Information Processing Systems, 2021. [Online].

        Args:
            downsampling_size (tuple, optional): If specified, shape to which
                the feature map shall be downsized. Defaults to None.

        Returns:
            tuple[Iterator[nn.Module], list]: Tuple of (feature extractors,
                feature layer names).
        """

        # Define layers of interest.
        # Here, we define all layers of interest except for the softmax layer.
        pd_layers = ["bn1"]
        eval_nodes = get_graph_node_names(self.model)[1]
        pd_layers = pd_layers + [node for node in eval_nodes if node[-6:] == ".1.add"]

        # Construct generator that yields the feature extractors. 
        def generate() -> Iterator[nn.Module]:
            # We can get all layers except the softmax layer with the build-in
            # feature extraction.
            for pd_layer in pd_layers:
                yield FeatureExtractor(self.model, feature_layer=pd_layer, downsampling_size=downsampling_size)

            # Add softmax layer.
            yield SoftmaxModel(self.model)

        return generate(), pd_layers + ["softmax"]