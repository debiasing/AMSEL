from typing import Callable, Optional
from os.path import isdir, join
import torch
from torch.utils.data import Subset
from torchvision.datasets import CelebA as _CelebA
import torchvision.transforms as T

from datasets.base import BiasDataset

class CelebA(BiasDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        bias: str = "all",
        return_bias: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """CelebA dataset.

        CelebA is comprised of faces of celebrities. In total, the dataset
        presented by Liu et al. has 202,599 samples. In this implementation, we
        follow Sagawa et al. by setting the target attribute to "blond" and the
        considering the bias attribute "male". The data source accessed in this
        implementation is torchvision.datasets.

        Bias:
            The prediction target is "blond" and the bias attribute is "male".
            In the dataset, females are more likely to have blond hair than
            males. Thus, females with blond hair and males with dark hair are
            considered to be aligned with the bias. Contrariwise, females with
            dark hair and males with blond hair are considered to be conflicting
            examples to this bias.
            Note: Even though females are more likely to have blond hair than
            males, most females have dark hair. The exact dataset
            characteristics of the training dataset are as follows:
                Dark hair, female:  71629
                Blond, female:      22880
                Dark hair, male:    66874
                Blond, male:         1387

        Encoding:
            Label (blond):      0 -> dark hair, 1 -> blond
            Bias (male):        0 -> female, 1 -> male  

        Splits:
            Train split:        Biased as described above
            Validation split:   Biased as described above
            Test split:         Biased as described above
        
        Licensing:
            For the detailed licensing conditions of CelebA, should be indicated
            on the official download site. At the time of writing, this is
            http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Please check the
            up-to-date licensing before using this implementation to download
            and use CelebA.
            Please check the official license agreements for correctness, more
            detailed information and whether the information is up-to-date
            before using the dataset.
        
        References:
            Z. Liu, P. Luo, X. Wang, and X. Tang, "Deep Learning Face Attributes
               in the Wild," in Proceedings of the IEEE International Conference
               on Computer Vision (ICCV), 2015. URL:
               http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
            S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang,
               "Distributionally Robust Neural Networks," in International
               Conference on Learning Representations, 2020.

        Args:
            root (str):  File path in which the dataset should be stored. The
                folder has to exist (i.e., will not be created even if download
                is True).
            split (str, optional): Split of the dataset that is selected. Has to
                be one of "train", "val", or "test". Defaults to "train".
            bias (str, optional): Can be used to limit the data to only
                bias-aligned or bias-conflicting samples. Has to be one of
                "default", "all", "align" or "conflict". Defaults to "all".
            return_bias (bool, optional): Whether only (input, target) or
                (input, target, bias) is returned. Defaults to True.
            transform (Optional[Callable], optional): Transformation that is
                applied to the input image. Defaults to None.
            download (bool, optional): Whether the dataset should be downloaded
                if the root folder does not exist. Defaults to False.

        Raises:
            ValueError: Raised if an invalid split is selected.
            ValueError: Raised if an invalid bias configuration is selected.
            FileNotFoundError: Raised if the root directory does not exist.
            FileNotFoundError: Raised if the CelebA directory does not exist and
                download is False.
        """
        
        super().__init__(root=root, split=split, bias=bias, return_bias=return_bias, transform=transform)
        self._name = "CelebA"
        self._version = None
        self._class_labels = ["Dark Hair", "Blond"]
        self._bias_levels = ["Female", "Male"]

        if not split in ["train", "val", "test"]:
            raise ValueError("Invalid split specified. Please select one of \'train\', \'val\' and \'test\'.")
        if not self.bias in ["default", "all", "align", "conflict"]:
            raise ValueError("Invalid bias configuration specified. Please select one of \'default\', \'all\', \'align\' and \'conflict\'.")
        if self.bias == "default":
            self.bias = "all"
        if not isdir(self.root):
            raise FileNotFoundError("The specified root folder could not be located at " + str(self.root) + ". Please create the root folder before initializing the dataset!")
        if not download and not isdir(join(self.root, "celeba")):
            raise FileNotFoundError("The CelebA dataset could not be located in " + str(self.root) + ". You can specify download=True to download the dataset.")

        # Create desired version of CelebA.
        self.base_dataset = _CelebA(root=self.root, split="valid" if split == "val" else split, target_type="attr", transform=transform, target_transform=None, download=download)

        # Blond hair as target + gender bias.
        self._target_idx = self.base_dataset.attr_names.index("Blond_Hair")
        self._bias_idx = self.base_dataset.attr_names.index("Male")
        self._biases = self.base_dataset.attr[:, self._bias_idx]
        self._targets = self.base_dataset.attr[:, self._target_idx]

        # Limit to the desired bias configuration.
        self.dataset = self.base_dataset
        if self.bias == "all":
            self.dataset = self.base_dataset
        elif self.bias == "conflict":
            indices = torch.where(self._targets == self._biases)[0]
            self.dataset = Subset(self.base_dataset, indices)
            self._biases = self._biases[indices]
            self._targets = self._targets[indices]
        elif self.bias == "align":
            indices = torch.where(self._targets != self._biases)[0]
            self.dataset = Subset(self.base_dataset, indices)
            self._biases = self._biases[indices]
            self._targets = self._targets[indices]
        else:
            raise ValueError("Invalid bias mode selected. Please select one of \'all\', \'align\', and \'conflict\'.")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        img, attr = self.dataset[idx]
        if self.return_bias:
            return img, attr[self._target_idx], self.bias_attr(idx)
        else:
            return img, attr[self._target_idx]
        
    def label(self, idx: int) -> torch.Tensor: 
        """Returns the label of a dataset element.

        For CelebA, we consider the target "blond". Here, "0" corresponds to
        dark hair and "1" to blond.

        Args:
            idx (int): Dataset index.

        Returns:
            torch.Tensor: Label.
        """
        
        return self._targets[idx]

    def bias_attr(self, idx: int):
        """Returns the bias attribute of a dataset element.

        For CelebA, we consider the bias attribute "male". It is encoded as a
        binary attribute where "0" corresponds to a female and "1" to a male.

        Args:
            idx (int): Dataset index.

        Returns:
            torch.Tensor: Bias attribute.
        """

        return self._biases[idx]
    
    def bias_aligned(self, idx: int) -> bool:
        """Returns whether the element of the dataset is aligned with the bias.

        We consider the bias attribute "Male". In CelebA, females are more
        likely to have blond hair than males. Thus, females with blond hair and
        males with dark hair are aligned with the bias. Contrariwise, females
        with dark hair and males with blond hair are conflicting examples to
        this bias.
        Note: Even though females are more likely to have blond hair than males,
        most females have dark hair as well. Thus, both males and females are
        generally more likely to have dark hair than blond hair but blond males
        are far rarer than blond females.

        Distribution in the training dataset:
            Dark hair, female:  71629
            Blond, female:      22880
            Dark hair, male:    66874
            Blond, male:         1387

        Args:
            idx (int): Dataset index.

        Returns:
            bool: Whether the dataset element is aligned with the bias.
        """
        
        # Label: 0 -> dark hair, 1 -> blond hair
        # Bias attribute (gender): 0 -> female, 1 -> male
        return self._targets[idx] != self._biases[idx]

    @staticmethod
    def default_transform(normalize=True, augmentation=False) -> Callable:
        """Returns default transformation for this dataset.

        We follow the transformations applied by Izmailov et al.

        Args:
            normalize (bool, optional): Whether image statistics should be
                normalized to ImageNet statistics. Defaults to True.
            augmentation (bool, optional): Whether augmentation should be
                performed. Defaults to False.

        Returns:
            Callable: Transformation.
        """
        if not augmentation:
            transforms_list = [T.Resize([256, 256]),
                            T.CenterCrop([224, 224])]
        else:
            transforms_list = [T.RandomResizedCrop(
                                    (224, 224),
                                    scale=(0.7, 1.0),
                                    ratio=(0.75, 1.3333333333333333),
                                    interpolation=2),
                               T.RandomHorizontalFlip()]

        transforms_list += [T.ToTensor()]

        if normalize:
            transforms_list += [T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

        return T.Compose(transforms_list)