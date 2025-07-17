from typing import Callable, Optional
from os import mkdir
from os.path import join, isdir, isfile, join, basename, abspath, dirname
from tqdm import tqdm
import hashlib
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import gdown
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms as T

from datasets.base import BiasDataset

class InvalidChecksumError(Exception):
    """Raised if the checksum of a file does not match the expected checksum."""
    pass

def _md5sum(filename: str, chunk_size: int = 65536):
    if not isfile(filename):
        raise FileNotFoundError("File to calculate MD5 hash sum for could not be located: " + filename)

    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(chunk_size)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(chunk_size)

    return file_hash.hexdigest()

def _read_file_to_list(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

class ChestXRay14(BiasDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        bias: str = "all",
        return_bias: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """ChestX-ray14 dataset.

        The ChestX-ray14 dataset consists of chest x-rays collected by the NIH
        Clinical Center. In this implementation, we follow Murali et al. by
        considering the target to be 'Pneumothorax' and the bias attribute to be
        whether the patient has a chest tube. The chest tube labels are taken
        from Murali et al.

        Bias:
            In this dataset, the target is 'Pneumothorax' and the bias attribute
            is 'chest tubes'. Specifically, patients with diagnosed pneumothorax
            are more likely to have chest tubes than patients with no diagnosed
            pneumothorax. Despite that, it is not a desirable feature because
            the chest tubes are usually placed after the patient is diagnosed
            with pneumothorax (or it is suspected).
            Thus, patients with diagnosed pneumothorax and chest tubes as well
            as patients with no diagnosed pneumothorax and no chest tubes are
            considered bias-aligned samples. The reversed combinations of
            pneumothorax and chest tubes are considered to be bias-conflicting
            samples.
            The exact dataset characteristics of the training dataset are as
            follows:
                No diagnosed pneumothorax, no chest tubes:  102193
                No diagnosed pneumothorax, chest tubes:       4629
                Diagnosed pneumothorax, no chest tubes:       3891
                Diagnosed pneumothorax, chest tubes:          1407

        Encoding:
            Label (pneumothorax):   0 -> no diagnosed pneumothorax, 1 -> diagnosed
                                    pneumothorax
            Bias (chest tubes):     0 -> no chest tubes, 1 -> chest tubes  

        Dataset Splits:
            The NIH dataset only defines a training vs. test split (i.e., no
            validation split). We split the training environment into 80% for
            training and 20% for validation. Note that we split patient ids and
            not images. Thus, the resulting composition is only an approximate
            80/20 split.
            We explicitly do not follow the splits proposed by Murali et al.
            because it conflicts with the original train/test split and in their
            composition, images of one patient can be in multiple splits.

            Train split:        Biased as described above, sorted by
                                patient/image id!
            Validation split:   Biased as described above, sorted by
                                patient/image id!
            Test split:         Biased as described above, sorted by
                                patient/image id!

        Licensing:
            The ChestX-ray14 x-ray data and labels are provided by the NIH
            Clinical Center. They ask to provide a link to the official download
            site: https://nihcc.app.box.com/v/ChestXray-NIHCC, cite their CVPR17
            paper (Wang et al., see below) and acknowledge them as the data
            provider.
            Murali et al. have published their repository
            (https://github.com/batmanlab/TMLR23_Dynamics_of_Spurious_Features/)
            under MIT license.
            Please check the official license agreements for correctness, more
            detailed information and whether the information is up-to-date
            before using the dataset.

        References:
            X. Wang, Y. Peng, Le Lu, Z. Lu, M. Bagheri, and R. M. Summers,
               "ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks
               on Weakly-Supervised Classification and Localization of Common
               Thorax Diseases," in Proceedings of the IEEE Conference on
               Computer Vision and Pattern Recognition (CVPR), 2017.
            N. Murali, A. Puli, K. Yu, R. Ranganath, and K. Batmanghelich,
               "Beyond Distribution Shift: Spurious Features Through the Lens of
               Training Dynamics," Transactions on Machine Learning Research,
               vol. 2023, 2023.

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
            InvalidChecksumError: Raised if the checksum of one or more of the
                downloaded files does not match the expected checksum.
        """

        super().__init__(root=root, split=split, bias=bias, return_bias=return_bias, transform=transform)
        self._name = "NIH ChestX-ray14"
        self._version = None
        self._class_labels = ["No Diagnosed Pneumothorax", "Diagnosed Pneumothorax"]
        self._bias_levels = ["No Tubes", "Tubes"]

        if not self.split in ["train", "val", "test"]:
            raise ValueError("Invalid split specified. Please select one of \'train\', \'val\' and \'test\'.")
        if not self.bias in ["default", "all", "align", "conflict"]:
            raise ValueError("Invalid bias configuration specified. Please select one of \'default\', \'all\', \'align\' and \'conflict\'.")
        if self.bias == "default":
            self.bias = "all"
        if not isdir(self.root):
            raise FileNotFoundError("The specified root folder could not be located at " + str(self.root) + ". Please create the root folder before initializing the dataset!")

        # Download dataset if required.
        self.root = join(self.root, "chestx-ray14")
        self.root = abspath(self.root)
        if download and not isdir(self.root):
            # Create root folder.
            mkdir(self.root)

            # Download dataset.
            print("Downloading metadata...")
            # 1. Metadata used for chest tube labels.
            file_id = "12hMy_fSK6XlA4tyGFAKrPTfjDCofv33p"
            url = "https://drive.google.com/uc?id=" + file_id
            md5 = "md5:821233f79998a202514e492ffe315a0c"
            gdown.cached_download(url=url, format="xlsx", path=join(self.root, "nih_full.xlsx"), hash=md5)

            # 2. Metadata of prediction depth probing set.
            file_id = "1kVZLBR4XJJY7bowCO-fXBlahSGnNTGe8"
            url = "https://drive.google.com/uc?id=" + file_id
            md5 = "md5:7055f2d8628d37c7884b4179c955b33c"
            gdown.cached_download(url=url, format="xlsx", path=join(self.root, "nih_subset.xlsx"), hash=md5)

            # Pre-process metadata.
            # Note: We drop the column 'split' because we do not follow the
            # split of Murali et al. We cannot follow their split because they
            # don't use the offical train/val vs. test split of the dataset.
            print("Processing metadata...")
            metadata = pd.read_excel(join(self.root, "nih_full.xlsx"))
            metadata = metadata.drop(columns=["split"])
            metadata["filename"] = metadata["path"].apply(basename)
            metadata["patient_id"] = metadata["filename"].apply(lambda f: f[:8])
            metadata = metadata.sort_values("filename", ascending=True)
            metadata.to_csv(join(self.root, "nih_full_processed.csv"), index=False)

            train_val_ids = _read_file_to_list(join(dirname(__file__), "_chestx-ray14", "train_val.txt"))
            test_ids = _read_file_to_list(join(dirname(__file__), "_chestx-ray14", "test.txt"))
            train_val_metadata = metadata.query("patient_id in @train_val_ids")
            test_metadata = metadata.query("patient_id in @test_ids")
            train_val_metadata.to_csv(join(self.root, "nih_train_val_processed.csv"), index=False)
            test_metadata.to_csv(join(self.root, "nih_test_processed.csv"), index=False)

            print("Downloading dataset...")
            links = {
                "images_001.tar.gz": 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
                "images_002.tar.gz": 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
                "images_003.tar.gz": 'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
                "images_004.tar.gz": 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
                "images_005.tar.gz": 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
                "images_006.tar.gz": 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
                "images_007.tar.gz": 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
                "images_008.tar.gz": 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
                "images_009.tar.gz": 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
                "images_010.tar.gz": 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
                "images_011.tar.gz": 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
                "images_012.tar.gz": 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
            }
            for filename, link in tqdm(links.items()):
                urllib.request.urlretrieve(link, join(self.root, filename)) # download the zip file

            checksums = {
                "images_001.tar.gz": "fe8ed0a6961412fddcbb3603c11b3698", 
                "images_002.tar.gz": "ab07a2d7cbe6f65ddd97b4ed7bde10bf", 
                "images_003.tar.gz": "2301d03bde4c246388bad3876965d574", 
                "images_004.tar.gz": "9f1b7f5aae01b13f4bc8e2c44a4b8ef6", 
                "images_005.tar.gz": "1861f3cd0ef7734df8104f2b0309023b", 
                "images_006.tar.gz": "456b53a8b351afd92a35bc41444c58c8", 
                "images_007.tar.gz": "1075121ea20a137b87f290d6a4a5965e", 
                "images_008.tar.gz": "b61f34cec3aa69f295fbb593cbd9d443", 
                "images_009.tar.gz": "442a3caa61ae9b64e61c561294d1e183", 
                "images_010.tar.gz": "09ec81c4c31e32858ad8cf965c494b74", 
                "images_011.tar.gz": "499aefc67207a5a97692424cf5dbeed5", 
                "images_012.tar.gz": "dc9fda1757c2de0032b63347a7d2895c", 
            }
            if not all([checksums[f] == _md5sum(join(self.root, f)) for f in checksums.keys()]):
                raise InvalidChecksumError("Download Error: Checksums do not match - please retry or check links for updates.")
            for filename in checksums.keys():
                gdown.extractall(join(self.root, filename))

        if not isdir(self.root):
            raise FileNotFoundError(f"Could not locate ChestX-ray14 dataset at '{self.root}'. You can enable automatic downloading by setting download=True.")
        
        required_files = ["nih_train_val_processed.csv", "nih_test_processed.csv", ]
        for required_file in required_files:
            if not isfile(join(self.root, required_file)):
                raise FileNotFoundError(f"Could not locate '{join(self.root, required_file)}'. This can be caused by the automatic download failing. Delete '{self.root}' and set download=True to restart automatic downloading.")

        # Load metadata.
        if self.split in ["train", "val"]:
            # Split train/val data into training, validation.
            # Here, we have to make sure that patients cannot be in both the 
            # training and the validation split.
            self.metadata = pd.read_csv(join(self.root, "nih_train_val_processed.csv"))
            train_ids, val_ids = train_test_split(sorted(self.metadata["patient_id"].unique()), train_size=0.8, random_state=42)
            if self.split == "train":
                self.metadata = self.metadata.query("patient_id in @train_ids")
            else:
                self.metadata = self.metadata.query("patient_id in @val_ids")
        elif self.split == "test":
            self.metadata = pd.read_csv(join(self.root, "nih_test_processed.csv"))
        else:
            raise ValueError("Invalid split selected: " + str(self.split))
        self.metadata = self.metadata.sort_values("path")
        
        # Select appropriate bias split.
        if self.bias == "align":
            self.metadata = self.metadata.query("Pneumothorax == tube_label")
        if self.bias == "conflict":
            self.metadata = self.metadata.query("Pneumothorax != tube_label")

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        filename, target = self.metadata[["filename", "Pneumothorax"]].iloc[idx]
        image = Image.open(join(self.root, "images", filename)).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        # Load image.
        if self.return_bias:
            return image, target, self.bias_attr(idx)
        else:
            return image, target

    def label(self, idx: int) -> torch.Tensor: 
        """Returns the label of a dataset element.

        For ChestX-ray14, we consider the target "pneumothorax". Here, "0"
        corresponds to no diagnosed pneumothorax and "1" to a diagnosed
        pneumothorax.

        Args:
            idx (int): Dataset index.

        Returns:
            torch.Tensor: Label.
        """
        
        return self.metadata["Pneumothorax"].iloc[idx]

    def bias_attr(self, idx: int):
        return self.metadata["tube_label"].iloc[idx]
    
    def bias_aligned(self, idx: int) -> bool:
        return self.metadata["Pneumothorax"].iloc[idx] == self.metadata["tube_label"].iloc[idx]
    
    @staticmethod
    def default_transform(normalize=True) -> Callable:
        """Returns the default transformation for this dataset.

        The default transformation of this dataset follows the transformation
        performed by Murali et al.

        Args:
            normalize (bool, optional): Whether a normalization of the values shall be performed. Defaults to True.

        Returns:
            Callable: Default transformation.
        """
        # Transformations.
        # We always at least resize, convert to tensor and center crop. The
        # order is 1) resize convert to tensor + center crop, 2) normalization.
        
        class center_crop(object):
            def crop_center(self, img):
                _, y, x = img.shape
                crop_size = np.min([y,x])
                startx = x // 2 - (crop_size // 2)
                starty = y // 2 - (crop_size // 2)
                return img[:, starty:starty + crop_size, startx:startx + crop_size]
            
            def __call__(self, img):
                return self.crop_center(img)

        class normalization(object):
            def normalize_(self, img, maxval=255):
                img = (img)/(maxval)
                return img
            
            def __call__(self, img):
                return self.normalize_(img)  

        transforms_list = [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Lambda(center_crop())
        ]

        if normalize:
            transforms_list += [T.Lambda(normalization())]

        return T.Compose(transforms_list)
    

class MuraliChestXRay14(ChestXRay14):
    def __init__(self, *args, **kwargs):
        """ChestX-ray14 with dataset splitting according to Murali et al.

        This dataset implements dataset splitting following Murali et al. This
        includes the option to create a balanced train split for prediction
        depth calculation by setting the split to 'prediction_depth_trainset'.
        For a more detailed documentation, see the ChestXRay14 dataset.

        References:
            X. Wang, Y. Peng, Le Lu, Z. Lu, M. Bagheri, and R. M. Summers,
               "ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks
               on Weakly-Supervised Classification and Localization of Common
               Thorax Diseases," in Proceedings of the IEEE Conference on
               Computer Vision and Pattern Recognition (CVPR), 2017.
            N. Murali, A. Puli, K. Yu, R. Ranganath, and K. Batmanghelich,
               "Beyond Distribution Shift: Spurious Features Through the Lens of
               Training Dynamics," Transactions on Machine Learning Research,
               vol. 2023, 2023.

        Raises:
            ValueError: Raised if an invalid split is selected.
        """
        
        # If split is "prediction_depth_trainset", we create an dummy dataset
        # (here: test split) to overwrite. This does that we unecessarily open
        # one csv file.
        modified_kwargs = kwargs.copy()
        modified_kwargs["split"] = "test" if kwargs["split"] == "prediction_depth_trainset" else kwargs["split"]
        super().__init__(*args, **modified_kwargs)

        # Reset self.split because super().__init__() would set it to the dummy
        # argument passed above.
        self.split = kwargs["split"]

        if self.split == "prediction_depth_trainset":
            self.metadata = pd.read_excel(join(self.root, "nih_subset.xlsx"))
        elif self.split in ["train", "val", "test"]:
            self.metadata = pd.read_excel(join(self.root, "nih_full.xlsx"))
            self.metadata = self.metadata[self.metadata["split"] == self.split]
        else:
            raise ValueError("Invalid split specified. Please select one of \'train\', \'val\', \'test\' and \'prediction_depth_trainset\'.")

        self.metadata["filename"] = self.metadata["path"].apply(basename)
        self.metadata["patient_id"] = self.metadata["filename"].apply(lambda f: f[:8])
        self.metadata = self.metadata.sort_values("path")
        
        # Select appropriate bias split.
        if self.bias == "align":
            self.metadata = self.metadata.query("Pneumothorax == tube_label")
        if self.bias == "conflict":
            self.metadata = self.metadata.query("Pneumothorax != tube_label")