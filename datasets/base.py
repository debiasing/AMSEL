from typing import Any, Callable, Optional
import warnings
from os import makedirs, listdir
from os.path import isdir, join, isfile
import shutil
from PIL.Image import Image
from PIL.Image import open as open_img
import torch
from torch.utils.data import Dataset, Subset

class BiasDataset(Dataset):

    def __init__(self,
        root: str,
        split: str = "train",
        bias: str = "all",
        return_bias: bool = True,
        transform: Optional[Callable] = None):
        
        self.root = root
        self.split = split
        self.bias = bias
        self.return_bias = return_bias
        self.transform = transform

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def class_labels(self) -> list:
        return self._class_labels
    
    @property
    def bias_levels(self) -> list:
        return self._bias_levels

    def label(self, idx: int):
        raise NotImplementedError("Method has to be implemented by subclass!")

    def bias_attr(self, idx: int):
        raise NotImplementedError("Method has to be implemented by subclass!")
    
    def bias_aligned(self, idx: int) -> bool:
        # label+bias attr
        raise NotImplementedError("Method has to be implemented by subclass!")

    @staticmethod
    def default_transform(normalize=True, augmentation=False) -> Callable:
        return torch.nn.Identity()
    

def buffer_biased_dataset(dataset_class: type[BiasDataset]) -> type[BiasDataset]:
    """Adds buffering to an existing BiasDataset.

    This function allows to add buffering to an existing BiasDataset. With it,
    all inputs to the network (i.e., the first entry of __getitem__) are only
    calculated once and saved to disk. In all subsequent calls to __getitem__
    the input is directly loaded from disk. Currently, only images and tensors
    can be buffered.

    Args:
        dataset_class (type[BiasDataset]): BiasDataset that should be converted
            into a buffered variant.

    Returns:
        type[BiasDataset]: Buffered version of the dataset.
    """
    class BufferedBiasDataset(dataset_class):
        def __init__(self, buffer_dirname: str, buffer_overwrite: bool=False, after_buffer_transform: Optional[Callable] = None, *args, **kwargs):
            """Buffered version of a BiasDataset.

            If the buffer directory already exists, all files in it are treated
            as valid bufferings. However, you can explicitly set
            buffer_overwrite to True to overwrite old dataset configurations.

            Important Notes:
                1. Note that no information on the dataset configuration is
                   stored. We recommend to include the dataset configuration
                   (split, bias configuration, transformations, ...) in the name
                   of the buffer directory!
                2. When applying augmentations, the dataset changes in each
                   execution. Thus, these transformations have to be applied
                   after loading the input from disk. You can implement this via
                   after_buffer_transform.

            Args:
                buffer_dirname (str): Name of the directory in which buffered
                    files are stored. Note that this is only the name of the
                    directory itself, it will be placed into the root of the
                    inherited BiasDataset. We recommend to include the dataset
                    configuration (split, bias configuration, transformations,
                    ...) in the name of the buffer directory!
                buffer_overwrite (bool, optional): Whether the buffer folder
                    should be overwritten if it already exists. This is only
                    necessary if the dataset configuration has changed w.r.t.
                    the previous execution. Defaults to False.
                after_buffer_transform (Optional[Callable], optional):
                    Transformation that is applied after loading the input from
                    buffer. This is recommended for non-static transforms like
                    data augmentation.

            Raises:
                ValueError: Raised if the inputs to buffer are not either Pillow
                    images or tensors.
            """

            super().__init__(*args, **kwargs)

            # Output path.
            assert len(buffer_dirname) > 0, "Name of the buffer directory cannot be empty string"
            self.buffer_path = join(self.root, "buffered", buffer_dirname)
            self.after_buffer_transform = after_buffer_transform
            
            # If the path already exists and is not empty, we either treat the
            # files as valid or overwrite the directory.
            if isdir(self.buffer_path) and not len(listdir(self.buffer_path)) == 0:
                if buffer_overwrite:
                    warnings.warn("Buffer path already exists but because of buffer_overwrite=True it will be overwritten (" + self.buffer_path + ")!")
                    shutil.rmtree(self.buffer_path)
                else:
                    warnings.warn("Buffer path already exists. Because buffer_overwrite=False all files will be treated as valid data (" + self.buffer_path + ")!")
            makedirs(self.buffer_path, exist_ok=True)

            # Identify and validate input type.
            self.input_type = type(super().__getitem__(0)[0])
            if self.input_type not in [Image, torch.Tensor]:
                raise ValueError("Currently only buffering of PIL images and torch tensors is supported!")

        def __getitem__(self, idx) -> Any:
            # Model input.
            # If this index has been loaded once, load from disk. Otherwise,
            # load from wrapped dataset and save to disk for future executions.
            # Currently, only tensors and Pillow images are supported.
            if self.input_type == Image:
                # Load pillow image.
                buffer_file = join(self.buffer_path, "sample_" + str(idx) + ".png")
                if isfile(buffer_file):
                    x = open_img(buffer_file)
                else:
                    x = super().__getitem__(idx)[0]
                    x.save(buffer_file)
            else:
                # Load tensor image.
                buffer_file = join(self.buffer_path, "sample_" + str(idx) + ".pt")
                if isfile(buffer_file):
                    x = torch.load(buffer_file)
                else:
                    x = super().__getitem__(idx)[0]
                    torch.save(x, f=buffer_file)

            # Transformation.
            # Some transformations - such as data augmentation - are dynamic and
            # can, thus, only be applied after loading from disk.
            if self.after_buffer_transform is not None:
                x = self.after_buffer_transform(x)

            # Return inputs, targets and, if specified, also the bias attribute.
            y = self.label(idx)
            if self.return_bias:
                return x, y, self.bias_attr(idx)
            else:
                return x, y

    return BufferedBiasDataset

class BiasSubset(Subset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_transform = self.dataset.default_transform

    @property
    def root(self) -> str:
        return self.dataset.root
    
    @property
    def split(self) -> str:
        return self.dataset.split

    @property
    def bias(self) -> str:
        return self.dataset.bias
    
    @property
    def return_bias(self) -> bool:
        return self.dataset.return_bias
    
    @property
    def transform(self) -> Optional[Callable]:
        return self.dataset.transform

    @property
    def name(self) -> str:
        return self.dataset.name
    
    @property
    def version(self) -> str:
        return self.dataset.version
    
    @property
    def class_labels(self) -> list:
        return self.dataset.class_labels
    
    @property
    def bias_levels(self) -> list:
        return self.dataset.bias_levels

    def label(self, idx: int):
        return self.dataset.label(self.indices[idx])

    def bias_attr(self, idx: int):
        return self.dataset.bias_attr(self.indices[idx])
    
    def bias_aligned(self, idx: int) -> bool:
        return self.dataset.bias_aligned(self.indices[idx])