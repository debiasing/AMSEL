# 🗂️ Datasets and Models

This directory is the central storage location for all datasets and pre-trained models used in the AMSEL experiments.

It is designed to be **populated automatically** when you run an experiment for the first time. The framework will download the necessary image files, annotations, and baseline ERM model weights into this folder.

## Directory Structure

The expected structure after a successful download will be:

```
data/
├── celeba/
│   ├── img_align_celeba/
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── models/
│   │   ├── izmailov_resnet50_erm_seed1/
│   │   │   └── best_model.th
│   │   └── ...
│   ├── identity_CelebA.txt
│   ├── list_attr_celeba.txt
│   ├── list_bbox_celeba.txt
│   ├── list_eval_partition.txt
│   └── list_landmarks_align_celeba.txt
│
└── chestx-ray14/
    ├── images/
    │   ├── 00000001_000.png
    │   └── ...
    ├── models/
    │   ├── murali_dense121_erm_seed1/
    │   │   └── best_model.th
    │   └── ...
    ├── nih_full.xlsx
    ├── nih_subset.xlsx
    ├── nih_full_processed.csv
    ├── nih_train_val_processed.csv
    └── nih_test_processed.csv
```

## Folder Contents

Here is a breakdown of the files and folders you will find inside this directory:

### 🖼️ Image Data
These folders contain the raw image files for the respective datasets.
-   **`celeba/img_align_celeba/`**: Contains over 200,000 celebrity face images.
-   **`chestx-ray14/images/`**: Contains over 100,000 frontal-view chest X-ray images.

### 📝 Annotations and Metadata
These files provide the ground-truth labels, group information (e.g., gender, presence of chest drains), and data splits required for the experiments.
-   **`celeba/*.txt`**: Official annotation files for the CelebA dataset, detailing attributes, identity, and evaluation partitions.
-   **`chestx-ray14/*.xlsx`, `*.csv`**: Annotation files for the ChestX-ray14 dataset, including the original NIH data and our pre-processed CSVs with spurious group labels.

### 🤖 Pre-trained ERM Models
These folders contain the weights for the baseline ERM models that serve as the fixed feature extractors for our method.
-   **`.../models/[model_name]/`**: Each subdirectory corresponds to a specific baseline ERM model (e.g., `izmailov_resnet50_erm_seed1`).
-   **`.../best_model.th`**: The PyTorch file containing the trained weights for that specific model.

> **💡 Note on Manual Setup:** If the automatic download fails (e.g., due to network issues or server rate limits), please refer to the **Manual Data & Model Setup** section in the main [`README.md`](../README.md) for detailed instructions on how to populate this directory yourself.