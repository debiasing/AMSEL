# 📦 Experiment Outputs

This directory stores all generated files from the experiments run in the main `adaptive_model_selection.ipynb` notebook.

When you run the experiments, this folder will be populated with subdirectories named after the dataset being evaluated (e.g., `celeba/`, `chestx-ray14/`). Each subdirectory will contain the results for that specific dataset.

## Directory Structure

The expected structure after running the experiments will be:

```
outputs/
├── celeba/
│   ├── celeba - ... - lineplot.pdf
│   ├── celeba - ... - on test subset.tex
│   ├── celeba_data_erm-pred-and-conf.pkl
│   ├── celeba_data_tasks.pkl
│   └── ...
├── chestx-ray14/
│   └── ... (similar files for the ChestX-ray14 dataset)
└── README.md
```

## Generated Files

Here is a breakdown of the files you will find inside each dataset-specific subdirectory:

### 📊 Plots (`.pdf`)
Visualizations of the experimental results, ready for inclusion in your paper.
-   **`... - lineplot.pdf`**: The main performance plots (Accuracy, WSG Accuracy, etc.) comparing AMSEL to baselines across different test distributions ($\theta$).
-   **`... - dependency consensus level and balancing factor.pdf`**: A plot showing the relationship between the consensus score and the true balancing parameter $\theta$.
-   **`... - Trade-Off Between Accuracy ... .pdf`**: A scatter plot visualizing the trade-off between performance on biased and unbiased test sets for the candidate models.

### 📋 LaTeX Tables (`.tex`)
Formatted tables containing the precise numerical results.
-   **`... on test subset.tex`**: Detailed performance metrics for all models at each tested value of $\theta$.
-   **`... - AuC Accuracy.tex`**: A summary table with the Area Under the Curve (AUC) for the main accuracy plot, providing a single-number performance comparison.

### 🗂️ Cached Data (`.pkl`)
To accelerate subsequent runs, the notebook caches intermediate results. These files can be large and are not typically meant for version control.
-   **`..._erm-pred-and-conf.pkl`**: Caches the predictions and confidence scores from the baseline ERM models on the test set.
-   **`..._data_tasks.pkl`**: Caches the most time-consuming computations: the extracted features for all data splits and the family of trained candidate classifier heads ($h_\theta$).

> **💡 Note:** If you wish to re-run the experiments from scratch, simply delete the `.pkl` files for the corresponding dataset. The notebook will automatically regenerate them.