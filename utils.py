from os.path import join, isdir
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import dill as pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def worst_subgroup_accuracy(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> float:
    assert len(y_true) == len(y_pred) == len(group_ids), "Each element of the dataset should be annotated with its individual group id"

    # Evaluate each subgroup separately.
    subgroup_accuracies = []
    for group_id in np.unique(group_ids):
        subgroup_accuracies.append(accuracy_score(y_true[group_ids == group_id], y_pred[group_ids == group_id]))

    return np.min(subgroup_accuracies)

def subgroup_accuracy(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray, aggregation="none") -> float:
    assert len(y_true) == len(y_pred) == len(group_ids), "Each element of the dataset should be annotated with its individual group id"

    # Evaluate each subgroup separately.
    subgroup_accuracies = []
    for group_id in np.unique(group_ids):
        subgroup_accuracies.append(accuracy_score(y_true[group_ids == group_id], y_pred[group_ids == group_id]))
    subgroup_accuracies = np.array(subgroup_accuracies)

    if aggregation == "none":
        return subgroup_accuracies
    elif aggregation == "min":
        return np.min(subgroup_accuracies)
    elif aggregation == "max":
        return np.max(subgroup_accuracies)
    elif aggregation == "mean":
        return np.mean(subgroup_accuracies)
    elif callable(aggregation):
        return aggregation(subgroup_accuracies)
    else:
        raise ValueError("Invalid aggregation mode specified. Please specify one of 'none', 'min', 'max', 'mean', or a custom callable that can be applied.")

def load_model(guild_home: str, run: str, model_name: str = "debiased_model.pkl") -> nn.Module:
    """_summary_

    Args:
        guild_home (str): Root directory of the guild run.
        run (str): Run from which to load the model.
        model_name (str, optional): Filename of the model to load. Defaults to "debiased_model.pkl".

    Raises:
        FileNotFoundError: Raised if the run directory does not exist.

    Returns:
        nn.Module: PyTorch model.
    """

    # Verify run directory.
    run_dir = join(guild_home, "runs", run)
    if not isdir(run_dir):
        raise FileNotFoundError("Run directory could not be located: " + str(run_dir))
    
    # Load model.
    with open(join(run_dir, model_name), 'rb') as f:
        model = pickle.load(f)
    
    return model

def get_runs_for_base_model(experiment_parameters: pd.DataFrame, base_model: str = "izmailov_resnet50_erm_seed1") -> list:
    """Returns list of runs corresponding to the given base model.
    
    Returns all runs corresponding to the given base model. The runs are sorted
    by ascending balancing factor.
    
    Args:
        experiment_parameters (pd.DataFrame): Experiment configuration as given
            guild.runs().guild_flags().
        base_model (str, optional): Name of the base model as used in
            experiment_parameters["model_name"]. Defaults to
            "izmailov_resnet50_erm_seed1".

    Returns:
        list: List of runs.
    """

    experiment_parameters = experiment_parameters.copy()
    
    experiment_parameters = experiment_parameters[experiment_parameters["model_name"] == base_model]
    experiment_parameters = experiment_parameters.sort_values(by=["balancing_factor"])
    
    
    return experiment_parameters["run"].to_list()

@torch.no_grad()
def extract_model_outputs_per_balancing_factor(experiment_parameters: pd.DataFrame, dataloader: DataLoader, base_model: str, guild_home: str, device: str) -> np.ndarray:
    # Get runs corresponding to the specified base model.
    runs = get_runs_for_base_model(experiment_parameters=experiment_parameters, base_model=base_model)

    # We know that all runs belong to the same base model. Thus, we only need to
    # extract the features of that base model *once*.
    # For this, we simply load the ERM model from the first of the specified
    # runs.
    model = load_model(guild_home, runs[0], model_name="erm_model.pkl").to(device)
    feature_extractor = model.get_feature_extractor().eval()

    features = [feature_extractor(x[0].to(device)).cpu() for x in tqdm(dataloader, desc="Extract features")]
    features = torch.row_stack(features).flatten(start_dim=1).numpy()
    del feature_extractor

    # For each balancing factor, predict probabilities based on the
    # extracted features.
    outputs_per_factor = []
    for run in tqdm(runs, desc="Performing classification per balancing factor"):
        model = load_model(guild_home, run, model_name="debiased_model.pkl").to(device)
        x = np.copy(features)

        # Preprocessing and classification.
        if model.preprocess:
            x = model.scaler.transform(x)
        x = model.classifier.predict_proba(x)
        outputs_per_factor.append(x)
    outputs_per_factor = np.array(outputs_per_factor)

    return outputs_per_factor

class FuncClassifier:
    def __init__(self, func, n_classes):
        self.func = func
        self.n_classes = n_classes
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        # If binary, simply apply and check whether >= 0.
        # Otherwise, we first group by class labels and then apply to the
        # grouped variant. Afterwards, we predict via argmax.
        X = self.predict_proba(X)
        if self.n_classes == 2:
            return (X >= 0.5).astype(int)
        else:
            # Predict via argmax.
            return np.argmax(X, axis=1)
    
    def predict_proba(self, X):
        # For binary classification, we directly apply the function because the
        # data only contains a single logit per balancing factor. For
        # multi-class classification, we first group by class labels and then
        # apply to the grouped variant. Afterwards, we predict via argmax.
        if self.n_classes == 2:
            return np.apply_along_axis(self.func, 1, X)
        else:
            # Reshape from [n_samples, n_balancing_factors x n_classes] to 
            # [n_samples, n_balancing_factors, n_classes]
            assert X.shape[1] % self.n_classes == 0, "The second data dimension is expected to be n_balancing_factors x n_classes but your dimension is not divisible by the specified number of classes: shape " + str(X.shape) + " for n_classes " + str(self.n_classes)
            X = np.array(np.split(X, X.shape[1] // self.n_classes, axis=1))
            X = X.transpose((1,0,2))

            # Apply function.
            return np.apply_along_axis(self.func, 1, X)

def extract_predictions(X: np.ndarray, n_classes: int) -> np.ndarray:
    if n_classes == 2:
        # In the case of a binary problem, we only consider one logit (namely
        # that of class 1). Thus, we can perform a simple thresholding to
        # generate class predictions.
        predictions = (X >= 0.5).astype(int)
    else:
        # For multi-class problems, we use a small trick: The FuncClassifier
        # splits the one-dimensional feature vectors back into the outputs of
        # each individual classifier. Then, we only have to apply np.argmax to
        # the extracted probabilites.
        clf = FuncClassifier((lambda x: x), n_classes=n_classes)
        predictions = clf.predict_proba(X)
        predictions = np.argmax(predictions, axis=2)

    return predictions

