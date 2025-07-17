from typing import Any
import inspect
import numpy as np
from copy import deepcopy
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import BiasDataset
from datasets.utils import BalancedSampler
from models import BiasModel
from models.utils import extract_features
from shortcut_removal.base import BaseShortcutRemover

class CustomLogisticRegression:
    """Custom logistic regressor that averages coefficient and intercepts.

    Following Izmailov et al., we do not just perform a single logistic
    regression. Instead we increase the robustness by repeating the logistic
    regression and averaging coefficients and intercepts over the multiple runs.
    In each run, we randomly sample a balanced dataset from the given inputs.
    Note that because we sample min{#samples in group} from each group, this
    means that for the smallest group we always sample the same samples.

    Balanced Sampling:
        By default, in each run a balanced subset of the dataset is drawn.
        However, you can also explicitly choose to draw unbalanced subsets. In
        this case, the size of each subset is n_classes*min{#samples in group}.
    """
    def __init__(self, num_retrains: int = 10, balanced_sampling_mode: str = "exact_under", balancing_factor: float = 1.0, *args, **kwargs):
        assert num_retrains > 0, "Number of retrains of the classifier has to be a positive integer!"

        self.num_retrains = num_retrains
        self.balanced_sampling_mode = balanced_sampling_mode
        self.balancing_factor = balancing_factor
        self.base_logistic_regression_classifier = LogisticRegression(*args, **kwargs)
        self.clf = None

    def fit(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
        assert len(group_ids) == len(X), "Number of group ids does not match number of samples!"

        coefs, intercepts = [], []
        seed = 0
        for _ in range(self.num_retrains):
            valid_sample = False
            while not valid_sample:
                if self.balanced_sampling_mode != "no_balanced_sampling":
                    # Balanced sampling.
                    # We sample a random balanced sample from given dataset.
                    sampler = BalancedSampler(X,
                                            mode=self.balanced_sampling_mode,
                                            balancing_factor=self.balancing_factor,
                                            labels=torch.from_numpy(group_ids),
                                            seed=seed)
                    indices = list(sampler)
                else:
                    # We subsample to the size of the balanced sampling set but do
                    # not perform any balancing.
                    warnings.warn("Implicitly subsampling to the size of the balanced dataset (but not performing any balancing). Please make sure that this is your desired behavior!")
                    
                    _, counts = np.unique(group_ids, return_counts=True)
                    n_samples = len(counts) * np.min(counts)
                    indices = np.arange(len(X))
                    indices = np.random.default_rng(seed).choice(indices, size=n_samples, replace=False)
                
                # Are all labels present?
                # If the subgroups are based on the labels, the subgroup
                # balancing ensures presence of all classes as long as there is
                # at least one sample of each class in the original dataset.
                # If, however, the subgroups are not based on the classes, the
                # subsampling might result in a subset where classes are missing
                # completely. This would cause problems later on because we
                # cannot merge the weights of the logistic regression if the
                # number of classes (and, thus, also the number of coefficients
                # of the logistic regression) differs between the runs.
                if set(y) == set(y[indices]):
                    valid_sample = True
                else:
                    print("Resampling because the subsampling with seed " + str(seed) + "(randomly) resulted in a split where at least one class is missing. Missing classes: " + str(set(y).difference(set(y[indices]))))
                
                # Increase seed.
                # Usually, we increase the seed after every retrain, but we also
                # need to increase it if not valid sample was drawn (as to not
                # draw the same sample again).
                seed += 1
            _X, _y = X[indices], y[indices]

            # Train logistic regressor on the given subset.
            clf = deepcopy(self.base_logistic_regression_classifier)
            clf.fit(_X, _y)
            coefs.append(clf.coef_)
            intercepts.append(clf.intercept_)
        
        # Create final classifier.
        # In principle, we could also create a new logistic regressor object
        # since we overwrite the coefficients anyway, but we can also just adapt
        # the one created last.
        self.clf = clf

        self.clf.coef_ = np.mean(coefs, axis=0)
        self.clf.intercept_ = np.mean(intercepts, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise ValueError("Classifier not fitted. Please call fit() before calling predict_proba()!")
        
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise ValueError("Classifier not fitted. Please call fit() before calling predict()!")
        
        return self.clf.predict(X)
    
    def __sklearn_is_fitted__(self):
        return (self.clf is not None)

class DeepFeatureReweighting(BaseShortcutRemover):
    """Deep Feature Reweighting as described by Kirichenko et al.

    Deep Feature Reweighting (DFR) originates from last layer re-training.
    Hpwever, in practice, we do not exactly re-train the last layer but perform
    a variant of logistic regression instead. This implementation mainly follows
    Izmailov et al. instead of Kirichenko et al.

    Logistic Regression:
        Instead of performing one logistic regression on the given features,
        Izmailov et al. sample 10 balanced subsets and train a logistic
        regression on each. Afterward, they average the coefficients and
        intercepts.
        Note that because for a balanced sample, we sample the number of the
        lowest group count samples from each group, we always sample all samples
        from the smallest group.

    Alternative Classifiers:
        By default, DFR uses logistic regression as its classifier. However, if
        you pass a sklearn classifier - or another classifier implementing the 
        resepective interface - as `classifier` attribute in the initialization
        this classifier is used. 

    References:
        P. Kirichenko, P. Izmailov, and A. G. Wilson, "Last Layer Re-Training is
           Sufficient for Robustness to Spurious Correlations," in The Eleventh
           International Conference on Learning Representations, 2023.
        P. Izmailov, P. Kirichenko, N. Gruver, and A. G. Wilson, "On Feature
           Learning in the Presence of Spurious Correlations," in Advances in
           Neural Information Processing Systems, 2022.
    """
    
    C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
    
    def __init__(self, c: float = 1.0, preprocess: bool = True, penalty: str = "l1", classifier: Any = None):
        self.preprocess = preprocess
        self._name = "Deep Feature Reweighting"

        # Classifier.
        # By default, a classifier with the desired properties is created. If 
        # a classifier is specified, we use that one instead.
        if classifier is None:
            self.c = c
            self.penalty = penalty
            self.clf = CustomLogisticRegression(num_retrains=20, penalty=self.penalty, C=self.c, solver="liblinear", verbose=1)
        else:
            # Sanity check that the user did not pass arguments for logistic
            # regression and a custom classifier. This is not perfect, we don't
            # catch cases where the users explicitly passes the default
            # arguments.
            assert c == 1.0 and penalty == "l1", "You can only specify either parameters for logistic regression or a custom classifier but you have supplied both!"
            self.c, self.penalty = None, None
            self.clf = classifier

    def remove_shortcut(self, model: BiasModel, dataset: BiasDataset, group_ids: torch.Tensor, device:str="cpu") -> nn.Module:
        # If choosing a different classifier compared to the default one, you
        # might want to manually balance your dataset.

        # Validate group ids.
        group_ids = group_ids.detach().cpu().numpy()
        assert len(group_ids) == len(dataset), "Each element of the dataset should be annotated with its individual group id"
        # For DFR, groups are expected to be balanced. That is, their element count
        # should differ by at most one.
        # However, the CustomLogisticRegression automatically balances the data
        # so we don't have to care about this for now.
        # group_counts = np.unique(group_ids, return_counts=True)[1]
        # if np.max(group_counts) - np.min(group_counts) > 1:
        #     warnings.warn("Groups are expected to be balanced but their element counts differ by more than one!")
        
        # Extract features.
        # Since we will return an updated model, we use deepcopy to make sure
        # that we are definitely independent of changes to the old one.
        dataloader = DataLoader(
            dataset,
            batch_size=100,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )
        feature_extractor = model.get_feature_extractor()
        feature_extractor = deepcopy(feature_extractor)
        feature_extractor = feature_extractor.eval()
        embeddings, labels = extract_features(
            feature_extractor=feature_extractor,
            dataloader=dataloader,
            extract_labels=True,
            device=device
        )
        X, y = [x.detach().cpu().numpy() for x in (embeddings, labels)]
        
        return self._remove_shortcut(model, feature_extractor, X, y, group_ids, device)
    

    def _remove_shortcut(self, model: BiasModel, feature_extractor: nn.Module, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray, device:str="cpu") -> nn.Module:
        # Preprocessing.
        if self.preprocess:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        # Fit classifier.
        # Generally, we expect the provided classifiers to follow the interface
        # of fit(X, y, group_ids). However, you can also pass standard sklearn
        # classifiers that do not use the group_ids. Here, we only pass X and y.
        # To make sure that this is intentional, we throw a warning.
        if "group_ids" in inspect.signature(self.clf.fit).parameters.keys():
            self.clf.fit(X, y, group_ids=group_ids)
        else:
            warnings.warn("Classifier.fit() does not take argument 'group_ids'! We expect the classifiers to follow the interface Classifier.fit(X, y, group_ids). Since the classifier does not support it, only passing X and y.")
            self.clf.fit(X, y)

        # Create new nn.module with logistic regression as output.
        class DFRModel(nn.Module):
            """Bias Model with Deep Feature Reweighting.

            Using the feature extractor of a bias model as a backbone, we
            add a logistic regression for classification.
            Note that even though DFR by default uses logistic regression, you
            can also pass different sklearn classifiers.
            """
            def __init__(self, model, feature_extractor, scaler, classifier, preprocess=True):
                super().__init__()
                self.name = model.name + " [DFR Version]"
                self.version = model.version

                self.feature_extractor = feature_extractor
                self.feature_extractor = feature_extractor.eval()
                self.scaler = scaler
                self.preprocess = preprocess
                self.classifier = classifier
                self.training = False

            def forward(self, x):
                device = x.device

                # Extract features.
                x = self.feature_extractor(x).detach().cpu().numpy()

                # Scaling and classification.
                if self.preprocess:
                    x = self.scaler.transform(x)
                preds =  self.classifier.predict(x)
                
                return torch.tensor(preds).to(device)
    
            def get_feature_extractor(self):
                return self.feature_extractor
            
            def train(self, mode: bool = True):
                warnings.warn("The DFRModel is not differentiable and cannot be trained. Thus, DFRModel.train(...) has no effect. Model is always in evaluation mode!")
                return self
            
        return DFRModel(model, feature_extractor, self.scaler, self.clf, self.preprocess).to(device)

