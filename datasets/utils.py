import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import Sampler

def get_balanced_subset(dataset: Dataset, mode: str = "over", balancing_factor: float = 1.0, labels: torch.Tensor = None, seed: int = None) -> Subset:
    """Samples a balanced Subset.

    Args:
        dataset (Dataset): _description_
        mode (str, optional): Over- or undersampling. You can also specifiy
            whether you want random balanced sampling or exact balanced
            sampling. Has to be one of 'over' (equal to 'random_over'),
            'under' (equal to 'random_under'), 'random_over',
            'random_under', 'exact_over' and 'exact_under'. Defaults to
            "over".
        balancing_factor (float, optional): Determines whether the sampling
            is balanced or a mixture of balanced and unbalanced sampling. A
            factor of 0.1 corresponds to balanced sampling while a factor of
            0.0 corresponds to unbalanced sampling. Currently only supported
            for exact balanced sampling.
        labels (torch.Tensor, optional): Element-wise labels of the dataset
            w.r.t. which the dataset shall be balanced. By default, labels are
            extracted as the second element returned by the dataloader but you
            can also specify them explicitly if you want to balance w.r.t. a
            different variable.
        seed (int, optional): Seed for a reproducible subset. If None, a
            pseudo-random Subset is drawn. Defaults to None.

    Returns:
        Subset: Balanced Subset.
    """

    indices = list(iter(BalancedSampler(dataset=dataset, mode=mode, balancing_factor=balancing_factor, labels=labels, seed=seed)))
    balanced_subset = Subset(dataset=dataset, indices=indices)

    return balanced_subset

class BalancedSampler(Sampler):
    """Custom over- and undersampling of datasets."""

    def __init__(self, dataset: Dataset, mode: str = "over", balancing_factor: float = 1.0, labels: torch.Tensor = None, seed: int = None):
        """Over- or undersampling sampler.

        General:
            For oversampling, we sample more samples of the minority classes to
            create a balanced dataset, while for undersampling, we sample fewer
            samples of the majority classes to create a balanced dataset.
            When performing oversampling, we sample with replacement and when
            performing undersampling, we sample without replacement.
        
        Number of Samples:
            When performing undersampling, we undersample the dataset to the
            size of n_classes*n_minority_class, while for oversampling, we
            sample a dataset of size n_classes*n_majority_class.

        Randomized Balancing vs. Exact Balancing:
            1. Random Balancing:
                By default, stochastic over- and undersampling is performed w.r.t.
                the class weights. They are calculated as 1/<samples of the class>.
                In this case, each sample contains approximately the same number of
                elements per class but not exactly.
                In the case of oversampling, we sample with replacement while for
                undersampling we sample without replacement.
                
                Note:
                    For undersampling, we have to re-weight the samples after each
                    sampling step because the class count changes!
            
            2. Exact Balancing:
                However, you can also choose to perform exact sampling. That is, we
                do not just sample approximately equally from each class but rather
                exactly equally. For undersampling, we simply draw these samples
                without replacement. For oversampling, we duplicate the class
                samples as often as necessary and randomly draw the remaining
                samples if the desired sample size is not divisible by the class
                count. Afterwards, the resulting dataset is shuffled.
        
        Balancing Factor:
            Sometimes you might not want a perfectly balanced dataset. For such
            cases, we support sampling from a distribution that is interpolated
            between the balanced sampling distribution and the unbalanced
            distribution. Note that this mode is only supported for exact
            sampling but not for random sampling.

        Args:
            dataset (Dataset): Dataset to sample from.
            mode (str, optional): Over- or undersampling. You can also specifiy
                whether you want random balanced sampling or exact balanced
                sampling. Has to be one of 'over' (equal to 'random_over'),
                'under' (equal to 'random_under'), 'random_over',
                'random_under', 'exact_over' and 'exact_under'. Defaults to
                "over".
            balancing_factor (float, optional): Determines whether the sampling
                is balanced or a mixture of balanced and unbalanced sampling. A
                factor of 0.1 corresponds to balanced sampling while a factor of
                0.0 corresponds to unbalanced sampling. Currently only supported
                for exact balanced sampling.
            labels (torch.Tensor, optional): Element-wise labels of the dataset
                w.r.t. which the dataset shall be balanced. By default, labels
                are extracted as the second element returned by the dataloader
                but you can also specify them explicitly if you want to balance
                w.r.t. a different variable.
            seed (int, optional): Seed for the a reproducible pseudo-randomized
                drawing of samples. If None, a random initial seed is used.
                Defaults to None.

        Raises:
            ValueError: Raised if an invalid mode is selected.
        """

        assert mode in ["over", "under", "random_over", "random_under", "exact_over", "exact_under"], "mode has to be one of \'over\', \'under\', \'random_over\', \'random_under\', \'exact_over\',  or \'exact_under\'"
        if mode in ["over", "under"]:
            mode = ("random_over" if mode == "over" else "random_under")
        # assert balancing_factor >= 0.0 and balancing_factor <= 1.0, "Invalid balancing factor, should be in [0,1] but is: " + str(balancing_factor) # We allow for > 1.0 and < 0.0 if valid proportions are generated.
        if mode in ["random_over", "random_under"] and balancing_factor != 1.0:
            raise ValueError("Currently, balancing factors other than 1.0 are only supported for exact balancing but not for random balancing!")

        self.mode = mode
        self.balancing_factor = balancing_factor
        self.dataset = dataset
        self.random_generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        self.np_random_generator = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        self.indices = np.arange(len(dataset))
        if labels is not None:
            self.labels = labels.detach().cpu().numpy()
        else:
            self.labels = np.array(
                [
                    xy[1]
                    for xy in tqdm(dataset, desc="Extracting labels for weighted sampling")
                ]
            )

        # Generate a list of weights.
        # One weight for each index, that defines how likely it is to draw this
        # index. Here, we use 1/<class count>.
        values, counts = np.unique(self.labels, return_counts=True)
        n_classes = len(values)
        lookup_counts = {v: c for v, c in zip(values, counts)}
        assert set(lookup_counts.keys()) == set(range(n_classes)), "Expecting ascending class indices with samples of all classes present but missing samples for the following class ids: " + str(set(range(n_classes)).difference(set(lookup_counts.keys())))
        self.class_weights = [1 / lookup_counts[c] for c in range(n_classes)] 

        if self.mode == "random_over":
            # Supersample each group to the size of the majority class.
            self.num_samples = int(n_classes * np.max(counts))
        elif self.mode == "random_under":
            # Subsample each group to the size of the minority class.
            self.num_samples = int(n_classes * np.min(counts))
        elif self.mode in ["exact_over", "exact_under"]:
            # When performing exact sampling, calculating the number of examples
            # is more difficult because we need to take the balancing factor
            # into account.

            # 1. Calculate proportions.
            # Here, we interpolate between the class proportions for perfectly
            # balanced sampling and completely unbalanced sampling.
            _, counts = np.unique(self.labels, return_counts=True)
            n_classes = len(counts)
            balanced_sampling_proportions = np.array([1/n_classes] * n_classes)
            unbalanced_sampling_proportions = counts / len(self.labels)

            self.sampling_proportions = self.balancing_factor * balanced_sampling_proportions + (1 - self.balancing_factor) * unbalanced_sampling_proportions
            assert (self.sampling_proportions >= 0.0).all() and (self.sampling_proportions <= 1.0).all(), "Invalid balancing factor specified, sampling proportions not in [0,1]! This might be caused by an invalid balancing factor outside of [0,1]. The generated dataset for balancing factor " + str(self.balancing_factor) + " would have to have sampling proportions of: " + str(self.sampling_proportions) + "."

            # 2. Calculate number of samples.
            # For oversampling, the number of samples is determined by the
            # majority class, while for undersampling the minority class
            # determines the total number of samples (we over-/undersample such
            # that all classes have this class count).
            # In both cases, we can calculate the total number of samples via:
            #     n_samples_reference_class = n_total * sampling_proportions_reference_class
            # <-> n_total = n_samples_reference_class / sampling_proportions_reference_class
            if self.mode == "exact_over":
                self.num_samples = int(np.max(counts) / self.sampling_proportions[np.argmax(counts)])
            else:
                self.num_samples = int(np.min(counts) / self.sampling_proportions[np.argmin(counts)])

            # For negative balancing factors, we cannot select all samples from
            # the minority subgroup (because then we don‘t have enough from all
            # the other groups) but rather have to properly select the number as
            # the maximum possible number of samples.
            if balancing_factor < 0 and self.mode == "exact_under":
                warnings.warn("For negative balancing factors, the promised number of samples (i.e., n_classes*n_minority_class) can not be held. Thus, we sample as many samples as possible when drawing without replacement!")   
                self.num_samples = int(np.min([count / sampling_proportion for count, sampling_proportion in zip(counts, self.sampling_proportions)]))
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.mode in ["exact_over", "exact_under"]:
            # We perform the sampling in two steps:
            #   1. Calculate the proportion of samples from each group in the
            #      resulting dataset as specified by self.balancing_factor. This
            #      already happens in the __init__().
            #   2. Actually sample the indices for the dataset.
            # For undersampling, we sample without replacement. For
            # oversampling, we first take all samples from the original dataset
            # and then oversample as much as necessary.

            # Sample the indices.
            # We first sample from each class. Afterwards, we permutate the
            # result.
            # Note that the resulting sample might contain slightly fewer
            # elements that self.num_samples if the desired total number of
            # samples is not divisible by n_classes.
            assert np.isclose(np.sum(self.sampling_proportions), 1.), "Invalid sampling proportions specified, should sum to 1!"

            class_labels, _ = np.unique(self.labels, return_counts=True)
            n_samples_per_class = [int(self.num_samples * proportion) for proportion in self.sampling_proportions]
            indices_per_class = [self.indices[self.labels == label] for label in class_labels]
            if self.mode == "exact_under":
                # For undersampling, we can simple draw without replacement.
                samples_per_class = [self.np_random_generator.choice(indices,
                                                                size=n_samples,
                                                                replace=False)
                                                                for n_samples, indices in zip(n_samples_per_class, indices_per_class)]
            else:
                # For oversampling, we want to repeat the indices as often as
                # possible. The remaining slots are filled with random elements
                # (but without replacement).
                # For this, we concatenate the shuffled lists (although we would
                # only need to shuffle the last one) and then select the desired
                # number of elements. Afterwards, the list is shuffled again.
                samples_per_class = [list(int(np.ceil(n_samples / len(indices))) * self.np_random_generator.permutation(indices).tolist())[:n_samples]
                                    for n_samples, indices in zip(n_samples_per_class, indices_per_class)]
            samples = self.np_random_generator.permutation(np.concatenate(samples_per_class))

        elif self.mode == "random_over":
            # For oversampling, we stochastically oversample elements according
            # to weights. For this, we need to enable sampling with
            # replacements.
            normalized_weights = np.array([self.class_weights[l] for l in self.labels]) / np.sum(np.array([self.class_weights[l] for l in self.labels]))
            samples = self.np_random_generator.choice(
                            self.indices,
                            size=self.num_samples,
                            p=normalized_weights,
                            replace=True,
                        )
        
        elif self.mode == "random_under":
            # For undersampling on the other hand, we draw without replacement.
            # NOTE: Because we sample without replacement, the class counts
            # change with each draw. Thus, we have to re-calculate the weights
            # in each run!
            samples = []
            indices_remaining = self.indices.copy()
            for _ in range(self.num_samples):
                labels = [self.labels[idx] for idx in indices_remaining]

                # Update weights.
                # Note that wee consider class labels based on self.labels to
                # avoid problems when no elements of a class are present anymore.
                # In this case, we also have to avoid dividing by zero.
                class_weights = [1. / class_count if class_count != 0. else 0. for class_count in [np.sum(labels==l) for l in np.unique(self.labels)]]
                weights = np.array([class_weights[l] for l in labels])
                normalized_weights = weights / np.sum(weights)

                # Draw next element.
                # We draw the next element and remove it from the list of remaining indices to avoid drawing the same sample multiple times.
                idx = self.np_random_generator.choice(
                             indices_remaining,
                             p=normalized_weights,
                         )
                samples.append(idx)
                indices_remaining = np.delete(indices_remaining, np.argwhere(indices_remaining == idx)[0,0])

            # Re-shuffle indices.
            # We re-shuffle the indices because of the varying number of samples
            # per class per step.
            samples = self.np_random_generator.permutation(samples)

        else:
            raise ValueError("Invalid mode selected: " + self.mode)

        # Return generator.
        return (s for s in samples)
    
class AdaptableLabelsDataset(Dataset):
    def __init__(self, dataset: Dataset, adapted_labels: torch.Tensor=None):
        self.dataset = dataset
        if adapted_labels is None:
            self.reset_labels()
        else:
            self.adapted_labels = adapted_labels.detach().cpu()

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple:
        x = list(self.dataset[idx])
        x[1] = self.adapted_labels[idx]
        return x
    
    def update_labels(self, adapted_labels: torch.Tensor):
        # Since the DataLoader has problems with labels already being on the
        # GPU, we move the labels to CPU.
        self.adapted_labels = adapted_labels.detach().cpu()

    def reset_labels(self):
        self.adapted_labels = torch.tensor([x[1] for x in tqdm(self.dataset, desc="Extracting labels")])