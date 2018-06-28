from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import torch

class SubsetSampler(Sampler):  # pylint: disable=too-few-public-methods
    """
    Return subset of dataset. For example to enforce overfitting.
    """

    def __init__(self, indices, subset_size, random_subset=False, shuffle=True):
        assert subset_size <= len(indices), (
            f"The subset size ({subset_size}) must be smaller "
            f"or equal to the sampler size ({len(indices)}).")
        self._subset_size = subset_size
        self._shuffle = shuffle
        self._random_subset = random_subset
        self._indices = indices
        self._subset = None
        self.set_subset()

    def set_subset(self):
        """Set subset from sampler with size self._subset_size"""
        if self._random_subset:
            perm = torch.randperm(len(self._indices))
            self._subset = self._indices[perm][:self._subset_size]
        else:
            self._subset = torch.Tensor(self._indices[:self._subset_size])

    def __iter__(self):
        """Iterate over same or shuffled subset."""
        if self._shuffle:
            perm = torch.randperm(self._subset_size)
            return iter(self._subset[perm].tolist())
        return iter(self._subset)

    def __len__(self):
        return len(self._subset)


#
# Transforms
#
