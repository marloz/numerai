import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def create_fold_groups(df, n_splits):
    zeroes = np.zeros(df.shape[0])
    zeroes_split = np.array_split(zeroes, n_splits)
    groups_list = [np.full(len(arr, ), val)
                   for val, arr in enumerate(zeroes_split)]
    return np.hstack(groups_list)


class CustomSplitter(GroupKFold):
    """ Sklearn type data splitter that is compatible with grid or randomized cv searches."""

    def __init__(self, n_splits):
        super().__init__(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        """ Uses GroupKFold splitter to generate train/test indices.
        Y can be both pd.Series or np.array, depending on where in the pipeline it is used.
        """
        for train_idx, test_idx in super().split(X, y, groups):
            y_train = pd.Series(y.iloc[train_idx].values, index=train_idx, name='target')
            train_idx = self._downsample_train_idx(y_train)
            yield train_idx, test_idx

    def _downsample_train_idx(self, y_train):
        """ Not all training indices are included,
        first smallest class count is found, then larger classes are downsampled"""
        least_frequent_class_count = y_train.value_counts().min()
        return (pd.DataFrame(y_train)
                .groupby('target')
                .sample(n=least_frequent_class_count, replace=False, random_state=42)
                .index)
