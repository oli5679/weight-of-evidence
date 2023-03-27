import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

from typing import Union


class TreeBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        category_type="category",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.binner_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
        )
        self.category_type = category_type
        self.bin_thresholds_ = {}

    def fit(self, X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array]):
        # Ensure X is a DataFrame
        X_ = pd.DataFrame(X.copy())

        # Fit a tree for each numeric column in X
        for col in X_.select_dtypes("number").columns:
            X_col = X_[[col]]
            self.binner_.fit(X_col, y)
            thresholds = sorted(
                self.binner_.tree_.threshold[self.binner_.tree_.threshold != -2]
            )
            labels = [
                "bin_{}".format(i) for i in range(1, len(thresholds) + 2)
            ]  # Create one more label than the number of thresholds            self.bin_thresholds_[col] = thresholds, labels
            self.bin_thresholds_[col] = thresholds, labels

        return self

    def transform(self, X: Union[pd.DataFrame, np.array]) -> pd.DataFrame:
        # Ensure X is a DataFrame
        X_transformed = pd.DataFrame(X.copy())

        # Apply the binning transformations to each column in X
        for col, thresh in self.bin_thresholds_.items():
            X_col = X_transformed[col]

            thresholds, labels = thresh

            # Use pd.cut to bin the data and replace the original column with the binned data
            X_transformed[col] = pd.cut(
                X_col,
                bins=np.concatenate(([-np.inf], thresholds, [np.inf])),
                labels=labels,
                include_lowest=True,
            ).astype(self.category_type)

        return X_transformed
