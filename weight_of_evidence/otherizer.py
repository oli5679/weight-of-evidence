import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Otherizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.1):
        """
        :param threshold: The threshold below which a value is replaced with 'other'. Default is 0.1, i.e., any value
                          that appears in less than 10% of columns is replaced.
        """
        self.threshold = threshold
        self.common_strings = {}

    def fit(self, X, y=None):
        """
        Fit the transformer on the input data and identify common strings for each column.
        """
        N = len(X)
        for col in X.select_dtypes("object").columns:
            counts = pd.Series(X[col]).value_counts()
            common_strings = counts[(counts / N) >= self.threshold].index
            self.common_strings[col] = set(common_strings)
        return self

    def transform(self, X):
        """
        Transform the input data, replacing uncommon strings with 'other'.
        """
        X_transformed = pd.DataFrame(X.copy())
        for col, common_strings in self.common_strings.items():
            X_transformed[col] = np.where(X[col].isin(common_strings), X[col], "other")
        return X_transformed
