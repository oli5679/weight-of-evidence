import pandas as pd
from sklearn.datasets import make_classification
import re
import pytest

from tree_binner import TreeBinner


def create_test_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 6)])
    y = pd.Series(y, name="target")
    return X, y


@pytest.mark.parametrize(
    "category_type, dtype_check",
    [
        ("category", pd.api.types.is_categorical_dtype),
        ("string", pd.api.types.is_string_dtype),
        ("object", pd.api.types.is_object_dtype),
    ],
)
def test_tree_binner_fit_transform_categories(category_type, dtype_check):
    X, y = create_test_data()

    tree_binner = TreeBinner(max_depth=3, category_type=category_type)
    X_transformed = tree_binner.fit_transform(X, y)
    # Test if all columns have been transformed into categorical type
    for col in X_transformed.columns:
        assert dtype_check(X_transformed[col])


def test_tree_binner_fit_transform():
    X, y = create_test_data()

    tree_binner = TreeBinner(max_depth=3)
    X_transformed = tree_binner.fit_transform(X, y)
    for col in X_transformed.columns:
        assert pd.api.types.is_categorical_dtype(X_transformed[col])


def test_tree_binner_output_shape_and_bins():
    X, y = create_test_data()

    max_depth = 3
    tree_binner = TreeBinner(max_depth=max_depth)
    X_transformed = tree_binner.fit_transform(X, y)

    # Test if the number of rows in X_transformed is the same as in X
    assert X_transformed.shape[0] == X.shape[0]

    # Test if the number of unique bins per column is less than or equal to the maximum allowed by the tree depth
    max_bins = 2 ** max_depth
    for col in X_transformed.columns:
        unique_bins = X_transformed[col].nunique()
        assert unique_bins <= max_bins


def test_tree_binner_non_numeric_columns():
    X, y = create_test_data()

    # Add a non-numeric column to the input DataFrame
    X["non_numeric"] = ["category_{}".format(i % 3) for i in range(len(X))]

    tree_binner = TreeBinner(max_depth=3)
    X_transformed = tree_binner.fit_transform(X, y)

    # Test if the transformed DataFrame has the same columns as the input DataFrame
    assert set(X_transformed.columns) == set(X.columns)

    # Test if the non-numeric column remains unchanged
    assert (X["non_numeric"] == X_transformed["non_numeric"]).all()


def test_tree_binner_valid_bin_labels():
    X, y = create_test_data()

    tree_binner = TreeBinner(max_depth=3)
    X_transformed = tree_binner.fit_transform(X, y)

    # Define a regex pattern for the bin labels
    pattern = re.compile(r"^bin_\d+$")

    # Test if all elements in the transformed DataFrame match the bin label pattern
    for col in X_transformed.columns:
        for value in X_transformed[col]:
            assert pattern.match(value) is not None


def test_tree_binner_no_missing_values():
    X, y = create_test_data()

    tree_binner = TreeBinner(max_depth=3)
    X_transformed = tree_binner.fit_transform(X, y)

    # Test if the transformed DataFrame contains any missing values
    assert not X_transformed.isnull().any().any()