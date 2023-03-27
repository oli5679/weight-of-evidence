
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from typing import List
import scipy


def plot_feature_importance(
    var_names, coefficients, n=10
):
    """
    Find logit regression feature importance
    Args:
        var_names (series): variable names
        coefficients (series): regression coefficients
        n (integer): how many features to plot
    """
    coef_df = pd.DataFrame()
    coef_df["var_names"] = var_names
    coef_df["coef_vals"] = coefficients
    coef_df["abs_vals"] = np.abs(coef_df.coef_vals)
    coef_df = coef_df.set_index("var_names").sort_values(by="abs_vals", ascending=True)
    plt.figure(figsize=(4, 8))
    ax = coef_df.tail(n).coef_vals.plot.barh()
    plt.title(f"Top {n} features - logistic regression \n")
    plt.show()

def plot_bins(X: pd.DataFrame, y: pd.Series, space="probability", max_bins=15):
    """
    Plots target rates & counts for bins of splits
    Numeric column, split by 'splits' thresholds
    Categorical column, split by category 
    Args:
        X (dataframe): columns to be plotted
        y (target): target series
        splits (dictionary): splits to be applied to X
        space (string): space to plot target rates in
            NOTE - must be '%' or 'log-odds'
    """
    assert space in ["probability", "log-odds"]
    data = X.copy()
    data["target"] = y
    data["obs_count"] = 1
    for col in X.columns:
        not_numerical = pd.api.types.is_numeric_dtype(X[col])
        not_too_many_categories = X[col].nunique() <= max_bins
        if not_numerical and not_too_many_categories:
            agg = data.groupby().agg({"target": "mean", "obs_count": "sum"})
            plot_single_bin(agg, col, space)


def plot_single_bin(df, col, space):
    df["target rate %"] = df.target * 100
    df["target rate log-odds"] = scipy.special.logit(df.target)

    ax = df["obs_count"].plot.bar(alpha=0.5, color="grey")
    ax.legend(["obs count"])
    plt.ylabel("obs count")
    plt.xlabel("bin group")

    ax2 = df[f"target rate {space}"].plot.line(secondary_y=True, ax=ax)
    ax2.legend([f"target rate {space}"])
    plt.ylabel(f"target rate {space}")
    plt.title(f"Target rate vs. binned {col} \n")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()
