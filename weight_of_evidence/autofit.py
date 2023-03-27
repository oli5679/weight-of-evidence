from sklearn.pipeline import Pipeline
from category_encoders.woe import WOEEncoder
from sklearn.linear_model import LogisticRegression
from scipy.stats import sp_randint, sp_uniform
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import logging 

from weight_of_evidence.tree_binner import TreeBinner

AUTO_BIN_PIPELINE = Pipeline([
    ('tree_binner', TreeBinner(max_depth=5,min_samples_split=4,min_samples_leaf=4)),
    ('woe_encoder', WOEEncoder(regularization=1)),
    ('logistic_regression', LogisticRegression(max_iter=10_000,C=0.01))
])

AUTO_BIN_PARAMS_GRID = {
    'tree_binner__max_depth': sp_randint(2, 6),
    'tree_binner__min_samples_split': sp_randint(2, 11),
    'tree_binner__min_samples_leaf': sp_randint(1, 5),
    'tree_binner__max_leaf_nodes': [None] + list(sp_randint(10, 21).rvs(size=2)),
    'woe_encoder__regularization': sp_uniform(0, 1),
    'logistic_regression__C': np.logspace(-3, 2, 6),
    'logistic_regression__class_weight': [None, 'balanced']
}

def find_best_model_params(model: Pipeline, params_grid: dict, X: pd.DataFrame, y: pd.Series, cv:int=5, n:int=50, scoring:str='roc_auc'):
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=params_grid, 
        n_iter=n, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1, 
        verbose=1, 
        random_state=42
    )
    random_search.fit(X, y)
    print(f'Best score: {random_search.best_score_:.3f}')
    print(f'Best parameters: {random_search.best_params_}')
    return random_search