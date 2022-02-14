import re

# to handle datasets
import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)

from regression_model.processing import features as pp
from regression_model.config.core import config



titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation',CategoricalImputer(
        imputation_method='missing', variables=config.model_config.categorical_vars)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(
    variables=config.model_config.numerical_vars)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median',variables = config.model_config.numerical_vars)),


    # Extract first letter from cabin
    ('extract_letter', pp.ExtractLetterTransformer(
    variables=config.model_config.cabin)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder',RareLabelEncoder(
    tol=0.05, variables=config.model_config.categorical_vars, replace_with='Rare', n_categories=1)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder',OneHotEncoder(
    drop_last=True, variables=config.model_config.categorical_vars)),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=config.model_config.C, random_state=config.model_config.random_state)),
])