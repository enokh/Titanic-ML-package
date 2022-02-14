

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
            
        self.variables = variables


    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        
        df = X.copy()
            
        for feature in self.variables:
            df[feature] = df[feature].str[0]
        
        return df