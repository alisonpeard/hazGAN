# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


class MangroveDamageModel(BaseEstimator):
    """Model to predict mangrove damage using ensemble of XGBoost and Linear Regression"""
    def __init__(self, scaling=True):
       # functions
       self.scaling = scaling
       self.base = LogisticRegression(fit_intercept=True)
       self.linear = IdentityModel()
       self.scaler = StandardScaler()
       self.transformer = Transformer()
       self.fitted = False
       self.metrics = {}

    def fit(self, X, y):
        """Fit the ensemble model."""
        # transform
        X = self.transformer.positive(X)
        X = self.transformer.transform(X)
        if self.scaling:
            X = self.scaler.fit_transform(X)
        # fit models
        self.base.fit(X, y)
        X_base = self.base.predict_proba(X)[:,1]
        self.linear.fit(X_base, y)
        self.fitted = True
    
    def predict(self, X):
        """Predict mangrove damages using the ensemble model."""
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        X = self.transformer.positive(X)
        X = self.transformer.transform(X)
        if self.scaling:
            X = self.scaler.transform(X)
        X_base = self.base.predict_proba(X)[:,1]
        y_pred = self.linear.predict(X_base)
        return y_pred

    def set_metrics(self, metrics:dict):
        self.metrics = metrics


class Transformer(object):
    """Make transformer an object to make it more flexible."""
    def positive(self, X):
        """Shift the data to be positive."""
        def shift(x):
            if np.min(x) <= 0:
                shifted = x + abs(np.min(x)) + 1
            else:
                shifted = x
            return shifted
        if len(X.shape) == 1:
            return shift(X)
        else:
            return np.apply_along_axis(shift, 0, X)
    
    def transform(self, X):
        return np.log10(X)
    
    def inverse_transform(self, X):
        return 10 ** X


class IdentityModel(BaseEstimator):
    """Use to ignore base or linear model in ensemble."""
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True

    def predict(self, X):
        return X.squeeze()
