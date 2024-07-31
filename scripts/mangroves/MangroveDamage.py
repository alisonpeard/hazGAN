# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

  
class MangroveDamageModel(object):
    """Model to predict mangrove damage using ensemble of XGBoost and Linear Regression"""
    def __init__(self):
       # functions
       self.base = XGBRegressor(n_estimators=15)
       self.linear = LinearRegression()
       self.scaler = StandardScaler()
       self.transformer = Transformer()
       self.fitted = False

    def fit(self, X, y):
        # transform
        X = self.transformer.positive(X)
        X = self.transformer.transform(X)
        X = self.scaler.fit_transform(X)
        y = self.transformer.transform(y)
        # fit models
        self.base.fit(X, y)
        X_base = self.base.predict(X).reshape(-1, 1)
        self.linear.fit(X_base, y)
        self.fitted = True
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        X = self.transformer.positive(X)
        X = self.transformer.transform(X)
        X = self.scaler.transform(X)
        X_base = self.base.predict(X).reshape(-1, 1)
        y_pred = self.linear.predict(X_base)
        y_pred = self.transformer.inverse_transform(y_pred)
        return y_pred
    
  
class Transformer(object):
    """Make transformer an object to make it more flexible."""
    def positive(self, X):
        """Shift the data to be positive."""
        def shift(x):
            if np.min(x) <= 0:
                x += abs(np.min(x)) + 1
            return x
        if len(X.shape) == 1:
            return shift(X)
        else:
            return  np.apply_along_axis(shift, 0, X)
    
    def transform(self, X):
        return np.log10(X)
    
    def inverse_transform(self, X):
        return 10 ** X

