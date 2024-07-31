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
       # variables to be fitted
       self.scaler_fit = None
       self.base_fit = None
       self.linear_fit = None

    def fit(self, X, y):
        # scale and transform
        self.scaler_fit = self.scaler.fit(X)
        X = self.transformer.transform(X)
        y = self.transformer.transform(y)
        # fit models
        self.base_fit = self.base.fit(X, y)
        X_base = self.base.predict(X).reshape(-1, 1)
        self.linear_fit = self.linear.fit(X_base, y)
        return self 
    
    def predict(self, X):
        X = self.scaler.transform(X)
        X = self.transformer.transform(X)
        X_base = self.base.predict(X).reshape(-1, 1)
        y_pred = self.linear.predict(X_base)
        y_pred = self.transformer.inverse_transform(y_pred)
        return y_pred
    
  
class Transformer(object):
    """Make transformer an object to make it more flexible."""
    def transform(self, x):
        return np.log10(shift_positive(x))
    
    def inverse_transform(self, x):
        return x ** 10


def shift_positive(x):
  if np.min(x) <= 0:
    x += abs(np.min(x)) + 1
  return x