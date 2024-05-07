#%%
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose
#%%
n = 1
x = np.random.rand(n, 5, 5, 1024)
y = Conv2DTranspose(512, (2, 4), 1, activation='relu')(x)
z = Conv2DTranspose(256, (4, 4), 2, activation='relu')(y)
l = Conv2DTranspose(128, (4, 4), 1, activation='relu')(z)
o = Conv2DTranspose(2, (4, 4), 1, activation='relu')(l)
print(x.shape)
print(y.shape)
print(z.shape)
print(l.shape)
print(o.shape)
# %%
