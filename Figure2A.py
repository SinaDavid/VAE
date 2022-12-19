#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')

import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from matplotlib.lines import Line2D
pd.options.mode.chained_assignment = None  # default='warn'

url = "https://raw.githubusercontent.com/SinaDavid/VAE/main/LatentFeatures.csv"
df1 = pd.read_csv(url)

url1 = "https://raw.githubusercontent.com/SinaDavid/VAE/main/colors_correlation.csv"
df=pd.read_csv(url1)
df3=df.values.tolist()


cm = plt.cm.get_cmap('RdYlBu_r')
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
sc =ax1.scatter(df1['xx'], df1['yy'], df1['zz'], c=df3, edgecolors='white',cmap=cm)

ax1.set_xlabel('Latent feature 1')
ax1.set_ylabel('Latent feature 2')
ax1.set_zlabel('Latent feature 3')



norm1 = plt.Normalize(df1['correlation'].min(), df1['correlation'].max())
sm = plt.cm.ScalarMappable(norm=norm1, cmap=cm)
cbar = fig.colorbar(sm, ax=None, orientation='vertical') 

