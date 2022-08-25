#!/usr/bin/env python
# coding: utf-8

# **DBSCAN DEMO**

# Importar Librerias

# In[74]:


import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'qt')


# Cargar Datos

# In[75]:


df = df = pd.read_csv('./casas.csv')
df=df.dropna()
df.sample(10)


# Analisis

# In[76]:


pred = DBSCAN(eps=2,min_samples=10).fit_predict(df)


# In[77]:


print(pred)


# Graficos

# In[80]:


plt.figure(figsize=(7.5, 7.5))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=pred, s=100)
plt.xlabel("Antigüedad de la Construcción en Años")
plt.ylabel("Precio de Casa en Pesos (1:100,000)")
plt.box(False)
plt.show()


# In[ ]:




