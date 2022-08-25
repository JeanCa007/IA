#!/usr/bin/env python
# coding: utf-8

# Librerias

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


# Cargo Datos

# In[20]:


df = df = pd.read_csv('./water_potability.csv')
df=df.dropna()
df.sample(10)


# Divido entre agua potable o no potable

# In[21]:


potable = df[df['Potability']==1]
No_potable= df[df['Potability']==0]


# Grafico Potable vs No Potable

# In[22]:


plt.scatter(potable["Sulfate"],potable["ph"],marker="*",s=150,color="blue",label="Agua Potable")
plt.scatter(No_potable["Sulfate"],No_potable["ph"],marker="*",s=150,color="red",label="Agua NO Potable")
plt.ylabel("ph")
plt.xlabel("Sulfate")
plt.legend(bbox_to_anchor=(1,0.2))
plt.show()


# Preparo los datos

# In[24]:


df = pd.get_dummies(data=df, drop_first=True)


# In[115]:


parametros = df.drop(columns='Potability')
clase = df.Potability


# Creacion del Modelo KNN

# In[116]:


model = KNeighborsClassifier(n_neighbors=10)


# In[117]:


model.fit(parametros,clase)


# Prediccion Test

# In[118]:


a = parametros.sample()


# In[119]:


a


# Prediccion

# In[120]:


y_pred = model.predict(a)
print("Es Potable",(y_pred))


# Probabilidad de que sea Potable

# In[121]:


print("Probabilidad de ser potable: ",(model.predict_proba(a)))


# Comparacion del Modelo con los hechos

# In[122]:


df['pred']= y_pred[0]
df.sample(10)[['Potability','pred']]


# In[ ]:




