#!/usr/bin/env python
# coding: utf-8

# Cargar Datos

# In[72]:


import pandas as pd


# In[73]:


df = pd.read_csv('./water_potability.csv')
df.sample(10)


# In[76]:


df=df.dropna()
df.sample(10)


# In[77]:


df = pd.get_dummies(data=df, drop_first=True)


# Selecciono las variables

# In[78]:


explicativas = df.drop(columns='Potability')
objetivo = df.Potability


# Entrenar modelo Arbol de Decision Clasificacion

# In[79]:


fit()


# In[80]:


from sklearn.tree import DecisionTreeClassifier


# In[81]:


model = DecisionTreeClassifier(max_depth=3)


# In[82]:


model.fit(X=explicativas,y=objetivo)


# Visualizar el Modelo

# In[83]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[84]:


plt.figure(figsize=(14,8))
plot_tree(decision_tree=model,feature_names=explicativas.columns,filled=True, fontsize=10);


# Calcular Prediccion

# In[85]:


a = explicativas.sample()


# In[86]:


a


# In[87]:


model.predict_proba(a)


# In[88]:


y_pred = model.predict(explicativas)


# Interpretar Modelo

# In[90]:


import seaborn as sns


# In[91]:


sns.histplot(x=df.Sulfate, hue=df.Potability)


# Que tan bueno es el modelo ?

# In[92]:


df['pred'] = y_pred


# In[93]:


df.sample(10)[['Potability','pred']]


# In[94]:


(df['Potability']==df['pred']).sum()


# In[95]:


(df['Potability']==df['pred']).mean()


# In[ ]:




