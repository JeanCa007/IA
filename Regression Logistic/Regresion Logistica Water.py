#!/usr/bin/env python
# coding: utf-8

# Importamos la librerias

# In[41]:


import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# In[42]:


df = pd.read_csv('./water_potability.csv')
df=df.dropna()
df.sample(10)


# Selecciono las variables

# In[43]:


parametros = df.drop(columns="Potability")
objetivo = df.Potability


# Implementacion del model

# Preparamos los datos de prueba

# In[44]:


X_train, X_test, y_train,y_test =train_test_split(parametros,objetivo,test_size=0.5)


# Escalamos los datos

# In[45]:


escalar = StandardScaler()


# In[46]:


X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)


# Defino el modelo

# In[47]:


model = LogisticRegression()


# Entreno el modelo

# In[48]:


model.fit(X_train,y_train)


# Realizo una prediccion

# In[49]:


y_pred = model.predict(X_test)


# Verifico la prediccion

# In[50]:


matriz = confusion_matrix(y_test,y_pred)
print("Matriz de Confusion",matriz)


# Precision del Modelo

# In[51]:


precision = precision_score(y_test,y_pred)
print("Precision del modelo",precision)


# Exactitud del Modelo

# In[52]:


exactitud = accuracy_score(y_test,y_pred)
print("Exactitud del modelo: ",exactitud)


# In[ ]:




