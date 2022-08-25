#!/usr/bin/env python
# coding: utf-8

# # Regularización de Red Neuronal

# # Ejemplo

# # Importar Librerias

# In[72]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')


# # Cargar Datos

# In[42]:


data = pd.read_csv('./water_potability.csv')
data = data.replace(np.nan,"0")


data.sample(10)


# # Definir Input/Output de la  Red Neuronal

# In[43]:


#Entradas a la red
X=data.drop(columns="Potability")
#Salida
y = data.Potability


# # Escalamiento

# In[45]:



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Entrenar Red Neuronal con Ridge

# La Regresión Rigde, también denominada regresión contraída o Tikhonov regularization, regulariza el modelo resultante imponiendo una penalización al tamaño de los coeficientes de la relación lineal entre las características predictivas y la variable objetivo. En este caso, los coeficientes calculados minimizan la suma de los cuadrados de los residuos penalizada al añadir el cuadrado de la norma L2 del vector formado por los coeficientes donde λ es un parámetro que controla el grado de penalización: cuanto mayor éste, los coeficientes serán menores resultando más robustos a la colinealidad

# In[70]:


modelo = RidgeClassifier(alpha=0.1,max_iter=500, solver='lbfgs',tol=0.000000001,random_state=21,positive=True)
modelo.fit(X_train,y_train)
predictions=modelo.predict(X_test)


# # Revisamos la Red Neuronal con Ridge

# In[71]:


accuracy= modelo.score(X_test, y_test)
print('Accuracy: ',accuracy)


# # Entrenar Red Neuronal con Laso

# Es un modelo lineal que penaliza el vector de coeficientes añadiendo su norma L1 (basada en la distancia Manhattan) a la función de coste.
# 
# Lasso tiende a generar "coeficientes dispersos": vectores de coeficientes en los que la mayoría de ellos toman el valor cero. Esto quiere decir que el modelo va a ignorar algunas de las características predictivas, lo que puede ser considerado un tipo de selección automática de características. El incluir menos características supone un modelo más sencillo de interpretar que puede poner de manifiesto las características más importantes del conjunto de datos. En el caso de que exista cierta correlación entre las características predictivas, Lasso tenderá a escoger una de ellas al azar.
# 
# Esto significa que, aunque Ridge es una buena opción por defecto, si sospechamos que la distribución de los datos viene determinada por un subconjunto de las características predictivas, Lasso podría devolver mejores resultados.

# In[67]:


modelo = linear_model.Lasso(alpha=0.1)
modelo.fit(X_train,y_train)
predictions= modelo.predict(X_test)


# # Revisamos la Red Neuronal con Lasso

# In[68]:


accuracy= modelo.score(X_test, y_test)
print('Accuracy: ',accuracy)


# # Entrenar Red Neuronal con Elastic Net

# Es un modelo de regresión lineal que normaliza el vector de coeficientes con las normas L1 y L2. Esto permite generar un modelo en el que solo algunos de los coeficientes sean no nulos, manteniendo las propiedades de regularización de Ridge.
# 
# El parámetro λ regula el peso dado a la regularización impuesta por Ridge y por Lasso. Desde este punto de vista Elastic Net es un superconjunto de ambos modelos.
# 
# En el caso de que exista cierta colinealidad entre varias características predictivas, Elastic Net tenderá a escoger una o todas (aun con coeficientes menores) en función de cómo haya sido parametrizado.

# In[76]:


modelo = SGDClassifier(alpha=0.1,max_iter=500,tol=0.000000001,loss='squared_error',penalty='elasticnet')
modelo.fit(X_train,y_train)
predictions= modelo.predict(X_test)


#  # Revisamos la Red Neuronal con Elastic Net

# In[77]:


accuracy= modelo.score(X_test, y_test)
print('Accuracy: ',accuracy)


# # Conclusion

# Para este conjunto de datos a pesar de no dar resultados mejores al 60% los algoritmos de regularizacion L2 (Ridge)
# y Elastic Net son los indicados superando el resultado de Lasso.
