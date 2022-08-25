#!/usr/bin/env python
# coding: utf-8

# # Red Neuronal

# Una red neuronal es un modelo simplificado que emula el modo en que el cerebro humano procesa la información: Funciona simultaneando un número elevado de unidades de procesamiento interconectadas que parecen versiones abstractas de neuronas.
# 
# Las unidades de procesamiento se organizan en capas. Hay tres partes normalmente en una red neuronal : una capa de entrada, con unidades que representan los campos de entrada; una o varias capas ocultas; y una capa de salida, con una unidad o unidades que representa el campo o los campos de destino. Las unidades se conectan con fuerzas de conexión variables (o ponderaciones). Los datos de entrada se presentan en la primera capa, y los valores se propagan desde cada neurona hasta cada neurona de la capa siguiente. al final, se envía un resultado desde la capa de salida.
# 
# La red aprende examinando los registros individuales, generando una predicción para cada registro y realizando ajustes a las ponderaciones cuando realiza una predicción incorrecta. Este proceso se repite muchas veces y la red sigue mejorando sus predicciones hasta haber alcanzado uno o varios criterios de parada.
# 
# Al principio, todas las ponderaciones son aleatorias y las respuestas que resultan de la red son, posiblemente, disparatadas. La red aprende a través del entrenamiento. Continuamente se presentan a la red ejemplos para los que se conoce el resultado, y las respuestas que proporciona se comparan con los resultados conocidos. La información procedente de esta comparación se pasa hacia atrás a través de la red, cambiando las ponderaciones gradualmente. A medida que progresa el entrenamiento, la red se va haciendo cada vez más precisa en la replicación de resultados conocidos. Una vez entrenada, la red se puede aplicar a casos futuros en los que se desconoce el resultado.

# # Ejemplo

# Se buscara clasificar el genero de un juego en base a los parametros de la plataforma, el publisher y sus ventas globales

# # Importar Librerias

# In[20]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


# # Cargar Datos

# In[8]:


juegos = pd.read_csv('./vgsales.csv')
juegos = juegos.replace(np.nan,"0")
juegos['Platform'] = juegos['Platform'].replace("2600","Atari")

juegos.sample(10)


# # Normalizacion

# In[12]:


encoder = LabelEncoder()
juegos['plataforma'] = encoder.fit_transform(juegos.Platform.values)
juegos['publica'] = encoder.fit_transform(juegos.Publisher.values)


# # Definir Input/Output de la  Red Neuronal

# In[16]:


#Entradas a la red
X=juegos[['plataforma','publica','Global_Sales']]
#Salida
y = juegos['Genre']


# # Escalamiento

# In[17]:



X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # Crear y Entrenar Red Neuronal

# In[18]:


mlp=MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, alpha=0.0001,
                     solver='adam', random_state=21,tol=0.000000001)

mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)


# # Revisamos la Red Neuronal

# In[21]:


print(classification_report(y_test,predictions))


# In[28]:


accuracy= mlp.score(X_test, y_test)
print('Accuracy: ',accuracy)


# In[ ]:




