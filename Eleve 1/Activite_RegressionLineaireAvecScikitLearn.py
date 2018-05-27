
# coding: utf-8

# In[1]:


# On importe les librairies dont nous aurons besoin pour cette activite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Présentation du dataset
# 
# On charge le fichier hubble_data.csv.
# Celui contient une ensemble de mesure décrivant la relation entre la distance qui nous sépare d'une galaxie, et la vitesse à laquelle elle s'éloigne de nous.
# 
# Le dataset à traiter est unidimensionel.
# Le nuage de point aura la distance en abscisse et la vitesse en ordonné.

# In[2]:


# On charge le dataset
hubble_data = pd.read_csv('hubble_data.csv')
#print(hubble_data.head())
# Description complete
#print(hubble_data.describe(include='all'))
X = hubble_data['distance'].to_frame()
Y = hubble_data['recession_velocity']


# In[3]:


# On affiche le nuage de points dont on dispose
plt.xlabel('Distance')
plt.ylabel('Vitesse d\'eloignement')
plt.plot(X, Y, 'ro', markersize=4)
plt.show()


# # Utilisation de la bibliothèque scikit-learn

# In[4]:


#Import nécessaire pour la bibliothéque d'apprentissage
from sklearn import linear_model


# In[5]:


#Génération du modèle
regr = linear_model.LinearRegression()
regr.fit(X, Y)
#Prédiction des données suivant le modèle
Ypred = regr.predict(X)


# In[6]:


print("Paramètres du modèle")
print("La liste des coeficients est : ", regr.coef_)
print("Le terme constant est : ", regr.intercept_)


# # Résultat final

# In[7]:


# Plot outputs
#plt.xticks(())
#plt.yticks(())
plt.xlabel('Distance')
plt.ylabel('Vitesse d\'eloignement')

plt.plot(X, Y, 'ro', markersize=4, color='black')
plt.plot(X, regr.predict(X), color='blue', linewidth=3)


plt.show()

