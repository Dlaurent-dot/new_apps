import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Charger le modèle de forêt aléatoire
with open('hotel.pkl', 'rb') as file:
    model = pickle.load(file)

# Fonction pour charger et préparer les données
def load_data():
    df = pd.read_csv("hotel.csv", delimiter=',')
    return df

# Charger les données pour obtenir les noms de colonnes et les types de caractéristiques
df = load_data()
X = df.drop('booking_status', axis=1)

# Titre de l'application
st.title('Application de Prédiction avec Forêt Aléatoire')

# Instructions pour l'utilisateur
st.write('Veuillez entrer les caractéristiques pour faire une prédiction :')

# Générer les champs de saisie pour chaque caractéristique
input_data = []
for col in X.columns:
    dtype = X[col].dtype
    if dtype == 'int64':
        value = st.number_input(f'{col}', step=1)
    elif dtype == 'float64':
        value = st.number_input(f'{col}')
    else:
        value = st.text_input(f'{col}')
    input_data.append(value)

# Convertir les entrées en tableau numpy
input_data = np.array([input_data])

# Préparer les données pour la prédiction
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Bouton pour effectuer la prédiction
if st.button('Prédire'):
    prediction = model.predict(input_data_scaled)
    st.write('La prédiction est :', prediction[0])
    
   # Vérifier la prédiction et afficher le statut de la réservation
    if prediction[0] == 0:
        st.write('La réservation est honorée.')
    else:
        st.write('La réservation est annulée.')
# Afficher des informations supplémentaires sur les caractéristiques
st.write("""
## Instructions :
- Veuillez entrer les valeurs pour chaque caractéristique. Les valeurs peuvent être des entiers, des flottants ou des chaînes de caractères selon le type de donnée.
""")

# Afficher le dataframe pour référence
st.write('Voici un aperçu des données :')
st.write(df.drop('booking_status', axis=1))