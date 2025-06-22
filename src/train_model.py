import pandas as pd
import pickle
from sklearn.linear_model import Ridge
import os

# Charger les datasets
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()


# Entraîner le modèle Ridge
best_params = {"alpha": 1.0}
model = Ridge(**best_params)
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Sauvegarder les paramètres 
with open("models/params.pkl", "wb") as f:
    pickle.dump(best_params, f)

print("Le modèle a été stocké dans models/model.pkl.")
