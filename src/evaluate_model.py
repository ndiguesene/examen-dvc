import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données de test
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Charger le modèle entraîné
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Faire des prédictions
y_pred = model.predict(X_test)

# stocker les predictions dans data
predictions = pd.DataFrame({"Actual": y_test.values.ravel(), "Predicted": y_pred})
predictions.to_csv("data/predictions.csv", index=False)

# Calculer les métriques d'évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {"MSE": mse, "R2_Score": r2}

# stocker  les métriques dans metrics
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Le modèle a été évalué et  les résultats sont disponibles dans 'metrics/scores.json'.")

