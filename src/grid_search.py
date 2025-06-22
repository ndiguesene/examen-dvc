import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# Charger les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()  
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()    

# Choisir un modèle de régression
model = RandomForestRegressor(random_state=42)

# Définir une grille de paramètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# GridSearchCV pour rechercher les meilleurs paramètres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Effectuer la recherche par grille
grid_search.fit(X_train, y_train)

# Enregistrer le meilleur modèle
with open('models/model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Enregistrer les meilleurs paramètres
with open('models/params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)

# Évaluer sur les données de test
y_pred = grid_search.best_estimator_.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)

print(f"Performance sur le test (MSE) : {mse_test}")
print(f"Meilleurs paramètres : {grid_search.best_params_}")
print("Le modèle et les paramètres optimaux ont été enregistrés dans 'models/'")
