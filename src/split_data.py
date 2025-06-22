import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Créer le dossier data/processed s'il n'existe pas
os.makedirs("data/processed", exist_ok=True)

# Charger le dataset raw.csv
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
data = pd.read_csv(url)
data.to_csv("data/raw/raw.csv", index=False)

# Split des données en ensemble d'entraînement et de test
X = data.drop(columns=["silica_concentrate"])
y = data["silica_concentrate"]

# Spliter en 4 datasets (X_test, X_train, y_test, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stocker les datasets dans data/processed.
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Les 4 datasets (X_test, X_train, y_test, y_train) ont été stockés dans data/processed.")


