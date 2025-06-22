import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Normalisation des données. Comme vous pouvez le noter, les données sont dans des échelles très variés donc une
#normalisation est nécessaire. Vous pouvez utiliser des fonctions pré-existantes pour la construction de ce script.
#En sortie, ce script créera deux nouveaux datasets : (X_train_scaled, X_test_scaled) que vous sauvegarderez également
# dans data/processed.
# Charger les datasets X_train et x_test
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Convertir les dates en timestamps
for col in X_train.select_dtypes(include=["object"]).columns:
    try:
        X_train[col] = pd.to_datetime(X_train[col], errors="coerce").astype("int64") // 10**9
        X_test[col] = pd.to_datetime(X_test[col], errors="coerce").astype("int64") // 10**9
    except:
        pass
# Initialiser le scaler
scaler = StandardScaler()

# Normaliser les datasets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir en DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarder les datasets X_train_scaled et X_test_scaled
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

print("Les datasets normalisés X_train_scaled et  X_test_scaled ont été stockés dans data/processed.")

