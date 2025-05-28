import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
import joblib
import os

# --- Helper function ---
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- Step 1: Train models for all Tox21 labels ---
label_list = retrieve_label_name_list('tox21')
models = {}
print("🔁 Training models on Tox21 labels:")
for label in label_list:
    tox = Tox(name='Tox21', label_name=label)
    data = tox.get_data().dropna(subset=['Drug', 'Y'])

    fps, labels = [], []
    for _, row in data.iterrows():
        fp = smiles_to_fp(row['Drug'])
        if fp is not None:
            fps.append(fp)
            labels.append(row['Y'])

    X = np.array(fps)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"✅ Trained {label} | AUC: {auc:.3f}")

    models[label] = clf
    joblib.dump(clf, f"tox_model_{label}.pkl")

# --- Step 2: Load your dataset ---
print("\n📦 Loading your target dataset: chembl_qm_lipinski_filtered.csv")
input_df = pd.read_csv("../data/chembl_qm_lipinski_filtered.csv")  # Must contain a 'smiles' column
input_df["fp"] = input_df["smiles"].apply(smiles_to_fp)
input_df = input_df[input_df["fp"].notnull()].reset_index(drop=True)

fps = np.stack(input_df["fp"].values)
tox_scores = []

print("🧪 Scoring toxicity across all 12 endpoints...")
for label in label_list:
    model = joblib.load(f"tox_model_{label}.pkl")
    probs = model.predict_proba(fps)[:, 1]
    tox_scores.append(probs)

tox_array = np.array(tox_scores).T  # shape: [n_molecules, 12]

# --- Step 3: Add scores to DataFrame ---
input_df["tox_mean"] = tox_array.mean(axis=1)
input_df["tox_max"] = tox_array.max(axis=1)
input_df["tox_flags"] = (tox_array > 0.5).sum(axis=1)

# --- Step 4: Save output ---
input_df.drop(columns=["fp"]).to_csv("chembl_qm_tox_scored.csv", index=False)
print("✅ Saved toxicity-scored dataset: chembl_qm_tox_scored.csv")

