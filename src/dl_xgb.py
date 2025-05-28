import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

# --- Load and label data ---
df1 = pd.read_csv("../data/pytdc_chembl_valid.smi", names=["smiles"])
df1["label"] = 1
df2 = pd.read_csv("../data/zinc_molgen_druglike_candidates.smi", names=["smiles"])
df2["label"] = 0
df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

# --- Validate molecules ---
def is_valid_rdkit_mol(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return mol
    except:
        return None

df["mol"] = df["smiles"].apply(is_valid_rdkit_mol)
df = df[df["mol"].notnull()].reset_index(drop=True)

# --- Fingerprints ---
fp_gen = GetMorganGenerator(radius=2, fpSize=2048)

def mol_to_fp(mol):
    try:
        arr = np.zeros((2048,))
        fp = fp_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None

fps = [mol_to_fp(m) for m in df["mol"]]
valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
df = df.iloc[valid_indices].reset_index(drop=True)
X = np.array([fps[i] for i in valid_indices])
y = df["label"].values

# --- Train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
preds = (proba > 0.5).astype(int)

print("XGBoost:")
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
joblib.dump(model, "model_xgb.pkl")

