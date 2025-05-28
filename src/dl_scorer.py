import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Load dataset
df = pd.read_csv("chembl_qm_tox_scored.csv")

# Load model
model = joblib.load("model_nn.pkl")

# Generate fingerprints
def mol_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        arr = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

fps = df["smiles"].apply(mol_to_fp)
valid = fps.notnull()
X = np.stack(fps[valid])
drug_likeness = np.zeros(len(df))
drug_likeness[valid] = model.predict_proba(X)[:, 1]
df["drug_likeness_score"] = drug_likeness

df.to_csv("chembl_qm_tox_scored.csv", index=False)
print("✅ Added 'drug_likeness_score' and saved to chembl_qm_tox_scored.csv")
