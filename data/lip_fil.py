from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd

df = pd.read_csv("zinc_qm_summary_smiles.csv")  # or your 500k dataset

def passes_lipinski(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    except:
        return False

df["passes_lipinski"] = df["smiles"].apply(passes_lipinski)
df_filtered = df[df["passes_lipinski"]].drop(columns=["passes_lipinski"])

df_filtered.to_csv("chembl_qm_lipinski_filtered.csv", index=False)
print(f"✅ {len(df_filtered)} molecules passed Lipinski filter.")

