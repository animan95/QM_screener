from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pandas as pd

# Load ZINC SMILES
zinc_df = pd.read_csv("zinc_molgen_druglike_candidates.smi", names=["smiles"])

# Filter with Lipinski Rule of 5
def passes_lipinski(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        return (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
    except:
        return False

zinc_df["passes_lipinski"] = zinc_df["smiles"].apply(passes_lipinski)
filtered_zinc = zinc_df[zinc_df["passes_lipinski"]].drop(columns=["passes_lipinski"])

# Save result
filtered_zinc.to_csv("zinc_filtered_lipinski.smi", index=False, header=False)
print(f"✅ Filtered ZINC molecules: {len(filtered_zinc)} saved to 'zinc_filtered_lipinski.smi'")

