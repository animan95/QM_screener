from tdc.generation import MolGen
from rdkit import Chem
import pandas as pd

def is_valid_smiles(smi):
    try:
        smi = str(smi).strip()
        if smi.lower() == 'nan' or smi == '':
            return False
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except:
        return False

# Load ChEMBL molecules from MolGen
print("⏳ Loading ChEMBL molecules from PyTDC...")
gen = MolGen(name="ChEMBL")
df_gen = gen.get_data()

print(f"📦 Raw molecules loaded: {len(df_gen)}")

# Filter valid SMILES
valid_smiles = set()
for smi in df_gen["smiles"].unique():
    if is_valid_smiles(smi):
        valid_smiles.add(smi)

print(f"✅ Valid SMILES retained: {len(valid_smiles)}")

# Save to .smi file
with open("pytdc_chembl_valid.smi", "w") as f:
    for smi in sorted(valid_smiles):
        f.write(smi + "\n")

print("🎉 Saved to: pytdc_chembl_valid.smi")

