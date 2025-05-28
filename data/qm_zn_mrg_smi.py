import pandas as pd
from rdkit import Chem

# --- Step 1: Load ZINC candidate SMILES ---
zinc = pd.read_csv("pytdc_chembl_valid.smi", names=["smiles"])
zinc["smiles"] = zinc["smiles"].apply(
    lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
)
zinc = zinc.dropna().drop_duplicates(subset=["smiles"])

# --- Step 2: Load QMugs summary.csv ---
summary = pd.read_csv("summary.csv")
summary["smiles"] = summary["smiles"].apply(
    lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
)
summary = summary.dropna().drop_duplicates(subset=["smiles"])

# --- Step 3: Merge on canonical SMILES ---
merged = pd.merge(zinc, summary, on="smiles", how="inner")

# --- Step 4: Select and rename relevant QM features ---
columns_to_keep = [
    "smiles",
    "DFT_HOMO_ENERGY", "DFT_LUMO_ENERGY", "DFT_HOMO_LUMO_GAP",
    "DFT_DIPOLE_X", "DFT_DIPOLE_Y", "DFT_DIPOLE_Z", "DFT_DIPOLE_TOT",
    "DFT_TOTAL_ENERGY"
]

merged_filtered = merged[columns_to_keep].rename(columns={
    "DFT_HOMO_ENERGY": "homo",
    "DFT_LUMO_ENERGY": "lumo",
    "DFT_HOMO_LUMO_GAP": "gap",
    "DFT_DIPOLE_X": "dipole_x",
    "DFT_DIPOLE_Y": "dipole_y",
    "DFT_DIPOLE_Z": "dipole_z",
    "DFT_DIPOLE_TOT": "dipole_total",
    "DFT_TOTAL_ENERGY": "total_energy"
})

# --- Step 5: Save result and report ---
merged_filtered.to_csv("zinc_qm_summary_smiles.csv", index=False)
print(f"✅ SMILES-based match: {len(merged_filtered)} molecules saved to 'zinc_qm_summary_smiles.csv'")

