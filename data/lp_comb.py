import pandas as pd
from itertools import product

# Load ligand and protein data
ligand_df = pd.read_csv("ligands_filtered.csv")           # Should include: smiles, homo, lumo, gap, etc.
protein_df = pd.read_csv("bindingdb_proteins.csv")        # Should include: sequence column (can be "sequence" or "Target")

# Rename protein column if needed
if 'Target' in protein_df.columns:
    protein_df = protein_df.rename(columns={'Target': 'protein_seq'})
elif 'sequence' in protein_df.columns:
    protein_df = protein_df.rename(columns={'sequence': 'protein_seq'})

# Perform Cartesian product (all ligand–protein pairs)
ligand_df['key'] = 1
protein_df['key'] = 1
combined_df = pd.merge(ligand_df, protein_df, on='key').drop('key', axis=1)

# Optional: save to file
combined_df.to_csv("ligand_protein_pairs.csv", index=False)
print(f"✅ Saved {combined_df.shape[0]} ligand–protein pairs to 'ligand_protein_pairs.csv'")

