import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import os

CSV = "top20_with_pdbs.csv"
OUTPUT_DIR = "protein_ligand_html"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)

index_entries = []

for i, row in df.iterrows():
    smiles = row["smiles"]
    pdb_path = row["pdb_path"]

    if not isinstance(pdb_path, str) or not os.path.exists(pdb_path):
        print(f"❌ Missing PDB for row {i}")
        continue

    # Generate ligand conformer from SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if success != 0:
        print(f"❌ Failed to generate 3D structure for ligand {i}")
        continue
    AllChem.UFFOptimizeMolecule(mol)
    mol_block = Chem.MolToMolBlock(mol)

    with open(pdb_path, "r") as f:
        protein_str = f.read()

    # Generate 3D viewer
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(protein_str, "pdb")
    viewer.setStyle({"cartoon": {"color": "spectrum"}})
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"model": 1}, {"stick": {}})
    viewer.zoomTo()

    html = viewer._make_html()
    html_file = f"ligand_{i}.html"
    with open(os.path.join(OUTPUT_DIR, html_file), "w") as f:
        f.write(html)

    index_entries.append(f'<li><a href="{html_file}" target="_blank">Ligand {i}</a></li>')

# Create index.html
index_html = "<html><body><h2>Top Ligand–Protein 3D Viewer</h2><ul>"
index_html += "\n".join(index_entries)
index_html += "</ul></body></html>"

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(index_html)

print("✅ All HTML viewers saved in:", OUTPUT_DIR)

