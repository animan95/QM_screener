import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

# Load data (should include 'cluster' column from previous step)
df = pd.read_csv("top200_with_clusters.csv")
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]

def mol_to_img_tag(mol):
    img = Draw.MolToImage(mol, size=(200, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" />'

df["mol"] = mols
df["img_tag"] = df["mol"].apply(mol_to_img_tag)

html = "<html><body><h2>Ligand Viewer</h2><table border='1'><tr><th>Image</th><th>SMILES</th><th>KIBA</th><th>Drug Score</th><th>Tox</th><th>Cluster</th></tr>"
for _, row in df.iterrows():
    html += f"<tr><td>{row['img_tag']}</td><td>{row['smiles']}</td><td>{row['predicted_kiba']:.3f}</td><td>{row['drug_likeness_score']:.2f}</td><td>{row['tox_flags']}</td><td>{row['cluster']}</td></tr>"
html += "</table></body></html>"

with open("ligand_viewer.html", "w") as f:
    f.write(html)

print("✅ Viewer generated: ligand_viewer.html")

