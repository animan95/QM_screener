import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
df = pd.read_csv("top200_kiba_predictions.csv")

# === Optional Filtering ===
filtered = df[(df["tox_flags"] == 0) & (df["drug_likeness_score"] >= 0.8)]
print(f"✅ {len(filtered)} drug-like, non-toxic hits found")

# === 1. Drug-likeness vs Predicted KIBA ===
plt.figure(figsize=(8, 6))
plt.scatter(df["drug_likeness_score"], df["predicted_kiba"], c=df["tox_flags"], cmap="coolwarm", edgecolor="k")
plt.colorbar(label="Toxicity Flags")
plt.xlabel("Drug-likeness Score")
plt.ylabel("Predicted KIBA Score")
plt.title("Drug-Likeness vs Predicted Binding Affinity")
plt.tight_layout()
plt.savefig("drug_likeness_vs_kiba.png")
plt.show()

# === 2. Top Proteins with Strongest Ligand Binders ===
top_by_protein = df.groupby("protein_seq")["predicted_kiba"].max().reset_index()
top10_proteins = top_by_protein.sort_values(by="predicted_kiba", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="predicted_kiba", y="protein_seq", data=top10_proteins, palette="viridis")
plt.xlabel("Top Predicted KIBA Score")
plt.ylabel("Protein Sequence")
plt.title("Top Proteins with Strongest Ligand Binders")
plt.tight_layout()
plt.savefig("top_proteins_kiba.png")
plt.show()

# === 3. Optional: RDKit Visualization of Top Ligands ===
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = [Chem.MolFromSmiles(smi) for smi in filtered["smiles"].head(12)]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200))
    img.save("top_ligands.png")
    print("✅ Saved RDKit ligand grid as top_ligands.png")
except ImportError:
    print("⚠️ RDKit not found — skipping ligand visualization.")

