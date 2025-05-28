import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError

sns.set(style="whitegrid")

# --- Load SMILES datasets ---
drug_like = pd.read_csv("../data/pytdc_chembl_valid.smi", names=["smiles"])
drug_like["source"] = "Drug"
candidates = pd.read_csv("../data/zinc_molgen_druglike_candidates.smi", names=["smiles"])
candidates["source"] = "Candidate"
df = pd.concat([drug_like, candidates]).drop_duplicates().reset_index(drop=True)

# --- Strict molecule validation ---
def mol_from_smiles_strict(s):
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return None
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return mol
    except:
        return None

df["mol"] = df["smiles"].apply(mol_from_smiles_strict)
df = df[df["mol"].notnull()].reset_index(drop=True)

# --- Compute descriptors + Lipinski violations ---
def compute_props(mol):
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": CalcTPSA(mol),
        "HDonors": Lipinski.NumHDonors(mol),
        "HAcceptors": Lipinski.NumHAcceptors(mol),
        "RingCount": Descriptors.RingCount(mol),
        "LipinskiViolations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10
        ])
    })

df = df.join(df["mol"].apply(compute_props))

# --- Compute Morgan fingerprints ---
def mol_to_fp(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    arr = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = np.array([mol_to_fp(m) for m in df["mol"]])

# --- t-SNE with failure handling ---
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)
    df["tSNE1"] = X_embedded[:, 0]
    df["tSNE2"] = X_embedded[:, 1]
except Exception as e:
    print(f"⚠️ t-SNE failed: {e}")
    df["tSNE1"] = np.nan
    df["tSNE2"] = np.nan

# --- Plot t-SNE chemical space ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="tSNE1", y="tSNE2", hue="source", alpha=0.7, s=40)
plt.title("t-SNE of Morgan Fingerprints")
plt.tight_layout()
plt.savefig("tsne_chemspace.png", dpi=300)
plt.show()

# --- Plot descriptor histograms ---
for prop in ["MolWt", "LogP", "TPSA", "HDonors", "HAcceptors", "LipinskiViolations"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=prop, hue="source", kde=True, element="step", stat="density")
    plt.title(f"Distribution of {prop}")
    plt.tight_layout()
    plt.savefig(f"hist_{prop}.png", dpi=300)
    plt.close()

# --- Save descriptor table ---
df.drop(columns=["mol"]).to_csv("chemspace_descriptors.csv", index=False)
print("✅ Saved descriptors to chemspace_descriptors.csv")
print("📊 Saved plot: tsne_chemspace.png and per-descriptor histograms")

