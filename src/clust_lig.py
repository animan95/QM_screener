import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

# Load ligand data
df = pd.read_csv("top200_kiba_predictions.csv")

# Convert SMILES to fingerprints
mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

# Convert to numpy array
fp_array = np.zeros((len(fps), 1024))
for i, fp in enumerate(fps):
    DataStructs.ConvertToNumpyArray(fp, fp_array[i])

# Compute Jaccard distance matrix
distance_matrix = pairwise_distances(fp_array, metric="jaccard")

# Cluster ligands
clustering = AgglomerativeClustering(n_clusters=6, metric='precomputed', linkage='average')
df["cluster"] = clustering.fit_predict(distance_matrix)

# Save and plot
df.to_csv("top200_with_clusters.csv", index=False)
sns.countplot(x="cluster", data=df)
plt.title("Ligand Clusters by Structure")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("ligand_clusters.png")
plt.show()

