# Quantum-Enhanced Protein–Ligand Screening

A machine learning pipeline for predicting protein–ligand binding affinity using GNNs for ligands, CNNs for proteins, and quantum mechanical descriptors. This project combines cheminformatics, quantum features, and bioinformatics to identify promising drug candidates.

---

## 🚀 Features

- 📊 Binding affinity prediction using pretrained KIBA-based GNN–CNN models
- 🧪 Ligand filtering based on quantum descriptors (HOMO, LUMO, dipole moment)
- 💊 Toxicity and drug-likeness scoring (pretrained classifier)
- 🧬 Protein sequence extraction from BindingDB and AlphaFold structure matching
- 🔬 Structural clustering of top hits using molecular fingerprints
- 🌐 Interactive 3D visualization with `py3Dmol` for protein–ligand complexes
- 🔍 PubChem similarity search to identify novel compounds

---

## 📁 Project Structure

```bash
.
├── data/                     # Input datasets (ligands, proteins, QM, etc.)
├── src/                      # Core pipeline scripts
│   ├── tox_scr.py        # Runs toxicity screening on dataset
│   ├── dl_scorer.py   # Gives drug likeness score
│   ├── lig_filt.py    # Filters ligands based on toxicity, drug-likenes and QM properties
│   ├── kib_inf.py     # Adds KIBA binding affinity score for each ligand-protein combination
│   └── main.py     # Main script which runs the pipeline
├── vis/            #Visualization tools to visualize final drug-like set
└── README.md
```

---

## 🛠️ Requirements

```bash
pip install torch pandas rdkit py3Dmol biopython scikit-learn requests
```

---

## 🧪 Running the Pipeline

1. **Filter ligands** by QM, toxicity, and drug-likeness and predict binding scores**:
   ```bash
   python src/main.py
   ```

2. **Cluster and visualize hits**:
   ```bash
   python src/clust_lig.py
   python src/lig_vwr.py
   ```

3. **Check novelty against PubChem**:
   ```bash
   python vis/nov_look.py
   ```

---

## 📊 Example Output

- `top20_filtered_hits.csv` with SMILES, sequences, predicted KIBA scores, QM features
- `protein_ligand_html/index.html` for browsing 3D visualizations
- `cluster_plot.png` showing ligand clustering

---

## 📄 License

MIT License

---

## 👤 Author

Aniket Mandal – [@animan95](https://github.com/animan95)

For questions or collaborations, feel free to reach out!
