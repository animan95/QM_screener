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
│   ├── train_model.py        # Training script for GNN + CNN models
│   ├── predict_affinity.py   # Binding prediction for new ligand–protein pairs
│   ├── cluster_ligands.py    # Clustering by structure
│   ├── viewer.py             # 3D viewer generation with py3Dmol
│   └── pubchem_check.py      # PubChem novelty check
├── top_hits.csv              # Final 200 predicted ligand–protein pairs
└── README.md
```

---

## 🛠️ Requirements

```bash
pip install torch pandas rdkit py3Dmol biopython scikit-learn requests
```

---

## 🧪 Running the Pipeline

1. **Filter ligands** by QM, toxicity, and drug-likeness:
   ```bash
   python src/filter_ligands.py
   ```

2. **Prepare ligand–protein combinations**:
   ```bash
   python src/make_combinations.py
   ```

3. **Predict binding scores**:
   ```bash
   python src/predict_affinity.py
   ```

4. **Cluster and visualize hits**:
   ```bash
   python src/cluster_ligands.py
   python src/viewer.py
   ```

5. **Check novelty against PubChem**:
   ```bash
   python src/pubchem_check.py
   ```

---

## 📊 Example Output

- `top_hits.csv` with SMILES, sequences, predicted KIBA scores, QM features
- `protein_ligand_html/index.html` for browsing 3D visualizations
- `cluster_plot.png` showing ligand clustering

---

## 📄 License

MIT License

---

## 👤 Author

Aniket Mandal – [@animan95](https://github.com/animan95)

For questions or collaborations, feel free to reach out!
