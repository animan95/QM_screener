import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch as GraphBatch
from tqdm import tqdm

# --- Import your model and featurization components ---
from lig_gnn import LigandGCN
from prot_cnn import ProteinCNN
from aff_pred_bay import AffinityPredictor
from graph_utils import mol_to_graph
from seq_utils import encode_sequence

# === Custom Dataset for Inference ===
class LigandProteinDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = mol_to_graph(row['smiles'])
        sequence = encode_sequence(row['protein_seq'])
        qm_tensor = torch.tensor([
            row["homo"],
            row["lumo"],
            row["gap"],
            row["dipole_x"],
            row["dipole_y"],
            row["dipole_z"],
            row["dipole_total"]
        ], dtype=torch.float)
        tox_mean = row["tox_mean"]
        tox_flags = row["tox_flags"]
        drug_score = row["drug_likeness_score"]
        return graph, sequence, qm_tensor, row['smiles'], row['protein_seq'], tox_mean, tox_flags, drug_score

# === Collate function for batching ===
def collate_fn(batch):
    graphs, sequences, qms, smis, seqs, tox_means, tox_flags, drug_scores = zip(*batch)
    graph_batch = GraphBatch.from_data_list(graphs)
    sequence_batch = pad_sequence(sequences, batch_first=True)
    qm_batch = torch.stack(qms)
    return graph_batch, sequence_batch, qm_batch, smis, seqs, tox_means, tox_flags, drug_scores

# === Load model from checkpoint ===
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_ligand = LigandGCN()
    model_protein = ProteinCNN()
    model_predictor = AffinityPredictor(
        ligand_dim=135,
        protein_dim=128,
        dropout=0.2233207781090059
    )
    model_ligand.load_state_dict(checkpoint['ligand_model'])
    model_protein.load_state_dict(checkpoint['protein_model'])
    model_predictor.load_state_dict(checkpoint['predictor'])

    model_ligand.eval()
    model_protein.eval()
    model_predictor.eval()
    return model_ligand, model_protein, model_predictor

# === Run Inference ===
def run_inference(csv_path, model_path, output_path, batch_size=64):
    dataset = LigandProteinDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model_ligand, model_protein, model_predictor = load_model(model_path)

    preds = []
    with torch.no_grad():
        for graph_batch, prot_batch, qm_batch, smis, seqs, tox_means, tox_flags, drug_scores in tqdm(dataloader):
            lig_embed = model_ligand(graph_batch)
            lig_embed = torch.cat([lig_embed, qm_batch], dim=1)
            prot_embed = model_protein(prot_batch)
            outputs = model_predictor(lig_embed, prot_embed)

            for s, p, score, tox_m, tox_f, dscore in zip(smis, seqs, outputs.squeeze().tolist(), tox_means, tox_flags, drug_scores):
                preds.append((s, p, score, tox_m, tox_f, dscore))

    # Create DataFrame
    pred_df = pd.DataFrame(preds, columns=[
        'smiles', 'protein_seq', 'predicted_kiba',
        'tox_mean', 'tox_flags', 'drug_likeness_score'
    ])

    # Sort and filter top 200
    pred_df = pred_df.sort_values(by="predicted_kiba", ascending=False).head(200)

    # Save to CSV
    pred_df.to_csv(output_path, index=False)
    print(f"✅ Top 200 predictions saved to {output_path}")

# === Main Entry Point ===
if __name__ == "__main__":
    run_inference(
        csv_path="../data/ligand_protein_pairs.csv",
        model_path="kiba_best_qm.pt",
        output_path="top200_kiba_predictions.csv",
        batch_size=64
    )

