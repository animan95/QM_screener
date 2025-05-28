import torch
from torch.utils.data import Dataset
import pandas as pd
from graph_utils import mol_to_graph
from seq_utils import encode_sequence

class KIBADataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = mol_to_graph(row['SMILES'])
        sequence = encode_sequence(row['Sequence'])
        #sequence = row['Sequence']
        label = torch.tensor([row['KIBA_Score']], dtype=torch.float)
        #label = torch.tensor([row["AffinityNorm"]], dtype=torch.float)
        qm_tensor = torch.tensor([
            row["homo"],
            row["lumo"],
            row["lumo"]-row["homo"],
            row["dipole_x"],
            row["dipole_y"],
            row["dipole_z"],
            row["dipole_total"]
        ], dtype=torch.float)
        return graph, sequence, label, qm_tensor

