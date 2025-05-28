import pandas as pd
import requests
import os
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# === INPUT ===
CSV = "../src/top20_filtered_hits.csv"  # Must contain a 'Sequence' column
OUTPUT_DIR = "alphafold_pdbs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def blast_sequence_to_uniprot(sequence: str) -> str:
    """BLAST a sequence and return the best UniProt ID if found"""
    record = SeqRecord(Seq(sequence), id="query")
    result_handle = NCBIWWW.qblast("blastp", "swissprot", record.format("fasta"))
    blast_record = NCBIXML.read(result_handle)
    for alignment in blast_record.alignments:
        accession = alignment.accession
        return accession  # e.g., 'P00533'
    return None

def download_alphafold_pdb(uniprot_id: str, save_path: str) -> bool:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        with open(save_path, "w") as f:
            f.write(r.text)
        return True
    return False

# === PROCESS ===
df = pd.read_csv(CSV)
df["uniprot_id"] = None
df["pdb_path"] = None

for i, row in df.iterrows():
    seq = row["protein_seq"]
    print(f"🔎 BLASTing sequence {i}...")
    try:
        uniprot_id = blast_sequence_to_uniprot(seq)
        df.at[i, "uniprot_id"] = uniprot_id
        if uniprot_id:
            save_path = os.path.join(OUTPUT_DIR, f"{uniprot_id}.pdb")
            if not os.path.exists(save_path):
                success = download_alphafold_pdb(uniprot_id, save_path)
                if success:
                    df.at[i, "pdb_path"] = save_path
                    print(f"✅ Downloaded AlphaFold PDB for {uniprot_id}")
                else:
                    print(f"❌ Failed to download AlphaFold PDB for {uniprot_id}")
            else:
                df.at[i, "pdb_path"] = save_path
    except Exception as e:
        print(f"⚠️ Error processing sequence {i}: {e}")

df.to_csv("top20_with_pdbs.csv", index=False)
print("✅ All AlphaFold PDBs saved and mapped.")

