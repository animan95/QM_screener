from tdc.multi_pred import DTI
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def extract_sequences(dataset_name='BindingDB_Ki', max_sequences=1000):
    print(f"📡 Downloading protein sequences from TDC dataset: {dataset_name}")
    
    # Load dataset
    dataset = DTI(name=dataset_name)
    data = dataset.get_data()

    # Extract unique protein sequences
    unique_sequences = data['Target'].dropna().unique()
    print(f"🔍 Found {len(unique_sequences)} unique protein sequences")

    # Limit to max_sequences
    selected = unique_sequences[:max_sequences]
    return selected

def save_csv(sequences, filename="bindingdb_proteins.csv"):
    df = pd.DataFrame({"sequence": sequences})
    df.to_csv(filename, index=False)
    print(f"📤 Saved CSV: {filename} ({len(df)} sequences)")

def save_fasta(sequences, filename="bindingdb_proteins.fasta"):
    records = [SeqRecord(Seq(seq), id=f"bindingdb_prot{i+1}", description="") for i, seq in enumerate(sequences)]
    SeqIO.write(records, filename, "fasta")
    print(f"📤 Saved FASTA: {filename} ({len(records)} sequences)")

def main():
    sequences = extract_sequences(dataset_name='BindingDB_Ki', max_sequences=1000)
    save_csv(sequences, "bindingdb_proteins.csv")
    save_fasta(sequences, "bindingdb_proteins.fasta")

if __name__ == "__main__":
    main()

