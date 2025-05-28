from tdc.generation import MolGen
from rdkit import Chem
from rdkit.Chem import Descriptors

def is_valid_druglike(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])
    return violations <= 1

# Load from PyTDC
data = MolGen(name='ZINC')
split = data.get_split()

# Combine all sets and deduplicate
all_smiles = set()
for part in ['train', 'valid', 'test']:
    df = split[part]
    all_smiles.update(df['smiles'].unique())

print(f"🔍 Loaded {len(all_smiles)} total SMILES")

# Filter drug-like molecules
filtered = [smi for smi in all_smiles if is_valid_druglike(smi)]
print(f"✅ Found {len(filtered)} valid drug-like candidates")

# Save to file
with open("zinc_molgen_druglike_candidates.smi", "w") as f:
    for smi in sorted(filtered):
        f.write(smi + "\n")

