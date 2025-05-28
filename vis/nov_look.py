import pandas as pd
import requests
import time

# === INPUT ===
CSV_PATH = "../src/top200_kiba_predictions.csv"  # must have "smiles" column
df = pd.read_csv(CSV_PATH)

def query_pubchem_cid(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    return None

# === QUERY EACH SMILES ===
df["PubChem_CID"] = None

for i, row in df.iterrows():
    smiles = row["smiles"]
    cid = query_pubchem_cid(smiles)
    df.at[i, "PubChem_CID"] = cid
    print(f"[{i+1}/{len(df)}] {smiles} → CID: {cid}")
    time.sleep(0.3)  # To avoid rate limiting

df["is_novel"] = df["PubChem_CID"].isnull()
df.to_csv("top200_pubchem_checked.csv", index=False)
print("✅ Saved with PubChem results.")

