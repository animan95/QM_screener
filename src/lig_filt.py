import pandas as pd

# Load dataset
df = pd.read_csv("chembl_qm_tox_scored.csv")  # Replace with your actual file path

# ---- Define thresholds ----
thresholds = {
    "homo": -0.25,                  # HOMO must be more negative (stable)
    "lumo": -0.08,                  # LUMO must be more negative (reactivity)
    "gap_min": 0.12,# Moderate HOMO–LUMO gap
    "gap_max" : 0.22,
    "dipole_total": 6.5,         # Limit overall polarity
    "tox_mean": 0.3,              # Low predicted toxicity
    "tox_flags": 0,               # At most 1 toxicity alert
    "drug_likeness_score": 0.8    # Only compounds with decent drug-likeness
}

# ---- Apply filters ----
filtered_df = df[
    (df['homo'] <= thresholds['homo']) &
    (df['lumo'] >= thresholds['lumo']) &
    (df['gap'] >= thresholds['gap_min']) &
    (df['gap'] <= thresholds['gap_max']) &
    (df['dipole_total'] <= thresholds['dipole_total']) &
    (df['tox_mean'] <= thresholds['tox_mean']) &
    (df['tox_flags'] <= thresholds['tox_flags']) &
    (df['drug_likeness_score'] >= thresholds['drug_likeness_score'])
].copy()

# Save the filtered ligands
filtered_df.to_csv("ligands_filtered.csv", index=False)
print(f"✅ Filtered ligands saved: {filtered_df.shape[0]} entries retained.")

