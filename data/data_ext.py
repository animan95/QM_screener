import subprocess

# List of scripts to run in order
scripts = [
    "ext_drug.py",
    "gen_cad.py",
    "qm_zn_mrg_smi.py",
    "lip_fil.py",
    "ext_prot.py",
    "lp_comb.py"
]

for script in scripts:
    print(f"🔄 Running: {script}")
    try:
        result = subprocess.run(["python", script], check=True)
        print(f"✅ Completed: {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {script} (exit code: {e.returncode})")
        break

