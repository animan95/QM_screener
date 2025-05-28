import subprocess

# List of scripts to run in order
scripts = [
    "tox_scr.py",
    "dl_nn.py",
    "dl_scorer.py",
    "lig_filt.py",
    "kib_inf.py"
]

for script in scripts:
    print(f"🔄 Running: {script}")
    try:
        result = subprocess.run(["python", script], check=True)
        print(f"✅ Completed: {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {script} (exit code: {e.returncode})")
        break

