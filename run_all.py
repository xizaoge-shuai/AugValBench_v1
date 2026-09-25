import subprocess
import sys

STEPS = [
    "prepare_amazon.py",
    "build_probes.py",
    "probe_knowledge.py",
    "run_rerank.py",
    "run_baselines.py",
    "analyze_gain.py",
]

for script in STEPS:
    print("\n" + "=" * 80)
    print("RUN:", script)
    print("=" * 80)
    subprocess.run([sys.executable, script], check=True)
