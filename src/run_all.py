import subprocess
import sys


MODULES = [
    "src.prepare_amazon",
    "src.build_probes",
    "src.probe_knowledge",
    "src.run_rerank",
    "src.run_baselines",
    "src.analyze_gain",
]


def main():
    for module in MODULES:
        print("\n" + "=" * 80)
        print("RUN:", module)
        print("=" * 80)

        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
        )


if __name__ == "__main__":
    main()
