from pathlib import Path
import argparse
import datetime
import tarfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

root = Path.cwd()
cutoff = time.time() - 3 * 86400

search_dirs = [
    Path("logs"),
    Path("runs"),
    Path("baseline_runs"),
    Path("benchmark_runs"),
    Path("cross_model_runs"),
    Path("method_runs"),
    Path("results"),
]

def is_diagnostic(p):
    name = p.name

    return (
        name.endswith("_training_history.csv")
        or name == "prediction_diagnostics.csv"
        or name == "conditional_feature_importance.csv"
    )

candidates = []

for folder in search_dirs:

    if not folder.exists():
        continue

    for p in folder.rglob("*"):

        if not p.is_file():
            continue

        stat = p.stat()

        if stat.st_mtime >= cutoff:
            continue

        if (
            (folder == Path("logs") and p.suffix == ".log")
            or is_diagnostic(p)
        ):
            candidates.append(p)

candidates.sort()

total = sum(p.stat().st_size for p in candidates)

print("=" * 70)
print("OLD DIAGNOSTIC CLEANUP")
print("=" * 70)

print("Files:", len(candidates))
print("Size GiB:", round(total / 1024**3, 4))

for p in sorted(
    candidates,
    key=lambda x: x.stat().st_size,
    reverse=True,
)[:40]:
    print(
        f"{p.stat().st_size / 1024**2:10.2f} MiB",
        p,
    )

if not args.apply:
    print()
    print("DRY RUN ONLY")
    print("执行 --apply 才会归档并删除上述原文件")
    raise SystemExit(0)

if not candidates:
    print("没有符合条件的文件")
    raise SystemExit(0)

archive_dir = root / "archives"
archive_dir.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

archive = archive_dir / f"old_diagnostics_{timestamp}.tar.gz"

# 先压缩保存
with tarfile.open(archive, "w:gz") as tf:
    for p in candidates:
        tf.add(p, arcname=str(p))

# 验证归档内容能完整读取
with tarfile.open(archive, "r:gz") as tf:

    members = {
        m.name: m
        for m in tf.getmembers()
        if m.isfile()
    }

    assert len(members) == len(candidates)

    for p in candidates:
        name = str(p)
        assert name in members
        assert members[name].size == p.stat().st_size

        f = tf.extractfile(members[name])
        assert f is not None

        while f.read(1024 * 1024):
            pass

# 成功验证以后再删除原文件
for p in candidates:
    p.unlink()

print()
print("ARCHIVE:", archive)
print("ARCHIVE SIZE GiB:",
      round(archive.stat().st_size / 1024**3, 4))
print("REMOVED ORIGINAL FILES:", len(candidates))
print("CLEANUP COMPLETE")
