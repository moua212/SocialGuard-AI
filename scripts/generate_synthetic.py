"""Generate synthetic accounts/posts CSVs.   python scripts/generate_synthetic.py --n 2000 --out ml/data/synthetic"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data.synthetic import generate_dataset

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="ml/data/synthetic"); a = ap.parse_args()
acc, posts = generate_dataset(a.n, a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
acc.to_csv(out / "accounts.csv", index=False); posts.to_csv(out / "posts.csv", index=False)
flat = posts.merge(acc.drop(columns=["label", "profile", "label_noise_flag"]), on="account_id")
flat.head(20000).to_csv(out / "sample_upload.csv", index=False)
print(f"{len(acc)} accounts, {len(posts)} posts -> {out}  (train on it: python -m ml.training.train --data-dir {out})")
