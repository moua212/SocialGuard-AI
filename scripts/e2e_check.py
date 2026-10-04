"""End-to-end validation: upload -> validate -> features -> predict -> risk -> SHAP -> export, plus security probes.
Run against a live API:  python scripts/e2e_check.py http://localhost:8000 admin@example.com AdminPass"""
import sys, io, httpx
base, email, pw = sys.argv[1:4]
c = httpx.Client(base_url=base, timeout=120)
tok = c.post("/api/auth/login", json={"email": email, "password": pw}).json()["access_token"]
h = {"Authorization": f"Bearer {tok}"}
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data.synthetic import generate_dataset
a, p = generate_dataset(20, seed=5)
data = p.merge(a.drop(columns=["label", "profile", "label_noise_flag"]), on="account_id").to_csv(index=False).encode()
r = c.post("/api/analyze/upload", files={"file": ("e2e.csv", data)}, headers=h); r.raise_for_status(); d = r.json()
print("scored", d["n_scored"], "top:", d["results"][0]["account"], d["results"][0]["risk_level"])
assert c.get(d["export_url"], headers=h).text.startswith("account,"), "export failed"
assert c.post("/api/analyze/upload", files={"file": ("x.exe", b"MZ")}, headers=h).status_code == 415
assert c.get("/api/predictions").status_code == 401
print("E2E OK")
