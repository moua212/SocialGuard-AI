import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="sg-tests-"))
os.environ.update({
    "SG_ARTIFACTS_DIR": str(TMP / "artifacts"), "SG_DOCS_DIR": str(TMP / "docs"), "DATABASE_URL": f"sqlite:///{TMP}/test.db",
    "SECRET_KEY": "test-secret-key-not-for-production-0123456789", "ADMIN_EMAIL": "admin@example.com", "ADMIN_PASSWORD": "AdminPass12345",
    "RATE_LIMIT_PER_MIN": "1000", "LOGIN_RATE_LIMIT_PER_MIN": "1000", "MAX_UPLOAD_MB": "1", "APP_ENV": "test"})
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    from ml.training.train import train
    version, metrics = train(n_accounts=320, fast=True, make_plots=False, write_report=False, verbose=False)
    return version, metrics


@pytest.fixture(scope="session")
def client(trained_model):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def _login(client, email, pw):
    r = client.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin_h(client):
    return _login(client, "admin@example.com", "AdminPass12345")


@pytest.fixture(scope="session")
def analyst_h(client, admin_h):
    r = client.post("/api/auth/register", json={"email": "analyst@example.com", "password": "AnalystPass123"})
    assert r.status_code == 201
    assert client.patch(f"/api/users/{r.json()['id']}/role", json={"role": "ANALYST"}, headers=admin_h).status_code == 200
    return _login(client, "analyst@example.com", "AnalystPass123")


@pytest.fixture(scope="session")
def viewer_h(client):
    assert client.post("/api/auth/register", json={"email": "viewer@example.com", "password": "ViewerPass123"}).status_code == 201
    return _login(client, "viewer@example.com", "ViewerPass123")


@pytest.fixture(scope="session")
def sample_flat_csv():
    from ml.data.synthetic import generate_dataset
    a, p = generate_dataset(8, seed=11)
    flat = p.merge(a.drop(columns=["label", "profile", "label_noise_flag"]), on="account_id")
    return a, p, flat.to_csv(index=False).encode()
