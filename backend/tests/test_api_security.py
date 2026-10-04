import io
import json

from app.security import reset_rate_limits

GOOD = {"account_age_days": 540, "followers": 1200, "following": 450, "posts_per_day": 38, "mean_interval": 37.2, "interval_std": 3.1}


def test_health_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_protected_routes_require_auth(client):
    for method, url in [("get", "/api/models"), ("get", "/api/predictions"), ("post", "/api/predict"), ("post", "/api/predict/batch"),
                        ("post", "/api/analyze/account"), ("post", "/api/analyze/upload"), ("get", "/api/audit-logs")]:
        assert getattr(client, method)(url).status_code == 401, url
    assert client.get("/api/models", headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


def test_roles_are_enforced(client, viewer_h, analyst_h, admin_h):
    assert client.get("/api/models", headers=viewer_h).status_code == 200
    assert client.post("/api/predict", json=GOOD, headers=viewer_h).status_code == 403
    assert client.get("/api/audit-logs", headers=analyst_h).status_code == 403
    assert client.get("/api/audit-logs", headers=admin_h).status_code == 200
    assert client.patch("/api/users/1/role", json={"role": "ADMIN"}, headers=analyst_h).status_code == 403


def test_registration_rules(client):
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "weak@b.co", "password": "alllowercase1234"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "viewer@example.com", "password": "ViewerPass123"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "viewer@example.com", "password": "wrong-password"}).status_code == 401


def test_predict_returns_expected_shape(client, analyst_h):
    r = client.post("/api/predict", json=GOOD, headers=analyst_h)
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("bot_probability", "human_probability", "risk_score", "risk_level", "top_factors", "model_version"):
        assert k in d
    assert abs(d["bot_probability"] + d["human_probability"] - 1) < 1e-3


def test_predict_rejects_invalid_payloads(client, analyst_h):
    assert client.post("/api/predict", json={"evil_feature": 1}, headers=analyst_h).status_code == 422
    assert client.post("/api/predict", json={"posts_per_day": "lots"}, headers=analyst_h).status_code == 422
    assert client.post("/api/predict", json={"posts_per_day": -5}, headers=analyst_h).status_code == 422
    assert client.post("/api/predict", json={"posts_per_day": 1e99}, headers=analyst_h).status_code == 422
    assert client.post("/api/predict", json={}, headers=analyst_h).status_code == 422
    assert client.post("/api/predict", content=b"not json", headers={**analyst_h, "Content-Type": "application/json"}).status_code == 422


def test_batch_and_stored_predictions(client, analyst_h, viewer_h):
    r = client.post("/api/predict/batch", json={"items": [GOOD, {**GOOD, "posts_per_day": 2, "mean_interval": 600, "interval_std": 900}]}, headers=analyst_h)
    assert r.status_code == 200 and len(r.json()["results"]) == 2
    assert client.post("/api/predict/batch", json={"items": []}, headers=analyst_h).status_code == 422
    lst = client.get("/api/predictions?limit=5", headers=viewer_h).json()["items"]
    assert lst and client.get(f"/api/predictions/{lst[0]['id']}", headers=viewer_h).status_code == 200
    assert client.get("/api/predictions/999999", headers=viewer_h).status_code == 404
    assert client.get("/api/predictions?risk_level=BOGUS", headers=viewer_h).status_code == 422


def test_analyze_account_end_to_end(client, analyst_h, sample_flat_csv):
    a, p, _ = sample_flat_csv
    aid = a.iloc[0]["account_id"]
    posts = p[p.account_id == aid][["timestamp", "text"]].to_dict("records")
    body = {"platform": "twitter", "account": {"account_id": aid, "account_age_days": 300, "followers_count": 100, "following_count": 200},
            "posts": posts, "include_llm": True}
    r = client.post("/api/analyze/account", json=body, headers=analyst_h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["assessment"]["risk_level"] in ("LOW", "MEDIUM", "HIGH") and d["charts"]["heatmap"] and d["recommendations"]
    assert d["llm"]["source"] == "template" and d["llm"]["uncertainty"]       # LLM disabled → deterministic template
    assert "not identity verification" in d["disclaimer"]
    assert client.post("/api/analyze/account", json={**body, "posts": posts[:2]}, headers=analyst_h).status_code == 422


def test_upload_bulk_and_export(client, analyst_h, viewer_h, sample_flat_csv):
    _, _, data = sample_flat_csv
    r = client.post("/api/analyze/upload", files={"file": ("accounts.csv", data, "text/csv")}, headers=analyst_h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_scored"] == 8 and d["results"][0]["rank"] == 1
    assert d["results"] == sorted(d["results"], key=lambda x: -x["bot_probability"])
    ex = client.get(d["export_url"], headers=viewer_h)
    assert ex.status_code == 200 and ex.text.startswith("account,platform,risk_level")


def test_json_upload_supported(client, analyst_h, sample_flat_csv):
    a, p, _ = sample_flat_csv
    payload = {"accounts": a.drop(columns=["label", "profile", "label_noise_flag"]).head(3).to_dict("records"),
               "posts": p[p.account_id.isin(a.account_id.head(3))].to_dict("records")}
    r = client.post("/api/analyze/upload", files={"file": ("d.json", json.dumps(payload).encode(), "application/json")}, headers=analyst_h)
    assert r.status_code == 200 and r.json()["n_scored"] == 3


def test_malicious_uploads_rejected(client, analyst_h):
    def up(name, content):
        return client.post("/api/analyze/upload", files={"file": (name, content, "text/csv")}, headers=analyst_h)
    assert up("evil.exe", b"MZ\x90\x00").status_code == 415
    assert up("evil.csv.exe", b"account_id,timestamp\n").status_code == 415
    assert up("x.csv", b"account_id,timestamp,text\n\x00\x01\x02\n").status_code == 415
    assert up("x.csv", b"<html><script>alert(1)</script></html>").status_code == 415
    assert up("x.csv", b"").status_code == 400
    assert up("x.csv", b"foo,bar\n1,2\n").status_code == 400                                   # missing columns
    assert up("x.json", b"{not valid json").status_code == 400
    assert up("x.csv", b"\xff\xfe\xfa binary").status_code in (400, 415)
    assert up("big.csv", b"account_id,timestamp,text\n" + b"a,2026-01-01T00:00:00Z,hello\n" * 80000).status_code == 413   # > 1 MB limit
    assert client.post("/api/analyze/upload", headers=analyst_h).status_code == 422


def test_formula_injection_is_neutralized(client, analyst_h, viewer_h):
    rows = ["account_id,timestamp,text"]
    for i in range(12):
        rows.append(f'=cmd|calc!A0,2026-01-0{1 + i % 9}T0{i % 9}:00:00Z,"@x hello {i}"')
    r = client.post("/api/analyze/upload", files={"file": ("f.csv", "\n".join(rows).encode(), "text/csv")}, headers=analyst_h)
    assert r.status_code == 200, r.text
    acc = r.json()["results"][0]["account"]
    assert not acc.startswith(("=", "+", "-", "@")) and acc.startswith("h_")
    csv_text = client.get(r.json()["export_url"], headers=viewer_h).text
    assert "=cmd" not in csv_text


def test_rate_limiting(client, analyst_h, monkeypatch):
    reset_rate_limits()
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    codes = [client.post("/api/predict", json=GOOD, headers=analyst_h).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
    reset_rate_limits()


def test_login_rate_limiting(client, monkeypatch):
    reset_rate_limits()
    monkeypatch.setenv("LOGIN_RATE_LIMIT_PER_MIN", "4")
    codes = [client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "x" * 12}).status_code for _ in range(6)]
    assert 429 in codes
    reset_rate_limits()


def test_audit_log_records_actions_and_security_headers(client, admin_h):
    logs = client.get("/api/audit-logs", headers=admin_h).json()
    actions = {l["action"] for l in logs}
    assert {"login_ok", "login_failed", "register"} <= actions
    h = client.get("/api/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"


def test_models_endpoints(client, viewer_h):
    m = client.get("/api/models", headers=viewer_h).json()
    v = m["active"]
    d = client.get(f"/api/models/{v}", headers=viewer_h)
    assert d.status_code == 200 and "curves" in d.json() and d.json()["features"]
    assert client.get("/api/models/not-a-version", headers=viewer_h).status_code == 404
    assert client.get("/api/dashboard/stats", headers=viewer_h).status_code == 200
