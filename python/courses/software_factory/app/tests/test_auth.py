from helpdesk import security
from helpdesk.models import ApiKey


def test_health_needs_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_requires_a_bearer_key(client, acme):
    r = client.get("/api/v1/tickets")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/v1/tickets", headers={"Authorization": "Basic abc"}).status_code == 401


def test_unknown_and_tampered_keys_are_rejected(client, acme):
    assert client.get("/api/v1/me", headers=acme.auth("hd_nope")).status_code == 401
    tampered = acme.read_key[:-2] + ("AA" if not acme.read_key.endswith("AA") else "BB")
    assert client.get("/api/v1/me", headers=acme.auth(tampered)).status_code == 401


def test_revoked_key_stops_working(client, acme, db):
    keys = client.get("/api/v1/api-keys", headers=acme.admin_h).json()
    read = next(k for k in keys if k["name"] == "read")
    assert client.delete(f"/api/v1/api-keys/{read['id']}", headers=acme.admin_h).status_code == 204
    assert client.get("/api/v1/me", headers=acme.read_h).status_code == 401


def test_scopes_are_enforced(client, acme):
    body = {"subject": "Hi", "requester_email": "a@b.test"}
    assert client.post("/api/v1/tickets", json=body, headers=acme.read_h).status_code == 403
    assert client.post("/api/v1/tickets", json=body, headers=acme.write_h).status_code == 201
    assert client.get("/api/v1/api-keys", headers=acme.write_h).status_code == 403
    assert client.get("/api/v1/api-keys", headers=acme.admin_h).status_code == 200


def test_admin_scope_implies_the_others(client, acme):
    r = client.post(
        "/api/v1/tickets",
        json={"subject": "Hi", "requester_email": "a@b.test"},
        headers=acme.admin_h,
    )
    assert r.status_code == 201


def test_me_reports_the_key(client, acme):
    me = client.get("/api/v1/me", headers=acme.read_h).json()
    assert me == {
        "tenant": {"slug": "acme", "name": "Acme"},
        "actor": "API key: read",
        "scopes": ["tickets:read"],
    }


def test_using_a_key_records_last_used(client, acme, db):
    client.get("/api/v1/me", headers=acme.read_h)
    db.expire_all()
    key = db.query(ApiKey).filter_by(name="read").one()
    assert key.last_used_at is not None


def test_created_key_is_shown_once_and_stored_hashed(client, acme, db):
    r = client.post(
        "/api/v1/api-keys", json={"name": "ci", "scopes": ["tickets:read"]}, headers=acme.admin_h
    )
    assert r.status_code == 201
    token = r.json()["key"]
    assert token.startswith("hd_")
    assert client.get("/api/v1/me", headers=acme.auth(token)).status_code == 200
    listed = client.get("/api/v1/api-keys", headers=acme.admin_h).json()
    assert all("key" not in k for k in listed)
    stored = db.query(ApiKey).filter_by(name="ci").one()
    assert token not in (stored.key_hash, stored.prefix)


def test_password_hashing_round_trip():
    stored = security.hash_password("s3cret-pass")
    assert stored.startswith("pbkdf2_sha256$")
    assert security.verify_password("s3cret-pass", stored)
    assert not security.verify_password("wrong", stored)
    assert not security.verify_password("s3cret-pass", "garbage")
