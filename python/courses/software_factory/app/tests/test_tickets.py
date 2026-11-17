from conftest import new_ticket


def test_create_ticket_defaults(client, acme):
    t = new_ticket(client, acme, description="It's smoking.")
    assert t["number"] == 1
    assert t["status"] == "open"
    assert t["priority"] == "normal"
    assert t["assignee"] is None
    assert t["description"] == "It's smoking."
    assert t["created_at"].endswith("Z")


def test_numbers_count_up_per_workspace(client, acme, globex):
    assert [new_ticket(client, acme)["number"] for _ in range(3)] == [1, 2, 3]
    assert new_ticket(client, globex)["number"] == 1


def test_create_rejects_bad_input(client, acme):
    bad = [
        {"subject": "", "requester_email": "a@b.test"},
        {"subject": "Hi", "requester_email": "not-an-email"},
        {"subject": "Hi", "requester_email": "a@b.test", "priority": "whenever"},
        {"subject": "Hi", "requester_email": "a@b.test", "tenant_id": 2},
    ]
    for body in bad:
        r = client.post("/api/v1/tickets", json=body, headers=acme.write_h)
        assert r.status_code == 422, body


def test_get_ticket_and_missing_ticket(client, acme):
    t = new_ticket(client, acme)
    assert client.get(f"/api/v1/tickets/{t['number']}", headers=acme.read_h).json() == t
    r = client.get("/api/v1/tickets/999", headers=acme.read_h)
    assert r.status_code == 404
    assert r.json() == {"detail": "ticket 999 not found"}


def test_update_only_changes_sent_fields(client, acme):
    t = new_ticket(client, acme, priority="high")
    r = client.patch(
        f"/api/v1/tickets/{t['number']}", json={"status": "pending"}, headers=acme.write_h
    )
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["priority"] == "high"


def test_assign_and_unassign(client, acme):
    t = new_ticket(client, acme)
    url = f"/api/v1/tickets/{t['number']}"
    r = client.patch(url, json={"assignee_id": acme.agent.id}, headers=acme.write_h)
    assert r.json()["assignee"]["email"] == acme.agent.email
    r = client.patch(url, json={"assignee_id": None}, headers=acme.write_h)
    assert r.json()["assignee"] is None


def test_status_cannot_be_nulled(client, acme):
    t = new_ticket(client, acme)
    r = client.patch(f"/api/v1/tickets/{t['number']}", json={"status": None}, headers=acme.write_h)
    assert r.status_code == 422


def test_list_filters(client, acme):
    a = new_ticket(client, acme, priority="urgent")
    b = new_ticket(client, acme)
    client.patch(f"/api/v1/tickets/{b['number']}", json={"status": "solved"}, headers=acme.write_h)
    c = new_ticket(client, acme, assignee_id=acme.agent.id)

    def numbers(query: str) -> set[int]:
        r = client.get(f"/api/v1/tickets{query}", headers=acme.read_h)
        assert r.status_code == 200, r.text
        return {t["number"] for t in r.json()}

    assert numbers("") == {a["number"], b["number"], c["number"]}
    assert numbers("?status=solved") == {b["number"]}
    assert numbers("?status=open&status=pending") == {a["number"], c["number"]}
    assert numbers("?priority=urgent") == {a["number"]}
    assert numbers(f"?assignee_id={acme.agent.id}") == {c["number"]}


def test_list_is_most_recently_updated_first(client, acme):
    first = new_ticket(client, acme)
    second = new_ticket(client, acme)
    client.patch(
        f"/api/v1/tickets/{first['number']}", json={"priority": "high"}, headers=acme.write_h
    )
    listed = client.get("/api/v1/tickets", headers=acme.read_h).json()
    assert [t["number"] for t in listed] == [first["number"], second["number"]]


def test_list_limit_is_capped(client, acme):
    assert client.get("/api/v1/tickets?limit=500", headers=acme.read_h).status_code == 422


def test_users_endpoint_lists_the_workspace_team(client, acme, globex):
    users = client.get("/api/v1/users", headers=acme.read_h).json()
    assert {u["email"] for u in users} == {acme.admin.email, acme.agent.email}
    assert all("password_hash" not in u for u in users)
