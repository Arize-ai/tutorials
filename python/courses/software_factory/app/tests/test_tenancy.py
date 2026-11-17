"""One workspace must never see or change another's data, whatever IDs it guesses."""

from conftest import new_ticket


def test_tickets_are_invisible_across_workspaces(client, acme, globex):
    t = new_ticket(client, acme, subject="Acme secret")
    n = t["number"]
    assert client.get("/api/v1/tickets", headers=globex.read_h).json() == []
    assert client.get(f"/api/v1/tickets/{n}", headers=globex.read_h).status_code == 404
    patch = client.patch(f"/api/v1/tickets/{n}", json={"status": "closed"}, headers=globex.write_h)
    assert patch.status_code == 404
    assert client.get(f"/api/v1/tickets/{n}/comments", headers=globex.read_h).status_code == 404
    comment = client.post(
        f"/api/v1/tickets/{n}/comments", json={"body": "x"}, headers=globex.write_h
    )
    assert comment.status_code == 404
    assert client.get(f"/api/v1/tickets/{n}", headers=acme.read_h).json()["status"] == "open"


def test_same_number_resolves_to_each_workspaces_own_ticket(client, acme, globex):
    new_ticket(client, acme, subject="Acme one")
    new_ticket(client, globex, subject="Globex one")
    assert client.get("/api/v1/tickets/1", headers=acme.read_h).json()["subject"] == "Acme one"
    assert client.get("/api/v1/tickets/1", headers=globex.read_h).json()["subject"] == "Globex one"


def test_cannot_assign_another_workspaces_agent(client, acme, globex):
    r = client.post(
        "/api/v1/tickets",
        json={"subject": "Hi", "requester_email": "a@b.test", "assignee_id": globex.agent.id},
        headers=acme.write_h,
    )
    assert r.status_code == 422
    t = new_ticket(client, acme)
    r = client.patch(
        f"/api/v1/tickets/{t['number']}",
        json={"assignee_id": globex.agent.id},
        headers=acme.write_h,
    )
    assert r.status_code == 422


def test_api_keys_and_webhooks_are_per_workspace(client, acme, globex):
    globex_keys = client.get("/api/v1/api-keys", headers=globex.admin_h).json()
    assert (
        client.delete(f"/api/v1/api-keys/{globex_keys[0]['id']}", headers=acme.admin_h).status_code
        == 404
    )
    hook = client.post(
        "/api/v1/webhooks",
        json={"url": "https://globex.test/hook", "events": ["ticket.created"]},
        headers=globex.admin_h,
    ).json()
    assert client.get("/api/v1/webhooks", headers=acme.admin_h).json() == []
    assert client.delete(f"/api/v1/webhooks/{hook['id']}", headers=acme.admin_h).status_code == 404
    assert (
        client.get(f"/api/v1/webhooks/{hook['id']}/deliveries", headers=acme.admin_h).status_code
        == 404
    )


def test_webhooks_only_fire_for_their_own_workspace(client, acme, globex, receiver):
    client.post(
        "/api/v1/webhooks",
        json={"url": "https://globex.test/hook", "events": ["ticket.created"]},
        headers=globex.admin_h,
    )
    new_ticket(client, acme)
    assert receiver.requests == []


def test_users_list_is_per_workspace(client, acme, globex):
    emails = {u["email"] for u in client.get("/api/v1/users", headers=globex.read_h).json()}
    assert acme.agent.email not in emails
