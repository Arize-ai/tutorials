from conftest import PASSWORD, csrf, login, new_ticket


def test_pages_redirect_to_login(client, acme):
    r = client.get("/tickets/1", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=%2Ftickets%2F1"


def test_login_and_logout(client, acme):
    r = login(client, acme)
    assert r.status_code == 303
    assert r.headers["location"] == "/tickets"
    page = client.get("/tickets")
    assert acme.agent.name in page.text
    client.post("/logout", data={"csrf_token": csrf(page.text)})
    assert client.get("/tickets", follow_redirects=False).status_code == 303


def test_wrong_password_and_wrong_workspace(client, acme, globex):
    token = csrf(client.get("/login").text)
    for workspace, password in [("acme", "nope"), ("globex", PASSWORD)]:
        r = client.post(
            "/login",
            data={
                "csrf_token": token,
                "workspace": workspace,
                "email": acme.agent.email,
                "password": password,
            },
        )
        assert r.status_code == 401
        assert "don&#39;t match" in r.text


def test_login_ignores_offsite_next(client, acme):
    token = csrf(client.get("/login").text)
    for target in ["https://evil.test/", "//evil.test/", "/\\evil.test"]:
        r = client.post(
            "/login",
            data={
                "csrf_token": token,
                "workspace": "acme",
                "email": acme.agent.email,
                "password": PASSWORD,
                "next": target,
            },
            follow_redirects=False,
        )
        assert r.headers["location"] == "/tickets"
        token = csrf(client.get("/login").text)


def test_forms_need_the_csrf_token(client, acme):
    login(client, acme)
    r = client.post("/tickets", data={"subject": "x", "requester_email": "a@b.test"})
    assert r.status_code == 403
    r = client.post(
        "/tickets", data={"subject": "x", "requester_email": "a@b.test", "csrf_token": "forged"}
    )
    assert r.status_code == 403


def test_create_reply_and_update_through_the_ui(client, acme):
    login(client, acme)
    token = csrf(client.get("/tickets/new").text)
    r = client.post(
        "/tickets",
        data={
            "csrf_token": token,
            "subject": "Can't log in",
            "requester_email": "Pat@Customer.test",
            "priority": "high",
            "assignee_id": str(acme.agent.id),
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/tickets/1"
    page = client.get("/tickets/1")
    assert "Can&#39;t log in" in page.text
    assert "pat@customer.test" in page.text

    client.post("/tickets/1/comments", data={"csrf_token": token, "body": "Reset your password."})
    client.post(
        "/tickets/1/comments", data={"csrf_token": token, "body": "VIP", "internal": "true"}
    )
    client.post("/tickets/1", data={"csrf_token": token, "status": "solved", "priority": "high"})

    page = client.get("/tickets/1").text
    assert "Reset your password." in page
    assert "Internal note" in page
    assert '<option value="solved" selected>' in page
    assert '<option value="">Unassigned</option>' in page


def test_new_ticket_validation_keeps_the_form(client, acme):
    login(client, acme)
    token = csrf(client.get("/tickets/new").text)
    r = client.post(
        "/tickets", data={"csrf_token": token, "subject": "Kept", "requester_email": "nope"}
    )
    assert r.status_code == 422
    assert 'value="Kept"' in r.text


def test_ticket_views(client, acme):
    a = new_ticket(client, acme, subject="Still open")
    b = new_ticket(client, acme, subject="All done")
    client.patch(f"/api/v1/tickets/{b['number']}", json={"status": "closed"}, headers=acme.write_h)
    login(client, acme)
    active = client.get("/tickets").text
    assert a["subject"] in active
    assert b["subject"] not in active
    assert b["subject"] in client.get("/tickets?view=closed").text
    assert b["subject"] in client.get("/tickets?view=all").text
    assert "No tickets here" in client.get("/tickets?mine=true").text


def test_ui_is_isolated_per_workspace(client, acme, globex):
    new_ticket(client, globex, subject="Globex only")
    login(client, acme)
    assert "Globex only" not in client.get("/tickets?view=all").text
    assert client.get("/tickets/1").status_code == 404


def test_settings_are_for_admins_only(client, acme):
    login(client, acme)
    assert client.get("/settings").status_code == 403


def test_admin_creates_and_revokes_an_api_key(client, acme):
    login(client, acme, acme.admin)
    token = csrf(client.get("/settings").text)
    page = client.post(
        "/settings/api-keys",
        data={"csrf_token": token, "name": "Zapier", "scopes": ["tickets:read"]},
    ).text
    assert "Copy the key now" in page
    shown = page.split('<code class="secret">')[1].split("</code>")[0]
    assert client.get("/api/v1/me", headers=acme.auth(shown)).status_code == 200
    assert shown not in client.get("/settings").text  # only shown once

    keys = client.get("/api/v1/api-keys", headers=acme.admin_h).json()
    zapier = next(k for k in keys if k["name"] == "Zapier")
    client.post(f"/settings/api-keys/{zapier['id']}/revoke", data={"csrf_token": token})
    assert client.get("/api/v1/me", headers=acme.auth(shown)).status_code == 401


def test_admin_adds_a_webhook(client, acme, receiver):
    login(client, acme, acme.admin)
    token = csrf(client.get("/settings").text)
    page = client.post(
        "/settings/webhooks",
        data={"csrf_token": token, "url": "https://hooks.test/in", "events": ["ticket.created"]},
    ).text
    assert "signing secret" in page
    new_ticket(client, acme)
    assert len(receiver.requests) == 1
    assert "ticket.created" in client.get("/settings").text
