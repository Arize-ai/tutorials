import hashlib
import hmac
import json

from conftest import new_ticket

from helpdesk.security import sign_payload


def add_hook(client, ws, events=("ticket.created", "ticket.updated", "comment.created")):
    r = client.post(
        "/api/v1/webhooks",
        json={"url": "https://hooks.customer.test/helpdesk", "events": list(events)},
        headers=ws.admin_h,
    )
    assert r.status_code == 201, r.text
    return r.json()


def verify(secret: str, request) -> dict:
    """What a receiver does: check the signature, then parse the body."""
    parts = dict(p.split("=", 1) for p in request.headers["x-helpdesk-signature"].split(","))
    expected = hmac.new(
        secret.encode(), f"{parts['t']}.".encode() + request.content, hashlib.sha256
    ).hexdigest()
    assert hmac.compare_digest(parts["v1"], expected)
    return json.loads(request.content)


def test_ticket_created_is_delivered_and_signed(client, acme, receiver):
    hook = add_hook(client, acme)
    assert hook["secret"].startswith("whsec_")
    t = new_ticket(client, acme, subject="Webhook me")
    assert len(receiver.requests) == 1
    req = receiver.requests[0]
    assert str(req.url) == "https://hooks.customer.test/helpdesk"
    assert req.headers["x-helpdesk-event"] == "ticket.created"
    body = verify(hook["secret"], req)
    assert body["event"] == "ticket.created"
    assert body["data"]["number"] == t["number"]
    assert body["data"]["subject"] == "Webhook me"


def test_update_and_comment_events(client, acme, receiver):
    hook = add_hook(client, acme)
    t = new_ticket(client, acme)
    client.patch(f"/api/v1/tickets/{t['number']}", json={"status": "pending"}, headers=acme.write_h)
    client.post(
        f"/api/v1/tickets/{t['number']}/comments", json={"body": "Hi"}, headers=acme.write_h
    )
    events = [verify(hook["secret"], r) for r in receiver.requests]
    assert [e["event"] for e in events] == ["ticket.created", "ticket.updated", "comment.created"]
    assert events[1]["data"]["changed"] == ["status"]
    assert events[2]["data"]["comment"]["body"] == "Hi"


def test_no_op_update_sends_nothing(client, acme, receiver):
    add_hook(client, acme, events=["ticket.updated"])
    t = new_ticket(client, acme)
    client.patch(f"/api/v1/tickets/{t['number']}", json={"status": "open"}, headers=acme.write_h)
    assert receiver.requests == []


def test_only_subscribed_events_are_sent(client, acme, receiver):
    add_hook(client, acme, events=["comment.created"])
    new_ticket(client, acme)
    assert receiver.requests == []


def test_deleted_webhook_stops_receiving(client, acme, receiver):
    hook = add_hook(client, acme)
    assert client.delete(f"/api/v1/webhooks/{hook['id']}", headers=acme.admin_h).status_code == 204
    new_ticket(client, acme)
    assert receiver.requests == []


def test_failed_deliveries_are_recorded(client, acme, receiver):
    hook = add_hook(client, acme)
    receiver.status = 500
    new_ticket(client, acme)
    receiver.fail = True
    new_ticket(client, acme)
    deliveries = client.get(
        f"/api/v1/webhooks/{hook['id']}/deliveries", headers=acme.admin_h
    ).json()
    assert [(d["status_code"], d["delivered"]) for d in deliveries] == [(None, False), (500, False)]
    assert deliveries[0]["error"].startswith("ConnectError")
    assert deliveries[1]["error"] == "HTTP 500"


def test_webhook_url_must_be_http(client, acme):
    r = client.post(
        "/api/v1/webhooks",
        json={"url": "ftp://x.test/", "events": ["ticket.created"]},
        headers=acme.admin_h,
    )
    assert r.status_code == 422


def test_sign_payload_format():
    sig = sign_payload("whsec_test", 1700000000, b'{"a":1}')
    expected = hmac.new(b"whsec_test", b'1700000000.{"a":1}', hashlib.sha256).hexdigest()
    assert sig == f"t=1700000000,v1={expected}"
