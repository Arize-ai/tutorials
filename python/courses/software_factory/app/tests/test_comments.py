from conftest import new_ticket


def test_add_and_list_comments(client, acme):
    t = new_ticket(client, acme)
    url = f"/api/v1/tickets/{t['number']}/comments"
    r = client.post(url, json={"body": "On it."}, headers=acme.write_h)
    assert r.status_code == 201
    assert r.json()["author_name"] == "API key: write"
    assert r.json()["internal"] is False
    client.post(
        url,
        json={"body": "Customer is on the enterprise plan.", "internal": True},
        headers=acme.write_h,
    )
    comments = client.get(url, headers=acme.read_h).json()
    assert [(c["body"], c["internal"]) for c in comments] == [
        ("On it.", False),
        ("Customer is on the enterprise plan.", True),
    ]


def test_comment_needs_a_body_and_write_scope(client, acme):
    t = new_ticket(client, acme)
    url = f"/api/v1/tickets/{t['number']}/comments"
    assert client.post(url, json={"body": ""}, headers=acme.write_h).status_code == 422
    assert client.post(url, json={"body": "hi"}, headers=acme.read_h).status_code == 403


def test_comment_on_missing_ticket(client, acme):
    r = client.post("/api/v1/tickets/42/comments", json={"body": "hi"}, headers=acme.write_h)
    assert r.status_code == 404
