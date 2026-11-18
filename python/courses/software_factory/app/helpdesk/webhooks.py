"""Outgoing webhooks.

A change to a ticket queues one WebhookDelivery per subscribed webhook, in the same
transaction as the change. Deliveries are sent after the response, as a background
task, and each attempt is recorded on the delivery row. There are no retries yet.
"""

from __future__ import annotations

import enum
import json
import logging
import time

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .db import utcnow
from .models import Webhook, WebhookDelivery
from .security import sign_payload

log = logging.getLogger(__name__)


class Event(enum.StrEnum):
    ticket_created = "ticket.created"
    ticket_updated = "ticket.updated"
    comment_created = "comment.created"


def enqueue(db: Session, tenant_id: int, event: Event, data: dict) -> list[WebhookDelivery]:
    """Queue a delivery of this event to each of the tenant's subscribed webhooks."""
    hooks = db.scalars(
        select(Webhook).where(Webhook.tenant_id == tenant_id, Webhook.active.is_(True))
    ).all()
    queued = []
    for hook in hooks:
        if event.value not in hook.event_list:
            continue
        body = {"event": event.value, "created_at": utcnow().isoformat(), "data": data}
        delivery = WebhookDelivery(
            tenant_id=tenant_id, webhook=hook, event=event.value, payload=json.dumps(body)
        )
        db.add(delivery)
        queued.append(delivery)
    return queued


def deliver(delivery: WebhookDelivery, client: httpx.Client) -> None:
    body = delivery.payload.encode()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "helpdesk-webhooks/1",
        "X-Helpdesk-Event": delivery.event,
        "X-Helpdesk-Delivery": str(delivery.id),
        "X-Helpdesk-Signature": sign_payload(delivery.webhook.secret, int(time.time()), body),
    }
    try:
        response = client.post(delivery.webhook.url, content=body, headers=headers)
    except httpx.HTTPError as e:
        delivery.status_code = None
        delivery.error = f"{type(e).__name__}: {e}"[:500]
    else:
        delivery.status_code = response.status_code
        delivery.error = None if response.is_success else f"HTTP {response.status_code}"
    delivery.attempted_at = utcnow()


def deliver_pending(session_factory: sessionmaker, client: httpx.Client) -> int:
    """Send every delivery that hasn't been attempted. Returns how many were sent."""
    with session_factory() as db:
        pending = db.scalars(
            select(WebhookDelivery)
            .where(WebhookDelivery.attempted_at.is_(None))
            .order_by(WebhookDelivery.id)
        ).all()
        for delivery in pending:
            deliver(delivery, client)
            db.commit()
            if delivery.error:
                log.warning("webhook delivery %s failed: %s", delivery.id, delivery.error)
        return len(pending)
