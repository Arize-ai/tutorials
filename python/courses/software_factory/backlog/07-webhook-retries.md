---
title: Retry failed webhook deliveries
expect: human
why: Needs new delivery columns (attempt count, next attempt time), which touches the data model, a sensitive path. There are no migrations, so the release notes must say existing databases need recreating.
---
When a customer's endpoint is down for a minute, they miss events for good: each
delivery is attempted once.

Please retry failed deliveries (network errors and 5xx responses, not 4xx) up to 5 times
with increasing delays, and show the attempt count in the delivery log in Settings and
in `GET /api/v1/webhooks/{id}/deliveries`. We can run the retries from a command, for
example `python -m helpdesk deliver-webhooks` on a cron. A background worker isn't
needed yet.
