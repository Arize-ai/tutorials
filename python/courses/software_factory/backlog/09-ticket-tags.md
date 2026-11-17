---
title: Tag tickets
expect: human
why: A bigger feature with a new table (data model, a sensitive path) across the API, the UI, filters and webhook payloads. It tests whether the builder keeps scope tight and whether reviewers check tag lookups are workspace-scoped.
---
We want to tag tickets (`billing`, `bug`, `vip` and so on) so we can group and filter
them.

- Agents can add and remove tags on the ticket page.
- The ticket list can be filtered by a tag.
- Tags are included on tickets in the API, can be set with `POST` and `PATCH`, and can be
  filtered with `GET /api/v1/tickets?tag=billing`.
- Each workspace has its own tags.
