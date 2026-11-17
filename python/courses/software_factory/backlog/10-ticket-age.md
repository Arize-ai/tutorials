---
title: Show how long each ticket has been waiting
expect: automerge
why: A low-risk UI change. The interesting part is correctness at the edges (timezones, just-created tickets, singular and plural), which is the correctness reviewer's job.
---
On the ticket list, add a **Waiting** column that shows how long ago each ticket was
created, in a compact form like `5m`, `3h` or `2d`. Hovering over it should show the
exact creation time.

Urgent tickets that have been open for more than 4 hours should stand out.
