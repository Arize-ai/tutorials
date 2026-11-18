---
title: Replying to a ticket doesn't move it to the top of the queue
expect: automerge
why: A small bug with an obvious test. Adding a comment never touches the ticket row, so `updated_at` and the queue order stay stale.
---
When someone replies to a ticket or adds an internal note, the ticket's **Last updated**
time doesn't change, and the ticket stays where it was in the queue. The queue is sorted
by most recently updated, so tickets with fresh replies end up buried under older ones.

Steps to reproduce:

1. Open any ticket that isn't at the top of the queue.
1. Add a reply.
1. Go back to the ticket list.

Expected: the ticket is at the top and shows the time of the reply.
Actual: it's in the same place with the old time. The same happens with replies added
through `POST /api/v1/tickets/{number}/comments`.
