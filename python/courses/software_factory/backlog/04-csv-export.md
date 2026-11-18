---
title: Export tickets to CSV
expect: human
why: Ticket text is customer-written, so a naive export enables CSV formula injection, and an export query is easy to leave unscoped. The security reviewer should flag both, and the spec should mark it at least medium.
---
Workspace admins want a monthly export of their tickets for reporting.

Add an **Export CSV** button on the ticket list that downloads the tickets in the current
view, with these columns: number, subject, status, priority, assignee, requester, created,
updated. An API endpoint for the same export would be great too, so it can be scheduled.
