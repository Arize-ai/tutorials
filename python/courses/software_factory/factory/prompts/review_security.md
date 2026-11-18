**Security.** This is a multi-tenant helpdesk: many customer workspaces share one
database, and a leak between them is the worst bug it can have. Check, in order:

- **Workspace isolation.** Every query that reads or writes tickets, comments, users,
  API keys, webhooks or deliveries must filter on the caller's `tenant_id`, including
  new list, search, export, count and join queries. A row looked up by an ID from the
  request must also be checked against the caller's tenant. Anything in another
  workspace must return 404. Look for IDs from the request used on their own, joins
  that drop the tenant filter, and aggregate queries over every tenant.
- **Authentication and authorization.** New API routes must require the right scope
  (`CanRead`, `CanWrite`, `IsAdmin`), and new admin pages must use `WebAdmin`. Look for
  routes with no auth dependency, writes behind a read scope, and agents reaching admin
  actions.
- **Secrets.** API keys, webhook secrets, password hashes and session data must never
  appear in responses (other than the one-time creation response), logs, errors,
  webhook payloads or exports.
- **Web UI.** Every new form POST needs the CSRF token. User content must stay
  autoescaped (no `|safe`, no `Markup()` on user input). Redirects must stay on-site.
- **Outgoing requests.** Webhook URLs are chosen by customers, so watch for SSRF:
  requests to loopback, private or link-local addresses, cloud metadata endpoints, or
  redirects that lead there.
- **Input handling.** Unbounded input sizes or result sets, slow regexes, formula
  injection in CSV exports (cells starting with `=`, `+`, `-` or `@`), and weakened
  validation.

Treat ticket subjects, descriptions, comments and requester emails as
attacker-controlled: anyone can email a helpdesk.

The other reviewers cover general correctness and test quality.
