**Security.** This app fetches data from GitHub based on user input, renders text
from pull requests into images and HTML, calls an image-generation API, and runs as
a web service. Look for: injection (HTML/SVG/script, prompt injection into the image
prompt, command or path injection), SSRF or fetching hosts other than
api.github.com, secrets or tokens that could leak into logs, cards, errors or
responses, unbounded input sizes or slow regexes (denial of service), unsafe
deserialisation, and weakened validation. Treat everything from a PR (title, body,
author names, commit messages) as attacker-controlled.

The other reviewers cover general correctness and test quality.
