# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (the Security tab, "Report a vulnerability"). Please do not open a public issue for a vulnerability. Include a description, the steps to reproduce it, its impact and a fix if you have one.

Lesto is a personal project: reports are read, and fixes are made when time allows, with no promised response time.

## Security model

Lesto is an inference server for one machine or a trusted local network. It is not built for exposure to the internet or to untrusted networks.

- **The default bind is `0.0.0.0`** (port 11234), so a server started with no `--host` is reachable from every device on the local network. `--host 127.0.0.1` keeps it on the machine.
- **Authentication is off by default.** When a key is set with `--api-key <token>` (or `--api-key-env <VAR>`, which keeps the key out of the process table), once the server is ready, every request needs it except `GET /health` and any `OPTIONS` request (answered with 204). Connections from a loopback address (127.0.0.0/8 or ::1) are exempt unless `--api-key-strict` is set; a client on the same machine that connects through the machine's network address is not. No key is set when `--api-key` is empty or the `--api-key-env` variable is unset or empty (and no earlier option set one); the server then answers everyone, and nothing warns about it, so check that the key is really set. While a model passed with `--model` is still loading, the server checks no key: it answers every request with 503, except that a request for `/v1/load-progress` (any method) from a loopback address gets 200 and a load-progress report.
- **There is no TLS.** Traffic, including the API key, is plain HTTP. Put a reverse proxy that terminates TLS in front of it if the server must be reached across a network you do not control.
- **The server runs no tools.** It renders chat templates and parses a model's tool calls into the response; it never executes a command or touches files on a model's behalf. A client that does so is responsible for confining it.
- **Models are code you load.** Load models only from sources you trust, and do not run the server as root.
