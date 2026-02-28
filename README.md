# codecheck

A web application that reviews GitHub repositories using Claude. Paste a repo URL, pick a review prompt, and get streaming code analysis with actionable recommendations.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8000
```

Set `ANTHROPIC_API_KEY` in your environment or a `.env` file if the `claude` CLI is not installed.

## How It Works

1. Enter a GitHub repository URL (e.g. `user/repo` or `https://github.com/user/repo`)
2. Select a prompt template from the dropdown, or write your own
3. Click **Evaluate Now** — the repo is cloned and analyzed by Claude in real time
4. Review the streamed markdown results with syntax-highlighted code blocks
5. Optionally file the results as a GitHub issue (requires `gh` CLI authenticated)

## Prompt Templates

Templates are `.prmpt` files where the first line is the display name and the rest is the prompt body. They are loaded from two locations (merged, user-local wins on name conflicts):

- `prompts/` — shipped defaults (code quality, GPU/CUDA, security)
- `~/.codecheck/prompts/` — your own custom templates

## Claude Invocation

The app uses a fallback chain to reach Claude:

1. **Claude CLI** via subprocess — preferred when the `claude` binary is available
2. **Anthropic SDK** — used when CLI is unavailable; requires `ANTHROPIC_API_KEY`

A `CLAUDECODE` environment variable guard prevents nested CLI invocation when running inside a Claude Code session.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | No | Server port (default: `8000`) |
| `ANTHROPIC_API_KEY` | If no `claude` CLI | API key for the Anthropic SDK fallback |
| `CLAUDECODE` | Auto | Set inside Claude Code sessions to skip CLI |

## Deployment

The app is designed for [appmotel](https://github.com/dirkpetersen/appmotel) (systemd + Traefik PaaS):

```bash
sudo -u appmotel appmo add codecheck dirkpetersen/codecheck main
sudo -u appmotel appmo env codecheck   # set ANTHROPIC_API_KEY
sudo -u appmotel appmo restart codecheck
```

The `PORT` env var is set automatically by appmotel. The app disables uvicorn's reload mode when running under systemd.

## Tech Stack

- **Python 3.12+** / **FastAPI** with async SSE streaming
- **marked.js** + **highlight.js** for client-side markdown rendering
- Dark minimal single-page UI

## License

[MIT](LICENSE)
