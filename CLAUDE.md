# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**codecheck** is a Python web application that reviews GitHub repositories using Claude Code. Users paste a GitHub repo URL, select a prompt template, and click "Evaluate Now" to get code quality feedback and improvement suggestions. Optionally, results can be filed as a GitHub issue (requires `gh` CLI authenticated against GitHub).

## Application Architecture

The app has three user-facing inputs:
1. **GitHub URL field** — accepts `https://github.com/user/repo` or shorthand `user/repo`
2. **Prompt template dropdown** — populated from `.prmpt` files on the filesystem; the first line of each file is shown as the label
3. **Textarea** — pre-filled with the selected `.prmpt` file contents; editable before submission

On submit:
- Clone the target repo locally (shallow, depth 1)
- Use the textarea content as a prompt with Claude CLI (or Bedrock/Azure SDK fallback) to analyze the repo
- Stream results to the UI via SSE
- If `gh` CLI is authenticated, offer a button to file results as a GitHub issue

## Tech Stack

- **Language**: Python 3.12+
- **Web framework**: FastAPI (async, SSE streaming for live analysis output)
- **Prompt templates**: `.prmpt` files — first line is display name, rest is prompt body
- **Primary Claude invocation**: Claude CLI via `asyncio.create_subprocess_exec` using `stream-json` output format
- **Fallback Claude invocation**: Anthropic SDK via AWS Bedrock or Azure AI Foundry (`stream_sdk`)
- **External dependencies**: `gh` CLI (GitHub auth), `claude` CLI, `git`, `anthropic` SDK

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server (auto-reload)
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# App is served at http://localhost:8000
```

## Project Structure

```
app.py                  # FastAPI backend (routes, Claude invocation, SSE streaming)
static/index.html       # Single-page frontend (dark UI, markdown rendering)
prompts/*.prmpt         # Shipped prompt templates
~/.codecheck/prompts/   # User-local prompt templates (merged at runtime)
requirements.txt        # Python dependencies
claude-skills/          # Symlink to sibling repo with Claude invocation reference code
```

## Invoking Claude from Python

The app uses a two-tier fallback:

### Tier 1: Claude CLI via subprocess (preferred)
`stream_claude_cli` in `app.py` runs the Claude CLI with `--output-format stream-json --verbose` and parses newline-delimited JSON. Two event types carry content:
- `assistant` events: iterate `message.content[]` for `type=="text"` blocks
- `result` events: read the top-level `result` string

```python
cmd = [claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose"]
proc = await asyncio.create_subprocess_exec(*cmd, cwd=repo_dir, ...)
# parse JSON lines from proc.stdout
```

### Tier 2: Anthropic SDK via AWS Bedrock or Azure AI Foundry (fallback when CLI unavailable)
`stream_sdk` in `app.py` builds a repo context string from file contents, then streams via `AnthropicBedrock` or the Anthropic client with Azure base URL. The model defaults to `us.anthropic.claude-sonnet-4-6-20250514` and can be overridden with `ANTHROPIC_MODEL`. Set `AZURE_AI_FOUNDRY=1` to use Azure instead of Bedrock.

### Self-invocation guard
Claude Code **cannot invoke itself** (nested CLI calls crash). The `CLAUDECODE` environment variable is set when running inside a Claude Code session. `get_claude_bin()` returns `None` when `CLAUDECODE` is set, causing automatic fallback to the SDK path:
```python
claude_bin = shutil.which("claude") if not os.environ.get("CLAUDECODE") else None
```

### Key environment variables
| Variable | Purpose |
|----------|---------|
| `CLAUDECODE` | Set inside Claude Code sessions — skip CLI, use SDK fallback |
| `PORT` | Override default port 8000 |
| `ANTHROPIC_MODEL` | Override model (default: `us.anthropic.claude-sonnet-4-6-20250514`) |
| `AWS_PROFILE` | AWS profile for Bedrock (default: `bedrock`) |
| `AWS_DEFAULT_REGION` | Bedrock region (default: `us-west-2`) |
| `AZURE_AI_FOUNDRY` | Set to `1` to use Azure AI Foundry instead of Bedrock |
| `AZURE_ENDPOINT` | Azure AI Foundry endpoint URL |
| `AZURE_API_KEY` | Azure API key |
| `AZURE_API_VERSION` | Azure API version (default: `2025-04-01`) |
| `GH_TOKEN` / `GITHUB_TOKEN` | GitHub API auth (higher rate limits for cloning) |
| `SYSTEMD_EXEC_PID` | Set by systemd — disables uvicorn auto-reload |

## UI Design Decisions

- **Single process**: FastAPI serves both the HTML page and the `/api/evaluate` SSE endpoint
- **Single page app**: Form at top, streaming results appear below after submission
- **Dark, minimal style**: GitHub dark mode aesthetic (CSS vars in `static/index.html`)
- **Rendered markdown**: Claude's response is parsed with `marked.js` and syntax-highlighted with `highlight.js`
- **Session history**: Past evaluations kept in-browser during the session; displayed as a clickable sidebar list
- **Streaming via SSE**: `/api/evaluate` yields `chunk`, `status`, `error`, and `done` events; frontend appends chunks and re-renders markdown incrementally

## Conventions

- Prompt template files use the `.prmpt` extension; loaded from **both** `prompts/` in the repo root and `~/.codecheck/prompts/` (user-local). User-local templates take precedence on filename collision.
- GitHub auth state is determined by running `gh auth status`
- Repo cloning uses a `tempfile.mkdtemp` directory, cleaned up in a `finally` block after analysis
- The `_build_repo_context` function (Bedrock/Azure fallback path) caps context at 200KB and skips `.git`, `node_modules`, `__pycache__`, `venv`, `vendor`, `dist`, `build`
