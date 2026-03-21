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
- Collect any `.md` files Claude created during analysis; serve them via `/api/files/{session_id}/{path}`
- Keep the session alive (repo clone + Claude context) for up to 2 hours to support follow-up questions
- If `gh` CLI is authenticated, offer a button to file results as a GitHub issue
- Reports can be shared via `/api/share` → `/share/{share_id}` (persisted in `_FILES_BASE/shares/`)

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

### Tier 1: Claude CLI via subprocess (always preferred when installed)
The app **always uses the Claude Code CLI** when the binary is found (`~/.local/bin/claude`, `~/bin/claude`, or `PATH`). `stream_claude_cli` runs it with `--output-format stream-json --verbose --dangerously-skip-permissions` and parses newline-delimited JSON. Initial evals use `--model sonnet`, follow-ups use `--model opus` with `--continue` (resumes prior CLI session). Two event types carry content:
- `assistant` events: iterate `message.content[]` for `type=="text"` blocks
- `result` events: read the top-level `result` string

```python
cmd = [claude_bin, "-p", prompt, "--model", model, "--output-format", "stream-json", "--verbose"]
proc = await asyncio.create_subprocess_exec(*cmd, cwd=repo_dir, ...)
# parse JSON lines from proc.stdout
```

### Tier 2: Anthropic SDK via AWS Bedrock or Azure AI Foundry (fallback when CLI unavailable)
`stream_sdk` in `app.py` is **only used when the Claude Code CLI is not installed**. It builds a repo context string from file contents, then streams via `AnthropicBedrock` or the Anthropic client with Azure base URL. Initial evals use Sonnet (`ANTHROPIC_DEFAULT_SONNET_MODEL`), follow-ups use Opus (`ANTHROPIC_DEFAULT_OPUS_MODEL`). Set `CLAUDE_CODE_USE_FOUNDRY=1` to use Azure instead of Bedrock.

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
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus model for follow-ups (Bedrock: `global.anthropic.claude-opus-4-6-v1`, Foundry: `claude-opus-4-6`) |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet model for initial eval (Bedrock: `global.anthropic.claude-sonnet-4-6`, Foundry: `claude-sonnet-4-6`) |
| `CLAUDE_CODE_USE_BEDROCK` | Set to `1` to use AWS Bedrock |
| `AWS_PROFILE` | AWS profile for Bedrock (default: `codecheck`) |
| `AWS_DEFAULT_REGION` | Bedrock region (default: `us-west-2`) |
| `CLAUDE_CODE_USE_FOUNDRY` | Set to `1` to use Azure AI Foundry instead of Bedrock |
| `ANTHROPIC_FOUNDRY_BASE_URL` | Azure AI Foundry endpoint URL |
| `ANTHROPIC_FOUNDRY_API_KEY` | Azure AI Foundry API key |
| `GH_TOKEN` / `GITHUB_TOKEN` | GitHub API auth (higher rate limits for cloning) |
| `SYSTEMD_EXEC_PID` | Set by systemd — disables uvicorn auto-reload |

## UI Design Decisions

- **Single process**: FastAPI serves both the HTML page and the `/api/evaluate` SSE endpoint
- **Single page app**: Form at top, streaming results appear below after submission
- **Dark, minimal style**: GitHub dark mode aesthetic (CSS vars in `static/index.html`)
- **Rendered markdown**: Claude's response is parsed with `marked.js` and syntax-highlighted with `highlight.js`
- **Session history**: Past evaluations kept in-browser during the session; displayed as a clickable sidebar list
- **Streaming via SSE**: `/api/evaluate` and `/api/followup` yield these event types:
  - `session_id` — UUID for this session (first event)
  - `status` — status message string
  - `chunk` — text fragment to append
  - `report` — final result string from CLI `result` event
  - `file` — JSON `{"name": "...", "url": "..."}` for each `.md` file Claude created
  - `error` — error message string
  - `done` — signals stream end

## Prompt preamble

Every prompt (initial and SDK follow-ups) is prefixed with `_PREAMBLE` (defined in `app.py`), which instructs Claude to:
- Only create files with `.md` extension
- Prefer folding short output into the main report rather than creating extra files
- Create multiple `.md` files only when the analysis is large and warrants distinct documents

Follow-up prompts sent via `--continue` (CLI path) do **not** get the preamble prepended, since Claude already has it in context.

## Conventions

- Prompt template files use the `.prmpt` extension; loaded from **both** `prompts/` in the repo root and `~/.codecheck/prompts/` (user-local). User-local templates take precedence on filename collision.
- GitHub auth state is determined by running `gh auth status`
- Repo cloning uses a `tempfile.mkdtemp` directory, cleaned up in a `finally` block after analysis
- The `_build_repo_context` function (Bedrock/Azure fallback path) caps context at 200KB and skips `.git`, `node_modules`, `__pycache__`, `venv`, `vendor`, `dist`, `build`
