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
- Show a progress bar while working
- Clone the target repo locally
- Use the textarea content as a prompt with Claude Code SDK to analyze the repo
- Display recommendations in the UI
- If `gh` CLI is authenticated, offer a button to file results as a GitHub issue

## Tech Stack

- **Language**: Python 3.12+
- **Web framework**: FastAPI (async, SSE streaming for live analysis output)
- **Prompt templates**: `.prmpt` files — first line is display name, rest is prompt body
- **Primary Claude invocation**: Claude CLI via `subprocess` (with SDK/Bedrock fallback per collect-skills.py patterns)
- **External dependencies**: `gh` CLI (GitHub auth), `claude` CLI, `git`, `anthropic` SDK (fallback)

## Invoking Claude from Python — Reference Patterns

The sibling repo `claude-skills/collect-skills.py` contains a battle-tested three-tier fallback for running Claude analysis from Python. These patterns should be considered when building the analysis backend.

### Tier 1: Claude CLI via subprocess (preferred for local dev)
```python
import subprocess, shutil
claude_bin = shutil.which("claude")
result = subprocess.run(
    [claude_bin, "-p", prompt, "--output-format", "text"],
    capture_output=True, text=True, timeout=120
)
output = result.stdout
```

### Tier 2: Anthropic Python SDK (fallback when CLI unavailable)
```python
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
msg = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=4096,
    messages=[{"role": "user", "content": prompt}]
)
output = msg.content[0].text
```
For large content (>=50KB), use streaming:
```python
with client.messages.stream(model=model, max_tokens=65536, messages=messages) as stream:
    output = stream.get_final_text()
```

### Tier 3: AWS Bedrock SDK (no API key needed, uses AWS credentials)
```python
client = anthropic.AnthropicBedrock(aws_region="us-west-2", aws_profile="bedrock")
msg = client.messages.create(
    model="us.anthropic.claude-sonnet-4-6", max_tokens=4096,
    messages=[{"role": "user", "content": prompt}]
)
```

### Self-invocation guard
Claude Code **cannot invoke itself** (nested CLI calls crash). The `CLAUDECODE` environment variable is set when running inside a Claude Code session. Always check it before attempting CLI invocation and fall back to SDK:
```python
claude_bin = shutil.which("claude") if not os.environ.get("CLAUDECODE") else None
```

### Key environment variables (from collect-skills.py)
| Variable | Purpose |
|----------|---------|
| `CLAUDECODE` | Set inside Claude Code sessions — skip CLI, use SDK |
| `ANTHROPIC_API_KEY` | Direct API authentication for Tier 2 |
| `GH_TOKEN` / `GITHUB_TOKEN` | GitHub API auth (higher rate limits) |
| `AWS_PROFILE` | AWS profile for Bedrock (default: `bedrock`) |
| `AWS_DEFAULT_REGION` | Bedrock region (default: `us-west-2`) |

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

## UI Design Decisions

- **Single process**: FastAPI serves both the HTML page and the `/api/evaluate` SSE endpoint
- **Single page app**: Form at top, streaming results appear below after submission
- **Dark, minimal style**: Dark background, clean typography, code-editor aesthetic (GitHub dark mode vibe)
- **Rendered markdown**: Claude's response is parsed as markdown with syntax-highlighted code blocks, headings, and lists
- **Session history**: Past evaluations kept in-browser during the session (lost on page refresh); displayed as a clickable list so users can revisit previous results
- **Streaming via SSE**: The `/api/evaluate` endpoint streams Claude CLI stdout line-by-line as `text/event-stream` events; the frontend appends and re-renders markdown incrementally

## Conventions

- Prompt template files use the `.prmpt` extension; loaded from **both** `prompts/` in the repo root (shipped defaults) and `~/.codecheck/prompts/` (user-local). Both locations are merged into the dropdown, with user-local templates taking precedence on name conflicts.
- GitHub auth state is determined by running `gh auth status`
- Repo cloning should be done to a temp directory and cleaned up after analysis
- See `claude-skills/collect-skills.py` for full implementation of the fallback chain, PDF extraction, GitHub API usage, and multi-file response parsing
