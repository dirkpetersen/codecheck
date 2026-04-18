# codecheck

A web application that reviews GitHub repositories using **Claude Code**. Paste a repo URL, pick a review prompt, and get streaming code analysis with actionable recommendations.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.default .env        # edit as needed
python app.py
# Open http://localhost:8000
```

<img width="956" height="761" alt="codecheck screenshot" src="https://github.com/user-attachments/assets/c82e2ac3-cec9-4d09-bfe3-aec2c862e28b" />

## How It Works

1. Enter a GitHub repository URL (e.g. `user/repo` or `https://github.com/user/repo`)
2. Select a prompt template from the dropdown, or write your own
3. Click **Evaluate Now** — the repo is cloned and analyzed by Claude Code in real time
4. Watch the live terminal chatter (tool calls, file reads, commands) as Claude Code works
5. When done, the final report appears below with rendered markdown and syntax highlighting
6. Ask follow-up questions — Claude Code continues in the same session with full context
7. Optionally file the results as a GitHub issue (requires `gh` CLI authenticated)

## Features

- **Live streaming** — SSE-based real-time output with a terminal-like chatter box showing Claude Code's tool calls
- **Follow-up questions** — continue the conversation in the same Claude Code session (`--continue`)
- **Persistent history** — past evaluations stored in the browser for 30 days, deduplicated per repo
- **Generated files** — markdown files Claude Code creates during analysis are linked as chips, persisted under `/tmp` for 30 days
- **Copy markdown** — one-click copy of the full report or individual generated files
- **File as GitHub Issue** — post results directly to the analyzed repo's issue tracker
- **Dark UI** — minimal single-page app with OSU Beaver Orange accent

## Claude Code Backends

The app requires **both** the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/getting-started) **and** an authentication method. If the CLI is not installed or no credentials are configured, the app will not function. Initial evaluations use Sonnet; follow-up questions use Opus.

The app checks for the `claude` binary in `~/.local/bin/claude`, `~/bin/claude`, then `PATH`. When found, it runs Claude Code via subprocess with `--output-format stream-json --verbose` — this gives the richest experience with tool calls, file operations, and the full agent loop streamed live.

You must configure **one** of the following authentication methods:

### Option 1: Anthropic API Key

The simplest setup — just set your API key:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

### Option 2: AWS Bedrock

Uses the Anthropic SDK via AWS Bedrock for authentication.

First configure your AWS credentials:

```bash
aws --profile codecheck configure
# Enter your AWS Access Key ID, Secret Access Key, and region (us-west-2) when prompted
```

Then set in `.env`:

```env
CLAUDE_CODE_USE_BEDROCK=1
AWS_PROFILE=codecheck
AWS_DEFAULT_REGION=us-west-2
ANTHROPIC_DEFAULT_OPUS_MODEL=global.anthropic.claude-opus-4-7
ANTHROPIC_DEFAULT_SONNET_MODEL=global.anthropic.claude-sonnet-4-6
```

### Option 3: Azure AI Foundry

Uses the Anthropic SDK via Microsoft Azure AI Foundry for authentication.

```env
CLAUDE_CODE_USE_FOUNDRY=1
ANTHROPIC_FOUNDRY_BASE_URL=https://<resource>.services.ai.azure.com
ANTHROPIC_FOUNDRY_API_KEY=your-azure-api-key
ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-7
ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6
```

See [Anthropic on Azure docs](https://docs.anthropic.com/en/docs/build-with-claude/azure).

### SDK Fallback (no CLI)

If the Claude Code CLI is **not installed**, the app falls back to using the Anthropic SDK directly (requires Option 2 or 3 above plus `pip install anthropic[bedrock]` or `pip install anthropic`). This path builds a text context from the repo's files and streams the response, but doesn't have Claude Code's tool-use capabilities.

## Prompt Templates

Templates are `.prmpt` files where the first line is the display name and the rest is the prompt body. They are loaded from two locations (merged, user-local wins on name conflicts):

- `prompts/` — shipped defaults (code quality, multi-GPU, security)
- `~/.codecheck/prompts/` — your own custom templates

## Configuration

Copy `.env.default` to `.env` and adjust:

```bash
cp .env.default .env
```

All settings with defaults are documented in `.env.default`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `CLAUDECODE` | — | Set to skip CLI (auto-set inside Claude Code sessions) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (Option 1) |
| `CLAUDE_CODE_USE_BEDROCK` | — | Set to `1` to use AWS Bedrock (Option 2) |
| `AWS_PROFILE` | `codecheck` | AWS profile (set up with `aws --profile codecheck configure`) |
| `AWS_DEFAULT_REGION` | `us-west-2` | AWS region for Bedrock |
| `CLAUDE_CODE_USE_FOUNDRY` | — | Set to `1` to use Azure AI Foundry (Option 3) |
| `ANTHROPIC_FOUNDRY_BASE_URL` | — | Azure AI Foundry endpoint URL |
| `ANTHROPIC_FOUNDRY_API_KEY` | — | Azure AI Foundry API key |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | per-backend | Opus model for follow-ups (SDK fallback only) |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | per-backend | Sonnet model for initial eval (SDK fallback only) |
| `GH_TOKEN` / `GITHUB_TOKEN` | — | GitHub token for higher clone rate limits |

## Deployment

The app is designed for [appmotel](https://github.com/dirkpetersen/appmotel) (systemd + Traefik PaaS):

```bash
sudo -u appmotel appmo add codecheck dirkpetersen/codecheck main
sudo -u appmotel appmo env codecheck   # set env vars
sudo -u appmotel appmo restart codecheck
```

The `PORT` env var is set automatically by appmotel. Uvicorn auto-reload is disabled when running under systemd (`SYSTEMD_EXEC_PID` is set).

## Tech Stack

- **Python 3.12+** / **FastAPI** with async SSE streaming
- **Claude Code CLI** (`claude --output-format stream-json --verbose`) or **Anthropic SDK** (Bedrock / Azure)
- **marked.js** + **highlight.js** for client-side markdown rendering
- Dark minimal single-page UI with localStorage-based history

## License

[MIT](LICENSE)
