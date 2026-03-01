#!/usr/bin/env python3
"""codecheck - A code review web app powered by Claude Code."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="codecheck")

REPO_ROOT = Path(__file__).parent
PROMPTS_DIR = REPO_ROOT / "prompts"
USER_PROMPTS_DIR = Path.home() / ".codecheck" / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts() -> list[dict]:
    """Load .prmpt files from both repo and user directories.

    Each .prmpt file: first line = display name, rest = prompt body.
    User-local templates override repo defaults on filename collision.
    """
    templates: dict[str, dict] = {}

    for directory in [PROMPTS_DIR, USER_PROMPTS_DIR]:
        if not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.prmpt")):
            text = f.read_text(encoding="utf-8")
            lines = text.strip().splitlines()
            if not lines:
                continue
            name = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            templates[f.stem] = {
                "id": f.stem,
                "name": name,
                "body": body,
                "source": "user" if directory == USER_PROMPTS_DIR else "builtin",
            }

    return list(templates.values())


def resolve_repo_url(raw: str) -> str:
    """Normalize user input to a cloneable git URL.

    Accepts 'user/repo', 'github.com/user/repo', or full https URLs.
    """
    raw = raw.strip().rstrip("/")
    if raw.startswith("git@") or raw.startswith("https://") or raw.startswith("http://"):
        return raw
    if raw.startswith("github.com/"):
        return f"https://{raw}.git"
    # Assume user/repo shorthand
    if "/" in raw and not raw.startswith("/"):
        return f"https://github.com/{raw}.git"
    return raw


def get_claude_bin() -> str | None:
    """Return path to claude CLI, or None if not found or inside Claude Code."""
    if os.environ.get("CLAUDECODE"):
        return None
    found = shutil.which("claude")
    if found:
        return found
    # ~/bin/claude not always in systemd/service PATH
    candidate = Path.home() / "bin" / "claude"
    if candidate.is_file():
        return str(candidate)
    return None


def check_gh_auth() -> bool:
    """Return True if gh CLI is installed and authenticated."""
    gh = shutil.which("gh")
    if not gh:
        return False
    try:
        result = subprocess.run(
            [gh, "auth", "status"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


async def stream_claude_cli(claude_bin: str, prompt: str, repo_dir: str):
    """Run claude CLI in batch mode, streaming output via stream-json format."""
    cmd = [claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose"]
    # Claude CLI requires both ~/bin and ~/.local/bin in PATH on startup
    env = os.environ.copy()
    home = Path.home()
    path_parts = env.get("PATH", "").split(":")
    prepend = [str(home / "bin"), str(home / ".local" / "bin")]
    extra = [p for p in prepend if p not in path_parts]
    if extra:
        env["PATH"] = ":".join(extra) + ":" + env.get("PATH", "")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=repo_dir,
        env=env,
    )

    line_buf = ""
    try:
        while True:
            chunk = await asyncio.wait_for(proc.stdout.read(256), timeout=300)
            if not chunk:
                break
            line_buf += chunk.decode("utf-8", errors="replace")
            while "\n" in line_buf:
                line, line_buf = line_buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            yield _sse_event("chunk", block["text"])
    except asyncio.TimeoutError:
        proc.kill()
        yield _sse_event("error", "Claude CLI timed out after 5 minutes.")
        return

    await proc.wait()

    if proc.returncode != 0:
        stderr = (await proc.stderr.read()).decode("utf-8", errors="replace")
        yield _sse_event("error", f"Claude CLI exited with code {proc.returncode}: {stderr[:500]}")
    else:
        yield _sse_event("done", "")


async def stream_bedrock_sdk(prompt: str, repo_dir: str):
    """Fallback when Claude CLI is unavailable: use Anthropic SDK via AWS Bedrock."""
    try:
        import anthropic
    except ImportError:
        yield _sse_event("error", "anthropic package not installed. Run: pip install anthropic")
        return

    context = _build_repo_context(repo_dir)
    full_prompt = f"{prompt}\n\n---\n\nRepository contents:\n\n{context}"

    try:
        client = anthropic.AnthropicBedrock(
            aws_profile="bedrock",
            aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        )
        model = os.environ.get("ANTHROPIC_MODEL", "global.anthropic.claude-sonnet-4-6")
        with client.messages.stream(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": full_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield _sse_event("chunk", text)
        yield _sse_event("done", "")
    except Exception as e:
        yield _sse_event("error", f"Bedrock error: {e}")


def _build_repo_context(repo_dir: str, max_bytes: int = 200_000) -> str:
    """Read text files from repo into a single context string."""
    extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".c", ".cpp", ".h",
        ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".bash", ".zsh",
        ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".md", ".txt",
        ".html", ".css", ".sql", ".r", ".R", ".jl", ".cu", ".cuh",
        ".cmake", ".makefile", ".dockerfile",
    }
    parts = []
    total = 0
    repo_path = Path(repo_dir)

    for f in sorted(repo_path.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in extensions and f.name.lower() not in {"makefile", "dockerfile", "cmakelists.txt"}:
            continue
        # Skip common non-essential dirs
        rel = f.relative_to(repo_path)
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "vendor", "dist", "build"}
        if any(part in skip_dirs for part in rel.parts):
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        header = f"### {rel}\n```{f.suffix.lstrip('.')}\n"
        footer = "\n```\n\n"
        entry = header + content + footer
        if total + len(entry) > max_bytes:
            parts.append(f"\n... (truncated, {max_bytes // 1000}KB limit reached)\n")
            break
        parts.append(entry)
        total += len(entry)

    return "".join(parts) if parts else "(empty repository)"


def _sse_event(event: str, data: str) -> str:
    """Format a server-sent event."""
    # Escape newlines in data for SSE protocol
    escaped = data.replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {escaped}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/prompts")
async def get_prompts():
    """Return available prompt templates."""
    return JSONResponse(load_prompts())


@app.get("/api/gh-auth")
async def get_gh_auth():
    """Check if gh CLI is authenticated."""
    return JSONResponse({"authenticated": check_gh_auth()})


@app.post("/api/evaluate")
async def evaluate(request: Request):
    """Clone repo, run Claude analysis, stream results via SSE."""
    body = await request.json()
    repo_url = resolve_repo_url(body.get("repo_url", ""))
    prompt = body.get("prompt", "").strip()

    if not repo_url or not prompt:
        return JSONResponse({"error": "repo_url and prompt are required"}, status_code=400)

    async def generate():
        yield _sse_event("status", "Cloning repository...")

        tmp_dir = tempfile.mkdtemp(prefix="codecheck_")
        try:
            # Clone the repo
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", repo_url, tmp_dir + "/repo",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

            if proc.returncode != 0:
                stderr = (await proc.stderr.read()).decode("utf-8", errors="replace")
                yield _sse_event("error", f"Git clone failed: {stderr[:500]}")
                return

            repo_dir = tmp_dir + "/repo"
            yield _sse_event("status", "Analyzing with Claude Code...")

            claude_bin = get_claude_bin()
            if claude_bin:
                async for event in stream_claude_cli(claude_bin, prompt, repo_dir):
                    yield event
            else:
                async for event in stream_bedrock_sdk(prompt, repo_dir):
                    yield event

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/file-issue")
async def file_issue(request: Request):
    """File analysis results as a GitHub issue."""
    body = await request.json()
    repo_url = body.get("repo_url", "")
    title = body.get("title", "Code Review - codecheck")
    content = body.get("content", "")

    if not repo_url or not content:
        return JSONResponse({"error": "repo_url and content required"}, status_code=400)

    # Extract owner/repo from URL
    raw = repo_url.strip().rstrip("/").removesuffix(".git")
    parts = raw.split("/")
    if len(parts) >= 2:
        owner_repo = f"{parts[-2]}/{parts[-1]}"
    else:
        return JSONResponse({"error": "Cannot parse repo owner/name from URL"}, status_code=400)

    gh = shutil.which("gh")
    if not gh:
        return JSONResponse({"error": "gh CLI not installed"}, status_code=500)

    try:
        result = subprocess.run(
            [gh, "issue", "create", "--repo", owner_repo, "--title", title, "--body", content],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            issue_url = result.stdout.strip()
            return JSONResponse({"url": issue_url})
        else:
            return JSONResponse({"error": result.stderr.strip()}, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "gh issue create timed out"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page application."""
    html_path = REPO_ROOT / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    reload = not os.environ.get("SYSTEMD_EXEC_PID")  # disable reload under systemd
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
