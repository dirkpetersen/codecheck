#!/usr/bin/env python3
"""codecheck - A code review web app powered by Claude Code."""

import asyncio
import html
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
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

# Persistent directory for Claude-created files (survives session expiry, cleared after 30 days or on /tmp wipe)
_FILES_BASE = Path(tempfile.gettempdir()) / "codecheck_files"
_FILES_BASE.mkdir(exist_ok=True)

# In-memory set of already-known files per session to avoid re-emitting: session_id -> set of rel paths
_known_files: dict[str, set[str]] = {}

# Active sessions kept alive for follow-up questions: session_id -> {tmp_dir, repo_dir, created}
_sessions: dict[str, dict] = {}


def _cleanup_sessions(max_age_secs: int = 7200):
    """Remove repo clone sessions older than max_age_secs (default 2 hours)."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.get("created", 0) > max_age_secs]
    for sid in expired:
        tmp = _sessions.pop(sid, {}).get("tmp_dir")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        _known_files.pop(sid, None)


def _cleanup_file_dirs(max_age_days: int = 30):
    """Remove persistent file dirs older than max_age_days."""
    cutoff = time.time() - max_age_days * 86400
    for d in _FILES_BASE.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)

REPO_ROOT = Path(__file__).parent
PROMPTS_DIR = REPO_ROOT / "prompts"
USER_PROMPTS_DIR = Path.home() / ".codecheck" / "prompts"

try:
    _GIT_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
except Exception:
    _GIT_COMMIT = ""

# Prepended to every prompt so Claude knows the output rules before it starts working.
_PREAMBLE = """\
OVERRIDING INSTRUCTIONS (take priority over everything else):
- If you create any files during this analysis, use ONLY Markdown format with a .md extension.
  Do NOT create .txt, .rst, .html, or any other file type — Markdown only.
- Be thoughtful about when to create additional files vs. keeping everything in your main report:
  - If the analysis is short or produces only one supplementary document, fold it into the main
    report rather than creating a separate file.
  - Create separate .md files only when the analysis is detailed and there are multiple distinct
    documents (e.g. per-component reports, code examples, migration guides) that would make the
    main report unwieldy.
- These instructions override any conflicting guidance in the prompt below.

---

"""


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


async def stream_claude_cli(claude_bin: str, prompt: str, repo_dir: str,
                            continue_conversation: bool = False):
    """Run claude CLI in batch mode, streaming output via stream-json format."""
    cmd = [claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose"]
    if continue_conversation:
        cmd.insert(1, "--continue")
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
                        elif block.get("type") == "tool_use":
                            tool = block.get("name", "")
                            inp = block.get("input", {})
                            detail = (
                                inp.get("file_path")
                                or inp.get("command")
                                or inp.get("pattern")
                                or inp.get("query")
                                or ""
                            )
                            label = f"**[{tool}]** {detail}\n" if detail else f"**[{tool}]**\n"
                            yield _sse_event("chunk", label)
                elif etype == "user":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") != "tool_result":
                            continue
                        raw = block.get("content", "") or block.get("output", "")
                        if isinstance(raw, list):
                            raw = "\n".join(
                                b.get("text", "") for b in raw if b.get("type") == "text"
                            )
                        if raw and isinstance(raw, str):
                            lines = raw.strip().splitlines()
                            preview = lines[0][:160] if lines else ""
                            suffix = f" _…({len(lines)} lines)_" if len(lines) > 1 else ""
                            yield _sse_event("chunk", f"  ↳ {preview}{suffix}\n")
                elif etype == "result":
                    result_text = event.get("result", "")
                    if result_text and isinstance(result_text, str):
                        yield _sse_event("report", result_text)
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


async def stream_sdk(prompt: str, repo_dir: str):
    """Fallback when Claude CLI is unavailable: use Anthropic SDK via Bedrock or Azure."""
    try:
        import anthropic
    except ImportError:
        yield _sse_event("error", "anthropic package not installed. Run: pip install anthropic")
        return

    context = _build_repo_context(repo_dir)
    full_prompt = f"{prompt}\n\n---\n\nRepository contents:\n\n{context}"
    model = os.environ.get("ANTHROPIC_MODEL", "us.anthropic.claude-sonnet-4-6-20250514")

    try:
        if os.environ.get("AZURE_AI_FOUNDRY"):
            # Azure AI Foundry: Anthropic-compatible endpoint
            base_url = os.environ.get("AZURE_ENDPOINT", "").rstrip("/")
            api_version = os.environ.get("AZURE_API_VERSION", "2025-04-01")
            client = anthropic.Anthropic(
                base_url=f"{base_url}/anthropic/v1/",
                api_key=os.environ.get("AZURE_API_KEY", ""),
                default_headers={"api-version": api_version},
            )
        else:
            client = anthropic.AnthropicBedrock(
                aws_profile=os.environ.get("AWS_PROFILE", "bedrock"),
                aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            )

        with client.messages.stream(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": full_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield _sse_event("chunk", text)
        yield _sse_event("done", "")
    except Exception as e:
        backend = "Azure" if os.environ.get("AZURE_AI_FOUNDRY") else "Bedrock"
        yield _sse_event("error", f"{backend} error: {e}")


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


def _collect_output_files(repo_dir: str, session_id: str) -> dict[str, str]:
    """Find new .md files Claude created (not in original git tree, not already known).
    Copies them to _FILES_BASE/session_id/ for persistent serving."""
    repo_path = Path(repo_dir)
    try:
        r = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=repo_dir, timeout=10)
        tracked = set(r.stdout.strip().splitlines()) if r.returncode == 0 else set()
    except Exception:
        tracked = set()

    already_known = _known_files.get(session_id, set())

    new_files: dict[str, str] = {}
    for f in sorted(repo_path.rglob("*.md")):
        if not f.is_file():
            continue
        rel = f.relative_to(repo_path)
        if ".git" in rel.parts:
            continue
        rel_str = str(rel)
        if rel_str in tracked or rel_str in already_known:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            new_files[rel_str] = content
            # Write to persistent files dir
            dest = _FILES_BASE / session_id / rel_str
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        except Exception:
            pass

    if new_files:
        _known_files.setdefault(session_id, set()).update(new_files)
    return new_files


_FILE_VIEWER = """\
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PLACEHOLDER_FILENAME</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--surface:#161b22;--inset:#0d1117;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#E8520A;
  --font:'DM Sans',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;--r:8px}
html{font-size:15px;-webkit-font-smoothing:antialiased}
body{background:var(--bg);color:var(--text);font-family:var(--font);line-height:1.6}
.topbar{padding:14px 32px;border-bottom:1px solid var(--border);background:var(--surface);
  font-family:var(--mono);font-size:.82rem;color:var(--muted)}
.topbar strong{color:var(--text)}
.wrap{max-width:860px;margin:0 auto;padding:40px 32px 80px}
.md h1,.md h2,.md h3,.md h4{margin-top:1.4em;margin-bottom:.5em;font-weight:600;
  border-bottom:1px solid var(--border);padding-bottom:6px}
.md h1{font-size:1.5rem;border:none}.md h2{font-size:1.2rem}
.md h3{font-size:1.05rem;border:none}.md h4{font-size:.95rem;border:none}
.md p{margin:.6em 0}.md ul,.md ol{margin:.6em 0;padding-left:1.6em}.md li{margin:.25em 0}
.md code{font-family:var(--mono);font-size:.82em;background:rgba(110,118,129,.15);
  padding:2px 6px;border-radius:4px;color:#c9d1d9}
.md pre{margin:.8em 0;border-radius:var(--r);border:1px solid var(--border);overflow-x:auto}
.md pre code{display:block;padding:16px;background:var(--inset);font-size:.8rem;
  line-height:1.55;border-radius:0}
.md blockquote{border-left:3px solid var(--accent);padding:.4em 1em;margin:.8em 0;
  color:var(--muted);background:rgba(232,82,10,.06);border-radius:0 var(--r) var(--r) 0}
.md table{border-collapse:collapse;width:100%;margin:.8em 0;font-size:.85rem}
.md th,.md td{border:1px solid var(--border);padding:8px 12px;text-align:left}
.md th{background:#1c2129;font-weight:600}
.md a{color:var(--accent);text-decoration:none}.md a:hover{text-decoration:underline}
.md hr{border:none;border-top:1px solid var(--border);margin:1.5em 0}
.md strong{color:#fff}.md{font-size:1rem;line-height:1.8}
</style></head><body>
<div class="topbar"><strong>PLACEHOLDER_FILENAME</strong></div>
<div class="wrap"><div class="md" id="out"></div>
<div style="display:flex;justify-content:flex-end;margin-top:20px;padding-top:16px;border-top:1px solid var(--border)">
<button id="cpBtn" onclick="navigator.clipboard.writeText(md).then(()=>{this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy markdown'},1500)})" style="display:inline-flex;align-items:center;gap:5px;background:transparent;border:1px solid var(--border);color:var(--muted);font-family:var(--font);font-size:.75rem;padding:5px 14px;border-radius:var(--r);cursor:pointer">Copy markdown</button>
</div></div>
<script>
const md=PLACEHOLDER_CONTENT_JSON;
document.getElementById('out').innerHTML=marked.parse(md);
document.querySelectorAll('pre code:not(.hljs)').forEach(el=>hljs.highlightElement(el));
</script></body></html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/version")
async def get_version():
    return JSONResponse({"commit": _GIT_COMMIT})


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
        _cleanup_sessions()
        _cleanup_file_dirs()

        session_id = str(uuid.uuid4())
        yield _sse_event("session_id", session_id)
        yield _sse_event("status", "Cloning repository...")

        tmp_dir = tempfile.mkdtemp(prefix="codecheck_")
        keep_session = False
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
            yield _sse_event("status", "Claude Code is analyzing...")

            full_prompt = _PREAMBLE + prompt
            claude_bin = get_claude_bin()
            if claude_bin:
                async for event in stream_claude_cli(claude_bin, full_prompt, repo_dir):
                    yield event
            else:
                async for event in stream_sdk(full_prompt, repo_dir):
                    yield event

            # Collect any .md files Claude wrote during analysis
            output_files = _collect_output_files(repo_dir, session_id)
            for fname in sorted(output_files):
                url = f"/api/files/{session_id}/{fname}"
                yield _sse_event("file", json.dumps({"name": fname, "url": url}))

            # Keep session alive for follow-up questions
            _sessions[session_id] = {
                "tmp_dir": tmp_dir, "repo_dir": repo_dir, "created": time.time(),
            }
            keep_session = True

        finally:
            if not keep_session:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/followup")
async def followup(request: Request):
    """Ask follow-up questions using the same Claude session and repo."""
    body = await request.json()
    session_id = body.get("session_id", "")
    prompt = body.get("prompt", "").strip()

    if not session_id or not prompt:
        return JSONResponse({"error": "session_id and prompt are required"}, status_code=400)

    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session expired — please run a new evaluation"}, status_code=404)

    async def generate():
        repo_dir = session["repo_dir"]
        yield _sse_event("status", "Continuing analysis...")

        full_prompt = _PREAMBLE + prompt
        claude_bin = get_claude_bin()
        if claude_bin:
            async for event in stream_claude_cli(claude_bin, full_prompt, repo_dir,
                                                 continue_conversation=True):
                yield event
        else:
            async for event in stream_sdk(full_prompt, repo_dir):
                yield event

        # Collect any new .md files from this follow-up
        output_files = _collect_output_files(repo_dir, session_id)
        for fname in sorted(output_files):
            url = f"/api/files/{session_id}/{fname}"
            yield _sse_event("file", json.dumps({"name": fname, "url": url}))

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


@app.get("/api/files/{session_id}/{filename:path}", response_class=HTMLResponse)
async def get_file(session_id: str, filename: str):
    """Serve a Claude-created markdown file as a rendered HTML page."""
    # Resolve and confirm the path stays within the expected session directory (prevent path traversal)
    session_dir = (_FILES_BASE / session_id).resolve()
    file_path = (session_dir / filename).resolve()
    if not str(file_path).startswith(str(session_dir) + os.sep):
        return HTMLResponse("<h2>Invalid path.</h2>", status_code=400)
    if not file_path.is_file():
        return HTMLResponse("<h2>File not found or session expired.</h2>", status_code=404)
    content = file_path.read_text(encoding="utf-8", errors="replace")
    safe_filename = html.escape(filename)
    page = (_FILE_VIEWER
            .replace("PLACEHOLDER_FILENAME", safe_filename)
            .replace("PLACEHOLDER_CONTENT_JSON", json.dumps(content)))
    return HTMLResponse(page)


@app.post("/api/share")
async def share_report(request: Request):
    """Save a report and return a shareable URL."""
    body = await request.json()
    content = body.get("content", "").strip()
    repo_name = body.get("repo_name", "report")
    if not content:
        return JSONResponse({"error": "No content to share"}, status_code=400)

    share_id = uuid.uuid4().hex[:12]
    shares_dir = _FILES_BASE / "shares"
    shares_dir.mkdir(exist_ok=True)
    share_path = shares_dir / f"{share_id}.md"
    share_path.write_text(content, encoding="utf-8")
    # Store repo name as metadata
    (shares_dir / f"{share_id}.meta").write_text(repo_name, encoding="utf-8")
    return JSONResponse({"id": share_id, "url": f"/share/{share_id}"})


@app.get("/share/{share_id}", response_class=HTMLResponse)
async def get_share(share_id: str):
    """Serve a shared report as a rendered HTML page."""
    # Validate share_id format (hex only, prevent traversal)
    if not share_id.isalnum() or len(share_id) > 20:
        return HTMLResponse("<h2>Invalid share link.</h2>", status_code=400)
    share_path = _FILES_BASE / "shares" / f"{share_id}.md"
    if not share_path.is_file():
        return HTMLResponse("<h2>Shared report not found or expired.</h2>", status_code=404)
    content = share_path.read_text(encoding="utf-8", errors="replace")
    meta_path = _FILES_BASE / "shares" / f"{share_id}.meta"
    repo_name = meta_path.read_text(encoding="utf-8").strip() if meta_path.is_file() else "Report"
    safe_title = html.escape(f"codecheck — {repo_name}")
    page = (_FILE_VIEWER
            .replace("PLACEHOLDER_FILENAME", safe_title)
            .replace("PLACEHOLDER_CONTENT_JSON", json.dumps(content)))
    return HTMLResponse(page)


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
