"""
GitHub-connector plugin for Opti-Oignon.

First external service connector -- demonstrates network plugins
with authentication. Provides slash commands for GitHub API
interaction and auto-enrichment of GitHub references in responses.

Commands:
    /gh auth <token>         Store GitHub PAT (validated via /user)
    /gh auth status          Show auth status (username, scopes)
    /gh auth revoke          Remove stored token
    /gh issues [owner/repo]  List open issues
    /gh pr list [owner/repo] List open pull requests
    /gh repo info <o/r>      Repository details
    /gh search <query>       Search repositories
    /gh gist create <desc>   Create gist from last code block

All API calls use urllib.request (stdlib). Token stored in plugin
directory SQLite (never echoed in responses).
"""

import json
import logging
import re
import sqlite3

# S136 audit fix
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

__plugin_name__: str = "github-connector"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_MAX_RESULTS = 10
_GITHUB_API_BASE = "https://api.github.com"
_USER_AGENT = "Opti-Oignon-GitHub-Plugin/1.0"

# =========================================================================
# GitHub API layer
# =========================================================================


def _github_api(
    method: str,
    path: str,
    token: str,
    body: dict | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make a GitHub API request.

    All API interactions go through this single helper for consistency
    and error handling.

    Parameters
    ----------
    method : str
        HTTP method (GET, POST, PATCH, DELETE).
    path : str
        API path (e.g. '/user', '/repos/owner/repo/issues').
    token : str
        GitHub Personal Access Token.
    body : dict, optional
        JSON body for POST/PATCH requests.
    params : dict, optional
        Query parameters appended to the URL.

    Returns
    -------
    dict
        Parsed JSON response with added '_status' and '_rate_limit' keys.
    """
    url = f"{_GITHUB_API_BASE}{path}"

    if params:
        query_parts = [f"{k}={urllib.request.quote(str(v))}" for k, v in params.items()]
        url += "?" + "&".join(query_parts)

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": _USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"token {token}"

    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = resp.read().decode("utf-8")
            parsed = json.loads(resp_body) if resp_body else {}

            # Wrap list responses
            if isinstance(parsed, list):
                parsed = {"items": parsed}

            parsed["_status"] = resp.status
            parsed["_rate_limit"] = {
                "remaining": resp.headers.get("X-RateLimit-Remaining", "?"),
                "limit": resp.headers.get("X-RateLimit-Limit", "?"),
                "reset": resp.headers.get("X-RateLimit-Reset", "?"),
            }
            return parsed

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
            error_data = json.loads(error_body)
            message = error_data.get("message", error_body[:200])
        except Exception:
            message = error_body[:200] if error_body else str(exc)

        return {
            "_status": exc.code,
            "_error": message,
            "_rate_limit": {
                "remaining": exc.headers.get("X-RateLimit-Remaining", "?") if exc.headers else "?",
                "limit": exc.headers.get("X-RateLimit-Limit", "?") if exc.headers else "?",
                "reset": exc.headers.get("X-RateLimit-Reset", "?") if exc.headers else "?",
            },
        }

    except urllib.error.URLError as exc:
        return {
            "_status": 0,
            "_error": f"Network error: {exc.reason}",
            "_rate_limit": {"remaining": "?", "limit": "?", "reset": "?"},
        }

    except Exception as exc:
        return {
            "_status": 0,
            "_error": f"Request failed: {exc}",
            "_rate_limit": {"remaining": "?", "limit": "?", "reset": "?"},
        }


def _format_rate_limit(rate_info: dict, show: bool) -> str:
    """Format rate limit info string."""
    if not show:
        return ""
    remaining = rate_info.get("remaining", "?")
    limit = rate_info.get("limit", "?")
    if remaining == "?" or limit == "?":
        return ""
    try:
        rem = int(remaining)
        if rem < 10:
            return f"\n\n**Warning:** GitHub API rate limit low ({remaining}/{limit} remaining)"
        return f"\n\n*Rate limit: {remaining}/{limit}*"
    except (ValueError, TypeError):
        return ""


# =========================================================================
# Token storage (SQLite)
# =========================================================================


class TokenStore:
    """SQLite-backed GitHub token storage.

    Token stored in plugin directory. Never echoed in responses.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _safe_connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                token TEXT NOT NULL,
                username TEXT NOT NULL DEFAULT '',
                scopes TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        conn.commit()

    def store_token(self, token: str, username: str, scopes: str) -> None:
        """Store or replace the GitHub token."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO auth (id, token, username, scopes, created_at) "
            "VALUES (1, ?, ?, ?, ?)",
            (token, username, scopes, time.time()),
        )
        conn.commit()

    def get_token(self) -> str | None:
        """Retrieve the stored token, or None."""
        conn = self._get_conn()
        row = conn.execute("SELECT token FROM auth WHERE id = 1").fetchone()
        return row[0] if row else None

    def get_auth_info(self) -> dict[str, Any] | None:
        """Retrieve auth info (username, scopes, created_at)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT username, scopes, created_at FROM auth WHERE id = 1"
        ).fetchone()
        if not row:
            return None
        return {
            "username": row[0],
            "scopes": row[1],
            "created_at": row[2],
        }

    def revoke_token(self) -> bool:
        """Hard delete the stored token."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM auth WHERE id = 1")
        conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# =========================================================================
# Module-level state
# =========================================================================

_token_store: TokenStore | None = None


def _get_store(ctx: Any) -> TokenStore:
    """Get or create the token store."""
    global _token_store
    if _token_store is not None:
        return _token_store

    plugin_dir = ""
    if hasattr(ctx, "metadata"):
        plugin_dir = ctx.metadata.get("plugin_dir", "")

    if plugin_dir:
        db_path = Path(plugin_dir) / "github_auth.db"
    else:
        import tempfile
        db_path = Path(tempfile.gettempdir()) / "opti_github_auth.db"

    _token_store = TokenStore(db_path)
    return _token_store


def _require_token(ctx: Any) -> tuple[str | None, dict | None]:
    """Get token or return error response."""
    store = _get_store(ctx)
    token = store.get_token()
    if not token:
        return None, {
            "response": (
                "No GitHub token configured. "
                "Use `/gh auth <your-token>` to authenticate."
            ),
            "handled": True,
        }
    return token, None


# =========================================================================
# Command handlers
# =========================================================================


def _parse_owner_repo(arg: str, config: dict) -> str | None:
    """Parse owner/repo from argument or fall back to config default."""
    arg = arg.strip()
    if arg and "/" in arg:
        return arg
    default = config.get("default_repo", "")
    if default and "/" in default:
        return default
    return None


def _cmd_auth(args: str, ctx: Any) -> dict[str, Any]:
    """Handle /gh auth <token>, /gh auth status, /gh auth revoke."""
    args = args.strip()

    if args == "status":
        store = _get_store(ctx)
        info = store.get_auth_info()
        if not info:
            return {
                "response": "Not authenticated. Use `/gh auth <token>` to connect.",
                "handled": True,
            }
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(info["created_at"]))
        scopes = info["scopes"] or "none detected"
        return {
            "response": (
                f"**GitHub Auth Status**\n"
                f"- User: {info['username']}\n"
                f"- Scopes: {scopes}\n"
                f"- Connected: {created}"
            ),
            "handled": True,
        }

    if args == "revoke":
        store = _get_store(ctx)
        revoked = store.revoke_token()
        if revoked:
            return {"response": "GitHub token revoked.", "handled": True}
        return {"response": "No token to revoke.", "handled": True}

    # /gh auth <token>
    token = args
    if not token or len(token) < 10:
        return {
            "response": "Invalid token. Usage: `/gh auth <your-github-pat>`",
            "handled": True,
        }

    # Validate token via /user endpoint
    result = _github_api("GET", "/user", token)
    if "_error" in result:
        return {
            "response": f"Token validation failed: {result['_error']}",
            "handled": True,
        }

    username = result.get("login", "unknown")
    scopes = result.get("_rate_limit", {}).get("scopes", "")

    store = _get_store(ctx)
    store.store_token(token, username, scopes)

    return {
        "response": f"Authenticated as **{username}**.",
        "handled": True,
    }


def _cmd_issues(args: str, ctx: Any, config: dict) -> dict[str, Any]:
    """Handle /gh issues [owner/repo]."""
    token, error = _require_token(ctx)
    if error:
        return error

    repo = _parse_owner_repo(args, config)
    if not repo:
        return {
            "response": "Usage: `/gh issues owner/repo` or set a default_repo in config.",
            "handled": True,
        }

    max_results = config.get("max_results", _DEFAULT_MAX_RESULTS)
    show_rate = config.get("show_rate_limit", False)

    result = _github_api(
        "GET", f"/repos/{repo}/issues", token,
        params={"state": "open", "per_page": str(max_results)},
    )

    if "_error" in result:
        return {
            "response": f"Error fetching issues: {result['_error']}",
            "handled": True,
        }

    issues = result.get("items", [])
    if not issues:
        msg = f"No open issues in **{repo}**."
    else:
        lines = [f"**Open issues in {repo}** ({len(issues)}):"]
        for issue in issues:
            # Filter out pull requests (GitHub API returns PRs as issues too)
            if "pull_request" in issue:
                continue
            number = issue.get("number", "?")
            title = issue.get("title", "Untitled")
            labels = ", ".join(l.get("name", "") for l in issue.get("labels", []))
            label_str = f" [{labels}]" if labels else ""
            lines.append(f"- #{number}: {title}{label_str}")
        msg = "\n".join(lines)

    msg += _format_rate_limit(result.get("_rate_limit", {}), show_rate)
    return {"response": msg, "handled": True}


def _cmd_pr_list(args: str, ctx: Any, config: dict) -> dict[str, Any]:
    """Handle /gh pr list [owner/repo]."""
    token, error = _require_token(ctx)
    if error:
        return error

    repo = _parse_owner_repo(args, config)
    if not repo:
        return {
            "response": "Usage: `/gh pr list owner/repo` or set a default_repo.",
            "handled": True,
        }

    max_results = config.get("max_results", _DEFAULT_MAX_RESULTS)
    show_rate = config.get("show_rate_limit", False)

    result = _github_api(
        "GET", f"/repos/{repo}/pulls", token,
        params={"state": "open", "per_page": str(max_results)},
    )

    if "_error" in result:
        return {
            "response": f"Error fetching PRs: {result['_error']}",
            "handled": True,
        }

    prs = result.get("items", [])
    if not prs:
        msg = f"No open pull requests in **{repo}**."
    else:
        lines = [f"**Open PRs in {repo}** ({len(prs)}):"]
        for pr in prs:
            number = pr.get("number", "?")
            title = pr.get("title", "Untitled")
            user = pr.get("user", {}).get("login", "?")
            draft = " [draft]" if pr.get("draft", False) else ""
            lines.append(f"- #{number}: {title} by {user}{draft}")
        msg = "\n".join(lines)

    msg += _format_rate_limit(result.get("_rate_limit", {}), show_rate)
    return {"response": msg, "handled": True}


def _cmd_repo_info(args: str, ctx: Any, config: dict) -> dict[str, Any]:
    """Handle /gh repo info <owner/repo>."""
    token, error = _require_token(ctx)
    if error:
        return error

    repo = _parse_owner_repo(args, config)
    if not repo:
        return {
            "response": "Usage: `/gh repo info owner/repo`",
            "handled": True,
        }

    show_rate = config.get("show_rate_limit", False)

    result = _github_api("GET", f"/repos/{repo}", token)

    if "_error" in result:
        return {
            "response": f"Error fetching repo info: {result['_error']}",
            "handled": True,
        }

    name = result.get("full_name", repo)
    description = result.get("description", "No description")
    stars = result.get("stargazers_count", 0)
    forks = result.get("forks_count", 0)
    language = result.get("language", "Unknown")
    open_issues = result.get("open_issues_count", 0)
    license_info = result.get("license", {})
    license_name = license_info.get("spdx_id", "None") if license_info else "None"
    archived = " [ARCHIVED]" if result.get("archived", False) else ""

    msg = (
        f"**{name}**{archived}\n"
        f"{description}\n\n"
        f"- Language: {language}\n"
        f"- Stars: {stars}\n"
        f"- Forks: {forks}\n"
        f"- Open issues: {open_issues}\n"
        f"- License: {license_name}"
    )

    msg += _format_rate_limit(result.get("_rate_limit", {}), show_rate)
    return {"response": msg, "handled": True}


def _cmd_search(args: str, ctx: Any, config: dict) -> dict[str, Any]:
    """Handle /gh search <query>."""
    token, error = _require_token(ctx)
    if error:
        return error

    query = args.strip()
    if not query:
        return {
            "response": "Usage: `/gh search <query>`",
            "handled": True,
        }

    max_results = config.get("max_results", _DEFAULT_MAX_RESULTS)
    show_rate = config.get("show_rate_limit", False)

    result = _github_api(
        "GET", "/search/repositories", token,
        params={"q": query, "per_page": str(max_results), "sort": "stars"},
    )

    if "_error" in result:
        return {
            "response": f"Search error: {result['_error']}",
            "handled": True,
        }

    items = result.get("items", [])
    total = result.get("total_count", 0)

    if not items:
        msg = f"No repositories found for '{query}'."
    else:
        lines = [f"**Search results for '{query}'** ({total} total, showing {len(items)}):"]
        for item in items:
            name = item.get("full_name", "?")
            stars = item.get("stargazers_count", 0)
            desc = item.get("description", "")
            if desc and len(desc) > 80:
                desc = desc[:77] + "..."
            desc_str = f" -- {desc}" if desc else ""
            lines.append(f"- **{name}** ({stars} stars){desc_str}")
        msg = "\n".join(lines)

    msg += _format_rate_limit(result.get("_rate_limit", {}), show_rate)
    return {"response": msg, "handled": True}


def _cmd_gist_create(args: str, ctx: Any, config: dict) -> dict[str, Any]:
    """Handle /gh gist create <description>."""
    token, error = _require_token(ctx)
    if error:
        return error

    description = args.strip() or "Created by Opti-Oignon"

    # Find last code block in conversation
    conversation = ctx.data.get("conversation", [])
    last_code = ""
    last_lang = "txt"

    # Search conversation in reverse for a code block
    code_block_re = re.compile(r"```(\w*)\n([\s\S]*?)```", re.DOTALL)
    for msg in reversed(conversation):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        matches = list(code_block_re.finditer(content))
        if matches:
            last_match = matches[-1]
            last_lang = last_match.group(1) or "txt"
            last_code = last_match.group(2)
            break

    if not last_code:
        return {
            "response": "No code block found in conversation history.",
            "handled": True,
        }

    show_rate = config.get("show_rate_limit", False)

    filename = f"snippet.{last_lang}" if last_lang != "txt" else "snippet.txt"
    body = {
        "description": description,
        "public": False,
        "files": {
            filename: {"content": last_code},
        },
    }

    result = _github_api("POST", "/gists", token, body=body)

    if "_error" in result:
        return {
            "response": f"Gist creation failed: {result['_error']}",
            "handled": True,
        }

    gist_url = result.get("html_url", "")
    msg = f"Gist created: {gist_url}"
    msg += _format_rate_limit(result.get("_rate_limit", {}), show_rate)
    return {"response": msg, "handled": True}


# =========================================================================
# Command router
# =========================================================================

_CMD_RE = re.compile(
    r"^/gh\s+"
    r"(auth|issues|pr\s+list|repo\s+info|search|gist\s+create)"
    r"(?:\s+(.*))?$",
    re.DOTALL,
)


def route_command(user_input: str, ctx: Any) -> dict[str, Any] | None:
    """Route /gh commands to the appropriate handler.

    Parameters
    ----------
    user_input : str
        Raw user input.
    ctx : Any
        Hook context.

    Returns
    -------
    dict or None
        Response dict if command matched, None otherwise.
    """
    m = _CMD_RE.match(user_input.strip())
    if not m:
        return None

    command = m.group(1).strip()
    args = (m.group(2) or "").strip()
    config = ctx.config or {}

    if command == "auth":
        return _cmd_auth(args, ctx)
    elif command == "issues":
        return _cmd_issues(args, ctx, config)
    elif command == "pr list":
        return _cmd_pr_list(args, ctx, config)
    elif command == "repo info":
        return _cmd_repo_info(args, ctx, config)
    elif command == "search":
        return _cmd_search(args, ctx, config)
    elif command == "gist create":
        return _cmd_gist_create(args, ctx, config)

    return None


# =========================================================================
# Auto-link: GitHub reference detection (post_inference)
# =========================================================================

# Patterns: owner/repo#123, #123 (with default_repo), full URLs
_REF_OWNER_REPO_ISSUE = re.compile(
    r"(?<!\w)([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)#(\d+)(?!\w)"
)
_REF_BARE_ISSUE = re.compile(
    r"(?<!\w)#(\d+)(?!\w)"
)
_REF_GITHUB_URL = re.compile(
    r"https?://github\.com/"
    r"([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
    r"/(?:issues|pull)/(\d+)"
)

# Code block detection for exclusion
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _strip_code_regions(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Identify code regions to exclude from reference scanning.

    Returns the text and a list of (start, end) ranges to skip.
    """
    regions: list[tuple[int, int]] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        regions.append((m.start(), m.end()))
    for m in _INLINE_CODE_RE.finditer(text):
        regions.append((m.start(), m.end()))
    return text, regions


def _in_code_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    """Check if a position falls within any code region."""
    for start, end in regions:
        if start <= pos < end:
            return True
    return False


def detect_github_refs(
    text: str,
    default_repo: str = "",
) -> list[dict[str, str]]:
    """Detect GitHub references in text.

    Parameters
    ----------
    text : str
        LLM response text.
    default_repo : str
        Default owner/repo for bare #123 references.

    Returns
    -------
    list[dict]
        List of detected references with 'repo', 'number', 'type' keys.
    """
    _, code_regions = _strip_code_regions(text)
    refs: list[dict[str, str]] = []
    seen: set[str] = set()

    # owner/repo#123
    for m in _REF_OWNER_REPO_ISSUE.finditer(text):
        if _in_code_region(m.start(), code_regions):
            continue
        repo = m.group(1)
        number = m.group(2)
        key = f"{repo}#{number}"
        if key not in seen:
            seen.add(key)
            refs.append({"repo": repo, "number": number, "type": "issue_or_pr"})

    # Full GitHub URLs
    for m in _REF_GITHUB_URL.finditer(text):
        if _in_code_region(m.start(), code_regions):
            continue
        repo = m.group(1)
        number = m.group(2)
        key = f"{repo}#{number}"
        if key not in seen:
            seen.add(key)
            refs.append({"repo": repo, "number": number, "type": "issue_or_pr"})

    # Bare #123 (only with default_repo set)
    if default_repo and "/" in default_repo:
        for m in _REF_BARE_ISSUE.finditer(text):
            if _in_code_region(m.start(), code_regions):
                continue
            number = m.group(1)
            key = f"{default_repo}#{number}"
            if key not in seen:
                seen.add(key)
                refs.append({
                    "repo": default_repo,
                    "number": number,
                    "type": "issue_or_pr",
                })

    return refs


def enrich_refs(
    refs: list[dict[str, str]],
    token: str,
) -> list[dict[str, Any]]:
    """Fetch metadata for detected references.

    Parameters
    ----------
    refs : list[dict]
        Detected references.
    token : str
        GitHub PAT.

    Returns
    -------
    list[dict]
        Enriched references with title, state, type info.
    """
    enriched: list[dict[str, Any]] = []

    for ref in refs:
        repo = ref["repo"]
        number = ref["number"]

        # Try issue endpoint first
        result = _github_api("GET", f"/repos/{repo}/issues/{number}", token)

        if "_error" in result:
            enriched.append({
                "repo": repo,
                "number": number,
                "error": result["_error"],
            })
            continue

        title = result.get("title", "")
        state = result.get("state", "")
        is_pr = "pull_request" in result
        ref_type = "PR" if is_pr else "Issue"

        enriched.append({
            "repo": repo,
            "number": number,
            "title": title,
            "state": state,
            "ref_type": ref_type,
        })

    return enriched


def format_footnotes(enriched: list[dict[str, Any]]) -> str:
    """Format enriched references as markdown footnotes.

    Parameters
    ----------
    enriched : list[dict]
        Enriched reference data.

    Returns
    -------
    str
        Footnote block to append to response.
    """
    if not enriched:
        return ""

    lines = ["", "---", "**GitHub references:**"]
    for ref in enriched:
        repo = ref["repo"]
        number = ref["number"]

        if "error" in ref:
            lines.append(f"- {repo}#{number}: could not fetch ({ref['error']})")
            continue

        title = ref.get("title", "")
        state = ref.get("state", "")
        ref_type = ref.get("ref_type", "Issue")
        state_badge = f" [{state}]" if state else ""
        lines.append(f"- {repo}#{number} ({ref_type}){state_badge}: {title}")

    return "\n".join(lines)


# =========================================================================
# Hook implementations
# =========================================================================


def hook_tool_call(ctx: Any) -> dict[str, Any] | None:
    """Tool call hook: handle /gh commands."""
    user_input = ctx.data.get("user_input", "") or ctx.data.get("prompt", "")
    if not user_input:
        return None

    user_input = user_input.strip()
    if not user_input.startswith("/gh"):
        return None

    return route_command(user_input, ctx)


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: auto-detect GitHub references and enrich.

    Scans the LLM response for GitHub references (owner/repo#123,
    full URLs) and appends metadata footnotes.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    if not config.get("auto_link", True):
        return None

    default_repo = config.get("default_repo", "")

    # Detect references
    refs = detect_github_refs(response, default_repo)
    if not refs:
        return None

    # Need a token for enrichment
    try:
        store = _get_store(ctx)
        token = store.get_token()
    except Exception:
        token = None

    if not token:
        return None

    # Enrich references (network calls)
    enriched = enrich_refs(refs, token)
    footnotes = format_footnotes(enriched)

    if not footnotes:
        return None

    return {
        "response": response + footnotes,
        "github_refs": len(enriched),
    }


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "tool_call": hook_tool_call,
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown: close token store."""
    global _token_store
    if _token_store is not None:
        _token_store.close()
        _token_store = None
