#!/usr/bin/env python3
"""
repo_kb.py — LIVING REPOS + LOCAL KNOWLEDGE MODE
════════════════════════════════════════════════════════════════════════════

Your work repo becomes a private, always-up-to-date knowledge base that a
LOCAL model (Ollama, on the Mac) answers from. Nothing leaves the machine and
no answer costs money.

HOW IT FITS TOGETHER (read this first)
──────────────────────────────────────
1. SYNC — works like rsync / "git pull"
   You drop the same repo every Friday. The browser first sends a MANIFEST:
   path + size + modified-time + fingerprint (SHA-256) of every file, but no
   file content. The server compares it with what it already has:
        new  /  changed  /  unchanged  /  missing
   You see that preview, click Apply, and ONLY new + changed files are
   uploaded. One repo that grows over time — never two copies.

2. SAFETY — the "double-triple lock"
   • Uploads land in a STAGING folder first. The repo only changes when the
     whole sync is committed, so a half-finished upload changes nothing.
   • A changed file's previous version is moved to repo-history/, never
     overwritten.
   • A file missing from a new drop is KEPT and flagged "removed", never deleted.
   • A repo is LOCKED by default. Deleting it needs: unlock + type its name,
     and even then it is only moved to repo-trash/ on disk.

3. UNDERSTANDING THE STRUCTURE
   Every path is read as data:      dev/grafana/otel-collector-crash/notes.md
        environment = dev   product = grafana   work item = otel-collector-crash
   Versions in names (nexus-3.68, 3.95/) are parsed as real numbers, so
   3.95 > 3.68 and 3.100 > 3.95.

4. "LATEST" IS DECIDED BY CODE, NOT BY THE AI
   A language model is good at explaining and bad at reliably comparing dates.
   So this code picks the latest work item (file dates + version numbers),
   marks it ★, and the model only explains it. That is what keeps
   "what's my latest work on X" exact instead of "probably right".

5. ANSWERING
   The local model is called through Ollama's HTTP API (localhost:11434).
   Search = keyword (SQLite FTS5) + meaning (nomic-embed-text vectors), fused.

Wiring: app.py calls init(app, ...) once at startup and routes any chat whose
model id starts with "ollama:" to handle_local_message(). The cloud (Claude /
OpenAI) path is untouched.
"""

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context

try:                        # numpy makes vector search ~50x faster; optional
    import numpy as _np
except Exception:           # pragma: no cover - plain Python fallback works too
    _np = None

log = logging.getLogger("localllm")
bp = Blueprint("repo_kb", __name__)

# Filled by init() — functions and folders borrowed from app.py
_D: dict = {}
DROP_DIR = REPOS_DIR = HISTORY_DIR = STAGING_DIR = TRASH_DIR = None

OLLAMA_URL = os.environ.get("LOCALLLM_OLLAMA_URL", "http://localhost:11434").rstrip("/")

# ── Limits ───────────────────────────────────────────────────────────────────
MAX_FILE_BYTES   = 90 * 1024 * 1024   # Cloudflare rejects requests over 100MB
MAX_READ_BYTES   = 3 * 1024 * 1024    # read at most 3MB of a text file for search
MAX_INDEX_CHARS  = 400_000            # …and index at most this many characters
CHUNK_CHARS      = 1400               # one search "chunk" ≈ a short section
CHUNK_OVERLAP    = 150
DEFAULT_EMBED    = "nomic-embed-text"
DEFAULT_NUM_CTX  = 16384              # Ollama's default (2–4k) silently cuts context

# Folders/files that are never part of your knowledge (same list in the browser)
IGNORE_DIRS  = {".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
                ".idea", ".vscode", ".terraform", ".pytest_cache", ".mypy_cache", ".cache"}
IGNORE_FILES = {".ds_store", "thumbs.db", "desktop.ini"}

TEXT_EXTS = {
    "md", "markdown", "txt", "text", "rst", "adoc", "org", "log", "out",
    "yaml", "yml", "json", "jsonc", "toml", "ini", "cfg", "conf", "config", "properties",
    "env", "xml", "csv", "tsv", "sql", "tf", "tfvars", "hcl", "tpl", "j2", "jinja", "gotmpl",
    "sh", "bash", "zsh", "ksh", "fish", "ps1", "bat", "cmd", "py", "rb", "pl", "go", "rs",
    "java", "kt", "kts", "groovy", "gradle", "scala", "js", "mjs", "cjs", "ts", "tsx", "jsx",
    "c", "h", "cpp", "hpp", "cs", "php", "swift", "lua", "r", "html", "htm", "css", "scss",
    "dockerfile", "containerfile", "service", "timer", "socket", "repo", "pem-info", "ldif",
    "cnf", "pp", "erb", "vcl", "nginx", "rules", "list", "lst", "diff", "patch", "http",
}
DOC_EXTS = {"pdf", "docx"}
BINARY_EXTS = {
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tif", "tiff", "heic", "svgz",
    "mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "ogg", "flac",
    "zip", "gz", "tgz", "tar", "bz2", "xz", "7z", "rar", "jar", "war", "ear", "whl",
    "exe", "dll", "so", "dylib", "bin", "iso", "img", "dmg", "pkg", "deb", "rpm",
    "p12", "pfx", "jks", "keystore", "der", "class", "pyc", "o", "a",
    "xlsx", "xls", "pptx", "ppt", "doc", "odt", "ods", "odp", "sqlite", "db",
}


# ═════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═════════════════════════════════════════════════════════════════════════════
def init(app, *, get_db, now_iso, load_settings, save_setting, export_chat_txt,
         extract_text, base_dir, drop_dir):
    """Called once by app.py. Creates folders + tables, registers the routes."""
    global DROP_DIR, REPOS_DIR, HISTORY_DIR, STAGING_DIR, TRASH_DIR
    _D.update(get_db=get_db, now_iso=now_iso, load_settings=load_settings,
              save_setting=save_setting, export_chat_txt=export_chat_txt,
              extract_text=extract_text)
    base_dir = Path(base_dir)
    DROP_DIR    = Path(drop_dir)
    REPOS_DIR   = DROP_DIR / "living-repos"          # current files, browsable in Finder
    HISTORY_DIR = base_dir / "repo-history"          # every replaced version of a file
    STAGING_DIR = base_dir / "repo-staging"          # uploads wait here until commit
    TRASH_DIR   = base_dir / "repo-trash"            # deleted repos (still on disk)
    for d in (REPOS_DIR, HISTORY_DIR, STAGING_DIR, TRASH_DIR):
        d.mkdir(parents=True, exist_ok=True)
    _init_schema()
    _recover_interrupted_syncs()
    _cleanup_stale_staging()
    app.register_blueprint(bp)
    threading.Thread(target=_lock_existing_files, daemon=True).start()
    _kick_embeddings()      # finish any embeddings left over from last run
    log.info("Living Repos ready: %s", REPOS_DIR)


@contextmanager
def _db():
    """A short-lived DB connection that commits on success and always closes."""
    conn = _D["get_db"]()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _now() -> str:
    return _D["now_iso"]()


def _setting(key: str, default: str = "") -> str:
    try:
        return _D["load_settings"]().get(key, default) or default
    except Exception:
        return default


_SCHEMA = """
-- One row per living repo. slug = lower-case name = the sync identity.
CREATE TABLE IF NOT EXISTS lr_repos (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    slug          TEXT NOT NULL UNIQUE,
    locked        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    last_sync_at  TEXT,
    sync_count    INTEGER NOT NULL DEFAULT 0
);

-- One row per file path in a repo (the CURRENT state of that path).
--   env/product/item/version  = what the path means (see path_facets)
--   mtime      = last-modified time reported by your laptop
--   date_hint  = a date written in the file name or the file's header
--   recency    = the timestamp used to decide "latest"
--   status     = active | removed  (removed = missing from a later drop, kept)
CREATE TABLE IF NOT EXISTS lr_files (
    id            TEXT PRIMARY KEY,
    repo_id       TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    size          INTEGER NOT NULL DEFAULT 0,
    mtime         TEXT DEFAULT '',
    first_seen    TEXT NOT NULL,
    last_changed  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    removed_at    TEXT,
    env           TEXT DEFAULT '',
    product       TEXT DEFAULT '',
    item          TEXT DEFAULT '',
    version       TEXT DEFAULT '',
    version_key   TEXT DEFAULT '',
    title         TEXT DEFAULT '',
    date_hint     TEXT DEFAULT '',
    recency       TEXT NOT NULL,
    is_text       INTEGER NOT NULL DEFAULT 1,
    UNIQUE(repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_lr_files_facets ON lr_files(repo_id, status, env, product);

-- Every earlier version of a file that a sync replaced (file kept on disk).
CREATE TABLE IF NOT EXISTS lr_file_versions (
    id            TEXT PRIMARY KEY,
    file_id       TEXT NOT NULL,
    sha256        TEXT,
    size          INTEGER,
    mtime         TEXT,
    archived_path TEXT,
    replaced_at   TEXT NOT NULL
);

-- Search chunks (sections of files). rowid is shared with lr_chunks_fts.
CREATE TABLE IF NOT EXISTS lr_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       TEXT NOT NULL,
    repo_id       TEXT NOT NULL,
    ord           INTEGER NOT NULL,
    heading       TEXT DEFAULT '',
    text          TEXT NOT NULL,
    embedding     TEXT,
    emb_model     TEXT
);
CREATE INDEX IF NOT EXISTS idx_lr_chunks_file ON lr_chunks(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS lr_chunks_fts USING fts5(path, heading, text);

-- One row per sync (the plan, then its result). plan = JSON manifest.
CREATE TABLE IF NOT EXISTS lr_syncs (
    id            TEXT PRIMARY KEY,
    repo_id       TEXT,
    repo_name     TEXT NOT NULL,
    status        TEXT NOT NULL,      -- planned | committing | committed | failed | cancelled
    plan          TEXT NOT NULL,
    result        TEXT DEFAULT '',
    progress_done INTEGER DEFAULT 0,
    progress_total INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);

-- What the local chat was last talking about, so "how did I fix it?" works.
CREATE TABLE IF NOT EXISTS lr_chat_focus (
    chat_id       TEXT PRIMARY KEY,
    data          TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


def _init_schema():
    with _db() as conn:
        conn.executescript(_SCHEMA)
        # Every living repo has a permanent NUMBER (#1, #2, …). The name can change
        # (the source folder gets renamed); the number never does and is never
        # reused, so "sync into #1" always means the same repo.
        if "ref_no" not in {r[1] for r in conn.execute("PRAGMA table_info(lr_repos)")}:
            conn.execute("ALTER TABLE lr_repos ADD COLUMN ref_no INTEGER")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_lr_repos_ref ON lr_repos(ref_no)")
        for r in conn.execute("SELECT id FROM lr_repos WHERE ref_no IS NULL ORDER BY created_at").fetchall():
            conn.execute("UPDATE lr_repos SET ref_no=? WHERE id=?", (_next_ref(conn), r["id"]))


def _next_ref(conn) -> int:
    """Next repo number. Remembered in settings so a deleted repo's number is
    never handed out again."""
    row = conn.execute("SELECT value FROM settings WHERE key='living_repo_last_ref'").fetchone()
    last = max(int(row[0]) if row and str(row[0]).isdigit() else 0,
               conn.execute("SELECT COALESCE(MAX(ref_no), 0) FROM lr_repos").fetchone()[0])
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('living_repo_last_ref', ?)",
                 (str(last + 1),))
    return last + 1


def _cleanup_stale_staging():
    """Uploads left in repo-staging/ by a sync that was cancelled, closed or
    finished are just copies (the originals are on your laptop) — tidy them.
    A preview left open for over a day is cancelled; nothing in the repo changes."""
    cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat(timespec="seconds")
    with _db() as conn:
        conn.execute("UPDATE lr_syncs SET status='cancelled', finished_at=? "
                     "WHERE status='planned' AND created_at < ?", (_now(), cutoff))
        live = {r[0] for r in conn.execute("SELECT id FROM lr_syncs WHERE status IN ('planned','committing')")}
    for d in STAGING_DIR.iterdir():
        if d.is_dir() and d.name not in live:
            shutil.rmtree(d, ignore_errors=True)
            log.info("Living Repos: removed leftover upload folder %s", d.name)


def _recover_interrupted_syncs():
    """A sync that was mid-commit when the app stopped is marked failed. Each
    file move is individually safe (old version archived first), so dropping
    the folder again finishes the job — nothing is lost."""
    with _db() as conn:
        n = conn.execute(
            "UPDATE lr_syncs SET status='failed', finished_at=?, "
            "result=json_object('error','Interrupted (app restarted). Drop the folder again — nothing was lost.') "
            "WHERE status='committing'", (_now(),)).rowcount
    if n:
        log.warning("Living Repos: %d interrupted sync(s) marked failed", n)


# ═════════════════════════════════════════════════════════════════════════════
#  SMALL HELPERS — names, paths, time
# ═════════════════════════════════════════════════════════════════════════════
def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._").lower()
    return s[:80]


def clean_relpath(p: str) -> str:
    """Normalise a browser-supplied relative path and refuse anything that could
    escape the repo folder (.., absolute paths, drive letters)."""
    p = (p or "").replace("\\", "/")
    parts = [s for s in p.split("/") if s not in ("", ".")]
    if not parts or any(s == ".." for s in parts) or re.match(r"^[A-Za-z]:$", parts[0]):
        raise ValueError(f"unsafe path: {p!r}")
    parts = [re.sub(r"[\x00-\x1f]", "", s)[:200] for s in parts]
    return "/".join(parts)


def is_ignored(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in IGNORE_DIRS for p in parts[:-1]):
        return True
    name = parts[-1].lower()
    return name in IGNORE_FILES or name.endswith(".pyc") or name.startswith("~$")


def _inside(base: Path, rel: str) -> Path:
    """base/rel, guaranteed to stay inside base."""
    target = (base / rel).resolve()
    if not str(target).startswith(str(base.resolve()) + os.sep):
        raise ValueError("path escapes its folder")
    return target


def _ms_to_iso(ms) -> str:
    """Browser File.lastModified (milliseconds) → the app's UTC ISO format."""
    try:
        ms = float(ms)
        if ms <= 0:
            return ""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None) \
            .isoformat(timespec="seconds")
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ── Finder-level lock (the macOS "Locked" flag) ──────────────────────────────
# Every file in a living repo, and every archived old version, gets the same
# "Locked" flag you can tick in Finder → Get Info. Finder then won't bin it
# without an extra "it's locked — continue?" confirmation, and other apps can't
# overwrite it. The app lifts the flag only for the instant it archives a file
# during a sync. (On Linux these helpers do nothing.)
_IMMUTABLE = getattr(stat, "UF_IMMUTABLE", 0)


def _fs_lock(p: Path) -> None:
    if not _IMMUTABLE or not hasattr(os, "chflags"):
        return
    try:
        flags = getattr(os.lstat(p), "st_flags", 0)
        if not flags & _IMMUTABLE:
            os.chflags(p, flags | _IMMUTABLE)
    except OSError as e:
        log.debug("lock %s: %s", p, e)


def _fs_unlock(p: Path) -> None:
    if not _IMMUTABLE or not hasattr(os, "chflags"):
        return
    try:
        flags = getattr(os.lstat(p), "st_flags", 0)
        if flags & _IMMUTABLE:
            os.chflags(p, flags & ~_IMMUTABLE)
    except OSError as e:
        log.debug("unlock %s: %s", p, e)


def _fs_is_locked(p: Path) -> bool:
    try:
        return bool(_IMMUTABLE and getattr(os.lstat(p), "st_flags", 0) & _IMMUTABLE)
    except OSError:
        return False


def _lock_existing_files() -> None:
    """Startup: files synced before these protections existed get their laptop
    'Date Modified' back (it was recorded at sync time), then every repo file and
    archived version gets the Finder lock."""
    try:
        with _db() as conn:
            rows = conn.execute("SELECT r.slug, f.path, f.mtime FROM lr_files f "
                                "JOIN lr_repos r ON r.id = f.repo_id WHERE f.mtime != ''").fetchall()
        for r in rows:
            fp = REPOS_DIR / r["slug"] / r["path"]
            if fp.is_file() and not _fs_is_locked(fp):
                _set_original_mtime(fp, r["mtime"])
    except Exception as e:
        log.warning("Living Repos: restoring original dates failed: %s", e)
    if not _IMMUTABLE or not hasattr(os, "chflags"):
        return
    n = 0
    for root in (REPOS_DIR, HISTORY_DIR):
        for dirpath, _, names in os.walk(root):
            for name in names:
                if name != "SYNC-LOG.txt":
                    _fs_lock(Path(dirpath) / name)
                    n += 1
    log.info("Living Repos: %d files carry the Finder lock", n)


def _set_original_mtime(p: Path, iso: str) -> None:
    """Give the Mac copy the same 'Date Modified' it had on the work laptop."""
    if not iso:
        return
    try:
        ts = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()
        os.utime(p, (ts, ts))
    except (ValueError, OSError) as e:
        log.debug("mtime %s: %s", p, e)


def _ext(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1].lower()
    if name in ("dockerfile", "containerfile", "makefile", "jenkinsfile", "vagrantfile",
                "procfile", "readme", "license", "notes", "todo", "changelog"):
        return "txt"
    return name.rsplit(".", 1)[-1] if "." in name else ""


# ═════════════════════════════════════════════════════════════════════════════
#  UNDERSTANDING A PATH — environment, product, work item, version, dates
# ═════════════════════════════════════════════════════════════════════════════
# Canonical environment → folder names that mean it.
_ENV_GROUPS = {
    "prod":    ("prod", "production", "prd", "live"),
    "maint":   ("maint", "maintenance", "main", "mgmt", "k8mgmt", "management"),
    "staging": ("staging", "stage", "stg", "preprod", "pre-prod"),
    "test":    ("test", "testing", "tst", "qa"),
    "uat":     ("uat",),
    "dev":     ("dev", "development", "develop"),
    "sandbox": ("sandbox", "lab", "playground", "poc"),
}
_ENV_LOOKUP = {a: canon for canon, aliases in _ENV_GROUPS.items() for a in aliases}
# In Naveen's setup the maint cluster IS production: if one is asked for and the
# repo has no folder for it, the other is used (and the answer says so).
_ENV_EQUIV = {"prod": ("maint",), "maint": ("prod",)}
# Words that can sit next to an env name in a folder ("dev-cluster", "k8s-prod")
_ENV_FILLER = {"cluster", "clusters", "env", "envs", "environment", "environments", "k8s",
               "kube", "kubernetes", "ocp", "openshift", "aks", "eks", "gke", "rke", "rke2",
               "k3s", "namespace", "ns", "infra", "site"}
# Env words that are also ordinary English — in a QUESTION they only count when
# they clearly name an environment ("main cluster", "in test", "on live").
_ENV_AMBIGUOUS_IN_QUERY = {"main", "live", "test", "testing", "lab", "stage", "qa", "management"}

KNOWN_PRODUCTS = {
    "nexus", "grafana", "kafka", "strimzi", "keycloak", "haproxy", "ldap", "prometheus",
    "alertmanager", "loki", "tempo", "mimir", "promtail", "alloy", "opentelemetry", "jaeger",
    "argocd", "jenkins", "gitlab", "github", "harbor", "vault", "consul", "postgres", "mysql",
    "mariadb", "mongodb", "redis", "elasticsearch", "opensearch", "kibana", "logstash",
    "fluentd", "fluentbit", "longhorn", "ceph", "rook", "minio", "velero", "cert-manager",
    "ingress-nginx", "nginx", "traefik", "istio", "linkerd", "metallb", "calico", "cilium",
    "flannel", "rancher", "sonarqube", "artifactory", "rabbitmq", "zookeeper", "airflow",
    "spark", "tomcat", "wildfly", "jboss", "apache", "ansible", "terraform", "openshift",
    "docker", "podman", "containerd", "etcd", "coredns", "kubelet", "helm", "kyverno",
    "gatekeeper", "falco", "trivy", "sealed-secrets", "external-secrets", "external-dns",
    "kafka-connect", "schema-registry", "kafdrop", "akhq", "pgbouncer", "patroni", "zabbix",
    "nagios", "icinga", "graylog", "splunk", "dynatrace", "datadog", "victoriametrics",
    "thanos", "cortex", "openldap", "freeipa", "samba", "bind", "dnsmasq", "squid", "keepalived",
    "openvpn", "wireguard", "pihole", "portainer", "awx", "tekton", "flux", "kustomize",
    "nfs", "glusterfs", "proxmox", "vmware", "vsphere", "nextcloud", "mattermost", "confluence",
    "jira", "bitbucket", "teamcity", "bamboo", "nexus-iq", "cassandra", "clickhouse", "influxdb",
    "telegraf", "memcached", "varnish", "envoy", "kong", "oauth2-proxy", "dex", "authentik",
}
_PRODUCT_SYNONYMS = {
    "otel": "opentelemetry", "open-telemetry": "opentelemetry", "opentelemetry-collector": "opentelemetry",
    "otel-collector": "opentelemetry", "postgresql": "postgres", "pg": "postgres",
    "nexus3": "nexus", "nexus-repository": "nexus", "nexus-repo": "nexus", "sonatype": "nexus",
    "elastic": "elasticsearch", "argo-cd": "argocd", "argo": "argocd", "certmanager": "cert-manager",
    "nginx-ingress": "ingress-nginx", "kube-prometheus-stack": "prometheus", "kc": "keycloak",
    "grafana-stack": "grafana", "lgtm": "grafana", "strimzi-kafka": "kafka", "es": "elasticsearch",
    "fluent-bit": "fluentbit", "victoria-metrics": "victoriametrics", "ldaps": "ldap",
}
# Folder names that organise but don't name a product.
_GENERIC_DIRS = {
    "clusters", "cluster", "environments", "environment", "envs", "env", "apps", "applications",
    "app", "services", "service", "projects", "project", "deployments", "deployment", "docs",
    "doc", "documentation", "notes", "note", "tickets", "ticket", "issues", "issue", "work",
    "misc", "general", "common", "shared", "infra", "infrastructure", "platform", "tools",
    "k8s", "kubernetes", "kube", "helm", "charts", "chart", "manifests", "manifest", "src",
    "files", "config", "configs", "configuration", "scripts", "script", "yaml", "yamls",
    "values", "resources", "base", "overlays", "archive", "old", "backup", "backups", "tmp",
    "temp", "draft", "drafts", "wip", "repo", "repos", "knowledge", "local-knowledge", "kb",
}

_VER_RE = re.compile(r"(?<![\d.])v?(\d{1,4})\.(\d{1,4})(?:\.(\d{1,5}))?(?:\.(\d{1,5}))?(?!\d)")


def versions_in(text: str) -> list:
    """All version numbers in a string → [((3, 95), '3.95'), …]. Dates like
    2026.06.12 are not versions and are skipped."""
    out = []
    for m in _VER_RE.finditer((text or "").lower()):
        parts = tuple(int(g) for g in m.groups() if g is not None)
        if parts[0] >= 1990 and len(parts) >= 2 and parts[1] <= 12:
            continue
        out.append((parts, m.group(0).lstrip("v")))
    return out


def version_key(parts) -> str:
    """(3, 95) → '00003.00095.00000.00000' — sorts correctly as text, so
    3.100 comes after 3.95 (plain text sorting would get that wrong)."""
    p = (list(parts) + [0, 0, 0, 0])[:4]
    return ".".join(f"{int(x):05d}" for x in p)


def _env_of_segment(seg: str):
    s = (seg or "").lower().strip()
    if s in _ENV_LOOKUP:
        return _ENV_LOOKUP[s]
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if not toks or len(toks) > 4:
        return None
    envs = {_ENV_LOOKUP[t] for t in toks if t in _ENV_LOOKUP}
    rest = [t for t in toks if t not in _ENV_LOOKUP and t not in _ENV_FILLER and not t.isdigit()]
    if len(envs) == 1 and not rest:
        return envs.pop()
    return None


def _norm_product(seg: str) -> str:
    s = (seg or "").lower().strip()
    for _, raw in versions_in(s):
        s = s.replace(raw, " ")
    s = re.sub(r"\.(md|txt|ya?ml|json|log|sh|conf|pdf|docx)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"(^|-)v($|-)", "-", s).strip("-")
    return _PRODUCT_SYNONYMS.get(s, s)


def _known_product(seg: str):
    n = _norm_product(seg)
    if n in KNOWN_PRODUCTS:
        return n
    first = n.split("-")[0] if n else ""
    first = _PRODUCT_SYNONYMS.get(first, first)
    return first if first in KNOWN_PRODUCTS else None


def _is_version_dir(seg: str, product: str) -> bool:
    """True for folders that only say WHICH VERSION: '3.95', 'v3.68', 'nexus-3.95'."""
    vs = versions_in(seg)
    if not vs:
        return False
    s = seg.lower()
    for _, raw in vs:
        s = s.replace(raw, " ")
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    toks = [t for t in toks if t not in ("v", "version", "release", "rel", "ver")
            and t != product and _PRODUCT_SYNONYMS.get(t) != product]
    return not toks


def path_facets(path: str) -> dict:
    """Read meaning out of a repo path.

        dev/grafana/otel-collector-crash/notes.md
          → env=dev, product=grafana, item=dev/grafana/otel-collector-crash
        prod/nexus/ha-deployment/3.95/values.yaml
          → env=prod, product=nexus, item=prod/nexus/ha-deployment, version=3.95
        nexus/prod/nexus-3.68/upgrade.md        (product first also works)
          → env=prod, product=nexus, item=nexus/prod, version=3.68

    The WORK ITEM is the unit of "what did I work on": the first folder below
    the environment/product (version folders are folded into it), or the file
    itself when it sits directly in the product folder.
    """
    parts = path.split("/")
    dirs, fname = parts[:-1], parts[-1]

    env, env_idx = "", None
    for i, d in enumerate(dirs[:3]):           # environments live near the top
        e = _env_of_segment(d)
        if e:
            env, env_idx = e, i
            break

    product, prod_idx = "", None
    for i, d in enumerate(dirs):               # 1) a well-known product name anywhere
        if i == env_idx:
            continue
        k = _known_product(d)
        if k:
            product, prod_idx = k, i
            break
    if prod_idx is None:                       # 2) the folder right after the env
        start = env_idx + 1 if env_idx is not None else 0
        for i in range(start, len(dirs)):
            n = _norm_product(dirs[i])
            if not n or n in _GENERIC_DIRS or _env_of_segment(dirs[i]) or _is_version_dir(dirs[i], ""):
                continue
            product, prod_idx = n, i
            break
    if not product:                            # 3) a product named in the file name
        k = _known_product(fname.rsplit(".", 1)[0])
        if k:
            product = k

    anchor = max([x for x in (env_idx, prod_idx) if x is not None], default=-1)
    # Version folders are folded away, so 3.68/ha and 3.95/ha are the SAME
    # work item with two versions.
    item_parts = dirs[:anchor + 1]
    item = ""
    saw_version_dir = False
    for i in range(anchor + 1, len(dirs)):
        if _is_version_dir(dirs[i], product):
            saw_version_dir = True
            continue
        item = "/".join(item_parts + [dirs[i]])
        break
    if not item:
        item = "/".join(item_parts) if (saw_version_dir and item_parts) else path

    # Version: from the deepest segment below the anchor that mentions one.
    version, vkey = "", ""
    scan = dirs[anchor + 1:] + [fname.rsplit(".", 1)[0]]
    for seg in reversed(scan):
        vs = versions_in(seg)
        if vs:
            best = max(vs, key=lambda v: v[0])
            version, vkey = best[1], version_key(best[0])
            break

    return {"env": env, "product": product, "item": item, "version": version, "version_key": vkey}


# ── Dates written in names / headers ─────────────────────────────────────────
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "mär": 3, "maer": 3, "apr": 4, "may": 5, "mai": 5,
           "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "okt": 10, "nov": 11,
           "dec": 12, "dez": 12}
_MON_RE = r"(jan|feb|mar|mär|maer|apr|may|mai|jun|jul|aug|sep|oct|okt|nov|dec|dez)[a-zä]*\.?"
_DATE_RES = [
    (re.compile(r"(?<!\d)(20\d{2})[-_./](0?[1-9]|1[0-2])[-_./](0?[1-9]|[12]\d|3[01])(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(0?[1-9]|[12]\d|3[01])[./-](0?[1-9]|1[0-2])[./-](20\d{2})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\w)(\d{1,2})(?:st|nd|rd|th)?\.?\s+" + _MON_RE + r",?\s+(20\d{2})", re.I), "d_mon_y"),
    (re.compile(r"(?<!\w)" + _MON_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})", re.I), "mon_d_y"),
]
_YM_RE = re.compile(r"(?<!\d)(20\d{2})[-_.](0[1-9]|1[0-2])(?![\d])")
_HEADER_DATE_LINE = re.compile(
    r"^\s*[#*>\-\s]*(date|datum|updated|last updated|last update|created|stand|when|day)\b", re.I)


def _dates_in(text: str, year_month_ok: bool = False) -> list:
    found = []
    now = datetime.utcnow()
    for rx, kind in _DATE_RES:
        for m in rx.finditer(text or ""):
            try:
                g = m.groups()
                if kind == "ymd":
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                elif kind == "dmy":
                    d, mo, y = int(g[0]), int(g[1]), int(g[2])
                elif kind == "d_mon_y":
                    d, mo, y = int(g[0]), _MONTHS[g[1].lower()], int(g[2])
                else:
                    mo, d, y = _MONTHS[g[0].lower()], int(g[1]), int(g[2])
                dt = datetime(y, mo, d)
                if datetime(2000, 1, 1) <= dt <= now + timedelta(days=2):
                    found.append(dt)
            except (ValueError, KeyError):
                continue
    if year_month_ok and not found:
        for m in _YM_RE.finditer(text or ""):
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), 1)
                if dt <= now + timedelta(days=2):
                    found.append(dt)
            except ValueError:
                continue
    return found


def date_hint(rel: str, text: str) -> str:
    """A date the file itself claims: first from its path/name ('2026-09-26-otel.md',
    '26.09.2026'), else from a header line ('Date: …', 'Datum: …') or the first
    few lines. Returns ISO or ''."""
    ds = _dates_in(rel.replace("/", " "), year_month_ok=True)
    if not ds and text:
        head = text[:4000].splitlines()
        lines = [ln for ln in head[:60] if _HEADER_DATE_LINE.match(ln)] + head[:6]
        ds = _dates_in("\n".join(lines))
    return max(ds).isoformat(timespec="seconds") if ds else ""


def _recency(mtime: str, hint: str, fallback: str) -> str:
    """The moment that counts for 'latest' = the newest of (laptop's modified time,
    date written in the file). If neither is known, when the content last changed."""
    cands = [x for x in (mtime, hint) if x]
    return max(cands) if cands else fallback


def _title_of(text: str, rel: str) -> str:
    for ln in (text or "").splitlines()[:60]:
        m = re.match(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$", ln)
        if m:
            return m.group(1)[:140]
    for ln in (text or "").splitlines()[:15]:
        s = ln.strip().strip("#*-=>:`").strip()
        if len(s) >= 6 and not s.startswith(("apiVersion", "---", "{", "<?xml", "#!/")):
            return s[:140]
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"[-_]+", " ", stem)[:140]


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT + SEARCH INDEX
# ═════════════════════════════════════════════════════════════════════════════
def _read_text(abs_path: Path, rel: str):
    """→ (text, is_text). PDFs/DOCX go through app.py's extractor; binaries → ''."""
    ext = _ext(rel)
    try:
        if ext in DOC_EXTS:
            raw = abs_path.read_bytes()
            return (_D["extract_text"](rel.rsplit("/", 1)[-1], raw) or "")[:MAX_INDEX_CHARS], True
        if ext in BINARY_EXTS:
            return "", False
        with open(abs_path, "rb") as f:
            raw = f.read(MAX_READ_BYTES)
        if b"\x00" in raw[:8192]:
            return "", False
        return raw.decode("utf-8", errors="replace")[:MAX_INDEX_CHARS], True
    except Exception as e:
        log.warning("Living Repos: could not read %s: %s", rel, e)
        return "", False


def _split_long(text: str):
    """Split text into ~CHUNK_CHARS pieces on paragraph/line boundaries."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        if end < n:
            for sep in ("\n\n", "\n", ". ", " "):
                j = text.rfind(sep, i + CHUNK_CHARS // 2, end)
                if j != -1:
                    end = j + len(sep)
                    break
        out.append(text[i:end])
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return out


def chunk_document(text: str, rel: str):
    """→ [(heading_trail, chunk_text)]. Markdown is split by its headings so a
    chunk knows it belongs to e.g. 'Grafana OTel crash › Fix'."""
    if not text.strip():
        return []
    sections = []
    if _ext(rel) in ("md", "markdown"):
        trail, buf = [], []
        for ln in text.splitlines():
            m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", ln)
            if m:
                if "".join(buf).strip():
                    sections.append((" › ".join(trail), "\n".join(buf).strip()))
                level = len(m.group(1))
                trail = trail[:level - 1] + [m.group(2)[:80]]
                buf = [ln]
            else:
                buf.append(ln)
        if "".join(buf).strip():
            sections.append((" › ".join(trail), "\n".join(buf).strip()))
    else:
        sections = [("", text)]
    out = []
    for heading, body in sections:
        for piece in _split_long(body):
            if piece.strip():
                out.append((heading, piece))
    return out


def _delete_chunks(conn, file_id: str):
    ids = [r[0] for r in conn.execute("SELECT id FROM lr_chunks WHERE file_id=?", (file_id,))]
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        q = ",".join("?" * len(part))
        conn.execute(f"DELETE FROM lr_chunks_fts WHERE rowid IN ({q})", part)
        conn.execute(f"DELETE FROM lr_chunks WHERE id IN ({q})", part)
    if ids:
        _VEC["stamp"] += 1


def _index_file(conn, file_id: str, repo_id: str, rel: str, text: str, is_text: bool):
    """(Re)build the search chunks of one file."""
    _delete_chunks(conn, file_id)
    chunks = chunk_document(text, rel) if is_text else []
    if not chunks:   # binary/empty: index the path so the file can still be found by name
        chunks = [("", f"[{'binary' if not is_text else 'empty'} file: {rel}]")]
    path_words = rel.replace("/", " / ").replace("-", " ").replace("_", " ")
    for ord_, (heading, body) in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO lr_chunks (file_id, repo_id, ord, heading, text) VALUES (?,?,?,?,?)",
            (file_id, repo_id, ord_, heading, body))
        conn.execute("INSERT INTO lr_chunks_fts (rowid, path, heading, text) VALUES (?,?,?,?)",
                     (cur.lastrowid, path_words, heading, body))
    _VEC["stamp"] += 1


# ── Embeddings (meaning-based search) — run in the background ───────────────
_EMB = {"running": False, "last_error": "", "done": 0}
_EMB_LOCK = threading.Lock()
_VEC = {"stamp": 0, "loaded": -1, "model": "", "ids": [], "files": [], "mat": None}


def _embed_model() -> str:
    return _setting("local_embed_model", DEFAULT_EMBED)


def _http_json(url: str, payload=None, timeout: float = 30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed_texts(texts: list, kind: str = "document"):
    """→ list of unit-length vectors, or None if Ollama / the model isn't there.
    nomic-embed-text expects 'search_document:' / 'search_query:' prefixes."""
    model = _embed_model()
    if "nomic" in model:
        pre = "search_query: " if kind == "query" else "search_document: "
        texts = [pre + t for t in texts]
    texts = [t[:2000] for t in texts]
    vecs = None
    try:
        vecs = _http_json(f"{OLLAMA_URL}/api/embed", {"model": model, "input": texts},
                          timeout=120).get("embeddings")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            _EMB["last_error"] = f"embed HTTP {e.code}"
            return None
        try:   # older Ollama: one text per call
            vecs = [_http_json(f"{OLLAMA_URL}/api/embeddings", {"model": model, "prompt": t},
                               timeout=60).get("embedding") for t in texts]
        except Exception as e2:
            _EMB["last_error"] = str(e2)[:120]
            return None
    except Exception as e:
        _EMB["last_error"] = str(e)[:120]
        return None
    if not vecs or len(vecs) != len(texts) or not all(vecs):
        return None
    out = []
    for v in vecs:
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


def _kick_embeddings():
    if not _EMB["running"]:
        threading.Thread(target=_embed_worker, daemon=True).start()


def _embed_worker():
    if not _EMB_LOCK.acquire(blocking=False):
        return
    _EMB["running"] = True
    try:
        model = _embed_model()
        while True:
            with _db() as conn:
                rows = conn.execute(
                    "SELECT c.id, c.heading, c.text, f.path FROM lr_chunks c "
                    "JOIN lr_files f ON f.id = c.file_id "
                    "WHERE c.embedding IS NULL OR c.emb_model IS NOT ? "
                    "ORDER BY c.id LIMIT 32", (model,)).fetchall()
            if not rows:
                _EMB["last_error"] = ""
                break
            vecs = embed_texts([f"{r['path']}\n{r['heading']}\n{r['text']}" for r in rows])
            if vecs is None:
                if not _EMB["last_error"]:
                    _EMB["last_error"] = f"embedding model '{model}' not available"
                break
            with _db() as conn:
                for r, v in zip(rows, vecs):
                    conn.execute("UPDATE lr_chunks SET embedding=?, emb_model=? WHERE id=?",
                                 (json.dumps([round(x, 5) for x in v]), model, r["id"]))
            _EMB["done"] += len(rows)
            _VEC["stamp"] += 1
    except Exception as e:
        _EMB["last_error"] = str(e)[:160]
        log.warning("Living Repos: embedding worker stopped: %s", e)
    finally:
        _EMB["running"] = False
        _EMB_LOCK.release()


def _vector_scores(qvec, allowed_files: set) -> dict:
    """chunk_id → cosine similarity, for chunks of the allowed files."""
    model = _embed_model()
    if _VEC["loaded"] != _VEC["stamp"] or _VEC["model"] != model:
        with _db() as conn:
            rows = conn.execute("SELECT id, file_id, embedding FROM lr_chunks "
                                "WHERE embedding IS NOT NULL AND emb_model=?", (model,)).fetchall()
        ids = [r["id"] for r in rows]
        files = [r["file_id"] for r in rows]
        vecs = [json.loads(r["embedding"]) for r in rows]
        _VEC.update(ids=ids, files=files, model=model, loaded=_VEC["stamp"],
                    mat=(_np.array(vecs, dtype="float32") if (_np is not None and vecs) else vecs))
    if not _VEC["ids"]:
        return {}
    if _np is not None:
        sims = _VEC["mat"] @ _np.array(qvec, dtype="float32")
        return {cid: float(s) for cid, fid, s in zip(_VEC["ids"], _VEC["files"], sims)
                if fid in allowed_files}
    return {cid: sum(a * b for a, b in zip(qvec, vec))
            for cid, fid, vec in zip(_VEC["ids"], _VEC["files"], _VEC["mat"]) if fid in allowed_files}


# ═════════════════════════════════════════════════════════════════════════════
#  SYNC — plan → upload → commit
# ═════════════════════════════════════════════════════════════════════════════
def _repo_by_slug(conn, slug: str):
    return conn.execute("SELECT * FROM lr_repos WHERE slug=?", (slug,)).fetchone()


def _repo_by_ref(conn, ref):
    return conn.execute("SELECT * FROM lr_repos WHERE ref_no=?", (ref,)).fetchone()


def _parse_ref(v):
    """'3', 3 or '#3' → 3; anything else → None."""
    try:
        n = int(str(v).strip().lstrip("#"))
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _best_move_source(rel: str, cands: list) -> str:
    """Several missing files can share one fingerprint (e.g. identical READMEs).
    Pick the one that looks most like the same file: same file name first, then
    the most path parts in common from the end."""
    def score(c):
        common = 0
        for x, y in zip(reversed(c.split("/")), reversed(rel.split("/"))):
            if x != y:
                break
            common += 1
        return (c.rsplit("/", 1)[-1] == rel.rsplit("/", 1)[-1], common)
    return max(cands, key=score)


def _rename_groups(pairs: list) -> list:
    """[(old, new), …] → [{'from': 'dev/kafka-a', 'to': 'dev/kafka-strimzi', 'files': 12}]
    by dropping the path parts both sides share at the end."""
    groups = {}
    for old, new in pairs:
        a, b = old.split("/"), new.split("/")
        while len(a) > 1 and len(b) > 1 and a[-1] == b[-1]:
            a, b = a[:-1], b[:-1]
        key = ("/".join(a), "/".join(b))
        groups[key] = groups.get(key, 0) + 1
    return [{"from": k[0], "to": k[1], "files": n}
            for k, n in sorted(groups.items(), key=lambda kv: -kv[1])]


@bp.route("/api/repos/match", methods=["POST"])
def api_repo_match():
    """Does a dropped folder look like one of the living repos — even under a new
    name? Compares fingerprints (and, without them, paths) with every repo."""
    data = request.get_json(silent=True) or {}
    shas, paths = set(), set()
    for f in data.get("files") or []:
        sha = (f.get("sha256") or "").lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha):
            shas.add(sha)
        try:
            paths.add(clean_relpath(f.get("path", "")))
        except ValueError:
            pass
    out = []
    with _db() as conn:
        for r in conn.execute("SELECT * FROM lr_repos"):
            rows = conn.execute("SELECT path, sha256 FROM lr_files WHERE repo_id=? AND status='active'",
                                (r["id"],)).fetchall()
            same_content = sum(1 for x in rows if x["sha256"] in shas)
            same_path = sum(1 for x in rows if x["path"] in paths)
            score = max(same_content, same_path)
            if score:
                out.append({"ref": r["ref_no"], "name": r["name"], "slug": r["slug"], "files": len(rows),
                            "same_content": same_content, "same_path": same_path, "score": score})
    out.sort(key=lambda x: -x["score"])
    return jsonify({"dropped": len(paths), "candidates": out[:3]})


@bp.route("/api/repos/plan", methods=["POST"])
def api_repo_plan():
    """Step 1. The browser sends the manifest (no file content). We answer with
    what is new / changed / unchanged / missing and which files to upload."""
    data = request.get_json(silent=True) or {}
    ref = _parse_ref(data.get("ref")) if data.get("ref") not in (None, "") else None
    if data.get("ref") not in (None, "") and ref is None:
        return jsonify({"error": "The repo number must be a number, e.g. 1"}), 400
    if ref:
        with _db() as conn:
            target = _repo_by_ref(conn, ref)
        if not target:
            return jsonify({"error": f"There is no living repo #{ref}."}), 404
        name, slug = target["name"], target["slug"]
    else:
        name = (data.get("name") or "").strip()
        slug = _slug(name)
        if not slug:
            return jsonify({"error": "Give the repo a name (letters, digits, - _ .)"}), 400
    manifest, skipped, seen, too_big = {}, [], set(), set()
    for f in data.get("files") or []:
        try:
            rel = clean_relpath(f.get("path", ""))
        except ValueError:
            skipped.append({"path": str(f.get("path"))[:200], "why": "unsafe path"})
            continue
        if is_ignored(rel) or rel in seen:
            continue
        seen.add(rel)
        size = int(f.get("size") or 0)
        if size > MAX_FILE_BYTES:
            skipped.append({"path": rel, "why": f"larger than {MAX_FILE_BYTES // 1048576}MB"})
            too_big.add(rel)
            continue
        sha = (f.get("sha256") or "").lower()
        manifest[rel] = {"size": size, "mtime": _ms_to_iso(f.get("mtime")),
                         "sha": sha if re.fullmatch(r"[0-9a-f]{64}", sha or "") else ""}
    if not manifest:
        return jsonify({"error": "No usable files in this folder (empty, or only ignored files like .git)."}), 400

    with _db() as conn:
        repo = _repo_by_slug(conn, slug)
        existing = {}
        if repo:
            for r in conn.execute("SELECT path, sha256, size, mtime, status FROM lr_files WHERE repo_id=?",
                                  (repo["id"],)):
                existing[r["path"]] = dict(r)

    missing = sorted(p for p, r in existing.items()
                     if r["status"] == "active" and p not in manifest and p not in too_big)
    # MOVED / RENAMED: a file at a new path whose fingerprint equals a file that
    # is missing from this drop is the SAME file under a new name or folder.
    # It keeps its history and dates, and nothing is uploaded again.
    by_sha = {}
    for p_ in missing:
        by_sha.setdefault(existing[p_]["sha256"], []).append(p_)
    moved_pairs = []

    counts = {"new": 0, "changed": 0, "check": 0, "unchanged": 0, "restored": 0, "moved": 0, "missing": 0}
    samples = {k: [] for k in counts}
    upload = []
    for rel, m in manifest.items():
        old = existing.get(rel)
        if not old:
            cls = "new"
            if m["sha"] and by_sha.get(m["sha"]):
                src = _best_move_source(rel, by_sha[m["sha"]])
                by_sha[m["sha"]].remove(src)
                cls, m["from"] = "moved", src
                moved_pairs.append((src, rel))
        elif m["sha"]:
            cls = ("restored" if old["status"] == "removed" else "unchanged") \
                if m["sha"] == old["sha256"] else "changed"
        elif m["size"] == old["size"] and m["mtime"] and m["mtime"] == old["mtime"]:
            cls = "restored" if old["status"] == "removed" else "unchanged"
        else:
            cls = "check"      # no fingerprint from browser — server compares after upload
        m["cls"] = cls
        counts[cls] += 1
        if len(samples[cls]) < 200:
            samples[cls].append(f"{m['from']}  →  {rel}" if cls == "moved" else rel)
        if cls in ("new", "changed", "check"):
            upload.append(rel)
    moved_from = {a for a, _ in moved_pairs}
    missing = [p_ for p_ in missing if p_ not in moved_from]
    counts["missing"] = len(missing)
    samples["missing"] = missing[:200]

    warnings = []
    active_before = sum(1 for r in existing.values() if r["status"] == "active")
    if repo and active_before >= 10 and len(missing) > active_before * 0.5:
        warnings.append(f"This drop is missing {len(missing)} of the {active_before} files already in "
                        f"'{repo['name']}'. Did you pick the right folder (not a sub-folder)? "
                        "Missing files are kept either way — nothing gets deleted.")
    if repo and active_before >= 10 and counts["new"] > len(manifest) * 0.8:
        warnings.append("Almost every file looks new. If the top folder changed (an extra or missing "
                        "level), paths won't match — check the samples before applying.")

    plan_id = uuid.uuid4().hex
    plan = {"slug": slug, "name": repo["name"] if repo else name, "files": manifest,
            "missing": missing, "upload": upload, "counts": counts,
            "repo_id": repo["id"] if repo else None}
    with _db() as conn:
        conn.execute("INSERT INTO lr_syncs (id, repo_id, repo_name, status, plan, created_at) "
                     "VALUES (?,?,?,?,?,?)",
                     (plan_id, repo["id"] if repo else None, plan["name"], "planned",
                      json.dumps(plan), _now()))
    return jsonify({
        "plan_id": plan_id,
        "repo": {"name": plan["name"], "slug": slug, "exists": bool(repo),
                 "ref": repo["ref_no"] if repo else None,
                 "locked": bool(repo["locked"]) if repo else True,
                 "files_before": active_before},
        "counts": counts, "samples": samples, "upload": upload,
        "renames": _rename_groups(moved_pairs)[:50],
        "skipped": skipped[:200], "warnings": warnings,
    })


def _load_plan(plan_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", plan_id or ""):
        return None, None
    with _db() as conn:
        row = conn.execute("SELECT * FROM lr_syncs WHERE id=?", (plan_id,)).fetchone()
    if not row:
        return None, None
    return row, json.loads(row["plan"])


@bp.route("/api/repos/upload", methods=["POST"])
def api_repo_upload():
    """Step 2. Receive a batch of the files the plan asked for → staging folder."""
    row, plan = _load_plan(request.form.get("plan_id", ""))
    if not row:
        return jsonify({"error": "Unknown sync plan — start the drop again."}), 404
    if row["status"] != "planned":
        return jsonify({"error": f"This sync is already {row['status']}."}), 409
    wanted = set(plan["upload"])
    files = request.files.getlist("files")
    paths = request.form.getlist("paths")
    if len(files) != len(paths):
        return jsonify({"error": "files/paths mismatch"}), 400
    stage = STAGING_DIR / row["id"]
    got = 0
    for f, p in zip(files, paths):
        try:
            rel = clean_relpath(p)
        except ValueError:
            continue
        if rel not in wanted:
            continue
        dest = _inside(stage, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dest))
        got += 1
    return jsonify({"ok": True, "received": got})


@bp.route("/api/repos/commit", methods=["POST"])
def api_repo_commit():
    """Step 3. All uploads are in staging → apply the sync in the background."""
    data = request.get_json(silent=True) or {}
    row, plan = _load_plan(data.get("plan_id", ""))
    if not row:
        return jsonify({"error": "Unknown sync plan."}), 404
    if row["status"] != "planned":
        return jsonify({"error": f"This sync is already {row['status']}."}), 409
    stage = STAGING_DIR / row["id"]
    not_here = [p for p in plan["upload"] if not (stage / p).is_file()]
    if not_here:
        return jsonify({"error": f"{len(not_here)} file(s) did not arrive — nothing was changed. "
                                 "Drop the folder again.", "missing_uploads": not_here[:50]}), 409
    with _db() as conn:
        conn.execute("UPDATE lr_syncs SET status='committing', progress_total=? WHERE id=?",
                     (len(plan["files"]) + len(plan["missing"]), row["id"]))
    threading.Thread(target=_commit_sync, args=(row["id"],), daemon=True).start()
    return jsonify({"ok": True, "plan_id": row["id"]})


@bp.route("/api/repos/cancel", methods=["POST"])
def api_repo_cancel():
    data = request.get_json(silent=True) or {}
    row, _ = _load_plan(data.get("plan_id", ""))
    if row and row["status"] == "planned":
        shutil.rmtree(STAGING_DIR / row["id"], ignore_errors=True)
        with _db() as conn:
            conn.execute("UPDATE lr_syncs SET status='cancelled', finished_at=? WHERE id=?",
                         (_now(), row["id"]))
    return jsonify({"ok": True})


@bp.route("/api/repos/sync/<plan_id>", methods=["GET"])
def api_repo_sync_status(plan_id):
    row, _ = _load_plan(plan_id)
    if not row:
        return jsonify({"error": "unknown sync"}), 404
    return jsonify({"status": row["status"], "done": row["progress_done"],
                    "total": row["progress_total"],
                    "result": json.loads(row["result"]) if row["result"] else None})


_SYNC_LOCK = threading.Lock()


def _commit_sync(plan_id: str):
    """Apply a sync. Per file: archive the old version → move the new one in →
    update the DB row → re-index. Missing files are only flagged."""
    with _SYNC_LOCK:
        row, plan = _load_plan(plan_id)
        stage = STAGING_DIR / plan_id
        now = _now()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        res = {"added": 0, "changed": 0, "unchanged": 0, "restored": 0, "removed": 0, "moved": 0,
               "same_content": 0, "errors": [],
               "paths": {"added": [], "changed": [], "removed": [], "restored": [], "moved": []}}
        try:
            with _db() as conn:
                repo = None
                if plan.get("repo_id"):              # by identity — survives a rename
                    repo = conn.execute("SELECT * FROM lr_repos WHERE id=?", (plan["repo_id"],)).fetchone()
                if not repo:
                    repo = _repo_by_slug(conn, plan["slug"])
                if not repo:
                    rid = uuid.uuid4().hex
                    conn.execute("INSERT INTO lr_repos (id, name, slug, locked, created_at, updated_at, ref_no) "
                                 "VALUES (?,?,?,1,?,?,?)",
                                 (rid, plan["name"], plan["slug"], now, now, _next_ref(conn)))
                    repo = _repo_by_slug(conn, plan["slug"])
                conn.execute("UPDATE lr_syncs SET repo_id=? WHERE id=?", (repo["id"], plan_id))
                # Files missing from this drop, by fingerprint: an uploaded "new" file
                # with one of these fingerprints is really a move (caught here when the
                # browser couldn't fingerprint before uploading, e.g. plain http).
                pool = {}
                for rel_ in plan["missing"]:
                    r_ = conn.execute("SELECT sha256 FROM lr_files WHERE repo_id=? AND path=? AND status='active'",
                                      (repo["id"], rel_)).fetchone()
                    if r_:
                        pool.setdefault(r_["sha256"], []).append(rel_)
            ctx = {"pool": pool, "consumed": set()}
            mirror = REPOS_DIR / repo["slug"]
            mirror.mkdir(parents=True, exist_ok=True)
            hist = HISTORY_DIR / repo["slug"] / stamp

            done = 0
            for rel, m in plan["files"].items():
                try:
                    _apply_one(repo, rel, m, stage, mirror, hist, now, res, ctx)
                except Exception as e:
                    res["errors"].append(f"{rel}: {e}")
                    log.warning("Living Repos: %s failed: %s", rel, e)
                done += 1
                if done % 25 == 0:
                    with _db() as conn:
                        conn.execute("UPDATE lr_syncs SET progress_done=? WHERE id=?", (done, plan_id))

            if res["moved"]:
                _prune_empty_dirs(mirror)            # folders that were renamed away
            with _db() as conn:
                for rel in plan["missing"]:
                    if rel in ctx["consumed"]:
                        continue                     # it moved — not missing
                    n = conn.execute("UPDATE lr_files SET status='removed', removed_at=? "
                                     "WHERE repo_id=? AND path=? AND status='active'",
                                     (now, repo["id"], rel)).rowcount
                    res["removed"] += n
                    if n:
                        res["paths"]["removed"].append(rel)
                conn.execute("UPDATE lr_repos SET updated_at=?, last_sync_at=?, sync_count=sync_count+1 "
                             "WHERE id=?", (now, now, repo["id"]))
                sync_no = conn.execute("SELECT sync_count FROM lr_repos WHERE id=?",
                                       (repo["id"],)).fetchone()[0]
                conn.execute("UPDATE lr_syncs SET status='committed', finished_at=?, result=?, "
                             "progress_done=progress_total WHERE id=?",
                             (_now(), json.dumps(res), plan_id))
            log.info("Living Repos: synced '%s' — %s", repo["name"],
                     {k: v for k, v in res.items() if k not in ("errors", "paths")})
            _write_sync_log(repo, sync_no, stamp, res)
        except Exception as e:
            res["error"] = str(e)
            log.exception("Living Repos: sync %s failed", plan_id)
            with _db() as conn:
                conn.execute("UPDATE lr_syncs SET status='failed', finished_at=?, result=? WHERE id=?",
                             (_now(), json.dumps(res), plan_id))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    _kick_embeddings()


def _write_sync_log(repo, sync_no: int, stamp: str, res: dict) -> None:
    """repo-history/<repo>/SYNC-LOG.txt — one readable entry per sync, newest
    last: what came in, what changed (and where the old version is), what was
    missing from the drop (kept)."""
    p = HISTORY_DIR / repo["slug"] / "SYNC-LOG.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    paths = res["paths"]
    out = ["═" * 72,
           f"Sync #{sync_no} · {datetime.now():%a %d %b %Y, %H:%M} · repo #{repo['ref_no']} {repo['name']}",
           f"  + {res['added']} new   ~ {res['changed']} changed   - {res['removed']} missing from the drop (kept)"
           f"   = {res['unchanged'] + res['same_content']} unchanged"
           + (f"   ↪ {res['moved']} moved/renamed" if res["moved"] else "")
           + (f"   ↺ {res['restored']} back again" if res["restored"] else "")]
    if paths["changed"]:
        out.append(f"  Old versions of changed files: repo-history/{repo['slug']}/{stamp}/")
    out += [f"  + {x}" for x in paths["added"]]
    out += [f"  ~ {x}" for x in paths["changed"]]
    out += [f"  - {x}   (still in living-repos, flagged 'removed')" for x in paths["removed"]]
    out += [f"  ↺ {x}" for x in paths["restored"]]
    out += [f"  ↪ {x}" for x in paths["moved"]]
    out += [f"  ! {e}" for e in res["errors"]]
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n\n")
    except OSError as e:
        log.warning("sync log write failed: %s", e)


def _prune_empty_dirs(root: Path) -> None:
    """Remove folders left empty inside a repo copy (e.g. a folder that was renamed).
    Only empty folders — never a file."""
    for dirpath, _, _ in os.walk(root, topdown=False):
        d = Path(dirpath)
        if d != root:
            try:
                left = [x for x in d.iterdir() if x.name != ".DS_Store"]
                if not left:
                    for x in d.iterdir():
                        x.unlink()
                    d.rmdir()
            except OSError:
                pass


def _archive(src: Path, hist: Path, rel: str) -> Path:
    """Move a repo file into this sync's history folder (never delete it)."""
    archived = _inside(hist, rel)
    archived.parent.mkdir(parents=True, exist_ok=True)
    _fs_unlock(src)
    shutil.move(str(src), str(archived))
    _fs_lock(archived)
    return archived


def _move_file(conn, repo, srow, rel, m, mirror, hist, now, res, staged=None):
    """The same file (same fingerprint) now lives at a new path: move it there and
    update its row — it keeps its id, first-seen date, history and 'latest' date."""
    src, dest = _inside(mirror, srow["path"]), _inside(mirror, rel)
    if dest.exists():
        _archive(dest, hist, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if staged is not None:                    # an identical copy was uploaded anyway
        shutil.move(str(staged), str(dest))
        if src.exists():
            _archive(src, hist, srow["path"])
    else:
        _fs_unlock(src)
        shutil.move(str(src), str(dest))
    _set_original_mtime(dest, m.get("mtime") or srow["mtime"] or "")
    _fs_lock(dest)
    text, is_text = _read_text(dest, rel)
    fac = path_facets(rel)
    hint = date_hint(rel, text)
    rec = max(srow["recency"], hint) if hint else srow["recency"]     # a rename isn't new work
    conn.execute(
        "UPDATE lr_files SET path=?, mtime=?, status='active', removed_at=NULL, env=?, product=?, "
        "item=?, version=?, version_key=?, title=?, date_hint=?, recency=?, is_text=? WHERE id=?",
        (rel, m.get("mtime") or srow["mtime"], fac["env"], fac["product"], fac["item"], fac["version"],
         fac["version_key"], _title_of(text, rel), hint, rec, int(is_text), srow["id"]))
    _index_file(conn, srow["id"], repo["id"], rel, text, is_text)
    res["moved"] += 1
    res["paths"]["moved"].append(f"{srow['path']}  →  {rel}")


def _apply_one(repo, rel, m, stage, mirror, hist, now, res, ctx=None):
    ctx = ctx if ctx is not None else {"pool": {}, "consumed": set()}
    cls = m["cls"]
    with _db() as conn:
        old = conn.execute("SELECT * FROM lr_files WHERE repo_id=? AND path=?",
                           (repo["id"], rel)).fetchone()
        if cls == "moved":
            srow = conn.execute("SELECT * FROM lr_files WHERE repo_id=? AND path=?",
                                (repo["id"], m["from"])).fetchone()
            if not srow or not _inside(mirror, m["from"]).exists():
                raise RuntimeError(f"moved from {m['from']}, but that file is no longer there — drop again")
            _move_file(conn, repo, srow, rel, m, mirror, hist, now, res)
            ctx["consumed"].add(m["from"])
            return
        if cls == "unchanged":
            res["unchanged"] += 1
            return
        if cls == "restored":
            conn.execute("UPDATE lr_files SET status='active', removed_at=NULL WHERE id=?", (old["id"],))
            res["restored"] += 1
            res["paths"]["restored"].append(rel)
            return

        staged = stage / rel
        sha = _sha256_file(staged)
        if not old and ctx["pool"].get(sha):
            src_rel = _best_move_source(rel, ctx["pool"][sha])
            ctx["pool"][sha].remove(src_rel)
            srow = conn.execute("SELECT * FROM lr_files WHERE repo_id=? AND path=?",
                                (repo["id"], src_rel)).fetchone()
            if srow:
                _move_file(conn, repo, srow, rel, m, mirror, hist, now, res, staged=staged)
                ctx["consumed"].add(src_rel)
                return
        if old and old["sha256"] == sha:
            # Same content as we already have (only the timestamp differed):
            # keep the file's "latest" date — nothing new was written in it.
            staged.unlink()
            conn.execute("UPDATE lr_files SET status='active', removed_at=NULL WHERE id=?", (old["id"],))
            res["same_content"] += 1
            return

        dest = _inside(mirror, rel)
        if dest.exists():
            # Never overwrite: the current copy goes to repo-history first.
            archived = _archive(dest, hist, rel)
            if old:
                conn.execute("INSERT INTO lr_file_versions (id, file_id, sha256, size, mtime, archived_path, "
                             "replaced_at) VALUES (?,?,?,?,?,?,?)",
                             (uuid.uuid4().hex, old["id"], old["sha256"], old["size"], old["mtime"],
                              str(archived.relative_to(HISTORY_DIR)), now))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(dest))
        _set_original_mtime(dest, m.get("mtime") or "")
        _fs_lock(dest)

        text, is_text = _read_text(dest, rel)
        fac = path_facets(rel)
        hint = date_hint(rel, text)
        mtime = m.get("mtime") or ""
        rec = _recency(mtime, hint, now)
        title = _title_of(text, rel)
        size = dest.stat().st_size
        if old:
            fid = old["id"]
            conn.execute(
                "UPDATE lr_files SET sha256=?, size=?, mtime=?, last_changed=?, status='active', "
                "removed_at=NULL, env=?, product=?, item=?, version=?, version_key=?, title=?, "
                "date_hint=?, recency=?, is_text=? WHERE id=?",
                (sha, size, mtime, now, fac["env"], fac["product"], fac["item"], fac["version"],
                 fac["version_key"], title, hint, rec, int(is_text), fid))
            res["changed"] += 1
            res["paths"]["changed"].append(rel)
        else:
            fid = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO lr_files (id, repo_id, path, sha256, size, mtime, first_seen, last_changed, "
                "status, env, product, item, version, version_key, title, date_hint, recency, is_text) "
                "VALUES (?,?,?,?,?,?,?,?,'active',?,?,?,?,?,?,?,?,?)",
                (fid, repo["id"], rel, sha, size, mtime, now, now, fac["env"], fac["product"],
                 fac["item"], fac["version"], fac["version_key"], title, hint, rec, int(is_text)))
            res["added"] += 1
            res["paths"]["added"].append(rel)
        _index_file(conn, fid, repo["id"], rel, text, is_text)


# ═════════════════════════════════════════════════════════════════════════════
#  REPO MANAGEMENT — list, overview, lock, delete, normal folder upload
# ═════════════════════════════════════════════════════════════════════════════
@bp.route("/api/repos", methods=["GET"])
def api_repos_list():
    with _db() as conn:
        repos = conn.execute("SELECT * FROM lr_repos ORDER BY updated_at DESC").fetchall()
        out = []
        for r in repos:
            c = conn.execute(
                "SELECT SUM(status='active') AS active, SUM(status='removed') AS removed, "
                "COUNT(DISTINCT CASE WHEN status='active' AND env!='' THEN env END) AS envs, "
                "COUNT(DISTINCT CASE WHEN status='active' AND product!='' THEN product END) AS products "
                "FROM lr_files WHERE repo_id=?", (r["id"],)).fetchone()
            pend = conn.execute("SELECT COUNT(*) FROM lr_chunks WHERE repo_id=? AND embedding IS NULL",
                                (r["id"],)).fetchone()[0]
            out.append({"name": r["name"], "slug": r["slug"], "ref": r["ref_no"], "locked": bool(r["locked"]),
                        "files": c["active"] or 0, "removed": c["removed"] or 0,
                        "envs": c["envs"] or 0, "products": c["products"] or 0,
                        "last_sync_at": r["last_sync_at"], "sync_count": r["sync_count"],
                        "embedding_pending": pend})
    return jsonify({"repos": out, "dir": str(REPOS_DIR)})


@bp.route("/api/repos/<slug>/overview", methods=["GET"])
def api_repo_overview(slug):
    """What the app understood: environment → product → work items (newest first)."""
    with _db() as conn:
        repo = _repo_by_slug(conn, _slug(slug))
        if not repo:
            return jsonify({"error": "no such repo"}), 404
        rows = conn.execute("SELECT path, env, product, item, version, version_key, title, recency "
                            "FROM lr_files WHERE repo_id=? AND status='active'", (repo["id"],)).fetchall()
    tree = {}
    for r in rows:
        env = r["env"] or "(no environment)"
        prod = r["product"] or "(other)"
        it = tree.setdefault(env, {}).setdefault(prod, {}).setdefault(
            r["item"], {"files": 0, "recency": "", "title": "", "versions": set()})
        it["files"] += 1
        if r["version"]:
            it["versions"].add((r["version_key"], r["version"]))
        if r["recency"] > it["recency"]:
            it["recency"], it["title"] = r["recency"], r["title"]
    envs = []
    for env in sorted(tree):
        prods = []
        for prod in sorted(tree[env]):
            items = sorted(tree[env][prod].items(), key=lambda kv: kv[1]["recency"], reverse=True)
            prods.append({
                "product": prod, "items": len(items),
                "latest": {"item": items[0][0].rsplit("/", 1)[-1], "date": items[0][1]["recency"][:10],
                           "title": items[0][1]["title"]},
                "versions": [v for _, v in sorted({v for _, d in items for v in d["versions"]})],
            })
        envs.append({"env": env, "products": prods})
    return jsonify({"repo": repo["name"], "ref": repo["ref_no"], "envs": envs})


@bp.route("/api/repos/<slug>/verify", methods=["GET"])
def api_repo_verify(slug):
    """Re-check every file on disk against the fingerprint taken when it arrived.
    Proves the Mac copy is exactly what you dropped, and that old versions and
    'missing' files are all still there."""
    with _db() as conn:
        repo = _repo_by_slug(conn, _slug(slug))
        if not repo:
            return jsonify({"error": "no such repo"}), 404
        files = conn.execute("SELECT path, sha256, status FROM lr_files WHERE repo_id=?",
                             (repo["id"],)).fetchall()
        versions = conn.execute("SELECT v.archived_path FROM lr_file_versions v JOIN lr_files f "
                                "ON f.id=v.file_id WHERE f.repo_id=?", (repo["id"],)).fetchall()
    mirror = REPOS_DIR / repo["slug"]
    ok, changed, missing, locked = 0, [], [], 0
    for f in files:
        p = mirror / f["path"]
        if not p.is_file():
            missing.append(f["path"])
        elif _sha256_file(p) != f["sha256"]:
            changed.append(f["path"])
        else:
            ok += 1
            locked += _fs_is_locked(p)
    hist_ok = sum(1 for v in versions if (HISTORY_DIR / v["archived_path"]).is_file())
    return jsonify({
        "repo": repo["name"], "ref": repo["ref_no"], "checked": len(files), "intact": ok,
        "changed_outside_app": changed[:50], "missing_on_disk": missing[:50],
        "removed_but_kept": sum(1 for f in files if f["status"] == "removed"),
        "old_versions": len(versions), "old_versions_on_disk": hist_ok,
        "finder_locked": locked, "lock_supported": bool(_IMMUTABLE and hasattr(os, "chflags")),
        "folder": str(mirror),
    })


@bp.route("/api/repos/<slug>/rename", methods=["POST"])
def api_repo_rename(slug):
    """Give a living repo a new name (e.g. the source folder was renamed).
    Its number, files, history and search index stay exactly the same; its
    folders in living-repos/ and repo-history/ are renamed to match."""
    data = request.get_json(silent=True) or {}
    new_name = (data.get("name") or "").strip()
    new_slug = _slug(new_name)
    if not new_slug:
        return jsonify({"error": "Give a name (letters, digits, - _ .)"}), 400
    with _SYNC_LOCK:                                   # never while a sync is being applied
        with _db() as conn:
            repo = _repo_by_slug(conn, _slug(slug))
            if not repo:
                return jsonify({"error": "no such repo"}), 404
            other = _repo_by_slug(conn, new_slug)
            if other and other["id"] != repo["id"]:
                return jsonify({"error": f"Living repo #{other['ref_no']} is already called '{other['name']}'."}), 409
            old_slug = repo["slug"]
            if new_slug != old_slug:
                for base in (REPOS_DIR, HISTORY_DIR):
                    if (base / new_slug).exists():
                        return jsonify({"error": f"A folder '{new_slug}' already exists in {base}."}), 409
                for base in (REPOS_DIR, HISTORY_DIR):
                    if (base / old_slug).exists():
                        os.rename(str(base / old_slug), str(base / new_slug))   # locked files can stay locked
                conn.execute(
                    "UPDATE lr_file_versions SET archived_path = ? || substr(archived_path, ?) "
                    "WHERE archived_path LIKE ? AND file_id IN (SELECT id FROM lr_files WHERE repo_id=?)",
                    (new_slug, len(old_slug) + 1, old_slug + "/%", repo["id"]))
            conn.execute("UPDATE lr_repos SET name=?, slug=?, updated_at=? WHERE id=?",
                         (new_name, new_slug, _now(), repo["id"]))
    log.info("Living Repos: repo #%s renamed '%s' → '%s'", repo["ref_no"], repo["name"], new_name)
    return jsonify({"ok": True, "ref": repo["ref_no"], "name": new_name, "slug": new_slug})


@bp.route("/api/repos/<slug>/lock", methods=["POST"])
def api_repo_lock(slug):
    data = request.get_json(silent=True) or {}
    with _db() as conn:
        repo = _repo_by_slug(conn, _slug(slug))
        if not repo:
            return jsonify({"error": "no such repo"}), 404
        if data.get("locked") is False:
            if (data.get("confirm") or "").strip() != repo["name"]:
                return jsonify({"error": f"Type the repo name exactly ('{repo['name']}') to unlock."}), 400
            conn.execute("UPDATE lr_repos SET locked=0 WHERE id=?", (repo["id"],))
        else:
            conn.execute("UPDATE lr_repos SET locked=1 WHERE id=?", (repo["id"],))
    return jsonify({"ok": True})


@bp.route("/api/repos/<slug>/delete", methods=["POST"])
def api_repo_delete(slug):
    """Only for an UNLOCKED repo, only with its exact name typed, and the files
    are moved to repo-trash/ — not erased."""
    data = request.get_json(silent=True) or {}
    with _db() as conn:
        repo = _repo_by_slug(conn, _slug(slug))
        if not repo:
            return jsonify({"error": "no such repo"}), 404
        if repo["locked"]:
            return jsonify({"error": "This repo is locked. Unlock it first."}), 403
        if (data.get("confirm") or "").strip() != repo["name"]:
            return jsonify({"error": f"Type the repo name exactly ('{repo['name']}') to delete."}), 400
    dest = TRASH_DIR / f"{repo['slug']}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    src = REPOS_DIR / repo["slug"]
    if src.exists():
        os.rename(str(src), str(dest))        # files keep their Finder lock in the trash
    with _db() as conn:
        fids = [r[0] for r in conn.execute("SELECT id FROM lr_files WHERE repo_id=?", (repo["id"],))]
        for fid in fids:
            _delete_chunks(conn, fid)
        conn.execute("DELETE FROM lr_files WHERE repo_id=?", (repo["id"],))
        conn.execute("DELETE FROM lr_repos WHERE id=?", (repo["id"],))
    log.warning("Living Repos: deleted repo '%s' (files moved to %s)", repo["name"], dest)
    return jsonify({"ok": True, "moved_to": str(dest)})


@bp.route("/api/dropbox/upload-folder", methods=["POST"])
def api_dropbox_upload_folder():
    """A NORMAL folder drop (not a living repo): saved with its sub-folders intact
    under local-llm-dropbox/<timestamp>_<folder>/ — Google-Drive style."""
    target = re.sub(r"[^\w.\- ]+", "_", request.form.get("target", "")).strip(" ._")[:120]
    if not target or target.lower() == "living-repos":
        return jsonify({"error": "bad target folder"}), 400
    base = DROP_DIR / target
    files = request.files.getlist("files")
    paths = request.form.getlist("paths")
    if len(files) != len(paths):
        return jsonify({"error": "files/paths mismatch"}), 400
    saved = 0
    for f, p in zip(files, paths):
        try:
            rel = clean_relpath(p)
        except ValueError:
            continue
        if is_ignored(rel):
            continue
        dest = _inside(base, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dest))
        saved += 1
    return jsonify({"ok": True, "saved": saved, "folder": str(base)})


# ═════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL — understand the question, pick the right work item
# ═════════════════════════════════════════════════════════════════════════════
_LATEST_WORDS = {"latest", "last", "recent", "recently", "newest", "current", "currently", "status",
                 "now", "today", "yesterday", "lately", "ongoing"}
_LIST_WORDS = {"list", "all", "every", "overview", "everything", "inventory", "catalog", "catalogue"}
_HISTORY_WORDS = {"history", "timeline", "evolution", "changed", "changes", "previous", "older",
                  "earlier", "before", "compare", "comparison", "originally", "versions"}
_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "done", "doing", "have", "has",
    "had", "it", "its", "that", "this", "these", "those", "there", "here", "what", "whats", "which",
    "who", "whom", "where", "when", "why", "how", "my", "me", "i", "we", "our", "you", "your", "can",
    "could", "would", "should", "will", "shall", "please", "tell", "show", "give", "explain", "more",
    "detail", "details", "summary", "summarise", "summarize", "exactly", "just", "also", "any", "some",
    "one", "two", "first", "second", "other", "again", "then", "than", "so", "if", "not", "no", "yes",
    "ok", "okay", "thanks", "thank", "hey", "hi", "hello", "jarvis", "something", "anything", "know",
    "remember", "recall", "find", "look", "check", "get", "got", "went", "going", "happen", "happened",
    # words that describe the KIND of question, not its subject
    "issue", "issues", "ticket", "tickets", "problem", "problems", "work", "worked", "working", "task",
    "tasks", "deploy", "deployed", "deploying", "deployment", "deployments", "setup", "set", "info",
    "information", "update", "updates", "thing", "things", "stuff", "cluster", "clusters",
    "environment", "env", "version", "fix", "fixed", "fixing", "solve", "solved", "solution",
    "resolve", "resolved", "week", "month", "far", "away", "status", "state", "progress", "repo",
    "notes", "note", "did", "wrong", "cause", "root", "steps", "step", "between", "part", "bit",
    "regarding", "related", "around", "like", "really", "much", "many", "lot", "way", "ways",
}
_TERM_SYNONYMS = {"otel": ["opentelemetry"], "opentelemetry": ["otel"], "k8s": ["kubernetes"],
                  "kubernetes": ["k8s"], "cert": ["certificate", "tls"], "certificate": ["cert", "tls"],
                  "tls": ["ssl", "certificate"], "ssl": ["tls"], "pg": ["postgres"],
                  "postgres": ["postgresql"], "ha": ["high availability"], "db": ["database"]}
_ACK_RE = re.compile(r"^\s*(ok(ay)?|thanks?( you)?|thx|cool|great|nice|perfect|got it|understood|"
                     r"alright|good|super|danke)[\s.!]*$", re.I)


def _lev1(a: str, b: str) -> bool:
    """True if a and b differ by at most one edit (catches typos like 'grafna')."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return edits + (len(b) - j) + (len(a) - i) <= 1


def parse_query(q: str, vocab: dict) -> dict:
    """Turn a question into filters. Deterministic on purpose — this is the part
    that must be right for 'latest' to be right.
      'what is the latest on grafana in dev?'
        → env=dev, product=grafana, intent=latest, terms=[]
      'how did I fix the otel crash on the main cluster'
        → env=maint, terms=[otel, crash]"""
    low = (q or "").lower()
    toks = [t.strip("._-") for t in re.findall(r"[a-z0-9äöüß][a-z0-9äöüß._-]*", low)]
    toks = [t for t in toks if t]
    out = {"env": "", "product": "", "version": "", "version_key": "", "version_prefix": "", "repo": "",
           "intent": "auto", "terms": []}
    used = set()

    for i, t in enumerate(toks):
        canon = _ENV_LOOKUP.get(t)
        if not canon:
            continue
        if t in _ENV_AMBIGUOUS_IN_QUERY:
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            prv = toks[i - 1] if i > 0 else ""
            if not (nxt in ("cluster", "env", "environment", "clusters", "system")
                    or prv in ("in", "on", "at", "from")):
                continue
        out["env"] = canon
        used.add(t)
        break

    products = vocab.get("products", set())
    for t in toks:
        if t in used or t in _STOP or len(t) < 2:
            continue
        n = _PRODUCT_SYNONYMS.get(t, t)
        hit = n if n in products else None
        if not hit and len(t) >= 4:
            for p in products:
                if ((len(t) >= 5 and p.startswith(t)) or (t.startswith(p) and len(p) >= 4)
                        or (len(t) >= 5 and len(p) >= 5 and _lev1(t, p))):
                    hit = p
                    break
        if hit:
            out["product"] = hit
            used.add(t)
            break

    vs = versions_in(low)
    if vs:
        best = max(vs, key=lambda v: v[0])
        out["version"], out["version_key"] = best[1], version_key(best[0])
        out["version_prefix"] = ".".join(f"{x:05d}" for x in best[0])
        used.update(raw for _, raw in vs)

    # A well-known product the repo has NO folder for ("jenkins") is remembered,
    # so the answer can say "nothing about jenkins" instead of guessing.
    out["unknown_products"] = []
    for t in toks:
        n = _PRODUCT_SYNONYMS.get(t, t)
        if n in KNOWN_PRODUCTS and n not in products and n != out["product"]:
            out["unknown_products"].append(t)

    for slug, rid in vocab.get("repos", {}).items():
        if slug and slug in low:
            out["repo"] = rid
            used.add(slug)

    words = set(toks)
    if words & _HISTORY_WORDS:
        out["intent"] = "history"
    elif words & _LIST_WORDS:
        out["intent"] = "list"
    elif words & _LATEST_WORDS or "this week" in low or "last week" in low:
        out["intent"] = "latest"

    for t in toks:
        if (t in used or t in _STOP or t in _ENV_LOOKUP or t in _LATEST_WORDS or t in _LIST_WORDS
                or t in _HISTORY_WORDS or len(t) < 3 or t.isdigit() or versions_in(t)):
            continue
        if t not in out["terms"]:
            out["terms"].append(t)
    return out


def _is_followup(parsed: dict, q: str) -> bool:
    if parsed["env"] or parsed["product"]:
        return False
    low = q.lower()
    if re.search(r"\b(it|that|this|those|these|there|same|again|above|more|further|"
                 r"that one|the other|second one|first one|how did i|what did i|why did|"
                 r"what was the|what exactly)\b", low):
        return True
    return len(q.split()) <= 8


def _fts_query(terms: list) -> str:
    words = []
    for t in terms:
        for w in [t] + _TERM_SYNONYMS.get(t, []):
            for piece in re.findall(r"[a-z0-9äöüß]+", w.lower()):
                if len(piece) >= 2 and piece not in words:
                    words.append(piece)
    return " OR ".join(f'"{w}"*' if len(w) >= 4 else f'"{w}"' for w in words[:16])


SEM_FLOOR = 0.45      # meaning-similarity needed to count at all…
SEM_ALONE = 0.62      # …and to count WITHOUT any keyword match (avoids "always finds something")


def _hybrid_search(question: str, terms: list, allowed_files: set, k: int = 40) -> dict:
    """chunk_id → fused score (keyword BM25 rank + meaning rank, Reciprocal Rank
    Fusion — the same method the existing KB uses).

    Guard: a chunk found ONLY by meaning (no keyword hit) must be a strong match.
    Meaning search always returns *something* vaguely similar; without this guard
    "latest on jenkins" would happily answer about Grafana."""
    if not allowed_files:
        return {}
    kw = []
    fq = _fts_query(terms)
    if fq:
        try:
            with _db() as conn:
                rows = conn.execute(
                    "SELECT f.rowid AS id, c.file_id FROM lr_chunks_fts f JOIN lr_chunks c ON c.id=f.rowid "
                    "WHERE lr_chunks_fts MATCH ? ORDER BY bm25(lr_chunks_fts, 4.0, 2.0, 1.0) LIMIT 400",
                    (fq,)).fetchall()
            kw = [r["id"] for r in rows if r["file_id"] in allowed_files][:k]
        except Exception as e:
            log.warning("Living Repos FTS failed: %s", e)
    sem = []
    qv = embed_texts([" ".join(terms) or question], kind="query")   # embed the TOPIC, not "latest in dev"
    if qv:
        scores = _vector_scores(qv[0], allowed_files)
        kw_set = set(kw)
        sem = [cid for cid, s in sorted(scores.items(), key=lambda kv: -kv[1])
               if s >= (SEM_FLOOR if cid in kw_set else SEM_ALONE)][:k]
    fused = {}
    for rank, cid in enumerate(kw):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    for rank, cid in enumerate(sem):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    return fused


def _load_focus(chat_id):
    if not chat_id:
        return None
    with _db() as conn:
        r = conn.execute("SELECT data FROM lr_chat_focus WHERE chat_id=?", (chat_id,)).fetchone()
    return json.loads(r["data"]) if r else None


def _save_focus(chat_id, focus: dict):
    if not chat_id or not focus:
        return
    with _db() as conn:
        conn.execute("INSERT INTO lr_chat_focus (chat_id, data, updated_at) VALUES (?,?,?) "
                     "ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                     (chat_id, json.dumps(focus), _now()))


def _day(iso: str) -> str:
    return (iso or "")[:10] or "????-??-??"


def retrieve(question: str, chat_id=None, budget_chars: int = 30000) -> dict:
    """Find what the question is about and build the context for the model.
    Returns {context, sources, understood, found, focus}."""
    with _db() as conn:
        repos = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM lr_repos")}
        files = [dict(r) for r in conn.execute(
            "SELECT id, repo_id, path, env, product, item, version, version_key, title, recency, "
            "status, is_text FROM lr_files")]
    empty = {"context": "", "sources": [], "understood": "", "found": False, "focus": None,
             "repos": len(repos)}
    if not repos:
        return empty
    active = [f for f in files if f["status"] == "active"]
    vocab = {"products": {f["product"] for f in active if f["product"]},
             "envs": {f["env"] for f in active if f["env"]},
             "repos": {r["slug"]: rid for rid, r in repos.items()}}
    p = parse_query(question, vocab)
    focus = _load_focus(chat_id)
    pool = files if p["intent"] == "history" else active
    notes = []

    def _filter(e, prod):
        return [f for f in pool if (not e or f["env"] == e) and (not prod or f["product"] == prod)
                and (not p["repo"] or f["repo_id"] == p["repo"])]

    # FOLLOW-UPS. "how did I fix it?" → stay on the item we just discussed.
    # "and in prod?" / "what about kafka?" → keep the half the user didn't change.
    follow = bool(focus) and _is_followup(p, question)
    env, product = p["env"], p["product"]
    if follow:
        env, product = focus.get("env", ""), focus.get("product", "")
        if p["terms"]:
            # A new topic word ("latest otel issue?") that doesn't appear anywhere in
            # the focused area means the user moved on — drop the focus.
            focus_ids = {f["id"] for f in _filter(env, product)}
            if not _hybrid_search(question, p["terms"], focus_ids, k=5):
                follow, env, product = False, "", ""
    elif focus and re.match(r"^\s*(and|what about|how about|same (for|in|on))\b", question.lower()) \
            and len(question.split()) <= 7:
        env = env or focus.get("env", "")
        product = product or focus.get("product", "")

    cand = _filter(env, product)
    if env and not cand:
        for alt in _ENV_EQUIV.get(env, ()):
            alt_c = _filter(alt, product)
            if alt_c:
                notes.append(f"Your repo has no '{env}' folder for this, so I used '{alt}' "
                             "(maint is treated as production).")
                env, cand = alt, alt_c
                break
    if p["version_key"]:
        vc = [f for f in cand if f["version_key"].startswith(p["version_prefix"])]
        if vc:
            cand = vc
        else:
            notes.append(f"No files for version {p['version']} matched; showing what exists.")

    items = {}
    for f in cand:
        it = items.setdefault(f["item"], {"files": [], "recency": "", "env": f["env"],
                                          "product": f["product"], "repo_id": f["repo_id"]})
        it["files"].append(f)
        if f["recency"] > it["recency"]:
            it["recency"] = f["recency"]

    # Relevance (only when the question names a topic beyond env/product)
    rel, chunk_rank = {}, {}
    if p["terms"]:
        by_id = {f["id"]: f for f in cand}
        chunk_rank = _hybrid_search(question, p["terms"], set(by_id))
        if chunk_rank:
            with _db() as conn:
                q = ",".join("?" * len(chunk_rank))
                cmap = {r["id"]: r["file_id"] for r in conn.execute(
                    f"SELECT id, file_id FROM lr_chunks WHERE id IN ({q})", list(chunk_rank))}
            for cid, sc in chunk_rank.items():
                f = by_id.get(cmap.get(cid))
                if f:
                    rel[f["item"]] = max(rel.get(f["item"], 0.0), sc)

    # Asked about a product the repo has no folder for, and it isn't mentioned in
    # any file text either → honest "not found", never a lookalike.
    missing_products = []
    for t in p.get("unknown_products", []):
        fq = _fts_query([t])
        with _db() as conn:
            hit = conn.execute("SELECT 1 FROM lr_chunks_fts WHERE lr_chunks_fts MATCH ? LIMIT 1",
                               (fq,)).fetchone() if fq else None
        if not hit:
            missing_products.append(t)

    selected, why = None, ""
    if missing_products:
        items = {}
        product = missing_products[0]
    elif follow and focus.get("item") in items and not (rel and focus.get("item") not in rel):
        selected, why = focus["item"], "follow-up on the item we were just discussing"
    elif rel:
        ranked = sorted(rel, key=lambda i: -rel[i])
        top = rel[ranked[0]]
        close = [i for i in ranked if rel[i] >= top * (0.6 if p["intent"] == "latest" else 0.85)]
        selected = max(close, key=lambda i: items[i]["recency"])
        why = ("the most recent item that matches " if p["intent"] == "latest" or len(close) > 1
               else "the best match for ") + ", ".join(p["terms"])
    elif items and (env or product or p["repo"] or p["intent"] == "latest"):
        selected = max(items, key=lambda i: items[i]["recency"])
        why = "the most recently edited work item" + (
            f" (nothing mentions {', '.join(p['terms'])} there)" if p["terms"] else "")
        if p["terms"]:
            notes.append(f"Nothing in this area mentions {', '.join(p['terms'])}; showing the most recent work.")

    understood = " · ".join(x for x in [
        f"env={env}" if env else "", f"product={product}" if product else "",
        f"version={p['version']}" if p["version"] else "",
        f"topic={'/'.join(p['terms'])}" if p["terms"] else "",
        f"intent={p['intent']}" if p["intent"] != "auto" else "",
        "follow-up" if follow and selected == (focus or {}).get("item") else ""] if x) or "general question"

    # ── Build the context text ───────────────────────────────────────────────
    parts, sources, used = [], [], 0

    def add(s):
        nonlocal used
        room = budget_chars - used
        if room <= 200:
            return False
        if len(s) > room:
            s = s[:room - 30] + "\n…[cut to fit]\n"
        parts.append(s)
        used += len(s)
        return True

    repo_line = "; ".join(f"{r['name']} (last synced {_day(r['last_sync_at'])})" for r in repos.values())
    add("═══ REPO CONTEXT (searched locally on the Mac) ═══\n"
        f"Repos: {repo_line}\nQuestion understood as: {understood}\n")
    for n in notes:
        add(f"NOTE: {n}\n")

    catalog_src = items
    if not items:   # nothing in the filter → show what DOES exist so the model can say so
        env_part = [f for f in active if not env or f["env"] == env]
        catalog_src = {}
        for f in (env_part or active):
            it = catalog_src.setdefault(f["item"], {"files": [], "recency": "", "env": f["env"],
                                                    "product": f["product"], "repo_id": f["repo_id"]})
            it["files"].append(f)
            it["recency"] = max(it["recency"], f["recency"])
        what = " ".join(x for x in [env, product, p["version"]] if x) or "that"
        add(f"\nNOTHING in the repo matches {what}. Say so plainly. What the repo does contain "
            f"({'in ' + env if env_part else 'overall'}) is listed below.\n")

    if p["intent"] == "list" or not selected:
        cat_n = 40 if p["intent"] == "list" else 12
    else:
        cat_n = 12
    cat = sorted(catalog_src.items(), key=lambda kv: kv[1]["recency"], reverse=True)[:cat_n]
    if cat:
        add("\nCATALOG — work items, newest first (★ = the one the system selected):\n")
        for n, (key, it) in enumerate(cat, 1):
            title = max(it["files"], key=lambda f: f["recency"])["title"]
            vers = sorted({(f["version_key"], f["version"]) for f in it["files"] if f["version"]})
            vtxt = f"  versions: {', '.join(v for _, v in vers)}" if vers else ""
            mark = "★" if key == selected else " "
            add(f" {mark}{n:>2}. [{_day(it['recency'])}] {it['env'] or '-'} / {it['product'] or '-'} / "
                f"{key.rsplit('/', 1)[-1]}  ({len(it['files'])} file{'s' if len(it['files']) != 1 else ''})"
                f"  — {title[:90]}{vtxt}\n")
        same = [k for k, it in cat if k != selected and it["product"] == (items.get(selected) or {}).get("product")
                and it["env"] == (items.get(selected) or {}).get("env")]
        if selected and same:
            add("(If the question could mean one of the other items above — e.g. a second Kafka — "
                "say which one you describe and mention the other.)\n")

    sel_files = []
    if selected:
        it = items[selected]
        sel_files = it["files"]
        vers = sorted({f["version_key"] for f in sel_files if f["version_key"]})
        if p["version_key"]:
            pass
        elif vers and p["intent"] != "history":
            top_v = vers[-1]
            sel_files = [f for f in sel_files if f["version_key"] in ("", top_v)]
        sel_files = sorted(sel_files, key=lambda f: f["recency"], reverse=True)
        vlabel = ""
        if vers:
            vlabel = f" — version {next(f['version'] for f in it['files'] if f['version_key'] == vers[-1])}" \
                     f" is the highest of {len(vers)}"
        add(f"\n★ SELECTED: {it['env'] or '-'} / {it['product'] or '-'} / {selected.rsplit('/', 1)[-1]}{vlabel}\n"
            f"   Folder: {selected}\n   Why: {why}. Last edited {_day(it['recency'])}.\n")
        repo = repos[it["repo_id"]]
        matched = {}           # file_id → its chunks that matched the question, best first
        if chunk_rank:
            with _db() as conn:
                q = ",".join("?" * len(chunk_rank))
                for r in conn.execute(f"SELECT id, file_id, heading, text FROM lr_chunks WHERE id IN ({q})",
                                      list(chunk_rank)):
                    matched.setdefault(r["file_id"], []).append(r)
            for v in matched.values():
                v.sort(key=lambda r: -chunk_rank[r["id"]])
        for f in sel_files:
            abs_path = REPOS_DIR / repo["slug"] / f["path"]
            text, is_text = _read_text(abs_path, f["path"]) if abs_path.exists() else ("", False)
            status = "  [REMOVED from the latest drop — historical]" if f["status"] == "removed" else ""
            header = f"\n--- FILE: {f['path']}  (last edited {_day(f['recency'])}){status} ---\n"
            body = text if is_text and text.strip() else "(binary or empty file — name only)"
            if len(body) > budget_chars - used and matched.get(f["id"]):
                # Too big to send whole: send the sections that match the question
                body = "(large file — the sections matching the question:)\n" + "\n…\n".join(
                    (f"[{r['heading']}]\n" if r["heading"] else "") + r["text"] for r in matched[f["id"]][:6])
            if not add(header + body.strip() + "\n"):
                break
            sources.append({"path": f["path"], "repo": repo["name"], "date": _day(f["recency"])})

    # Other matching excerpts (outside the selected item)
    if chunk_rank and used < budget_chars - 800:
        sel_ids = {f["id"] for f in sel_files}
        best = sorted(chunk_rank.items(), key=lambda kv: -kv[1])[:12]
        with _db() as conn:
            q = ",".join("?" * len(best))
            rows = {r["id"]: r for r in conn.execute(
                f"SELECT c.id, c.heading, c.text, f.path, f.recency, f.id AS fid FROM lr_chunks c "
                f"JOIN lr_files f ON f.id=c.file_id WHERE c.id IN ({q})", [b[0] for b in best])}
        extra = [rows[c] for c, _ in best if c in rows and rows[c]["fid"] not in sel_ids][:5]
        if extra:
            add("\nOTHER MATCHING EXCERPTS (older or different items — use only if relevant):\n")
            for r in extra:
                if not add(f"\n--- {r['path']}  (last edited {_day(r['recency'])})"
                           f"{' › ' + r['heading'] if r['heading'] else ''} ---\n{r['text'][:1500]}\n"):
                    break

    new_focus = None
    if selected:
        it = items[selected]
        new_focus = {"env": it["env"], "product": it["product"], "item": selected,
                     "repo_id": it["repo_id"]}
    return {"context": "".join(parts), "sources": sources, "understood": understood,
            "found": bool(selected), "focus": new_focus, "repos": len(repos)}


@bp.route("/api/repos/ask", methods=["GET"])
def api_repo_ask():
    """Debug/accuracy check: what would the local mode pick for this question?
    (No AI call — shows exactly which item and files would be sent.)"""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "add ?q=your question"}), 400
    r = retrieve(q, None)
    return jsonify({"understood": r["understood"], "found": r["found"],
                    "selected": r["focus"], "sources": r["sources"], "context": r["context"]})


# ═════════════════════════════════════════════════════════════════════════════
#  OLLAMA — local models
# ═════════════════════════════════════════════════════════════════════════════
_TAGS = {"t": 0.0, "data": None}
_EMBED_NAME_HINTS = ("embed", "nomic", "bge", "minilm", "e5-", "gte-", "snowflake", "rerank", "mxbai")
# Best first. Matched as substrings against what you have installed.
_PREFERRED = ("qwen3.6", "gemma4:26b", "gemma4:31b", "qwen3.5:27b", "gemma4", "qwen3.5", "qwen3:14b",
              "qwen3", "gpt-oss", "gemma3:12b", "gemma3", "qwen2.5:14b", "qwen2.5", "llama3.1",
              "llama3", "mistral", "phi")


def ollama_models(force: bool = False):
    """Installed Ollama models (list of names), or None if Ollama isn't running."""
    if not force and time.time() - _TAGS["t"] < 15 and _TAGS["data"] is not None:
        return _TAGS["data"]
    try:
        data = _http_json(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        names = sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        names = None
    _TAGS.update(t=time.time(), data=names)
    return names


def is_embed_model(name: str) -> bool:
    return any(h in name.lower() for h in _EMBED_NAME_HINTS)


def chat_models():
    return [m for m in (ollama_models() or []) if not is_embed_model(m)]


def pick_local_model() -> str:
    installed = chat_models()
    want = _setting("local_model", "")
    if want and want in installed:
        return want
    for pref in _PREFERRED:
        for m in installed:
            if pref in m:
                return m
    return installed[0] if installed else ""


def _pretty(name: str) -> str:
    base, _, tag = name.partition(":")
    tag = "" if tag == "latest" else tag.upper()
    return f"{base[:1].upper()}{base[1:]} {tag}".strip()


def local_model_entries() -> list:
    """Entries for the model dropdown (provider 'ollama', always free)."""
    best = pick_local_model()
    out = []
    for m in chat_models():
        out.append({"id": f"ollama:{m}", "name": f"{_pretty(m)} · local", "group": "Local — free & private",
                    "provider": "ollama", "desc": "Runs on your Mac. Answers from your living repos.",
                    "in": 0.0, "out": 0.0, "free": True, "ctx": "local", "recommended": m == best})
    return out


def _ollama_chat_stream(model: str, messages: list, num_ctx: int):
    """Yield ('text', piece) … then ('done', stats). Raises RuntimeError on errors."""
    body = {"model": model, "messages": messages, "stream": True, "think": False,
            "keep_alive": "30m", "options": {"num_ctx": num_ctx, "temperature": 0.2}}

    def _open(b):
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=600)

    try:
        resp = _open(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        if e.code == 400 and "think" in detail.lower():   # model without a thinking switch
            body.pop("think", None)
            resp = _open(body)
        elif e.code == 404:
            raise RuntimeError(f"Model '{model}' is not installed. On the Mac run:  ollama pull {model}")
        else:
            raise RuntimeError(f"Ollama HTTP {e.code}: {detail}")
    except urllib.error.URLError:
        raise RuntimeError("Ollama is not running. Start it (open the Ollama app, or run: ollama serve).")
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            piece = (obj.get("message") or {}).get("content", "")
            if piece:
                yield "text", piece
            if obj.get("done"):
                yield "done", {"tin": obj.get("prompt_eval_count", 0) or 0,
                               "tout": obj.get("eval_count", 0) or 0}


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL CHAT — called by app.py for any chat whose model is "ollama:<name>"
# ═════════════════════════════════════════════════════════════════════════════
LOCAL_SYSTEM_PROMPT = """You are Jarvis in LOCAL KNOWLEDGE MODE — Naveen's personal secretary for his own DevOps work. You run fully offline on his MacBook. Naveen is a DevOps engineer (Kubernetes, Nexus, Grafana, Kafka, Keycloak, HAProxy, LDAP). He often asks you during meetings, so be fast and clear.

The system has ALREADY searched his work repository and put the relevant parts in REPO CONTEXT inside his message.

RULES
1. REPO CONTEXT is the truth about his work. Answer from it. Never invent hostnames, versions, dates, commands, ticket names or fixes that are not in it.
2. "Latest" has ALREADY been decided by the system using file dates and version numbers. The item marked ★ SELECTED is the one to talk about. Do not pick a different item as "latest" and do not second-guess the dates.
3. Explain, don't recite. Start with 2-4 plain-English sentences he could say out loud in a meeting: what it was, which environment/cluster, when (the date shown), and the outcome. Then the details he asked for (the issue, the root cause, how he fixed it, key commands/config) — short and concrete. Quote a command or config line only when it matters.
4. If the catalog shows several similar items (for example two Kafka setups in dev), say which one you are describing and mention the other one exists.
5. If REPO CONTEXT says nothing matches, say so plainly ("I can't find that in your repo") and suggest what IS there. Never fill the gap with a guess.
6. If there is no REPO CONTEXT and the question is general knowledge, answer it and add "(general knowledge — not from your repo)".
7. If a NOTE says a different environment was used, mention it in one line.
8. Plain English, no preamble, no filler."""


def _content_to_text(content) -> str:
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return content
        content = parsed
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content or "")


def _strip_footer(text: str) -> str:
    return text.split("\n\n---\n📁", 1)[0].split("\n\n---\n🏠", 1)[0]


def _read_attachments(files) -> tuple:
    blocks, names = [], []
    for f in files:
        name = f.filename or "file"
        raw = f.read()
        ext = _ext(name)
        names.append(name)
        if ext in DOC_EXTS:
            txt = _D["extract_text"](name, raw) or ""
        elif ext in BINARY_EXTS or b"\x00" in raw[:8192]:
            blocks.append(f"[Attached {name}: binary/image — the local model can't read it. Use Cloud mode for images.]")
            continue
        else:
            txt = raw.decode("utf-8", errors="replace")
        blocks.append(f"Attached file {name}:\n```\n{txt[:20000]}\n```")
    return "\n\n".join(blocks), names


def handle_local_message(chat_id: str, model_id: str, apply_auto_title=None):
    """Answer one message in a local chat. Same request/streaming format as the
    cloud path, so the existing UI works unchanged. Cost is always $0."""
    model = model_id.split(":", 1)[1] if ":" in model_id else model_id

    if request.content_type and "multipart" in request.content_type:
        text = (request.form.get("message") or "").strip()
        files = request.files.getlist("files")
    else:
        text = ((request.get_json(silent=True) or {}).get("message") or "").strip()
        files = []
    attach_text, attach_names = _read_attachments(files) if files else ("", [])
    if not text and not attach_text:
        return jsonify({"error": "Message is empty"}), 400

    # Check Ollama BEFORE saving anything, so a failed turn leaves no orphan message
    installed = ollama_models(force=True)
    if installed is None:
        return jsonify({"error": "Local mode needs Ollama, and it isn't running on the Mac. "
                                 "Open the Ollama app (or run: ollama serve), then try again."}), 503
    if model not in installed:
        return jsonify({"error": f"Local model '{model}' isn't installed. On the Mac run:  "
                                 f"ollama pull {model}   (or pick another model in Settings → Local AI)"}), 400

    user_db = text + (("\n\n" + attach_text) if attach_text else "")
    with _db() as conn:
        # History is read BEFORE saving the new message (timestamps have 1-second
        # resolution, so "everything except the newest row" could pick wrongly).
        hist_rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? "
                                 "ORDER BY created_at, rowid", (chat_id,)).fetchall()
        conn.execute("INSERT INTO messages (id, chat_id, role, content, tokens_in, tokens_out, model, created_at) "
                     "VALUES (?,?, 'user', ?, 0, 0, ?, ?)", (str(uuid.uuid4()), chat_id, user_db, model_id, _now()))
    if not hist_rows and apply_auto_title and text:
        try:
            apply_auto_title(chat_id, text)
        except Exception as e:
            log.warning("auto-title failed: %s", e)

    num_ctx = int(_setting("local_num_ctx", str(DEFAULT_NUM_CTX)) or DEFAULT_NUM_CTX)
    budget = max(6000, int(num_ctx * 2.6) - 12000)

    history = []
    for r in hist_rows[-8:]:
        t = _content_to_text(r["content"])
        if not t.strip() or t.startswith(("error:", "❌")):
            continue
        history.append({"role": r["role"], "content": _strip_footer(t)[:2500]})

    retrieval = {"context": "", "sources": [], "found": False, "focus": None, "understood": "", "repos": 0}
    if text and not _ACK_RE.match(text):
        try:
            retrieval = retrieve(text, chat_id, budget_chars=budget)
        except Exception as e:
            log.exception("Living Repos retrieval failed")
            retrieval["context"] = f"(Repo search failed: {e})"
    if retrieval["context"]:
        prompt = f"{retrieval['context']}\n═══ END OF REPO CONTEXT ═══\n\nNaveen's question: {text}"
    elif retrieval.get("repos", 0) == 0 and text and not _ACK_RE.match(text):
        prompt = ("(No living repo has been synced yet, so there is no REPO CONTEXT. If this is about "
                  f"his own work, tell him to drop his repo into LLM Dropbox first.)\n\nNaveen's question: {text}")
    else:
        prompt = text
    if attach_text:
        prompt += "\n\n" + attach_text
    messages = [{"role": "system", "content": LOCAL_SYSTEM_PROMPT}] + history + [{"role": "user", "content": prompt}]
    log.info("Local answer: model=%s ctx=%d chars understood=[%s] sources=%d",
             model, len(retrieval["context"]), retrieval["understood"], len(retrieval["sources"]))

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def gen():
        q = queue.Queue()

        def run():
            try:
                for kind, val in _ollama_chat_stream(model, messages, num_ctx):
                    q.put((kind, val))
            except Exception as e:
                q.put(("error", str(e)))
            q.put(("end", None))

        threading.Thread(target=run, daemon=True).start()
        full, stats = "", {"tin": 0, "tout": 0}
        while True:
            try:
                kind, val = q.get(timeout=8)
            except queue.Empty:
                yield ": ping\n\n"           # keeps Cloudflare from closing a slow first token
                continue
            if kind == "text":
                full += val
                yield sse({"type": "text", "text": val})
            elif kind == "done":
                stats = val
            elif kind == "error":
                yield sse({"type": "error", "text": val})
                return
            elif kind == "end":
                break
        if not full.strip():
            yield sse({"type": "error", "text": "The local model returned an empty answer."})
            return

        footer = ""
        if retrieval["sources"]:
            seen, lines = set(), []
            for s in retrieval["sources"][:6]:
                if s["path"] not in seen:
                    seen.add(s["path"])
                    lines.append(f"`{s['path']}` ({s['date']})")
            footer += "\n\n---\n📁 **From your repo:** " + " · ".join(lines)
        footer += f"\n\n---\n🏠 Local · {model} · $0.00" + (
            f" · understood: {retrieval['understood']}" if retrieval["understood"] else "")
        yield sse({"type": "text", "text": footer})
        full += footer

        with _db() as conn:
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, tokens_in, tokens_out, model, created_at, cost_usd) "
                "VALUES (?,?, 'assistant', ?, ?, ?, ?, ?, 0.0)",
                (str(uuid.uuid4()), chat_id, full, stats["tin"], stats["tout"], model_id, _now()))
            conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now(), chat_id))
        if retrieval.get("focus"):
            _save_focus(chat_id, retrieval["focus"])
        try:
            _D["export_chat_txt"](chat_id)
        except Exception as e:
            log.warning("chat.txt export failed: %s", e)
        yield sse({"type": "done", "tokens_in": stats["tin"], "tokens_out": stats["tout"],
                   "cost": 0, "free": True, "local": True})

    def with_first_byte(g):
        yield ": keep-alive\n\n"
        yield from g

    return Response(stream_with_context(with_first_byte(gen())), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL AI STATUS + SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
@bp.route("/api/local/status", methods=["GET"])
def api_local_status():
    installed = ollama_models(force=True)
    emb = _embed_model()
    with _db() as conn:
        pending = conn.execute("SELECT COUNT(*) FROM lr_chunks WHERE embedding IS NULL OR emb_model IS NOT ?",
                               (emb,)).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM lr_chunks").fetchone()[0]
    if installed is not None and pending and not _EMB["running"]:
        _kick_embeddings()
    embed_ready = bool(installed) and any(m.split(":")[0] == emb.split(":")[0] for m in installed)
    return jsonify({
        "running": installed is not None,
        "url": OLLAMA_URL,
        "chat_models": [m for m in (installed or []) if not is_embed_model(m)],
        "embed_models": [m for m in (installed or []) if is_embed_model(m)],
        "selected": pick_local_model() if installed else "",
        "saved_choice": _setting("local_model", ""),
        "embed_model": emb,
        "embed_ready": embed_ready,
        "chunks_total": total,
        "chunks_pending": pending,
        "embedding_running": _EMB["running"],
        "embedding_error": _EMB["last_error"],
        "num_ctx": int(_setting("local_num_ctx", str(DEFAULT_NUM_CTX)) or DEFAULT_NUM_CTX),
        "suggest": {"chat": "qwen3.5:9b", "chat_big": "gemma4:26b", "embed": DEFAULT_EMBED},
    })


@bp.route("/api/local/settings", methods=["POST"])
def api_local_settings():
    data = request.get_json(silent=True) or {}
    if "local_model" in data:
        _D["save_setting"]("local_model", str(data["local_model"] or "").strip()[:120])
    if data.get("embed_model"):
        _D["save_setting"]("local_embed_model", str(data["embed_model"]).strip()[:120])
        _VEC["stamp"] += 1
        _kick_embeddings()
    if data.get("num_ctx"):
        try:
            n = max(4096, min(131072, int(data["num_ctx"])))
            _D["save_setting"]("local_num_ctx", str(n))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "selected": pick_local_model()})
