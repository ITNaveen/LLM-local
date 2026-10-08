
# LocalLLM — Personal Claude AI
### Built for Naveen | Token-optimised | 100% local data | DevOps-focused

---

## Local knowledge mode + Living Repos (October 2026)

Your work repo becomes a private knowledge base that a **local** model answers
from — on the Mac, no internet, **$0 per question**. Cloud mode (Claude/OpenAI)
is unchanged and optional.

### One-time setup on the Mac
```bash
brew install ollama            # or install the Ollama app from ollama.com
ollama pull qwen3.5:9b         # the answer model (~6 GB; fine on 24 GB RAM)
ollama pull nomic-embed-text   # the search model (~270 MB) — you may already have it
```
With 48 GB RAM you can use a bigger, smarter model instead: `ollama pull gemma4:26b`,
then pick it in **Settings → Local AI**. `LocalLLM.command` starts Ollama for you.

### Every week
1. Open **LLM Dropbox** → drag your repo folder in (or press **📁 Folder**).
2. First time: choose **🧠 Living repo** and keep the folder name as the repo name.
3. Next weeks: drop the same folder again (or press **⟳ Sync** on the repo).
   You get a preview — *new / changed / unchanged / missing* — and only the new
   and changed files are uploaded when you press **Apply sync**.
4. Switch the top bar to **🏠 Local** and ask:
   *"what's the latest on grafana in dev?"* → *"how did I fix it?"*

> **Folder location:** keep `local-llm-app`, `local-llm-db` and `local-llm-dropbox` side by side —
> e.g. all three in `~/Documents/LLM/` (paths below use `~/Documents/`). The app and
> `LocalLLM.command` find them wherever they are; `LOCALLLM_ROOT=/path` overrides.

### The safety rules (the "lock")
| What happens | Where |
|---|---|
| Current files of each repo (browsable in Finder) | `~/Documents/local-llm-dropbox/living-repos/<repo>/` |
| A changed file's previous version is kept, never overwritten | `~/Documents/local-llm-db/repo-history/<repo>/<date>/` |
| A file missing from a new drop is **kept** and flagged "removed" | stays in place |
| Uploads wait in staging; the repo changes only when the whole sync commits | `~/Documents/local-llm-db/repo-staging/` |
| Repos are **locked**. Delete = unlock + type the name + it's only *moved* | `~/Documents/local-llm-db/repo-trash/` |
| Every repo file (and old version) carries macOS's **Locked** flag, so Finder won't bin it without an extra confirmation | Finder → Get Info → "Locked" |
| Files keep their original **Date Modified** from your work laptop | Finder shows the real dates |
| One readable entry per sync: what was added, changed, missing | `~/Documents/local-llm-db/repo-history/<repo>/SYNC-LOG.txt` |

Press **✓** next to a repo to re-check every file against the fingerprint taken
when it arrived — proof the Mac copy is exactly what you dropped.

### Reading your chats in Finder
`~/Documents/local-llm-db/chats/` mirrors the sidebar, readable like a book:
```
chats/
  INDEX - all chats.txt                          ← every chat, grouped like the sidebar
  Unfiled/
    2026-10-05 · make me statefulset for Postgres__7b126e90/chat.txt
  Kafka/                                          ← your UI folders, same names
    2026-10-05 · Kafka mTLS KafkaUser cert rotation__5c94832f/chat.txt
    _Trash_/                                      ← chats deleted in the app (restorable)
  _Orphan/                                        ← "Delete forever" chats, kept as a record
```
`chat.txt` is the conversation word for word (commands and code included). The
folder follows the chat when you rename it or move it between UI folders. The
`__7b126e90` tail is the chat's id — the link back to `chats.db`, which stays the
master copy. Older folders are reorganised automatically on the first start.

### How "latest" is decided (code, not AI guesswork)
* The path is read as data: `dev/grafana/otel-collector-crash/notes.md`
  → environment **dev**, product **grafana**, work item **otel-collector-crash**.
  Product-first layouts (`nexus/prod/3.95/…`) work too.
* Versions are compared as numbers: 3.95 > 3.68, and 3.100 > 3.95. "Latest Nexus"
  only sends the highest version's files — 3.68 is never mixed in unless you ask for it.
* A file's date = the newer of its modified time on your laptop and any date written
  in its name/header (`2026-09-26-…`, `Datum: 26.09.2026`).
* The code picks the item and marks it ★; the model only explains it.
* "prod" falls back to "maint" (and back) when one of them has no folder — the answer says so.
* Asking about something that isn't in the repo gives "I can't find that", not a lookalike.

**Check its accuracy without asking the AI:** the *Check what it would pick…* box
under Living repos shows which item and files a question would use.
Developers: `python3 -m unittest tests/test_living_repos.py -v` runs the whole
weekly-drop scenario offline.

---

## What This Is
A self-hosted Claude AI interface that:
- Runs on **http://localhost:5000** on your MacBook M4 Pro
- Stores ALL chats in `~/Documents/local-llm/chats.db` (SQLite)
- Uses **85–95% fewer tokens** than a naive LLM frontend
- Works on mobile (Android/iPhone) on the same WiFi
- Full file upload support (code, logs, YAML, PDF, images)
- Live streaming responses with cost tracker

---

## Token Conservation — How It Works

Your boss's trick is **rolling context compression**. Here's exactly what this app does:

| Scenario | Naive LLM | LocalLLM |
|---|---|---|
| 50-message chat, next question | ~80,000 tokens sent | ~3,000 tokens sent |
| System prompt (every request) | ~200 tokens | ~0 tokens (cached) |
| 500KB log file uploaded | All 500KB | 8KB (smart truncation) |
| **Net saving** | baseline | **~93% reduction** |

### The 5 mechanisms:
1. **Rolling compression** — After 10 messages, old history compressed into ~400-token summary using cheapest Haiku model
2. **Raw window** — Only last 6 messages sent in full (captures current working context)
3. **Anthropic prompt caching** — System prompt marked `cache_control: ephemeral` → saved on Anthropic's side, charged at 10% rate
4. **Smart file truncation** — Files over 8KB: first 4KB + last 4KB sent (most relevant parts)
5. **Haiku for summaries** — Internal compression uses `claude-haiku-3-0` at $0.25/MTok instead of Sonnet

### Cost example for your usage:
- 1M tokens = €5 credit
- With 93% saving: effectively **14M tokens for €5**
- Sonnet 3.5: $3/MTok input — that's ~**4,666 DevOps questions per €5**

---

## Quick Deploy on macOS M4 Pro

### Step 1 — Copy files
```bash
# Create app directory
mkdir -p ~/Documents/local-llm-app
# Copy all files there: app.py, templates/index.html, requirements.txt, start.sh
```

### Step 2 — Install dependencies
```bash
pip3 install flask flask-cors anthropic --break-system-packages
```

### Step 3 — Run
```bash
cd ~/Documents/local-llm-app
python3 app.py
# → Open http://localhost:5000
```

Or double-click `start.sh` (make executable first):
```bash
chmod +x start.sh
./start.sh
```

### Step 4 — Add API Key
- Open http://localhost:5000
- Click **Settings ⚙️** (bottom-left)
- Paste your Anthropic API key (starts with `sk-ant-`)
- Click Save

---

## File Structure
```
~/Documents/local-llm/          ← All data lives here
├── chats.db                    ← SQLite: all chats, messages, settings
├── uploads/                    ← Uploaded files
└── chats/                      ← One folder per chat UUID
    ├── <chat-id-1>/
    ├── <chat-id-2>/
    └── ...

~/Documents/local-llm-app/      ← App source code
├── app.py                      ← Flask backend
├── templates/
│   └── index.html              ← Full UI
├── requirements.txt
└── start.sh
```

---

## Use on Phone (same WiFi)

Find your Mac's local IP:
```bash
ipconfig getifaddr en0
# e.g. 192.168.1.42
```

Open on phone: `http://192.168.1.42:5000`

The UI is fully mobile-responsive.

---

## Recommended Models by Task

| Task | Model | Why |
|---|---|---|
| K8s/Helm debugging | Sonnet 3.5 | Best code understanding |
| Log analysis | Haiku 3.5 | Cheap, fast pattern recognition |
| Complex arch decisions | Opus 3.5 | Deep reasoning |
| Keycloak/OIDC config | Sonnet 3.5 | Well-trained on auth topics |
| Quick bash scripts | Haiku 3 | Cheapest, instant |
| Large codebase review | Opus 4.5 | Max context + intelligence |

---

## Auto-start on Mac Login (optional)

Create a LaunchAgent:
```bash
cat > ~/Library/LaunchAgents/com.localllm.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.localllm</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/Users/YOUR_USERNAME/Documents/local-llm-app/app.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
EOF

# Replace YOUR_USERNAME, then:
launchctl load ~/Library/LaunchAgents/com.localllm.plist
```

---

## Future: Custom Domain (for company laptop access)

Once you want to expose this on your network with a domain:

```bash
# Install nginx
brew install nginx

# Add to nginx.conf:
server {
    listen 80;
    server_name llm.yourcompany.local;
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_buffering off;  # Critical for SSE streaming
    }
}
```

Then add `llm.yourcompany.local` to your company DNS or local `/etc/hosts`.

---

## Security Notes
- API key stored in local SQLite only — never logged or sent anywhere except `api.anthropic.com`
- All chat data stays on your Mac in `~/Documents/local-llm/`
- No telemetry, no analytics, no cloud sync
- For production network exposure: add HTTPS + basic auth in nginx config
=======
# LLM-local
>>>>>>> 3a8623ac25005a62cf0f81c7c5495578f47aaabe
