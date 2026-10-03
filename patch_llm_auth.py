#!/usr/bin/env python3
"""
Adds HTTP Basic Auth to app.py using Flask's before_request hook, gating
EVERY route (including static files) behind a username/password check.

Reads the SAME credential files as the translator, so one login works
for both apps:
  ~/Documents/jarvis-auth/username.txt
  ~/Documents/jarvis-auth/password.txt
"""

path = "app.py"

with open(path, "r") as f:
    content = f.read()

changes = []

# 1. Add imports needed for Basic Auth
old_import = "from flask_cors import CORS"
new_import = (
    "from flask_cors import CORS\n"
    "import secrets as _secrets\n"
    "import base64 as _base64"
)
if old_import in content and "_secrets" not in content:
    content = content.replace(old_import, new_import, 1)
    changes.append(("OK", "Added auth-related imports"))
else:
    changes.append(("SKIPPED", "Imports (already present or pattern not found)"))

# 2. Add the before_request auth check, right after the Flask app is created
old_app_line = 'app = Flask(__name__, static_folder="static", template_folder="templates")'
auth_block = '''app = Flask(__name__, static_folder="static", template_folder="templates")

# ── HTTP Basic Auth (gates every route, including static files) ────────────
_AUTH_DIR = Path.home() / "Documents" / "jarvis-auth"
_AUTH_USER = (_AUTH_DIR / "username.txt").read_text().strip()
_AUTH_PASS = (_AUTH_DIR / "password.txt").read_text().strip()

@app.before_request
def _require_basic_auth():
    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() == "basic":
                decoded = _base64.b64decode(credentials).decode("utf-8")
                username, _, password = decoded.partition(":")
                user_ok = _secrets.compare_digest(username, _AUTH_USER)
                pass_ok = _secrets.compare_digest(password, _AUTH_PASS)
                if user_ok and pass_ok:
                    return None  # credentials OK, let the request through
        except Exception:
            pass
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Jarvis LLM"'},
    )'''

if old_app_line in content and "_require_basic_auth" not in content:
    content = content.replace(old_app_line, auth_block, 1)
    changes.append(("OK", "Added before_request auth check"))
else:
    changes.append(("SKIPPED", "Auth block (already present or pattern not found)"))

with open(path, "w") as f:
    f.write(content)

print()
for status, desc in changes:
    print(f"{status}: {desc}")
print()
print("Done.")
