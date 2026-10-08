#!/usr/bin/env python3
# =============================================================================
#
#   CREDIT UPDATE  —  LocalLLM Balance Setter
#
#   USE THIS WHEN:  You recharged your Anthropic account and want the tool's
#                   "Remaining Credit" to match what the Console shows.
#
#   YOU ONLY EDIT THE TWO LINES IN THE BOX BELOW.  NOTHING ELSE.
#
# =============================================================================
#
#   HOW IT WORKS (plain English):
#
#   STEP 1 - Type the dollar amount your Anthropic Console shows right now,
#            in the CONSOLE_BALANCE line.  Just the number, no dollar sign.
#            Example: Console says US 24.68  ->  write   24.68
#
#   STEP 2 - Change the SWITCH line to say   APPLY   (instead of OFF).
#
#   STEP 3 - Double-click your normal desktop start file. The balance updates
#            ONCE, then this switch turns itself back to OFF automatically.
#            So it can never reset your balance by accident on later starts.
#
#   Next recharge: come back here, change the number, set SWITCH to APPLY,
#   double-click start again. That's it.
#
# =============================================================================


# vvvvvvvvvvvvvvvvvvvvvv   EDIT THESE TWO LINES ONLY   vvvvvvvvvvvvvvvvvvvvvvvvv

CONSOLE_BALANCE = 9.19                          # STEP 1: your Console balance

SWITCH = "OFF"     #@arm@                        # STEP 2: "APPLY" or "OFF"

# ^^^^^^^^^^^^^^^^^^^^^^   EDIT THESE TWO LINES ONLY   ^^^^^^^^^^^^^^^^^^^^^^^^^


# =============================================================================
#   DO NOT EDIT ANYTHING BELOW THIS LINE
# =============================================================================

import os
import sqlite3
import sys
import re

# Same folder logic as app.py: local-llm-db next to this app's folder
# (e.g. ~/Documents/LLM/), LOCALLLM_ROOT overrides, ~/Documents is the fallback.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("LOCALLLM_ROOT"):
    _ROOT = os.path.expanduser(os.environ["LOCALLLM_ROOT"])
elif os.path.isdir(os.path.join(_PARENT, "local-llm-db")):
    _ROOT = _PARENT
else:
    _ROOT = os.path.expanduser("~/Documents")
DB_PATH = os.path.join(_ROOT, "local-llm-db", "chats.db")


def disarm_self():
    """Flip the SWITCH line back to OFF, using the @arm@ marker so only the
    real code line is touched (never the comments)."""
    try:
        me = os.path.abspath(__file__)
        with open(me, "r", encoding="utf-8") as f:
            src = f.read()
        new = re.sub(
            r'SWITCH = "OFF"     #@arm@',
            'SWITCH = "OFF"     #@arm@',
            src,
            count=1,
        )
        with open(me, "w", encoding="utf-8") as f:
            f.write(new)
    except Exception as e:
        print(f"   (note: could not auto-disarm switch: {e})")


def main():
    sw = (SWITCH or "").strip().upper()

    if sw != "APPLY":
        print("Credit update: SWITCH is OFF - nothing changed. (This is normal.)")
        return

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    db_cost_row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS c FROM chats"
    ).fetchone()
    db_cost = float(db_cost_row["c"] or 0.0)

    lt_row = conn.execute(
        "SELECT value FROM settings WHERE key='lifetime_cost'"
    ).fetchone()
    lt_cost = float(lt_row["value"]) if lt_row else 0.0

    total_cost = max(db_cost, lt_cost)

    # remaining = starting_credit - total_cost  ==  CONSOLE_BALANCE
    # so starting_credit = CONSOLE_BALANCE + total_cost
    new_starting = round(CONSOLE_BALANCE + total_cost, 6)

    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('starting_credit', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(new_starting),),
    )
    conn.commit()
    conn.close()

    print("=" * 55)
    print("CREDIT UPDATED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Console balance you entered : US$ {CONSOLE_BALANCE:.2f}")
    print(f"  Already spent (from DB)     : US$ {total_cost:.4f}")
    print(f"  New starting_credit anchor  : US$ {new_starting:.4f}")
    print(f"  Tool will now show          : US$ {CONSOLE_BALANCE:.2f} remaining")
    print("=" * 55)

    disarm_self()
    print("Switch automatically set back to OFF. Safe to run start anytime.")
    print()


if __name__ == "__main__":
    main()
