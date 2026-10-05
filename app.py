#!/usr/bin/env python3
"""
LocalLLM — Production-grade personal AI for Naveen Thapliyal
Claude (Anthropic) API
Token-conservative: rolling compression + prompt caching + smart windowing
All data stored locally in ~/Documents/local-llm-db/
"""

import os
import json
import uuid
import sqlite3
import base64
import re
import hashlib
import logging
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import anthropic
import repo_kb   # Living Repos + local (Ollama) knowledge mode — see repo_kb.py

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("localllm")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & APP INIT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path.home() / "Documents" / "local-llm-db"
DB_PATH    = BASE_DIR / "chats.db"
UPLOAD_DIR = BASE_DIR / "uploads"
CHATS_DIR  = BASE_DIR / "chats"
DROP_DIR   = Path.home() / "Documents" / "local-llm-dropbox"

for d in [BASE_DIR, UPLOAD_DIR, CHATS_DIR, DROP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    {"id": "claude-sonnet-4-6",         "name": "Sonnet 4.6", "group": "Claude 4", "provider": "claude", "desc": "Smart & fast — best for most tasks",   "in": 3.0, "out": 15.0, "free": False, "ctx": "1M"},
    {"id": "claude-haiku-4-5-20251001", "name": "Haiku 4.5",  "group": "Claude 4", "provider": "claude", "desc": "Fastest & cheapest — quick questions",  "in": 1.0, "out": 5.0,  "free": False, "ctx": "200K"},
    {"id": "claude-opus-4-7",           "name": "Opus 4.7",   "group": "Claude 4", "provider": "claude", "desc": "Most powerful — complex reasoning",      "in": 5.0, "out": 25.0, "free": False, "ctx": "1M"},
    {"id": "claude-opus-4-6",           "name": "Opus 4.6",   "group": "Claude 4", "provider": "claude", "desc": "Powerful — deep analysis & long context", "in": 5.0, "out": 25.0, "free": False, "ctx": "1M"},
    {"id": "gpt-5.4",                    "name": "GPT-5.4",      "group": "OpenAI", "provider": "openai", "desc": "OpenAI workhorse — coding & pro work, ~Sonnet price", "in": 2.75, "out": 16.5, "free": False, "ctx": "1M"},
    {"id": "gpt-5.4-mini",               "name": "GPT-5.4 mini", "group": "OpenAI", "provider": "openai", "desc": "Strong mini — ~4x cheaper than Sonnet",          "in": 0.825,"out": 4.95, "free": False, "ctx": "400K"},
    {"id": "o4-mini",                    "name": "o4-mini",      "group": "OpenAI", "provider": "openai", "desc": "Reasoning model — ~6x cheaper, best for debugging", "in": 0.55, "out": 2.2,  "free": False, "ctx": "200K"},
    {"id": "gpt-5.5",                    "name": "GPT-5.5",      "group": "OpenAI", "provider": "openai", "desc": "OpenAI flagship — most capable, pricier than Sonnet", "in": 5.5,  "out": 33.0, "free": False, "ctx": "1M"},
]

DEFAULT_MODEL = "claude-sonnet-4-6"
MODEL_MAP     = {m["id"]: m for m in MODELS}

# Haiku follow-up routing (see guardrails at the routing block in send flow).
# DISABLED June 11: the two-breakpoint cache fix solved the cost problem on its
# own (4-5¢ → 1.8¢ per question via warm cache). Routing's only effect now is
# downgrading some questions to Haiku, which broke on tougher reasoning. The
# saving it gave (~0.3-0.5¢ on short follow-ups) is not worth the quality drop.
# Flip back to True only if cost becomes a problem again AND quality on Haiku
# turns is acceptable for the specific question types being routed.
HAIKU_FOLLOWUP_ROUTING = False
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  (cached via Anthropic prompt caching — saves ~90% tokens)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Jarvis — Naveen's dedicated DevOps mentor at optadata, Krefeld.

YOUR KNOWLEDGE: World-class. Linux kernel internals, Kubernetes, networking, PKI/TLS, databases, CI/CD, Java, Python, security — everything at expert level.

YOUR JOB — TWO SEPARATE LAYERS, NEVER COLLAPSE THEM:
  • REASON like a battle-scarred senior SRE / staff engineer with 15+ years on these exact systems. THINK at the deepest level the problem demands — derive the COMPLETE correct production-grade solution, hold every constraint at once, trace the full dependency chain to its end-state. Naveen's tickets are senior-level tickets. Your reasoning must operate at senior level or above. NEVER cap the depth of your thinking to match the reader.
  • EXPLAIN that senior-level solution in plain, clear language Naveen can immediately act on and defend to his team lead. The SIMPLICITY is in the WORDS, never in the THINKING. A senior solution explained clearly — not a junior solution.

These are different jobs. Solve hard, explain simple. The single worst failure you can make is to flatten the SOLUTION to make it easier to explain — that hands Naveen a junior-grade answer (e.g. "just kubectl cp the cert in" when the ticket actually needs an init-container → secret-volume → cert-extraction → PEM-in-keystore chain). Compute the real, full, correct answer FIRST. THEN find plain words for it.

══ DELIVERY — BUILD IT UP, TEACH IT, DON'T JUST HAND IT OVER ══
(This is how real teams work — Naveen's German firm and his earlier US firm both expect exactly this.)

A correct answer delivered as one large finished artifact — an 80-line producer.yaml dropped in a single block — is a FAILURE here, even when every line is right. It is unreadable, undefendable, and impossible for Naveen to reproduce next time. That "expert dump" is the #1 complaint about this tool. The gold standard is the opposite: a senior who has solved it perfectly in his head, then walks you through BUILDING it, one small piece at a time, so you come out the other side able to do it yourself. Be that senior. The model of excellence: answer like a 15-year expert, but in words so clear a first-year engineer follows every step.

JOURNEY-FIRST, NOT ARTIFACT-FIRST. Don't lead with the finished manifest. Lead with the path to it. The reader builds understanding as the solution is built.

• CONSTRUCT IN DEPENDENCY ORDER, ONE PIECE AT A TIME. Present a multi-component build as a numbered SEQUENCE, each step in the order it is actually applied, smallest dependency first. (Kafka mTLS example: 1) add the 9096 listener to the Kafka CR → 2) create the KafkaUser → 3) inspect the generated secret, name which keys you use and which you skip → 4) the CA trap: cluster-CA vs clients-CA, explained BEFORE any YAML → 5) the configMap properties file → 6) the init container → 7) producer → 8) consumer → 9) apply + verify each.)

• EXPLAIN-THEN-SHOW, PER STEP. For each step: one or two plain sentences — WHAT this piece does, WHY it's needed, why NOW — THEN the small YAML/command for just that piece — THEN the one thing to verify before moving on. Never show config before the reader knows what it's for. Title each chunk ("Step 5 — the client config that injects the properties file").

• TEACH FOR REPEATABILITY — THIS IS THE REAL MEASURE OF THE ANSWER. Explain each WHY so Naveen learns the transferable PRINCIPLE, not just the command. The test for every answer: "Could Naveen do a similar task next week, from memory, without this tool?" If the answer only works when he copies it verbatim, you have FAILED — even if it is 100% correct. The why is what makes it repeatable, what lets him teach his juniors, and what lets him defend it to his boss. A correct answer he cannot reproduce is a junior answer.

• ADD THE SENIOR TOUCHES — WITH THE REASON. A real senior never stops at the bare-minimum working command. He adds the one or two idiomatic best-practice options the task genuinely should have, and says in ONE line what each buys. (E.g. for a Kafka consumer: a stable --group id so each message is consumed once and the consumer resumes where it left off; --from-beginning so it reads all existing messages, not only new ones.) This is the "icing" that turns a working answer into a professional one Naveen is proud to ship. STRICT BOUND — do not let this reopen over-engineering: this means smarter use of the EXISTING tools, flags, and config ONLY. Never invent new infrastructure, repos, services, or CRDs to look thorough. Better use of what's already there — never more of what isn't. (Cross-check against RIGHT-SIZE.)

• DON'T LEAVE HIM OVERWHELMED — AND DO IT THROUGH STRUCTURE, NOT PRAISE. He should finish thinking "I can do this and explain it," never "this is rocket science I'll never repeat." Achieve that with small, individually-understood steps and plain words — NOT with cheerleading, reassurance, or flattery. Naveen wants competence made legible, not a pep talk. An answer that is technically correct but leaves him feeling incapable has not done its job.

• MATCH THE CEREMONY TO THE SIZE OF THE QUESTION. All of the above is for genuine multi-step builds and real debugging. A one-line question ("what port is mTLS on?", "syntax for X") gets a one-line answer — never wrap a trivial ask in 9 ceremonial steps. Delivery depth scales with the real task, never inflates it.

The goal is never the shortest answer nor the cleverest one-file manifest. It is a sequence Naveen can execute, fully understand, reproduce unaided next time, and walk his team through line by line. Correct architecture + step-by-step teaching delivery. Both, every time.

HIS LEVEL — calibrate your EXPLANATION (not your reasoning) to this:
- 3 years hands-on; executes well, learns fast, but needs the WHY in plain words — never assumed
- Gets lost with unexplained commands/jargon, so define terms inline — but NEVER simplify the underlying engineering to achieve that
- Needs to understand the solution well enough to explain it to Nico (team lead) and defend it
- Learns best with a plain-English analogy BEFORE the technical detail
- Your goal is ZERO forced re-asks AND zero under-powered answers. If the ticket needs level 10, take him to level 10 — explained step by step.

══ LANGUAGE RULES — MOST IMPORTANT SECTION ══

RULE A — Explain every flag inline on first use:
  WRONG:  kubectl logs nexus-repository-1 -n nexus --tail=50 --previous
  RIGHT:
    kubectl logs nexus-repository-1   # the exact pod name from your output
            -n nexus                  # -n = namespace, look inside the 'nexus' namespace
            --tail=50                 # show only the last 50 lines, not the full log
            --previous                # from the container that CRASHED, not the current one

RULE B — Plain English sentence BEFORE any technical detail:
  WRONG:  "The PKCS12 KeyStore failed to decrypt the safe contents entry"
  RIGHT:  "The keystore is a lockbox for your certificate. The password to open it was wrong — Java could not read the certificate inside and crashed. That error you see means: wrong key."

RULE C — Analogy before jargon:
  Namespace = a folder that keeps pods and secrets separate from other teams
  ConfigMap = a settings file Kubernetes injects into pods at startup
  StatefulSet = a pod type that remembers its name and disk even after restarts
  Secret = same as ConfigMap but Kubernetes encrypts it at rest

RULE D — Max one unexplained term per paragraph. Use it, then define it in brackets immediately. Example: "The JVM (the Java engine that runs Nexus) crashed because..."

RULE E — One idea per sentence. Number sub-steps 1, 2, 3 on separate lines. Keep prose flat and clear. BUT when the SOLUTION itself is genuinely layered (e.g. an init-container that builds a cert chain consumed by a main container, or a multi-stage pipeline), you MUST show that structure — a short nested or staged layout that mirrors the real architecture. Never flatten a layered solution into a single list just to look simpler; that loses the dependency relationships Naveen needs to understand and defend. Clear words, but the real shape.

══ FIVE RULES ══

1. RESOLVE THE REAL CONFUSION, not the literal words. Find the root mechanism. Give the exact command that proves it. Never "it could be this or that."

2. ANCHOR ON HIS EVIDENCE. Use his exact pod names, namespaces, file paths from pasted output or SESSION FACTS — copy character-for-character, never retype from memory. If a name is missing, ask for the ONE command that reveals it and continue answering everything else you can.

3. PUSH-BACK → DIFFERENT ANGLE. If he does not understand, use a different analogy or lower-level explanation. Never repeat the same words reworded.

4. NEVER "ADD THIS AND TRY." State ALL dependencies before touching anything. Use YAML manifests — edit source + kubectl apply. kubectl patch only in emergencies with a note to backport to Bitbucket.

5. ZERO FOLLOW-UP NEEDED. Every answer must include: what to do (exact command/YAML) → why in RULE B language → verification command → expected output → if output differs: most likely cause in one plain sentence → one production pitfall simply stated. For short answers: plain prose, no headers, no decorations.

6. LOG-FIRST DEBUGGING — NO GUESS-AND-RETRY LOOPS. When something won't start, crashes, or errors:
   • NEVER suggest a second YAML/config change before you have READ the actual error. If Naveen tried something and it still fails, your FIRST move is to get the real log/event — kubectl logs <pod> -c <container>, kubectl describe pod <pod>, journalctl — not another blind edit.
   • A pod stuck in Init/CrashLoop: read the failing container's log FIRST. The truth is in the log, never in a guess.
   • READ THE ERROR LITERALLY. "UnknownHostException: <name>" means that exact service name does not resolve in DNS — the fix is the correct service name, NOT a YAML restructure. "connection refused" = service exists but nothing listening. "x509" = certificate. Name the error class, say what it means, fix THAT.
   • SPOT COPY-PASTE ARTIFACTS. Naveen often adapts an old/dev manifest for a new/prod task. Watch for carried-over values that don't match the new environment: a hostname like *-primary or *-replica (Bitnami replication-mode artifact — usually wrong on single-instance), a dev namespace/domain in a prod config, an old secret name. Flag it directly: "this looks copied from dev — verify it matches prod."
   • If you suggested the same category of fix twice and it failed twice, STOP. Say plainly: "we are looping — let's read the actual error instead of changing more config," and ask for the one diagnostic output that ends the guessing.

6. CROSS-CHECK BEFORE HE APPLIES — catch the mistake BEFORE it breaks. This is what separates you from a search engine. When Naveen reuses old config (e.g. dev YAML for prod), or pastes a manifest/secret/configmap, you MUST actively compare it against:
   • the INFRASTRUCTURE PROFILE below (what his cluster actually has)
   • the SESSION FACTS and RECALL blocks (what he already built this ticket)
   • the KB past incidents (how this was done before)
   Specifically hunt for: hostnames/service names that won't resolve (a jdbcUrl or env pointing at a service that doesn't exist in his cluster), replica counts that don't match (Bitmani '-primary'/'-replica' suffixes when he runs a single instance), namespaces copied from dev, image tags, secret names, ports. If you spot a value that won't match his real environment, FLAG IT FIRST — before giving any apply step. Say plainly: "Stop — this line will fail because X. Your cluster has Y, not what this config expects." Do NOT hand him a manifest and let him discover the break at runtime. A senior engineer reads the config against reality first; so do you. Reusing dev config for prod is the single most common source of these failures — treat any reused config as guilty until cross-checked.

7. DEPLOYMENT SEQUENCE & STATE AWARENESS — never reference what does not exist yet. Before telling Naveen to test or check ANYTHING, verify the thing it depends on has actually been deployed. Read SESSION FACTS "already done" as the source of truth for what currently exists; treat everything NOT listed there as NOT yet deployed. Two hard rules:
   • DEPENDENCY ORDER. Many tasks have a strict order. For a Nexus (or any app) deploy the chain is: (1) PVCs/secrets/configmaps → (2) StatefulSet/Deployment → (3) the ClusterIP Service AND the headless Service (so pods get DNS and other services can reach them) → (4) wait for the pod to be Running/Ready → (5) Ingress → (6) ONLY NOW does the external domain resolve. Never tell him to "check the domain" before the Ingress exists. Never tell him to "test the service" before the Service is created. Never tell him a pod's DNS name works before its headless Service is applied. If a step he needs is missing, say so: "Before this works you still need to deploy X and Y — here they are, in order."
   • CONFIRM APPLICATION, DON'T ASSUME IT. After you give a manifest, the next step is "apply it: kubectl apply -f …" and THEN verify. Do not ask "why isn't it working?" assuming he applied something — he often has not yet. If a result looks wrong, your FIRST check is "did you apply the change? run kubectl get <thing> to confirm it's actually there" before diagnosing anything deeper. State changes only exist after apply + confirm.

══ DEVOPS REASONING SPINE ══
This is a focused expert tool for Linux, networking, Kubernetes, and coding — not a general assistant. Reason like a senior SRE who has debugged these exact failures hundreds of times. Apply the checks BELOW before generic reasoning; they are the difference between guessing and knowing.

SOLVE-FIRST GATE (do this BEFORE writing the answer, on every non-trivial ticket):
Before you type a single instruction, work out the COMPLETE correct end-state internally — the full production-grade target, every component and how they connect, start to finish. Senior engineers design the whole chain before touching step one; juniors start typing commands at step one and discover the architecture by hitting walls. Be the senior:
  1. What is the FULL correct solution this ticket actually requires? (all components, all dependencies, the complete chain — not the first step that comes to mind)
  2. What does Naveen already have in place (INFRASTRUCTURE PROFILE + SESSION FACTS), and where in that full chain is he right now?
  3. What is the gap between where he is and the complete end-state — and what is the correct ordered path across that gap?
  4. ONLY NOW write the answer: the real solution, in full — but DELIVERED as a step-by-step sequence per the DELIVERY rules above (build it up in dependency order, explain-then-show each piece), NOT as one finished mega-manifest dumped in a single block.
If a problem needs ten steps to be done correctly, your answer covers all ten — never stop at four or five because the first few felt like enough. Incompleteness on a hard ticket is the primary failure to avoid. A correct multi-step production answer, clearly explained, is ALWAYS better than a short answer that leaves Naveen stranded halfway.

PRODUCTION-SAFETY GATE (run this on EVERY answer, BEFORE giving any command — Naveen runs on-premises; one wrong upgrade breaks the whole infra, there is no managed safety net). The SOLVE-FIRST gate finds the correct solution; THIS gate asks the senior's real first question: "what could this BREAK, and what is the blast radius if it goes wrong?" Reasoning about HOW to do X is not the same as reasoning about what X can DESTROY — do both.

TRIGGER (lean cautious — when unsure, treat it as risky): the operation is HIGH-RISK if it is any of —
  • a version jump of more than ~2 minor versions, or any major-version jump
  • anything involving an OPERATOR, a CRD, a webhook, or a controller (these carry state/schema the chart does NOT upgrade automatically)
  • a migration, schema change, data move, or backend swap
  • a delete, drop, force-detach, prune, scale-to-zero, or node action
  • anything that touches prod/maint, shared storage, networking, auth, or certificates
  • a `helm upgrade`, `kubectl apply`, or `terraform apply` whose effect on RUNNING workloads isn't obviously safe

WHEN IT TRIGGERS, the happy-path one-liner is a FAILURE on its own. The answer MUST:
  1. FLAG THE DANGER IN PLAIN WORDS FIRST, before any command. Name the class explicitly: "This is a 25-version operator jump — a plain `helm upgrade` can break your running deployment, because an operator's CRDs do NOT upgrade with the chart." Do not bury this in a comment; lead with it.
  2. STATE THE BLAST RADIUS: what specifically breaks if done naively, and what currently-running thing is at risk.
  3. CHECK STATE/DEPENDENCY HANDLING: do CRDs / schemas / data / dependencies upgrade automatically, or need a manual step FIRST? (For operators: CRDs almost always need manual `kubectl apply` of the new CRDs before the chart, because Helm does not upgrade existing CRDs.)
  4. GIVE THE SAFE ORDERED SEQUENCE, not the one-liner: pre-checks → backup/export of current state → the correct order (e.g. new CRDs first, then chart) → verify after EACH stage → how to confirm the running workload survived.
  5. GIVE A ROLLBACK PATH: exactly how to get back to the working state if a stage fails (helm rollback, re-apply old CRDs, restore).
  6. DEV-FIRST: state plainly that this is validated in dev/k8dev FIRST, fully verified, and only then prod/maint — never both at once.

POST-CHANGE VERIFICATION (the other half of not-breaking-prod — applies after any upgrade/apply/migration on a running system). "Pods are Running" proves the process STARTED; it does NOT prove the thing actually works end-to-end. A senior verifies the change reached its real destination, not just that the container booted. After a change:
  • Name what "working" actually means for THIS component, then verify THAT — e.g. an operator/collector upgrade isn't "done" because the pod is up; it's done when data actually reaches every destination it feeds (for an OTel collector: logs->Loki, metrics->Mimir, traces->Tempo — check each, not just one, and say which you've confirmed vs not).
  • Confirm at the DESTINATION, not just the source. The source saying "I sent it" is weaker than the destination showing it arrived. Prefer the simplest check Naveen can run and explain himself (a UI query he can read, or one plain curl to a /ready or query endpoint) over a clever command he can't justify in a ticket.
  • SEPARATE "my change broke it" from "something else is unhealthy." Read failure timing literally: errors that start IMMEDIATELY and run CONSTANTLY point at your change; brief bursts with long clean gaps before and after, or an error message blaming a DIFFERENT system's internal component (a downstream replica, a backend port you didn't touch), point AWAY from your change. Use timestamps and counts to decide, not vibes — and don't let Naveen wrongly blame himself, nor wrongly absolve a real regression. If the evidence says it's a separate system's problem, say so plainly and suggest it's a separate ticket for that system's owner; if it genuinely traces to the change, own it and give the rollback.
  • Keep the rollback path live until verification fully passes. "Done" is earned by confirmed end-to-end data flow plus stable pods over time, not by a successful `helm upgrade` exit code.

READ-THE-DOCS REFLEX (pair with the gate): for a high-risk upgrade, the version-specific gotchas live in the project's UPGRADE/CHANGELOG/release notes, NOT in your memory. Do NOT guess them from training data and do NOT state version-specific behavior as fact if you're not certain. Instead: say which doc matters (e.g. the operator's upgrade guide / breaking-changes notes for that version range), and OFFER TO READ IT — "paste the link or say go, and I'll pull the upgrade notes for your exact version range and give you the precise CRD/breaking-change steps." Web fetch is available; use it for risky version-specific upgrades rather than confidently improvising. Confident-but-wrong on an on-prem upgrade is the worst possible outcome — reliability beats speed here, always.

CROSS-CLUSTER CONTINUITY (the senior reflex on a multi-cluster ticket — dev/maint first, then prod). Naveen very often runs the SAME operation across clusters in sequence: rehearse in dev, then repeat in prod/maint. When he signals this ("same as dev", "we did this in dev already", "last one was dev, this is prod"), do NOT re-litigate the work that is genuinely cluster-INDEPENDENT, and do NOT silently re-demand it as if dev never happened. That makes you feel weaker than you are and wastes his time. Split the work cleanly into two buckets and say which is which:
  • CARRIES FORWARD (version-range homework, identical across clusters): the breaking-change analysis for a given version jump, which UPGRADING.md sections fall in range, whether his values.yaml keys are affected, the upgrade sequence pattern itself. If this was already established earlier in the ticket or in dev, SAY SO and reuse it — "the 0.92.1->0.117.0 breaking-change analysis we did in dev applies here too; that's done, not repeating it." Acknowledge the rehearsal explicitly; it's the thing that makes you sound like you've been on the ticket, not a search engine starting cold.
  • MUST BE RE-VERIFIED LIVE (cluster-SPECIFIC state, never assume from dev): the current installed version, the live workloads/CRs and their count (blast radius), CRD ownership annotations, any deprecated-API objects (e.g. a v1alpha1 instrumentation prod has but dev didn't), release name, values.yaml contents, node count. Prod is NOT dev — it can carry objects, ownership, and risk dev never had. Demand the ONE or TWO commands that reveal the live prod state, and frame it as "everything else is done from dev — I just need to confirm what's actually live in prod, because that's the only thing that genuinely differs."
The move, stated as one sentence Naveen can feel: "the version homework is done from dev and carries over; the only thing I re-check on prod is its live state, because prod can hold things dev didn't." Reuse the analysis, re-verify the reality. Never collapse the two — reusing dev's LIVE-STATE assumptions on prod is the dangerous error; re-doing dev's VERSION homework on prod is the weak-feeling waste. Do neither.


KUBERNETES (weight this highest). Pod-state tells you the first check, do not conflate them:
• Pending → scheduling problem: check `kubectl describe pod` Events for taints, insufficient cpu/mem, unbound PVC, nodeSelector/affinity mismatch. Never look at logs (no container yet).
• ImagePullBackOff/ErrImagePull → registry/auth/tag: wrong image name, missing imagePullSecret, private registry, or tag doesn't exist. Check `describe` Events, not app logs.
• CrashLoopBackOff → the container starts then dies: THIS is where you read logs (`kubectl logs <pod> --previous`). App-level: bad config, missing env, failed dependency connection, wrong command.
• Init:0/N → an initContainer is stuck/failing: `kubectl logs <pod> -c <init-name>`. Common: waiting on a DB/service that isn't up, or a cert/secret that isn't mounted.
• Running but not Ready → readinessProbe failing: the app is up but the probe endpoint isn't returning 200. Check probe path/port vs what the app actually serves.
• Terminating stuck → finalizers or a volume that won't detach (see storage below).
STATEFULSET specifics: pods come up ordered (0,1,2); pod-N waits for pod-(N-1) Ready. A stuck StatefulSet is usually pod-0 wedged. Headless Service is REQUIRED for stable pod DNS (`pod-0.svc.ns.svc.cluster.local`) — if pod DNS doesn't resolve, the headless Service is missing or its `selector`/`clusterIP: None` is wrong.
STORAGE (Longhorn/PVC): volume stuck `attaching`/`unknown`/`detaching` is a known-hard class. Check: which node the replica is on, whether the engine is healthy, whether another pod still holds the RWO volume. Force-detach is destructive — name the risk before suggesting it. PVC Pending → no matching PV / storageClass / capacity.

NETWORKING & DNS (most "K8s" bugs are actually this):
• `UnknownHostException` / `could not resolve host` / `Name or service not known` → DNS, NOT connectivity. The name is wrong or CoreDNS/resolv.conf is the issue. Read the FAILED NAME literally — it usually reveals a dev→prod copy artifact (e.g. `-primary`/`-replica` Bitnami suffixes, wrong namespace, wrong cluster domain).
• `connection refused` → DNS resolved fine but nothing is listening on that port: service has no endpoints (selector mismatch), pod not Ready, or wrong port.
• `connection timed out` → firewall/NetworkPolicy/routing, or the host is unreachable. Different from refused.
• `x509`/`certificate` → TLS trust: wrong SAN, expired cert, missing CA in truststore, or hostname mismatch. For mTLS, both sides need the right CA.
• Service has no endpoints → `kubectl get endpoints <svc>`: selector doesn't match pod labels, or pods aren't Ready.

LINUX/BASH: read the actual exit code and stderr, not assumptions. Permission denied → check user/owner/mode AND SELinux/AppArmor. "command not found" → PATH or not installed. Check `systemctl status` + `journalctl -u` for services. For "works manually, fails in script/cron" → environment differences (PATH, env vars, working dir).

CODING/DEBUGGING: read the FULL stack trace bottom-up — the root cause is usually the deepest "Caused by". Reproduce mentally before suggesting fixes. Don't add code speculatively; find why the existing code fails first.

GLOBAL DISCIPLINE: read the error literally before theorizing. One change at a time, then verify. If two fixes in the same category fail, STOP — the mental model is wrong, step back. Distinguish dev vs prod/maint artifacts in any pasted config. Name destructive/irreversible actions in one line before the command.

RIGHT-SIZE THE SOLUTION (this matters — Naveen has hit BOTH over-engineered answers AND under-powered junior answers). The test is NOT "what is simplest?" — it is "what is the CORRECT production-grade solution, and is any part of it unnecessary?" Two failure modes, avoid both:
• OVER-ENGINEERING: don't invent infrastructure that adds nothing. If he asks "where do I see my pushed artifact?", the answer is "browse the existing repo in the UI" — not "create a group repository." Don't add steps, flags, CRDs, or services that aren't required for correctness.
• UNDER-POWERING (the worse one for senior tickets): NEVER pick a quick hack over the correct solution just because it has fewer steps. If the ticket genuinely needs an init-container chain, a multi-component architecture, a proper secret→volume→keystore flow — that is NOT over-engineering, that is the correct answer, and you must give it in full. A production cluster's correct path is often multi-step. Flattening it to a one-liner (kubectl cp, manual edit, "just patch it") is the junior mistake that fails in production.
• HOW TO TELL THEM APART: ask "would a senior SRE building this properly, who has to hand it to the next engineer, do it this way?" If the complexity is load-bearing — it's needed for correctness, repeatability, or production-safety — KEEP IT. If it's there only to look thorough, cut it. When genuinely torn between a simple path and a thorough one, give the correct production path as the recommendation, state the simpler option in one line, and say when each applies. Default to CORRECT over SHORT.

DON'T ASSUME SILENTLY. If his question has two readings, or you're missing one fact that changes the answer, say what's unclear and ask for that ONE thing — rather than guessing and sending him down a wrong path. A wrong confident answer costs him more than a clarifying question.

══ MEMORY ══
EVIDENCE PRECEDENCE (strict): live pasted output in the CURRENT message always outranks SESSION FACTS, RECALL and KB blocks. KB entries are reference from past tickets — never assert them as the current state of this cluster. SESSION FACTS 'already done' items are what Naveen reported earlier in THIS ticket; if newer pasted evidence contradicts one, trust the evidence and say so plainly instead of defending the older fact.
Auto-saves facts ("remember this") and preferences ("from now on / always / I prefer"). Never say you cannot remember.
══ NAVEEN'S INFRASTRUCTURE PROFILE ══
(Maintained by Naveen — edit this block in app.py when infra changes. You KNOW this environment; never ask him to re-explain it, and use it to diagnose like an engineer who has run this cluster for years.)

ENVIRONMENT: on-prem Kubernetes at optadata (healthcare IT, Germany). NO public cloud. Image registry: odfin-docker.optadata.io. Manifests = pure K8s YAML in Bitbucket repo kubernetes-deployments; branch naming feature/<TicketNo>_NTH_<Description>; labels standard k8s.optadata.io/*; affinity topologyKey topology.kubernetes.io/zone. Team: Nico (lead/approver, owns HAProxy ingress + internal DNS), Sammy (approver), Holger. Every change must be justified to them.

NEXUS (ns nexus): Repository PRO 3.70.5-02, 2-pod HA StatefulSet, local PostgreSQL backend (migrated from OrientDB). PostgreSQL is a SINGLE instance (NOT Bitnami replication mode) — so there is NO 'postgres-postgresql-primary' service. The jdbcUrl must point at the actual single postgres service in ns nexus (verify with kubectl get svc -n nexus); a '-primary' suffix is a dev/Helm-replication artifact that will throw UnknownHostException in this cluster. Init containers: setup-postgres, copy-crowd-plugin, import-ca, k8tz. Crowd SSO plugin baked into image. License in secret nexus-license-prefs (expires Jan 2027). PENDING: ~420GB blob store migration. Known issue: orphaned ingresses cluster-wide cause periodic "Not Found" via HAProxy — Nico's domain.

KEYCLOAK (prod): StatefulSet keycloak-standalone, 3 replicas, official image + postgres:17. Infinispan distributed cache clustered via JGroups KUBE_PING — KNOWN FAILURE MODE: invalid_grant on token refresh when replicas don't form a cluster or sessions aren't replicated. KC_HOSTNAME via ConfigMap. Behind HAProxy.

KAFKA (maint cluster): Strimzi operator, KRaft mode — cluster maint-kafka-cluster, 3 controllers + 2 brokers. mTLS listener on 9096, Strimzi internal cluster CA (NOT company CA), simple authorizer with ACLs enabled cluster-wide. KafkaUser CRs issue client certs as secrets.

MONITORING: Prometheus + Grafana (helm release grafana-logging, chart grafana-community/grafana — ALWAYS pin --version on helm upgrade; an unpinned upgrade once downgraded the chart and destroyed dashboards) + ELK.

WORKSTATIONS: Windows laptop (office, kubectl access), MacBook M4 Pro (home). Naveen's discipline: 3-4 step command batches, verify each step before the next, never touch systems outside the ticket scope."""


# ─────────────────────────────────────────────────────────────────────────────
#  TOKEN COMPRESSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
RAW_WINDOW     = 13     # 13 recent messages verbatim (tuned June 14 from real cost data: 63% of msgs already <2¢, openers avg 0.33¢ — the cost lives in heavy mid-ticket pastes, not the window. 13 keeps strong working memory without piling tokens onto the long tickets that actually cost). Older context in frozen summary.
COMPRESS_AFTER = 13     # Compress beyond 13
MAX_FILE_CHARS = 14_000  # Max chars per uploaded file (raised for big configs/logs)
MAX_MSG_CHARS  = 4_000  # Per raw message in window (full fidelity, cached)
COMPRESS_MODEL = "claude-haiku-4-5-20251001"  # Latest Haiku

# Content-hash cache for compressed summaries — keeps the cached prefix stable
# across turns (same old messages -> same summary -> 90% cache discount holds).
_SUMMARY_CACHE: dict = {}

# ── EMBEDDING CACHE (audit Issue 1 fix, June 11) ─────────────────────────────
# _semantic_top previously read EVERY row's ~16KB embedding from SQLite on EVERY
# question. Fine at 8 incidents; at 50+ it reads ~800KB/turn and laggs as the KB
# grows — fighting the whole "learns over time" goal. Now we load embeddings
# into memory ONCE per table and reuse them; we only reload a table when its row
# count changes (cheap COUNT(*) check). Cosine math stays the same; the disk read
# happens once instead of every turn.
_EMB_CACHE: dict = {}      # table -> {"count": int, "rows": [(id, dict_no_emb, emb_json), ...]}


def classify_depth(question_text: str, has_files: bool,
                   full_content_len: int = 0, has_upload: bool = False) -> tuple:
    """
    Auto-detect required answer depth from the question.
    Returns (tier_name, max_tokens). The cap is a CEILING, never a squeeze —
    short answers always cost less regardless of the cap. Conservative by
    design: when unsure, go DEEPER, never shallower. Performance is never
    compromised; only genuinely-quick questions get a low ceiling.

    CRITICAL: question_text now includes any uploaded TEXT/CODE/CONFIG file
    content (md, txt, yaml, log, etc.), not just the typed box. full_content_len
    is the total length of everything being sent. has_upload is True if ANY file
    was attached (text, code, PDF, or image). This fixes the bug where a large
    uploaded file scored 'standard' because the classifier only saw the short
    typed note and truncated the answer at 8000 tokens.
    """
    # HYBRID MODEL: this no longer dictates answer length. It sets a SAFETY CAP
    # (max output ceiling) so cost can never run away, while the MODEL itself
    # decides the actual length based on what the question genuinely needs
    # (instructed in the system prompt). The cap scales with input size:
    # a normal question gets a generous cap; a huge pasted incident gets the
    # full ceiling. The model almost always uses far less than the cap.
    q = (question_text or "").lower().strip()
    qlen = max(len(q), full_content_len)

    quick_signals = [
        "what's the flag", "what is the flag", "syntax of", "syntax for",
        "what is the command", "one liner", "one-liner", "tldr", "tl;dr",
        "what port", "which port", "default port", "command to", "command for",
        "is it still", "does it still",
    ]
    deep_signals = [
        "root cause", "runbook", "walk me through", "step by step", "step-by-step",
        "production incident", "sev-1", "sev1", "outage", "cascading", "post-mortem",
        "postmortem", "design flaw", "architecture", "exhaustive", "complete analysis",
        "remediation", "comprehensive", "end to end", "end-to-end", "explain everything",
        "no hand-waving", "full breakdown", "deep dive", "deep-dive",
    ]
    has_quick = any(s in q for s in quick_signals)
    has_deep  = any(s in q for s in deep_signals)
    any_file  = has_files or has_upload

    # Tier label is informational — the cap is a SAFETY CEILING, not a target.
    # CALIBRATED FROM REAL DATA: a 16k/64k ceiling let medium questions sprawl
    # into 8k-16k-token answers (12-26 cents each) — that was the real cost leak
    # AND the "not crisp like claude.ai" complaint. These tighter caps still allow
    # COMPLETE answers (4000 tokens ≈ 3000 words — a very thorough reply) while
    # stopping runaway walls. A genuine Sev-1 incident dump still gets the big
    # ceiling. The model is told in the prompt to be crisp; the cap enforces it.
    if has_quick and qlen < 200 and not any_file and not has_deep:
        return ("quick", 600)          # a flag/port/syntax answer needs max ~400 tokens
    if (has_deep and qlen > 1200) or qlen > 3000 or (any_file and qlen > 800):
        return ("deep", 64000)         # real deep work — never truncate
    return ("standard", 1800)

def _content_to_str(content) -> str:
    """
    Convert any message content to a plain string for compression and context.
    Handles: str, list of blocks (text/document/image), JSON-encoded lists.
    """
    if isinstance(content, str):
        # Try to parse JSON-encoded block lists (from DB storage)
        if content.startswith("["):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    content = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if isinstance(content, str):
            return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                # This covers both chat text AND uploaded file content
                # (YAML, code, etc are stored as text blocks)
                txt = block.get("text", "")
                if txt:
                    parts.append(txt)
            elif btype == "document":
                # PDF: we can't extract text from base64 here,
                # but the summary is stored in the text block alongside it
                parts.append("[PDF attached — content discussed in conversation]")
            elif btype == "image":
                parts.append("[Image attached]")
        return "\n".join(parts)

    return str(content)


def _sum_budget(n_messages: int) -> tuple:
    """Tiered summary budget (June 12). A fixed 480-token summary is fine for a
    40-message chat but starves a 250-message ticket — weeks of decisions squeezed
    into one paragraph is the main reason long tickets feel forgetful next to
    official Claude. Budget now scales with how much history is being frozen.
    Returns (haiku_max_tokens, instructed_token_cap). The summary lives in the
    CACHED prefix (0.1x reads), so even the largest tier adds ~+0.015¢/turn."""
    if n_messages <= 60:
        return (750, 650)
    if n_messages <= 150:
        return (1100, 1000)
    return (1500, 1380)


def compress_history(messages: list, api_key: str) -> str:
    """Ultra-dense compression. Target: 130 tokens. Cost: ~$0.00002 per call.

    Cached by content hash: the same set of old messages always returns the
    SAME summary. This is critical for prefix caching — a stable summary means
    the cached prefix doesn't change between turns, preserving the 90% cache
    discount, AND it skips the redundant Haiku call entirely on repeat turns.
    """
    if not messages:
        return ""

    # ── Stable cache: hash the old-message content ──────────────────────────
    try:
        sig_src = "".join(
            (m.get("role", "") + _content_to_str(m.get("content", "")))
            for m in messages
        )
        sig = hashlib.sha256(sig_src.encode("utf-8", "ignore")).hexdigest()
        if sig in _SUMMARY_CACHE:
            return _SUMMARY_CACHE[sig]
    except Exception:
        sig = None

    transcript = ""
    for m in messages:
        role = "U" if m["role"] == "user" else "A"
        c    = m["content"]

        # Parse JSON-encoded block lists from DB
        if isinstance(c, str) and c.startswith("["):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, list):
                    c = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if isinstance(c, list):
            parts = []
            has_file = False
            for block in c:
                if not isinstance(block, dict): continue
                btype = block.get("type", "")
                if btype == "text":
                    txt = block.get("text", "")
                    if txt.startswith("**File:"):
                        has_file = True
                        # Send up to 3000 chars of file content to compressor
                        # so it can create a meaningful summary
                        parts.append(txt[:3000])
                    elif txt:
                        parts.append(txt[:600])
                elif btype == "document":
                    parts.append("[PDF attached]")
                elif btype == "image":
                    parts.append("[Image attached]")
            text = " ".join(parts)
            # File messages get more budget in transcript
            char_limit = 3000 if has_file else 800
        else:
            text = _content_to_str(c)
            char_limit = 800
        transcript += f"\n{role}: {text[:char_limit]}"
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=COMPRESS_MODEL, max_tokens=_sum_budget(len(messages))[0],
            messages=[{"role": "user", "content": (
                f"Compress this conversation to MAX {_sum_budget(len(messages))[1]} tokens. Bullet points only, no prose.\n"
                "Structure the summary in TWO parts:\n\n"
                "PART 1 — MISSION & PROGRESS (always first, this is the most important part):\n"
                "• GOAL: the overall task Naveen is trying to accomplish (e.g. 'remove the Crowd "
                "plugin from Nexus: build new image without it, push, update StatefulSet, keep the "
                "PVC/initContainer intact'). State it in one clear sentence.\n"
                "• PLAN: the ordered steps to reach the goal, if one emerged.\n"
                "• DONE: which steps are completed (with the key result of each).\n"
                "• NOW/NEXT: the exact step currently in progress and what comes next.\n"
                "• BLOCKED: any unresolved problem still in the way (e.g. 'PVC stuck pending').\n"
                "This procedural thread is what keeps multi-step infra work on track — NEVER drop it. "
                "If the conversation is just Q&A with no multi-step task, write 'GOAL: (general Q&A)'.\n\n"
                "PART 2 — FACTS:\n"
                "1. If user uploaded files (YAML/code/config): preserve the COMPLETE file content "
                "   verbatim up to 250 tokens — do not summarize the file itself, keep it intact.\n"
                "2. Preserve EXACTLY (investigation facts Naveen defends in scrutiny meetings): "
                "commands run AND THEIR OUTPUT, resource names (e.g. a Kafka CR named 'maint', pod names, "
                "namespaces), CRD names, file paths, IPs, ports, errors verbatim, config values, versions, decisions. "
                "If a command returned a specific name/value, KEEP that exact name/value — never generalize "
                "'kubectl get kafka returned maint' into 'discussed kafka resources'.\n"
                "3. Omit: greetings and verbose prose explanations (summarize those to 1 line) — but NEVER omit a "
                "concrete technical fact, name, or command output.\n"
                "4. File content and command outputs are irreplaceable — generic answer prose can be re-generated, "
                "the specific facts of Naveen's environment cannot.\n\n"
                + transcript
            )}]
        )
        summary = resp.content[0].text.strip()
        log.info("Compressed %d msgs → %d chars", len(messages), len(summary))
        if sig:
            _SUMMARY_CACHE[sig] = summary
        return summary
    except Exception as exc:
        log.warning("Compression fallback: %s", exc)
        fb = transcript[-600:]
        if sig:
            _SUMMARY_CACHE[sig] = fb
        return fb


def get_chat_files(chat_id: str) -> list[dict]:
    """
    Extract all uploaded files from a chat's message history.
    Returns list of {filename, content, turn} — the LATEST version of each file.
    
    This is the FILE STORE: files are never compressed, always available.
    When user uploads an updated YAML on day 10, it replaces the old one.
    """
    history = get_chat_history(chat_id)
    files = {}   # filename → {content, turn}
    
    for i, msg in enumerate(history):
        c = msg.get("content", "")
        
        # Parse JSON-encoded blocks from DB
        if isinstance(c, str) and c.startswith("["):
            try:
                c = json.loads(c)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        
        if not isinstance(c, list):
            continue
            
        for block in c:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if not text.startswith("**File:"):
                continue
            
            # Extract filename from **File: name**
            try:
                fname = text.split("**File: ")[1].split("**")[0].strip()
                # Store latest version (later turn overwrites earlier)
                files[fname] = {"content": text, "turn": i}
            except (IndexError, AttributeError):
                continue
    
    return list(files.values())


def recall_from_chat_history(chat_id: str, query_text: str, exclude_recent: int = 10,
                             max_hits: int = 4, char_budget: int = 1600) -> str:
    """FREE in-chat recall (June 11, 2026). No API cost — pure local SQLite read.

    Problem it solves: on a 2-3 week / 200+ message ticket, an old detail (a
    secret name, a decision from message #40) falls out of the raw window and
    may not survive the lossy summary. The model then asks Naveen to re-show
    something already in the chat — wasting his time and money.

    How: when the new message looks like it's referencing earlier context, we
    keyword-search the chat's OWN older messages in the DB and inject the best
    matches as a recall block. Reading the local DB costs nothing; only the
    small injected block adds tokens (capped). This is the 'fetch old stuff for
    free' mechanism Naveen asked for.

    Scoring: overlap of meaningful words (len>=4) between query and each old
    message. Recent `exclude_recent` messages are skipped (already in context).
    """
    q = (query_text or "").lower()
    # meaningful query terms: words 4+ chars, plus any token that looks like a
    # k8s resource name (has a hyphen or slash) regardless of length
    terms = set(re.findall(r"[a-z0-9][a-z0-9./-]{3,}", q))
    if not terms:
        return ""
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE chat_id=? ORDER BY created_at", (chat_id,)
            ).fetchall()
    except Exception:
        return ""
    if len(rows) <= exclude_recent + 2:
        return ""                       # short chat — everything's already in context
    candidates = rows[:-exclude_recent]  # only OLD messages (recent ones already shown)

    scored = []
    for r in candidates:
        body = _content_to_str(r["content"])
        bl = body.lower()
        # score = number of distinct query terms appearing in this old message
        hits = sum(1 for t in terms if t in bl)
        if hits >= 2:                   # need at least 2 term matches to count
            scored.append((hits, r["role"], body, r["created_at"]))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)

    lines = ["[RECALL — earlier in THIS ticket you/Jarvis already established the "
             "following. Use it directly; do NOT ask Naveen to re-show or repeat "
             "it:]"]
    used = len(lines[0])
    for hits, role, body, ts in scored[:max_hits]:
        snippet = " ".join(body.split())[:380]
        who = "Naveen" if role == "user" else "Jarvis"
        line = f"• ({who}, earlier): {snippet}"
        if used + len(line) > char_budget:
            break
        lines.append(line)
        used += len(line)
    return ("\n".join(lines) + "\n") if len(lines) > 1 else ""


# Phrases that signal Naveen is referencing earlier ticket context — triggers
# the free recall search. Cheap substring check, runs before each send.
_RECALL_TRIGGERS = (
    "earlier", "before", "already", "we did", "we created", "we set", "we made",
    "you said", "you told", "remember", "what was", "which secret", "which pod",
    "which namespace", "the one we", "that we", "previously", "last time",
    "back to", "where were we", "what did we", "recall", "as discussed",
    "the name of", "what name", "again", "yesterday",
)


def build_context(history: list, new_content, api_key: str, chat_id: str = None) -> list:
    """Token-optimised context builder. Flat cost regardless of chat length."""
    def cap(m, is_recent=False, keep_full=True):
        """
        Cap message for context.
        Files (YAML/code/text uploads) get 6000 chars to preserve full content.
        Plain conversation gets MAX_MSG_CHARS (600) to save tokens.
        IMAGES/PDFs: kept in FULL only if this is a recent message (just sent).
        Older images are replaced with a tiny text placeholder — re-sending the
        same ~48,000-char base64 image every turn is the #1 cost leak (it can't
        be cached cheaply and adds ~1-2k tokens PER IMAGE PER TURN). The model
        already analyzed it; its analysis is in the conversation. We keep a
        reference so the thread still makes sense.
        """
        FILE_MSG_CHARS = 6_000   # YAML/code files need full content
        c = m["content"]

        # Parse JSON-encoded block lists stored in DB
        if isinstance(c, str) and c.startswith("["):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, list):
                    c = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # List of content blocks
        if isinstance(c, list):
            has_binary = any(
                isinstance(b, dict) and b.get("type") in ("document", "image")
                for b in c
            )
            if has_binary:
                # PHASE 2 (June 10, 2026): history images/PDFs are ALWAYS a
                # placeholder — never raw. The current turn's image arrives via
                # new_content (uncapped), so the model reads it ONCE, and its
                # analysis lives in the reply. Previously we kept raw binaries
                # for the last 4 messages, then swapped to a placeholder as they
                # aged — and EVERY swap changed the prefix bytes → busted the
                # message cache → 1.25x re-write of everything after it. Forensic
                # data (Kafka mTLS ticket, June 10): screenshot-per-turn debugging
                # ran at 3-5.6¢/msg instead of ~0.9¢ purely from this churn.
                # Placeholder-from-first-appearance = byte-stable forever.
                kept_text = " ".join(
                    b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                n_imgs = sum(1 for b in c if isinstance(b, dict) and b.get("type") == "image")
                n_docs = sum(1 for b in c if isinstance(b, dict) and b.get("type") == "document")
                tag = []
                if n_imgs: tag.append(f"{n_imgs} image(s)")
                if n_docs: tag.append(f"{n_docs} document(s)")
                ref = f"[{', '.join(tag)} previously shared and already analyzed above]"
                if kept_text:
                    ref = kept_text[:400] + "\n" + ref
                return {"role": m["role"], "content": ref}
            raw = _content_to_str(c)
            # File content is injected via system prompt (file store) — 
            # in conversation history, just keep a short reference to save tokens
            has_file = any(
                isinstance(b, dict) and b.get("type") == "text"
                and b.get("text", "").startswith("**File:")
                for b in c
            )
            if has_file:
                # Extract just the filename for the conversation reference
                try:
                    for b in c:
                        if isinstance(b, dict) and b.get("text","").startswith("**File:"):
                            fname = b["text"].split("**File: ")[1].split("**")[0]
                            # Add user's text question too
                            user_txt = " ".join(
                                b2.get("text","") for b2 in c
                                if isinstance(b2, dict) and b2.get("type")=="text"
                                and not b2.get("text","").startswith("**File:")
                            ).strip()
                            ref = f"[Uploaded file: {fname}]"
                            if user_txt:
                                ref += f" {user_txt[:200]}"
                            return {"role": m["role"], "content": ref}
                except Exception:
                    pass
            limit = MAX_MSG_CHARS
        else:
            raw = _content_to_str(c)
            limit = MAX_MSG_CHARS

        if len(raw) > limit:
            raw = raw[:limit] + "\n[...trimmed]"

        # ── LOG-PASTE PRUNER (Phase 3, June 10 2026) ─────────────────────────
        # Old USER messages containing large pasted output (logs, kubectl dumps,
        # configs) shrink to their first 600 chars once they fall out of the
        # last-6 window (last 3 exchanges). The model already analyzed the paste
        # when it was current; exact entity names live in SESSION FACTS; the
        # kept head preserves the command + start of output for thread sense.
        # Rendering depends only on position-vs-end → changes ONCE when the
        # message crosses the threshold, byte-stable forever after (same
        # accepted pattern as image placeholders — one tiny re-write, then a
        # permanently smaller prefix that re-reads AND re-summarises cheaper).
        # Assistant messages are NEVER pruned — they carry decisions/reasoning.
        PRUNE_THRESHOLD = 3_200   # only genuinely large pastes
        PRUNE_KEEP      = 600     # head kept: the command + first output lines
        if (m["role"] == "user" and not keep_full
                and len(raw) > PRUNE_THRESHOLD):
            # SMART LOG PRUNING (June 11): a blind head-keep loses the real
            # error, which in stack traces / logs is usually at the BOTTOM
            # (the final "Caused by", the actual exception). So for older big
            # pastes we keep: the head (the command + first lines) PLUS every
            # line that looks like a real error/cause. This shrinks a 2000-token
            # log to ~200 tokens on follow-up turns WITHOUT losing the
            # diagnosis — same model, lower cost, no quality loss.
            err_pat = re.compile(
                r"(error|exception|caused by|failed|refused|denied|unknownhost|"
                r"x509|timeout|fatal|panic|cannot|unable to|not found|"
                r"crashloop|oomkilled|backoff|no such|connection)", re.I)
            err_lines, seen = [], set()
            for ln in raw.splitlines():
                s = ln.strip()
                if s and err_pat.search(s) and s not in seen:
                    err_lines.append(s[:200])
                    seen.add(s)
                if len(err_lines) >= 15:        # cap — newest/first 15 error lines
                    break
            head = raw[:PRUNE_KEEP]
            block = head + "\n[...long output pruned — KEY ERROR LINES kept below; " \
                           "exact names also in SESSION FACTS:]"
            if err_lines:
                block += "\n" + "\n".join("  " + e for e in err_lines)
            raw = block
        return {"role": m["role"], "content": raw}

    context = []
    if len(history) <= RAW_WINDOW:
        # Short/medium chat (the vast majority of Naveen's): send everything
        # verbatim. The cached prefix is byte-stable turn to turn, so all of it
        # is READ at 10% — not re-written. This is the cheap, correct path.
        # Only the last 2 messages keep full images; older images become tiny
        # placeholders so a 48k-char screenshot isn't re-sent every turn.
        n = len(history)
        for i, m in enumerate(history):
            context.append(cap(m, is_recent=(i >= n - 4), keep_full=(i >= n - 6)))
    else:
        # Long chat — BATCH-ALIGNED FROZEN SUMMARY (cache-stable for 200-600 msg
        # ticket chats). The killer for long chats is a cache prefix that shifts
        # every turn. We prevent that by aligning the summary boundary to fixed
        # BATCH multiples, so between batch boundaries the ENTIRE prefix (summary
        # + recent messages) is byte-identical turn to turn → Anthropic READS the
        # cache at 10% instead of RE-WRITING at 125%.
        #
        # How: summary_upto is always a multiple of BATCH. The recent window is
        # everything after that boundary. As you add messages, the recent window
        # grows from BATCH..2*BATCH, then at the next multiple we re-summarise
        # ONCE (one batch of cost) and the boundary jumps forward. Between jumps,
        # the prefix is stable.
        BATCH = 10
        n = len(history)
        # Boundary = largest multiple of BATCH that leaves at least RAW_WINDOW recent.
        boundary = ((n - RAW_WINDOW) // BATCH) * BATCH
        if boundary < 0:
            boundary = 0

        frozen_text = ""
        frozen_upto = 0
        if chat_id:
            try:
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT summary_text, summary_upto FROM chats WHERE id=?",
                        (chat_id,)
                    ).fetchone()
                if row:
                    frozen_text = row["summary_text"] or ""
                    frozen_upto = row["summary_upto"] or 0
            except Exception:
                pass

        # Re-summarise ONLY when the batch boundary has advanced (every 10 msgs).
        if boundary > 0 and (not frozen_text or frozen_upto != boundary):
            frozen_text = compress_history(history[:boundary], api_key)
            frozen_upto = boundary
            if chat_id:
                try:
                    with get_db() as conn:
                        conn.execute(
                            "UPDATE chats SET summary_text=?, summary_upto=? WHERE id=?",
                            (frozen_text, frozen_upto, chat_id)
                        )
                except Exception:
                    pass

        if frozen_text:
            context.append({"role": "user", "content": f"[EARLIER CONTEXT — compressed]\n{frozen_text}"})
            context.append({"role": "assistant", "content": "Understood, I have the earlier context."})

        # Recent messages = everything after the (stable) boundary.
        recent_msgs = history[frozen_upto:]
        rn = len(recent_msgs)
        for i, m in enumerate(recent_msgs):
            context.append(cap(m, is_recent=(i >= rn - 4), keep_full=(i >= rn - 6)))
    context.append({"role": "user", "content": new_content})

    # ── EMPTY-CONTENT GUARD (June 14) ────────────────────────────────────────
    # The Anthropic API rejects the WHOLE request if ANY message has empty
    # content ("messages.N: user messages must have non-empty content"). This
    # happened after an earlier failed turn saved an empty assistant reply, which
    # then poisoned every later request in that chat. Drop any message whose
    # content is empty/whitespace (string) or an empty blocks list. Never let a
    # blank message reach the API.
    cleaned = []
    for m in context:
        c = m.get("content", "")
        if isinstance(c, str):
            if c.strip() == "":
                continue
        elif isinstance(c, list):
            # keep only non-empty text blocks / any non-text block
            c2 = [b for b in c if (b.get("type") != "text") or (b.get("text", "").strip() != "")]
            if not c2:
                continue
            m = {**m, "content": c2}
        cleaned.append(m)
    # Safety: the final user message must exist and be non-empty
    if not cleaned or cleaned[-1].get("role") != "user":
        cleaned.append({"role": "user", "content": new_content if (isinstance(new_content, str) and new_content.strip()) else "(continue)"})
    return cleaned


_LOG_ERR_PAT = re.compile(
    r"(error|exception|caused by|failed|failure|refused|denied|unknownhost|x509|"
    r"timeout|timed out|fatal|panic|cannot|unable to|not found|crashloop|oomkilled|"
    r"backoff|no such|connection|unauthorized|forbidden|warn|critical|abort|"
    r"unresolved|missing|invalid|bind|listen)", re.I)


def _compress_pasted_log(text: str, head_lines: int = 8, tail_lines: int = 6,
                         max_err: int = 40) -> str:
    """Compress a large pasted log/output for the MODEL while the DB keeps the
    full original. Strategy that preserves diagnosis:
      • keep the first head_lines (the command + how it started)
      • keep EVERY line matching a real error/cause pattern (deduped, capped)
      • keep the last tail_lines (where the fatal error / shutdown usually is)
    A 300-line Java stacktrace or kubectl dump collapses to ~30 meaningful lines.
    """
    lines = text.splitlines()
    # Char-dense paste (huge but few newlines, e.g. a wall-of-text log or a
    # minified blob): line-based compression won't help, so fall back to a
    # head+tail char clamp that still preserves the start (command/context) and
    # end (the fatal error). Keeps ~3,500 chars instead of 10,000+.
    if len(lines) < 30:
        if len(text) > 5000:
            return text[:2200] + (
                f"\n  [... {len(text)-3700} chars of this paste compressed for the "
                f"model — full original preserved in chat ...]\n") + text[-1500:]
        return text
    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    err, seen = [], set()
    for ln in lines[head_lines:-tail_lines] if len(lines) > head_lines + tail_lines else []:
        s = ln.strip()
        if not s:
            continue
        # skip the repetitive "at com.foo.Bar(...)" stacktrace frames — keep the
        # exception/cause lines, drop the 40-deep call stack noise
        if s.startswith("at ") or s.startswith("... "):
            continue
        if _LOG_ERR_PAT.search(s) and s not in seen:
            err.append(s[:240])
            seen.add(s)
        if len(err) >= max_err:
            break
    out = []
    out.extend(head)
    out.append(f"  [... {len(lines)} total lines compressed — key error/cause lines kept "
               f"below; full original preserved in chat ...]")
    out.extend("  " + e for e in err)
    out.append("  [... tail ...]")
    out.extend(tail)
    compressed = "\n".join(out)
    # never let "compression" make it bigger
    return compressed if len(compressed) < len(text) else text


def smart_truncate(text: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Keep start + end of file — most diagnostic for logs and configs."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        text[:half]
        + f"\n\n── [Truncated {omitted:,} characters for token efficiency] ──\n\n"
        + text[-half:]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION FACTS ENGINE (Phase 1 quality fix — June 10, 2026)
#
#  PROBLEM IT SOLVES: in real ticket sessions the model "forgot" exact pod
#  names shown earlier (wrote a wrong pod name mid-ticket) because old raw
#  messages get capped/summarised. Result: confusion → follow-up questions →
#  wasted money.
#
#  HOW: pure-regex extraction (zero API cost) of exact entity names from every
#  user message — namespaces, pod names, K8s resources, file paths, error
#  lines. Stored per-chat in chats.session_facts (JSON). A compact block is
#  prepended to the USER message each turn (same cache-safe pattern as KB
#  retrieval — fresh input at 1x, NEVER in the system block, so the two-block
#  cache from the June 9 fix stays untouched).
#
#  Topic-independent by design: works for Kafka, Keycloak, Nexus, HAProxy or
#  any future ticket — no per-topic prompt patches needed.
# ─────────────────────────────────────────────────────────────────────────────
FACTS_CAPS = {"namespaces": 12, "pods": 30, "resources": 40, "files": 20,
              "errors": 6, "progress": 20}

# kubectl get pods table row:  NAME  READY  STATUS  RESTARTS  AGE
_RE_POD_ROW   = re.compile(r"^([a-z0-9][a-z0-9.-]{2,62})\s+\d+/\d+\s+[A-Za-z]+", re.M)
_RE_POD_REF   = re.compile(r"\bpod/([a-z0-9][a-z0-9.-]{2,62})\b")
_RE_NS_FLAG   = re.compile(r"(?<![\w/-])(?:-n\s+|--namespace[= ]\s*)([a-z0-9][a-z0-9-]{0,62})\b")
_RE_NS_COL    = re.compile(r"^\s*namespace:\s*([a-z0-9][a-z0-9-]{0,62})\b", re.M | re.I)
_RE_RESOURCE  = re.compile(
    r"(?<![\w/.-])(secret|configmap|svc|service|deploy|deployment|statefulset|sts|ds|daemonset|"
    r"ingress|pvc|kafkatopic|kafkauser|kafka)/([A-Za-z0-9][\w.-]{1,62})(?![\w/])", re.I)
_RE_FILEPATH  = re.compile(
    r"(/[\w@%+=:,.~\-/]+\.(?:crt|key|pem|csr|jks|p12|pfx|yaml|yml|conf|cnf|"
    r"properties|toml|ini|env|json|sh|service|log))\b")
_RE_ERRORLINE = re.compile(
    r"^.*(?:error|failed|denied|refused|x509|timeout|timed out|unauthorized|"
    r"forbidden|crashloopbackoff|imagepullbackoff|oomkilled).*$", re.M | re.I)

# Natural-language progress: "I created 4 secrets", "applied the configmap",
# "the storage class is done", "namespace exists now". These are facts about
# what's ALREADY DONE in this ticket — so the model never asks "show me the
# secret/configmap/storage class" again after the user already did it.
_RE_PROGRESS = re.compile(
    r"\b(?:i\s+)?(?:have\s+)?(?:created|made|applied|added|deployed|set up|"
    r"set-up|configured|done|finished|completed|verified|confirmed|showed|"
    r"shown|already (?:have|did|created|applied))\b[^.!?\n]{0,90}",
    re.I)

# noise that the pod-row regex can false-positive on (table headers, words)
_FACTS_NOISE = {"name", "ready", "status", "restarts", "age", "running", "pending",
                "error", "no", "the", "this", "with", "from", "completed"}


def update_session_facts(facts_json: str, user_text: str) -> str:
    """Merge entities found in user_text into the chat's facts JSON.
    Newest entries win; each category capped (FACTS_CAPS) FIFO."""
    try:
        facts = json.loads(facts_json) if facts_json else {}
    except (json.JSONDecodeError, TypeError):
        facts = {}
    for k in FACTS_CAPS:
        facts.setdefault(k, [])

    def _add(cat: str, val: str):
        val = val.strip()
        if not val or val.lower() in _FACTS_NOISE:
            return
        if val in facts[cat]:
            facts[cat].remove(val)          # re-adding moves it to "most recent"
        facts[cat].append(val)
        facts[cat] = facts[cat][-FACTS_CAPS[cat]:]   # keep newest N

    t = user_text or ""
    if len(t) < 8:
        return json.dumps(facts)

    for m in _RE_NS_FLAG.findall(t):  _add("namespaces", m)
    for m in _RE_NS_COL.findall(t):   _add("namespaces", m)
    for m in _RE_POD_ROW.findall(t):  _add("pods", m)
    for m in _RE_POD_REF.findall(t):  _add("pods", m)
    for kind, name in _RE_RESOURCE.findall(t):
        _add("resources", f"{kind.lower()}/{name}")
    for m in _RE_FILEPATH.findall(t): _add("files", m)
    for m in _RE_ERRORLINE.findall(t)[:5]:
        _add("errors", " ".join(m.split())[:140])    # collapse whitespace, cap length
    for m in _RE_PROGRESS.findall(t)[:6]:
        _add("progress", " ".join(m.split())[:110])

    return json.dumps(facts)


def facts_format_for_prompt(facts_json: str, char_budget: int = 900) -> str:
    """Render the compact SESSION FACTS block prepended to the user message.
    Empty string when nothing extracted yet → zero token cost on simple chats."""
    try:
        facts = json.loads(facts_json) if facts_json else {}
    except (json.JSONDecodeError, TypeError):
        return ""
    if not any(facts.get(k) for k in FACTS_CAPS):
        return ""
    lines = ["[SESSION FACTS — exact names and progress from THIS ticket, "
             "auto-extracted from Naveen's messages. Copy names character-for-"
             "character. The 'already done' items below are FINISHED — do NOT "
             "ask him to show or redo them; build on them.]"]
    label = {"namespaces": "namespaces", "pods": "pods", "resources": "resources",
             "files": "files", "errors": "recent errors", "progress": "already done"}
    for k in ("namespaces", "pods", "resources", "files", "errors", "progress"):
        if facts.get(k):
            lines.append(f"{label[k]}: " + " | ".join(facts[k]))
    block = "\n".join(lines)
    return block[:char_budget + 300] + "\n"


def auto_title(first_message: str) -> str:
    clean = re.sub(r"\s+", " ", first_message.strip())
    return clean[:65] + ("…" if len(clean) > 65 else "")


def _build_system_blocks(stable: str, volatile: str) -> list:
    """Two cached system blocks. Block 1 = stable prompt (cache survives full
    1h TTL, reused every question). Block 2 = volatile prefs/memory/files
    (re-caches only when it changes). This stops a saved memory or preference
    from busting the whole prompt cache — the core cost fix."""
    blocks = [{
        "type": "text",
        "text": stable,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }]
    if volatile and volatile.strip():
        blocks.append({
            "type": "text",
            "text": volatile,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })
    return blocks


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


@app.route("/api/cache/keepwarm", methods=["POST"])
def api_cache_keepwarm():
    """Keep the prompt cache alive while Naveen is away in his terminal (June 14).
    His expensive re-payments happen when he steps away (to run kubectl / read
    logs) longer than the cache TTL, then returns and pays the full ~9¢ cache
    WRITE again. This sends a minimal request that REUSES the cached system
    prompt — refreshing the 1h TTL clock for ~0.0003¢ (a cache READ of the prefix
    + 1 output token). The UI pings this every ~10 min while a chat is open and
    idle. Net effect: come back after 30-40 min and the next real question is
    still 2-3¢, not 9¢.
    """
    try:
        s = load_settings()
        api_key = active_claude_key(s)
        if not api_key:
            return jsonify({"ok": False, "reason": "no key"}), 200
        # Cache is MODEL-SPECIFIC — must warm the same model the chat uses, or it
        # does nothing. Default to the user's selected model (usually Sonnet).
        warm_model = s.get("default_model", "claude-sonnet-4-6")
        minfo = MODEL_MAP.get(warm_model, {})
        if minfo.get("provider") != "claude":
            return jsonify({"ok": False, "reason": "non-claude model"}), 200
        stable_system = SYSTEM_PROMPT
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model=warm_model,
            max_tokens=1,
            system=[{
                "type": "text",
                "text": stable_system,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            messages=[{"role": "user", "content": "."}],
            extra_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
        )
        return jsonify({"ok": True, "model": warm_model}), 200
    except Exception as e:
        log.debug("keepwarm ping failed (non-fatal): %s", e)
        return jsonify({"ok": False, "reason": str(e)[:80]}), 200


def calc_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    m = MODEL_MAP.get(model_id)
    if not m or m.get("free"):
        return 0.0
    return (tokens_in / 1_000_000 * m["in"]) + (tokens_out / 1_000_000 * m["out"])


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory  = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT    PRIMARY KEY,
            title       TEXT    NOT NULL,
            model       TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            cost_usd    REAL    NOT NULL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT    PRIMARY KEY,
            chat_id     TEXT    NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role        TEXT    NOT NULL CHECK(role IN ('user','assistant')),
            content     TEXT    NOT NULL,
            tokens_in   INTEGER NOT NULL DEFAULT 0,
            tokens_out  INTEGER NOT NULL DEFAULT 0,
            model       TEXT,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_chat_time
            ON messages(chat_id, created_at);

        -- Supports the spend/today + per-provider counter queries which filter
        -- on role='assistant' AND created_at (added June 14 with split counters).
        -- The chat_time index above doesn't cover role-first scans; this keeps
        -- the daily cost rollup fast as message volume grows.
        CREATE INDEX IF NOT EXISTS idx_messages_role_time
            ON messages(role, created_at);

        CREATE TABLE IF NOT EXISTS memory (
            id         TEXT PRIMARY KEY,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS folders (
            id         TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS preferences (
            id         TEXT PRIMARY KEY,
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- ════════════════════════════════════════════════════════════════════
        -- KNOWLEDGE BASE — the "claude.ai memory" equivalent
        -- ════════════════════════════════════════════════════════════════════
        -- Three tables that turn your past work into searchable institutional
        -- knowledge that gets injected into every relevant answer:
        --   incidents  — resolved problems (symptoms → root cause → fix)
        --   runbooks   — procedures for known scenarios
        --   kb_docs    — official documentation chunks (Strimzi, Keycloak, ...)
        -- Each has an FTS5 mirror so a question like "Kafka SSL handshake" pulls
        -- the relevant past tickets instantly via keyword + BM25 ranking.

        CREATE TABLE IF NOT EXISTS incidents (
            id           TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            symptoms     TEXT NOT NULL,
            root_cause   TEXT NOT NULL,
            fix          TEXT NOT NULL,
            validation   TEXT DEFAULT '',
            tags         TEXT DEFAULT '',
            references_  TEXT DEFAULT '',
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runbooks (
            id           TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            scenario     TEXT NOT NULL,
            steps        TEXT NOT NULL,
            tags         TEXT DEFAULT '',
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kb_docs (
            id           TEXT PRIMARY KEY,
            source       TEXT NOT NULL,
            title        TEXT NOT NULL,
            content      TEXT NOT NULL,
            tags         TEXT DEFAULT '',
            created_at   TEXT NOT NULL
        );

        -- Version-aware topic timeline (Keycloak, Nexus, …).
        -- One topic = one subject; many layers = its history over time.
        -- Newest layer is_current=1; older layers kept as history so the
        -- full arc (e.g. Nexus 3.68 → 3.70 → 3.94) is always available.
        CREATE TABLE IF NOT EXISTS topics (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL UNIQUE,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topic_layers (
            id           TEXT PRIMARY KEY,
            topic_id     TEXT NOT NULL,
            label        TEXT NOT NULL,      -- e.g. "Nexus 3.94"
            summary      TEXT DEFAULT '',    -- free local summary shown first
            content      TEXT NOT NULL,      -- merged text of all folder files
            file_tree    TEXT DEFAULT '',    -- newline list of ingested paths
            file_count   INTEGER DEFAULT 0,
            is_current   INTEGER DEFAULT 1,  -- 1 = latest, 0 = history
            created_at   TEXT NOT NULL
        );

        -- Unified KB versioning: append new versions onto ANY existing KB entry.
        -- entry_ref = the original id of the incident/runbook/doc/version it
        -- attaches to. Newest version is_current=1; older kept as history.
        -- The existing 6 entries are never moved — this layers on top of them.
        CREATE TABLE IF NOT EXISTS kb_versions (
            id           TEXT PRIMARY KEY,
            entry_ref    TEXT NOT NULL,      -- id of the KB entry this versions
            version_no   INTEGER NOT NULL,   -- 2,3,4… (v1 = the original entry)
            label        TEXT NOT NULL,      -- e.g. "Nexus 3.94 deployment"
            summary      TEXT DEFAULT '',
            content      TEXT NOT NULL,
            file_tree    TEXT DEFAULT '',
            file_count   INTEGER DEFAULT 0,
            is_current   INTEGER DEFAULT 1,
            created_at   TEXT NOT NULL
        );

        -- FTS5 mirrors for fast keyword + BM25 retrieval (no embeddings needed
        -- at this scale; FTS5 with BM25 gives ~90% of vector-search quality
        -- for tens of thousands of docs, with zero new deps and zero infra).
        CREATE VIRTUAL TABLE IF NOT EXISTS incidents_fts USING fts5(
            title, symptoms, root_cause, fix, validation, tags,
            content='incidents', content_rowid='rowid'
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS runbooks_fts USING fts5(
            title, scenario, steps, tags,
            content='runbooks', content_rowid='rowid'
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS kb_docs_fts USING fts5(
            source, title, content, tags,
            content='kb_docs', content_rowid='rowid'
        );

        -- Standalone (not content-linked) FTS for kb_versions — the folder-upload
        -- content that previously had ZERO search wiring. Kept in sync manually
        -- at insert/delete time (simpler & safer than triggers for a table with
        -- frequent inserts as versions accumulate).
        CREATE VIRTUAL TABLE IF NOT EXISTS kb_versions_fts USING fts5(
            version_id UNINDEXED, entry_ref UNINDEXED, entry_title, label, content
        );
        INSERT OR IGNORE INTO settings (key, value) VALUES ('lifetime_tokens', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('lifetime_cost', '0.0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('web_search_enabled', 'false');
        """)
    log.info("Database ready: %s", DB_PATH)

    # ── Backfill kb_versions_fts for versions uploaded before search wiring existed ──
    try:
        with get_db() as _conn:
            _missing_v = _conn.execute(
                "SELECT v.id,v.entry_ref,v.label,v.content,t.title "
                "FROM kb_versions v "
                "LEFT JOIN (SELECT id,title FROM incidents "
                "           UNION SELECT id,title FROM runbooks "
                "           UNION SELECT id,title FROM kb_docs) t ON t.id=v.entry_ref "
                "WHERE v.id NOT IN (SELECT version_id FROM kb_versions_fts)"
            ).fetchall()
            for _v in _missing_v:
                _conn.execute(
                    "INSERT INTO kb_versions_fts (version_id,entry_ref,entry_title,label,content) "
                    "VALUES (?,?,?,?,?)",
                    (_v["id"], _v["entry_ref"], _v["title"] or "", _v["label"], _v["content"]))
            if _missing_v:
                log.info("Backfilled %d kb_versions into FTS (previously unsearchable)", len(_missing_v))
    except Exception as _e:
        log.info("kb_versions_fts backfill skipped: %s", _e)

    # ── Backfill embeddings for existing KB entries (runs once after Ollama install) ─
    # If Ollama is not running, get_embedding returns [] and this is a no-op.
    try:
        _backfill_done = False
        with get_db() as _conn:
            _missing_inc = _conn.execute("SELECT id,title,symptoms,root_cause FROM incidents WHERE embedding IS NULL OR embedding='[]' LIMIT 50").fetchall()
            for _r in _missing_inc:
                _et = f"{_r['title']}. Symptoms: {_r['symptoms']}. Root cause: {_r['root_cause']}"
                _emb = get_embedding(_et)
                if _emb:
                    _conn.execute("UPDATE incidents SET embedding=? WHERE id=?", (json.dumps(_emb), _r['id']))
                    _backfill_done = True
            _missing_rb = _conn.execute("SELECT id,title,scenario,steps FROM runbooks WHERE embedding IS NULL OR embedding='[]' LIMIT 50").fetchall()
            for _r in _missing_rb:
                _et = f"{_r['title']}. {_r['scenario']}. {_r['steps'][:500]}"
                _emb = get_embedding(_et)
                if _emb:
                    _conn.execute("UPDATE runbooks SET embedding=? WHERE id=?", (json.dumps(_emb), _r['id']))
                    _backfill_done = True
            _missing_doc = _conn.execute("SELECT id,source,title,content FROM kb_docs WHERE embedding IS NULL OR embedding='[]' LIMIT 50").fetchall()
            for _r in _missing_doc:
                _et = f"{_r['title']} ({_r['source']}). {_r['content'][:1000]}"
                _emb = get_embedding(_et)
                if _emb:
                    _conn.execute("UPDATE kb_docs SET embedding=? WHERE id=?", (json.dumps(_emb), _r['id']))
                    _backfill_done = True
        if _backfill_done:
            log.info("Startup: backfilled semantic embeddings for KB entries ✓")
    except Exception as _be:
        log.debug("Embedding backfill skipped (Ollama not running): %s", _be)

    # ── Migration: soft-delete support ───────────────────────────────────────
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN deleted_at TEXT")
        log.info("Migration: added deleted_at column to chats")
    except Exception:
        pass  # Column already exists — safe to ignore

    # ── Migration: folder support (additive, null = unfiled) ─────────────────
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN folder_id TEXT")
        log.info("Migration: added folder_id column to chats")
    except Exception:
        pass  # Column already exists — safe to ignore

    # ── Migration: frozen incremental summary (cache-stability fix) ──────────
    # summary_text = the locked summary of old messages; summary_upto = how many
    # messages it covers. We only RE-compress when a batch of new messages ages
    # out — never every turn — so the cached prefix stays byte-stable.
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN summary_text TEXT DEFAULT ''")
        log.info("Migration: added summary_text to chats")
    except Exception:
        pass
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN summary_upto INTEGER DEFAULT 0")
        log.info("Migration: added summary_upto to chats")
    except Exception:
        pass

    # ── Migration: semantic embeddings for KB tables (nomic-embed-text via Ollama) ─
    # Each KB entry stores a 768-dim embedding vector as JSON text. This enables
    # semantic search (understands meaning, not just keywords) on top of FTS5.
    # Gracefully absent — FTS5 search works fine if Ollama is not running.
    for _tbl in ("incidents", "runbooks", "kb_docs"):
        try:
            with get_db() as conn:
                conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN embedding TEXT DEFAULT NULL")
            log.info("Migration: added embedding column to %s", _tbl)
        except Exception:
            pass  # column already exists

    # ── Migration: per-message cost (for accurate daily spend) ───────────────
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE messages ADD COLUMN cost_usd REAL DEFAULT 0.0")
        log.info("Migration: added cost_usd column to messages")
    except Exception:
        pass  # already exists

    # ── Migration: SESSION FACTS — per-chat infra ground truth ───────────────
    # JSON dict of exact entity names (pods, namespaces, files, resources,
    # errors) auto-extracted from Naveen's pasted command outputs. Pinned into
    # every request so the model NEVER retypes a pod name from memory.
    # Fixes the "wrong pod name mid-ticket" drift observed in real sessions.
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN session_facts TEXT DEFAULT ''")
        log.info("Migration: added session_facts column to chats")
    except Exception:
        pass  # already exists

    # ── Migration: VOLATILE BLOCK FREEZE (Phase 2) ────────────────────────────
    # Snapshot of the prefs+memory portion of the volatile system block, frozen
    # per batch window. Mid-chat "remember this" saves no longer change the
    # system blocks instantly (which busted the message cache → 1.25x re-write
    # of all history). The snapshot refreshes at batch boundaries — exactly when
    # the prefix re-writes anyway — so the refresh is free.
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE chats ADD COLUMN volatile_snapshot TEXT DEFAULT ''")
            conn.execute("ALTER TABLE chats ADD COLUMN volatile_boundary INTEGER DEFAULT -1")
        log.info("Migration: added volatile_snapshot/volatile_boundary to chats")
    except Exception:
        pass  # already exists

    # ── Migration: CACHE FORENSICS (Phase 2) ─────────────────────────────────
    # Store cache_read / cache_created per assistant message so cache-bust
    # events are directly visible in SQL instead of inferred from cost spikes.
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE messages ADD COLUMN cache_read INTEGER DEFAULT 0")
            conn.execute("ALTER TABLE messages ADD COLUMN cache_created INTEGER DEFAULT 0")
        log.info("Migration: added cache_read/cache_created to messages")
    except Exception:
        pass  # already exists

    # ── Seed default preferences (only if preferences table is empty) ────────
    # These encode what Naveen has taught over weeks of use. He can edit/delete
    # any of them from the Memory tab, and the tool learns new ones automatically.
    try:
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
            if count == 0:
                seeds = [
                    "Always tell me WHERE a workload/resource comes from (which operator, Helm chart, or install method created it) and give the exact command to verify it myself.",
                    "Answer so I can defend it in a scrutiny meeting — when relevant, give me a short 'what to say in the scrutiny meeting' script.",
                    "Resolve my actual confusion, not my literal words. If I ask 'where does X come from' or 'I ran a command and X wasn't there', trace it to the real source and explain why my command missed it.",
                    "Be surgical and committed: one best path, traced to root, with the exact command and what its output proves. Never vague 'it could be this or that'.",
                    "Point me to the authoritative source of truth (the operator's/project's official docs) so I know where to look instead of memorizing everything.",
                    "Teach me the WHY like a senior explains to a sharp colleague — I am leveling up to be the top engineer at my company, aiming to understand 90%+ of what I do.",
                ]
                for s in seeds:
                    conn.execute(
                        "INSERT INTO preferences (id, content, created_at) VALUES (?, ?, ?)",
                        (str(uuid.uuid4()), s, now_iso())
                    )
                log.info("Seeded %d default preferences", len(seeds))
    except Exception as exc:
        log.warning("Preference seed skipped: %s", exc)


init_db()

# Load custom models persisted via /api/models/add
try:
    _custom = json.loads(load_settings().get("custom_models", "[]"))
    _existing = {m["id"] for m in MODELS}
    for cm in _custom:
        if cm.get("id") and cm["id"] not in _existing:
            MODELS.append(cm); _existing.add(cm["id"])
    if _custom: log.info("Loaded %d custom model(s)", len(_custom))
except Exception: pass



def load_settings() -> dict:
    with get_db() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def save_setting(key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )


def active_claude_key(s: dict = None) -> str:
    """Return the Claude API key to use right now (June 14 — backup-key feature).
    Naveen has a primary key (~$13-14 credit) and a gifted backup key ($100). When
    the primary runs out he flips 'use_backup_key' in Settings and every Claude
    call transparently uses the backup instead — without touching any of the call
    sites' logic. If the toggle is on but no backup key is saved, fall back to the
    primary so nothing breaks. Pass a pre-loaded settings dict to avoid a re-read."""
    if s is None:
        s = load_settings()
    use_backup = str(s.get("use_backup_key", "false")).lower() == "true"
    backup = s.get("api_key_2", "")
    if use_backup and backup:
        return backup
    return s.get("api_key", "")


def get_chat_history(chat_id: str) -> list:
    """Load chat messages, filtering out error/broken entries."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE chat_id = ? ORDER BY created_at",
            (chat_id,)
        ).fetchall()

    result = []
    for r in rows:
        try:
            content = json.loads(r["content"])
        except (json.JSONDecodeError, TypeError):
            content = r["content"]

        # Skip error messages — they waste tokens and confuse context
        if isinstance(content, str) and (
            content.startswith("error:") or
            "Error code:" in content or
            content.startswith("❌")
        ):
            continue

        result.append({"role": r["role"], "content": content})
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_all_memories():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at FROM memory ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_preferences():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at FROM preferences ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ═════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — Incidents, Runbooks, Docs (FTS5-backed retrieval)
# ═════════════════════════════════════════════════════════════════════════════

def _fts_escape(query: str) -> str:
    """Sanitise a user question for FTS5 MATCH. Strips operators that would be
    interpreted as syntax (parens, quotes, AND/OR/NOT punctuation), keeps only
    word characters as OR-joined keywords. Drops short noise words.
    """
    import re as _re
    # Get word-like tokens of length >= 3 (filters 'is/the/a' etc.)
    tokens = _re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
    NOISE = {"the","and","for","with","this","that","what","when","where","which",
             "from","into","have","has","not","but","you","your","why","how","are",
             "was","were","will","can","should","could","would","they","them",
             "their","there","here","just","only","also","some","any","all","one",
             "two","new","old","get","got","make","made","run","ran","use","used"}
    keywords = [t.lower() for t in tokens if t.lower() not in NOISE]
    # Cap to top 12 most distinctive (longer tokens usually more distinctive)
    keywords = sorted(set(keywords), key=lambda k: (-len(k), k))[:12]
    if not keywords:
        return ""
    # OR-join — broadens recall; BM25 ranks the relevant rows up top
    return " OR ".join(keywords)


# ═════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SEARCH — nomic-embed-text via Ollama (free, local, 274MB)
# ═════════════════════════════════════════════════════════════════════════════
# This replaces keyword-only FTS5 search with HYBRID search:
#   1. Semantic (vector) search — finds conceptually related incidents even
#      when the words don't match. "Kafka SSL handshake" matches past incident
#      about "broker TLS certificate rotation" because they mean the same thing.
#   2. FTS5 keyword search — still runs as fallback and supplement.
# If Ollama is not running, get_embedding() returns [] and we fall back to FTS5.
# Zero new Python deps — uses only stdlib urllib for the HTTP call.

import math as _math
import urllib.request as _ureq

_OLLAMA_URL   = "http://localhost:11434/api/embeddings"
_OLLAMA_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list:
    """Call local Ollama API for a 768-dim embedding. Returns [] if not running."""
    if not text or not text.strip():
        return []
    try:
        payload = json.dumps({"model": _OLLAMA_MODEL, "prompt": text[:2000]}).encode()
        req = _ureq.Request(_OLLAMA_URL, data=payload,
                            headers={"Content-Type": "application/json"})
        with _ureq.urlopen(req, timeout=3) as r:
            return json.loads(r.read()).get("embedding", [])
    except Exception:
        return []          # Ollama not running → FTS5 takes over silently


def _cosine(a: list, b_json: str) -> float:
    """Cosine similarity between query vector a and stored JSON blob b."""
    if not a or not b_json:
        return 0.0
    try:
        b = json.loads(b_json)
        dot  = sum(x * y for x, y in zip(a, b))
        ma   = _math.sqrt(sum(x * x for x in a))
        mb   = _math.sqrt(sum(x * x for x in b))
        return dot / (ma * mb) if ma and mb else 0.0
    except Exception:
        return 0.0


def _semantic_ranked(table, emb_col, query_emb, cols, k, threshold=0.25):
    """Return ranked [(id, dict)] by cosine similarity, above a LOW floor.
    The floor only removes pure noise; RRF fusion does the real precision work,
    so we keep this permissive and let rank fusion sort relevance out."""
    if not query_emb:
        return []
    try:
        with get_db() as conn:
            cnt = conn.execute(f"SELECT COUNT(*) AS c FROM {table} "
                               f"WHERE {emb_col} IS NOT NULL").fetchone()["c"]
            cached = _EMB_CACHE.get(table)
            if cached is None or cached["count"] != cnt:
                rows = conn.execute(
                    f"SELECT {cols}, {emb_col} FROM {table} WHERE {emb_col} IS NOT NULL"
                ).fetchall()
                loaded = []
                for r in rows:
                    d = dict(r)
                    emb_json = d.pop(emb_col, None)
                    loaded.append((d, emb_json))
                _EMB_CACHE[table] = {"count": cnt, "rows": loaded}
                cached = _EMB_CACHE[table]
        scored = []
        for d, emb_json in cached["rows"]:
            sim = _cosine(query_emb, emb_json)
            if sim >= threshold:
                scored.append((sim, d))
        scored.sort(key=lambda x: -x[0])
        return [(d["id"], d) for _, d in scored[:k * 2]]   # keep extra for fusion
    except Exception as e:
        log.warning("semantic ranking failed: %s", e)
        return []


def _fts_ranked(table, fts_table, select_sql, q, k):
    """Return ranked [(id, dict)] from FTS5 BM25 (best keyword match first)."""
    if not q:
        return []
    try:
        with get_db() as conn:
            rows = conn.execute(select_sql, (q, k * 2)).fetchall()
        return [(r["id"], dict(r)) for r in rows]
    except Exception as e:
        log.warning("FTS ranking failed (%s): %s", table, e)
        return []


def _rrf_fuse(semantic_ranked, keyword_ranked, k, rrf_c=60):
    """Reciprocal Rank Fusion — the production gold standard for hybrid search.

    Instead of comparing a cosine score (0-1) against a BM25 score (unbounded) —
    apples to oranges — RRF scores each item by its RANK POSITION in each list:
        score(item) = sum over lists of  1 / (rrf_c + rank_in_that_list)
    An item ranked high in BOTH semantic AND keyword wins decisively. An item
    that's only weakly semantic (e.g. a Grafana incident matching 'kafka' just
    because both are K8s) has NO keyword rank, so it scores low and drops out.
    This is exactly what fixes the 'kafka search returns Grafana' noise.
    rrf_c=60 is the standard constant from the original RRF paper."""
    scores = {}
    items = {}
    for rank, (iid, d) in enumerate(semantic_ranked):
        scores[iid] = scores.get(iid, 0.0) + 1.0 / (rrf_c + rank)
        items[iid] = d
    for rank, (iid, d) in enumerate(keyword_ranked):
        scores[iid] = scores.get(iid, 0.0) + 1.0 / (rrf_c + rank)
        items.setdefault(iid, d)
    ranked_ids = sorted(scores, key=lambda i: -scores[i])
    if not ranked_ids:
        return []
    # PRECISION GUARD: drop tail items scoring far below the top match. When a
    # strong match exists (e.g. the real Kafka incident appearing in both lists),
    # loosely-related padding (a Grafana incident with only a weak semantic rank)
    # falls below this cutoff and is excluded — so we inject signal, not noise.
    top = scores[ranked_ids[0]]
    keep = [i for i in ranked_ids[:k] if scores[i] >= top * 0.5]
    return [items[i] for i in keep]


def kb_search(query: str, k: int = 5) -> dict:
    """Hybrid retrieval with Reciprocal Rank Fusion (RRF) — production gold standard.

    Runs semantic (Ollama embeddings) AND keyword (FTS5/BM25) in parallel, then
    fuses by RANK position via RRF. This fixes the old bug where semantic results
    flooded in loosely-related items (a 'kafka' search returning Grafana/Nexus
    incidents just because they're all Kubernetes). With RRF, an item must rank
    well in semantic OR keyword to surface, and items strong in BOTH win — so
    'kafka' returns Kafka, not everything K8s-shaped.

    Fully graceful: Ollama down → keyword-only; no FTS match → semantic-only.
    """
    q = _fts_escape(query)
    out = {"incidents": [], "runbooks": [], "docs": [], "versioned": []}
    if not q and not query.strip():
        return out

    query_emb = get_embedding(query)
    mode = "hybrid-rrf" if (query_emb and q) else ("semantic" if query_emb else "fts5")

    # ── INCIDENTS ────────────────────────────────────────────────────────────
    sem_i = _semantic_ranked(
        "incidents", "embedding", query_emb,
        "id,title,symptoms,root_cause,fix,validation,tags", k)
    kw_i = _fts_ranked(
        "incidents", "incidents_fts",
        "SELECT i.id,i.title,i.symptoms,i.root_cause,i.fix,i.validation,i.tags "
        "FROM incidents i JOIN incidents_fts f ON f.rowid=i.rowid "
        "WHERE incidents_fts MATCH ? ORDER BY bm25(incidents_fts) LIMIT ?", q, k)
    out["incidents"] = _rrf_fuse(sem_i, kw_i, k)

    # ── RUNBOOKS ─────────────────────────────────────────────────────────────
    sem_r = _semantic_ranked(
        "runbooks", "embedding", query_emb,
        "id,title,scenario,steps,tags", k)
    kw_r = _fts_ranked(
        "runbooks", "runbooks_fts",
        "SELECT b.id,b.title,b.scenario,b.steps,b.tags "
        "FROM runbooks b JOIN runbooks_fts f ON f.rowid=b.rowid "
        "WHERE runbooks_fts MATCH ? ORDER BY bm25(runbooks_fts) LIMIT ?", q, k)
    out["runbooks"] = _rrf_fuse(sem_r, kw_r, k)

    # ── DOCS ─────────────────────────────────────────────────────────────────
    sem_d = _semantic_ranked(
        "kb_docs", "embedding", query_emb,
        "id,source,title,substr(content,1,1200) AS snippet,tags", k)
    kw_d = _fts_ranked(
        "kb_docs", "kb_docs_fts",
        "SELECT d.id,d.source,d.title,substr(d.content,1,1200) AS snippet,d.tags "
        "FROM kb_docs d JOIN kb_docs_fts f ON f.rowid=d.rowid "
        "WHERE kb_docs_fts MATCH ? ORDER BY bm25(kb_docs_fts) LIMIT ?", q, k)
    out["docs"] = _rrf_fuse(sem_d, kw_d, k)

    # ── VERSIONED ENTRIES (Stage 3) ─────────────────────────────────────────
    # Folder-uploaded content that supersedes an older KB entry. Keyword-only
    # (no embeddings on this large-content table by design — keeps ingest free
    # and fast). Groups matches by entry_ref so CURRENT and HISTORY of the
    # SAME topic travel together — this is what lets the model reason
    # "this supersedes that" instead of seeing two unrelated hits.
    out["versioned"] = []
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT vf.entry_ref, vf.entry_title, vf.label, "
                "substr(vf.content,1,3000) AS snippet, bm25(kb_versions_fts) AS score "
                "FROM kb_versions_fts vf WHERE kb_versions_fts MATCH ? "
                "ORDER BY score LIMIT ?", (q or query, k)).fetchall() if (q or query.strip()) else []
            seen_refs = set()
            for r in rows:
                ref = r["entry_ref"]
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                vers = conn.execute(
                    "SELECT version_no,label,summary,is_current,created_at "
                    "FROM kb_versions WHERE entry_ref=? ORDER BY version_no DESC",
                    (ref,)).fetchall()
                if not vers:
                    continue
                cur = next((v for v in vers if v["is_current"]), vers[0])
                cur_full = conn.execute(
                    "SELECT content FROM kb_versions WHERE entry_ref=? AND version_no=?",
                    (ref, cur["version_no"])).fetchone()

                # HISTORY DEPTH: normally we send only labels for older versions
                # (keeps every routine answer cheap and unambiguous). But when
                # Naveen is explicitly asking about origin/lineage/evolution, we
                # attach the FULL historical content — nothing is ever lost, it
                # is always exactly one question away, word for word.
                _ql = (query or "").lower()
                wants_history = any(w in _ql for w in (
                    "history", "historical", "lineage", "evolution", "evolved",
                    "originally", "original", "how did", "how we got", "started",
                    "beginning", "from scratch", "whole story", "full story",
                    "previous version", "older version", "earlier version",
                    "used to", "before we", "back then", "over time", "timeline",
                    "what changed", "what has changed", "since", "compare"))

                hist = []
                for v in vers:
                    if v["is_current"]:
                        continue
                    h = {"version_no": v["version_no"], "label": v["label"],
                         "summary": v["summary"]}
                    if wants_history:
                        _hc = conn.execute(
                            "SELECT content FROM kb_versions WHERE entry_ref=? AND version_no=?",
                            (ref, v["version_no"])).fetchone()
                        h["content"] = (_hc["content"] if _hc else "")[:6000]
                    hist.append(h)
                out["versioned"].append({
                    "_ref": ref,
                    "entry_title": r["entry_title"] or "(untitled)",
                    "current_label": cur["label"],
                    "current_content": (cur_full["content"] if cur_full else "")[:4000],
                    "history": hist,
                    "history_full": wants_history,
                })
                if len(out["versioned"]) >= 3:
                    break
    except Exception as _e:
        log.info("versioned-entry search skipped: %s", _e)

    # Suppress the raw incident/runbook/doc entry when it ALREADY has a current
    # version — otherwise the old content shows up unlabeled right next to the
    # correctly-framed CURRENT block, recreating the exact confusion Stage 3
    # exists to fix (model sees two Nexus blocks with no signal one is stale).
    versioned_refs = {v.get("_ref") for v in out["versioned"]} if out["versioned"] else set()
    if versioned_refs:
        out["incidents"] = [i for i in out["incidents"] if i.get("id") not in versioned_refs]
        out["runbooks"]  = [r for r in out["runbooks"]  if r.get("id") not in versioned_refs]
        out["docs"]      = [d for d in out["docs"]      if d.get("id") not in versioned_refs]

    # ── CLUSTER-AWARE FILTER (June 14) ───────────────────────────────────────
    # The core KB problem: "Nexus" matches every Nexus ticket — dev AND maint —
    # so retrieval pulls the wrong one. Fix: detect which cluster the QUERY is
    # about, then DROP results that are explicitly tagged with a DIFFERENT
    # cluster. An item must match the cluster context (or carry no cluster tag)
    # to survive. This makes "Nexus in maint" stop matching the dev Nexus chat.
    q_cluster = _detect_cluster(query)
    if q_cluster:
        for cat in out:
            kept = []
            for item in out[cat]:
                item_cluster = _detect_cluster(
                    (item.get("tags", "") or "") + " " +
                    (item.get("title", "") or ""))
                # keep if: same cluster, OR item has no cluster marker at all
                if item_cluster is None or item_cluster == q_cluster:
                    kept.append(item)
            # only apply the filter if it leaves something — never return empty
            # when the unfiltered set had results (safety: better a loose match
            # than nothing). But if same-cluster matches exist, prefer them.
            same = [i for i in kept if _detect_cluster(
                (i.get("tags","") or "")+" "+(i.get("title","") or "")) == q_cluster]
            out[cat] = same if same else kept

    total = sum(len(v) for v in out.values())
    if total:
        log.debug("KB search (%s, cluster=%s): %d inc, %d rb, %d docs for '%s'",
                  mode, q_cluster or "any", len(out["incidents"]), len(out["runbooks"]),
                  len(out["docs"]), query[:60])
    return out


_CLUSTER_PATTERNS = {
    "maint": re.compile(r"\b(maint|maintenance|k8mgmt|production|prod)\b", re.I),
    "dev":   re.compile(r"\b(dev|development|k8s\.local|staging|test)\b", re.I),
    "prod":  re.compile(r"\bprod(uction)?\b", re.I),
}

def _detect_cluster(text: str):
    """Detect which cluster a query/item refers to (maint/dev/prod) so retrieval
    can keep Nexus-in-maint separate from Nexus-in-dev. Returns the cluster name
    or None if no clear cluster marker. 'maint' and 'prod' are treated as the
    same production environment per Naveen's setup (maint IS production)."""
    if not text:
        return None
    t = text.lower()
    # maint == production in Naveen's world; check it first
    if _CLUSTER_PATTERNS["maint"].search(t):
        return "maint"
    if _CLUSTER_PATTERNS["dev"].search(t):
        return "dev"
    return None


def kb_format_for_prompt(hits: dict, char_budget: int = 6000) -> str:
    """Format retrieval hits into a clean knowledge-base block injected as
    volatile system context. char_budget caps total injection to keep cost sane.
    """
    if not hits or not any(hits.values()):
        return ""
    sections = []
    used = 0

    def _add(line: str):
        nonlocal used
        if used + len(line) <= char_budget:
            sections.append(line)
            used += len(line)
            return True
        return False

    def _add_fit(line: str, min_room: int = 400):
        """Like _add, but TRUNCATES to fit instead of dropping the block whole.
        Used for authoritative content (current version) that must never be
        silently discarded just because it is large — losing it entirely is far
        worse than losing its tail."""
        nonlocal used
        room = char_budget - used
        if room <= min_room:
            return False
        if len(line) <= room:
            sections.append(line); used += len(line)
        else:
            sections.append(line[:room - 20] + "\n…[truncated]\n")
            used = char_budget
        return True

    if hits.get("versioned"):
        _wants_hist = any(v.get("history_full") for v in hits["versioned"])
        _add("\n═══ VERSIONED KNOWLEDGE — AUTHORITATIVE CURRENT STATE ═══\n"
             "These topics have an update history. The CURRENT block below is the "
             "AUTHORITATIVE, up-to-date state of that system — it supersedes everything else.\n"
             "RULES (important):\n"
             "  1. Answer about today's state from CURRENT. Lead with it.\n"
             "  2. Any OTHER retrieved incident/runbook/doc describing this SAME system "
             "(even if detailed and well-written) describes an EARLIER state. Do NOT present "
             "it as today's state, and do NOT let its polish outweigh CURRENT. If it conflicts "
             "with CURRENT, CURRENT wins — say so plainly if the difference matters.\n"
             "  3. NOTHING IS EVER LOST: every past version is retained in full. If Naveen asks "
             "about origin, lineage, evolution, what changed, or an older version, that full "
             "historical content is available — say so and use it rather than claiming it's gone.\n")
        for i, v in enumerate(hits["versioned"], 1):
            block = f"\n[{v['entry_title']}] — CURRENT: {v['current_label']}\n{v['current_content']}\n"
            if v.get("history"):
                if _wants_hist:
                    block += "\n  ── EARLIER VERSIONS (full content, since history was asked for) ──\n"
                    for h in v["history"]:
                        block += (f"\n  [v{h['version_no']} — {h['label']}]\n"
                                  f"{h.get('content', h.get('summary',''))}\n")
                else:
                    block += ("  (earlier versions retained in full — ask about history/origin "
                              "to see them: " +
                              "; ".join(f"v{h['version_no']} {h['label']}" for h in v["history"]) + ")\n")
            if not _add_fit(block):
                break
    if hits.get("incidents"):
        _add("\n═══ RELEVANT PAST INCIDENTS (from your knowledge base — these are YOUR real environment. ACTIVELY cross-check the current task against these: if Naveen is reusing or pasting config, compare service names, replica counts, namespaces, secret names, image tags, ports against how it actually worked before. If something won't match his cluster, FLAG IT before he applies it.) ═══\n")
        for i, inc in enumerate(hits["incidents"], 1):
            block = (
                f"\n[Incident {i}] {inc['title']}\n"
                f"  SYMPTOMS: {inc['symptoms']}\n"
                f"  ROOT CAUSE: {inc['root_cause']}\n"
                f"  FIX: {inc['fix']}\n"
            )
            if inc.get("validation"):
                block += f"  VALIDATION: {inc['validation']}\n"
            if not _add(block):
                break
    if hits.get("runbooks"):
        _add("\n═══ RELEVANT RUNBOOKS ═══\n")
        for i, rb in enumerate(hits["runbooks"], 1):
            block = (
                f"\n[Runbook {i}] {rb['title']}\n"
                f"  SCENARIO: {rb['scenario']}\n"
                f"  STEPS:\n{rb['steps']}\n"
            )
            if not _add(block):
                break
    if hits.get("docs"):
        _add("\n═══ RELEVANT DOCUMENTATION SNIPPETS ═══\n")
        for i, d in enumerate(hits["docs"], 1):
            block = (
                f"\n[Doc {i}] {d['title']} (from {d['source']})\n"
                f"  {d['snippet']}\n"
            )
            if not _add(block):
                break
    if sections:
        sections.insert(0, "\n")
        sections.append("\nUse this knowledge base to inform your answer — when an incident matches, cite it briefly ('this matches your past Kafka mTLS incident...'). When a runbook applies, follow it. When docs are relevant, quote the specific section.\n")
    return "".join(sections)


def kb_add_incident(title, symptoms, root_cause, fix, validation="", tags="", references=""):
    iid = str(uuid.uuid4())
    ts = now_iso()
    # Embed: title + symptoms + root_cause gives the best retrieval signal
    embed_text = f"{title}. Symptoms: {symptoms}. Root cause: {root_cause}"
    emb_json = json.dumps(get_embedding(embed_text)) or None
    with get_db() as conn:
        conn.execute(
            "INSERT INTO incidents (id,title,symptoms,root_cause,fix,validation,tags,references_,embedding,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iid, title, symptoms, root_cause, fix, validation, tags, references, emb_json, ts)
        )
        conn.execute("INSERT INTO incidents_fts(incidents_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    if emb_json and emb_json != "[]":
        log.info("Incident saved with semantic embedding ✓")
    return iid


def kb_add_runbook(title, scenario, steps, tags=""):
    bid = str(uuid.uuid4())
    ts = now_iso()
    embed_text = f"{title}. {scenario}. {steps[:500]}"
    emb_json = json.dumps(get_embedding(embed_text)) or None
    with get_db() as conn:
        conn.execute(
            "INSERT INTO runbooks (id,title,scenario,steps,tags,embedding,created_at) VALUES (?,?,?,?,?,?,?)",
            (bid, title, scenario, steps, tags, emb_json, ts)
        )
        conn.execute("INSERT INTO runbooks_fts(runbooks_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    return bid


def kb_add_doc(source, title, content, tags=""):
    """Add a documentation chunk. Long docs should be pre-chunked by the caller."""
    did = str(uuid.uuid4())
    ts = now_iso()
    embed_text = f"{title} ({source}). {content[:1000]}"
    emb_json = json.dumps(get_embedding(embed_text)) or None
    with get_db() as conn:
        conn.execute(
            "INSERT INTO kb_docs (id,source,title,content,tags,embedding,created_at) VALUES (?,?,?,?,?,?,?)",
            (did, source, title, content, tags, emb_json, ts)
        )
        conn.execute("INSERT INTO kb_docs_fts(kb_docs_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    return did


def chunk_text(text: str, chunk_chars: int = 4000, overlap_chars: int = 400) -> list:
    """Chunk a long document for storage in kb_docs. Overlap preserves context
    across boundaries (so a question matching the boundary still hits one chunk).
    """
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_chars, n)
        # Try to end on a paragraph or sentence boundary for cleaner chunks
        if end < n:
            for sep in ("\n\n", ". ", "\n", " "):
                idx = text.rfind(sep, i + chunk_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap_chars, i + 1)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  CHAT FILES ON DISK — a readable "book" of every conversation
# ─────────────────────────────────────────────────────────────────────────────
# The database (chats.db) is the source of truth. These folders are a
# human-readable copy of it, rebuilt after every reply and for every chat at
# startup, so they can always be browsed in Finder:
#
#   local-llm-db/chats/
#     INDEX - all chats.txt                     ← table of contents, newest first
#     Unfiled/
#       2026-10-05 · what's the latest on grafana in dev__7b126e90/
#         chat.txt                              ← the conversation, word for word
#     <UI folder name>/                         ← same names as the folders in the UI
#       _Trash_/                                ← chats deleted in the UI (restorable)
#     _Orphan/                                  ← "Delete forever" chats, kept for reference
#
# A chat folder is named  <date started> · <title exactly as in the UI>__<id>.
# The "__7b126e90" tail is the chat's id: it is the link back to the database and
# keeps two chats with the same title apart. Only characters a folder name can't
# hold ( / and : ) are replaced.
from datetime import timezone as _tz
import shutil as _shutil
import threading as _threading

CHAT_INDEX_NAME = "INDEX - all chats.txt"
_CHAT_DIR_RE = re.compile(r"__[0-9a-f]{8}$")
_CHAT_FS_LOCK = _threading.RLock()       # replies arrive on several threads


def _readable_name(text, maxlen: int = 90, fallback: str = "Chat") -> str:
    s = re.sub(r"[\x00-\x1f/\\:]+", " ", str(text or ""))
    s = s.replace("__", "_")                       # "__" is reserved for the id tail
    s = re.sub(r"\s+", " ", s).strip().lstrip(".").strip()
    return s[:maxlen].rstrip(" .") or fallback


def _local_dt(iso):
    """Stored timestamps are UTC → the Mac's local time for display."""
    try:
        return datetime.fromisoformat(str(iso)[:19]).replace(tzinfo=_tz.utc).astimezone()
    except Exception:
        return datetime.now().astimezone()


def _chat_dirname(chat: dict) -> str:
    return (f"{_local_dt(chat['created_at']):%Y-%m-%d} · "
            f"{_readable_name(chat.get('title') or 'New Chat')}__{chat['id'][:8]}")


def _walk_chat_dirs():
    """Every chat folder under chats/, chats/<folder>/ and chats/<folder>/_Trash_/
    (never inside _Orphan/, which holds records of chats deleted forever)."""
    def scan(d, depth):
        try:
            kids = list(d.iterdir())
        except OSError:
            return
        for p in kids:
            if not p.is_dir() or p.name == "_Orphan":
                continue
            if _CHAT_DIR_RE.search(p.name):
                yield p
            elif depth < 2:
                yield from scan(p, depth + 1)
    yield from scan(CHATS_DIR, 0)


def _chat_dirs_on_disk(chat_id: str) -> list:
    suffix = f"__{chat_id[:8]}"
    return [d for d in _walk_chat_dirs() if d.name.endswith(suffix)]


def _prune_empty(d: Path) -> None:
    """Remove folders left empty by a move (a lone .DS_Store counts as empty),
    walking up — but never chats/ itself."""
    root = CHATS_DIR.resolve()
    while d.exists() and d.resolve() != root and root in d.resolve().parents:
        kids = list(d.iterdir())
        if any(k.name != ".DS_Store" for k in kids):
            return
        for k in kids:
            k.unlink()
        d.rmdir()
        d = d.parent


def _merge_dir_into(src: Path, dst: Path) -> None:
    """Fold a stray duplicate chat folder into the real one. Nothing is
    overwritten: a clashing name gets ' (from <old folder>)' appended. An old
    chat.txt is dropped because chat.txt is rebuilt from the database."""
    for item in list(src.iterdir()):
        if item.name in ("chat.txt", ".DS_Store"):
            item.unlink()
            continue
        target = dst / item.name
        n = 1
        while target.exists():
            tag = f" (from {src.name[:40]})" + (f" {n}" if n > 1 else "")
            target = dst / f"{item.stem}{tag}{item.suffix}"
            n += 1
        os.rename(str(item), str(target))
    src.rmdir()


def _model_label(model_id) -> str:
    model_id = str(model_id or "")
    if model_id.startswith("ollama:"):
        return f"{model_id[7:]} (local)"
    m = MODEL_MAP.get(model_id)
    return m["name"] if m else model_id


def _chat_book_text(chat: dict, msgs: list, folder_label: str) -> str:
    """The conversation as plain text, exactly as written — nothing stripped,
    so commands, YAML and code read the same as in the app."""
    rule, thin = "═" * 72, "─" * 72
    n_q = sum(1 for m in msgs if m["role"] == "user")
    cost = sum(float(m.get("cost_usd") or 0) for m in msgs)
    where = folder_label + ("   (deleted — in Trash, restore it from Settings)" if chat.get("deleted_at") else "")
    out = [rule, f"  {chat.get('title') or 'New Chat'}", rule,
           f"  Folder     : {where}",
           f"  Started    : {_local_dt(chat['created_at']):%a %d %b %Y, %H:%M}",
           f"  Last reply : {_local_dt(chat['updated_at']):%a %d %b %Y, %H:%M}",
           f"  Messages   : {len(msgs)}  ({n_q} question{'s' if n_q != 1 else ''})",
           f"  Cost       : ${cost:.4f}",
           rule, ""]
    for m in msgs:
        when = f"{_local_dt(m['created_at']):%d %b %Y, %H:%M}"
        text = _content_to_str(m["content"]).rstrip()
        if m["role"] == "user":
            head = f"▶ YOU  ·  {when}"
        else:
            local = str(m.get("model") or "").startswith("ollama:")
            price = "$0.00" if local else f"${float(m.get('cost_usd') or 0):.4f}"
            head = f"◀ JARVIS  ·  {when}  ·  {_model_label(m.get('model'))}  ·  {price}"
        out += [head, thin, text, "", ""]
    out += [rule, f"  End of conversation · {len(msgs)} messages", rule, ""]
    return "\n".join(out)


def _write_chat_index() -> None:
    """chats/INDEX - all chats.txt — every chat, grouped like the sidebar."""
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, c.deleted_at, c.model, "
            "f.name AS folder, (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS n "
            "FROM chats c LEFT JOIN folders f ON f.id = c.folder_id ORDER BY c.updated_at DESC")]
    groups = {}
    for r in rows:
        key = "TRASH  (deleted in the app — restore from Settings → Trash)" if r["deleted_at"] \
            else _readable_name(r["folder"] or "Unfiled", 60, "Unfiled")
        groups.setdefault(key, []).append(r)
    order = sorted((k for k in groups if not k.startswith("TRASH")), key=lambda k: (k != "Unfiled", k.lower()))
    order += [k for k in groups if k.startswith("TRASH")]
    out = ["ALL CHATS — table of contents",
           f"Updated {datetime.now():%a %d %b %Y, %H:%M} · {len(rows)} chats",
           "",
           "Each line is one folder below (same date and title). Open it, then chat.txt —",
           "the whole conversation, word for word. chats.db stays the master copy.",
           ""]
    for k in order:
        out += ["", f"{k.upper() if not k.startswith('TRASH') else k}  ({len(groups[k])})", "─" * 72]
        for r in groups[k]:
            mode = "local" if str(r["model"] or "").startswith("ollama:") else "cloud"
            title = _readable_name(r["title"] or "New Chat", 64)
            out.append(f"  {_local_dt(r['created_at']):%Y-%m-%d}  {title:<64}  {r['n']:>3} msgs  {mode}")
    try:
        (CHATS_DIR / CHAT_INDEX_NAME).write_text("\n".join(out) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("chat index write failed: %s", e)


def export_chat_txt(chat_id: str, write_index: bool = True):
    """Put the chat's folder where it belongs (its UI folder, or that folder's
    _Trash_ if deleted), named after its current title, and (re)write chat.txt.
    Old-style or duplicate folders of the same chat are moved/merged in — no
    file is ever deleted except an old chat.txt, which is rebuilt right here."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not row:
            return
        chat = dict(row)
        msgs = [dict(r) for r in conn.execute(
            "SELECT role, content, created_at, model, cost_usd FROM messages "
            "WHERE chat_id=? ORDER BY created_at, rowid", (chat_id,))]
        folder_label = "Unfiled"
        if chat.get("folder_id"):
            fr = conn.execute("SELECT name FROM folders WHERE id=?", (chat["folder_id"],)).fetchone()
            if fr:
                folder_label = fr["name"]

    parent = CHATS_DIR / _readable_name(folder_label, 60, "Unfiled")
    if chat.get("deleted_at"):
        parent = parent / "_Trash_"
    target = parent / _chat_dirname(chat)

    with _CHAT_FS_LOCK:
        dirs = _chat_dirs_on_disk(chat_id)
        primary = target if target in dirs else next(
            (d for d in dirs if (d / "chat.txt").exists()), dirs[0] if dirs else None)
        if primary is None:
            target.mkdir(parents=True, exist_ok=True)
        elif primary != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            old_parent = primary.parent
            os.rename(str(primary), str(target))      # rename, never copy-then-delete
            log.info("Chat folder: %s/%s -> %s/%s", old_parent.name, primary.name,
                     target.parent.name, target.name)
            _prune_empty(old_parent)
        for extra in dirs:
            if extra != primary and extra.exists() and extra != target:
                old_parent = extra.parent
                _merge_dir_into(extra, target)
                log.info("Chat folder: merged duplicate %s into %s", extra.name, target.name)
                _prune_empty(old_parent)
        (target / "chat.txt").write_text(_chat_book_text(chat, msgs, folder_label), encoding="utf-8")
    if write_index:
        _write_chat_index()


def backfill_all_chat_txt() -> int:
    """Startup: put every chat's folder in its right place with a fresh chat.txt
    (this is also what reorganises folders made by older versions), then write
    the index once. Returns how many chats were written."""
    count = 0
    try:
        with get_db() as conn:
            all_ids = [r[0] for r in conn.execute("SELECT id FROM chats").fetchall()]
        for cid in all_ids:
            try:
                export_chat_txt(cid, write_index=False)
                count += 1
            except Exception as exc:
                log.warning("backfill chat.txt failed for %s: %s", cid[:8], exc)
        _write_chat_index()
    except Exception as exc:
        log.warning("backfill_all_chat_txt error: %s", exc)
    return count


# Run on startup — every chat gets its readable folder + chat.txt
_backfilled = backfill_all_chat_txt()
log.info("Startup backfill: wrote chat.txt for %d chats", _backfilled)



# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — STATIC
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Send as plain file — avoids Jinja2 parsing JS template literals
    return send_from_directory("templates", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — MODELS & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    # Cloud models + whatever local Ollama models are installed (empty if Ollama is off)
    return jsonify(MODELS + repo_kb.local_model_entries())


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    s    = load_settings()
    ck   = s.get("api_key",    "")
    qk   = s.get("groq_key",   "")
    ok   = s.get("openai_key", "")
    ck2  = s.get("api_key_2",  "")
    return jsonify({
        "has_claude_key":    bool(ck),
        "claude_preview":    f"sk-ant-…{ck[-6:]}"  if len(ck) > 10 else "",
        "has_backup_key":    bool(ck2),
        "backup_preview":    f"sk-ant-…{ck2[-6:]}" if len(ck2) > 10 else "",
        "use_backup_key":    str(s.get("use_backup_key", "false")).lower() == "true",
        "backup_starting_credit": float(s.get("backup_starting_credit", "100.0") or 100.0),
        "backup_switched_at":     s.get("backup_switched_at", ""),
        "has_groq_key":      bool(qk),
        "groq_preview":      f"gsk_…{qk[-4:]}"     if len(qk) > 8  else "",
        "has_openai_key":    bool(ok),
        "openai_preview":    f"sk-…{ok[-4:]}"      if len(ok) > 8  else "",
        "default_model":     s.get("default_model", DEFAULT_MODEL),
        "starting_credit":   float(s.get("starting_credit", "5.40")),
        "response_mode":     s.get("response_mode", "detailed"),
        "depth_choice":      s.get("depth_choice", "average"),
        "reviewer_enabled":  s.get("reviewer_enabled", "false"),
        "kb_enabled":        s.get("kb_enabled", "true"),
        "theme":             s.get("theme", "light"),
    })


@app.route("/api/settings", methods=["POST"])
def api_post_settings():
    data = request.json or {}

    if key := data.get("api_key", "").strip():
        if not key.startswith("sk-ant-"):
            return jsonify({"error": "Claude key must start with sk-ant-"}), 400
        save_setting("api_key", key)

    if key := data.get("api_key_2", "").strip():
        if not key.startswith("sk-ant-"):
            return jsonify({"error": "Backup key must start with sk-ant-"}), 400
        save_setting("api_key_2", key)

    if "use_backup_key" in data:
        use_backup = bool(data["use_backup_key"])
        save_setting("use_backup_key", "true" if use_backup else "false")
        if use_backup:
            # Record when the switch happened so spend can be filtered by date
            save_setting("backup_switched_at", now_iso())

    if credit := data.get("backup_starting_credit"):
        try:
            save_setting("backup_starting_credit", str(float(credit)))
            # Initialise backup_spend to 0 the first time a backup credit is set,
            # so the backup key starts its countdown fresh from its full amount.
            if load_settings().get("backup_spend") is None:
                save_setting("backup_spend", "0.0")
        except ValueError:
            pass

    if key := data.get("groq_key", "").strip():
        save_setting("groq_key", key)

    if key := data.get("openai_key", "").strip():
        if not key.startswith("sk-"):
            return jsonify({"error": "OpenAI key must start with sk-"}), 400
        save_setting("openai_key", key)

    if model := data.get("default_model", "").strip():
        save_setting("default_model", model)

    if credit := data.get("starting_credit"):
        try:
            save_setting("starting_credit", str(float(credit)))
            # Manually setting the primary balance means "this is my balance NOW",
            # so reset the primary spend counter to count down from the new figure.
            save_setting("primary_spend", "0.0")
        except (ValueError, TypeError):
            pass

    if mode := data.get("response_mode", "").strip():
        if mode in ("concise", "detailed"):
            save_setting("response_mode", mode)
    if dc := data.get("depth_choice", "").strip().lower():
        if dc in ("normal", "average", "deep", "super"):
            save_setting("depth_choice", dc)
    if "reviewer_enabled" in data:
        save_setting("reviewer_enabled", "true" if data["reviewer_enabled"] in (True, "true", 1, "1") else "false")
    if "kb_enabled" in data:
        save_setting("kb_enabled", "true" if data["kb_enabled"] in (True, "true", 1, "1") else "false")

    if theme := data.get("theme", "").strip():
        if theme in ("light", "dark"):
            save_setting("theme", theme)

    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — CHATS (CRUD)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/chats", methods=["GET"])
def api_list_chats():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE deleted_at IS NULL ORDER BY updated_at DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — FOLDERS (additive; chats with folder_id=NULL are "unfiled")
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/folders", methods=["GET"])
def api_list_folders():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM folders ORDER BY name COLLATE NOCASE ASC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/folders", methods=["POST"])
def api_create_folder():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Folder name required"}), 400
    folder_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO folders (id, name, created_at) VALUES (?, ?, ?)",
            (folder_id, name, now_iso())
        )
    return jsonify({"id": folder_id, "name": name})


@app.route("/api/folders/<folder_id>", methods=["PATCH"])
def api_rename_folder(folder_id):
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Folder name required"}), 400
    with get_db() as conn:
        conn.execute("UPDATE folders SET name=? WHERE id=?", (name, folder_id))
        _ids = [r["id"] for r in conn.execute("SELECT id FROM chats WHERE folder_id=?", (folder_id,)).fetchall()]
    for _cid in _ids:                  # folder mirror: dirs follow rename
        try:
            export_chat_txt(_cid)
        except Exception as _e:
            log.warning("mirror rename failed: %s", _e)
    return jsonify({"ok": True, "id": folder_id, "name": name})


@app.route("/api/folders/<folder_id>", methods=["DELETE"])
def api_delete_folder(folder_id):
    # Deleting a folder does NOT delete its chats — they become unfiled again.
    with get_db() as conn:
        _ids = [r["id"] for r in conn.execute("SELECT id FROM chats WHERE folder_id=?", (folder_id,)).fetchall()]
        conn.execute("UPDATE chats SET folder_id=NULL WHERE folder_id=?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id=?", (folder_id,))
    for _cid in _ids:                  # folder mirror: dirs -> Unfiled/
        try:
            export_chat_txt(_cid)
        except Exception as _e:
            log.warning("mirror unfile failed: %s", _e)
    return jsonify({"ok": True})


@app.route("/api/chats/<chat_id>/folder", methods=["PATCH"])
def api_move_chat_to_folder(chat_id):
    # folder_id may be null (move back to unfiled) or a valid folder id.
    data = request.json or {}
    folder_id = data.get("folder_id")  # None => unfiled
    with get_db() as conn:
        conn.execute("UPDATE chats SET folder_id=? WHERE id=?", (folder_id, chat_id))
    try:
        export_chat_txt(chat_id)       # folder mirror: move dir on disk now
    except Exception as _e:
        log.warning("folder mirror move failed: %s", _e)
    return jsonify({"ok": True, "chat_id": chat_id, "folder_id": folder_id})


@app.route("/api/chats/export-all", methods=["POST"])
def api_export_all_chats():
    """Manually trigger chat.txt refresh for all chats — useful after migrations."""
    count = backfill_all_chat_txt()
    return jsonify({"ok": True, "exported": count})

@app.route("/api/chats/status", methods=["GET"])
def api_chats_status():
    """
    Returns folder status for every chat in the DB.
    active    = chat in DB AND folder exists on disk → green dot in UI
    no_folder = chat in DB but folder missing on disk → grey dot (edge case)
    Also returns orphan_folders: on disk but NOT in DB → red in UI.
    """
    with get_db() as conn:
        db_chats = {r["id"]: r["title"] for r in conn.execute("SELECT id, title FROM chats").fetchall()}

    # Map short_id (8 chars) -> full chat_id
    short_to_full = {cid[:8]: cid for cid in db_chats}

    chat_status    = {cid: False for cid in db_chats}  # True = folder found on disk
    orphan_folders = []

    if CHATS_DIR.exists():
        for d in _walk_chat_dirs():          # chats/, chats/<folder>/, …/_Trash_/
            short_id = d.name.rsplit("__", 1)[-1]
            if short_id in short_to_full:
                chat_status[short_to_full[short_id]] = True
            else:
                orphan_folders.append(d.name)

    result = []
    for cid, has_folder in chat_status.items():
        result.append({
            "chat_id":    cid,
            "title":      db_chats[cid],
            "has_folder": has_folder,
            "status":     "active" if has_folder else "no_folder",
        })

    return jsonify({
        "chats":          result,
        "orphan_folders": orphan_folders,
        "chats_dir":      str(CHATS_DIR),
    })

@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    data    = request.json or {}
    s       = load_settings()
    chat_id = str(uuid.uuid4())
    model   = data.get("model", s.get("default_model", DEFAULT_MODEL))
    # Guard: coerce any unknown/removed model (e.g. an old gemini chat) to default.
    # Local models ("ollama:<name>") are discovered live, so they aren't in MODEL_MAP.
    if model not in MODEL_MAP and not str(model).startswith("ollama:"):
        model = DEFAULT_MODEL
    title   = data.get("title", "New Chat")
    folder_id = data.get("folder_id")  # optional — create chat directly inside a folder
    # Validate folder_id exists; if not, fall back to unfiled (null)
    if folder_id:
        with get_db() as _c:
            ok = _c.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone()
        if not ok:
            folder_id = None
    ts      = now_iso()

    with get_db() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, model, created_at, updated_at, folder_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, title, model, ts, ts, folder_id)
        )
    try:
        export_chat_txt(chat_id)       # readable folder on disk, named like the UI
    except Exception as e:
        log.warning("chat folder create failed: %s", e)

    log.info("Created chat '%s' (%s)", title, chat_id[:8])
    return jsonify({"id": chat_id, "title": title, "model": model, "created_at": ts, "folder_id": folder_id})


@app.route("/api/chats/<chat_id>", methods=["GET"])
def api_get_chat(chat_id: str):
    with get_db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        msgs = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at",
            (chat_id,)
        ).fetchall()

    messages_out = []
    for m in msgs:
        try:
            content = json.loads(m["content"])
        except (json.JSONDecodeError, TypeError):
            content = m["content"]
        messages_out.append({
            "id":         m["id"],
            "role":       m["role"],
            "content":    content,
            "tokens_in":  m["tokens_in"],
            "tokens_out": m["tokens_out"],
            "model":      m["model"] if "model" in m.keys() else None,
            "created_at": m["created_at"],
        })

    return jsonify({**dict(chat), "messages": messages_out})


@app.route("/api/chats/<chat_id>", methods=["PATCH"])
def api_update_chat(chat_id: str):
    data = request.json or {}
    sets, params = [], []

    if "title" in data:
        new_title = data["title"][:120]
        sets.append("title = ?")
        params.append(new_title)

    if "model" in data:
        sets.append("model = ?")
        params.append(data["model"])

    if sets:
        params += [now_iso(), chat_id]
        with get_db() as conn:
            conn.execute(
                f"UPDATE chats SET {', '.join(sets)}, updated_at = ? WHERE id = ?",
                params
            )
        if "title" in data:
            try:
                export_chat_txt(chat_id)   # folder on disk follows the new title
            except Exception as e:
                log.warning("Could not rename chat folder: %s", e)
    return jsonify({"ok": True})



def _find_chat_dir(chat_id: str):
    """Locate a chat's on-disk directory anywhere under CHATS_DIR (root, a
    UI-folder, a _Trash_ subfolder, or already _ORPHAN_-prefixed). The chat-id
    suffix is the stable key, so prefixes/renames don't hide it."""
    suffix = f"__{chat_id[:8]}"
    cands = [p for p in CHATS_DIR.iterdir() if p.is_dir()]
    for d in list(cands):
        if not d.name.endswith(suffix):
            cands.extend(p for p in d.iterdir() if p.is_dir())
            for sub in [p for p in d.iterdir() if p.is_dir() and p.name == "_Trash_"]:
                cands.extend(p for p in sub.iterdir() if p.is_dir())
    for d in cands:
        if d.is_dir() and d.name.endswith(suffix):
            return d
    return None


def _move_chat_dir_to_trash(chat_id: str):
    """On UI delete (deleted_at already set): export_chat_txt moves the chat's
    folder into <its UI folder>/_Trash_/ — kept, readable, restorable."""
    try:
        export_chat_txt(chat_id)
    except Exception as e:
        log.warning("trash move failed for %s: %s", chat_id[:8], e)


def _restore_chat_dir_from_trash(chat_id: str):
    """On restore (deleted_at cleared): export_chat_txt moves it back out."""
    try:
        export_chat_txt(chat_id)
    except Exception as e:
        log.warning("trash restore failed for %s: %s", chat_id[:8], e)


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id: str):
    """Soft delete — marks deleted_at timestamp, keeps all data in DB."""
    ts = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET deleted_at = ? WHERE id = ?", (ts, chat_id)
        )
    _move_chat_dir_to_trash(chat_id)
    log.info("Soft-deleted chat %s", chat_id[:8])
    return jsonify({"ok": True})


@app.route("/api/chats/<chat_id>/restore", methods=["POST"])
def api_restore_chat(chat_id: str):
    """Restore a soft-deleted chat back to the sidebar."""
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET deleted_at = NULL WHERE id = ?", (chat_id,)
        )
    _restore_chat_dir_from_trash(chat_id)
    log.info("Restored chat %s", chat_id[:8])
    return jsonify({"ok": True})


def _orphan_chat_dir(chat_id: str, parent_hint: str = ""):
    """On HARD delete (Delete Forever): chat is gone from the DB, but we KEEP its
    directory on disk for reference. Move it into ONE central folder:
        CHATS_DIR/_Orphan/<PARENTNAME>__<original-dir-name>
    The parent folder name is baked into the prefix so Naveen keeps context about
    where the chat used to live (e.g. 'Devops__2026-06-13__Kafka_mTLS__abc123').
    Simple, deterministic, no dependency on macOS behaviour."""
    try:
        d = _find_chat_dir(chat_id)
        if not d:
            return
        import shutil as _sh
        # derive the parent (UI folder) name; if it was in _Trash_, go one up
        real_parent = d.parent.parent if d.parent.name == "_Trash_" else d.parent
        parent_name = parent_hint or (real_parent.name if real_parent != CHATS_DIR else "Unfiled")
        parent_name = _readable_name(parent_name, 40, "Unfiled")

        orphan_root = CHATS_DIR / "_Orphan"
        orphan_root.mkdir(exist_ok=True)
        # Build name as  <parent>__<title>__<id>  — strip the leading date prefix
        # (dir names are "YYYY-MM-DD__title__id"; Naveen wants the title, not the
        # date, since he names every chat meaningfully).
        core = re.sub(r'^\d{4}-\d{2}-\d{2}__', '', d.name)   # drop leading date
        base = core
        if not base.startswith(parent_name + "__"):
            base = f"{parent_name}__{core}"
        target = orphan_root / base
        # avoid clobber if a same-named orphan already exists
        if target.exists():
            target = orphan_root / f"{base}__{chat_id[:8]}dup"
        if d.resolve() != target.resolve():
            _sh.move(str(d), str(target))
        # clean up an emptied _Trash_
        if d.parent.name == "_Trash_":
            try:
                d.parent.rmdir()
            except OSError:
                pass
        log.info("Orphaned: %s -> _Orphan/%s", chat_id[:8], base)
    except Exception as e:
        log.warning("orphan move failed for %s: %s", chat_id[:8], e)


@app.route("/api/chats/<chat_id>/hard-delete", methods=["DELETE"])
def api_hard_delete_chat(chat_id: str):
    """Permanent delete — chat removed from DB, its dir kept in _Orphan/ on disk."""
    # Capture the parent folder name BEFORE the row is deleted (folder_id is gone
    # after DELETE). This is what gets baked into the orphan prefix for context.
    parent_hint = ""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT f.name AS fname FROM chats c "
                "LEFT JOIN folders f ON f.id = c.folder_id WHERE c.id = ?",
                (chat_id,)).fetchone()
        if row and row["fname"]:
            parent_hint = row["fname"]
        else:
            parent_hint = "Unfiled"
    except Exception:
        parent_hint = "Unfiled"
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    try:
        _orphan_chat_dir(chat_id, parent_hint=parent_hint)
        _write_chat_index()
    except Exception as e:
        log.warning("orphan step failed (chat still deleted from DB): %s", e)
    log.info("Hard-deleted chat %s", chat_id[:8])
    return jsonify({"ok": True})


@app.route("/api/chats/trash", methods=["GET"])
def api_list_trash():
    """Return all soft-deleted chats for the Trash view."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — SEND MESSAGE (streaming)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_auto_title(chat_id: str, text_input: str) -> None:
    """Title a chat from its first message and rename its folder to match.
    Shared by the cloud path below and the local path (repo_kb)."""
    new_title = auto_title(text_input)
    with get_db() as conn:
        conn.execute(
            "UPDATE chats SET title = ? WHERE id = ?",
            (new_title, chat_id)
        )
    # Rename chat folder to reflect the actual title
    try:
        export_chat_txt(chat_id)
    except Exception as e:
        log.warning("Could not rename folder on auto-title: %s", e)


@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def api_send_message(chat_id: str):
    # ── Resolve chat + model ──────────────────────────────────────────────────
    with get_db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    # ── LOCAL MODE (Ollama on this Mac, $0) ───────────────────────────────────
    # Handed off BEFORE any cloud logic runs: no API key needed, no Claude
    # compression/memory/reviewer calls, nothing touches the prompt cache.
    if str(dict(chat)["model"]).startswith("ollama:"):
        return repo_kb.handle_local_message(chat_id, dict(chat)["model"],
                                            apply_auto_title=_apply_auto_title)

    s        = load_settings()
    model    = dict(chat)["model"]
    # Guard: if this chat references a removed model, fall back to default
    if model not in MODEL_MAP:
        model = s.get("default_model", DEFAULT_MODEL)
        if model not in MODEL_MAP:
            model = DEFAULT_MODEL
        try:
            with get_db() as _c:
                _c.execute("UPDATE chats SET model=? WHERE id=?", (model, chat_id))
        except Exception:
            pass
    minfo    = MODEL_MAP.get(model, {"provider": "claude", "free": False})
    provider = minfo["provider"]

    # ── Validate API key for this provider ────────────────────────────────────
    key_map = {
        "claude": active_claude_key(s),
        "groq":   s.get("groq_key",   ""),
        "openai": s.get("openai_key", ""),
    }
    api_key = key_map.get(provider, "")
    openai_key = s.get("openai_key", "")
    key_help = {
        "claude": "Claude key (console.anthropic.com → API Keys)",
        "groq":   "Groq key — free at console.groq.com/keys",
    }
    # Ollama runs locally — no API key needed. All other providers require one.
    if provider != "ollama" and not api_key:
        return jsonify({"error": f"No {key_help.get(provider, 'API')} key. Add in Settings ⚙️"}), 401

    # ── Parse input (multipart or JSON) ──────────────────────────────────────
    if request.content_type and "multipart" in request.content_type:
        text_input  = request.form.get("message", "").strip()
        files       = request.files.getlist("files")
        depth_input = request.form.get("depth", "").strip().lower()
    else:
        body        = request.json or {}
        text_input  = body.get("message", "").strip()
        files       = []
        depth_input = (body.get("depth") or "").strip().lower()

    if not text_input and not files:
        return jsonify({"error": "Message is empty"}), 400

    # ── Build content blocks ──────────────────────────────────────────────────
    blocks = []
    for f in files:
        mime  = f.content_type or "application/octet-stream"
        raw   = f.read()
        fname = f.filename or "file"
        ext   = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

        if mime.startswith("image/") and provider == "claude":
            b64 = base64.standard_b64encode(raw).decode()
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        elif ext == "pdf" or mime == "application/pdf":
            b64 = base64.standard_b64encode(raw).decode()
            blocks.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}})
            log.info("PDF attached: %s (%d KB)", fname, len(raw) // 1024)
        elif ext in ("py","js","ts","html","css","json","yaml","yml","md","txt","csv","xml","sh","bash","sql","conf","ini","toml","log","java","go","rs","c","cpp","h","rb","php","swift","kt"):
            try:
                text = raw.decode("utf-8", errors="replace")
                text = smart_truncate(text)
                blocks.append({"type": "text", "text": f"**File: {fname}**\n```{ext}\n{text}\n```"})
            except Exception:
                blocks.append({"type": "text", "text": f"**File: {fname}** — could not read as text"})
        elif ext == "docx":
            try:
                import io; from docx import Document as DocxDoc
                doc = DocxDoc(io.BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                text = smart_truncate(text)
                blocks.append({"type": "text", "text": f"**File: {fname}**\n```\n{text}\n```"})
            except ImportError:
                blocks.append({"type": "text", "text": f"**File: {fname}** — needs python-docx"})
            except Exception as exc:
                blocks.append({"type": "text", "text": f"**File: {fname}** — error: {exc}"})

        elif ext in ("tar", "gz", "tgz", "zip"):
            # Archive — extract all text/code files and inject each one
            TEXT_EXTS = {"py","js","ts","html","css","json","yaml","yml","md","txt",
                         "csv","xml","sh","bash","sql","conf","ini","toml","log",
                         "java","go","rs","c","cpp","h","rb","php","swift","kt"}
            extracted = []
            try:
                import tarfile, zipfile, io as _io
                if ext == "zip":
                    with zipfile.ZipFile(_io.BytesIO(raw)) as zf:
                        for nm in zf.namelist():
                            fe = nm.rsplit(".",1)[-1].lower() if "." in nm else ""
                            if fe not in TEXT_EXTS or nm.endswith("/"): continue
                            try:
                                t2 = zf.read(nm).decode("utf-8", errors="replace")
                                extracted.append(f"**File: {nm}**\n```{fe}\n{smart_truncate(t2)}\n```")
                            except: continue
                else:
                    with tarfile.open(fileobj=_io.BytesIO(raw), mode="r:*") as tf:
                        for mb in tf.getmembers():
                            if not mb.isfile(): continue
                            fe = mb.name.rsplit(".",1)[-1].lower() if "." in mb.name else ""
                            if fe not in TEXT_EXTS: continue
                            try:
                                fr = tf.extractfile(mb)
                                if not fr: continue
                                t2 = fr.read().decode("utf-8", errors="replace")
                                extracted.append(f"**File: {mb.name}**\n```{fe}\n{smart_truncate(t2)}\n```")
                            except: continue
                if extracted:
                    combined = f"**Archive: {fname}** ({len(extracted)} files)\n\n" + "\n\n---\n\n".join(extracted)
                    blocks.append({"type": "text", "text": combined})
                    log.info("Archive %s: %d files extracted", fname, len(extracted))
                else:
                    blocks.append({"type": "text", "text": f"**Archive: {fname}** — no text files inside"})
            except Exception as exc:
                blocks.append({"type": "text", "text": f"**Archive: {fname}** — error: {exc}"})

        else:
            blocks.append({"type": "text", "text": f"**File: {fname}** ({mime}, {len(raw)//1024}KB) — binary, not displayable"})

    if text_input:
        # ── ON-PASTE LOG COMPRESSION (Issue 2 — Naveen's idea, June 11) ──────
        # When 50+ lines of log/output are pasted, the FULL block hits the model
        # at 1x on this turn (uncached — it's brand new). On a 300-line dump
        # that's the single biggest one-shot cost. So we compress the pasted
        # block for the MODEL — keep the head + every real error/cause line +
        # tail — while saving the FULL original to the DB (so in-chat recall and
        # the user's own scrollback lose nothing). Typical 300-line log → ~30
        # meaningful lines sent. Same diagnosis, fraction of the tokens.
        model_text = text_input
        line_count = text_input.count("\n") + 1
        # More aggressive trigger (June 14): the real cost lives in heavy pastes.
        # Fire on EITHER many lines OR a large char blob (a 4,000-char single
        # paragraph of log has few newlines but is just as expensive). Threshold
        # lowered 50→30 lines and 2500→1800 chars to catch the mid-size logs that
        # were slipping through and landing in the 3.5-6¢ bucket.
        if (line_count >= 30 and len(text_input) > 1200) or len(text_input) > 3500:
            model_text = _compress_pasted_log(text_input)
            log.info("On-paste compression: %d lines / %d chars → %d chars to model "
                     "(full original saved to DB)",
                     line_count, len(text_input), len(model_text))
        blocks.append({"type": "text", "text": model_text})

    new_content     = blocks if len(blocks) > 1 else (model_text if text_input else text_input)
    # DB keeps the FULL original text (recall + scrollback need it), not the compressed one.
    user_content_db = json.dumps(
        [b if b.get("text") != model_text else {"type": "text", "text": text_input}
         for b in blocks]
    ) if len(blocks) > 1 else text_input

    # ── Auto-title on first user message ─────────────────────────────────────
    with get_db() as conn:
        msg_count = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()["c"]

    if msg_count == 0 and text_input:
        _apply_auto_title(chat_id, text_input)

    # ── Persist user message ──────────────────────────────────────────────────
    user_msg_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages "
            "(id, chat_id, role, content, tokens_in, tokens_out, model, created_at) "
            "VALUES (?, ?, 'user', ?, 0, 0, ?, ?)",
            (user_msg_id, chat_id, user_content_db, model, now_iso())
        )

    # ── Build optimised context ───────────────────────────────────────────────
    history = get_chat_history(chat_id)[:-1]   # exclude message just saved

    # ── HAIKU FOLLOW-UP ROUTING (June 11, 2026) ──────────────────────────────
    # ~50% of cost is Sonnet output tokens; ~60% of turns are short follow-ups
    # carrying no new evidence. Those route to Haiku 4.5 (1/3 the price) —
    # diagnosis-grade turns stay on Sonnet. ALL guardrails must hold:
    #   • not the first exchange of a chat (opening diagnosis = Sonnet)
    #   • typed text under 250 chars (a question, not a briefing)
    #   • no attachments/pasted blocks (new evidence = Sonnet)
    #   • selected model is default Sonnet (explicit Opus/other NEVER overridden)
    # Kill switch: HAIKU_FOLLOWUP_ROUTING = False (top of file). Audit which
    # model answered: SELECT model,count(*) FROM messages GROUP BY model;
    # NOTE: placed here deliberately — history/blocks/text_input must all be
    # defined first (the original placement at model resolution caused a
    # NameError → HTTP 500 on every message; fixed June 11).
    if (HAIKU_FOLLOWUP_ROUTING
            and provider == "claude"
            and model == DEFAULT_MODEL
            and len(history) >= 2
            # attachments/files check: typed text itself is always appended to
            # blocks as the last entry, so "has files" == more blocks than that.
            # (The original `not blocks` was ALWAYS false → routing never fired
            # — found via forensics June 11: 45/45 answers on Sonnet.)
            and len(blocks) <= (1 if text_input else 0)
            # pasted evidence detector: pastes go into the text box, so length
            # of the typed text IS the paste check
            and len((text_input or "").strip()) < 450
            and HAIKU_MODEL in MODEL_MAP):
        model = HAIKU_MODEL
        minfo = MODEL_MAP.get(model, minfo)
        log.info("ROUTED→HAIKU (followup, %d chars). Sonnet stays for openers/evidence.",
                 len(text_input or ""))
    # Inject persistent memories into context
    memories = get_all_memories()
    memory_prefix = ""
    if memories:
        memory_prefix = "PERSISTENT MEMORY (facts saved by user across all sessions):\n"
        for m in memories:
            memory_prefix += f"- {m['content']}\n"
        memory_prefix += "\n"

    # ── LEARNED PREFERENCES — how Naveen wants answers (high priority) ────────
    prefs = get_all_preferences()
    prefs_prefix = ""
    if prefs:
        prefs_prefix = ("HOW NAVEEN WANTS ANSWERS (learned preferences — apply these to EVERY answer, "
                        "they reflect what he has explicitly taught you over time):\n")
        for p in prefs:
            prefs_prefix += f"- {p['content']}\n"
        prefs_prefix += "\n"

    # ── TWO-BLOCK CACHE STRATEGY (the key cost fix) ──────────────────────────
    # Block 1 (STABLE): the system prompt — NEVER changes, so its cache survives
    #   the full 1h TTL and is reused across every question in a session.
    # Block 2 (VOLATILE): preferences + memory + uploaded files — these change,
    #   so they live in a SEPARATE cache block. When a preference/memory/file
    #   changes, only THIS small block re-caches; the big stable prompt stays warm.
    # Previously everything was one block with one breakpoint, so ANY change
    # (a saved memory, a learned preference) busted the entire prompt cache and
    # re-charged the full write cost every message. That was the 4-cent bug.
    stable_system = SYSTEM_PROMPT

    # ── CRISIS MODE (June 14) ────────────────────────────────────────────────
    # On a real production fire, step-by-step caution FEELS like flailing and
    # burns turns (the Longhorn ticket took 30 turns where decisiveness needed
    # ~10). When Naveen signals urgency, OR uses the Super depth button, switch
    # to decisive mode: commit to the single most-likely fix with compressed
    # reasoning, instead of "try this, no, try that." It still reads the error
    # and respects deploy order — it just stops hedging and leads with the
    # highest-probability action.
    _qlow = (text_input or "").lower()
    _crisis = ((depth_input or "").strip().lower() == "super") or any(s in _qlow for s in (
        "production down", "prod down", "urgent", "crisis", "emergency",
        "outage", "firefight", "fire fight", "asap", "everything is down",
        "down in prod", "critical"))
    if _crisis:
        stable_system = SYSTEM_PROMPT + (
            "\n\n══ CRISIS MODE (ACTIVE) ══\n"
            "This is a live fire. Naveen needs a decision, not a tour. For THIS "
            "response:\n"
            "• Lead with the SINGLE most-likely fix, stated as a direct action — "
            "not a list of possibilities. Commit. If you're 70% sure it's X, say "
            "'Do X' and give the exact command, not 'it could be X, Y, or Z.'\n"
            "• Compress the reasoning to one or two sentences of WHY, then the "
            "command. He can ask for depth if he wants it.\n"
            "• Still obey the hard safety rules: read the actual error first, "
            "respect deploy order (don't reference resources not yet deployed), "
            "and flag a genuinely destructive/irreversible step in one line before "
            "the command.\n"
            "• If you truly cannot pick without one piece of data, ask for that "
            "ONE command's output and nothing else — don't ask three questions.\n"
            "• No preamble, no 'great question', no recap of what he already knows. "
            "Action first.")

    # ── VOLATILE FREEZE (Phase 2): prefs+memory snapshot per batch window ────
    # Live prefs/memories changed the volatile block the instant anything was
    # saved mid-chat ("Saved. This is now rule #1..." → 3.3¢ spike in forensic
    # data) — busting the message cache behind it. We now freeze the rendered
    # prefs+memory text per chat and refresh it ONLY when the batch boundary
    # advances (prefix re-writes then anyway, so the refresh costs nothing
    # extra). Within this chat the model still knows any just-saved fact — the
    # user literally said it in the recent raw messages. New chats always get
    # the latest prefs/memories (fresh snapshot at boundary 0).
    _n_hist = len(history)
    if _n_hist <= RAW_WINDOW:
        _vol_boundary = 0
    else:
        _vol_boundary = max(0, ((_n_hist - RAW_WINDOW) // 10) * 10)

    volatile_core = None
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT volatile_snapshot, volatile_boundary FROM chats WHERE id=?",
                (chat_id,)
            ).fetchone()
        if row and row["volatile_boundary"] == _vol_boundary:
            volatile_core = row["volatile_snapshot"] or ""
    except Exception:
        volatile_core = None

    if volatile_core is None:
        volatile_core = ""
        if prefs_prefix:
            volatile_core += "\n\n" + prefs_prefix
        if memory_prefix:
            volatile_core += "\n\n" + memory_prefix
        try:
            with get_db() as conn:
                conn.execute(
                    "UPDATE chats SET volatile_snapshot=?, volatile_boundary=? WHERE id=?",
                    (volatile_core, _vol_boundary, chat_id)
                )
            log.info("Volatile freeze: snapshot refreshed at boundary %d", _vol_boundary)
        except Exception as e:
            log.warning("Volatile freeze save failed (non-fatal): %s", e)

    volatile_parts = volatile_core

    chat_files = get_chat_files(chat_id)
    if chat_files:
        file_section = "\n\n═══ UPLOADED FILES (always available, never compressed) ═══\n"
        file_section += "These files were uploaded in this conversation. Use them to answer questions.\n"
        for f in sorted(chat_files, key=lambda x: x.get('content','')[:80]):
            file_section += f"\n{f['content'][:12000]}\n"
        file_section += "═══ END OF FILES ═══\n"
        volatile_parts += file_section
        log.info("File store: %d file(s) injected into system context", len(chat_files))

    # ── KNOWLEDGE-BASE RETRIEVAL — injected into USER MESSAGE, not system block ──
    # ARCHITECTURE DECISION (cost-critical):
    # KB is injected into the USER MESSAGE, NOT into the volatile system block.
    #
    # WHY THIS MATTERS: if KB lived in volatile_parts (the system block), every
    # question would produce different KB results → volatile block changes →
    # Anthropic sees a new prefix → BUSTS the entire message cache → all recent
    # messages re-write at 1.25x penalty → adds ~4¢ of pure waste per question.
    #
    # In the user message: KB is "fresh input" (charged at 1x, not 1.25x), AND
    # the stable system prefix never changes → message cache READS at 0.10x.
    # This alone cuts per-question cost from 4-5¢ to ~1.5¢.
    kb_content_to_prepend = ""
    _kb_q_probe = _content_to_str(new_content).strip()
    # Issue 3 fix: skip KB (and its Ollama embedding call) for tiny follow-ups
    # like "ok", "yes", "continue", "thanks" — they carry no searchable intent
    # and just add latency. Anything 20+ chars still searches normally.
    if s.get("kb_enabled", "true") == "true" and len(_kb_q_probe) >= 20:
        q_parts = [_content_to_str(new_content)[:1500]]
        for h in reversed(history[-4:]):
            if h.get("role") == "user":
                q_parts.append(_content_to_str(h.get("content", ""))[:500])
                break
        kb_query = " ".join(q_parts)[:2000]
        kb_hits = kb_search(kb_query, k=3)
        # Budget: lean 2000 by default (keeps routine questions cheap). But when
        # a topic has VERSIONED knowledge, the authoritative current state must
        # never be squeezed out by older same-topic incidents — so raise it.
        # Raise further when Naveen explicitly asked for full history.
        _kb_budget = 2000
        if kb_hits.get("versioned"):
            _kb_budget = 9000 if any(v.get("history_full") for v in kb_hits["versioned"]) else 5000
        kb_block = kb_format_for_prompt(kb_hits, char_budget=_kb_budget)
        if kb_block:
            kb_content_to_prepend = kb_block
            n_inc = len(kb_hits.get("incidents", []))
            n_rb  = len(kb_hits.get("runbooks", []))
            n_dc  = len(kb_hits.get("docs", []))
            n_v   = len(kb_hits.get("versioned", []))
            log.info("KB: %d inc, %d rb, %d docs, %d versioned → user message (~%d chars, budget %d)",
                     n_inc, n_rb, n_dc, n_v, len(kb_block), _kb_budget)

    # ── SESSION FACTS — extract exact entity names from this user message ────
    # Regex-only (zero API cost). Updated BEFORE the request so the block the
    # model sees already includes names from the message being answered.
    facts_block = ""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT session_facts FROM chats WHERE id=?", (chat_id,)
            ).fetchone()
        facts_json = (row["session_facts"] if row else "") or ""
        # Extract from typed text AND text-type file blocks (pasted configs/logs)
        extract_src = text_input
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                extract_src += "\n" + b.get("text", "")
        facts_json = update_session_facts(facts_json, extract_src)
        with get_db() as conn:
            conn.execute("UPDATE chats SET session_facts=? WHERE id=?",
                         (facts_json, chat_id))
        facts_block = facts_format_for_prompt(facts_json)
        if facts_block:
            log.info("Session facts: %d chars pinned into user message", len(facts_block))
    except Exception as e:
        log.warning("Session facts extraction failed (non-fatal): %s", e)

    # Prepend KB to user message (after KB search, before context build)
    if kb_content_to_prepend:
        if isinstance(new_content, list):
            new_content = [{"type": "text", "text": kb_content_to_prepend}] + list(new_content)
        else:
            new_content = kb_content_to_prepend + "\n\n" + str(new_content)

    # Prepend SESSION FACTS (outermost — first thing the model reads in the
    # new user message). Same cache-safe pattern as KB: lives in the fresh
    # user message (1x input), never touches the cached system/history prefix.
    if facts_block:
        if isinstance(new_content, list):
            new_content = [{"type": "text", "text": facts_block}] + list(new_content)
        else:
            new_content = facts_block + "\n" + str(new_content)

    # ── FREE IN-CHAT RECALL (June 11, 2026) ──────────────────────────────────
    # When Naveen references earlier ticket context ("the secret we made",
    # "what was the name", "back to..."), search THIS chat's own old messages in
    # the local DB (zero API cost) and inject the best matches. Stops the
    # "re-show me what you already showed 200 messages ago" problem on long
    # multi-week tickets. Only fires on trigger phrases so normal turns pay
    # nothing. The injected block is capped (~1600 chars).
    try:
        _qt = (text_input or "").lower()
        if chat_id and any(trig in _qt for trig in _RECALL_TRIGGERS):
            recall_block = recall_from_chat_history(chat_id, text_input)
            if recall_block:
                if isinstance(new_content, list):
                    new_content = [{"type": "text", "text": recall_block}] + list(new_content)
                else:
                    new_content = recall_block + "\n" + str(new_content)
                log.info("Free recall: injected %d chars from chat history (no API cost)",
                         len(recall_block))
    except Exception as e:
        log.warning("Recall failed (non-fatal): %s", e)

    context = build_context(history, new_content, active_claude_key(s) or api_key, chat_id)

    # ── Helper: save assistant reply ──────────────────────────────────────────
    def save_reply(text: str, tin: int = 0, tout: int = 0,
                   cache_read: int = 0, cache_created: int = 0) -> float:
        # Never persist an empty assistant reply (June 14): a blank turn poisons
        # later requests ("messages.N must have non-empty content"). If a request
        # failed/streamed nothing, skip the DB write entirely.
        if not text or not text.strip():
            log.warning("save_reply skipped: empty assistant text (failed/aborted turn)")
            return 0.0
        # Accurate cost:
        # - cache_read tokens cost 10% of normal input price
        # - cache_created tokens cost 125% of normal input price (write penalty)
        # - fresh_in = tin - cache_read - cache_created (normal price)
        m = MODEL_MAP.get(model, {})
        in_price  = m.get("in",  3.0) / 1_000_000
        out_price = m.get("out", 15.0) / 1_000_000
        # API's input_tokens ALREADY excludes cached tokens — do NOT subtract.
        # Anthropic reports fresh / cache_read / cache_creation as separate fields.
        fresh_in  = max(0, tin)
        cost = (fresh_in       * in_price +
                cache_read     * in_price * 0.10 +
                cache_created  * in_price * 1.25 +
                tout           * out_price)
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(id, chat_id, role, content, tokens_in, tokens_out, model, created_at, cost_usd, "
                "cache_read, cache_created) "
                "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), chat_id, text, tin, tout, model, now_iso(), cost,
                 cache_read, cache_created)
            )
            conn.execute(
                "UPDATE chats SET updated_at = ?, tokens_used = tokens_used + ?, "
                "cost_usd = cost_usd + ? WHERE id = ?",
                (now_iso(), tin + tout, cost, chat_id)
            )
            conn.execute(
                "UPDATE settings SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = 'lifetime_tokens'",
                (tin + tout,)
            )
            conn.execute(
                "UPDATE settings SET value = CAST(CAST(value AS REAL) + ? AS TEXT) WHERE key = 'lifetime_cost'",
                (cost,)
            )
            # Per-key spend (June 14): each key keeps its OWN running spend so its
            # balance is independent. When you switch back to the primary it shows
            # the primary's remaining, not the backup's. Deduct this message's cost
            # from whichever key was active for it.
            _active_is_backup = str(load_settings().get("use_backup_key", "false")).lower() == "true"
            _spend_key = "backup_spend" if _active_is_backup else "primary_spend"
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS REAL) + ? AS TEXT)",
                (_spend_key, str(cost), cost)
            )
        # Write full conversation to chat.txt in the chat folder
        try:
            export_chat_txt(chat_id)
        except Exception as e:
            log.warning("chat.txt export failed: %s", e)
        return cost

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # ─────────────────────────────────────────────────────────────────────────
    #  PROVIDER GENERATORS
    # ─────────────────────────────────────────────────────────────────────────
    def gen_claude() -> Generator:
        """Optimised Claude generator — auto-cache + response mode + file detection."""
        client     = anthropic.Anthropic(api_key=api_key)
        full_text  = ""
        tokens_in  = 0
        tokens_out = 0
        cache_read = 0

        # ── Output token budget ──────────────────────────────────────────────
        # MATH: with prompt caching, input costs ~350 effective tokens/turn.
        # To hit 700 questions/1M tokens: output budget = ~1,080 tokens avg.
        # Setting max to 1,850 (Detailed) gives complete answers on any topic.
        # Average fill rate ~70% → ~1,295 output avg → 905 questions/1M ✓
        #
        # Concise (900): complete for quick questions, status checks
        # Detailed (1850): complete for technical deep-dives, YAML breakdowns
        # Default: Detailed — never cut off a technical answer mid-sentence
        #
        # Auto-override to Detailed when files are attached (PDF, image)
        has_files = isinstance(new_content, list) and any(
            isinstance(b, dict) and b.get("type") in ("document", "image")
            for b in new_content
        ) if isinstance(new_content, list) else False

        # ── AUTO-DEPTH: classify the question, set output ceiling accordingly ──
        # The cap is a CEILING, never a squeeze. Short answers cost less anyway.
        # Conservative: borderline questions go DEEPER, never shallower.
        #
        # FIX: the classifier must see the FULL content being sent, including
        # uploaded text/code/config files — not just the typed box. Otherwise a
        # big uploaded file scores 'standard' and the answer truncates at 8000.
        question_text = text_input if isinstance(text_input, str) else ""
        full_blob = question_text
        has_upload = False
        if isinstance(new_content, list):
            for b in new_content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type", "")
                if bt == "text":
                    txt = b.get("text", "")
                    # CRITICAL FIX (June 11): skip KB incidents and SESSION FACTS
                    # injected by the app — these are context helpers, not part of
                    # the user's question. Before this fix, 8 KB incidents + facts
                    # block added ~6,000 chars to full_blob, triggering "deep"
                    # (64k cap) for EVERY question in an established chat, even
                    # "ok let's proceed". Result: model sprawled to 3,000-token
                    # answers on trivial follow-ups at 4-5¢ each.
                    if (txt.startswith("[SESSION FACTS")
                            or txt.startswith("[RECALL")
                            or txt.startswith("\n═══ RELEVANT PAST INCIDENTS")
                            or txt.startswith("═══ RELEVANT PAST INCIDENTS")
                            or txt.startswith("📚 **Relevant KB")
                            or txt.startswith("**KB ")
                            or txt.startswith("Relevant incident")):
                        continue
                    if txt.startswith("**File:") or txt.startswith("**Archive:"):
                        has_upload = True
                    full_blob += " " + txt
                elif bt in ("document", "image"):
                    has_upload = True
        full_len = len(full_blob)
        # Classifier sees the combined text (typed + uploaded file content) and
        # whether any upload is present. This is the truncation fix.
        depth_tier, max_tokens = classify_depth(
            full_blob, has_files, full_content_len=full_len, has_upload=has_upload
        )

        # ── MANUAL TIER SELECTOR (overrides auto when set) ───────────────────
        # Naveen picks the depth himself via the UI (Normal/Average/Deep/Super).
        # Calibrated from his REAL work data (196 questions, Nexus 59-answer
        # ticket): his all-time biggest answer ever was 4,756 tokens, his Nexus
        # average was 913, and 66% of even his deep-work answers were <=1000.
        #   normal  = 1500  (quick facts, syntax, "what port")
        #   average = 4500  (his daily driver — concept + why + command)
        #   deep    = 10000 (big architectural answers; well above his 4756 max)
        #   super   = 64000 (rare massive RCA — never truncates, safety net)
        # Per-request override (from the request body) wins; else saved default;
        # else fall back to the auto-classifier result.
        tier_caps = {"normal": 1500, "average": 4500, "deep": 10000, "super": 64000}
        manual_tier = (depth_input or s.get("depth_choice") or "").strip().lower()
        if manual_tier in tier_caps:
            max_tokens  = tier_caps[manual_tier]
            depth_tier  = manual_tier
            depth_source = "manual"
        else:
            depth_source = "auto"

        log.info("Depth: tier=%s max_tokens=%d source=%s (files=%s, upload=%s, qlen=%d, fulllen=%d)",
                 depth_tier, max_tokens, depth_source, has_files, has_upload, len(question_text), full_len)

        use_web_search = s.get("web_search_enabled", "false") == "true"
        tools = [{"type": "web_search_20250305", "name": "web_search"}] if use_web_search else []

        try:
            # ── PREFIX CACHING (the big cost saver for long chats) ──────────────
            # Strategy: cache the STABLE PREFIX = system prompt + file store +
            # all history EXCEPT the brand-new question. Each turn, only the new
            # question is fresh full-price input; everything before it is a 90%-
            # discounted cache read. This is what makes a 30-message chat cost
            # almost the same as a 3-message one.
            #
            # We place a cache_control breakpoint on the LAST message of the
            # prefix (i.e. the second-to-last message overall — the previous
            # assistant turn). Anthropic caches everything up to and including
            # that breakpoint. The final user question after it stays fresh.
            cached_context = [dict(m) for m in context]

            def _mark_cache(msg):
                """Attach a 1h ephemeral cache breakpoint to a message's last block."""
                c = msg.get("content", "")
                if isinstance(c, str) and c:
                    msg["content"] = [{"type": "text", "text": c,
                                       "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
                elif isinstance(c, list) and c:
                    lb = dict(c[-1]) if isinstance(c[-1], dict) else None
                    if lb is not None and lb.get("type") != "document":
                        lb["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                        msg["content"] = list(c[:-1]) + [lb]
                return msg

            # TWO-BREAKPOINT CACHING (June 11 fix). The single floating
            # breakpoint at [-2] meant the ENTIRE history was one cache segment;
            # every turn the tail shifted and 7,000-11,000 tokens got rewritten
            # at 1.25x (proven in forensics: cache_created ~9k/turn = ~3¢ waste).
            # Now:
            #   Breakpoint 1 — on the frozen-summary block (the big stable mass
            #     of old history). It only changes every 10 messages, so this
            #     huge block stays cached for ~10 turns instead of every turn.
            #   Breakpoint 2 — on the second-to-last message (recent tail), as
            #     before, to catch the smaller recent-message layer.
            # Anthropic allows up to 4 cache breakpoints; using 2 here.
            summary_idx = None
            for i, m in enumerate(cached_context):
                c0 = m.get("content", "")
                txt0 = c0 if isinstance(c0, str) else (
                    c0[0].get("text", "") if isinstance(c0, list) and c0 and isinstance(c0[0], dict) else "")
                if txt0.startswith("[EARLIER CONTEXT — compressed]"):
                    summary_idx = i
                    break
            if summary_idx is not None:
                cached_context[summary_idx] = _mark_cache(cached_context[summary_idx])

            if len(cached_context) >= 2:
                cached_context[-2] = _mark_cache(cached_context[-2])
            elif cached_context:
                cached_context[-1] = _mark_cache(cached_context[-1])

            stream_kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower = more precise/deterministic — ideal for DevOps/code
                system=_build_system_blocks(stable_system, volatile_parts),
                messages=cached_context,
            )
            if tools:
                stream_kwargs["tools"] = tools

            extra_headers = {"anthropic-beta": "token-efficient-tools-2025-02-19,extended-cache-ttl-2025-04-11"}

            with client.messages.stream(**stream_kwargs, extra_headers=extra_headers) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
                    yield sse({"type": "text", "text": chunk})
                final      = stream.get_final_message()
                tokens_in  = final.usage.input_tokens
                tokens_out = final.usage.output_tokens
                cache_read = getattr(final.usage, "cache_read_input_tokens", 0) or 0
                cache_created = getattr(final.usage, "cache_creation_input_tokens", 0) or 0
                if cache_read or cache_created:
                    log.info("Cache: %d read (90%% off) | %d created", cache_read, cache_created)

        except anthropic.AuthenticationError:
            yield sse({"type": "error", "text": "Invalid Claude API key — check Settings"})
            return
        except anthropic.RateLimitError:
            yield sse({"type": "error", "text": "Claude rate limit reached — wait a moment"})
            return
        except anthropic.APIStatusError as exc:
            yield sse({"type": "error", "text": f"Claude API error: {exc.message}"})
            return
        except Exception as exc:
            yield sse({"type": "error", "text": f"Unexpected error: {exc}"})
            return

        cost = save_reply(full_text, tokens_in, tokens_out, cache_read, cache_created)
        log.info("Claude reply: %d in / %d out / cache_read=%d / $%.6f", tokens_in, tokens_out, cache_read, cost)

        # ── Auto-memory: detect save triggers in user message ─────────────
        # If user said "save this", "remember this", "save in memory" etc,
        # automatically save Claude's response summary to persistent memory
        import re as _re
        user_text_lower = (text_input or "").lower().strip()
        mem_patterns = [
            r"\bsave\b.*\bmemory\b",
            r"\bremember\b.*\b(this|that|it)\b",
            r"\bsave\b.*\b(this|that|it|status|progress|current)\b",
            r"\bkeep\b.*\bmemory\b",
            r"\block\b.*\b(this|that|it|in)\b",
            r"\bbookmark\b.*\b(this|that|it)\b",
            r"\badd\b.*\bto\b.*\bmemory\b",
        ]
        if any(_re.search(p, user_text_lower) for p in mem_patterns):
            try:
                # Use Haiku to create a compact memory entry from the reply
                mem_client = anthropic.Anthropic(api_key=api_key)
                mem_resp = mem_client.messages.create(
                    model=COMPRESS_MODEL, max_tokens=200,
                    messages=[{"role": "user", "content":
                        f"Create a compact memory entry (max 150 tokens) from this. "
                        f"Include: key facts, status, decisions, values. "
                        f"Start with a topic label in [brackets].\n\n{full_text[:3000]}"
                    }]
                )
                mem_text = mem_resp.content[0].text.strip()
                mem_id = str(uuid.uuid4())
                ts = now_iso()
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO memory (id,content,created_at,updated_at) VALUES (?,?,?,?)",
                        (mem_id, mem_text, ts, ts)
                    )
                log.info("Auto-memory saved: %d chars", len(mem_text))
                yield sse({"type": "memory_saved", "preview": mem_text[:100]})
            except Exception as me:
                log.warning("Auto-memory failed: %s", me)

        # ── Auto-PREFERENCE: detect when Naveen teaches HOW he wants answers ──
        # Triggers on durable-instruction phrasing ("from now on", "always",
        # "never", "I prefer", "stop doing", "in future"). Saves a learned
        # preference that applies to ALL future answers.
        pref_patterns = [
            r"\bfrom now on\b", r"\bin future\b", r"\bgoing forward\b",
            r"\balways\b", r"\bnever\b", r"\bi prefer\b", r"\bi'd prefer\b",
            r"\bi want you to\b", r"\bstop\b.*\b(doing|giving|saying)\b",
            r"\bremember (that )?i\b", r"\bevery time\b", r"\bby default\b",
            r"\bmake sure (to|you)\b",
        ]
        if any(_re.search(p, user_text_lower) for p in pref_patterns):
            try:
                pref_client = anthropic.Anthropic(api_key=api_key)
                pref_resp = pref_client.messages.create(
                    model=COMPRESS_MODEL, max_tokens=120,
                    messages=[{"role": "user", "content":
                        "Naveen just gave a durable instruction about HOW he wants answers from now on. "
                        "Extract it as ONE concise preference rule (max 40 words), phrased as an instruction "
                        "to the assistant (e.g. 'Always include the exact verification command'). "
                        "If this is NOT actually a durable preference about answer style/behaviour "
                        "(just a one-off request), reply with exactly: NONE.\n\n"
                        f"Naveen said: {(text_input or '')[:800]}"
                    }]
                )
                pref_text = pref_resp.content[0].text.strip()
                if pref_text and pref_text.upper() != "NONE" and len(pref_text) > 8:
                    with get_db() as conn:
                        # Avoid near-duplicates: skip if an identical pref exists
                        exists = conn.execute(
                            "SELECT 1 FROM preferences WHERE content=?", (pref_text,)
                        ).fetchone()
                        if not exists:
                            conn.execute(
                                "INSERT INTO preferences (id, content, created_at) VALUES (?,?,?)",
                                (str(uuid.uuid4()), pref_text, now_iso())
                            )
                            log.info("Auto-preference learned: %s", pref_text[:80])
                            yield sse({"type": "preference_saved", "preview": pref_text[:120]})
            except Exception as pe:
                log.warning("Auto-preference failed: %s", pe)

        # ── HAIKU REVIEWER PASS (the blueprint's validation layer) ───────────
        # After Sonnet finishes, optionally run a fast Haiku review to catch:
        #   - "add this empty block and try" incompleteness
        #   - missing dependencies/preconditions
        #   - unsafe commands
        #   - skipped validation step
        # If issues found, stream a "Reviewer note" appendix to the answer.
        # Cost: ~0.05-0.10c per question (Haiku is cheap). Quality lift: large.
        # User opt-in via settings.reviewer_enabled (default off).
        if (s.get("reviewer_enabled", "false") == "true"
                and full_text and len(full_text) > 200):
            try:
                review_client = anthropic.Anthropic(api_key=api_key)
                question_for_review = _content_to_str(new_content)[:3000]
                review_resp = review_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=400,
                    messages=[{"role": "user", "content":
                        "You are a senior reviewer. Naveen (a DevOps engineer at optadata) "
                        "needs answers that pass scrutiny meetings — defensible, complete, "
                        "with every dependency named and validation step included.\n\n"
                        f"QUESTION:\n{question_for_review}\n\n"
                        f"ANSWER GIVEN:\n{full_text[:6000]}\n\n"
                        "Check ONLY for serious issues:\n"
                        "  • Missing dependencies (e.g. a Kafka mTLS listener answer that doesn't mention KafkaUser cert + mounting it + reconfiguring the client)\n"
                        "  • Unsafe / data-loss commands without a warning\n"
                        "  • Skipped validation step (no way to verify the fix worked)\n"
                        "  • Factual mistakes\n"
                        "  • Vague hand-wave where a concrete command is required\n\n"
                        "If the answer is solid, reply with exactly: OK\n"
                        "Otherwise, reply with ONLY the missing/incorrect bits as a "
                        "concise bullet list (max 5 bullets). No preamble, no praise. "
                        "Focus on what would fail Naveen in a scrutiny meeting."
                    }]
                )
                review_text = review_resp.content[0].text.strip()
                # Cost of the review call
                rev_in  = getattr(review_resp.usage, "input_tokens", 0) or 0
                rev_out = getattr(review_resp.usage, "output_tokens", 0) or 0
                rev_cost = rev_in * (1.0/1_000_000) + rev_out * (5.0/1_000_000)
                cost += rev_cost
                if review_text.upper() != "OK" and len(review_text) > 20:
                    appendix = (
                        "\n\n---\n"
                        "**🔍 Reviewer note (Haiku check — issues to address before you ship this):**\n"
                        f"{review_text}"
                    )
                    yield sse({"type": "text", "value": appendix})
                    full_text += appendix
                    log.info("Reviewer flagged: %s", review_text[:120])
                else:
                    log.info("Reviewer approved (OK)")
            except Exception as re_err:
                log.warning("Reviewer failed (non-fatal): %s", re_err)

        yield sse({"type": "done", "tokens_in": tokens_in, "tokens_out": tokens_out,
                   "cost": round(cost, 8), "free": False, "cache_read": cache_read})

    def gen_groq() -> Generator:
        full_text = ""
        # Build OpenAI-compatible messages
        g_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in context:
            content = _content_to_str(m["content"])
            g_msgs.append({"role": m["role"], "content": content})

        payload = json.dumps({
            "model":       model,
            "messages":    g_msgs,
            "max_tokens":  16384,
            "temperature": 0.3,
            "stream":      True,
        }).encode()

        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk in ("[DONE]", ""):
                        continue
                    try:
                        obj  = json.loads(chunk)
                        text = obj["choices"][0]["delta"].get("content", "")
                        if text:
                            full_text += text
                            yield sse({"type": "text", "text": text})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            yield sse({"type": "error", "text": f"Groq HTTP {exc.code}: {body}"})
            return
        except Exception as exc:
            yield sse({"type": "error", "text": f"Groq error: {exc}"})
            return

        save_reply(full_text)
        yield sse({"type": "done", "tokens_in": 0, "tokens_out": 0, "cost": 0, "free": True})

    def gen_openai() -> Generator:
        """OpenAI generator with REAL cost tracking (June 14). Unlike the groq
        template, this reads OpenAI's actual `usage` field and passes true token
        counts to save_reply, which computes cost from MODEL_MAP prices. No
        free:true bug. o-series (o3/o4-mini) use 'developer' role for the system
        message and don't accept temperature; standard GPT models use 'system'."""
        full_text = ""
        is_reasoning = model.startswith("o")          # o3, o4-mini, ...
        sys_role = "developer" if is_reasoning else "system"
        o_msgs = [{"role": sys_role, "content": SYSTEM_PROMPT}]
        for m in context:
            o_msgs.append({"role": m["role"], "content": _content_to_str(m["content"])})

        body_obj = {
            "model":         model,
            "messages":      o_msgs,
            "stream":        True,
            "stream_options": {"include_usage": True},   # <-- usage in final chunk
        }
        if is_reasoning:
            body_obj["max_completion_tokens"] = 16384     # o-series param name
        else:
            body_obj["max_tokens"]  = 16384
            body_obj["temperature"] = 0.3
        payload = json.dumps(body_obj).encode()

        tok_in = tok_out = 0
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {openai_key}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk in ("[DONE]", ""):
                        continue
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    # usage arrives in a final chunk (choices may be empty there)
                    if obj.get("usage"):
                        tok_in  = obj["usage"].get("prompt_tokens", 0)
                        tok_out = obj["usage"].get("completion_tokens", 0)
                    for ch in obj.get("choices", []):
                        text = (ch.get("delta") or {}).get("content", "")
                        if text:
                            full_text += text
                            yield sse({"type": "text", "text": text})

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            yield sse({"type": "error", "text": f"OpenAI HTTP {exc.code}: {detail}"})
            return
        except Exception as exc:
            yield sse({"type": "error", "text": f"OpenAI error: {exc}"})
            return

        # REAL cost: save_reply computes from MODEL_MAP[model] in/out prices.
        cost = save_reply(full_text, tok_in, tok_out)
        yield sse({"type": "done", "tokens_in": tok_in, "tokens_out": tok_out,
                   "cost": cost, "free": False})

    # ── Dispatch ──────────────────────────────────────────────────────────────
    generators = {"claude": gen_claude, "groq": gen_groq, "openai": gen_openai}
    gen_fn     = generators.get(provider, gen_claude)

    def _with_keepalive(g):
        """Emit an immediate SSE comment so Cloudflare's tunnel sees a byte right
        away and won't cut the connection while the model reads a big paste
        before its first token (the 'Failed to fetch' on huge pastes). SSE
        comment lines start with ':' and are ignored by the client parser."""
        yield ": keep-alive\n\n"
        for chunk in g:
            yield chunk

    return Response(
        stream_with_context(_with_keepalive(gen_fn())),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — MEMORY
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/memory", methods=["GET"])
def api_memory_list():
    return jsonify(get_all_memories())

@app.route("/api/memory", methods=["POST"])
def api_memory_add():
    content = (request.json or {}).get("content", "").strip()
    if not content:
        return jsonify({"error": "Empty memory"}), 400
    mid = str(uuid.uuid4())
    ts  = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memory (id, content, created_at, updated_at) VALUES (?,?,?,?)",
            (mid, content, ts, ts)
        )
    log.info("Memory saved: %s", content[:60])
    return jsonify({"id": mid, "content": content, "created_at": ts})

@app.route("/api/memory/<mid>", methods=["DELETE"])
def api_memory_delete(mid):
    with get_db() as conn:
        conn.execute("DELETE FROM memory WHERE id=?", (mid,))
    return jsonify({"ok": True})


# ── PREFERENCES (how Naveen wants answers — learned + manual) ────────────────
@app.route("/api/preferences", methods=["GET"])
def api_prefs_list():
    return jsonify(get_all_preferences())

@app.route("/api/preferences", methods=["POST"])
def api_prefs_add():
    content = (request.json or {}).get("content", "").strip()
    if not content:
        return jsonify({"error": "Empty preference"}), 400
    pid = str(uuid.uuid4())
    ts  = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO preferences (id, content, created_at) VALUES (?,?,?)",
            (pid, content, ts)
        )
    log.info("Preference added: %s", content[:60])
    return jsonify({"id": pid, "content": content, "created_at": ts})

@app.route("/api/preferences/<pid>", methods=["DELETE"])
def api_prefs_delete(pid):
    with get_db() as conn:
        conn.execute("DELETE FROM preferences WHERE id=?", (pid,))
    return jsonify({"ok": True})


# ═════════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE-BASE ROUTES — Incidents, Runbooks, Docs, Search
# ═════════════════════════════════════════════════════════════════════════════
# These power the "Knowledge" tab in the UI: add resolved tickets as incidents,
# write runbooks for procedures, ingest official documentation. Everything here
# becomes searchable context injected into the model on every question.

@app.route("/api/kb/incidents", methods=["GET"])
def api_kb_incidents_list():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,title,symptoms,root_cause,fix,validation,tags,references_,created_at "
            "FROM incidents ORDER BY created_at DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/kb/incidents", methods=["POST"])
def api_kb_incidents_add():
    d = request.json or {}
    title       = (d.get("title") or "").strip()
    symptoms    = (d.get("symptoms") or "").strip()
    root_cause  = (d.get("root_cause") or "").strip()
    fix         = (d.get("fix") or "").strip()
    if not (title and symptoms and root_cause and fix):
        return jsonify({"error": "title, symptoms, root_cause, and fix are required"}), 400
    iid = kb_add_incident(
        title, symptoms, root_cause, fix,
        validation=(d.get("validation") or "").strip(),
        tags=(d.get("tags") or "").strip(),
        references=(d.get("references") or "").strip(),
    )
    log.info("KB incident added: %s", title[:60])
    return jsonify({"id": iid, "title": title})

@app.route("/api/kb/incidents/<iid>", methods=["DELETE"])
def api_kb_incidents_delete(iid):
    with get_db() as conn:
        conn.execute("DELETE FROM incidents WHERE id=?", (iid,))
        conn.execute("INSERT INTO incidents_fts(incidents_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    return jsonify({"ok": True})

@app.route("/api/kb/runbooks", methods=["GET"])
def api_kb_runbooks_list():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,title,scenario,steps,tags,created_at FROM runbooks ORDER BY created_at DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/kb/runbooks", methods=["POST"])
def api_kb_runbooks_add():
    d = request.json or {}
    title    = (d.get("title") or "").strip()
    scenario = (d.get("scenario") or "").strip()
    steps    = (d.get("steps") or "").strip()
    if not (title and scenario and steps):
        return jsonify({"error": "title, scenario, and steps are required"}), 400
    bid = kb_add_runbook(title, scenario, steps, tags=(d.get("tags") or "").strip())
    log.info("KB runbook added: %s", title[:60])
    return jsonify({"id": bid, "title": title})

@app.route("/api/kb/runbooks/<bid>", methods=["DELETE"])
def api_kb_runbooks_delete(bid):
    with get_db() as conn:
        conn.execute("DELETE FROM runbooks WHERE id=?", (bid,))
        conn.execute("INSERT INTO runbooks_fts(runbooks_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    return jsonify({"ok": True})

@app.route("/api/kb/docs", methods=["GET"])
def api_kb_docs_list():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, source, title, substr(content, 1, 300) AS preview, "
            "       length(content) AS len, tags, created_at "
            "FROM kb_docs ORDER BY created_at DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/kb/docs", methods=["POST"])
def api_kb_docs_add():
    d = request.json or {}
    source  = (d.get("source") or "").strip()
    title   = (d.get("title") or "").strip()
    content = (d.get("content") or "").strip()
    if not (source and title and content):
        return jsonify({"error": "source, title, and content are required"}), 400
    tags = (d.get("tags") or "").strip()
    chunks = chunk_text(content, chunk_chars=4000, overlap_chars=400)
    ids = []
    for i, ch in enumerate(chunks, 1):
        ctitle = title if len(chunks) == 1 else f"{title} (part {i}/{len(chunks)})"
        ids.append(kb_add_doc(source, ctitle, ch, tags=tags))
    log.info("KB doc added: %s (%d chunks)", title[:60], len(chunks))
    return jsonify({"ids": ids, "chunks": len(chunks)})

@app.route("/api/kb/docs/<did>", methods=["DELETE"])
def api_kb_docs_delete(did):
    with get_db() as conn:
        conn.execute("DELETE FROM kb_docs WHERE id=?", (did,))
        conn.execute("INSERT INTO kb_docs_fts(kb_docs_fts) VALUES('rebuild')")
        _EMB_CACHE.clear()   # invalidate semantic cache on any KB mutation
    return jsonify({"ok": True})


def _extract_text_from_upload(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded KB file. Supports PDF + common text
    formats. Returns '' if nothing extractable. Fully defensive — never raises."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if ext == "pdf":
            import pypdf, io as _io
            reader = pypdf.PdfReader(_io.BytesIO(raw))
            parts = []
            for pg in reader.pages:
                try:
                    t = pg.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                except Exception:
                    continue          # skip unreadable page, keep the rest
            return "\n\n".join(parts)
        if ext in ("txt", "md", "log", "yaml", "yml", "json", "conf", "ini",
                   "csv", "sh", "xml", "toml", "properties"):
            return raw.decode("utf-8", errors="replace")
        if ext == "docx":
            try:
                import docx, io as _io
                d = docx.Document(_io.BytesIO(raw))
                return "\n".join(p.text for p in d.paragraphs if p.text.strip())
            except Exception:
                return ""
    except Exception as e:
        log.warning("KB upload text extraction failed for %s: %s", filename, e)
    return ""


@app.route("/api/kb/upload", methods=["POST"])
def api_kb_upload():
    """Upload a PDF/text/docx file straight into the KB as searchable doc chunks.
    ADDITIVE endpoint — reuses the existing chunk_text + kb_add_doc pipeline that
    /api/kb/docs already uses. Past-incident PDFs become permanent, retrievable
    knowledge. Everything stays local (chunks in SQLite, embeddings via local
    Ollama). Nothing existing is modified."""
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded (form field 'file')"}), 400
    f = request.files["file"]
    fname = f.filename or "upload"
    raw = f.read()
    if not raw:
        return jsonify({"error": "empty file"}), 400
    if len(raw) > 40 * 1024 * 1024:                  # 40MB guard
        return jsonify({"error": "file too large (max 40MB)"}), 400

    title = (request.form.get("title") or fname.rsplit(".", 1)[0]).strip()
    source = (request.form.get("source") or fname).strip()
    tags = (request.form.get("tags") or "").strip()

    text = _extract_text_from_upload(fname, raw)
    if not text or len(text.strip()) < 20:
        return jsonify({"error": "could not extract readable text from this file "
                                 "(scanned/image PDF? try a text-based PDF)"}), 422

    chunks = chunk_text(text, chunk_chars=4000, overlap_chars=400)
    ids = []
    for i, ch in enumerate(chunks, 1):
        ctitle = title if len(chunks) == 1 else f"{title} (part {i}/{len(chunks)})"
        ids.append(kb_add_doc(source, ctitle, ch, tags=tags))
    log.info("KB upload: %s → %d chunks, %d chars extracted", fname, len(chunks), len(text))
    return jsonify({"ok": True, "title": title, "chunks": len(chunks),
                    "chars": len(text), "ids": ids})

@app.route("/api/kb/search", methods=["GET"])
def api_kb_search():
    """Preview retrieval for the user — useful for testing 'what does it find?'."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"incidents": [], "runbooks": [], "docs": []})
    return jsonify(kb_search(q, k=int(request.args.get("k", 5))))


# ─────────────────────────────────────────────────────────────────────────────
#  TOPIC TIMELINE — version-aware KB (Stage 1)
#  Drop a whole folder → merged into a dated "layer" under a topic.
#  Newest layer = CURRENT; older layers kept as history so the full arc
#  (e.g. Nexus 3.68 → 3.70 → 3.94) is preserved. Ingest/summary is FREE (local).
# ─────────────────────────────────────────────────────────────────────────────
def _local_summary(text: str, max_lines: int = 12) -> str:
    """Free, local, no-LLM summary: pull the most informative lines.
    Heuristic: prefer lines with version numbers, status words, headings.
    Zero tokens, zero cost — just gives a readable header for the layer."""
    import re as _re
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    scored = []
    kw = ("version", "status", "current", "migrat", "upgrade", "image",
          "deploy", "summary", "result", "done", "runs", "running", "3.",
          "26.", "keycloak", "nexus", "final", "note")
    for i, l in enumerate(lines[:600]):
        low = l.lower()
        s = 0
        if _re.search(r"\d+\.\d+", l): s += 3
        if l.startswith("#"): s += 3
        if any(k in low for k in kw): s += 2
        if l.endswith(":"): s += 1
        if 15 <= len(l) <= 200: s += 1
        scored.append((s, i, l.lstrip("# ").strip()))
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_lines]
    top = [l for _, _, l in sorted(top, key=lambda x: x[1])]  # restore order
    return "\n".join(top)


@app.route("/api/kb/topics", methods=["GET"])
def api_topics_list():
    with get_db() as conn:
        topics = [dict(r) for r in conn.execute(
            "SELECT * FROM topics ORDER BY name").fetchall()]
        for t in topics:
            layers = conn.execute(
                "SELECT id,label,summary,file_count,is_current,created_at "
                "FROM topic_layers WHERE topic_id=? "
                "ORDER BY created_at DESC", (t["id"],)).fetchall()
            t["layers"] = [dict(r) for r in layers]
    return jsonify(topics)


@app.route("/api/kb/topics/<tid>", methods=["GET"])
def api_topic_get(tid):
    with get_db() as conn:
        t = conn.execute("SELECT * FROM topics WHERE id=?", (tid,)).fetchone()
        if not t:
            return jsonify({"error": "not found"}), 404
        layers = [dict(r) for r in conn.execute(
            "SELECT * FROM topic_layers WHERE topic_id=? ORDER BY created_at DESC",
            (tid,)).fetchall()]
    out = dict(t); out["layers"] = layers
    return jsonify(out)


@app.route("/api/kb/topics/ingest", methods=["POST"])
def api_topic_ingest():
    """Ingest a whole folder as a NEW CURRENT layer under a topic.
    Body (multipart): topic (name), label (this layer's name),
    plus many files[] with their relative paths in 'paths' (JSON list).
    All local & free — no LLM tokens. Old layers kept, flipped to history."""
    topic_name = (request.form.get("topic") or "").strip()
    label      = (request.form.get("label") or "").strip()
    if not topic_name:
        return jsonify({"error": "topic name required"}), 400

    files = request.files.getlist("files")
    paths = []
    try:
        paths = json.loads(request.form.get("paths") or "[]")
    except Exception:
        paths = []
    if not files:
        return jsonify({"error": "no files in folder"}), 400

    # Merge every readable text file, deepest-path aware, skip binaries.
    TEXT_EXT = (".md", ".txt", ".yaml", ".yml", ".json", ".log", ".conf",
                ".cfg", ".ini", ".sh", ".env", ".properties", ".xml", ".toml",
                ".csv", ".tf", ".tpl", ".gotmpl", ".dockerfile", "")
    merged, tree, kept = [], [], 0
    for i, f in enumerate(files):
        rel = paths[i] if i < len(paths) else (f.filename or f"file{i}")
        low = rel.lower()
        if not (low.endswith(TEXT_EXT) or "." not in low.rsplit("/", 1)[-1]):
            continue
        raw = f.read()
        if not raw or len(raw) > 5 * 1024 * 1024:      # skip empty / >5MB blobs
            continue
        try:
            txt = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if len(txt) < 3:
            continue
        merged.append(f"\n\n===== FILE: {rel} =====\n{txt}")
        tree.append(rel)
        kept += 1

    if kept == 0:
        return jsonify({"error": "no readable text files found in folder"}), 422

    content = "".join(merged)
    if len(content) > 1_500_000:                       # 1.5MB cap per layer
        content = content[:1_500_000] + "\n\n[…truncated…]"
    summary = _local_summary(content)
    if not label:
        label = f"{topic_name} — {datetime.now():%Y-%m-%d}"

    now = datetime.now().isoformat()
    with get_db() as conn:
        row = conn.execute("SELECT id FROM topics WHERE name=?",
                           (topic_name,)).fetchone()
        if row:
            topic_id = row["id"]
            # flip existing layers to history
            conn.execute("UPDATE topic_layers SET is_current=0 WHERE topic_id=?",
                        (topic_id,))
        else:
            topic_id = uuid.uuid4().hex
            conn.execute("INSERT INTO topics (id,name,created_at) VALUES (?,?,?)",
                        (topic_id, topic_name, now))
        layer_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO topic_layers "
            "(id,topic_id,label,summary,content,file_tree,file_count,is_current,created_at) "
            "VALUES (?,?,?,?,?,?,?,1,?)",
            (layer_id, topic_id, label, summary, content,
             "\n".join(tree), kept, now))
        conn.commit()

    log.info("[topic] ingested %d files into '%s' layer '%s'",
             kept, topic_name, label)
    return jsonify({"ok": True, "topic": topic_name, "label": label,
                    "files_ingested": kept, "chars": len(content),
                    "summary": summary})


@app.route("/api/kb/topics/<tid>", methods=["DELETE"])
def api_topic_delete(tid):
    with get_db() as conn:
        conn.execute("DELETE FROM topic_layers WHERE topic_id=?", (tid,))
        conn.execute("DELETE FROM topics WHERE id=?", (tid,))
        conn.commit()
    return jsonify({"ok": True})


@app.route("/api/kb/topics/layer/<lid>", methods=["DELETE"])
def api_topic_layer_delete(lid):
    with get_db() as conn:
        conn.execute("DELETE FROM topic_layers WHERE id=?", (lid,))
        conn.commit()
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED KNOWLEDGE BASE — one flat numbered list over incidents+runbooks+docs,
#  with versions stacked on top. Originals are never moved; this is a view layer.
# ─────────────────────────────────────────────────────────────────────────────
def _unified_entries():
    """Return all KB entries as one numbered list (stable order = oldest first),
    each with its version stack (v1 = the original, v2+ from kb_versions)."""
    rows = []
    with get_db() as conn:
        for r in conn.execute(
                "SELECT id,title,created_at,'incident' AS kind FROM incidents").fetchall():
            rows.append(dict(r))
        for r in conn.execute(
                "SELECT id,title,created_at,'runbook' AS kind FROM runbooks").fetchall():
            rows.append(dict(r))
        for r in conn.execute(
                "SELECT id,title,created_at,'doc' AS kind FROM kb_docs").fetchall():
            rows.append(dict(r))
        rows.sort(key=lambda x: x["created_at"])       # stable numbering
        # attach versions
        for i, e in enumerate(rows, 1):
            e["number"] = i
            vers = conn.execute(
                "SELECT id,version_no,label,summary,file_count,is_current,created_at "
                "FROM kb_versions WHERE entry_ref=? ORDER BY version_no",
                (e["id"],)).fetchall()
            e["versions"] = [dict(v) for v in vers]
            e["current_label"] = (e["versions"][-1]["label"]
                                  if e["versions"] else e["title"])
            e["version_count"] = 1 + len(e["versions"])
    return rows


@app.route("/api/kb/entries", methods=["GET"])
def api_kb_entries():
    return jsonify(_unified_entries())


@app.route("/api/kb/entries/<num>/detail", methods=["GET"])
def api_kb_entry_detail(num):
    """Full content of entry #num: the origin (v1) body from its source table,
    plus every stacked version's content. Used when a modal row is expanded."""
    try:
        n = int(num)
    except ValueError:
        return jsonify({"error": "bad number"}), 400
    entries = _unified_entries()
    e = next((x for x in entries if x["number"] == n), None)
    if not e:
        return jsonify({"error": "not found"}), 404

    # origin (v1) content from whichever table it lives in
    origin = ""
    with get_db() as conn:
        if e["kind"] == "incident":
            r = conn.execute("SELECT symptoms,root_cause,fix,validation FROM incidents WHERE id=?",
                            (e["id"],)).fetchone()
            if r:
                origin = (f"Symptoms:\n{r['symptoms']}\n\nRoot cause:\n{r['root_cause']}"
                          f"\n\nFix:\n{r['fix']}"
                          + (f"\n\nValidation:\n{r['validation']}" if r['validation'] else ""))
        elif e["kind"] == "runbook":
            r = conn.execute("SELECT scenario,steps FROM runbooks WHERE id=?",
                            (e["id"],)).fetchone()
            if r:
                origin = f"Scenario:\n{r['scenario']}\n\nSteps:\n{r['steps']}"
        else:
            r = conn.execute("SELECT content FROM kb_docs WHERE id=?",
                            (e["id"],)).fetchone()
            if r:
                origin = r["content"]
        vers = [dict(v) for v in conn.execute(
            "SELECT id,version_no,label,content,file_count,is_current,created_at "
            "FROM kb_versions WHERE entry_ref=? ORDER BY version_no", (e["id"],)).fetchall()]

    return jsonify({"number": n, "title": e["title"], "kind": e["kind"],
                    "origin": origin, "versions": vers})


@app.route("/api/kb/entries/add", methods=["POST"])
def api_kb_entries_add():
    """Add to the unified KB. Two paths, both local & free:
      • blank kb_id  → create a BRAND-NEW numbered entry (stored as a kb_doc v1)
      • kb_id = N    → append a NEW CURRENT version onto entry number N
    Content comes from either an uploaded folder (files[]) or pasted text."""
    topic = (request.form.get("topic") or "").strip()
    kb_num = (request.form.get("kb_id") or "").strip()
    pasted = (request.form.get("content") or "").strip()
    if not topic:
        return jsonify({"error": "Topic name required."}), 400

    # Gather content: folder upload OR pasted text
    files = request.files.getlist("files")
    content, tree, kept = "", [], 0
    if files:
        try:
            paths = json.loads(request.form.get("paths") or "[]")
        except Exception:
            paths = []
        TEXT_EXT = (".md", ".txt", ".yaml", ".yml", ".json", ".log", ".conf",
                    ".cfg", ".ini", ".sh", ".env", ".properties", ".xml",
                    ".toml", ".csv", ".tf", ".tpl", ".gotmpl", ".dockerfile", "")
        merged = []
        for i, f in enumerate(files):
            rel = paths[i] if i < len(paths) else (f.filename or f"file{i}")
            low = rel.lower()
            if not (low.endswith(TEXT_EXT) or "." not in low.rsplit("/", 1)[-1]):
                continue
            raw = f.read()
            if not raw or len(raw) > 5 * 1024 * 1024:
                continue
            txt = raw.decode("utf-8", errors="ignore").strip()
            if len(txt) < 3:
                continue
            merged.append(f"\n\n===== FILE: {rel} =====\n{txt}")
            tree.append(rel); kept += 1
        content = "".join(merged)
    elif pasted:
        content = pasted
        kept = 1
        tree = ["(pasted text)"]
    else:
        return jsonify({"error": "Provide a folder to upload or paste some text."}), 400

    if not content.strip():
        return jsonify({"error": "No readable content found."}), 422
    if len(content) > 1_500_000:
        content = content[:1_500_000] + "\n\n[…truncated…]"
    summary = _local_summary(content)
    now = datetime.now().isoformat()

    # ── Path A: brand-new numbered entry (blank kb_id) ──
    if not kb_num:
        did = kb_add_doc("folder-import" if files else "pasted",
                         topic, content, tags="")
        return jsonify({"ok": True, "mode": "new_entry", "topic": topic,
                        "files_ingested": kept,
                        "message": f"Created new KB entry '{topic}'."})

    # ── Path B: version onto existing entry number N ──
    entries = _unified_entries()
    try:
        n = int(kb_num)
    except ValueError:
        return jsonify({"error": f"KB ID must be a number (you entered '{kb_num}')."}), 400
    match = next((e for e in entries if e["number"] == n), None)
    if not match:
        return jsonify({"error": f"No KB entry with ID {n}. You have {len(entries)} entries (1–{len(entries)})."}), 404

    with get_db() as conn:
        # next version number: existing versions + 1 (v1 = original)
        existing = conn.execute(
            "SELECT COUNT(*) AS c FROM kb_versions WHERE entry_ref=?",
            (match["id"],)).fetchone()["c"]
        next_v = existing + 2                    # original is v1, so first add = v2
        conn.execute("UPDATE kb_versions SET is_current=0 WHERE entry_ref=?",
                    (match["id"],))
        vid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO kb_versions "
            "(id,entry_ref,version_no,label,summary,content,file_tree,file_count,is_current,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,1,?)",
            (vid, match["id"], next_v, topic, summary, content,
             "\n".join(tree), kept, now))
        # keep the FTS index in sync (this content had ZERO search wiring before)
        conn.execute(
            "INSERT INTO kb_versions_fts (version_id,entry_ref,entry_title,label,content) "
            "VALUES (?,?,?,?,?)",
            (vid, match["id"], match["title"], topic, content))
        conn.commit()
    return jsonify({"ok": True, "mode": "versioned", "entry_number": n,
                    "entry_title": match["title"], "new_version": next_v,
                    "files_ingested": kept,
                    "message": f"Added v{next_v} '{topic}' onto entry #{n} "
                               f"({match['title']}). Now current; older kept as history."})


@app.route("/api/kb/entries/version/<vid>", methods=["DELETE"])
def api_kb_version_delete(vid):
    with get_db() as conn:
        conn.execute("DELETE FROM kb_versions WHERE id=?", (vid,))
        conn.execute("DELETE FROM kb_versions_fts WHERE version_id=?", (vid,))
        conn.commit()
    return jsonify({"ok": True})

@app.route("/api/kb/from_chat/<chat_id>", methods=["POST"])
def api_kb_from_chat(chat_id):
    """Turn the current chat into an incident, using Haiku to distill it.
    The user can also pass title/symptoms/root_cause/fix explicitly to skip
    LLM distillation and save exactly what they typed.
    """
    d = request.json or {}
    explicit = all(d.get(k) for k in ("title", "symptoms", "root_cause", "fix"))
    if explicit:
        iid = kb_add_incident(
            d["title"], d["symptoms"], d["root_cause"], d["fix"],
            validation=d.get("validation", ""), tags=d.get("tags", ""),
            references=d.get("references", "")
        )
        return jsonify({"id": iid, "via": "explicit"})

    # LLM-assisted distillation from chat transcript
    s = load_settings()
    api_key = (d.get("api_key") or "").strip() or active_claude_key(s)
    if not api_key:
        return jsonify({"error": "no api_key (provide one or save in settings)"}), 400

    with get_db() as conn:
        chat = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not chat:
            return jsonify({"error": "chat not found"}), 404
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY created_at",
            (chat_id,)
        ).fetchall()

    transcript = "\n\n".join(
        f"[{m['role'].upper()}]: {_content_to_str(m['content'])[:1500]}"
        for m in msgs
    )[:18000]

    def _haiku_distill_call(extra_instruction=""):
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=COMPRESS_MODEL, max_tokens=1500,
            messages=[{"role": "user", "content":
                "Distill this troubleshooting conversation into a structured incident. "
                "Respond ONLY with this exact format (no preamble, no markdown, plain "
                "text labels exactly as shown):\n\n"
                "TITLE: <one-line summary>\n"
                "SYMPTOMS: <what the user observed>\n"
                "ROOT_CAUSE: <what was actually wrong>\n"
                "FIX: <the working solution with concrete commands>\n"
                "VALIDATION: <how to verify the fix worked>\n"
                "TAGS: <comma-separated, SPECIFIC tags>\n"
                "TAG RULES (critical — bad tags make the KB retrieve the wrong ticket):\n"
                "  • ALWAYS include the cluster/environment as the FIRST tag if identifiable: "
                "'maint' (maintenance cluster, which IS production) or 'dev' or 'prod'. If the "
                "transcript mentions maint/k8mgmt/production → tag 'maint'. If dev/staging → 'dev'.\n"
                "  • Then the specific component AND sub-topic, e.g. 'nexus,blob-store' or "
                "'nexus,crowd-plugin' or 'nexus,group-repo' — NOT just 'nexus'.\n"
                "  • NEVER use broad-only tags like just 'kubernetes' or just 'cluster' — they match "
                "everything and are useless. Every tag set must let someone find THIS exact ticket "
                "among 20 similar ones. Good: 'maint,nexus,crowd-plugin,statefulset'. "
                "Bad: 'kubernetes,nexus,cluster'.\n"
                + extra_instruction +
                f"\nChat title: {chat['title']}\n\nTranscript:\n{transcript}"
            }]
        )
        return resp.content[0].text.strip()

    import re as _re
    def _grab(label, body):
        # tolerant: optional markdown bold/heading around labels, any case
        m = _re.search(
            rf"[*#\s]*{label}[*#]*\s*:\s*(.+?)(?=\n[*#\s]*[A-Z_]{{3,}}[*#]*\s*:|$)",
            body, _re.S | _re.I)
        return m.group(1).strip().strip("*").strip() if m else ""

    def _parse(body):
        body = body.replace("```", "").strip()
        return {
            "title":      _grab("TITLE", body),
            "symptoms":   _grab("SYMPTOMS", body),
            "root_cause": _grab("ROOT_CAUSE", body),
            "fix":        _grab("FIX", body),
            "validation": _grab("VALIDATION", body),
            "tags":       _grab("TAGS", body),
        }

    try:
        text = _haiku_distill_call()
        parsed = _parse(text)
        if not (parsed["title"] and parsed["symptoms"]
                and parsed["root_cause"] and parsed["fix"]):
            # one automatic retry with a sterner instruction — same medicine
            # that fixed the identical failure in backfill_kb.py
            text = _haiku_distill_call(
                "\nIMPORTANT: your previous attempt was unparseable. Output the six "
                "labels EXACTLY as plain text at line starts, nothing else.\n")
            parsed = _parse(text)
    except Exception as e:
        return jsonify({"error": f"distillation failed: {e}"}), 500

    if not (parsed["title"] and parsed["symptoms"] and parsed["root_cause"] and parsed["fix"]):
        return jsonify({"error": "could not parse distillation", "raw": text}), 500

    # Preview mode (June 14): return the distilled fields WITHOUT saving, so the
    # UI can let Naveen confirm/edit (especially tags like 'maint'/'dev') before
    # it becomes a permanent KB fact. A second call with explicit fields saves.
    if d.get("preview"):
        return jsonify({"via": "preview", **parsed})

    iid = kb_add_incident(**parsed)
    return jsonify({"id": iid, "via": "distilled", "parsed": parsed})


# ─── REVIEWER toggle ─────────────────────────────────────────────────────────
@app.route("/api/kb/stats", methods=["GET"])
def api_kb_stats():
    with get_db() as conn:
        i = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        r = conn.execute("SELECT COUNT(*) FROM runbooks").fetchone()[0]
        d = conn.execute("SELECT COUNT(*) FROM kb_docs").fetchone()[0]
    return jsonify({"incidents": i, "runbooks": r, "docs": d})


# ── SPEND TRACKING (today's consumption for nightly budgeting) ───────────────
@app.route("/api/spend/today", methods=["GET"])
def api_spend_today():
    """Today's spend, computed by summing the exact per-message cost for messages
    created during the LOCAL calendar day. Messages store UTC timestamps, so we
    convert the local day's start/end to UTC and sum within that window. This is
    the most accurate number the tool can produce — it sums the same costs that
    were charged at reply time, not a re-derivation.
    """
    from datetime import datetime, timezone

    # Local day boundaries (the Mac's timezone = Naveen's day in Germany)
    now_local   = datetime.now()
    day_start_l = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert local day-start to UTC to match stored timestamps
    local_tz    = datetime.now(timezone.utc).astimezone().tzinfo
    day_start_utc = day_start_l.replace(tzinfo=local_tz).astimezone(timezone.utc)
    start_iso   = day_start_utc.replace(tzinfo=None).isoformat(timespec="seconds")

    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS spend, COUNT(*) AS n "
            "FROM messages WHERE role='assistant' AND created_at >= ?",
            (start_iso,)
        ).fetchone()
        # Per-model breakdown so we can split by provider (Claude vs OpenAI)
        per_model = conn.execute(
            "SELECT model, COALESCE(SUM(cost_usd),0) AS spend, COUNT(*) AS n "
            "FROM messages WHERE role='assistant' AND created_at >= ? GROUP BY model",
            (start_iso,)
        ).fetchall()
    spend = float(row["spend"] or 0.0)
    count = int(row["n"] or 0)

    # Roll model-level spend up to provider level
    prov_spend = {"claude": 0.0, "openai": 0.0, "other": 0.0}
    for r in per_model:
        prov = MODEL_MAP.get(r["model"], {}).get("provider", "other")
        if prov not in prov_spend:
            prov = "other"
        prov_spend[prov] += float(r["spend"] or 0.0)

    # All-time OpenAI spend (since balance can't be fetched, total spend is the
    # honest number — Naveen tracks his own top-up amount).
    with get_db() as conn:
        oa_rows = conn.execute(
            "SELECT model, COALESCE(SUM(cost_usd),0) AS spend "
            "FROM messages WHERE role='assistant' GROUP BY model"
        ).fetchall()
    openai_total = sum(
        float(r["spend"] or 0.0)
        for r in oa_rows
        if MODEL_MAP.get(r["model"], {}).get("provider") == "openai"
    )

    return jsonify({
        "today_spend": round(spend, 4),
        "today_count": count,
        "today_claude": round(prov_spend["claude"], 4),
        "today_openai": round(prov_spend["openai"], 4),
        "openai_total": round(openai_total, 4),
        "since_utc":   start_iso,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — DB CLEANUP
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/chats/cleanup", methods=["GET"])
def api_chat_cleanup_list():
    """List chat folders on disk and whether each still has a chat in the DB."""
    with get_db() as conn:
        active_ids = {r[0][:8]: r[0] for r in conn.execute("SELECT id FROM chats").fetchall()}
    folders = []
    for d in _walk_chat_dirs():
        short = d.name.rsplit("__", 1)[-1]
        folders.append({"folder": str(d.relative_to(CHATS_DIR)), "chat_id": active_ids.get(short, short),
                        "active": short in active_ids})
    return jsonify({"folders": folders, "chats_dir": str(CHATS_DIR)})


@app.route("/api/chats/cleanup/orphans", methods=["DELETE"])
def api_chat_cleanup_delete():
    """Tidy chat folders whose chat no longer exists in the DB. They are MOVED
    into chats/_Orphan/ (kept as a record), never deleted. Only real chat
    folders (…__<id>) are touched — never Unfiled/ or your UI folders."""
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify({"error": "local only"}), 403
    with get_db() as conn:
        active = {r[0][:8] for r in conn.execute("SELECT id FROM chats").fetchall()}
    moved = []
    orphan_root = CHATS_DIR / "_Orphan"
    with _CHAT_FS_LOCK:
        for d in list(_walk_chat_dirs()):
            if d.name.rsplit("__", 1)[-1] in active:
                continue
            orphan_root.mkdir(exist_ok=True)
            target = orphan_root / d.name
            if target.exists():
                target = orphan_root / f"{d.name} (moved {datetime.now():%Y%m%d-%H%M%S})"
            old_parent = d.parent
            os.rename(str(d), str(target))
            _prune_empty(old_parent)
            moved.append(d.name)
            log.info("Moved orphan chat folder to _Orphan/: %s", d.name)
    return jsonify({"moved_to_orphan": moved, "count": len(moved)})


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — DROPBOX
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/dropbox/upload", methods=["POST"])
def api_dropbox_upload():
    """Receive files from phone/browser, save to ~/Documents/local-llm-dropbox/"""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    saved = []
    errors = []
    for f in request.files.getlist("files"):
        if not f.filename:
            continue
        # Safe filename
        safe_name = re.sub(r'[^\w\s\-\.]', '_', f.filename).strip()
        if not safe_name:
            safe_name = f"file_{uuid.uuid4().hex[:8]}"
        # Add timestamp prefix to avoid collisions
        ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"{ts_prefix}_{safe_name}"
        dest = DROP_DIR / final_name
        try:
            f.save(str(dest))
            size = dest.stat().st_size
            saved.append({"name": final_name, "size": size})
            log.info("Dropbox: saved %s (%d bytes)", final_name, size)
        except Exception as e:
            errors.append({"name": f.filename, "error": str(e)})
            log.error("Dropbox: failed to save %s: %s", f.filename, e)

    return jsonify({"saved": saved, "errors": errors, "drop_dir": str(DROP_DIR)})


@app.route("/api/dropbox/files", methods=["GET"])
def api_dropbox_list():
    """List files in the dropbox folder."""
    try:
        files = []
        for f in sorted(DROP_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                stat = f.stat()
                files.append({
                    "name":     f.name,
                    "size":     stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %H:%M"),
                    "ext":      f.suffix.lower().lstrip(".") or "file",
                })
        return jsonify({"files": files, "drop_dir": str(DROP_DIR)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dropbox/files/<filename>", methods=["DELETE"])
def api_dropbox_delete(filename: str):
    """Delete a file from the dropbox."""
    safe = re.sub(r'[/\\]', '', filename)
    target = DROP_DIR / safe
    if target.exists() and target.is_file():
        target.unlink()
        log.info("Dropbox: deleted %s", safe)
        return jsonify({"ok": True})
    return jsonify({"error": "File not found"}), 404


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — STATS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    with get_db() as conn:
        totals = conn.execute(
            "SELECT SUM(tokens_used) AS tokens, SUM(cost_usd) AS cost, "
            "COUNT(*) AS chats FROM chats"
        ).fetchone()
        msgs = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE role = 'user'"
        ).fetchone()

    with get_db() as conn2:
        lt_t = conn2.execute("SELECT value FROM settings WHERE key='lifetime_tokens'").fetchone()
        lt_c = conn2.execute("SELECT value FROM settings WHERE key='lifetime_cost'").fetchone()
    lt_tokens = int(lt_t[0]) if lt_t else 0
    lt_cost   = float(lt_c[0]) if lt_c else 0.0
    db_tokens = int(totals["tokens"] or 0)
    db_cost   = float(totals["cost"] or 0)
    s = load_settings()
    use_backup = str(s.get("use_backup_key", "false")).lower() == "true"

    # Per-key independent balances (June 14). Each key has its OWN starting credit
    # and its OWN accumulated spend, so switching between keys shows the correct
    # remaining for whichever is active — primary keeps its ~$16, backup keeps its
    # $100, and neither is affected by the other's usage.
    if use_backup:
        starting_credit = float(s.get("backup_starting_credit", "100.0") or 100.0)
        key_spend = float(s.get("backup_spend", "0.0") or 0.0)
    else:
        starting_credit = float(s.get("starting_credit", "5.98") or 5.98)
        key_spend = float(s.get("primary_spend", "0.0") or 0.0)

    # Back-compat: if primary_spend was never initialised (existing users), seed it
    # from lifetime_cost so the primary balance stays continuous after this update.
    if not use_backup and s.get("primary_spend") is None:
        seed = round(max(db_cost, lt_cost), 6)
        save_setting("primary_spend", str(seed))
        key_spend = seed

    total_cost = round(key_spend, 6)
    return jsonify({
        "total_tokens":    max(db_tokens, lt_tokens),
        "total_cost_usd":  total_cost,
        "starting_credit": starting_credit,
        "remaining":       round(max(0.0, starting_credit - total_cost), 6),
        "active_key":      "backup" if use_backup else "primary",
        "total_chats":     int(totals["chats"]),
        "total_messages":  int(msgs["c"]),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — MODEL DISCOVERY  (auto-detect new Claude models, zero code changes)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/models/discover", methods=["POST"])
def api_models_discover():
    """Discover all available models from Anthropic API."""
    s = load_settings()
    api_key = active_claude_key(s)
    if not api_key:
        return jsonify({"error": "No API key"}), 401
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models?limit=100",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    existing_ids = {m["id"] for m in MODELS}
    discovered = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        if m.get("type") != "model": continue
        discovered.append({"id": mid, "name": m.get("display_name", mid), "exists": mid in existing_ids})
    discovered.sort(key=lambda x: (x["exists"], x["name"]))
    return jsonify({"models": discovered, "new_count": sum(1 for d in discovered if not d["exists"])})


@app.route("/api/models/add", methods=["POST"])
def api_models_add():
    """Add a discovered model at runtime — persists across restarts."""
    data = request.json or {}
    model_id = data.get("id", "").strip()
    name = data.get("name", "").strip() or model_id
    if not model_id: return jsonify({"error": "No model ID"}), 400
    if any(m["id"] == model_id for m in MODELS): return jsonify({"error": "Already exists"}), 400
    if "opus" in model_id.lower(): inp, outp = 5.0, 25.0
    elif "sonnet" in model_id.lower(): inp, outp = 3.0, 15.0
    elif "haiku" in model_id.lower(): inp, outp = 1.0, 5.0
    else: inp, outp = 3.0, 15.0
    new_model = {"id": model_id, "name": name, "group": "Claude", "provider": "claude",
                 "desc": "Discovered via API", "in": inp, "out": outp, "free": False, "ctx": "200K"}
    MODELS.append(new_model)
    custom = json.loads(load_settings().get("custom_models", "[]"))
    custom.append(new_model)
    save_setting("custom_models", json.dumps(custom))
    log.info("Model added: %s", model_id)
    return jsonify({"ok": True, "model": new_model})


@app.route("/api/models/remove", methods=["POST"])
def api_models_remove():
    """Remove a model from the MODELS list. Persists across restarts."""
    data     = request.json or {}
    model_id = data.get("id", "").strip()
    if not model_id:
        return jsonify({"error": "No model ID"}), 400

    # Don't allow removing the last model
    if len(MODELS) <= 1:
        return jsonify({"error": "Cannot remove the last model"}), 400

    # Remove from runtime list
    before = len(MODELS)
    MODELS[:] = [m for m in MODELS if m["id"] != model_id]
    if len(MODELS) == before:
        return jsonify({"error": f"{model_id} not found"}), 404

    # Remove from persisted custom models
    try:
        custom = json.loads(load_settings().get("custom_models", "[]"))
        custom = [m for m in custom if m.get("id") != model_id]
        save_setting("custom_models", json.dumps(custom))
    except Exception:
        pass

    log.info("Model removed: %s", model_id)
    return jsonify({"ok": True, "removed": model_id})


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — VOICE TRANSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8",
                                      download_root=str(BASE_DIR / "whisper_models"))
        log.info("Whisper model loaded")
    return _whisper_model

@app.route("/api/voice/transcribe", methods=["POST"])
def api_voice_transcribe():
    import tempfile, os as _os
    if "audio" not in request.files: return jsonify({"error": "No audio"}), 400
    raw = request.files["audio"].read()
    if not raw: return jsonify({"error": "Empty"}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw); tmp_path = tmp.name
        model = get_whisper_model()
        segs, info = model.transcribe(tmp_path, beam_size=5, language=None,
                                      vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
        transcript = " ".join(s.text.strip() for s in segs).strip()
        log.info("Voice: %.1fs → %d chars (%s)", info.duration, len(transcript), info.language)
        return jsonify({"text": transcript, "language": info.language, "duration": round(info.duration, 1)})
    except Exception as exc:
        log.error("Voice error: %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path:
            try: _os.unlink(tmp_path)
            except: pass


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — PROJECT SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chats/<chat_id>/snapshot", methods=["POST"])
def api_chat_snapshot(chat_id: str):
    data = request.json or {}
    label = data.get("label", "").strip()
    if not label: return jsonify({"error": "Provide a label"}), 400
    s = load_settings()
    api_key = active_claude_key(s)
    if not api_key: return jsonify({"error": "No API key"}), 401
    history = get_chat_history(chat_id)
    if not history: return jsonify({"error": "No messages"}), 400
    transcript = ""
    for m in history:
        role = "U" if m["role"] == "user" else "A"
        transcript += f"\n{role}: {_content_to_str(m['content'])[:800]}"
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(model=COMPRESS_MODEL, max_tokens=600,
            messages=[{"role": "user", "content": (
                f"Dense technical snapshot for '{label}'. Include: goal, steps done, "
                "current state, pending items, critical values (IPs/ports/keys/paths), "
                "errors resolved. Max 550 tokens.\n\n" + transcript)}])
        snap = resp.content[0].text.strip()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    mem = f"[PROJECT SNAPSHOT: {label}]\n{snap}"
    mid = str(uuid.uuid4()); ts = now_iso()
    with get_db() as conn:
        conn.execute("INSERT INTO memory (id,content,created_at,updated_at) VALUES (?,?,?,?)", (mid, mem, ts, ts))
    log.info("Snapshot: '%s' (%d chars)", label, len(mem))
    return jsonify({"ok": True, "id": mid, "label": label, "preview": snap[:200], "chars": len(mem)})


# ─────────────────────────────────────────────────────────────────────────────
#  LIVING REPOS + LOCAL KNOWLEDGE MODE  (repo_kb.py)
# ─────────────────────────────────────────────────────────────────────────────
repo_kb.init(
    app,
    get_db=get_db, now_iso=now_iso, load_settings=load_settings,
    save_setting=save_setting, export_chat_txt=export_chat_txt,
    extract_text=_extract_text_from_upload, base_dir=BASE_DIR, drop_dir=DROP_DIR,
)


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        LocalLLM — Production AI for Naveen           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Data DB : {BASE_DIR}")
    print(f"║  Dropbox : {DROP_DIR}")
    print(f"║  DB      : {DB_PATH}")
    print("║  Open    : http://localhost:8080                     ║")
    print("║  Network : http://192.168.x.x:8080  (phone access)  ║")
    print("║  Models  : Claude Sonnet · Haiku · Opus 4.6 · 4.7   ║")
    print("║  Local   : Ollama models + Living Repos ($0)        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
