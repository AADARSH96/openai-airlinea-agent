# Airline Multi‑Agent Assistant (OpenAI Agents SDK)

A tiny, production‑style example that shows how to build a **multi‑agent airline assistant** using the **OpenAI Agents SDK** with:

- **Sessions (conversation memory)** — the SDK automatically loads prior items before each run and saves new ones after; you pass only the latest user message.  
- **Multi‑agent handoffs** — a triage agent delegates to an FAQ agent or a Seat‑Booking agent.  
- **Tools** — Python functions exposed via `@function_tool`.  
- **Guardrails** — input guardrails (off‑topic) and output guardrails (seat‑update integrity).  
- **New chat with carryover context** — when a user starts a *new* chat, we create a fresh `session_id` and inject a short summary of the last chat so the agent starts fresh but remembers key facts.

References: Sessions, Guardrails, Handoffs, Runner input, and Tools are documented in the OpenAI Agents SDK docs.


## What you get

- **`app/main.py`** — Single-file runnable example with:
  - User→conversation mapping in a tiny SQLite table (`app.db`).
  - Persistent session storage for message history (`conversations.db`).
  - “New chat” flow that **creates a brand‑new session** and **seeds** it with a compact summary from the previous chat (if any).
  - Minimal guardrails and a sample tool (`faq_lookup_tool`) plus a seat update tool.
- **Works from the terminal** — run it and just type questions.


## Project structure (simple)

```
airline-agents/
├─ app/
│  └─ main.py                # the code you already have (single file)
├─ data/                     # databases will be created next to the repo root, but you can move them here
│  └─ .gitkeep
├─ requirements.txt          # pinned deps for reproducible installs
├─ .env.example              # template for env vars (OPENAI_API_KEY)
└─ README.md
```

> Prefer a more modular layout? Split `main.py` into `agents.py`, `tools.py`, `guardrails.py`, `sessions.py`, and `summarizer.py`. The logic doesn’t change; it’s just organization.


## Prerequisites

- **Python 3.10+**
- **OpenAI API key** — set `OPENAI_API_KEY` in your environment.


## Setup

```bash
git clone <your-repo-url> airline-agents
cd airline-agents

# (recommended) create a virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# set your API key (macOS/Linux)
export OPENAI_API_KEY="sk-..."
# Windows (PowerShell)
# setx OPENAI_API_KEY "sk-..."
```


## Run

```bash
python app/main.py
```

You’ll see something like:
```
Ready. (user_id=user-1, NEW session_id=...)
> 
```

Now just ask questions like:
- `Can I bring a carry-on bag?`
- `Change my seat to 12C. My confirmation is ABC123.`


## How “New chat + carryover context” works

- On app start, we treat it as **New chat**: we create a **fresh `session_id`** and `SQLiteSession(session_id, "conversations.db")`.
- We look up the user’s **previous session** (if any), fetch the last N items with `get_items(limit=N)`, summarize them using a tiny “Summarizer” agent, and **inject** that summary into the **new session** via `add_items([{"role":"system", "content": ...}])`.
- From then on, you just pass the user’s **string** input to `Runner.run(...)` and the SDK auto‑prepends the new session’s items on every turn.

This is the intended pattern: a **session = one chat thread**. Create a new session on “New chat”; reuse the same session across turns; seed a new session if you want to carry context forward.


## Config knobs you can tweak

- **Carryover limit**: how many items to pull from the last chat before summarizing (default in the code: `carry_limit=40`).
- **DB locations**: change `APP_DB_PATH` and `CONV_DB_PATH` if you want to store SQLite files under `data/`.
- **User ID**: currently hardcoded to `"user-1"` for simplicity; in an app, you’d pass a real user id from your platform.
- **Start mode**: if you want to *continue* the latest chat instead of starting new, replace the “NEW chat” block with a lookup of `_latest_session_for_user(...)` and reuse it.


## Requirements

Put this in `requirements.txt`:

```
openai-agents>=0.3.0
pydantic>=2.6.0
nest_asyncio>=1.6.0
```

> The code uses:
> - **Sessions** to persist/restore message history automatically (no manual `.to_input_list()`).
> - **Guardrails** to validate input/output and raise tripwires you can handle in app code.
> - **Handoffs** so the triage agent can delegate to specialist agents.


