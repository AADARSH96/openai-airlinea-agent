"""
Airline Support Multi-Agent Demo (Sessions + Guardrails + Handoffs)

This script showcases a minimal but realistic airline customer-support flow built on
an Agents SDK. It demonstrates:

- A NEW-CHAT flow that still carries over context from the user’s PREVIOUS chat thread
  via a compact summary injected as a system message.
- A tiny mapping database that relates a user_id to conversation session_ids.
- A triage agent that delegates to specialized agents (FAQ, Seat Booking).
- Tooling for FAQs and seat updates.
- Input and output guardrails:
  * Input guardrail blocks obviously off-topic queries at the triage boundary.
  * Output guardrail validates claims made by the Seat Booking agent.
- Handoff hooks that can inject context (e.g., a flight number) when transferring.

Run this file directly to start an interactive loop:

    python airline_agents_demo.py

Type airline-related questions or requests (e.g., "What's the baggage policy?"
or "Change my seat to 7C, my confirmation is ABC123").
"""

from __future__ import annotations as _annotations

import asyncio
import nest_asyncio
nest_asyncio.apply()
import random
import re
import uuid
import sqlite3
from pathlib import Path
from typing import List, Union
from datetime import datetime, timezone

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    SQLiteSession,          # ✅ Sessions API
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    input_guardrail,        # ✅ Guardrails API
    output_guardrail,       # ✅ Guardrails API
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

# =========================
# MINIMAL USER → CONVERSATION MAPPING
# =========================

# Paths for two SQLite databases:
# 1) app.db stores a *very small* user_id ↔ session_id mapping table for "Recent chats".
# 2) conversations.db is the Agents SDK session store that persists the actual message history.
APP_DB_PATH = Path("data/app.db")            # tiny mapping DB (user ↔ session_id)
CONV_DB_PATH = Path("data/conversations.db") # Agents SDK session store (messages)

def _init_mapping_db() -> None:
    """
    Initialize the tiny mapping database that associates a `user_id` with
    conversation `session_id`s. This powers a simple "recent chats" feature.

    Table: user_conversations
        - session_id (PRIMARY KEY): unique ID per conversation thread
        - user_id: who owns the thread
        - title: optional title for UI lists
        - created_at / updated_at: timestamps for sorting recent threads
    """
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_conversations (
              session_id TEXT PRIMARY KEY,
              user_id    TEXT NOT NULL,
              title      TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_conversations_user
            ON user_conversations (user_id, updated_at DESC)
        """)

def _create_session_for_user(user_id: str, title: str | None = None) -> str:
    """
    Create a new conversation row for the given user and return its session_id.

    Args:
        user_id: Logical user key supplied by your app.
        title: Optional display title for UI.

    Returns:
        The generated `session_id` (hex UUID string).
    """
    sid = uuid.uuid4().hex
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_conversations (session_id, user_id, title) VALUES (?, ?, ?)",
            (sid, user_id, title or "Conversation"),
        )
    return sid

def _latest_session_for_user(user_id: str) -> str | None:
    """
    Fetch the most recently updated session_id for a user, or None if none exists.

    This is used to locate the *previous* chat thread so we can summarize and
    carry over context into a brand-new chat.

    Args:
        user_id: Logical user key supplied by your app.

    Returns:
        The latest `session_id` for that user, or None.
    """
    with sqlite3.connect(APP_DB_PATH) as conn:
        row = conn.execute(
            "SELECT session_id FROM user_conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return row[0] if row else None

def _touch_session(session_id: str) -> None:
    """
    Update the `updated_at` timestamp for a session to keep it at the top of a
    “recent chats” list.

    Args:
        session_id: The conversation thread to bump.
    """
    with sqlite3.connect(APP_DB_PATH) as conn:
        conn.execute(
            "UPDATE user_conversations SET updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
            (session_id,),
        )

# Create the mapping DB (no-op if it already exists).
_init_mapping_db()

# =========================
# CONTEXT
# =========================

class AirlineAgentContext(BaseModel):
    """
    Shared, structured context available to all agents and tools in this flow.

    Fields:
        passenger_name: Optional name of the traveler.
        confirmation_number: The PNR/confirmation code for a booking.
        seat_number: The seat we most recently updated or referenced.
        flight_number: An example attribute that can be injected by a handoff.
    """
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


# =========================
# TOOLS
# =========================

@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    """
    Very simple FAQ lookup mock.

    This tool inspects a natural-language question and returns a canned airline
    policy answer for a handful of topics (baggage, seating, wifi). Anything
    else yields an “unknown” response.

    Args:
        question: The user’s last question.

    Returns:
        A short policy answer string.
    """
    question_lower = question.lower()
    if any(
        keyword in question_lower
        for keyword in ["bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"]
    ):
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif any(keyword in question_lower for keyword in ["seat", "seats", "seating", "plane"]):
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom."
        )
    elif any(
        keyword in question_lower
        for keyword in ["wifi", "internet", "wireless", "connectivity", "network", "online"]
    ):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    NOTE: This is a demo only—no real API calls are made. We perform:
      1) A minimal confirmation_number sanity check.
      2) An update of the *shared context* so the output guardrail can verify
         any later claims made by the Seat Booking agent.

    Args:
        context: SDK wrapper that exposes the AirlineAgentContext.
        confirmation_number: Alphanumeric booking reference (3–8 chars).
        new_seat: Desired seat label (e.g., "12A").

    Returns:
        A human-readable status message about the update.
    """
    # Minimal sanity check (demo)
    if not re.fullmatch(r"[A-Z0-9]{3,8}", confirmation_number):
        return "Invalid confirmation number format. Please provide a valid alphanumeric code."

    # Update context so the output guardrail can verify claims later
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat

    # Handoff hook should set this (assert for developer clarity)
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


# =========================
# HANDOFF HOOK
# =========================

async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """
    Handoff side-effect that seeds a fake flight number.

    This simulates a real system where—upon routing to a specific workflow—
    we might populate required fields (e.g., retrieving the active flight).
    """
    # Seed a fake flight number on handoff
    context.context.flight_number = f"FLT-{random.randint(100, 999)}"


# =========================
# GUARDRAILS
# =========================

# --- Input guardrail: block clearly off-topic requests before any agent runs ---
OFF_TOPIC_KEYWORDS = (
    "poem", "lyrics", "recipe", "algebra", "calculus", "java", "frontend",
    "build website", "write code for", "physics", "chemistry"
)

@input_guardrail
async def airline_off_topic_guardrail(
    ctx: RunContextWrapper[AirlineAgentContext],
    agent: Agent,
    input: Union[str, List[TResponseInputItem]],
) -> GuardrailFunctionOutput:
    """
    Lightweight off-topic filter for the *entry* (triage) agent.

    If the incoming user text clearly references unrelated domains (poetry,
    schoolwork, programming, etc.), we trip and prevent the agent graph from
    running.

    Args:
        ctx: Context wrapper (unused here, but part of the signature).
        agent: The agent the guardrail is attached to (triage).
        input: Raw user string or list of multi-part items.

    Returns:
        GuardrailFunctionOutput with tripwire_triggered=True when off-topic.
    """
    text = ""
    if isinstance(input, str):
        text = input
    else:
        # input is a list of items; collect user-side text
        parts = []
        for it in input:
            if isinstance(it, dict) and it.get("role") == "user":
                parts.append(str(it.get("content", "")))
        text = " ".join(parts)

    is_off_topic = any(k in text.lower() for k in OFF_TOPIC_KEYWORDS)
    return GuardrailFunctionOutput(
        output_info={"reason": "off_topic" if is_off_topic else "ok"},
        tripwire_triggered=is_off_topic,
    )


# --- Output guardrail: ensure seat-update claims match context state ---
class SeatBookingOutput(BaseModel):
    """
    Output schema for the Seat Booking agent.

    The agent is instructed to return a JSON with a single field `response`.
    We use this class so the output guardrail can access structured data.
    """
    response: str  # final assistant message for this turn

@output_guardrail
async def seat_update_integrity_guardrail(
    ctx: RunContextWrapper[AirlineAgentContext],
    agent: Agent,
    output: SeatBookingOutput,
) -> GuardrailFunctionOutput:
    """
    Validate that any claim like “updated seat” is backed by context state.

    If the Seat Booking agent asserts it changed a seat, then BOTH
    `ctx.context.confirmation_number` and `ctx.context.seat_number` must be set
    (by the `update_seat` tool). If not, we trip.

    Args:
        ctx: Context wrapper (holds AirlineAgentContext).
        agent: The agent the guardrail is attached to (Seat Booking).
        output: Structured output containing a final `response` string.

    Returns:
        GuardrailFunctionOutput with details and `tripwire_triggered` flag.
    """
    text = (output.response or "").lower()
    claims_update = any(
        phrase in text for phrase in ["updated seat", "seat has been updated", "changed your seat"]
    )
    if not claims_update:
        return GuardrailFunctionOutput(output_info={"reason": "no_claim"}, tripwire_triggered=False)

    ctx_ok = bool(ctx.context.confirmation_number) and bool(ctx.context.seat_number)
    return GuardrailFunctionOutput(
        output_info={
            "reason": "claimed_update_without_context" if not ctx_ok else "ok",
            "ctx_confirmation_number": ctx.context.confirmation_number,
            "ctx_seat_number": ctx.context.seat_number,
        },
        tripwire_triggered=not ctx_ok,
    )


# =========================
# AGENTS
# =========================

# FAQ agent: calls a lookup tool; delegates back to triage if it cannot answer.
faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

# Seat booking agent: collects confirmation + desired seat, calls update tool,
# and is protected by an output guardrail that checks integrity of its claims.
seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent.
    When you finish, return JSON with a single field 'response' containing your final reply.""",
    tools=[update_seat],
    # Apply output guardrail only when THIS agent is the finisher for the turn
    output_guardrails=[seat_update_integrity_guardrail],
    output_type=SeatBookingOutput,
)

# Triage agent: entry point that can route to FAQ or Seat Booking agents.
# The input guardrail is attached here to prevent off-topic runs early.
triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
    # Run input guardrails on the entry agent only
    input_guardrails=[airline_off_topic_guardrail],
)

# Allow agents to transfer back to triage if needed.
faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


# =========================
# NEW CHAT WITH CARRYOVER CONTEXT (MINIMAL FLOW)
# =========================

# Summarizer agent: compresses the tail of the previous conversation into <=150 words.
summarizer_agent = Agent[AirlineAgentContext](
    name="Summarizer",
    instructions=(
        "You are a concise assistant. Summarize the prior conversation below into <=150 words "
        "as bullet points, focusing on stable user facts (e.g., name, preferences), recent actions "
        "(e.g., seat updates), and any open tasks. Output only the summary text."
    )
)

async def seed_new_session_with_previous_summary(
    user_id: str, new_session: SQLiteSession, prev_session_id: str | None, carry_limit: int = 40
) -> None:
    """
    Inject a brief summary of the prior chat into a *new* session as a system item.

    This is the core of the “NEW chat, but retain useful memory” flow:
    - Find the user's previous session (if any).
    - Pull its last `carry_limit` items (SDK returns in chronological order).
    - Extract just user/assistant text.
    - Ask the `summarizer_agent` to produce a compact bullet summary.
    - Prepend that summary in the new session as a `system` item so all
      subsequent turns in this new chat have that context “for free”.

    Args:
        user_id: Logical user key for lookup (unused here; we get prev_session_id directly).
        new_session: The freshly created session that will receive the summary.
        prev_session_id: The previous session to summarize (or None if no history).
        carry_limit: How many recent items to consider from the prior chat.
    """
    if not prev_session_id:
        return

    prev_sess = SQLiteSession(prev_session_id, str(CONV_DB_PATH))
    # Latest N items in chronological order (SDK contract)  # see docs on get_items/add_items
    prev_items = await prev_sess.get_items(limit=carry_limit)

    if not prev_items:
        return

    # Build a compact text payload for summarization
    lines: List[str] = []
    for it in prev_items:
        role = it.get("role")
        content = it.get("content")
        # content may be a string or a list of parts; keep it simple
        if isinstance(content, list):
            content_str = " ".join([str(p) for p in content])
        else:
            content_str = str(content)
        if role in ("user", "assistant"):
            lines.append(f"{role.upper()}: {content_str}")
    convo_text = "\n".join(lines[-(carry_limit*2):])

    if not convo_text.strip():
        return

    # Summarize without a session (stateless)
    prompt = (
        "Summarize this prior chat:\n\n"
        f"{convo_text}\n\n"
        "Return only the bullet summary, <=150 words."
    )
    sum_result = await Runner.run(summarizer_agent, prompt, context=AirlineAgentContext())
    summary = (sum_result.final_output or "").strip()
    if not summary:
        return

    # Inject the summary into the new session as a system item,
    # so it is auto-prepended on every turn of this new chat.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    system_note = {
        "role": "system",
        "content": f"Carryover context (summarized from previous chat at {now}):\n{summary}"
    }
    await new_session.add_items([system_note])


# =========================
# RUN (Sessions + Guardrails + Tracing + AUTO user mapping)
# =========================

async def main():
    """
    Entry point for running the interactive demo.

    Flow:
        1) Create a NEW chat session for a fixed demo user_id.
        2) Look up the user's most recent *previous* session (if any).
        3) Summarize that previous chat and inject it into the NEW chat as a system note.
        4) Start an interactive loop:
            - Route input to the triage agent (with input guardrail).
            - Handle tool calls, handoffs, and output guardrails.
            - Print turn events for visibility.
            - Keep the session “touched” so it remains the most recent.
    """
    current_agent: Agent[AirlineAgentContext] = triage_agent
    context = AirlineAgentContext()

    # In a real app, your platform supplies this. Here we keep it simple.
    user_id = "user-1"

    # --- Treat this run as a NEW chat ---
    prev_session_id = _latest_session_for_user(user_id)    # may be None on first-ever chat
    session_id = _create_session_for_user(user_id, "Conversation")
    session = SQLiteSession(session_id, str(CONV_DB_PATH))  # fresh thread

    # Seed this new chat with a compact summary of the previous chat (if any)
    await seed_new_session_with_previous_summary(user_id, session, prev_session_id, carry_limit=40)

    print(f"Ready. (user_id={user_id}, NEW session_id={session_id})  Ask your question:")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue

        # Group traces by conversation to make inspection easier.
        with trace("Customer service", group_id=session_id):  # group traces by conversation
            try:
                # With Sessions, pass a plain string (or list of items) — not a bare dict
                result = await Runner.run(
                    current_agent,
                    user_input,      # pass a string; sessions prepend history & store new items
                    context=context,
                    session=session,
                )
            except InputGuardrailTripwireTriggered:
                # Triggered by airline_off_topic_guardrail
                print("System: Sorry, I can only help with airline-related questions.")
                continue
            except OutputGuardrailTripwireTriggered:
                # Triggered by seat_update_integrity_guardrail
                print("System: I couldn’t verify that seat update in our system. Let’s try again.")
                continue

            # Print events from this turn for developer/operator visibility
            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(f"Handoff: {new_item.source_agent.name} → {new_item.target_agent.name}")
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")

            # Persist which agent finished so we continue from the right place next turn
            current_agent = result.last_agent

            # Keep the “recent chats” ordering fresh
            _touch_session(session_id)

if __name__ == "__main__":
    # asyncio.run is sufficient here since we don’t nest event loops in this file.
    # (nest_asyncio.apply() at the top enables reentrancy if you ever embed this.)
    asyncio.run(main())
