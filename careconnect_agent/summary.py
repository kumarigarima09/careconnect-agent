"""CRM-friendly conversation summary generator.

Produces the format mandated by FR-22. Tries the LLM with a strict prompt for
the SUMMARY paragraph; if the LLM is unavailable, falls back to a deterministic
template so the system is still fully operational offline.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .scoring import ScoringResult
from .state import ConversationState

if TYPE_CHECKING:
    from .llm import LLMClientProtocol

logger = logging.getLogger("careconnect_agent.summary")


SECTION_HEADERS = (
    "PATIENT TYPE",
    "INTENT",
    "SERVICE",
    "LOCATION",
    "PREFERRED DATE",
    "PREFERRED TIME",
    "LEAD TEMPERATURE",
    "SUMMARY",
)


def _value(state: ConversationState, key: str) -> str:
    v = state.appointment.get(key) or ""
    return v.strip() or "(Not specified)"


def _sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([p for p in parts if p.strip()])


def _deterministic_summary(state: ConversationState, result: ScoringResult) -> str:
    patient_type = (state.appointment.get("patient_type") or state.patient_type.value).replace("_", " ").title()
    service = state.appointment.get("requested_service") or "a requested service"
    location = state.appointment.get("preferred_location") or "a CareConnect location"
    date = state.appointment.get("preferred_date") or "their preferred date"
    time = state.appointment.get("preferred_time") or "their preferred time"
    s1 = (
        f"{patient_type} is requesting {service} at the CareConnect {location} clinic."
    )
    s2 = (
        f"They prefer {date} during the {time}, and have provided contact details "
        f"for follow-up."
    )
    temp = result.temperature.value
    action = (
        "The clinic team should call or email the patient within 4 working hours to "
        "confirm the appointment slot and answer any questions."
        if temp == "HOT"
        else (
            "The clinic team should follow up within 1 working day to confirm the "
            "appointment and share any prep instructions relevant to the service."
            if temp == "WARM"
            else "The clinic team may add this patient to the nurture list for information "
                 "follow-up and reach out if they have not responded within 2-3 days."
        )
    )
    return f"{s1} {s2} {action}"


def _llm_summary(state: ConversationState, result: ScoringResult, llm: "LLMClientProtocol") -> str | None:
    data = state.appointment
    patient_type = data.get("patient_type") or state.patient_type.value
    service = data.get("requested_service") or "unspecified"
    location = data.get("preferred_location") or "unspecified"
    date = data.get("preferred_date") or "unspecified"
    time = data.get("preferred_time") or "unspecified"
    lead_temp = result.temperature.value
    prompt = (
        "You write concise CRM-friendly summaries for a clinic. Given structured fields, "
        "produce exactly 2 to 4 complete sentences. Do not use bullet points. The "
        "summary must cover: who the patient is, what they need, where and when they want "
        "it, and what the clinic team should do next. Do not invent details.\n\n"
        f"PATIENT TYPE: {patient_type}\n"
        f"INTENT: {state.intent.value}\n"
        f"REQUESTED SERVICE: {service}\n"
        f"LOCATION: {location}\n"
        f"DATE: {date}\n"
        f"TIME: {time}\n"
        f"LEAD TEMPERATURE: {lead_temp}\n"
        f"RECENT CONVERSATION SNIPPET:\n{state.transcript_text()[-600:]}\n\n"
        "SUMMARY (2-4 sentences only):"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return llm.chat(messages, response_format="text", temperature=0.3).strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM summary failed, using deterministic fallback: %s", e)
        return None


def generate_crm_summary(state: ConversationState,
                         result: ScoringResult,
                         llm: "LLMClientProtocol | None" = None) -> str:
    data = state.appointment
    patient_type = data.get("patient_type") or state.patient_type.value
    intent = state.intent.value

    summary_text: str
    if llm is not None:
        llm_attempt = _llm_summary(state, result, llm)
        if llm_attempt and 2 <= _sentence_count(llm_attempt) <= 5:
            summary_text = llm_attempt
        else:
            summary_text = _deterministic_summary(state, result)
    else:
        summary_text = _deterministic_summary(state, result)

    lines = [
        f"PATIENT TYPE: {patient_type}",
        f"INTENT: {intent}",
        f"SERVICE: {data.get('requested_service') or '(Not specified)'}",
        f"LOCATION: {data.get('preferred_location') or '(Not specified)'}",
        f"PREFERRED DATE: {data.get('preferred_date') or '(Not specified)'}",
        f"PREFERRED TIME: {data.get('preferred_time') or '(Not specified)'}",
        f"LEAD TEMPERATURE: {result.temperature.value} (score {result.score})",
        "",
        "SUMMARY:",
        summary_text,
    ]
    return "\n".join(lines)
