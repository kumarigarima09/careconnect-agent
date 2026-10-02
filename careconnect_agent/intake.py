"""Appointment intake controller.

Drives the conversational collection of required appointment fields, the
pre-submission confirmation step, and utility helpers for extracting
structured data from free-form user messages into the state.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .classifiers import Intent, PatientType, classify as classify_message
from .grounded import answer_from_kb
from .knowledge_base import KnowledgeBase
from .safety import (
    EHR_DISCLAIMER,
    FALLBACK_NO_INFO,
)
from .state import PRETTY_FIELD_LABELS, ConversationState, REQUIRED_APPOINTMENT_FIELDS
from .safety import evaluate_safety

if TYPE_CHECKING:
    from .llm import LLMClientProtocol

logger = logging.getLogger("careconnect_agent.intake")


OPENING_MESSAGE = (
    "Hi! I'm the CareConnect AI Assistant. I can help you learn about our "
    "clinics, services, doctors, or help you request an appointment.\n\n"
    "Are you:\n"
    "1. Visiting CareConnect for the first time?\n"
    "2. Already a CareConnect patient?\n"
    "3. Just looking for information?"
)


# Fields collected in order when the current intent is appointment-related.
# We ask 1-2 at a time.
APPOINTMENT_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("first_name", "May I have your first name?"),
    ("last_name", "And your last name?"),
    ("preferred_location", "Which CareConnect location would you prefer — Delhi, Noida, or Meerut?"),
    ("requested_service", "What type of appointment or specialty are you looking for?"),
    ("preferred_date", "What date would work best for you?"),
    ("preferred_time", "What time of day works best — morning, afternoon, or evening?"),
    ("phone", "What phone number can we reach you on?"),
    ("email", "And your email address, please?"),
)


def opening_message() -> str:
    return OPENING_MESSAGE


# ---------------------------------------------------------------------------
# Free-form data extraction helpers
# ---------------------------------------------------------------------------

_LOCATIONS = ("delhi", "noida", "meerut")

_SPECIALTY_MAP: dict[str, str] = {
    "skin": "Dermatology", "hair": "Dermatology", "acne": "Dermatology",
    "dermatology": "Dermatology", "dermatologist": "Dermatology", "derma": "Dermatology",
    "child": "Pediatrics", "baby": "Pediatrics", "kid": "Pediatrics",
    "pediatric": "Pediatrics", "pediatrics": "Pediatrics", "pediatrician": "Pediatrics",
    "bone": "Orthopedics", "joint": "Orthopedics", "knee": "Orthopedics",
    "ortho": "Orthopedics", "orthopedic": "Orthopedics", "orthopedics": "Orthopedics",
    "women": "Gynecology", "woman": "Gynecology", "pregnancy": "Gynecology",
    "period": "Gynecology", "pcod": "Gynecology", "pcos": "Gynecology",
    "gynea": "Gynecology", "gynecology": "Gynecology", "gynecologist": "Gynecology",
    "tooth": "Dentistry", "teeth": "Dentistry", "dental": "Dentistry", "cavity": "Dentistry",
    "dentistry": "Dentistry", "dentist": "Dentistry",
    "general": "General Medicine", "physician": "General Medicine", "fever": "General Medicine",
    "cold": "General Medicine", "cough": "General Medicine", "diabetes": "General Medicine",
    "bp": "General Medicine", "general medicine": "General Medicine",
    "diagnostic": "Diagnostics", "diagnostics": "Diagnostics", "blood test": "Diagnostics",
    "xray": "Diagnostics", "x-ray": "Diagnostics", "ultrasound": "Diagnostics",
    "ecg": "Diagnostics", "check up": "Preventive Health Checkups",
    "check-up": "Preventive Health Checkups", "checkup": "Preventive Health Checkups",
    "preventive": "Preventive Health Checkups", "health check": "Preventive Health Checkups",
    "full body": "Preventive Health Checkups",
}

_TIME_HINTS: dict[str, str] = {
    "morning": "Morning", "am": "Morning", "before noon": "Morning", "breakfast": "Morning",
    "9am": "Morning", "10am": "Morning", "11am": "Morning", "8am": "Morning",
    "afternoon": "Afternoon", "pm": "Afternoon", "noon": "Afternoon", "lunch": "Afternoon",
    "1pm": "Afternoon", "2pm": "Afternoon", "3pm": "Afternoon", "4pm": "Afternoon",
    "evening": "Evening", "night": "Evening", "dinner": "Evening", "late": "Evening",
    "5pm": "Evening", "6pm": "Evening", "7pm": "Evening",
}


def _extract_email(text: str) -> str | None:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> str | None:
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10:
        return digits
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1:]
    # Also capture +91 XXXXXXXXXX style as-is
    m = re.search(r"(\+?\d[\d\s()-]{7,}\d)", text)
    if m:
        return m.group(1).strip()
    return None


def _extract_location(text: str) -> str | None:
    t = text.lower()
    for loc in _LOCATIONS:
        if loc in t:
            return loc.capitalize()
    return None


def _extract_specialty(text: str) -> str | None:
    t = text.lower()
    for k, v in _SPECIALTY_MAP.items():
        if k in t:
            return v
    return None


def _extract_date(text: str) -> str | None:
    t = text.lower()
    # Specific day+month first: if user gives "Friday 14 November" prefer "14 November"
    m = re.search(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)", t)
    if m:
        return f"{m.group(1)} {m.group(2).title()}"
    m = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{1,2})", t)
    if m:
        return f"{m.group(2)} {m.group(1).title()}"
    # Relative keywords
    if "day after tomorrow" in t:
        return "Day after tomorrow"
    if "tomorrow" in t:
        return "Tomorrow"
    if "today" in t:
        return "Today"
    # Weekday fallback when no month/day given
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for w in weekdays:
        if w in t:
            return w.title()
    return None


def _extract_time(text: str) -> str | None:
    t = text.lower()
    # prefer broader categories first
    for k, v in _TIME_HINTS.items():
        if k in t:
            return v
    return None


_NAME_UNLIKELY_WORDS = (
    # Content verbs, nouns, signals that the sentence is NOT answering "what's your name"
    "need", "want", "book", "appointment", "doctor", "dr", "specialist",
    "dermatologist", "pediatrician", "gynecologist", "orthopedic", "dentist",
    "physician", "clinic", "consultation", "dermatology", "pediatrics",
    "gynecology", "orthopedics", "dentistry", "diagnostic", "medicine",
    "noida", "delhi", "meerut", "location", "available", "have", "offer",
    "morning", "afternoon", "evening", "today", "tomorrow", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "please", "thanks", "thank", "help", "looking", "find", "searching",
    "ask", "question", "information", "about", "visit", "visiting",
    "prefer", "preferred", "date", "time", "address", "price", "fee",
    "cost", "insurance", "email", "phone", "contact",
)


def _extract_name_field(text: str, state: ConversationState) -> str | None:
    cleaned = text
    cleaned = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", cleaned)
    cleaned = re.sub(r"(\+?\d[\d\s()-]{7,}\d)", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)?\b", " ", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = cleaned.replace("@", " ").replace("+", " ")

    lower = cleaned.lower()
    if "?" in cleaned:
        return None

    fillers = {
        "my", "name", "is", "it's", "its", "i", "am", "i'm", "im",
        "you", "can", "call", "me", "hello", "hi", "hey",
        "first", "last", "full", "myself",
        "a", "an", "the", "of", "and", "for", "to", "in", "at", "on", "by",
        "also", "too", "along", "with",
        "phone", "mobile", "contact", "email", "mail", "e-mail", "reach",
        "noida", "delhi", "meerut",
        "morning", "afternoon", "evening", "today", "tomorrow",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "please", "thanks", "thank", "help", "okay", "ok", "yes", "yeah",
        "prefer", "preferred", "date", "time", "address",
        "it", "s", "this", "that", "these", "those", "there", "here",
        "but", "or", "so", "if", "then", "than",
        "as", "be", "been", "being", "was", "were",
        "just", "very", "really",
        "patient", "time", "book", "appointment", "consultation",
        "specialist", "doctor", "dr", "clinic", "pediatrics", "dermatology",
        "gynecology", "orthopedics", "dentistry", "diagnostic", "medicine",
        "cardiology", "ent", "ophthalmology", "general",
        "insurance", "health", "star",
        "need", "want",
    }

    # Explicit "Name is X Y" / "I'm X Y" → extract 1–3 tokens IMMEDIATELY after the intro phrase.
    explicit_match = re.search(r"(my\s+)?name\s+is\s+([A-Za-z][A-Za-z\s\-]{0,60})", lower)
    if explicit_match:
        tail = explicit_match.group(2)
        raw_tokens = re.findall(r"[A-Za-z]+", tail)
        picked: list[str] = []
        for tok in raw_tokens:
            if tok.lower() in fillers:
                if picked:
                    break
                continue
            if len(tok) < 2:
                continue
            picked.append(tok)
            if len(picked) == 3:
                break
        if picked:
            return " ".join(w.capitalize() for w in picked)
    im_match = re.search(r"i'?m\s+([A-Za-z][A-Za-z\s\-]{0,60})", lower)
    if im_match:
        tail = im_match.group(1)
        raw_tokens = re.findall(r"[A-Za-z]+", tail)
        picked = []
        for tok in raw_tokens:
            if tok.lower() in fillers:
                if picked:
                    break
                continue
            if len(tok) < 2:
                continue
            picked.append(tok)
            if len(picked) == 3:
                break
        if picked:
            return " ".join(w.capitalize() for w in picked)

    tokens = re.findall(r"[A-Za-z]+", cleaned)
    if not tokens:
        return None
    remaining = [t for t in tokens if t.lower() not in fillers]
    if not remaining:
        return None
    remaining = [t for t in remaining if len(t) >= 2]
    if len(remaining) < 1 or len(remaining) > 3:
        return None
    heavy_signals = (
        "need", "want", "book", "appointment", "doctor", "dr", "specialist",
        "clinic", "consultation", "patient",
    )
    if any(w in lower for w in heavy_signals):
        return None
    return " ".join(w.capitalize() for w in remaining)


_EMAIL_REFUSAL_PATTERNS = (
    "don't have email", "dont have email", "no email", "i don't use email",
    "dont use email", "prefer not to share email", "won't share email",
    "wont share email", "don't want to share email", "no thanks, email",
    "don't have an email", "dont have an email", "don't have a email",
    "no email address", "i have no email", "no email id", "use phone only",
    "phone only", "only phone", "phone instead",
)


def _looks_like_email_refusal(text: str) -> bool:
    t = text.lower()
    if any(p in t for p in _EMAIL_REFUSAL_PATTERNS):
        return True
    if "email" in t:
        if ("don't" in t or "dont" in t or "do not" in t or "can't" in t or "cant" in t
                or "won't" in t or "wont" in t or "no" in t or "prefer not" in t
                or "rather not" in t or "use my phone" in t or "phone instead" in t
                or "phone only" in t or "only phone" in t):
            return True
    return False


_CONFIRM_POSITIVE = {
    "yes", "yea", "yeah", "yep", "sure", "ok", "okay", "okey", "fine",
    "go ahead", "confirm", "confirmed", "submit", "sounds good", "all good",
    "correct", "that's right", "that is right", "right", "exactly", "do it",
    "please do", "absolutely", "definitely", "yup",
}

_CONFIRM_NEGATIVE = {
    "no", "nope", "not yet", "cancel", "stop", "let me check", "i need to change",
    "change", "edit", "modify", "wait", "hold on",
}


def _looks_like_confirmation_yes(text: str) -> bool:
    t = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    tokens = set(t.split())
    if any(w in t for w in _CONFIRM_POSITIVE):
        return True
    if tokens & set(_CONFIRM_POSITIVE):
        return True
    return False


def _looks_like_confirmation_no(text: str) -> bool:
    t = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    if any(w in t for w in _CONFIRM_NEGATIVE):
        return True
    if "no" in t.split():
        return True
    return False


# ---------------------------------------------------------------------------
# Field extraction dispatcher
# ---------------------------------------------------------------------------

def extract_fields_from_message(text: str, state: ConversationState) -> list[str]:
    """Extract any recognizable structured fields from ``text`` into state.

    Always overwrites the state value when a field is detected in the current
    message.  This lets the user revise previously entered values (e.g.
    *"Actually, Noida instead of Delhi"*).  Only the first extraction of a
    new field is reported in the returned ``filled`` list, but the stored
    value is always refreshed.
    """
    filled: list[str] = []

    def _remember(field: str, value: str | None) -> None:
        if not value:
            return
        new_or_changed = not state.has(field) or state.appointment.get(field) != value
        state.remember(field, value)
        if new_or_changed and field not in filled:
            filled.append(field)

    spec = _extract_specialty(text)
    _remember("requested_service", spec)
    loc = _extract_location(text)
    _remember("preferred_location", loc)

    if state.patient_type in (PatientType.NEW_PATIENT, PatientType.EXISTING_PATIENT):
        if not state.has("patient_type") or state.appointment.get("patient_type") != state.patient_type.value:
            state.remember("patient_type", state.patient_type.value)
            filled.append("patient_type")
    # Menu selection / natural-language patient type
    lowered = text.lower().strip()
    if lowered.startswith("1") or "first time" in lowered:
        state.patient_type = PatientType.NEW_PATIENT
        _remember("patient_type", PatientType.NEW_PATIENT.value)
    elif lowered.startswith("2") or ("already" in lowered and "patient" in lowered) or "already a patient" in lowered:
        state.patient_type = PatientType.EXISTING_PATIENT
        _remember("patient_type", PatientType.EXISTING_PATIENT.value)
    elif lowered.startswith("3") or "just looking" in lowered or "just info" in lowered:
        state.patient_type = PatientType.UNKNOWN

    _remember("preferred_date", _extract_date(text))
    _remember("preferred_time", _extract_time(text))

    phone = _extract_phone(text)
    _remember("phone", phone)
    email = _extract_email(text)
    _remember("email", email)

    # Names heuristics: if the message looks like a name response and we are
    # still missing first_name or last_name, try to split them.
    if not state.has("first_name") or not state.has("last_name"):
        maybe = _extract_name_field(text, state)
        if maybe:
            parts = maybe.split()
            if state.has("first_name") and not state.has("last_name"):
                # First name already captured: treat this message as the last name (1-2 tokens)
                if len(parts) >= 1:
                    state.remember("last_name", " ".join(parts[:2]))
                    filled.append("last_name")
            else:
                if not state.has("first_name") and len(parts) >= 1:
                    state.remember("first_name", parts[0])
                    filled.append("first_name")
                if not state.has("last_name") and len(parts) >= 2:
                    state.remember("last_name", " ".join(parts[1:]))
                    filled.append("last_name")

    # Insurance provider: if message names an insurer or "insurance" + value
    if not state.has("insurance_provider"):
        from careconnect_agent.knowledge_base import INSURER_NAMES
        t = text.lower()
        for insurer in INSURER_NAMES:
            if insurer.lower() in t:
                state.remember("insurance_provider", insurer)
                filled.append("insurance_provider")
                break

    # Appointment intent (reschedule vs cancel vs new appointment)
    if not state.has("appointment_intent"):
        if state.intent == Intent.RESCHEDULE:
            state.remember("appointment_intent", "Reschedule")
            filled.append("appointment_intent")
        elif state.intent == Intent.CANCELLATION:
            state.remember("appointment_intent", "Cancellation")
            filled.append("appointment_intent")
        elif state.intent == Intent.APPOINTMENT_REQUEST:
            label = "New" if state.patient_type == PatientType.NEW_PATIENT else (
                "Follow-up" if state.patient_type == PatientType.EXISTING_PATIENT else "Request"
            )
            state.remember("appointment_intent", label)
            filled.append("appointment_intent")

    return filled


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class IntakeController:
    def __init__(self, kb: KnowledgeBase, llm: "LLMClientProtocol | None" = None) -> None:
        self.kb = kb
        self.llm = llm

    # -- top-level dispatcher per turn ------------------------------------
    def handle(self, text: str, state: ConversationState) -> str:
        # 1) Data extraction first - always try to grab fields
        extract_fields_from_message(text, state)

        # 2) Classify + update state intent/patient type
        cls = classify_message(text, prior_intent=state.intent, prior_patient=state.patient_type)
        state.intent = cls.intent
        if cls.patient_type != PatientType.UNKNOWN:
            state.patient_type = cls.patient_type
        # If patient_type became known, copy to form
        if state.patient_type in (PatientType.NEW_PATIENT, PatientType.EXISTING_PATIENT):
            if not state.has("patient_type"):
                state.remember("patient_type", state.patient_type.value)

        # 3) Safety short-circuits have been handled upstream; but EHR disclaimers
        # can still be needed here if the user asked for records.
        safety = evaluate_safety(text)
        if safety.message == EHR_DISCLAIMER:
            if state.intent in (Intent.APPOINTMENT_REQUEST, Intent.RESCHEDULE, Intent.EXISTING_PATIENT_SUPPORT):
                follow_up = self._next_questions(state, limit=1)
                return f"{safety.message}\n\n{follow_up}" if follow_up else safety.message
            return safety.message

        # 4) Appointment confirmation step?
        if state.awaiting_confirmation:
            return self._handle_confirmation_reply(text, state)

        # 5) Email refusal handling
        if state.awaiting_email:
            if _looks_like_email_refusal(text):
                state.awaiting_email = False
                # Placeholder "opt-out" value so required list can be skipped safely
                state.remember("email", "email not provided (user opted out)")
                return (
                    "No problem. We'll use your phone number as the primary contact. "
                    + self._maybe_summarize_or_next(state)
                )
            email = _extract_email(text)
            if email:
                state.remember("email", email)
                state.awaiting_email = False
                return "Thank you. " + self._maybe_summarize_or_next(state)

        # 5b) Detect whether the user message is a KB info-seeking question (even if
        # we are otherwise collecting appointment fields). When it is, answer it
        # FIRST (FR-5), then offer the next intake question(s). Topic resets like
        # "Actually, never mind. What are your contact details?" also trigger
        # this path so the new question is answered.
        inline_kb = _looks_like_kb_inline_question(text)
        topic_reset = _looks_like_topic_reset(text)
        pure_info_intent = state.intent in (
            Intent.GENERAL_INFORMATION, Intent.LOCATION,
            Intent.DOCTOR_INFORMATION, Intent.SERVICE_INFORMATION,
            Intent.PRICING, Intent.INSURANCE, Intent.OTHER,
        )
        collecting = self._collecting_appointment_fields(state)
        appointment_intent = self._is_appointment_intent(state.intent)
        should_answer_kb_first = inline_kb or topic_reset or (
            pure_info_intent and not appointment_intent and not collecting
        )
        if should_answer_kb_first:
            try:
                kb_ans = answer_from_kb(text, self.kb, self.llm)
            except Exception:  # noqa: BLE001
                kb_ans = FALLBACK_NO_INFO
            # If not collecting and not in an appointment intent, we are pure
            # Q&A mode — return KB answer + next offer only.
            if not collecting and not appointment_intent:
                return self._wrap_kb_answer_with_next(state, kb_ans, text)
            # Otherwise we were already collecting; answer the KB question first
            # (per FR-5) then proceed with the next intake question(s) or summary.
            missing = state.missing_required_fields()
            if not missing:
                next_part = self._summarize_and_ask_confirmation(state)
            else:
                next_part = self._next_questions_from_missing(state, missing, limit=2)
            if next_part:
                return f"{kb_ans}\n\n{next_part}"
            return kb_ans

        # 6) Drive intake when the intent is appointment-related, or when the
        #    user already started giving us structured data (≥1 required field
        #    filled).  We explicitly do NOT drive intake merely because the
        #    required fields list says items are still missing — that would
        #    force info-only queries into an unwanted form-filling conversation.
        if appointment_intent or collecting:
            return self._handle_appointment_intake(text, state)

        # 7) Default: KB answer + next helpful action.
        try:
            ans = answer_from_kb(text, self.kb, self.llm)
        except Exception:  # noqa: BLE001
            ans = FALLBACK_NO_INFO
        return self._wrap_kb_answer_with_next(state, ans, text)

    def _wrap_kb_answer_with_next(self, state, ans, text):
        if state.intent in (Intent.SERVICE_INFORMATION, Intent.DOCTOR_INFORMATION,
                            Intent.PRICING, Intent.LOCATION):
            extra = (
                "\n\nWould you like to know more about the available doctors, "
                "the next appointment slots, or request an appointment?"
            )
        elif state.intent == Intent.INSURANCE:
            extra = (
                "\n\nIf you'd like I can note your insurance provider and start "
                "an appointment request."
            )
        elif state.patient_type == PatientType.UNKNOWN and "appointment request" not in ans.lower():
            extra = (
                "\n\nMay I also help with any of the following? "
                "Request an appointment, speak with the clinic team, or just "
                "answer more questions."
            )
        else:
            extra = "\n\nIs there anything else I can help with?"
        return ans + extra

    # -- helpers ----------------------------------------------------------
    def _is_appointment_intent(self, intent: Intent) -> bool:
        return intent in (
            Intent.APPOINTMENT_REQUEST,
            Intent.RESCHEDULE,
            Intent.CANCELLATION,
            Intent.EXISTING_PATIENT_SUPPORT,
        )

    def _collecting_appointment_fields(self, state: ConversationState) -> bool:
        """Return True when at least one required appointment field is filled
        (even if not by an explicit appointment intent), meaning we should
        stay in the intake flow instead of dropping back to pure KB Q&A."""
        required_and_set = state.collected_fields & set(REQUIRED_APPOINTMENT_FIELDS)
        return len(required_and_set) >= 1

    def _handle_appointment_intake(self, text: str, state: ConversationState) -> str:
        # If the user's message itself answers a KB question AND drives intake,
        # answer it first. For simple service/location questions we already
        # extracted the fields; try a KB answer if the sentence starts with "Do you have..."
        kb_answer: str | None = None
        if _looks_like_kb_inline_question(text):
            try:
                kb_answer = answer_from_kb(text, self.kb, self.llm)
            except Exception:  # noqa: BLE001
                kb_answer = None

        missing = state.missing_required_fields()
        # Summarize and confirm when all required fields are in.
        if not missing:
            return self._summarize_and_ask_confirmation(state, prefix=kb_answer)
        # Otherwise ask the next 1-2 missing questions
        questions = self._next_questions_from_missing(state, missing, limit=2)
        if not questions:
            questions = self._next_questions_from_missing(state, missing, limit=1)
        if kb_answer:
            return f"{kb_answer}\n\n{questions}"
        return questions

    def _next_questions_from_missing(self,
                                     state: ConversationState,
                                     missing: list[str],
                                     limit: int) -> str:
        # Build ordered subset using APPOINTMENT_QUESTIONS order
        ordered: list[tuple[str, str]] = []
        for field, q in APPOINTMENT_QUESTIONS:
            if field in missing:
                ordered.append((field, q))
        # "patient_type" is not in APPOINTMENT_QUESTIONS but may be missing
        if "patient_type" in missing and not ordered:
            ordered.append(("patient_type",
                            "Before I proceed — are you a new patient or already registered with us?"))
        selected = ordered[:limit]
        # Special handling for email refusal: only ask email alone, flag state
        questions_text = " ".join(q for _, q in selected)
        # If email is among selected and >1 questions, ask it separately? Actually fine.
        # But if the missing fields are only [first_name, last_name], combine them nicely
        names = {f for f, _ in selected} & {"first_name", "last_name"}
        if names == {"first_name", "last_name"}:
            questions_text = "May I have your first and last name, please?"
        if "email" in {f for f, _ in selected}:
            state.awaiting_email = True
        return questions_text

    def _next_questions(self, state: ConversationState, limit: int) -> str:
        missing = state.missing_required_fields()
        if not missing:
            return ""
        return self._next_questions_from_missing(state, missing, limit=limit)

    def _summarize_and_ask_confirmation(self, state: ConversationState,
                                        prefix: str | None = None) -> str:
        lines = state.pretty_summary_lines()
        block = "Let me confirm the details:\n\n" + "\n".join(f"- {l}" for l in lines)
        question = "\n\nWould you like me to submit this appointment request?"
        state.awaiting_confirmation = True
        if prefix:
            return prefix + "\n\n" + block + question
        return block + question

    def _maybe_summarize_or_next(self, state: ConversationState) -> str:
        missing = state.missing_required_fields()
        if not missing:
            return self._summarize_and_ask_confirmation(state)
        next_q = self._next_questions_from_missing(state, missing, limit=1)
        return next_q

    def _handle_confirmation_reply(self, text: str, state: ConversationState) -> str:
        if _looks_like_confirmation_yes(text):
            state.awaiting_confirmation = False
            state.confirmed = True
            # Orchestrator will see state.confirmed and run CRM submission,
            # then send the submitted message. Return an empty-ish string so
            # the orchestrator can send the post-submit message instead.
            return "__SUBMIT__"
        if _looks_like_confirmation_no(text):
            state.awaiting_confirmation = False
            # Offer to edit fields.
            return (
                "No problem. Which details would you like to change — name, phone, "
                "email, location, service, date, or time?"
            )
        # If it's not clearly yes/no, try to treat it as a field edit
        changed = extract_fields_from_message(text, state)
        if changed:
            # Re-summarize
            return self._summarize_and_ask_confirmation(state, prefix=(
                f"Got it. I've updated: {', '.join(PRETTY_FIELD_LABELS.get(f,f) for f in changed)}."
            ))
        return (
            "Sorry, I didn't catch that. Would you like me to submit this appointment "
            "request? Please say yes (submit) or no (edit details)."
        )


def _looks_like_kb_inline_question(text: str) -> bool:
    t = text.lower().strip()
    starts_info = (
        t.startswith("do you") or t.startswith("does ") or t.startswith("is there") or t.startswith("what")
        or t.startswith("how much") or t.startswith("where") or t.startswith("when")
        or t.startswith("are you") or t.startswith("tell me about")
        or t.startswith("which") or t.startswith("who") or t.startswith("how to")
        or t.startswith("how do") or t.startswith("how can") or t.startswith("how is")
        or t.startswith("contact") or t.startswith("your contact")
    )
    if starts_info and ("?" in t or "tell me" in t or "about" in t or "which" in t or "details" in t or "information" in t):
        return True
    # Explicit topic reset / abandon + new question
    if ("actually" in t or "never mind" in t or "nevermind" in t or "instead" in t
            or "forget that" in t or "change of topic" in t) and ("?" in t or "what" in t or "your" in t or "contact" in t or "tell" in t):
        return True
    # Which cities / contact details / phone number info queries, no "?" needed
    if ("which cities" in t or "cities do you" in t or "what cities" in t
            or "all locations" in t or "which locations" in t or "what locations" in t
            or "contact details" in t or "contact information" in t or "your phone" in t
            or "customer support" in t or "how to reach" in t or "how do i contact" in t):
        return True
    return False


def _looks_like_topic_reset(new_message: str) -> bool:
    t = new_message.lower().strip()
    return (
        "actually" in t and ("never mind" in t or "instead" in t or "forget" in t or "not" in t)
    ) or "never mind" in t or "nevermind" in t or "forget that" in t or "change of topic" in t
