"""Healthcare safety guardrails and canned disclaimers.

All disclaimers below are fixed strings chosen at runtime by the pure rule
functions. The LLM never authoritatively answers diagnosis/medication
questions — instead these disclaimers are returned directly.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class SafetyAction(str, enum.Enum):
    NONE = "NONE"
    EMERGENCY_MSG = "EMERGENCY_MSG"
    NO_DIAGNOSIS_MSG = "NO_DIAGNOSIS_MSG"
    NO_MEDICATION_MSG = "NO_MEDICATION_MSG"
    HUMAN_HANDOFF_MSG = "HUMAN_HANDOFF_MSG"
    NO_EHR_MSG = "NO_EHR_MSG"


# ---------------------------------------------------------------------------
# Fixed safety + support response strings
# ---------------------------------------------------------------------------

EMERGENCY_RESPONSE = (
    "Your symptoms may require urgent medical attention. Please contact your local "
    "emergency service (108 / 112 in India) or go to the nearest emergency department "
    "immediately. I can provide CareConnect's emergency-care information if needed."
)

DIAGNOSIS_DISCLAIMER = (
    "I can provide general information from the clinic's approved resources, but I can't "
    "diagnose a medical condition. For a proper evaluation, please consult a qualified "
    "healthcare professional."
)

MEDICATION_DISCLAIMER = (
    "I can't prescribe or recommend medication. A qualified healthcare professional can "
    "assess your situation and advise you appropriately."
)

HUMAN_HANDOFF_OFFER = (
    "I don't want to give you inaccurate information. I can connect you with the clinic "
    "team so they can assist you directly. Would you like me to help with that?"
)

EHR_DISCLAIMER = (
    "I don't have direct access to your medical records through this chat. I can help "
    "you request a follow-up appointment or share general clinic information, and the "
    "clinic team can pull up your history when you arrive or call in."
)

FALLBACK_NO_INFO = (
    "I don't have that information available right now. I can help you contact the "
    "clinic team for confirmation."
)

APPOINTMENT_SUBMITTED = (
    "Your appointment request has been submitted to the CareConnect team. The clinic "
    "team will contact you on your phone or email to confirm the appointment. Thank you!"
)

APPOINTMENT_SUBMIT_FAILED = (
    "I'm sorry, I wasn't able to auto-submit your appointment request right now. Please "
    "contact the clinic directly by phone so the team can assist you immediately. You "
    "can find the clinic phone numbers in our contact information."
)


# ---------------------------------------------------------------------------
# Pure rule detectors
# ---------------------------------------------------------------------------

_EMERGENCY_KEYWORDS = (
    r"\bchest\s+pain\b", r"\bheart\s+(attack|pain)\b",
    r"\bdifficulty\s+breathing\b", r"\bshortness\s+of\s+breath\b", r"\bcan'?t\s+breathe\b",
    r"\bloss\s+of\s+consciousness\b", r"\bpassed?\s+out\b", r"\bfainted\b", r"\bunconscious\b",
    r"\bsevere\s+bleed(ing)?\b", r"\bheavy\s+bleed(ing)?\b", r"\bwon'?t\s+stop\s+bleeding\b",
    r"\bstroke\b", r"\bslurred\s+speech\b", r"\bsudden\s+weakness\b", r"\bface\s+drooping\b",
    r"\bsevere\s+allergic\b", r"\banaphylaxis\b",
    r"\bthroat\s+(closing|swollen|tight|clogged)\b", r"\bcan'?t\s+swallow\b",
    r"\bmajor\s+trauma\b", r"\bbad\s+accident\b", r"\bcar\s+crash\b",
    r"\bi\s+think\s+i'?m?\s+(dying|going\s+to\s+die)\b",
    r"\blife\s+threat(en|ing)\b", r"\bimmediate\s+danger\b",
)

_DIAGNOSIS_HINTS = (
    r"\bis\s+it\s+(a\s+)?(psoriasis|eczema|dengue|malaria|typhoid|cancer|tuberculosis|pneumonia|asthma|thyroid|diabetes|pcod|pcos|acidity|migraine|vertigo|infection|vitamin\s+d|deficiency|covid|corona|hiv|std|sti|allergy|fracture)\b",
    r"\bwhat\s+(disease|condition|illness|problem|sickness|diagnosis)\s+(do\s+i\s+have|is\s+this)\b",
    r"\bshould\s+i\s+worry\s+about\s+(cancer|tumor|lump|mass)\b",
    r"\bam\s+i\s+pregnant\b",
    r"\bis\s+this\s+(normal|serious|dangerous)\b",
    r"\bcan\s+you\s+(check|diagnose|tell\s+me\s+what'?s\s+wrong|identify|test|interpret|read)\b",
    r"\bwhat\s+do\s+(these?\s+)?(results?|reports?)\s+mean\b",
    r"\binterpret\s+my\b",
)

_MEDICATION_HINTS = (
    r"\bwhat\s+medicine\b", r"\bwhich\s+(tablet|pill|drug|medicine|injection|cream|ointment|syrup|antibiotic)\b",
    r"\bcan\s+i\s+take\s+(a\s+)?(paracetamol|dolo|crocin|ibuprofen|aspirin|antibiotic|steroid|cough\s+syrup)\b",
    r"\bprescribe\b", r"\brecommend\s+medication\b",
    r"\bhow\s+much\s+(dose|mg|ml)\b.*\b(medicine|pill|tablet)\b",
    r"\bshould\s+i\s+take\s+(an?\s+)?(antibiotic|pain\s+killer|tablet|medicine)\b",
    r"\bmedicine\s+for\b",
)

_HUMAN_HANDOFF_HINTS = (
    r"\b(speak|talk)\s+(to|with)\s+(a|real|live)?\s*(human|person|agent|representative|doctor|staff|manager|team)\b",
    r"\bi\s+want\s+(to\s+speak\s+to\s+)?(a\s+)?human\b",
    r"\btransfer\s+me\b",
    r"\bfrustrated\b", r"\bnot\s+help(ing|ful)\b", r"\buseless\b", r"\bstupid\b.*\bbot\b",
    r"\bstop\s+(this\s+)?(chat|bot|autoreply)\b",
    r"\byou\s+aren'?t\s+helping\b",
    r"\bcan\s+i\s+speak\b",
    r"\bconnect\s+me\b.*\bhuman\b",
)

_EHR_HINTS = (
    r"\b(my|the\s+)\s*(medical|patient|health)\s*(record|history|file|chart|notes?|reports?)\b",
    r"\b(pull\s+up|see|view|access|show|find|look\s+at|retrieve|pull)\s+(my\s+)?(records?|files?|history|notes?|last\s+visit)\b",
    r"\blast\s+(visit|appointment)\s+(notes?|summary|report)\b",
    r"\bprevious\s+(visit|report|notes?)\b",
    r"\bwhat\s+(did|was)\s+my\s+(last|previous)\b.*\b(doctor|report|result|diagnosis)\b",
)


def _match(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def looks_like_emergency(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _match(_EMERGENCY_KEYWORDS, text)


def looks_like_diagnosis_request(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _match(_DIAGNOSIS_HINTS, text)


def looks_like_medication_request(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _match(_MEDICATION_HINTS, text)


def looks_like_human_handoff(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _match(_HUMAN_HANDOFF_HINTS, text)


def looks_like_ehr_request(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _match(_EHR_HINTS, text)


@dataclass
class SafetyResponse:
    action: SafetyAction
    message: str

    @classmethod
    def none(cls) -> "SafetyResponse":
        return cls(action=SafetyAction.NONE, message="")


def evaluate_safety(text: str) -> SafetyResponse:
    """Apply safety rules in strict priority order and return the first match."""
    if looks_like_emergency(text):
        return SafetyResponse(SafetyAction.EMERGENCY_MSG, EMERGENCY_RESPONSE)
    if looks_like_diagnosis_request(text):
        return SafetyResponse(SafetyAction.NO_DIAGNOSIS_MSG, DIAGNOSIS_DISCLAIMER)
    if looks_like_medication_request(text):
        return SafetyResponse(SafetyAction.NO_MEDICATION_MSG, MEDICATION_DISCLAIMER)
    if looks_like_human_handoff(text):
        return SafetyResponse(SafetyAction.HUMAN_HANDOFF_MSG, HUMAN_HANDOFF_OFFER)
    if looks_like_ehr_request(text):
        return SafetyResponse(SafetyAction.NO_EHR_MSG, EHR_DISCLAIMER)
    return SafetyResponse.none()
