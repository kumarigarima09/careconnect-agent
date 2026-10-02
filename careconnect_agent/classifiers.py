"""Conversation enums and hybrid rule-based / LLM classifiers."""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class PatientType(str, enum.Enum):
    NEW_PATIENT = "NEW_PATIENT"
    EXISTING_PATIENT = "EXISTING_PATIENT"
    UNKNOWN = "UNKNOWN"


class Intent(str, enum.Enum):
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    LOCATION = "LOCATION"
    DOCTOR_INFORMATION = "DOCTOR_INFORMATION"
    SERVICE_INFORMATION = "SERVICE_INFORMATION"
    PRICING = "PRICING"
    INSURANCE = "INSURANCE"
    APPOINTMENT_REQUEST = "APPOINTMENT_REQUEST"
    RESCHEDULE = "RESCHEDULE"
    CANCELLATION = "CANCELLATION"
    EXISTING_PATIENT_SUPPORT = "EXISTING_PATIENT_SUPPORT"
    EMERGENCY = "EMERGENCY"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    OTHER = "OTHER"


_EXISTING_PATIENT_HINTS = (
    "already visited", "already been", "previously visited", "before visited",
    "i have been here before", "i've been here before",
    "existing patient", "returning patient", "come back", "coming back",
    "follow-up", "follow up", "followup",
    "last visit", "previous visit", "my doctor",
    "my last", "my previous", "my earlier",
    "patient id", "already booked", "already registered", "already had",
)

_NEW_PATIENT_HINTS = (
    "first time", "never visited", "never been", "never come",
    "new to careconnect", "new patient", "new here", "first visit",
    "visiting for the first",
)


def _make_emergency_patterns() -> tuple[str, ...]:
    return (
        "chest pain", "heart attack", "heart pain",
        "difficulty breathing", "shortness of breath", "can't breathe", "cant breathe",
        "loss of consciousness", "passed out", "pass out", "fainted", "unconscious",
        "severe bleeding", "heavy bleeding", "won't stop bleeding", "wont stop bleeding",
        "stroke", "slurred speech", "sudden weakness", "face drooping",
        "severe allergic", "anaphylaxis",
        "throat closing", "throat swollen", "throat tight", "can't swallow", "cant swallow",
        "major trauma", "car crash", "bad accident",
        "think i'm dying", "think im dying", "think i am dying", "going to die",
        "life threatening", "life threat", "immediate danger",
    )


def _make_intent_patterns() -> list[tuple[Intent, tuple[str, ...]]]:
    return [
        (Intent.EMERGENCY, _make_emergency_patterns()),
        (Intent.HUMAN_HANDOFF, (
            "speak to a human", "speak to human", "talk to a human", "talk to human",
            "speak to a real", "talk to a real", "want a human", "want to talk",
            "speak to someone", "talk to someone", "talk to a person", "speak to a person",
            "customer support", "transfer me", "connect me",
            "speak to staff", "talk to staff", "speak to team", "talk to team",
        )),
        (Intent.CANCELLATION, (
            "cancel appointment", "cancel my appointment", "cancelling appointment",
            "cancellation", "cancel booking",
        )),
        (Intent.RESCHEDULE, (
            "reschedule", "rescheduling",
            "change appointment", "change date", "change slot", "change time",
            "move appointment", "move my appointment",
        )),
        (Intent.APPOINTMENT_REQUEST, (
            "book an appointment", "book appointment", "schedule appointment",
            "request an appointment", "request appointment", "fix an appointment",
            "need an appointment", "need appointment", "need a consultation",
            "want an appointment", "want to book", "want appointment",
            "appointment please", "appointment today", "appointment tomorrow",
            "appointment available", "see a doctor", "see the doctor",
            "see the dermatologist", "consultation please",
            "come in today", "walk in", "walk-in",
        )),
        (Intent.PRICING, (
            "price of", "pricing", "cost of", "how much", "consultation fee",
            "fee of", "charges", "how much does",
        )),
        (Intent.INSURANCE, (
            "insurance", "insurer", "insured", "health insurance",
            "cashless", "claim", "premium", "insurance policy", "insurance covered",
        )),
        (Intent.DOCTOR_INFORMATION, (
            "do you have a doctor", "which doctor", "who is the doctor",
            "dermatologist", "pediatrician", "orthopedic", "gynecologist",
            "dentist", "physician", "specialist",
        )),
        (Intent.LOCATION, (
            "address", "where is", "location", "which area", "which sector",
            "which city", "branch", "clinic in", "nearest clinic", "closest clinic",
            "how to reach", "directions",
        )),
        (Intent.SERVICE_INFORMATION, (
            "do you have", "do you offer", "do you provide",
            "dermatology", "dermatologist", "pediatrics", "child doctor",
            "baby doctor", "orthopedics", "ortho", "gynecology", "gynea",
            "dentistry", "dental", "teeth", "general medicine", "check up",
            "checkup", "preventive health", "diagnostic",
            "services you offer", "what services", "what specialties",
            "skin", "hair",
        )),
        (Intent.EXISTING_PATIENT_SUPPORT, (
            "my records", "my report", "my last report", "my previous report",
            "my medical records", "my history", "my file",
            "pull up my", "see my records", "view my records", "access my records",
            "show my records", "my patient id", "patient id",
            "my last visit", "my last appointment", "my previous visit",
            "what did my doctor",
        )),
        (Intent.GENERAL_INFORMATION, (
            "about careconnect", "about your clinic", "overview",
            "what is careconnect", "tell me about",
            "opening hours", "opening time", "what time", "hours of operation",
            "open on sunday", "closed on", "open timings",
            "contact number", "phone number", "email id", "email address",
            "how to contact", "how to reach",
        )),
    ]


_INTENT_RULES: list[tuple[Intent, tuple[str, ...]]] = _make_intent_patterns()


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    h = haystack.lower()
    return any(n in h for n in needles)


def classify_patient_type_rule(text: str, prior: PatientType = PatientType.UNKNOWN) -> PatientType:
    if not isinstance(text, str) or not text.strip():
        return prior
    t = text.lower()
    if _contains_any(t, _EXISTING_PATIENT_HINTS):
        return PatientType.EXISTING_PATIENT
    if _contains_any(t, _NEW_PATIENT_HINTS):
        return PatientType.NEW_PATIENT
    if "first" in t and "time" in t:
        return PatientType.NEW_PATIENT
    if "already a patient" in t or "already patient" in t:
        return PatientType.EXISTING_PATIENT
    return prior


def classify_intent_rule(text: str, prior: Intent = Intent.OTHER) -> Intent:
    if not isinstance(text, str) or not text.strip():
        return prior
    for intent, patterns in _INTENT_RULES:
        if _contains_any(text, patterns):
            return intent
    return prior


LLM_INTENT_PROMPT = (
    "You are a classification assistant for a clinic chatbot. "
    "Given the user message, classify INTENT and PATIENT_TYPE as JSON only, with two keys: "
    "\"intent\" and \"patient_type\". "
    "INTENT must be one of: GENERAL_INFORMATION, LOCATION, DOCTOR_INFORMATION, "
    "SERVICE_INFORMATION, PRICING, INSURANCE, APPOINTMENT_REQUEST, RESCHEDULE, "
    "CANCELLATION, EXISTING_PATIENT_SUPPORT, EMERGENCY, HUMAN_HANDOFF, OTHER. "
    "PATIENT_TYPE must be one of: NEW_PATIENT, EXISTING_PATIENT, UNKNOWN. "
    "Return only the JSON object. No commentary."
)


@dataclass
class Classification:
    intent: Intent
    patient_type: PatientType


def classify(text: str,
             prior_intent: Intent = Intent.OTHER,
             prior_patient: PatientType = PatientType.UNKNOWN) -> Classification:
    intent = classify_intent_rule(text, prior=prior_intent)
    ptype = classify_patient_type_rule(text, prior=prior_patient)
    if ptype == PatientType.EXISTING_PATIENT and intent in (Intent.OTHER, Intent.GENERAL_INFORMATION):
        intent = Intent.EXISTING_PATIENT_SUPPORT
    if ptype == PatientType.NEW_PATIENT and intent == Intent.EXISTING_PATIENT_SUPPORT:
        intent = Intent.GENERAL_INFORMATION
    return Classification(intent=intent, patient_type=ptype)
