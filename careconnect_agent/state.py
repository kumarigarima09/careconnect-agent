"""Conversation state model and memory.

Holds patient type, intent, message history, the partially-filled appointment
form, and a ``confirmed`` flag. Pure data with deterministic helpers. The
orchestrator is the only writer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict

from .classifiers import Intent, PatientType


class AppointmentData(TypedDict, total=False):
    first_name: str
    last_name: str
    phone: str
    email: str
    preferred_location: str
    requested_service: str
    preferred_date: str
    preferred_time: str
    patient_type: str
    insurance_provider: str
    additional_notes: str
    appointment_intent: str


REQUIRED_APPOINTMENT_FIELDS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "phone",
    "email",
    "preferred_location",
    "requested_service",
    "preferred_date",
    "preferred_time",
)


OPTIONAL_APPOINTMENT_FIELDS: tuple[str, ...] = (
    "patient_type",
    "insurance_provider",
    "additional_notes",
    "appointment_intent",
)


PRETTY_FIELD_LABELS: dict[str, str] = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "phone": "Phone",
    "email": "Email",
    "preferred_location": "Preferred Location",
    "requested_service": "Requested Service",
    "preferred_date": "Preferred Date",
    "preferred_time": "Preferred Time",
    "patient_type": "Patient Type",
    "insurance_provider": "Insurance Provider",
    "additional_notes": "Additional Notes",
    "appointment_intent": "Appointment Intent",
}


@dataclass
class ConversationState:
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    patient_type: PatientType = PatientType.UNKNOWN
    intent: Intent = Intent.OTHER
    history: list[dict[str, str]] = field(default_factory=list)
    appointment: AppointmentData = field(default_factory=lambda: AppointmentData())
    collected_fields: set[str] = field(default_factory=set)
    confirmed: bool = False
    awaiting_confirmation: bool = False
    submitted: bool = False
    awaiting_email: bool = False
    awaiting_human: bool = False

    # ---- memory helpers --------------------------------------------------
    def remember(self, field: str, value: Any) -> None:
        """Record a value in the appointment form and mark it collected."""
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value:
            return
        self.appointment[field] = value  # type: ignore[literal-required]
        self.collected_fields.add(field)
        if field == "patient_type":
            if value.upper() == PatientType.NEW_PATIENT:
                self.patient_type = PatientType.NEW_PATIENT
            elif value.upper() == PatientType.EXISTING_PATIENT:
                self.patient_type = PatientType.EXISTING_PATIENT

    def has(self, field: str) -> bool:
        return field in self.collected_fields and bool(self.appointment.get(field))

    def get(self, field: str) -> Optional[str]:
        return self.appointment.get(field) if self.has(field) else None

    def missing_required_fields(self) -> list[str]:
        missing: list[str] = []
        for f in REQUIRED_APPOINTMENT_FIELDS:
            if not self.has(f):
                missing.append(f)
        # Patient type is no longer a *required* form field, but it IS used by
        # downstream consumers of the collected info. If enum already stores it, make sure the
        # typed-dict also carries it so builders see it.
        if self.patient_type in (PatientType.NEW_PATIENT, PatientType.EXISTING_PATIENT):
            self.appointment["patient_type"] = self.patient_type.value  # type: ignore[literal-required]
            self.collected_fields.add("patient_type")
        return missing

    def pretty_summary_lines(self) -> list[str]:
        lines: list[str] = []
        for f in REQUIRED_APPOINTMENT_FIELDS:
            if self.has(f):
                lines.append(f"{PRETTY_FIELD_LABELS[f]}: {self.appointment[f]}")
        for f in ("insurance_provider", "additional_notes", "appointment_intent"):
            if self.has(f):
                lines.append(f"{PRETTY_FIELD_LABELS[f]}: {self.appointment[f]}")
        return lines

    # ---- history helpers -------------------------------------------------
    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def transcript_text(self) -> str:
        lines: list[str] = []
        for m in self.history:
            role = "Agent" if m.get("role") == "assistant" else "Patient"
            lines.append(f"{role}: {m.get('content','')}")
        return "\n".join(lines)

    # ---- scoring-friendly attribute flags --------------------------------
    def scoring_flags(self) -> dict[str, bool]:
        def has_field(f: str) -> bool:
            return bool(self.appointment.get(f))
        requested = self.intent in (Intent.APPOINTMENT_REQUEST, Intent.RESCHEDULE, Intent.CANCELLATION) or bool(
            self.appointment.get("appointment_intent")
        ) or self.confirmed
        has_contact = has_field("phone") and has_field("email")
        return {
            "appointment_request": requested,
            "specific_service": has_field("requested_service"),
            "preferred_location": has_field("preferred_location"),
            "preferred_datetime": has_field("preferred_date") and has_field("preferred_time"),
            "contact_info": has_contact,
            "insurance_info": has_field("insurance_provider"),
        }
