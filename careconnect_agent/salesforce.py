"""Salesforce adapter (interface + mock implementation + payload builders).

The interface makes it trivial to swap the mock for real simple-salesforce or
REST API calls without touching orchestrator code.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from .scoring import LeadTemperature
from .state import ConversationState, PRETTY_FIELD_LABELS
from .classifiers import PatientType


logger = logging.getLogger("careconnect_agent.salesforce")


# ---------------------------------------------------------------------------
# Protocols + errors
# ---------------------------------------------------------------------------

class SalesforceError(RuntimeError):
    pass


class SalesforceAdapter(Protocol):
    def create_lead(self, payload: dict[str, Any]) -> str: ...
    def create_task(self, lead_id: str, payload: dict[str, Any]) -> str: ...


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def priority_for(temp: LeadTemperature) -> str:
    if temp == LeadTemperature.HOT:
        return "High"
    if temp == LeadTemperature.WARM:
        return "Normal"
    return "Low"


def build_lead_payload(state: ConversationState,
                       score: int,
                       temperature: LeadTemperature,
                       ai_summary: str) -> dict[str, Any]:
    data = state.appointment
    # For CRM tracking, unknown visitors are treated conservatively as NEW
    # patients.  Explicit EXISTING_PATIENT classification always wins.
    pt_raw = data.get("patient_type") or state.patient_type.value
    if pt_raw == PatientType.UNKNOWN.value:
        pt = PatientType.NEW_PATIENT.value
    else:
        pt = pt_raw
    return {
        "FirstName": data.get("first_name", ""),
        "LastName": data.get("last_name", ""),
        "Phone": data.get("phone", ""),
        "Email": data.get("email", ""),
        "LeadSource": "Website AI Agent",
        "Company": "CareConnect Clinics Lead",
        "Patient_Type__c": pt,
        "Preferred_Location__c": data.get("preferred_location", ""),
        "Requested_Service__c": data.get("requested_service", ""),
        "Preferred_Date__c": data.get("preferred_date", ""),
        "Preferred_Time__c": data.get("preferred_time", ""),
        "Insurance_Provider__c": data.get("insurance_provider", ""),
        "Appointment_Intent__c": data.get("appointment_intent") or state.intent.value,
        "Lead_Temperature__c": temperature.value,
        "Lead_Score__c": score,
        "AI_Summary__c": ai_summary,
        "Conversation_ID__c": state.conversation_id,
    }


def build_task_payload(state: ConversationState,
                       score: int,
                       temperature: LeadTemperature,
                       ai_summary: str,
                       crm_summary: str) -> dict[str, Any]:
    data = state.appointment
    pt_raw = data.get("patient_type") or state.patient_type.value
    pt = PatientType.NEW_PATIENT.value if pt_raw == PatientType.UNKNOWN.value else pt_raw
    description_parts = [
        "AI AGENT CONVERSATION",
        "",
        f"Patient Type: {pt}",
        f"Service: {data.get('requested_service', '(not specified)')}",
        f"Location: {data.get('preferred_location', '(not specified)')}",
        f"Preferred Date: {data.get('preferred_date', '(not specified)')}",
        f"Preferred Time: {data.get('preferred_time', '(not specified)')}",
        f"Contact: {data.get('phone', '')} | {data.get('email', '')}",
        f"Insurance: {data.get('insurance_provider', '(not provided)')}",
        f"Lead Temperature: {temperature.value} (score {score})",
        "",
        "CRM SUMMARY:",
        crm_summary,
        "",
        "FULL CONVERSATION:",
        state.transcript_text(),
    ]
    return {
        "Subject": "AI Agent - Appointment Request",
        "Status": "Not Started",
        "Priority": priority_for(temperature),
        "Description": "\n".join(description_parts),
        "TaskSubtype": "Task",
        "Type": "Follow-Up",
    }


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path(os.getcwd()) / ".out" / "salesforce"


@dataclass
class MockSalesforceAdapter:
    """Writes JSONL files under ``out_dir``; returns fake lead/task IDs."""
    out_dir: Path = DEFAULT_OUT_DIR
    fail_lead: bool = False
    fail_task: bool = False
    leads_written: list[dict[str, Any]] = field(default_factory=list)
    tasks_written: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.out_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str))
            fh.write("\n")

    def create_lead(self, payload: dict[str, Any]) -> str:
        if self.fail_lead:
            raise SalesforceError("Mock adapter lead failure injected for testing.")
        lead_id = "00Q" + uuid.uuid4().hex[:12].upper()
        record = {"id": lead_id, "payload": payload}
        self.leads_written.append(record)
        self._append_jsonl("leads.jsonl", record)
        return lead_id

    def create_task(self, lead_id: str, payload: dict[str, Any]) -> str:
        if self.fail_task:
            raise SalesforceError("Mock adapter task failure injected for testing.")
        task_id = "00T" + uuid.uuid4().hex[:12].upper()
        record = {"id": task_id, "what_id": lead_id, "payload": payload}
        self.tasks_written.append(record)
        self._append_jsonl("tasks.jsonl", record)
        return task_id
