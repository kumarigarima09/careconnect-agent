"""Unit tests for classifiers, safety, KB, scoring, state, salesforce payloads."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careconnect_agent.classifiers import (  # noqa: E402
    classify, PatientType, Intent,
)
from careconnect_agent.safety import (  # noqa: E402
    evaluate_safety, SafetyAction,
    looks_like_emergency, looks_like_diagnosis_request,
    looks_like_medication_request, looks_like_human_handoff,
    looks_like_ehr_request,
    EMERGENCY_RESPONSE, DIAGNOSIS_DISCLAIMER, MEDICATION_DISCLAIMER,
    HUMAN_HANDOFF_OFFER, EHR_DISCLAIMER, FALLBACK_NO_INFO,
)
from careconnect_agent.knowledge_base import (  # noqa: E402
    KnowledgeBase, SERVICE_NAMES, LOCATION_NAMES,
)
from careconnect_agent.scoring import (  # noqa: E402
    compute_score, temperature_for, compute_lead_score,
    LeadTemperature, WEIGHTS, MAX_SCORE,
)
from careconnect_agent.state import (  # noqa: E402
    ConversationState, REQUIRED_APPOINTMENT_FIELDS,
)
from careconnect_agent.salesforce import (  # noqa: E402
    build_lead_payload, build_task_payload, priority_for,
)
from careconnect_agent.classifiers import PatientType as PT, Intent as IT  # noqa: E402
from itertools import product  # noqa: E402


# ------------------------------------------------------
# Classifiers
# ------------------------------------------------------
def test_classifier_existing_patient():
    c = classify("I've already visited your Noida clinic.")
    assert c.patient_type == PatientType.EXISTING_PATIENT


def test_classifier_new_patient():
    c = classify("I've never visited your clinic before.")
    assert c.patient_type == PatientType.NEW_PATIENT


def test_classifier_follow_up_existing():
    c = classify("I need a follow-up with my doctor.")
    assert c.patient_type == PatientType.EXISTING_PATIENT
    assert c.intent in (Intent.EXISTING_PATIENT_SUPPORT, Intent.APPOINTMENT_REQUEST)


def test_classifier_emergency_intent_wins():
    c = classify("I'm having severe chest pain.")
    assert c.intent == Intent.EMERGENCY


def test_classifier_intent_transitions():
    c1 = classify("Do you have dermatology in Noida?")
    assert c1.intent == Intent.SERVICE_INFORMATION
    c2 = classify("Okay, I'd like to book an appointment.")
    assert c2.intent == Intent.APPOINTMENT_REQUEST


# ------------------------------------------------------
# Safety guardrails
# ------------------------------------------------------
def test_safety_diagnosis():
    assert looks_like_diagnosis_request("I have a rash. Is it psoriasis?")
    s = evaluate_safety("I have a rash. Is it psoriasis?")
    assert s.action == SafetyAction.NO_DIAGNOSIS_MSG
    assert s.message == DIAGNOSIS_DISCLAIMER


def test_safety_medication():
    assert looks_like_medication_request("What medicine should I take for a fever?")
    s = evaluate_safety("What medicine should I take for a fever?")
    assert s.action == SafetyAction.NO_MEDICATION_MSG
    assert s.message == MEDICATION_DISCLAIMER


def test_safety_human_handoff():
    assert looks_like_human_handoff("I want to speak to a human")
    s = evaluate_safety("I want to speak to a human")
    assert s.action == SafetyAction.HUMAN_HANDOFF_MSG
    assert s.message == HUMAN_HANDOFF_OFFER


def test_safety_emergency_detection():
    for phrase in (
        "Severe allergic reaction with throat closing",
        "I'm having severe chest pain and shortness of breath",
        "I passed out and now feel dizzy",
    ):
        assert looks_like_emergency(phrase), phrase
        assert evaluate_safety(phrase).action == SafetyAction.EMERGENCY_MSG
        assert evaluate_safety(phrase).message == EMERGENCY_RESPONSE


def test_safety_priority_emergency_beats_medication():
    s = evaluate_safety("severe chest pain — what medicine?")
    assert s.action == SafetyAction.EMERGENCY_MSG


def test_safety_ehr_disclaimer():
    assert looks_like_ehr_request("Can you see my last visit notes?")
    s = evaluate_safety("Can you see my last visit notes?")
    assert s.action == SafetyAction.NO_EHR_MSG
    assert s.message == EHR_DISCLAIMER


def test_safety_none_for_benign():
    s = evaluate_safety("What time does the Noida clinic open?")
    assert s.action == SafetyAction.NONE


# ------------------------------------------------------
# Knowledge base
# ------------------------------------------------------
def test_kb_3_locations():
    kb = KnowledgeBase()
    assert len(kb.locations) == 3
    for loc in kb.locations:
        for k in ("address", "hours_weekday", "hours_saturday", "phone", "email"):
            v = getattr(loc, k)
            assert v and len(v) > 5


def test_kb_8_services_and_fees():
    kb = KnowledgeBase()
    assert len(kb.services) == 8
    for name in SERVICE_NAMES:
        svc = kb.get_service(name)
        assert svc is not None, name
        if name not in ("Diagnostics", "Preventive Health Checkups"):
            assert svc.fee_inr > 0


def test_kb_search_relevance():
    kb = KnowledgeBase()
    for q in ("dermatology Noida", "doctor fees", "insurance"):
        hits = kb.search(q)
        assert 1 <= len(hits) <= 5


# ------------------------------------------------------
# Scoring (deterministic)
# ------------------------------------------------------
def test_scoring_95_hot_example():
    flags = dict(appointment_request=True, specific_service=True,
                  preferred_location=True, preferred_datetime=True,
                  contact_info=True, insurance_info=False)
    r = compute_lead_score(flags)
    assert r.score == 95
    assert r.temperature == LeadTemperature.HOT


def test_scoring_30_cold():
    flags = {k: False for k in WEIGHTS}
    flags["specific_service"] = True
    flags["preferred_location"] = True
    assert compute_score(flags) == 30
    assert temperature_for(30) == LeadTemperature.COLD


def test_scoring_40_warm():
    flags = {k: False for k in WEIGHTS}
    flags["appointment_request"] = True
    assert compute_score(flags) == 40
    assert temperature_for(40) == LeadTemperature.WARM


def test_scoring_max_100():
    flags = {k: True for k in WEIGHTS}
    assert compute_score(flags) == 100


def test_scoring_buckets_at_boundaries():
    assert temperature_for(39) == LeadTemperature.COLD
    assert temperature_for(40) == LeadTemperature.WARM
    assert temperature_for(69) == LeadTemperature.WARM
    assert temperature_for(70) == LeadTemperature.HOT
    assert temperature_for(100) == LeadTemperature.HOT
    assert temperature_for(0) == LeadTemperature.COLD


def test_scoring_all_64_combinations():
    for combo in product([False, True], repeat=6):
        flags = {k: combo[i] for i, k in enumerate(WEIGHTS)}
        raw = sum(WEIGHTS[k] * v for k, v in flags.items())
        expected = min(raw, MAX_SCORE)
        assert compute_score(flags) == expected
        temp = temperature_for(expected)
        buckets_correct = {
            LeadTemperature.HOT: 70 <= expected <= 100,
            LeadTemperature.WARM: 40 <= expected <= 69,
            LeadTemperature.COLD: 0 <= expected <= 39,
        }
        assert buckets_correct[temp], (expected, temp)


# ------------------------------------------------------
# State
# ------------------------------------------------------
def test_state_unique_conversation_ids():
    s1 = ConversationState()
    s2 = ConversationState()
    assert s1.conversation_id != s2.conversation_id
    assert len(s1.conversation_id) == 32


def test_state_remember_and_has():
    s = ConversationState()
    s.remember("requested_service", "Dermatology")
    s.remember("preferred_location", "Noida")
    assert s.has("requested_service")
    assert s.has("preferred_location")
    assert s.get("requested_service") == "Dermatology"
    assert not s.has("first_name")


def test_state_missing_required_when_all_filled():
    s = ConversationState()
    for f in REQUIRED_APPOINTMENT_FIELDS:
        s.remember(f, "X")
    assert s.missing_required_fields() == []
    # Also: patient_type stored via enum → not required but must be
    # included when auto-populated.
    s.patient_type = PatientType.NEW_PATIENT
    assert s.patient_type == PatientType.NEW_PATIENT


def test_state_transcript_roundtrip():
    s = ConversationState()
    s.add_user("Hello")
    s.add_assistant("Hi!")
    s.add_user("Need dermatologist")
    t = s.transcript_text()
    assert "Patient: Hello" in t
    assert "Agent: Hi!" in t
    assert "Patient: Need dermatologist" in t


# ------------------------------------------------------
# Salesforce payload builders
# ------------------------------------------------------
def _build_state_for_payload():
    s = ConversationState()
    s.remember("first_name", "Rahul")
    s.remember("last_name", "Sharma")
    s.remember("phone", "9876543210")
    s.remember("email", "rahul@example.com")
    s.remember("preferred_location", "Noida")
    s.remember("requested_service", "Dermatology")
    s.remember("preferred_date", "10 October")
    s.remember("preferred_time", "Afternoon")
    s.remember("patient_type", PatientType.NEW_PATIENT.value)
    s.remember("insurance_provider", "Star Health")
    s.patient_type = PatientType.NEW_PATIENT
    s.intent = Intent.APPOINTMENT_REQUEST
    s.add_user("I need a dermatologist")
    s.add_assistant("Which location?")
    return s


def test_salesforce_lead_payload():
    s = _build_state_for_payload()
    lp = build_lead_payload(s, 95, LeadTemperature.HOT, "summary text")
    assert lp["LeadSource"] == "Website AI Agent"
    assert lp["FirstName"] == "Rahul"
    assert lp["LastName"] == "Sharma"
    assert lp["Phone"] == "9876543210"
    assert lp["Email"] == "rahul@example.com"
    assert lp["Patient_Type__c"] == PatientType.NEW_PATIENT.value
    assert lp["Preferred_Location__c"] == "Noida"
    assert lp["Requested_Service__c"] == "Dermatology"
    assert lp["Preferred_Date__c"] == "10 October"
    assert lp["Preferred_Time__c"] == "Afternoon"
    assert lp["Lead_Temperature__c"] == "HOT"
    assert lp["Lead_Score__c"] == 95
    assert lp["AI_Summary__c"] == "summary text"
    assert lp["Conversation_ID__c"] == s.conversation_id
    for k in ("Insurance_Provider__c", "Appointment_Intent__c"):
        assert k in lp


def test_salesforce_task_payload():
    s = _build_state_for_payload()
    crm = "CRM SUMMARY BLOCK"
    tp = build_task_payload(s, 95, LeadTemperature.HOT, "ai", crm)
    assert tp["Subject"] == "AI Agent - Appointment Request"
    assert tp["Status"] == "Not Started"
    assert priority_for(LeadTemperature.HOT) == "High"
    assert priority_for(LeadTemperature.WARM) == "Normal"
    assert priority_for(LeadTemperature.COLD) == "Low"
    assert tp["Priority"] == "High"
    assert "AI AGENT CONVERSATION" in tp["Description"]
    assert "Patient: I need a dermatologist" in tp["Description"]
    assert "Lead Temperature: HOT (score 95)" in tp["Description"]
    assert crm in tp["Description"]


# ------------------------------------------------------
# Grounded answers rule-layer (offline, no LLM)
# ------------------------------------------------------
def test_grounded_service_in_location_offline():
    from careconnect_agent.grounded import answer_from_kb
    from careconnect_agent.llm import MockLLMClient
    kb = KnowledgeBase()
    mock = MockLLMClient(fallback="(no llm)")
    reply = answer_from_kb("Do you have dermatology in Noida?", kb, mock)
    assert "dermatology" in reply.lower()
    assert "Noida" in reply


def test_grounded_pricing_offline():
    from careconnect_agent.grounded import answer_from_kb
    from careconnect_agent.llm import MockLLMClient
    kb = KnowledgeBase()
    fee = kb.get_fee("Dermatology")
    reply = answer_from_kb("How much is the dermatology consultation?", kb, MockLLMClient())
    assert str(fee) in reply


def test_grounded_fallback_missing_info_offline():
    from careconnect_agent.grounded import answer_from_kb
    from careconnect_agent.llm import MockLLMClient
    kb = KnowledgeBase()
    # Question with no KB facts: exact phrase "robotic surgery" not in KB
    reply = answer_from_kb("Does Dr. Sharma perform robotic surgery?", kb, MockLLMClient())
    # Should contain fallback
    assert FALLBACK_NO_INFO in reply
