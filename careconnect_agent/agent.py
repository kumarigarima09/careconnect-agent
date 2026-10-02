"""Top-level CareConnectAgent orchestrator.

Wires safety guardrails, classifiers, the intake controller, grounded KB
answering, the lead scorer, summary generator, and the Salesforce adapter
into a single ``handle(user_message) -> str`` method.
"""
from __future__ import annotations

import logging
from typing import Optional

from .classifiers import Intent, PatientType, classify as classify_message
from .grounded import answer_from_kb
from .intake import IntakeController, opening_message
from .knowledge_base import KnowledgeBase
from .llm import LLMClientProtocol, MockLLMClient, build_default_client
from .salesforce import (
    MockSalesforceAdapter,
    SalesforceAdapter,
    SalesforceError,
    build_lead_payload,
    build_task_payload,
)
from .safety import (
    APPOINTMENT_SUBMITTED,
    APPOINTMENT_SUBMIT_FAILED,
    SafetyAction,
    evaluate_safety,
)
from .scoring import compute_lead_score
from .state import ConversationState
from .summary import generate_crm_summary


logger = logging.getLogger("careconnect_agent.agent")


class CareConnectAgent:
    def __init__(self,
                 kb: Optional[KnowledgeBase] = None,
                 llm: Optional[LLMClientProtocol] = None,
                 salesforce: Optional[SalesforceAdapter] = None) -> None:
        self.kb = kb or KnowledgeBase()
        self.llm = llm if llm is not None else build_default_client()
        self.salesforce = salesforce if salesforce is not None else MockSalesforceAdapter()
        self.intake = IntakeController(kb=self.kb, llm=self.llm)
        self.state = ConversationState()
        # Record open message on first `start()` / first handle call.
        self._started = False

    # ---- conversation lifecycle ----------------------------------------
    def start(self) -> str:
        if self._started:
            return self._last_assistant() or opening_message()
        self._started = True
        reply = opening_message()
        self.state.add_assistant(reply)
        return reply

    def _last_assistant(self) -> str:
        for m in reversed(self.state.history):
            if m.get("role") == "assistant":
                return m["content"] or ""
        return ""

    # ---- top-level handler ---------------------------------------------
    def handle(self, user_message: str) -> str:
        if not self._started:
            # Prime the conversation with the opening message on first user turn
            self._started = True
        user_message = (user_message or "").strip()
        if not user_message:
            return "I'm here to help. Could you please share what you're looking for?"

        self.state.add_user(user_message)

        # 1) Healthcare safety FIRST. Emergency + diagnosis + medication
        #    + human handoff + EHR disclaimers short-circuit normal flow.
        safety = evaluate_safety(user_message)
        if safety.action == SafetyAction.EMERGENCY_MSG:
            self.state.intent = Intent.EMERGENCY
            reply = safety.message
            self.state.add_assistant(reply)
            return reply
        if safety.action in (SafetyAction.NO_DIAGNOSIS_MSG, SafetyAction.NO_MEDICATION_MSG):
            reply = safety.message
            # Attach a gentle next-step offer for non-urgent disclaimers
            extra = (
                "\n\nI can still help with general clinic information or "
                "requesting an appointment if that would be useful."
            )
            reply = reply + extra
            self.state.add_assistant(reply)
            return reply
        if safety.action == SafetyAction.HUMAN_HANDOFF_MSG:
            self.state.intent = Intent.HUMAN_HANDOFF
            self.state.awaiting_human = True
            reply = safety.message
            self.state.add_assistant(reply)
            return reply

        # 2) EHR disclaimer handled by intake internally; but if safety says
        #    NO_EHR_MSG and user only asked that, return that + offer follow-up
        if safety.action == SafetyAction.NO_EHR_MSG:
            # Let intake merge EHR disclaimer + next question when relevant
            pass

        # 3) Classify intent + patient type
        cls = classify_message(user_message,
                               prior_intent=self.state.intent,
                               prior_patient=self.state.patient_type)
        self.state.intent = cls.intent
        if cls.patient_type != PatientType.UNKNOWN:
            self.state.patient_type = cls.patient_type

        # 4) Delegate to intake controller. Intake handles:
        #    - field extraction,
        #    - next 1-2 questions,
        #    - summary + confirmation prompt,
        #    - confirmation yes/no handling
        reply = self.intake.handle(user_message, self.state)

        # Intake returns the sentinel "__SUBMIT__" when user confirmed
        if reply == "__SUBMIT__":
            reply = self._submit_to_crm()
            self.state.add_assistant(reply)
            return reply

        # 5) If intake returned nothing (unlikely), fall back to KB answer.
        if not reply:
            try:
                reply = answer_from_kb(user_message, self.kb, self.llm)
            except Exception as e:  # noqa: BLE001
                logger.warning("KB fallback failed: %s", e)
                from .safety import FALLBACK_NO_INFO
                reply = FALLBACK_NO_INFO

        self.state.add_assistant(reply)
        return reply

    # ---- CRM submission -------------------------------------------------
    def _submit_to_crm(self) -> str:
        flags = self.state.scoring_flags()
        result = compute_lead_score(flags)
        crm_summary = generate_crm_summary(self.state, result,
                                           llm=None if isinstance(self.llm, MockLLMClient) else self.llm)
        ai_summary = self._extract_ai_summary(crm_summary)
        try:
            lead_payload = build_lead_payload(self.state, result.score, result.temperature, ai_summary)
            task_payload = build_task_payload(self.state, result.score, result.temperature,
                                               ai_summary, crm_summary)
            lead_id = self.salesforce.create_lead(lead_payload)
            self.salesforce.create_task(lead_id, task_payload)
            self.state.submitted = True
            logger.info("Created Salesforce Lead %s and Task for conversation %s",
                        lead_id, self.state.conversation_id)
            return APPOINTMENT_SUBMITTED
        except SalesforceError as e:
            logger.error("Salesforce submission failed: %s", e)
            return APPOINTMENT_SUBMIT_FAILED
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error during CRM submit: %s", e)
            return APPOINTMENT_SUBMIT_FAILED

    @staticmethod
    def _extract_ai_summary(crm_summary: str) -> str:
        # Use the "SUMMARY:" section of the CRM summary as the AI summary.
        if "SUMMARY:" in crm_summary:
            body = crm_summary.split("SUMMARY:", 1)[1].strip()
            # Take first 250 chars so custom fields don't overflow
            return (body[:247] + "...") if len(body) > 250 else body
        return (crm_summary[:247] + "...") if len(crm_summary) > 250 else crm_summary
