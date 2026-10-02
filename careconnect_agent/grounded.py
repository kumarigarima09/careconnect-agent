"""Grounded knowledge-base answering (RAG-lite).

Retrieves facts via KB.search, then prompts the LLM to answer STRICTLY from
those facts. If the facts cannot answer the query the LLM is instructed to
emit the fixed FALLBACK sentence defined in ``safety.py``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .knowledge_base import KnowledgeBase
from .llm import MockLLMClient
from .safety import FALLBACK_NO_INFO

if TYPE_CHECKING:
    from .llm import LLMClientProtocol


logger = logging.getLogger("careconnect_agent.grounded")


def _answer_without_llm(query: str, facts: list[tuple[str, str]]) -> str | None:
    """Rule-based fast path for simple FAQ-style queries.

    Returns None when the rule layer cannot answer and the caller should
    delegate to the LLM.
    """
    import re
    q = query.lower()
    from .knowledge_base import SERVICE_NAMES, LOCATION_NAMES, KnowledgeBase
    service_match = next((s for s in SERVICE_NAMES if s.lower() in q), None)
    loc_match = next((l for l in LOCATION_NAMES if l.lower() in q), None)
    # Which cities / locations / branches are you in?
    if ("which cities" in q or "cities do you" in q or "what cities" in q
            or "all locations" in q or "which locations" in q or "what locations" in q
            or "how many branches" in q or "branches do you" in q
            or "cities are you" in q or "locations are you" in q):
        return (
            "CareConnect Clinics have outpatient locations in Delhi, Noida, and Meerut. "
            "Each clinic offers on-site diagnostics, pharmacy counter, and all 8 specialties. "
            "Would you like the exact address and hours for a specific location, or help "
            "requesting an appointment?"
        )
    # Contact details
    if ("contact details" in q or "your contact" in q or "contact information" in q
            or "how to contact" in q or "how do i contact" in q or "phone number" in q
            or "email address" in q or "customer support" in q):
        kb = KnowledgeBase()
        locs_info = []
        for loc in kb.locations:
            locs_info.append(f"- {loc.name}: {loc.phone} ({loc.email})")
        return (
            "You can reach CareConnect Clinics here:\n"
            + "\n".join(locs_info)
            + "\nCentral support (9 AM–7 PM, Mon–Sat): +91-11-4000-0000 / hello@careconnectclinics.example. "
            + "Is there anything else I can help with?"
        )
    # "do you have SERVICE in LOCATION?"
    if ("have" in q or "offer" in q or "available" in q) and service_match:
        kb = KnowledgeBase()
        doctors_here = kb.get_doctors(location=loc_match, specialty=service_match)
        if loc_match and not doctors_here:
            any_doctor = kb.get_doctors(specialty=service_match)
            if not any_doctor:
                return None
            return (
                f"CareConnect Clinics offer {service_match}. {loc_match} location does not "
                f"have a {service_match} specialist on staff, but we offer it at other "
                f"locations. Would you like me to show which locations have {service_match} "
                f"doctors, or help you request an appointment?"
            )
        loc_msg = f"at the {loc_match} clinic " if loc_match else ""
        fee = kb.get_fee(service_match)
        fee_note = ""
        if fee:
            fee_note = f" The consultation fee for {service_match} is INR {fee}."
        return (
            f"Yes. CareConnect {loc_msg}offers {service_match} consultations.{fee_note} "
            f"Would you like information about the doctors or would you like to request "
            f"an appointment?"
        )
    # How much does SERVICE cost?
    if ("how much" in q or "price" in q or "fee" in q or "cost" in q) and service_match:
        kb = KnowledgeBase()
        fee = kb.get_fee(service_match)
        svc = kb.get_service(service_match)
        if fee:
            return (
                f"According to our clinic information, the {service_match} consultation "
                f"fee is INR {fee} ({svc.duration_min} mins). Would you like help requesting "
                f"an appointment?"
            )
        if svc and service_match in ("Diagnostics", "Preventive Health Checkups"):
            return (
                f"{service_match} pricing varies by test or package. The {service_match.lower()} "
                f"price card is available at each clinic. Would you like me to connect you with "
                f"the clinic team for specific pricing, or help you request an appointment?"
            )
        return FALLBACK_NO_INFO
    # Insurance-related
    if "insurance" in q:
        kb = KnowledgeBase()
        insurers = kb.get_insurers()
        return (
            f"CareConnect Clinics accept most major Indian health insurers including "
            f"{', '.join(insurers[:4])}, and others. Payment options include cash, UPI, "
            f"all major cards, and insurance cashless (where authorized). Would you like "
            f"more details or help with an appointment?"
        )
    # Clinic hours / opening
    if ("hour" in q or "open" in q or "timing" in q or "time" in q):
        kb = KnowledgeBase()
        if loc_match:
            loc = kb.get_location(loc_match)
            if loc:
                return (
                    f"CareConnect's {loc.name} clinic hours are: Mon–Fri {loc.hours_weekday}, "
                    f"Saturday {loc.hours_saturday}, Sunday {loc.hours_sunday}. The phone "
                    f"number is {loc.phone}. Would you like help with an appointment?"
                )
        kb = KnowledgeBase()
        return (
            "CareConnect Clinics are generally open Monday through Saturday with shorter "
            "hours on Sunday (Meerut is closed on Sunday). For exact hours by location, "
            "could you tell me which location you're interested in?"
        )
    # Doctor/specialist information at location: "Tell me about SERVICE specialists in LOC?"
    if loc_match and service_match:
        kb = KnowledgeBase()
        doctors = kb.get_doctors(location=loc_match, specialty=service_match)
        if doctors:
            lines = [f"Here are our {service_match} specialists at CareConnect {loc_match}:"]
            for d in doctors:
                lines.append(f"- {d.name} ({d.qualifications}, {d.experience_years} yrs exp; days: {d.consultation_days})")
            lines.append("Would you like to request an appointment with any of them?")
            return "\n".join(lines)
    # Just "Tell me about doctors in LOC" or just service OR just location doctor info
    if (("tell me about" in q or "who are the" in q or "which doctors" in q or "doctors available" in q)
            and (loc_match or service_match)):
        kb = KnowledgeBase()
        doctors = kb.get_doctors(location=loc_match, specialty=service_match)
        if doctors:
            loc_label = f" at {loc_match}" if loc_match else ""
            svc_label = f" for {service_match}" if service_match else ""
            lines = [f"Here are the doctors{svc_label}{loc_label}:"]
            for d in doctors[:5]:
                lines.append(f"- {d.name}, {d.specialty}, {d.location} ({d.experience_years} yrs; {d.consultation_days})")
            lines.append("Would you like to request an appointment?")
            return "\n".join(lines)
    return None


def answer_from_kb(query: str,
                   kb: KnowledgeBase,
                   llm: "LLMClientProtocol | None" = None) -> str:
    facts = kb.search(query, limit=5)
    # Fast path: pure rule-based answer if it covers the query
    rule_based = _answer_without_llm(query, facts)
    if rule_based is not None:
        return rule_based
    # Slow path: LLM grounded on facts
    if llm is None or isinstance(llm, MockLLMClient):
        # Without a real LLM, fall back to concise extracted facts + FALLBACK
        if not facts:
            return FALLBACK_NO_INFO
        lines = [f"- {title}: {body[:160]}{'…' if len(body) > 160 else ''}" for title, body in facts[:3]]
        return "\n".join(lines) + "\n" + FALLBACK_NO_INFO
    context_blocks = []
    for title, body in facts:
        context_blocks.append(f"[{title}]\n{body}")
    context = "\n\n".join(context_blocks)
    if not context:
        return FALLBACK_NO_INFO
    messages = [
        {"role": "system", "content": (
            "You are CareConnect Clinics' helpful AI assistant. Answer the user's question "
            "using ONLY the provided FACTS. Do not make up any doctors, fees, locations, "
            "hours, insurers, services, or policies. "
            "If the facts do not contain the answer, reply EXACTLY with this sentence:\n"
            f"{FALLBACK_NO_INFO}\n"
            "Keep the answer short and friendly. Answer the question first, then briefly "
            "offer the next logical action (e.g. appointment, more info) when appropriate."
        )},
        {"role": "user", "content": (
            f"FACTS:\n{context}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "Answer strictly from the FACTS above."
        )},
    ]
    try:
        response = llm.chat(messages, response_format="text", temperature=0.0)
        response = response.strip()
        if not response:
            return FALLBACK_NO_INFO
        # Safety check: if the LLM hallucinated something clearly off-script, fall back
        if "FALLBACK" in response:
            return FALLBACK_NO_INFO
        return response
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM grounded answer failed: %s", e)
        if facts:
            lines = [f"- {title}: {body[:140]}" for title, body in facts[:2]]
            return "\n".join(lines) + "\n" + FALLBACK_NO_INFO
        return FALLBACK_NO_INFO
