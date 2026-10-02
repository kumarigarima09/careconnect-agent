"""Deterministic lead scoring engine.

Pure-function, no LLM involvement. Weights and buckets match FR-12 / FR-13.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class LeadTemperature(str, enum.Enum):
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


WEIGHTS = {
    "appointment_request": 40,
    "specific_service": 20,
    "preferred_location": 10,
    "preferred_datetime": 15,
    "contact_info": 10,
    "insurance_info": 5,
}

MAX_SCORE = 100


def compute_score(flags: dict[str, bool]) -> int:
    score = 0
    for key, weight in WEIGHTS.items():
        if flags.get(key):
            score += weight
    return min(score, MAX_SCORE)


def temperature_for(score: int) -> LeadTemperature:
    if score >= 70:
        return LeadTemperature.HOT
    if score >= 40:
        return LeadTemperature.WARM
    return LeadTemperature.COLD


@dataclass(frozen=True)
class ScoringResult:
    score: int
    temperature: LeadTemperature


def compute_lead_score(flags: dict[str, bool]) -> ScoringResult:
    score = compute_score(flags)
    return ScoringResult(score=score, temperature=temperature_for(score))
