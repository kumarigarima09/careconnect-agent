"""CareConnect Clinics Knowledge Base.

All fictional prototype data. Exposes a typed ``KnowledgeBase`` class with
lookup helpers and a simple keyword-weighted ``search()`` used for RAG-style
grounded answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional


SERVICE_NAMES = (
    "General Medicine",
    "Dermatology",
    "Pediatrics",
    "Orthopedics",
    "Gynecology",
    "Dentistry",
    "Diagnostics",
    "Preventive Health Checkups",
)

LOCATION_NAMES = ("Delhi", "Noida", "Meerut")

INSURER_NAMES = (
    "Star Health and Allied Insurance",
    "ICICI Lombard General Insurance",
    "HDFC ERGO General Insurance",
    "Apollo Munich Health Insurance",
    "Max Bupa Health Insurance",
    "Religare Health Insurance",
    "Bajaj Allianz General Insurance",
    "SBI General Insurance",
)


@dataclass
class Doctor:
    name: str
    specialty: str
    location: str
    qualifications: str
    experience_years: int
    languages: tuple[str, ...] = field(default_factory=lambda: ("English", "Hindi"))
    consultation_days: str = "Mon - Sat"


@dataclass
class Location:
    name: str
    address: str
    phone: str
    email: str
    hours_weekday: str
    hours_saturday: str
    hours_sunday: str
    emergency: bool
    amenities: list[str]


@dataclass
class KBService:
    name: str
    fee_inr: int
    duration_min: int
    description: str


# ---------------------------------------------------------------------------
# Raw knowledge base data (all fictional)
# ---------------------------------------------------------------------------

OVERVIEW = (
    "CareConnect Clinics is a multi-specialty primary and secondary care chain with "
    "outpatient clinics across Delhi, Noida, and Meerut. We offer consultations across "
    "7 specialties plus on-site diagnostics and preventive health checkup packages. "
    "Our mission is to deliver consistent, compassionate, and evidence-based care to "
    "every patient, first-time or returning."
)

APPOINTMENT_PROCESS = (
    "Appointments can be requested through our website chat, by phone, or by walking in. "
    "When you request an appointment online, the clinic team reviews your request and "
    "contacts you on your registered phone or email to confirm the exact slot. Appointment "
    "requests are typically responded to within 4 working hours on weekdays and 8 hours on "
    "Sundays. Walk-in consultations are also available but may involve a wait time of "
    "30-60 minutes depending on clinic load."
)

CANCELLATION_POLICY = (
    "Appointments can be rescheduled or cancelled free of charge if you give us at least "
    "24 hours' notice. For cancellations made with less than 24 hours' notice, a nominal "
    "administrative fee of INR 150 may apply. Same-day cancellations due to genuine "
    "medical emergencies are waived on a case-by-case basis. To cancel or reschedule, "
    "call your clinic directly or use the link provided in your appointment confirmation "
    "message."
)

PAYMENT_INSURANCE = (
    "CareConnect Clinics accept cash, UPI (Google Pay, PhonePe, Paytm), all major credit "
    "and debit cards (Visa, Mastercard, RuPay, Amex), and leading Indian health insurance "
    "providers. For insurance claims, please carry your physical or digital insurance card, "
    "a valid government-issued photo ID, and any prior authorization letter issued by your "
    "insurer. Insurance claims are settled either via cashless (if your insurer is on our "
    "panel and authorization is approved) or reimbursement mode. Diagnostic and preventive "
    "health checkup packages are also covered by most insurers subject to policy terms."
)

DIAGNOSTIC_SERVICES = (
    "On-site diagnostic services available at all CareConnect Clinics include: complete "
    "blood counts and biochemistry panels, urine and stool analysis, ECG, X-ray, and "
    "ultrasound (USG) studies. MRI and CT scans are not performed on-site; however, we "
    "maintain empanelment with trusted partner imaging centers across Delhi, Noida, and "
    "Meerut and can provide a referral with priority appointment slots. Diagnostic reports "
    "are typically delivered within 4-24 hours depending on the test and are shared via "
    "email and SMS."
)

NEW_PATIENT_INFO = (
    "If you are visiting CareConnect Clinics for the first time, please carry a valid "
    "government-issued photo ID (Aadhaar, PAN, Passport, or Driving License). If you "
    "intend to use health insurance, also bring your insurance card and any prior "
    "authorization. Arriving 15 minutes before your requested appointment time helps "
    "complete registration smoothly. During your first visit you will be registered in "
    "our patient management system and provided with a unique Patient ID for all future "
    "visits."
)

EXISTING_PATIENT_INFO = (
    "Returning patients can request follow-up appointments, reschedule or cancel "
    "appointments, and request copies of their previous visit summaries by contacting "
    "any CareConnect clinic. Please have your Patient ID or registered phone number "
    "handy to help us retrieve your records quickly. Previous prescriptions and visit "
    "notes are stored in our clinic system and can be shared on request after verifying "
    "your identity for privacy reasons."
)

CONTACT_INFO = (
    "Central customer support: +91-11-4000-0000 (9 AM - 7 PM, Mon - Sat). "
    "Email: hello@careconnectclinics.example. For emergencies, please call your local "
    "emergency number (108 / 112 in India) or visit the nearest emergency department."
)

EMERGENCY_GUIDANCE = (
    "CareConnect Clinics are outpatient facilities and are not equipped to handle "
    "life-threatening emergencies. If you or someone near you is experiencing severe "
    "chest pain, difficulty breathing, loss of consciousness, heavy bleeding, sudden "
    "weakness or slurred speech suggestive of stroke, severe allergic reaction with "
    "swelling of the face or throat, major trauma, or any other condition that feels "
    "immediately life-threatening, please call 108 / 112 or go to the nearest "
    "hospital emergency department right away. Our front desk can help coordinate with "
    "nearby hospitals if you are already at a clinic."
)

LOCATIONS: list[Location] = [
    Location(
        name="Delhi",
        address="CareConnect Clinics, 221, Sector 12, Dwarka, New Delhi - 110075",
        phone="+91-11-4101-0200",
        email="delhi.clinic@careconnectclinics.example",
        hours_weekday="8:30 AM - 8:00 PM",
        hours_saturday="9:00 AM - 6:00 PM",
        hours_sunday="10:00 AM - 2:00 PM",
        emergency=False,
        amenities=[
            "On-site diagnostics (blood, ECG, X-ray, USG)",
            "Wheelchair access",
            "Pharmacy counter",
            "Kids' waiting area",
            "Free Wi-Fi",
        ],
    ),
    Location(
        name="Noida",
        address="CareConnect Clinics, C-56/11, Sector 62, Noida, Uttar Pradesh - 201309",
        phone="+91-120-422-3300",
        email="noida.clinic@careconnectclinics.example",
        hours_weekday="8:00 AM - 8:30 PM",
        hours_saturday="8:30 AM - 7:00 PM",
        hours_sunday="10:00 AM - 3:00 PM",
        emergency=False,
        amenities=[
            "On-site diagnostics (blood, ECG, X-ray, USG)",
            "Wheelchair access and ramp entry",
            "In-clinic pharmacy",
            "Kids' play corner",
            "Free parking for 2-wheelers and 4-wheelers",
        ],
    ),
    Location(
        name="Meerut",
        address="CareConnect Clinics, 14/4, Near Begum Bridge, Garh Rd, Meerut, Uttar Pradesh - 250001",
        phone="+91-121-450-9900",
        email="meerut.clinic@careconnectclinics.example",
        hours_weekday="8:30 AM - 7:30 PM",
        hours_saturday="9:00 AM - 5:30 PM",
        hours_sunday="Closed",
        emergency=False,
        amenities=[
            "On-site diagnostics (blood, ECG, X-ray, USG)",
            "Wheelchair access",
            "Pharmacy counter",
            "General waiting area with TV and drinking water",
        ],
    ),
]

DOCTORS: list[Doctor] = [
    # Delhi
    Doctor(name="Dr. Ananya Mehta", specialty="General Medicine", location="Delhi",
           qualifications="MBBS, MD (Internal Medicine)", experience_years=12,
           consultation_days="Mon - Sat"),
    Doctor(name="Dr. Rohit Khanna", specialty="Dermatology", location="Delhi",
           qualifications="MBBS, MD (Dermatology, Venereology & Leprosy)", experience_years=9,
           consultation_days="Mon, Wed, Fri"),
    Doctor(name="Dr. Priya Rastogi", specialty="Pediatrics", location="Delhi",
           qualifications="MBBS, DCH", experience_years=11,
           consultation_days="Tue, Thu, Sat"),
    Doctor(name="Dr. Sumit Arora", specialty="Orthopedics", location="Delhi",
           qualifications="MBBS, MS (Orthopedics)", experience_years=14,
           consultation_days="Mon, Wed, Sat"),
    Doctor(name="Dr. Sneha Bhatnagar", specialty="Gynecology", location="Delhi",
           qualifications="MBBS, MS (Obstetrics & Gynaecology)", experience_years=10,
           consultation_days="Tue, Thu, Fri"),
    Doctor(name="Dr. Karan Sabharwal", specialty="Dentistry", location="Delhi",
           qualifications="BDS, MDS (Conservative Dentistry & Endodontics)", experience_years=7,
           consultation_days="Mon - Sat"),

    # Noida
    Doctor(name="Dr. Aisha Kapoor", specialty="General Medicine", location="Noida",
           qualifications="MBBS, MD (General Medicine)", experience_years=10,
           consultation_days="Mon - Sat"),
    Doctor(name="Dr. Vikram Desai", specialty="Dermatology", location="Noida",
           qualifications="MBBS, DNB (Dermatology)", experience_years=8,
           consultation_days="Mon, Tue, Thu, Sat"),
    Doctor(name="Dr. Ritu Bansal", specialty="Pediatrics", location="Noida",
           qualifications="MBBS, MD (Pediatrics)", experience_years=13,
           consultation_days="Mon - Sat"),
    Doctor(name="Dr. Raghavendra Rao", specialty="Orthopedics", location="Noida",
           qualifications="MBBS, MS (Orthopaedics)", experience_years=16,
           consultation_days="Tue, Wed, Fri, Sat"),
    Doctor(name="Dr. Natasha Singh", specialty="Gynecology", location="Noida",
           qualifications="MBBS, DGO, DNB (Obstetrics & Gynaecology)", experience_years=11,
           consultation_days="Mon, Wed, Thu, Fri"),
    Doctor(name="Dr. Samarth Jain", specialty="Dentistry", location="Noida",
           qualifications="BDS, MDS (Prosthodontics)", experience_years=6,
           consultation_days="Mon - Sat"),

    # Meerut
    Doctor(name="Dr. Sanjay Agarwal", specialty="General Medicine", location="Meerut",
           qualifications="MBBS, MD (Internal Medicine)", experience_years=18,
           consultation_days="Mon - Sat"),
    Doctor(name="Dr. Monica Chaudhary", specialty="Dermatology", location="Meerut",
           qualifications="MBBS, MD (Skin & VD)", experience_years=10,
           consultation_days="Tue, Thu, Sat"),
    Doctor(name="Dr. Ramesh Kumar", specialty="Pediatrics", location="Meerut",
           qualifications="MBBS, MD (Paediatrics)", experience_years=15,
           consultation_days="Mon, Wed, Fri"),
    Doctor(name="Dr. Neeraj Tomar", specialty="Orthopedics", location="Meerut",
           qualifications="MBBS, MS (Orthopaedics)", experience_years=12,
           consultation_days="Mon, Wed, Sat"),
    Doctor(name="Dr. Shalini Goyal", specialty="Gynecology", location="Meerut",
           qualifications="MBBS, MS (Obstetrics & Gynaecology)", experience_years=9,
           consultation_days="Tue, Thu, Fri"),
    Doctor(name="Dr. Akash Verma", specialty="Dentistry", location="Meerut",
           qualifications="BDS, MDS (Oral & Maxillofacial Surgery)", experience_years=7,
           consultation_days="Mon - Sat"),
]

SERVICES: list[KBService] = [
    KBService(name="General Medicine", fee_inr=600, duration_min=20,
              description="Consultation for common adult illnesses, chronic condition management, fever, cough, diabetes, hypertension, and general health advice."),
    KBService(name="Dermatology", fee_inr=800, duration_min=25,
              description="Consultation for acne, eczema, psoriasis, hair loss, pigmentation, fungal infections, allergies, and general skin, hair and nail concerns."),
    KBService(name="Pediatrics", fee_inr=650, duration_min=20,
              description="Consultation for newborns, infants, children and adolescents up to 18 years of age, covering growth, immunization, and common childhood illnesses."),
    KBService(name="Orthopedics", fee_inr=900, duration_min=30,
              description="Consultation for bone, joint, muscle and spine concerns, sports injuries, fractures, back pain, and arthritis management."),
    KBService(name="Gynecology", fee_inr=900, duration_min=30,
              description="Consultation for women's health including menstrual concerns, pregnancy planning, antenatal care, PCOD/thyroid issues and routine checkups."),
    KBService(name="Dentistry", fee_inr=500, duration_min=20,
              description="Consultation for tooth pain, cavities, cleaning, whitening, crowns, root canal referrals, braces, and general oral health."),
    KBService(name="Diagnostics", fee_inr=0, duration_min=15,
              description="On-site sample collection for blood, urine, ECG, X-ray and ultrasound (USG). Fees vary by test; listed on the diagnostics price card at each clinic."),
    KBService(name="Preventive Health Checkups", fee_inr=0, duration_min=45,
              description="Packages starting from INR 1,999 for individuals and INR 3,499 for couples/families, including physician consultation and curated diagnostic panels."),
]

PREVENTIVE_PACKAGES = [
    {"name": "Basic Health Check", "price_inr": 1999,
     "includes": "CBC, RBS, Lipid Profile, Liver Profile, Kidney Profile, Urine Routine, ECG, Physician Consultation"},
    {"name": "Advanced Heart Check", "price_inr": 3299,
     "includes": "All of Basic + HbA1c, Thyroid Profile, ECG, 2D-Echo referral, Cardiologist Consultation"},
    {"name": "Family Package (2 Adults)", "price_inr": 3499,
     "includes": "Basic panel for 2 adults, Physician consultation for both, individual reports."},
    {"name": "Women's Wellness", "price_inr": 2799,
     "includes": "CBC, Thyroid Profile, HbA1c, Vitamin D & B12, Pap Smear (if indicated), Gynecologist Consultation"},
]


# ---------------------------------------------------------------------------
# KnowledgeBase class with typed lookup + search helpers
# ---------------------------------------------------------------------------

class KnowledgeBase:
    def __init__(self) -> None:
        self.overview = OVERVIEW
        self.locations = LOCATIONS
        self.doctors = DOCTORS
        self.services = SERVICES
        self.insurers = list(INSURER_NAMES)
        self.appointment_process = APPOINTMENT_PROCESS
        self.cancellation_policy = CANCELLATION_POLICY
        self.payment_and_insurance = PAYMENT_INSURANCE
        self.diagnostic_services = DIAGNOSTIC_SERVICES
        self.new_patient_info = NEW_PATIENT_INFO
        self.existing_patient_info = EXISTING_PATIENT_INFO
        self.contact_info = CONTACT_INFO
        self.emergency_guidance = EMERGENCY_GUIDANCE
        self.preventive_packages = PREVENTIVE_PACKAGES
        self._facts: Optional[list[tuple[str, str]]] = None

    # --- typed lookups -----------------------------------------------------
    def get_location(self, name: str) -> Optional[Location]:
        if not name:
            return None
        needle = name.strip().lower()
        for loc in self.locations:
            if loc.name.lower() == needle:
                return loc
        for loc in self.locations:
            if needle in loc.name.lower() or loc.name.lower() in needle:
                return loc
        return None

    def get_doctors(self, location: Optional[str] = None, specialty: Optional[str] = None) -> list[Doctor]:
        result: list[Doctor] = []
        for d in self.doctors:
            if location and location.lower() != d.location.lower():
                continue
            if specialty and specialty.lower() not in d.specialty.lower() and d.specialty.lower() not in specialty.lower():
                continue
            result.append(d)
        return result

    def get_fee(self, specialty: str) -> Optional[int]:
        if not specialty:
            return None
        needle = specialty.strip().lower()
        for s in self.services:
            if needle in s.name.lower() or s.name.lower() in needle:
                return s.fee_inr if s.fee_inr else None
        return None

    def get_insurers(self) -> list[str]:
        return list(self.insurers)

    def get_service(self, name: str) -> Optional[KBService]:
        if not name:
            return None
        needle = name.strip().lower()
        for s in self.services:
            if needle in s.name.lower() or s.name.lower() in needle:
                return s
        return None

    # --- RAG-style search --------------------------------------------------
    def _build_facts(self) -> list[tuple[str, str]]:
        if self._facts is not None:
            return self._facts
        facts: list[tuple[str, str]] = []

        def add(title: str, body: str) -> None:
            facts.append((title, body))

        add("Clinic Overview", self.overview)
        for loc in self.locations:
            add(
                f"Location: {loc.name}",
                (
                    f"Clinic name: CareConnect Clinics {loc.name}. "
                    f"Address: {loc.address}. Phone: {loc.phone}. Email: {loc.email}. "
                    f"Opening hours: Mon-Fri {loc.hours_weekday}, Saturday {loc.hours_saturday}, "
                    f"Sunday {loc.hours_sunday}. Amenities: {', '.join(loc.amenities)}."
                ),
            )
        for doc in self.doctors:
            add(
                f"Doctor: {doc.name}",
                (
                    f"{doc.name} is a {doc.specialty} specialist at CareConnect {doc.location}. "
                    f"Qualifications: {doc.qualifications}. Experience: {doc.experience_years} years. "
                    f"Languages spoken: {', '.join(doc.languages)}. Consultation days: {doc.consultation_days}."
                ),
            )
        for svc in self.services:
            fee_note = f"Consultation fee: INR {svc.fee_inr}." if svc.fee_inr else (
                "Fees vary by specific test or package; please ask staff for the current price card."
            )
            add(
                f"Service: {svc.name}",
                (
                    f"{svc.name} at CareConnect Clinics. {svc.description} "
                    f"Typical consultation duration: {svc.duration_min} minutes. {fee_note}"
                ),
            )
        for pkg in self.preventive_packages:
            add(
                f"Preventive Package: {pkg['name']}",
                (
                    f"{pkg['name']} costs INR {pkg['price_inr']}. Includes: {pkg['includes']}."
                ),
            )
        insurers_list = ", ".join(self.insurers)
        add("Insurance & Payment", f"{PAYMENT_INSURANCE} Panel insurers for cashless: {insurers_list}.")
        add("Appointment Process", self.appointment_process)
        add("Cancellation / Rescheduling Policy", self.cancellation_policy)
        add("Diagnostic Services", self.diagnostic_services)
        add("New Patient Information", self.new_patient_info)
        add("Existing Patient Information", self.existing_patient_info)
        add("Contact Information", self.contact_info)
        add("Emergency and Urgent Care Guidance", self.emergency_guidance)

        self._facts = facts
        return facts

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Return top-k facts by simple case-insensitive keyword overlap.

        Purely local; the LLM then synthesizes the answer strictly from the
        returned snippets so no hallucination of facts occurs.
        """
        if not query:
            return []
        tokens = {t for t in re.findall(r"[a-zA-Z0-9\u0900-\u097F]{2,}", query.lower())}
        # Expand location / specialty aliases so short queries match.
        for loc in LOCATION_NAMES:
            if loc.lower() in query.lower():
                tokens.add(loc.lower())
        for svc in SERVICE_NAMES:
            if svc.lower() in query.lower() or any(tok in svc.lower() for tok in tokens):
                tokens.add(svc.lower())
        scored: list[tuple[int, tuple[str, str]]] = []
        for fact in self._build_facts():
            title, body = fact
            haystack = f"{title} {body}".lower()
            score = sum(1 for tok in tokens if tok in haystack)
            # Boost location/service/exact-title matches slightly
            if any(loc.lower() in title.lower() for loc in LOCATION_NAMES) and any(
                loc.lower() in query.lower() for loc in LOCATION_NAMES
            ):
                score += 2
            if any(svc.lower() in title.lower() for svc in SERVICE_NAMES) and any(
                svc.lower() in query.lower() for svc in SERVICE_NAMES
            ):
                score += 2
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overview": self.overview,
            "locations": [loc.__dict__ for loc in self.locations],
            "doctors": [d.__dict__ for d in self.doctors],
            "services": [s.__dict__ for s in self.services],
            "insurers": self.insurers,
        }
