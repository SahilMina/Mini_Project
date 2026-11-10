"""
Module 4: Deterministic Triage Dispatcher
Rule-based engine mapping predicted stress tiers (Low, Medium, High)
to deterministic CBT exercises or clinical referral handoff pathways.
"""
from typing import Dict, Any, List


class TriageDispatcher:
    def __init__(self):
        pass

    def dispatch(self, tier: str, confidence: float, stressors: List[str]) -> Dict[str, Any]:
        """
        Applies deterministic triage rules based on predicted stress tier.
        Guarantees clear separation between self-guided CBT care and clinical escalation.
        """
        tier_normalized = tier.strip().capitalize()
        if tier_normalized not in ["Low", "Medium", "High"]:
            tier_normalized = "Medium"

        if tier_normalized == "Low":
            return self._triage_low(confidence, stressors)
        elif tier_normalized == "Medium":
            return self._triage_medium(confidence, stressors)
        else:
            return self._triage_high(confidence, stressors)

    def _triage_low(self, confidence: float, stressors: List[str]) -> Dict[str, Any]:
        return {
            "tier": "Low",
            "tier_label": "Mild Academic Tension / Normal Pressure",
            "urgency": "Routine / Self-Care",
            "badge_color": "emerald",
            "summary": "Stress levels are within normal academic bounds. Regular study breaks and routine balance are recommended.",
            "requires_clinical_escalation": False,
            "cbt_exercises": [
                {
                    "id": "study_pacing",
                    "title": "Engineering Milestone Pacing",
                    "category": "Time & Study Management",
                    "description": "Break massive project submissions and syllabus into 45-minute sprint blocks with 10-minute disconnection.",
                    "steps": [
                        "Select one pending module or lab report.",
                        "Set timer for 45 minutes; mute campus group chats.",
                        "Stop strictly at 45 minutes and step away from laptop screen for 10 minutes."
                    ]
                },
                {
                    "id": "gratitude_journaling",
                    "title": "Daily Win Journaling",
                    "category": "Cognitive Reinforcement",
                    "description": "Shift focus away from perfectionism by registering 3 concrete technical milestones completed today.",
                    "steps": [
                        "Note 1 bug fixed, code written, or concept understood today.",
                        "Acknowledge peer or faculty support received.",
                        "Reaffirm that academic worth is not defined solely by semester GPA."
                    ]
                }
            ],
            "action_guidance": "Continue tracking daily reflections. No clinical appointment needed."
        }

    def _triage_medium(self, confidence: float, stressors: List[str]) -> Dict[str, Any]:
        stressor_str = ", ".join(stressors[:2]) if stressors else "academic workload"
        return {
            "tier": "Medium",
            "tier_label": "Moderate Academic Distress",
            "urgency": "Active Self-Intervention (CBT Guided)",
            "badge_color": "amber",
            "summary": f"Notable anxiety detected around {stressor_str}. Structured Cognitive Behavioral Therapy (CBT) exercises are recommended to prevent burnout.",
            "requires_clinical_escalation": False,
            "cbt_exercises": [
                {
                    "id": "breathing_478",
                    "title": "4-7-8 Parasympathetic Vagus Breathing",
                    "category": "Physiological Regulation",
                    "description": "Activates the parasympathetic nervous system to rapidly lower heart rate and cortisol prior to exams or presentations.",
                    "timing": {"inhale": 4, "hold": 7, "exhale": 8, "cycles": 4},
                    "interactive": True
                },
                {
                    "id": "cognitive_reframing",
                    "title": "CBT Cognitive Reframing (Thought Record)",
                    "category": "Cognitive Restructuring",
                    "description": "Dissect automatic negative thoughts (e.g., 'If I fail this lab, my career is over') into objective evidence.",
                    "interactive": True,
                    "fields": [
                        {"key": "ant", "label": "Automatic Negative Thought (ANT)", "placeholder": "What catastrophizing thought is echoing right now?"},
                        {"key": "distortion", "label": "Identified Cognitive Distortion", "options": ["Catastrophizing", "All-or-Nothing Thinking", "Fortune Telling", "Mind Reading", "Disqualifying the Positive"]},
                        {"key": "rational_response", "label": "Balanced Alternative Reality", "placeholder": "What would I tell a close batchmate facing this exact situation?"}
                    ]
                },
                {
                    "id": "grounding_54321",
                    "title": "5-4-3-2-1 Sensory Grounding",
                    "category": "Acute Grounding",
                    "description": "Brings attention back to the present moment during overwhelming panic before exams or placement interviews.",
                    "interactive": True,
                    "sensory_steps": [
                        "5 things you can see in the room",
                        "4 things you can physically feel (e.g. keyboard, chair)",
                        "3 sounds you can hear right now",
                        "2 things you can smell or scents around you",
                        "1 thing you like about your resilience"
                    ]
                },
                {
                    "id": "box_breathing",
                    "title": "Box Breathing (4-4-4-4)",
                    "category": "Physiological Regulation",
                    "description": "Equalized breathing technique used for acute focus before high-stakes exams or viva voce.",
                    "timing": {"inhale": 4, "hold_in": 4, "exhale": 4, "hold_out": 4, "cycles": 4},
                    "interactive": True
                }
            ],
            "action_guidance": "Perform at least one interactive breathing or reframing exercise now. Re-evaluate reflection afterwards."
        }

    def _triage_high(self, confidence: float, stressors: List[str]) -> Dict[str, Any]:
        return {
            "tier": "High",
            "tier_label": "Severe Distress / Crisis Escalation",
            "urgency": "Immediate Clinical Referral Handoff",
            "badge_color": "rose",
            "summary": "Deep distress and severe emotional burden detected. Standard self-care exercises are insufficient; an anonymous clinical referral token has been prepared for professional counselor handoff.",
            "requires_clinical_escalation": True,
            "referral_instructions": "Take the generated sealed referral pass to the university counseling office or contact student healthcare. Your private diary entries remain encrypted and are never shared.",
            "crisis_helplines": [
                {
                    "name": "Tele-MANAS (Govt of India 24x7 Mental Health Helpline)",
                    "phone": "14416",
                    "toll_free": "1800-891-4416",
                    "type": "24/7 National Multilingual Support"
                },
                {
                    "name": "KIRAN National Mental Health Helpline",
                    "phone": "1800-599-0019",
                    "type": "Ministry of Social Justice & Empowerment"
                },
                {
                    "name": "Central University of Karnataka (CUK) Health & Counseling Center",
                    "location": "Administrative Block / Health Centre, Kadaganchi, Kalaburagi",
                    "details": "Walk-in counseling hours: 9:30 AM – 5:30 PM (Mon-Fri)"
                }
            ],
            "action_guidance": "Generate and present the sealed Anonymous Referral Slip to the campus counselor."
        }
