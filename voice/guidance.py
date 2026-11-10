"""
Voice Guidance & Audio Cue Generator for CBT Exercises
Academic Stress & Clinical Referral Ecosystem
Central University of Karnataka, Kadaganchi, Kalaburagi
Author: Sahil Mina (24UCOMP047) | Guide: Prof. Srikantaiah K C

Generates spoken guidance scripts and synthesized audio tone sequences
for interactive CBT modules:
- 4-7-8 Parasympathetic Vagus Breathing
- 5-4-3-2-1 Sensory Grounding
- Cognitive Reframing Thought Pacing
"""
import io
import math
import wave
import struct
import numpy as np
from typing import Dict, Any, List


class VoiceGuidanceGenerator:
    """
    Synthesizes therapeutic audio prompts and harmonic acoustic guidance tones
    for stress reduction and parasympathetic nervous system activation.
    """

    GUIDANCE_SCRIPTS: Dict[str, Dict[str, Any]] = {
        "breathing_478": {
            "title": "4-7-8 Breathing Voice Guidance",
            "scripts": {
                "en-IN": [
                    {"phase": "prepare", "text": "Sit comfortably with your back straight. Relax your shoulders. Let us begin 4-7-8 breathing."},
                    {"phase": "inhale", "duration": 4, "text": "Inhale gently through your nose for four seconds. Feel your lungs expand."},
                    {"phase": "hold", "duration": 7, "text": "Hold your breath calmly for seven seconds. Release the tension in your jaw."},
                    {"phase": "exhale", "duration": 8, "text": "Exhale completely through your mouth for eight seconds with a soft whoosh sound."},
                    {"phase": "complete", "text": "Well done. Notice the calm in your nervous system. You are ready for your next study block."}
                ],
                "hi-IN": [
                    {"phase": "prepare", "text": "आराम से बैठें और कंधे ढीले छोड़ें। अब हम 4-7-8 प्राणायाम शुरू करेंगे।"},
                    {"phase": "inhale", "duration": 4, "text": "नाक से चार सेकंड तक गहरी सांस अंदर लें।"},
                    {"phase": "hold", "duration": 7, "text": "सात सेकंड तक सांस को रोक कर रखें, शांत रहें।"},
                    {"phase": "exhale", "duration": 8, "text": "मुंह से आठ सेकंड में धीरे-धीरे पूरी सांस बाहर निकालें।"}
                ],
                "kn-IN": [
                    {"phase": "prepare", "text": "ಶಾಂತವಾಗಿ ಕುಳಿತುಕೊಳ್ಳಿ. ಹೆಗಲುಗಳನ್ನು ಸಡಿಲಗೊಳಿಸಿ. 4-7-8 ಉಸಿರಾಟವನ್ನು ಪ್ರಾರಂಭಿಸೋಣ."},
                    {"phase": "inhale", "duration": 4, "text": "ಮೂಗಿನಿಂದ 4 ಸೆಕೆಂಡುಗಳ ಕಾಲ ದೀರ್ಘವಾಗಿ ಉಸಿರನ್ನು ಒಳಗೆ ತೆಗೆದುಕೊಳ್ಳಿ."},
                    {"phase": "hold", "duration": 7, "text": "7 ಸೆಕೆಂಡುಗಳ ಕಾಲ ಉಸಿರನ್ನು ಹಿಡಿದಿಟ್ಟುಕೊಳ್ಳಿ."},
                    {"phase": "exhale", "duration": 8, "text": "ಬಾಯಿಯಿಂದ 8 ಸೆಕೆಂಡುಗಳ ಕಾಲ ನಿಧಾನವಾಗಿ ಉಸಿರನ್ನು ಹೊರಬಿಡಿ."}
                ]
            }
        },
        "grounding_54321": {
            "title": "5-4-3-2-1 Sensory Grounding Voice Guidance",
            "scripts": {
                "en-IN": [
                    {"step": 5, "text": "Look around your campus room or lab. Acknowledge 5 things you can see right now."},
                    {"step": 4, "text": "Acknowledge 4 things you can physically feel. The chair supporting you, your laptop keys, your feet on the ground."},
                    {"step": 3, "text": "Listen attentively. Acknowledge 3 distinct sounds you can hear in the distance or nearby."},
                    {"step": 2, "text": "Acknowledge 2 things you can smell or scents around you."},
                    {"step": 1, "text": "Acknowledge 1 positive fact about yourself: you are capable of handling today's challenges."}
                ]
            }
        }
    }

    def get_guidance_metadata(self, exercise_id: str, language: str = "en-IN") -> Dict[str, Any]:
        """Returns structured spoken text cues for the requested exercise."""
        exercise = self.GUIDANCE_SCRIPTS.get(exercise_id, self.GUIDANCE_SCRIPTS["breathing_478"])
        scripts = exercise["scripts"].get(language, exercise["scripts"]["en-IN"])
        return {
            "exercise_id": exercise_id,
            "title": exercise["title"],
            "language": language,
            "steps": scripts
        }

    def generate_harmonic_pacer_wav(self, exercise_id: str = "breathing_478") -> bytes:
        """
        Synthesizes a soothing binaural-harmonic pacing WAV audio (432Hz calming tuning)
        with distinct auditory tones for Inhale (rise tone), Hold (soft drone), and Exhale (descending tone).
        Zero external dependencies, purely synthesized in Python via numpy!
        """
        sr = 16000
        # 1 cycle: 4s inhale (rising tone), 7s hold (gentle soft drone), 8s exhale (descending harmonic)
        t_inhale = 4.0
        t_hold = 7.0
        t_exhale = 8.0

        # Inhale: 220Hz -> 432Hz rising frequency
        samples_inhale = int(sr * t_inhale)
        t_i = np.linspace(0, t_inhale, samples_inhale, endpoint=False)
        f_i = np.linspace(220.0, 432.0, samples_inhale)
        phase_i = 2 * np.pi * np.cumsum(f_i) / sr
        env_i = np.sin(np.pi * t_i / t_inhale) * 0.4
        sig_inhale = np.sin(phase_i) * env_i

        # Hold: 432Hz gentle drone with warm harmonics
        samples_hold = int(sr * t_hold)
        t_h = np.linspace(0, t_hold, samples_hold, endpoint=False)
        sig_hold = (0.25 * np.sin(2 * np.pi * 432.0 * t_h) +
                    0.08 * np.sin(2 * np.pi * 216.0 * t_h))

        # Exhale: 432Hz -> 160Hz descending soothing tone
        samples_exhale = int(sr * t_exhale)
        t_e = np.linspace(0, t_exhale, samples_exhale, endpoint=False)
        f_e = np.linspace(432.0, 160.0, samples_exhale)
        phase_e = 2 * np.pi * np.cumsum(f_e) / sr
        env_e = np.sin(np.pi * t_e / t_exhale) * 0.45
        sig_exhale = np.sin(phase_e) * env_e

        # Concatenate 1 complete cycle
        full_audio = np.concatenate([sig_inhale, sig_hold, sig_exhale])
        # Convert to 16-bit PCM WAV
        full_audio = np.clip(full_audio, -1.0, 1.0)
        pcm_data = (full_audio * 32767).astype(np.int16)

        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_data.tobytes())

        return bio.getvalue()

