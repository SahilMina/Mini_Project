"""
Dataset Module: Dreaddit & Academic Distress Corpus Mapping
Provides curated academic and student reflections mapped to three stress tiers:
0: Low, 1: Medium, 2: High.

Calibrated for realistic campus reflections:
- Tier 0 (Low): Confident, organized, productive, healthy study pacing, mild fatigue with good coping.
- Tier 1 (Medium): Situational academic anxiety, pre-exam nervousness, viva tension, lab deadlines, backlog worry, sleep loss.
- Tier 2 (High): Severe clinical distress, depression, hopelessness, crisis thoughts, self-harm risk, breaking point.
"""
from typing import List, Tuple, Dict
import random

ACADEMIC_STRESS_SAMPLES: List[Tuple[str, int]] = [
    # =========================================================================
    # TIER 0: LOW STRESS (Calm, organized, productive, mild fatigue, good coping)
    # =========================================================================
    ("Finished my computer networks lab report early. Feeling confident for the viva tomorrow.", 0),
    ("Going to study with batchmates in the library for two hours and then play football.", 0),
    ("The DBMS syllabus is long, but I have a steady timetable and feel in control.", 0),
    ("Submitted the mini project synopsis on time. The guide suggested minor tweaks.", 0),
    ("A bit tired after back to back lectures, but getting 8 hours of sleep tonight.", 0),
    ("Attended the placement briefing today. Interesting companies are coming next month.", 0),
    ("Debugging this React component took a while, but it is working smoothly now.", 0),
    ("Group study session helped clarify the dynamic programming concepts.", 0),
    ("Balancing coursework well this semester. My internal marks are solid.", 0),
    ("Taking a walk around Kadaganchi campus to refresh my mind before evening study.", 0),
    ("Ready for the mid-term exams. Solved previous year question papers.", 0),
    ("Prof gave constructive feedback on my operating systems assignment.", 0),
    ("Feeling relaxed this weekend after submitting all pending lab records.", 0),
    ("Looking forward to the hackathon next week with my team.", 0),
    ("Consistent daily study is really paying off for data structures.", 0),
    ("Cooking with hostel friends this Sunday, good break from academic work.", 0),
    ("Got an 'A' grade on my software engineering presentation today.", 0),
    ("The lab assistant explained the pointer memory allocation clearly.", 0),
    ("Campus life is peaceful today, preparing for the upcoming semester.", 0),
    ("Feeling grateful for supportive peers and faculty guidance.", 0),
    ("Had a productive morning reviewing algorithms notes over hot tea.", 0),
    ("Completed all exercises in unit 2. Progress is steady and satisfying.", 0),
    ("Enjoyed the technical seminar by visiting faculty on cloud computing.", 0),
    ("Everything is organized on my calendar for the upcoming exams.", 0),
    ("Our team finished the wireframes ahead of the engineering milestone deadline.", 0),
    ("Studied for 3 hours, took a nice dinner break, feeling grounded.", 0),
    ("Clear understanding of the math syllabus now after solving past papers.", 0),
    ("Great peer discussion on database normalization today.", 0),
    ("Slept well last night and feeling energized for classes.", 0),
    ("Faculty approved our Phase 1 project architecture without issues.", 0),
    ("Finished reading the assigned chapters in computer architecture.", 0),
    ("Campus weather is pleasant, studying in the garden with quiet focus.", 0),
    ("Managed my time well between classes, gym, and project coding.", 0),
    ("Scored well on the surprise quiz in discrete mathematics.", 0),
    ("Feel prepared for the upcoming lab internal evaluations.", 0),
    ("Tomorrow is exam but i have revised the syllabus and feel calm.", 0),
    ("Tomorrow is the viva, notes are prepared and ready.", 0),
    ("Exam tomorrow, getting good rest tonight and feeling positive.", 0),
    ("Ready for tomorrow's test. Practiced all the numerical problems.", 0),
    ("Tomorrow is internal exam, feeling confident and steady.", 0),
    ("Steady progress on my exam preparation, no stress.", 0),
    ("Reviewed the core concepts for tomorrow's viva voce. Feeling good.", 0),
    ("All lab code is verified and pushed to GitHub. Ready for tomorrow.", 0),
    ("Feeling calm and focused for the semester evaluation.", 0),
    ("Studied comfortably with batchmates, feeling well-equipped for exams.", 0),
    ("Well prepared for the technical presentation tomorrow morning.", 0),
    ("Checked off all syllabus topics for tomorrow's assessment.", 0),
    ("Looking forward to finishing exams and starting the semester break.", 0),

    # =========================================================================
    # TIER 1: MEDIUM STRESS (Situational anxiety, pre-exam nervousness, deadlines)
    # =========================================================================
    ("tomorrow is exam i'm anxious", 1),
    ("tomorrow is exam i am anxious", 1),
    ("tomorrow is my exam feeling anxious", 1),
    ("feeling anxious about the exam tomorrow", 1),
    ("tomorrow is exam and i am so nervous", 1),
    ("anxious about tomorrow's viva voce", 1),
    ("exam tomorrow feeling stressed out", 1),
    ("nervous about internal exam tomorrow", 1),
    ("tomorrow is lab exam feeling really anxious", 1),
    ("a bit anxious about the viva tomorrow", 1),
    ("worried about tomorrow's test", 1),
    ("having exam anxiety right now", 1),
    ("stressed about compiler design exam tomorrow", 1),
    ("nervous about placement drive tomorrow", 1),
    ("scared about failing tomorrow's paper", 1),
    ("anxious about my CGPA and upcoming tests", 1),
    ("so anxious about viva tomorrow", 1),
    ("feeling stressed about tomorrow's deadline", 1),
    ("anxious about lab records submission", 1),
    ("tension about viva tomorrow", 1),
    ("feeling anxious about results coming out", 1),
    ("afraid of failing the test tomorrow", 1),
    ("butterflies in my stomach before the exam tomorrow", 1),
    ("stressed out about the test tomorrow morning", 1),
    ("I have three lab exams in two days and I haven't even finished the records. Panic is setting in.", 1),
    ("My CGPA dropped below 7.5 and now I might be debarred from the top placement companies.", 1),
    ("Staying up until 4 AM cramming for compiler design. My head is throbbing and I feel sick.", 1),
    ("The project deadline is tonight and our code is throwing random segmentation faults.", 1),
    ("I am terrified of failing the viva voce tomorrow. The professor is known to be very strict.", 1),
    ("All my friends got placed in early rounds and I keep getting rejected in the first round.", 1),
    ("I cannot sleep at all thinking about the semester exam schedule. My chest feels tight.", 1),
    ("Attendance shortage in two subjects. If they don't allow condonation I will get backlogs.", 1),
    ("Overwhelmed by the sheer volume of assignments due this Monday. Idk where to start.", 1),
    ("My parents expect a 9+ CGPA and high LPA package, the pressure is suffocating me.", 1),
    ("Constant headache and nausea from staring at code all night. Exhausted af.", 1),
    ("I bombed my campus placement technical interview today. Felt humiliated and anxious.", 1),
    ("Mini project guide rejected our synopsis draft. We have to redo the whole pipeline in 3 days.", 1),
    ("Dreading tomorrow's lab viva. My mind goes completely blank whenever prof asks questions.", 1),
    ("I haven't slept properly in two weeks. Surviving on instant coffee and adrenaline.", 1),
    ("The algorithm concepts just refuse to click in my head. I feel so far behind my classmates.", 1),
    ("Arrears from 3rd semester are piling up. I don't know how I will clear them with 5th sem load.", 1),
    ("Stressed out about mock interviews and aptitude rounds. Heart beats fast whenever I think of it.", 1),
    ("So much campus politics and hostel drama making it impossible to concentrate on studies.", 1),
    ("My hands are shaking while typing this code. The submission portal closes in 20 minutes.", 1),
    ("Worrying constantly about whether I will clear the placement cutoffs.", 1),
    ("Another late night coding session, my eyes burn and anxiety won't let me relax.", 1),
    ("Stressed out because our group partner abandoned their share of the project code.", 1),
    ("Feel anxious every time faculty mentions the upcoming external lab evaluation.", 1),
    ("Feeling restless and nervous about my academic standing and future career.", 1),
    ("Procrastinated all week and now facing an impossible mountain of syllabus.", 1),
    ("My heart is racing because the aptitude test results are coming out in an hour.", 1),
    ("Afraid I will freeze during the technical interview presentation tomorrow.", 1),
    ("Too many lab records to write by hand before Friday. Wrist hurts and stressed.", 1),
    ("Worrying non-stop about placement rejection letters.", 1),
    ("Scared that my low test score will ruin my semester GPA permanently.", 1),
    ("Cannot focus on lectures because the fear of backlogs keeps spinning in my head.", 1),
    ("Nervous breakdown over this compiler code not linking properly.", 1),
    ("Stressed about hostel curfew and having no quiet place to study for exams.", 1),
    ("So much anxiety about whether I will ever get a software engineering job.", 1),
    ("Feeling butterflies and panic before tomorrow's programming practicals.", 1),
    ("Anxious that I won't finish the operating systems assignment in time.", 1),
    ("Can't stop overthinking about my upcoming project viva voce.", 1),
    ("Feeling nervous about the surprise quiz tomorrow morning.", 1),
    ("Stressed about lab internal marks, need a high score to pass.", 1),

    # =========================================================================
    # TIER 2: HIGH STRESS (Severe distress, clinical crisis, hopelessness)
    # =========================================================================
    ("I cannot take this anymore. Everything is falling apart and I want all of this pain to just end.", 2),
    ("I feel completely broken, hopeless and empty. No matter what I do, I am a burden to everyone.", 2),
    ("I have been crying alone in my hostel room for days. I can't eat or get out of bed.", 2),
    ("There is no way out of this nightmare. I have failed everyone and see no reason to live.", 2),
    ("Severe panic attack in the exam hall today. I couldn't breathe and felt like dying on the spot.", 2),
    ("I hate myself so much. The thoughts in my head won't stop and I am terrified I will do something drastic.", 2),
    ("Everything is dark and hopeless. I don't care about placements, exams, or my future anymore. I just want to disappear.", 2),
    ("I feel completely detached from reality. Nobody would notice or care if I wasn't here tomorrow.", 2),
    ("I am suffocating under this agonizing mental pressure. I have no one to talk to and feel totally abandoned.", 2),
    ("Having severe crisis thoughts every single night. The psychological agony is unbearable.", 2),
    ("I can't stop thinking about ending it all. The pressure has completely destroyed my mental state.", 2),
    ("Extreme emotional numbness and terror. I cannot function at all and feel completely shattered.", 2),
    ("I want to give up on everything. Life feels pointless and the pain never goes away.", 2),
    ("I am in deep crisis. I don't feel safe with my own thoughts anymore. Please help me.", 2),
    ("Crying uncontrollably and hyperventilating. I feel so utterly trapped and desolate.", 2),
    ("I feel like I'm drowning in despair and there is no lifeline left for me.", 2),
    ("I can't take this mental torment another day. The suicidal thoughts are getting louder.", 2),
    ("Completely paralyzed by agony and self-hatred. I have reached my absolute breaking point.", 2),
    ("I feel so worthless and unloved. Wishing I would just never wake up from sleep.", 2),
    ("Crushed by utter despair. Nothing matters anymore and I cannot bear to exist like this.", 2),
    ("Terrified of what I might do to myself tonight. The darkness in my mind is overwhelming.", 2),
    ("Total collapse of hope. I feel broken beyond repair and cannot see any future ahead.", 2),
    ("Crying in the bathroom so my roommates don't hear. I can't breathe and want the pain to stop.", 2),
    ("I have lost all will to survive this semester. The emotional pain is crushing my chest.", 2),
    ("Desperate for help but drowning in hopelessness and acute psychological suffering.", 2),
    ("I don't belong anywhere. I feel like a ghost and can't find any reason to keep going.", 2),
    ("The panic and despair are suffocating me. I feel entirely broken and helpless.", 2),
    ("I am terrified I am losing my mind. The unbearable agony won't stop.", 2),
    ("Everything hurts inside. I feel like an utter failure who has no right to live.", 2),
    ("Please make this agonizing mental suffering stop. I am at the edge of the abyss.", 2),
    ("I want to kill myself, the academic pressure and loneliness are too much to survive.", 2),
    ("Thinking about ending my life tonight, I have completely given up on hope.", 2),
    ("Intense suicidal ideation and depression. I cannot handle this existence anymore.", 2),
    ("I feel worthless, empty, and suicidal. I am at my absolute breaking point.", 2),
    ("No hope left inside me. I want all of this suffering and pain to end permanently.", 2)
]

TIER_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def get_dataset() -> Tuple[List[str], List[int]]:
    """Returns texts and corresponding tier labels (0, 1, 2)."""
    texts = [item[0] for item in ACADEMIC_STRESS_SAMPLES]
    labels = [item[1] for item in ACADEMIC_STRESS_SAMPLES]
    return texts, labels


def get_train_test_split(test_ratio: float = 0.20, seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Generates stratified train/test partitions for fine-tuning and evaluation."""
    random.seed(seed)
    by_tier: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for text, label in ACADEMIC_STRESS_SAMPLES:
        by_tier[label].append(text)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for label, texts in by_tier.items():
        shuffled = texts[:]
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_texts.extend(shuffled[:split_idx])
        train_labels.extend([label] * len(shuffled[:split_idx]))
        test_texts.extend(shuffled[split_idx:])
        test_labels.extend([label] * len(shuffled[split_idx:]))

    return train_texts, train_labels, test_texts, test_labels
