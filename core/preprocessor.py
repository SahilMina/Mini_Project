"""
Module 1: Text Ingestion and Preprocessing
Handles campus slang normalization, contractions expansion, academic stressor tagging,
and tokenization with padding up to 128 tokens.
"""
import re
from typing import Dict, List, Tuple, Any

# Domain-specific campus slang & academic abbreviation lexicon
CAMPUS_SLANG_MAP = {
    r"\bcgpa?\b": "cumulative grade point average",
    r"\bsgpa?\b": "semester grade point average",
    r"\barrears?\b": "failed subjects backlogs",
    r"\bbacklog(s)?\b": "failed academic backlog",
    r"\bkt\b": "allowed to keep term academic failure",
    r"\byear[- ]?back\b": "academic year back setback",
    r"\bviva\b": "oral examination viva voce",
    r"\bsem\b": "semester",
    r"\bmidsem(s)?\b": "mid semester exam",
    r"\bendsem(s)?\b": "end semester final exam",
    r"\binternals?\b": "internal assessment exam",
    r"\bprofs?\b": "professor",
    r"\bhod\b": "head of department",
    r"\bdsa\b": "data structures and algorithms exam",
    r"\bdbms\b": "database management systems exam",
    r"\bos\b": "operating systems exam",
    r"\bcn\b": "computer networks exam",
    r"\bppo\b": "pre placement offer",
    r"\bplacements?\b": "campus placement job drive",
    r"\bunplaced\b": "unplaced in campus job interviews",
    r"\boff[- ]?campus\b": "off campus job search",
    r"\bshortage\b": "attendance shortage debarment",
    r"\bfml\b": "overwhelmed hopeless distress",
    r"\bidk\b": "i do not know",
    r"\brn\b": "right now",
    r"\btbh\b": "to be honest",
    r"\bfr\b": "for real",
    r"\baf\b": "extremely",
    r"\bcramming\b": "last minute studying without sleep",
    r"\ball[- ]?nighter\b": "staying awake all night without sleep",
    r"\bdebarred?\b": "disqualified from exams due to low attendance",
    r"\bgpa\b": "grade point average",
    r"\blab records?\b": "laboratory practical record submission",
    r"\bsynopsis\b": "mini project synopsis review"
}

# Academic stressor topic taxonomies
ACADEMIC_STRESSOR_PATTERNS = {
    "Exam & Viva Anxiety": [
        r"exams?", r"tests?", r"vivas?", r"quiz(zes)?", r"practicals?", r"lab exams?", r"marks?", r"pass(ing)?", r"fail(ing|ed)?", r"midsems?", r"endsems?", r"internals?"
    ],
    "Placement & Career Pressure": [
        r"placements?", r"packages?", r"lpa", r"jobs?", r"interviews?", r"unplaced", r"resumes?", r"reject(ed|ion)?", r"offers?", r"careers?", r"ppo", r"aptitude"
    ],
    "Academic Backlog & Arrears": [
        r"backlogs?", r"arrears?", r"kt", r"year back", r"debarred?", r"attendance shortage", r"clearing subjects"
    ],
    "Project & Assignment Deadlines": [
        r"deadlines?", r"submissions?", r"projects?", r"mini projects?", r"synopsis", r"reports?", r"codes?", r"assignments?", r"reviews?"
    ],
    "Grade & CGPA Apprehension": [
        r"cgpa?", r"sgpa?", r"gpa", r"grades?", r"pointers?", r"percentages?", r"ranks?", r"scores?"
    ],
    "Sleep Deprivation & Burnout": [
        r"sleep(ing|less)?", r"tired", r"exhaust(ed|ion)", r"insomnia", r"all[- ]?nighter", r"burnout", r"drained", r"headaches?", r"breakdowns?", r"crying", r"fatigue"
    ],
    "Family & Financial Guilt": [
        r"parents?", r"family", r"loans?", r"father", r"mother", r"financial", r"expectations?", r"burden", r"fees?", r"middle class", r"disappointing"
    ],
    "Social & Campus Restrictions": [
        r"hostels?", r"lonel(y|iness)", r"isolated?", r"curfews?", r"wardens?", r"friends?", r"homesick", r"campus restrictions?", r"ragging", r"alienated"
    ]
}

CONTRACTIONS = {
    r"can't": "cannot",
    r"won't": "will not",
    r"n't": " not",
    r"'re": " are",
    r"'s": " is",
    r"'d": " would",
    r"'ll": " will",
    r"'ve": " have",
    r"'m": " am"
}

MAX_TOKENS = 128


class TextPreprocessor:
    def __init__(self, max_length: int = MAX_TOKENS):
        self.max_length = max_length
        # Built-in lightweight vocabulary mapping for deterministic edge tokenization
        self._vocab = self._build_subword_vocab()

    def _build_subword_vocab(self) -> Dict[str, int]:
        """Constructs a deterministic subword vocabulary dictionary."""
        base_words = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "the", "i", "and", "to", "my", "a", "is", "in", "am", "feel", "stressed",
            "anxious", "exam", "exams", "placement", "deadlines", "assignment", "failing",
            "sleep", "grades", "cgpa", "project", "lab", "viva", "cannot", "tired",
            "depressed", "hopeless", "panic", "suicide", "hurt", "pressure", "campus",
            "hostel", "career", "study", "studying", "backlog", "future", "family"
        ]
        return {word: idx for idx, word in enumerate(base_words)}

    def normalize(self, text: str) -> str:
        """Cleans, expands contractions, and normalizes campus slang."""
        if not text:
            return ""

        clean = text.strip()

        # Expand common contractions
        for pattern, replacement in CONTRACTIONS.items():
            clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)

        # Expand campus slang
        for pattern, replacement in CAMPUS_SLANG_MAP.items():
            clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)

        # Normalize irregular punctuation and excessive spaces
        clean = re.sub(r"[!]{2,}", " !", clean)
        clean = re.sub(r"[?]{2,}", " ?", clean)
        clean = re.sub(r"[.]{3,}", " ... ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        return clean

    def extract_stressors(self, text: str) -> List[str]:
        """Identifies specific academic stress drivers present in student reflection."""
        identified = []
        lower_text = text.lower()
        for category, patterns in ACADEMIC_STRESSOR_PATTERNS.items():
            for pat in patterns:
                if re.search(r"\b" + pat + r"\b", lower_text):
                    identified.append(category)
                    break
        return identified or ["General Academic Tension"]

    def tokenize_and_pad(self, normalized_text: str) -> Dict[str, Any]:
        """
        Tokenizes normalized text and pads/truncates to max_length (128).
        Returns token IDs, attention masks, and metadata.
        """
        words = re.findall(r"\b\w+\b|[^\w\s]", normalized_text.lower())
        
        # [CLS] + tokens + [SEP]
        token_ids = [self._vocab.get("[CLS]", 2)]
        for w in words:
            if len(token_ids) >= self.max_length - 1:
                break
            tid = self._vocab.get(w, self._vocab.get("[UNK]", 1))
            token_ids.append(tid)
        token_ids.append(self._vocab.get("[SEP]", 3))

        raw_length = len(token_ids)
        attention_mask = [1] * raw_length

        # Pad to max_length
        pad_id = self._vocab.get("[PAD]", 0)
        padding_needed = self.max_length - raw_length
        if padding_needed > 0:
            token_ids.extend([pad_id] * padding_needed)
            attention_mask.extend([0] * padding_needed)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "token_count": min(raw_length, self.max_length),
            "max_length": self.max_length,
            "is_truncated": len(words) > (self.max_length - 2)
        }

    def process(self, raw_text: str) -> Dict[str, Any]:
        """Complete ingestion pipeline: normalize, extract stressors, and tokenize."""
        normalized = self.normalize(raw_text)
        stressors = self.extract_stressors(raw_text + " " + normalized)
        token_data = self.tokenize_and_pad(normalized)

        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "stressors": stressors,
            "tokens": token_data,
            "word_count": len(raw_text.split()),
            "char_count": len(raw_text)
        }
