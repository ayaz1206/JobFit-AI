import re
from io import BytesIO
from typing import Dict

from pdfminer.high_level import extract_text as pdf_text
from docx import Document
from rapidfuzz import fuzz

COMMON_SECTIONS = [
    "experience", "work experience", "projects", "education", "skills",
    "summary", "objective", "certifications", "publications"
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")


def read_resume(file_bytes: BytesIO, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return pdf_text(file_bytes)
    if name.endswith(".docx"):
        doc = Document(file_bytes)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type. Please upload PDF or DOCX.")


def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def detect_sections(text: str) -> Dict[str, str]:
    chunks = {}
    lower = text.lower()
    for sec in COMMON_SECTIONS:
        idx = lower.find(sec)
        if idx != -1:
            chunks[sec] = idx
    ordered = sorted(chunks.items(), key=lambda kv: kv[1])
    ranges = []
    for i, (sec, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        ranges.append((sec, text[start:end].strip()))
    return dict(ranges)


def extract_skills(text: str, skill_list):
    found = set()
    lower = text.lower()
    # exact match first
    for sk in skill_list:
        sk_low = sk.lower().strip()
        if not sk_low:
            continue
        if re.search(rf"\b{re.escape(sk_low)}\b", lower):
            found.add(sk_low)
        else:
            # fuzzy partial; keep threshold conservative to avoid false positives
            if fuzz.partial_ratio(sk_low, lower) >= 90:
                found.add(sk_low)
    return sorted(found)


def extract_contact(text: str):
    return {
        "email": EMAIL_RE.search(text).group(0) if EMAIL_RE.search(text) else "",
        "phone": PHONE_RE.search(text).group(0) if PHONE_RE.search(text) else "",
    }