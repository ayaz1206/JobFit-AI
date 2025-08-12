import re
from typing import List, Tuple

from .score import keyword_ngrams

ACTION_VERBS = [
    "built", "developed", "designed", "implemented", "automated", "optimized",
    "reduced", "improved", "launched", "migrated", "deployed", "integrated",
    "led", "owned", "created", "analyzed", "trained", "tested"
]


def best_hint(jd_text: str) -> str:
    keys = keyword_ngrams(jd_text, top_k=10)
    return keys[0] if keys else "the role"


def normalize_bullet(b: str) -> str:
    return b.strip().rstrip('.')


def rewrite_once(bullet: str, jd_text: str, hint_kw: str) -> str:
    b = normalize_bullet(bullet)
    # Try to preserve numbers/metrics
    m = re.search(r"(\d+%?|\d+\+?)", b)
    metric = m.group(1) if m else None

    # Try to identify an action verb, otherwise add one
    words = b.split()
    if words and words[0].lower() in ACTION_VERBS:
        core = b
    else:
        core = f"{ACTION_VERBS[0].capitalize()} {b}"

    if metric:
        return f"{core} — achieving {metric} impact; aligned with {hint_kw}."
    return f"{core} — aligned with {hint_kw}."


def tailored_rewrites(bullets: List[str], jd_text: str, prefer_keywords: List[str] = None) -> List[Tuple[str, str]]:
    hint = prefer_keywords[0] if prefer_keywords else best_hint(jd_text)
    out = []
    for b in bullets:
        if len(b) < 5:
            continue
        out.append((b, rewrite_once(b, jd_text, hint)))
    return out