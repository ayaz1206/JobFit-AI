from functools import lru_cache
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def embedder():
    # Cached model (downloaded on first run)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_embed_sim(resume_text: str, jd_text: str) -> float:
    # Use sentence embeddings for better semantic matching
    model = embedder()
    emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
    # cosine similarity of two normalized vectors = dot product
    return float((emb[0] * emb[1]).sum())


def keyword_ngrams(text: str, top_k=20):
    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=1)
    X = vec.fit_transform([text])
    items = list(zip(vec.get_feature_names_out(), X.toarray()[0]))
    return [k for k, _ in sorted(items, key=lambda kv: kv[1], reverse=True)[:top_k]]


def weighted_skill_overlap(resume_skills: List[str], jd_text: str, skill_list: List[str]) -> float:
    jd_lower = jd_text.lower()
    lines = [ln.strip() for ln in jd_lower.split("\n") if ln.strip()]
    weights = {}

    for sk in skill_list:
        s = sk.lower()
        if s in jd_lower:
            weight = 1.0
            for i, ln in enumerate(lines):
                if s in ln:
                    if i < 5:
                        weight = max(weight, 1.5)  # early emphasis
                    if ln.endswith(":"):
                        weight = max(weight, 1.3)
                    break
            weights[s] = weight

    if not weights:
        return 0.0

    total = sum(weights.values())
    matched = sum(w for s, w in weights.items() if s in set(resume_skills))
    return matched / total


def top_missing_keywords(resume_text: str, jd_text: str, k=10):
    jd_keys = keyword_ngrams(jd_text, top_k=30)
    res_keys = set(keyword_ngrams(resume_text, top_k=60))
    return [kw for kw in jd_keys if kw not in res_keys][:k]


def missing_from(jd_skills: List[str], resume_skills: List[str]) -> List[str]:
    rs = set(resume_skills)
    return [s for s in jd_skills if s not in rs]


def level_from_text(t: str) -> str:
    t = t.lower()
    for lvl in ["intern", "junior", "entry", "mid", "senior", "staff", "lead"]:
        if lvl in t:
            return lvl
    return "unknown"


def level_match(resume_text: str, jd_text: str) -> float:
    r, j = level_from_text(resume_text), level_from_text(jd_text)
    if j == "unknown":
        return 0.5
    return 1.0 if r == j else 0.3


def overall_score(s_skills: float, s_embed: float, s_kw: float, s_level: float) -> float:
    # Tuned weights
    return 100 * (0.5 * s_skills + 0.25 * s_embed + 0.15 * s_kw + 0.10 * s_level)