import io
import time
from datetime import datetime
from typing import List, Dict

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from nlp.parse import read_resume, normalize_text, detect_sections, extract_skills, extract_contact
from nlp.score import (
    embedder, cosine_embed_sim, weighted_skill_overlap, level_match,
    top_missing_keywords, missing_from, overall_score
)
from nlp.rewrite import tailored_rewrites

APP_NAME = "AI Resume Tailor"

st.set_page_config(page_title=f"{APP_NAME}", page_icon="🧵", layout="wide")
st.title("🧵 AI Resume Tailor")
st.caption("Upload your resume and paste a job description to get a match score, gaps, and recruiter-ready bullet rewrites. Files are processed in-memory and not stored.")

# Sidebar – settings
with st.sidebar:
    st.header("Settings")
    st.write("Model: sentence-transformers/all-MiniLM-L6-v2")
    st.write("Weights: skills 50%, similarity 25%, keywords 15%, level 10%")
    show_debug = st.checkbox("Show debug details", value=False)

# Inputs
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"], help="We parse text only. Avoid image-only PDFs for best results.")
with col2:
    jd_text = st.text_area("Paste the job description", height=260, placeholder="Paste full job description here …")

analyze = st.button("Tailor my resume ✨", use_container_width=True, type="primary")

if analyze:
    if not (resume_file and jd_text.strip()):
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    # Read & normalize
    t0 = time.time()
    resume_bytes = io.BytesIO(resume_file.read())
    try:
        resume_raw = read_resume(resume_bytes, resume_file.name)
    except Exception as e:
        st.error(f"Could not read resume: {e}")
        st.stop()

    resume_text = normalize_text(resume_raw)
    jd_text_norm = normalize_text(jd_text)

    # Load skills catalog
    skills_df = pd.read_csv("nlp/skills_catalog.csv")
    catalog = skills_df["skill"].astype(str).str.lower().tolist()

    # Basic extractions
    sections = detect_sections(resume_text)
    resume_skills = extract_skills(resume_text, catalog)
    jd_skills = extract_skills(jd_text_norm, catalog)
    contact = extract_contact(resume_text)

    # Scoring components
    s_skills = weighted_skill_overlap(resume_skills, jd_text_norm, catalog)
    s_embed = cosine_embed_sim(resume_text, jd_text_norm)
    missing_kw = top_missing_keywords(resume_text, jd_text_norm)
    s_kw = 1.0 - min(len(missing_kw), 10) / 10.0
    s_level = level_match(resume_text, jd_text_norm)
    score = overall_score(s_skills, s_embed, s_kw, s_level)

    # Tailored rewrites
    bullets_src = sections.get("experience", "") or sections.get("work experience", "")
    bullets = [b.strip("-• \t").strip() for b in bullets_src.split("\n") if len(b.strip()) > 0][:8]
    rewrites = tailored_rewrites(bullets, jd_text_norm, prefer_keywords=missing_kw)

    # ATS checks
    ats_checks = [
        {"name": "Contact info present", "pass": bool(contact.get("email") or contact.get("phone"))},
        {"name": "Has 'Experience' section", "pass": ("experience" in sections or "work experience" in sections)},
        {"name": "Has 'Education' section", "pass": ("education" in sections)},
        {"name": "Detectable skills present", "pass": len(resume_skills) >= 3},
        {"name": "Resume length reasonable", "pass": 200 <= len(resume_text) <= 20000},
    ]

    t1 = time.time()

    # Layout: Score + Breakdown
    st.subheader("Match Score")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall", f"{score:.1f}/100")
        st.progress(min(max(score/100, 0.01), 0.99))
    with c2:
        st.write({
            "skills_overlap (50%)": round(s_skills, 2),
            "semantic_similarity (25%)": round(s_embed, 2),
            "keyword_coverage (15%)": round(s_kw, 2),
            "level_match (10%)": round(s_level, 2)
        })
        st.caption(f"Processed in {t1 - t0:.2f}s on this machine.")

    # Tabs for details
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gaps & Suggestions", "Tailored Bullets", "ATS Check", "Detected Sections", "Share Badge"
    ])

    with tab1:
        st.markdown("### Missing Skills")
        miss_sk = missing_from(jd_skills, resume_skills)
        if miss_sk:
            st.write(miss_sk)
        else:
            st.success("Nice — no critical skill gaps detected from your catalog.")

        st.markdown("### Missing Keywords (from JD)")
        st.write(missing_kw if missing_kw else "Good coverage of JD keywords.")

    with tab2:
        st.markdown("### Tailored Bullet Suggestions")
        if bullets:
            for before, after in rewrites:
                st.write(f"**• Before:** {before}")
                st.write(f"**• After:**  {after}")
                st.divider()
        else:
            st.info("Couldn’t auto-detect bullets. Put work items as plaintext under an 'Experience' header.")

    with tab3:
        st.markdown("### ATS Sanity Checks")
        df = pd.DataFrame(ats_checks)
        st.dataframe(df)
        if show_debug:
            st.code(contact, language="json")

    with tab4:
        st.markdown("### Detected Sections (naive)")
        for k, v in sections.items():
            with st.expander(k.title(), expanded=False):
                st.write(v[:1000] + ("…" if len(v) > 1000 else ""))

    with tab5:
        st.markdown("### Shareable Match Badge")
        role = st.text_input("Role title (optional)", value="Machine Learning Engineer Intern")
        company = st.text_input("Company (optional)", value="Acme")
        if st.button("Generate Badge"):
            img = generate_badge(score, role, company)
            st.image(img, caption="Right-click → Save image")

    st.success("Analysis complete. You can tweak your resume using the suggestions above.")


def generate_badge(score: float, role: str, company: str) -> Image.Image:
    """Create a simple shareable PNG badge with score, role and company."""
    w, h = 900, 300
    img = Image.new("RGB", (w, h), (249, 250, 251))  # near-white
    d = ImageDraw.Draw(img)

    # Basic fonts (fallback to default if no system fonts available)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 28)
        big_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        big_font = ImageFont.load_default()

    # Text blocks
    d.text((40, 30), "AI Resume Tailor", font=title_font, fill=(31, 41, 55))
    d.text((40, 110), f"{role} @ {company}", font=body_font, fill=(55, 65, 81))

    # Score circle-like block
    score_box_w, score_box_h = 260, 180
    score_x, score_y = w - score_box_w - 40, 60
    d.rounded_rectangle([(score_x, score_y), (score_x + score_box_w, score_y + score_box_h)], radius=24, fill=(229, 231, 235))
    d.text((score_x + 30, score_y + 40), f"{score:.1f}", font=big_font, fill=(17, 24, 39))
    d.text((score_x + 40, score_y + 130), "/ 100", font=body_font, fill=(75, 85, 99))

    d.text((40, 200), f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", font=body_font, fill=(107, 114, 128))
    return img