from nlp.parse import normalize_text, extract_skills

def test_normalize():
    assert normalize_text(" a  b \n c ") == "a b c"

def test_extract_skills():
    skills = ["python","pandas","docker"]
    text = "Worked with Python and Pandas on ETL pipelines."
    found = extract_skills(text, skills)
    assert "python" in found and "pandas" in found
