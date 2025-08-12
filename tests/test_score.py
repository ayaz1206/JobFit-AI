from nlp.score import skill_overlap, overall_score

def test_overlap():
    assert abs(skill_overlap(["python","sql"], ["python","java"])) == 0.5

def test_scoring_bounds():
    s = overall_score(0.5,0.5,0.5,0.5)
    assert 0 <= s <= 100
