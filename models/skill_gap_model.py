# ml_models/skill_gap_model.py

def compute_match_percentage(user_skills, required_skills):
    user_skills = set(skill.lower() for skill in user_skills)
    required_skills = set(skill.lower() for skill in required_skills)

    matched_skills = user_skills.intersection(required_skills)
    match_percent = len(matched_skills) / len(required_skills) * 100

    return round(match_percent, 2)
