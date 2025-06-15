# utils/skill_gap_recommender.py

def recommend_missing_skills(user_skills, required_skills):
    user_skills = set(skill.lower() for skill in user_skills)
    required_skills = set(skill.lower() for skill in required_skills)

    missing_skills = required_skills - user_skills

    recommendations = []
    for skill in missing_skills:
        recommendations.append({
            "skill": skill.title(),
            "suggested_course": f"https://www.coursera.org/search?query={skill}",
            "suggested_tool": f"https://www.google.com/search?q={skill}+tool",
        })

    return list(recommendations)
