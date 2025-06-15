import re

def generate_feature_vector(resume_data, aptitude_scores):
    skills_lower = [s.lower() for s in resume_data.get("skills", [])]
    
    # Degree level: 1 for bachelor, 2 for master/others
    degree_level = 0
    if resume_data.get("degree"):
        for degree in resume_data["degree"]:
            if "bachelor" in degree or "b.e" in degree or "b.tech" in degree or "b.sc" in degree:
                degree_level = 1
                break
            elif "master" in degree or "m.e" in degree or "m.tech" in degree or "m.sc" in degree:
                degree_level = 2
                break

    # Handle experience string like "3 years" or "Not specified"
    experience = resume_data.get("experience", "0")
    if isinstance(experience, str):
        match = re.search(r"\d+", experience)
        experience_years = int(match.group()) if match else 0
    else:
        experience_years = int(experience)

    feature_vector = {
        "python": int("python" in skills_lower),
        "java": int("java" in skills_lower),
        "degree_level": degree_level,
        "experience_years": experience_years,
        "logic_score": aptitude_scores.get("logic", 0),
        "comm_score": aptitude_scores.get("communication", 0)
    }
    return feature_vector


