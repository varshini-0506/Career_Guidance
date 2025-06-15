import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.skill_gap_model import compute_match_percentage
from utils.skill_gap_recommender import recommend_missing_skills

# Simulated inputs
user_skills = ["Python", "Pandas", "Machine Learning"]
selected_role = "Data Scientist"

# Simulated job database (you can expand this)
job_requirements = {
    "data scientist": ["Python", "SQL", "Pandas", "Machine Learning", "Tableau", "Communication"],
    "web developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Git"],
    "ai engineer": ["Python", "PyTorch", "Deep Learning", "TensorFlow", "ML Ops"]
}

required_skills = job_requirements[selected_role.lower()]

# Skill match % and recommendations
match = compute_match_percentage(user_skills, required_skills)
missing = recommend_missing_skills(user_skills, required_skills)

# Output
print(f"🎯 Role Selected: {selected_role}")
print(f"✅ Match Percentage: {match}%")
print("🧠 Recommendations to Improve:")
for item in missing:
    print(f" - {item['skill']}")
    print(f"   📘 Course: {item['suggested_course']}")
    print(f"   🛠️ Tool: {item['suggested_tool']}")
