import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.career_recommender import get_career_recommendations

sample_resume_data = {
    "skills": ["Python", "Java", "HTML"],
    "degree": "Bachelor of Engineering",
    "experience": 2
}

sample_aptitude_scores = {
    "logic": 0.8,
    "communication": 0.7
}

results = get_career_recommendations(sample_resume_data, sample_aptitude_scores)
print("Top career recommendations:")
for r in results:
    print(f"{r['career']}: {r['confidence']}%")
