from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.resume_parser import parse_resume
from utils.aptitude_analyzer import score_aptitude
from utils.feature_engineering import generate_feature_vector
from utils.career_recommender import get_career_recommendations
from utils.skill_gap_recommender import recommend_missing_skills
from utils.career_skill_mapping import career_required_skills
from utils.emotion_model_loader import load_model, predict_emotion

app = Flask(__name__)
CORS(app)  # Allow requests from tkinter frontend

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("resume")
    answers = request.form.getlist("answers")

    if not file or not answers:
        return jsonify({"error": "Resume file or answers missing"}), 400

    filepath = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    resume_data = parse_resume(filepath)
    aptitude_scores = score_aptitude(answers)
    features = generate_feature_vector(resume_data, aptitude_scores)
    career_results = get_career_recommendations(resume_data, aptitude_scores)
    top_career = career_results[0]["career"]  # e.g., "Software Developer"
    required_skills = career_required_skills.get(top_career, [])  # get skill list or empty list
    skill_gap_suggestions = recommend_missing_skills(resume_data["skills"], required_skills)

    return jsonify({
        "parsed_resume": resume_data,
        "aptitude_scores": aptitude_scores,
        "career_recommendations": career_results,
        "skill_gap": {
            "target_career": top_career,
            "suggestions": skill_gap_suggestions
        }
    })

@app.route("/analyze_behavior", methods=["POST"])
def analyze_behavior():
    try:
        # Get metrics and features from the request
        data = request.get_json()
        if not data or 'metrics' not in data or 'features' not in data:
            return jsonify({'error': 'Metrics or features missing in request'}), 400

        metrics = data['metrics']
        features = np.array(data['features'])

        # Load emotion model and predict emotion
        model = load_model()
        emotion, probs = predict_emotion(model, features)

        # Compute overall performance score
        eye_score = metrics.get("eye_contact_ratio", 0)
        smile_score = metrics.get("smile_ratio", 0)
        hand_score = metrics.get("hand_movement_score", 0)

        overall_score = (eye_score * 0.4 + smile_score * 0.3 + hand_score * 0.3) * 100

        # Generate suggestions
        suggestions = []
        suggestions_given = False

        if eye_score < 0.6:
            suggestions.append("Improve eye contact by looking more steadily at the camera.")
            suggestions_given = True
        if smile_score < 0.5:
            suggestions.append("Smile more naturally to convey friendliness.")
            suggestions_given = True
        if hand_score < 0.3:
            suggestions.append("Use your hands more confidently when you talk.")
            suggestions_given = True

        if not suggestions_given:
            suggestions.append("You're doing great! Keep it up 🎉")
        elif overall_score < 60:
            suggestions.append("Focus on building more expressiveness and presence.")
        elif overall_score > 85:
            suggestions.append("You're nearly interview-ready. Great presence!")
        else:
            suggestions.append("You're on the right track. Small tweaks will help polish your delivery.")

        # Prepare response
        result = {
            "metrics": {k: round(v, 3) for k, v in metrics.items()},
            "emotion": emotion,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "overall_score": round(overall_score, 2),
            "suggestions": suggestions
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Behavior analysis failed: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)