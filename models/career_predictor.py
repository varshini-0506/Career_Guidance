import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "career_predictor.pkl")

def load_career_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_career_track(features_dict):
    """
    Input: features_dict = {
        'python': 1,
        'java': 0,
        'degree_level': 1,
        'experience_years': 2,
        'logic_score': 0.8,
        'comm_score': 0.6
    }
    Output: list of top 3 careers with confidence
    """
    model = load_career_model()
    feature_order = ['python', 'java', 'degree_level', 'experience_years', 'logic_score', 'comm_score']
    input_vector = [features_dict[k] for k in feature_order]
    probas = model.predict_proba([input_vector])[0]
    classes = model.classes_

    top_3 = sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)[:3]
    return [{"career": c, "confidence": round(p * 100, 2)} for c, p in top_3]
