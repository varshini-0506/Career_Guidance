def extract_features(metrics):
    return [
        metrics.get("eye_contact_ratio", 0.0),
        metrics.get("smile_ratio", 0.0),
        metrics.get("hand_movement_score", 0.0)
    ]

def calculate_component_scores(metrics, predicted_emotion):
    eye = metrics.get("eye_contact_ratio", 0.0) * 10
    smile = metrics.get("smile_ratio", 0.0) * 10
    hand = metrics.get("hand_movement_score", 0.0) * 10
    face_score = (eye + smile) / 2

    emotion_boost = {
        'Confident': 3,
        'Enthusiastic': 2,
        'Uncertain': -2,
        'Stressed': -3
    }.get(predicted_emotion, 0)

    overall = (face_score + hand) / 2 + emotion_boost
    final_score = max(0, min(10, round(overall, 2)))

    return {
        "facial_expression_score": round(face_score, 2),
        "hand_gesture_score": round(hand, 2),
        "emotion_boost": emotion_boost,
        "final_score": final_score
    }