from models.career_predictor import predict_career_track

def get_career_recommendations(resume_data, aptitude_scores):
    from utils.feature_engineering import generate_feature_vector

    features = generate_feature_vector(resume_data, aptitude_scores)
    top_careers = predict_career_track(features)
    return top_careers
