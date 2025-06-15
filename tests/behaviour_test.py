import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.face_hand_tracker import EmotionTracker
from utils.emotion_model_loader import load_model, predict_emotion
from utils.feature_extractor import extract_features
from utils.visualizer import plot_metrics, plot_emotion_probabilities

# 🎥 Initialize tracker and webcam
tracker = EmotionTracker()
cap = cv2.VideoCapture(0)

print("🔴 Press 'q' to stop webcam and see results.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = tracker.process_frame(frame)
    cv2.imshow("🎥 Emotion & Engagement Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
tracker.release()

# 📊 Extract metrics
metrics = tracker.get_metrics()
print("\n📊 Extracted Metrics:")
for k, v in metrics.items():
    print(f" - {k}: {round(v, 3)}")

# 📈 Plot engagement metrics
plot_metrics(metrics)

# 🔍 Load emotion model and extract features
model = load_model()
features = extract_features(metrics)

# 🎯 Predict emotion
emotion, probs = predict_emotion(model, features)
print(f"\n🎯 Predicted Emotion: {emotion}")
print("📈 Class Probabilities:")
for label, prob in probs.items():
    print(f"   - {label}: {round(prob, 3)}")

# 📊 Show bar chart of emotion probabilities
plot_emotion_probabilities(probs)

# 🌟 Compute overall performance score
eye_score = metrics.get("eye_contact_ratio", 0)
smile_score = metrics.get("smile_ratio", 0)
hand_score = metrics.get("hand_movement_score", 0)

overall_score = (eye_score * 0.4 + smile_score * 0.3 + hand_score * 0.3) * 100
print(f"\n🌟 Your Overall Interview Performance Score: {round(overall_score, 2)} / 100")

# 💡 Suggestions
print("\n💡 Suggestions to Improve:")
suggestions_given = False

if eye_score < 0.6:
    print("- Improve eye contact by looking more steadily at the camera.")
    suggestions_given = True
if smile_score < 0.5:
    print("- Smile more naturally to convey friendliness.")
    suggestions_given = True
if hand_score < 0.3:
    print("- Use your hands more confidently when you talk.")
    suggestions_given = True

if not suggestions_given:
    print("✅ You're doing great! Keep it up 🎉")
elif overall_score < 60:
    print("📈 Focus on building more expressiveness and presence.")
elif overall_score > 85:
    print("🔥 You're nearly interview-ready. Great presence!")
else:
    print("👍 You're on the right track. Small tweaks will help polish your delivery.")