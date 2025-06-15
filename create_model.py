# run this once to create a model
from sklearn.ensemble import RandomForestClassifier
import pickle

# Fake training data
X = [
    [1, 0, 1, 2, 0.8, 0.7],  # Data Analyst
    [0, 1, 1, 1, 0.7, 0.5],  # Software Developer
    [1, 0, 2, 0, 0.6, 0.9]   # Project Manager
]
y = ["Data Analyst", "Software Developer", "Project Manager"]

clf = RandomForestClassifier()
clf.fit(X, y)

with open("models/career_predictor.pkl", "wb") as f:
    pickle.dump(clf, f)
