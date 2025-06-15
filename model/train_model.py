import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

print("🚀 Training script started")

df = pd.read_csv("data/synthetic_emotion_dataset.csv")
X = df[["eye_contact_ratio", "smile_ratio", "hand_movement_score"]].values
y = LabelEncoder().fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = EmotionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model/emotion_classifier.pt")
print("✅ Model trained and saved as emotion_classifier.pt")
