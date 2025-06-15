import torch
import torch.nn as nn

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

def load_model(path="../models/emotion_classifier.pt"):
    model = EmotionClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_emotion(model, input_list):
    input_tensor = torch.tensor([input_list]).float()

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()

    class_names = ['Confident', 'Stressed', 'Uncertain', 'Enthusiastic']
    emotion = class_names[predicted_idx]
    probs = dict(zip(class_names, probabilities))
    return emotion, probs
