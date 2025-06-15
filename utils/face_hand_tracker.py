import cv2
import mediapipe as mp
import numpy as np

class EmotionTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.hands = mp.solutions.hands.Hands()
        self.total_frames = 0
        self.eye_contact_frames = 0
        self.smile_frames = 0
        self.hand_movement_scores = []

    def process_frame(self, frame):
        self.total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self.face_mesh.process(rgb)
        hand_result = self.hands.process(rgb)

        if face_result.multi_face_landmarks:
            self.eye_contact_frames += 1
            self.smile_frames += 1  # Dummy smile score

        if hand_result.multi_hand_landmarks:
            self.hand_movement_scores.append(1.0)
        else:
            self.hand_movement_scores.append(0.0)

        return frame

    def get_metrics(self):
        return {
            "eye_contact_ratio": self.eye_contact_frames / self.total_frames,
            "smile_ratio": self.smile_frames / self.total_frames,
            "hand_movement_score": sum(self.hand_movement_scores) / self.total_frames,
            "total_frames": float(self.total_frames)
        }

    def release(self):
        self.face_mesh.close()
        self.hands.close()
