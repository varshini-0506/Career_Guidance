import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """
    Plots the soft skill metrics visually.
    """
    labels = ['Eye Contact', 'Smile', 'Hand Movement']
    values = [
        metrics['eye_contact_ratio'],
        metrics['smile_ratio'],
        metrics['hand_movement_score']
    ]

    colors = ['#58CC02', '#F4A900', '#379237']
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, 1.1)
    plt.title('Soft Skill Engagement Metrics')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.2, yval + 0.02, f'{yval:.2f}')
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt

def plot_emotion_probabilities(prob_dict):
    labels = list(prob_dict.keys())
    probabilities = list(prob_dict.values())
    colors = ['green' if prob == max(probabilities) else 'gray' for prob in probabilities]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, probabilities, color=colors)
    plt.title("Emotion Classification Probabilities")
    plt.ylim(0, 1)
    plt.ylabel("Probability")

    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{prob:.2f}", ha='center', va='bottom')

    plt.show()