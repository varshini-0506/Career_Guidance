import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import os
from tkinter import ttk
import cv2
import time
from PIL import Image, ImageTk
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Debug: Print the current working directory and script location
print("Current working directory:", os.getcwd())
print("Script location (app.py):", os.path.abspath(__file__))

# Add the parent directory of frontend/ to sys.path (which should contain utils/)
script_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

# Debug: Print sys.path after modification
print("sys.path after modification:", sys.path)
print("Parent directory added to sys.path:", parent_dir)

# Verify that utils exists in the parent directory
utils_path = os.path.join(parent_dir, 'utils')
print("Looking for utils at:", utils_path)
if not os.path.exists(utils_path):
    raise FileNotFoundError(f"'utils' directory not found at {utils_path}. Please check your project structure.")
else:
    print(f"Found 'utils' directory at {utils_path}")

# Check for __init__.py in utils/
init_path = os.path.join(utils_path, '__init__.py')
if not os.path.exists(init_path):
    raise FileNotFoundError(f"'__init__.py' not found in {utils_path}. This file is required for 'utils' to be treated as a package.")

# Attempt to import from utils
try:
    from utils.face_hand_tracker import EmotionTracker
    from utils.feature_extractor import extract_features
except ImportError as e:
    print(f"ImportError: {str(e)}")
    raise

API_URL = "http://127.0.0.1:5000"

class CareerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Career Guidance App")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f8ff")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'), background='#ff4500', foreground='white')
        self.style.configure('TLabel', background='#f0f8ff')
        
        # Define color scheme
        self.colors = {
            'bg': '#f0f8ff',  # Light blue background
            'card_bg': '#ffffff',  # White card background
            'heading': '#4682b4',  # Dark blue for headings
            'text': '#333333',  # Dark gray for text
            'accent': '#ff4500'  # Orange for accents
        }
        
        self.username = ""
        self.create_login_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_nav_bar(self):
        nav_frame = tk.Frame(self.root, bg=self.colors['heading'], height=50)
        nav_frame.pack(fill='x', side='top')
        
        tk.Button(nav_frame, text="Back to Home", command=self.show_dashboard, 
                 bg=self.colors['accent'], fg="white", font=('Helvetica', 10, 'bold'),
                 relief='flat', bd=0).pack(side='left', padx=10, pady=5)

    def create_login_screen(self):
        self.clear_screen()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        login_card = tk.Frame(main_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        login_card.pack(pady=50, padx=50, fill='both')
        login_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(login_card, text="Career Guidance", font=("Helvetica", 24, "bold"), 
                bg=self.colors['card_bg'], fg=self.colors['heading']).pack(pady=30)
        
        tk.Label(login_card, text="Username:", font=("Helvetica", 12), 
                bg=self.colors['card_bg'], fg=self.colors['text']).pack(pady=5)
        self.username_entry = ttk.Entry(login_card, font=("Helvetica", 12), width=30)
        self.username_entry.pack(pady=5)
        
        ttk.Button(login_card, text="Login", command=self.login, 
                  style='TButton').pack(pady=20)

    def login(self):
        self.username = self.username_entry.get()
        if not self.username:
            messagebox.showerror("Error", "Please enter a username")
            return
        self.show_dashboard()

    def show_dashboard(self):
        self.clear_screen()
        self.create_nav_bar()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        tk.Label(main_frame, text=f"Welcome {self.username}!", 
                font=("Helvetica", 20, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=20)
        
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        button_frame.pack(pady=30)
        
        ttk.Button(button_frame, text="Upload Resume", 
                  command=self.show_resume_upload,
                  style='TButton').pack(pady=10, padx=20, fill='x')
        
        ttk.Button(button_frame, text="Take Aptitude Test", 
                  command=self.show_aptitude_test,
                  style='TButton').pack(pady=10, padx=20, fill='x')

        ttk.Button(button_frame, text="Start Behavior Test", 
                  command=self.start_behavior_test,
                  style='TButton').pack(pady=10, padx=20, fill='x')

    def show_resume_upload(self):
        self.clear_screen()
        self.create_nav_bar()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        tk.Label(main_frame, text="Upload Your Resume", 
                font=("Helvetica", 18, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=20)
        
        upload_card = tk.Frame(main_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        upload_card.pack(pady=20, padx=50, fill='both')
        upload_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        ttk.Button(upload_card, text="Select Resume (PDF)", 
                  command=self.upload_resume,
                  style='TButton').pack(pady=30)

    def show_aptitude_test(self):
        self.clear_screen()
        self.create_nav_bar()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        tk.Label(main_frame, text="Aptitude Test", 
                font=("Helvetica", 18, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=20)
        
        canvas = tk.Canvas(main_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        question_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        question_card.pack(pady=10, padx=20, fill='both')
        question_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        self.questions = [
            "Do you enjoy solving logical problems?",
            "Are you comfortable communicating your ideas clearly?",
            "Do you like working with data and analysis?",
            "Are you interested in leading projects or teams?",
            "Do you enjoy coding and building software applications?",
            "Are you familiar with basic programming concepts?",
            "Can you explain technical concepts to non-technical people?"
        ]
        
        self.answers = []
        for i, q in enumerate(self.questions):
            tk.Label(question_card, text=f"{i+1}. {q}", bg=self.colors['card_bg'], 
                    font=("Helvetica", 11), fg=self.colors['text'], wraplength=600, 
                    justify='left').pack(anchor='w', padx=10, pady=5)
            entry = ttk.Entry(question_card, width=60)
            entry.pack(pady=5, padx=10)
            self.answers.append(entry)
        
        ttk.Button(question_card, text="Submit", command=self.submit_all, 
                  style='TButton').pack(pady=20)

    def start_behavior_test(self):
        self.clear_screen()
        self.create_nav_bar()

        self.main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        tk.Label(self.main_frame, text="Behavior Analysis", 
                font=("Helvetica", 18, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=10)

        tk.Label(self.main_frame, text="Analyzing your behavior... Click 'Stop Analysis' to finish.", 
                font=("Helvetica", 12), bg=self.colors['bg'], fg=self.colors['text']).pack(pady=5)

        # Create a label to display the webcam feed
        self.video_label = tk.Label(self.main_frame, bg=self.colors['bg'])
        self.video_label.pack(pady=10)

        # Create a "Stop Analysis" button, initially disabled
        self.stop_button = ttk.Button(self.main_frame, text="Stop Analysis", 
                                     command=self.stop_and_process, 
                                     style='TButton', state='disabled')
        self.stop_button.pack(pady=10)

        # Initialize webcam and tracker
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access webcam")
                self.show_dashboard()
                return

            time.sleep(1)

            self.tracker = EmotionTracker()
            self.start_time = time.time()
            self.is_recording = True

            self.root.after(20000, self.enable_stop_button)
            self.update_video_feed()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
            self.show_dashboard()

    def enable_stop_button(self):
        self.stop_button.config(state='normal')
        print("Stop Analysis button enabled")

    def update_video_feed(self):
        if self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                frame = self.tracker.process_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.config(image=photo)
                self.video_label.image = photo
                self.root.after(10, self.update_video_feed)
            else:
                self.stop_behavior_test()
                messagebox.showerror("Error", "Failed to capture video frame")

    def stop_and_process(self):
        self.is_recording = False
        self.stop_behavior_test()
        self.process_behavior_results()

    def stop_behavior_test(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'tracker'):
            self.tracker.release()
        self.video_label.config(image='')

    def process_behavior_results(self):
        try:
            metrics = self.tracker.get_metrics()
            print("Metrics:", metrics)
            features = extract_features(metrics)
            print("Features:", features)
            print("Type of features:", type(features))

            if isinstance(features, (list, tuple)):
                features_data = list(features)
            elif hasattr(features, 'tolist'):
                features_data = features.tolist()
            else:
                raise ValueError(f"Features must be a list or convertible to a list, got {type(features)}")

            request_data = {
                "metrics": metrics,
                "features": features_data
            }
            print("Data sent to backend:", request_data)

            response = requests.post(f"{API_URL}/analyze_behavior", json=request_data)
            print("Response status code:", response.status_code)
            print("Response content:", response.text)

            if response.status_code == 200:
                result = response.json()
                self.display_behavior_results(result)
            else:
                messagebox.showerror("Error", f"Behavior Analysis Failed: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Request failed: {str(e)}")
            self.show_dashboard()

    def display_behavior_results(self, result):
        print("Backend response:", result)
        self.clear_screen()
        self.create_nav_bar()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        tk.Label(main_frame, text="Behavior Analysis Report", 
                font=("Helvetica", 18, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=20)
        
        canvas = tk.Canvas(main_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Metrics Card with Seaborn Bar Plot
        metrics_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        metrics_card.pack(pady=10, padx=20, fill='both')
        metrics_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(metrics_card, text="Metrics", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        
        metrics = result.get("metrics", {})
        if metrics:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=list(metrics.values()), y=list(metrics.keys()), ax=ax, palette="Blues_d")
            ax.set_xlabel("Score", fontsize=10, color=self.colors['text'])
            ax.set_ylabel("Metric", fontsize=10, color=self.colors['text'])
            ax.set_xlim(0, 1)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_color(self.colors['text'])
            plt.tight_layout()
            
            canvas_plot = FigureCanvasTkAgg(fig, master=metrics_card)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(pady=10, padx=15, fill='x')
            plt.close(fig)
        
        # Emotion Card
        emotion_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        emotion_card.pack(pady=10, padx=20, fill='both')
        emotion_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(emotion_card, text="Predicted Emotion", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        tk.Label(emotion_card, text=result.get("emotion", "N/A"), 
                font=("Helvetica", 12), bg=self.colors['card_bg'], 
                fg=self.colors['text']).pack(anchor='w', padx=25, pady=5)
        
        # Emotion Probabilities Card with Matplotlib Pie Chart
        probs_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        probs_card.pack(pady=10, padx=20, fill='both')
        probs_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(probs_card, text="Emotion Probabilities", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        
        probabilities = result.get("probabilities", {})
        if probabilities:
            labels = list(probabilities.keys())
            values = list(probabilities.values())
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
                   colors=sns.color_palette("Blues_d", len(labels)))
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            plt.tight_layout()
            
            canvas_plot = FigureCanvasTkAgg(fig, master=probs_card)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(pady=10, padx=15, fill='x')
            plt.close(fig)
        
        # Overall Score Card
        score_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        score_card.pack(pady=10, padx=20, fill='both')
        score_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(score_card, text="Overall Performance Score", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        tk.Label(score_card, text=f"{result.get('overall_score', 0)} / 100", 
                font=("Helvetica", 12), bg=self.colors['card_bg'], 
                fg=self.colors['text']).pack(anchor='w', padx=25, pady=5)
        
        # Suggestions Card
        suggestions_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        suggestions_card.pack(pady=10, padx=20, fill='both')
        suggestions_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(suggestions_card, text="Suggestions to Improve", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        for suggestion in result.get("suggestions", []):
            tk.Label(suggestions_card, text=f"• {suggestion}", 
                    font=("Helvetica", 11), bg=self.colors['card_bg'], 
                    fg=self.colors['text'], wraplength=600, 
                    justify='left').pack(anchor='w', padx=25, pady=3)

    def upload_resume(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.resume_path = file_path
            messagebox.showinfo("Resume Uploaded", f"Uploaded: {os.path.basename(file_path)}")

    def submit_all(self):
        if not hasattr(self, 'resume_path'):
            messagebox.showerror("Error", "Please upload a resume")
            return

        answers_dict = {}
        for i, entry in enumerate(self.answers):
            answer = entry.get().strip()
            if not answer:
                messagebox.showerror("Error", "Please answer all questions")
                return
            key = f"q{i+1}"
            answers_dict[key] = answer

        files = {'resume': open(self.resume_path, 'rb')}
        data = {'answers': str(answers_dict)}

        try:
            response = requests.post(f"{API_URL}/analyze", files=files, data=data)
            if response.status_code == 200:
                self.display_results(response.json())
            else:
                messagebox.showerror("Error", f"API Error: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Request failed: {str(e)}")

    def display_results(self, result):
        self.clear_screen()
        self.create_nav_bar()
        
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        tk.Label(main_frame, text="Analysis Report", 
                font=("Helvetica", 18, "bold"), bg=self.colors['bg'], 
                fg=self.colors['heading']).pack(pady=20)
        
        canvas = tk.Canvas(main_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Parsed Resume Card
        parsed_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        parsed_card.pack(pady=10, padx=20, fill='both')
        parsed_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(parsed_card, text="Parsed Resume", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        
        parsed = result.get("parsed_resume", {})
        tk.Label(parsed_card, text=f"Skills: {', '.join(parsed.get('skills', [])) or 'N/A'}", 
                font=("Helvetica", 11), bg=self.colors['card_bg'], 
                fg=self.colors['text']).pack(anchor='w', padx=25, pady=3)
        tk.Label(parsed_card, text=f"Degree: {', '.join(parsed.get('degree', [])) or 'N/A'}", 
                font=("Helvetica", 11), bg=self.colors['card_bg'], 
                fg=self.colors['text']).pack(anchor='w', padx=25, pady=3)
        tk.Label(parsed_card, text=f"Experience: {parsed.get('experience', 'N/A')}", 
                font=("Helvetica", 11), bg=self.colors['card_bg'], 
                fg=self.colors['text']).pack(anchor='w', padx=25, pady=3)
        
        # Career Recommendations Card with Matplotlib Bar Chart
        career_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        career_card.pack(pady=10, padx=20, fill='both')
        career_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(career_card, text="Career Recommendations", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        
        career_recs = result.get("career_recommendations", [])
        if career_recs:
            careers = [item['career'] for item in career_recs]
            confidences = [item['confidence'] for item in career_recs]
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(careers, confidences, color=sns.color_palette("Blues_d", len(careers)))
            ax.set_xlabel("Confidence (%)", fontsize=10, color=self.colors['text'])
            ax.set_ylabel("Career", fontsize=10, color=self.colors['text'])
            ax.set_xlim(0, 100)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_color(self.colors['text'])
            plt.tight_layout()
            
            canvas_plot = FigureCanvasTkAgg(fig, master=career_card)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(pady=10, padx=15, fill='x')
            plt.close(fig)
        
        # Skill Gap Suggestions Card
        skill_gap_card = tk.Frame(scrollable_frame, bg=self.colors['card_bg'], bd=2, relief='flat')
        skill_gap_card.pack(pady=10, padx=20, fill='both')
        skill_gap_card.configure(highlightbackground=self.colors['heading'], highlightthickness=2)
        
        tk.Label(skill_gap_card, text="Skill Gap Suggestions", 
                font=("Helvetica", 14, "bold"), bg=self.colors['card_bg'], 
                fg=self.colors['heading']).pack(anchor='w', padx=15, pady=10)
        
        suggestions = result.get("skill_gap", {}).get("suggestions", [])
        if suggestions:
            for item in suggestions:
                tk.Label(skill_gap_card, text=f"• {item['skill']}: Learn via {item['suggested_course']}", 
                        font=("Helvetica", 11), bg=self.colors['card_bg'], 
                        fg=self.colors['text'], wraplength=600, 
                        justify='left').pack(anchor='w', padx=25, pady=3)
        else:
            tk.Label(skill_gap_card, text="No suggestions available.", 
                    font=("Helvetica", 11, "italic"), bg=self.colors['card_bg'], 
                    fg=self.colors['text']).pack(anchor='w', padx=25, pady=3)

if __name__ == '__main__':
    root = tk.Tk()
    app = CareerApp(root)
    root.mainloop()