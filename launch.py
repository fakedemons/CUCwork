import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque, Counter
from threading import Thread, Event
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_model(model_path, device):
    model = FaceCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_hist = cv2.equalizeHist(face_resized)
    face_normalized = face_hist.reshape(1, 48, 48) / 255.0
    face_tensor = torch.from_numpy(face_normalized).float().unsqueeze(0)
    return face_tensor

def predict_emotion(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

def voice_feedback(text):
    def speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    Thread(target=speak).start()

def update_plot(emotion_counter, emotions):
    plt.clf()
    emotion_counts = [emotion_counter[emotion] for emotion in emotions]
    plt.bar(emotions, emotion_counts)
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Real-time Emotion Distribution')
    plt.draw()

def recognize_emotions(device, model_path, emotions, feedback_folder, stop_event, emotion_counter, video_label):
    model = load_model(model_path, device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_window = deque(maxlen=20)

    while not stop_event.is_set():
        start_time = time.time()
        emotion_counter.clear()

        while time.time() - start_time < 10 and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                image_tensor = preprocess_image(face)
                emotion = predict_emotion(model, image_tensor, device)
                label = emotions[emotion]

                emotion_window.append(label)
                emotion_counter[label] += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        if emotion_window:
            most_common_emotion = Counter(emotion_window).most_common(1)[0][0]
            feedback_image = None

            if most_common_emotion == 'Happy':
                feedback_image = feedback_folder + 'bravo.jpg'
            elif most_common_emotion == 'Angry':
                feedback_image = feedback_folder + 'calm.jpg'
            elif most_common_emotion in ['Fear', 'Sad']:
                feedback_image = feedback_folder + 'warm.jpeg'
            elif most_common_emotion == 'Surprise':
                feedback_image = feedback_folder + 'surprise.jpeg'
            elif most_common_emotion == 'Disgust':
                feedback_image = feedback_folder + 'cute.jpeg'

            if feedback_image and os.path.isfile(feedback_image):
                img = cv2.imread(feedback_image)
                cv2.imshow('Feedback Image', img)
                cv2.waitKey(1000)
                cv2.destroyWindow('Feedback Image')

        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

def update_ui():
    if recognition_active:
        update_plot(emotion_counter, emotions)
        canvas.draw()
    root.after(1000, update_ui)

def start_recognition():
    global recognition_thread, recognition_active, stop_event
    if recognition_active:
        messagebox.showwarning("Warning", "Recognition is already running!")
        return
    recognition_active = True
    stop_event.clear()
    recognition_thread = Thread(target=recognize_emotions,
                                args=(device, model_path, emotions, feedback_folder, stop_event, emotion_counter, video_label))
    recognition_thread.start()

def pause_recognition():
    global recognition_active
    if not recognition_active:
        messagebox.showwarning("Warning", "Recognition is not running!")
        return
    recognition_active = False
    stop_event.set()

def resume_recognition():
    if recognition_active:
        messagebox.showwarning("Warning", "Recognition is already running!")
        return
    start_recognition()

def exit_program():
    if messagebox.askokcancel("Exit", "Are you sure you want to exit the program?"):
        global recognition_active, stop_event
        recognition_active = False
        stop_event.set()
        root.destroy()

def create_ui():
    global root, video_label, canvas, stop_event, fig, emotion_counter
    stop_event = Event()
    root = tk.Tk()
    root.title("Emotion Recognition System")
    root.geometry("1400x800")
    root.resizable(False, False)

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 14), padding=10)
    style.configure('TLabel', font=('Helvetica', 14))

    main_frame = ttk.Frame(root, padding=(10, 10, 10, 10))
    main_frame.pack(fill=tk.BOTH, expand=True)

    video_label_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding=(10, 10, 10, 10))
    video_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    video_label = ttk.Label(video_label_frame)
    video_label.pack(fill=tk.BOTH, expand=True)

    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill=tk.X)

    start_button = ttk.Button(control_frame, text="Start Recognition", command=start_recognition)
    start_button.pack(side=tk.LEFT, padx=5, pady=5)

    pause_button = ttk.Button(control_frame, text="Pause Recognition", command=pause_recognition)
    pause_button.pack(side=tk.LEFT, padx=5, pady=5)

    resume_button = ttk.Button(control_frame, text="Resume Recognition", command=resume_recognition)
    resume_button.pack(side=tk.LEFT, padx=5, pady=5)

    exit_button = ttk.Button(control_frame, text="Exit", command=exit_program)
    exit_button.pack(side=tk.LEFT, padx=5, pady=5)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#e6e6e6')
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    emotion_counter = Counter()

    root.after(1000, update_ui)
    root.mainloop()

# Initialize global variables
recognition_thread = None
recognition_active = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'model/model_o_cnn.pth'
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
feedback_folder = 'feedback_images/'

create_ui()
