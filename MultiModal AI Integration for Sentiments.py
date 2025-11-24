import os
import wave
import sounddevice as sd
import numpy as np
import cv2
from scipy.io.wavfile import write
from textblob import TextBlob
from vosk import Model, KaldiRecognizer


# === 1. Record Speech using Vosk ===
def record_audio(filename="speech.wav", duration=10):
    print("\nRecording speech for 10 seconds...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("Recording saved.")


# === 2. Speech to Text ===
def speech_to_text(filename="speech.wav"):
    model = Model(r"C:\Users\AKSHAT JAIN\OneDrive\Desktop\project.py\vosk-model-small-en-us-0.15")
    wf = wave.open(filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = eval(rec.Result())
            text += result.get("text", "") + " "
    return text.strip()


# === 3. Enhanced Text Sentiment with Polarity ===
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    # Emotion reasoning using both polarity and subjectivity
    if polarity > 0.3 and subjectivity > 0.5:
        sentiment = "Happy"
    elif polarity < -0.3 and subjectivity > 0.5:
        sentiment = "Angry or Sad"
    elif polarity < 0:
        sentiment = "Negative"
    elif polarity > 0:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    return sentiment, polarity


# === 4. Basic Voice Emotion Detection ===
def basic_voice_emotion(filename="speech.wav"):
    import scipy.io.wavfile as wav
    fs, data = wav.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0]

    data = data / np.max(np.abs(data))
    energy = np.sum(data ** 2) / len(data)
    zero_crossings = np.nonzero(np.diff(data > 0))[0]
    zcr = len(zero_crossings) / len(data)

    if energy > 0.01 and zcr > 0.1:
        return "Excited or Happy"
    elif energy < 0.005:
        return "Calm or Sad"
    else:
        return "Neutral"


# === 5. Webcam Image + Smile Detection ===
def capture_image():
    cap = cv2.VideoCapture(0)
    print("\nCapturing image from your in-built camera...")

    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        smile_detected = False

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22)
            if len(smiles) > 0:
                smile_detected = True
                break

        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image_filename = "captured_image.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Image saved as {image_filename}.")

        return "Happy" if smile_detected else "Neutral or Serious"
    else:
        return "Failed to capture image."


# === 6. Handle Input ===
def handle_input(input_type="text", input_data=None):
    if input_type == "text":
        return input_data
    elif input_type == "speech":
        return speech_to_text()
    elif input_type == "image":
        return capture_image()
    else:
        return ""


# === 7. Main Logic ===
def main():
    input_type = input(
        "Enter 'text' for manual text input, 'speech' to record speech, or 'image' to capture an image: ").strip().lower()

    if input_type == "text":
        user_input = input("Please enter your text: ").strip()
        sentiment, polarity = analyze_sentiment(user_input)
        print(f"\n📝 Sentiment: {sentiment}")
        print(f"📈 Polarity Score: {polarity}")
    elif input_type == "speech":
        record_audio()
        user_input = handle_input("speech")
        print(f"\n🗣️ Recognized Speech: {user_input}")
        sentiment, polarity = analyze_sentiment(user_input)
        emotion = basic_voice_emotion()
        print(f"📝 Sentiment: {sentiment}")
        print(f"📈 Polarity Score: {polarity}")
        print(f"🎙️ Voice Emotion Detected: {emotion}")
    elif input_type == "image":
        emotion = handle_input("image")
        print(f"\n📷 Detected Emotion from Image: {emotion}")
        user_input = input("How are you feeling today? ").strip()
        print("Have a Great Day!")
    else:
        print("Invalid input type! Please choose either 'text', 'speech', or 'image'.")


if __name__ == "__main__":
    main()
