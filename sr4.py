import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import os

# Initialize the speaker recognizer
recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                             savedir="pretrained_models/spkrec-ecapa-voxceleb")

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Directory to save speaker embeddings and audio samples
os.makedirs("saved_speakers", exist_ok=True)

# Dictionary to hold speaker embeddings
speaker_embeddings = {}

def record_audio(filename, duration):
    """Records audio from the microphone for a specified duration and saves it to a .wav file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save audio to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
def get_speaker_embedding(filename):
    """Extracts and returns the speaker embedding from the audio file."""
    signal, sr = torchaudio.load(filename, backend='soundfile')
    signal = (signal - signal.mean()) / signal.std()  # Normalize
    embedding = recognizer.encode_batch(signal)
    return embedding.squeeze(0)

def compare_embeddings(new_embedding):
    """Finds the closest match for a given embedding among saved speakers."""
    max_similarity = -1
    matched_name = None
    
    for name, embedding in speaker_embeddings.items():
        similarity = torch.cosine_similarity(new_embedding, embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            matched_name = name
    
    return matched_name, max_similarity

def record_1min_sample():
    """Records a 1-minute sample, extracts the embedding, and saves it."""
    speaker_name = entry_name.get().strip()
    if not speaker_name:
        messagebox.showerror("Error", "Please enter the speaker's name.")
        return
    
    filename = f"saved_speakers/{speaker_name}.wav"
    record_audio(filename, 60)
    
    embedding = get_speaker_embedding(filename)
    speaker_embeddings[speaker_name] = embedding
    messagebox.showinfo("Success", f"1-minute sample for {speaker_name} saved!")

def record_and_identify():
    """Records a 10-second sample and identifies the speaker."""
    filename = "temp_sample.wav"
    record_audio(filename, 10)
    
    new_embedding = get_speaker_embedding(filename)
    matched_name, similarity = compare_embeddings(new_embedding)
    
    if similarity > 0.5:
        messagebox.showinfo("Result", f"Matched Speaker: {matched_name} (Similarity: {similarity:.2f})")
    else:
        messagebox.showinfo("Result", "Speaker not recognized.")

# Load pre-existing embeddings
for file in os.listdir("saved_speakers"):
    if file.endswith(".wav"):
        name = os.path.splitext(file)[0]
        filepath = os.path.join("saved_speakers", file)
        speaker_embeddings[name] = get_speaker_embedding(filepath)

# Create the UI
root = tk.Tk()
root.title("Speaker Recognition System")

# Section for recording a 1-minute sample
frame_save = tk.Frame(root)
frame_save.pack(pady=10)
tk.Label(frame_save, text="Speaker Name:").grid(row=0, column=0, padx=5)
entry_name = tk.Entry(frame_save)
entry_name.grid(row=0, column=1, padx=5)
btn_save = tk.Button(frame_save, text="Record 1-Minute Sample", command=record_1min_sample)
btn_save.grid(row=0, column=2, padx=5)

# Section for identifying speaker from a 10-second sample
frame_identify = tk.Frame(root)
frame_identify.pack(pady=10)
btn_identify = tk.Button(frame_identify, text="Record 10-Second Sample", command=record_and_identify)
btn_identify.pack()

root.mainloop()