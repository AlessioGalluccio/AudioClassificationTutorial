import pyaudio
import numpy as np
from model import *
from transform import AudioUtil
import time


CHUNK = 32768
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
PATH_MODEL = "models/model_audio_OLD.pt"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

model = AudioClassifier()
model.load_state_dict(torch.load(PATH_MODEL))
model.eval()
ax = None

while True:
    # Read audio data from the stream
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16).copy()
    # Convert NumPy array to PyTorch tensor
    waveform_tensor = torch.from_numpy(audio_data).float()
    waveform_tensor = torch.unsqueeze(waveform_tensor, 0)
    spectrogram = AudioUtil.spectrogram(waveform_tensor, RATE)
    ax = dynamic_spectrogram_plot(spectrogram, ax)

    with torch.no_grad():
        output = model(torch.unsqueeze(spectrogram, 0))
        predicted_class = torch.argmax(output).item()
        print(f"Predicted class: {predicted_class}")




