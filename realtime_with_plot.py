import pyaudio
import numpy as np
from model import *
from transform import AudioUtil
import time
import wave


CHUNK = 176400
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
PATH_MODEL = "models/model_audio_OLD.pt"
def wave_output_filename(iteration):
    return f"models/output_{iteration}.wav"

classes = {
    '0': "air_conditioner",
    '1': "car_horn",
    '2': "children_playing",
    '3': "dog_bark",
    '4': "drilling",
    '5': "engine_idling",
    '6': "gun_shot",
    '7': "jackhammer",
    '8': "siren",
    '9': "street_music"
}

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

def compute_spectrogram():
    # Read audio data from the stream
    audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16).copy()
    # Convert NumPy array to PyTorch tensor
    waveform_tensor = torch.from_numpy(audio_data).float()
    waveform_tensor = torch.unsqueeze(waveform_tensor, 0)
    spectrogram = AudioUtil.spectrogram(waveform_tensor, RATE)
    spectrogram_output = spectrogram.numpy()[0]

    # save audio file
    frames = []
    data = stream.read(CHUNK)
    frames.append(data)
    wf = wave.open(wave_output_filename(1), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    with torch.no_grad():
        output = model(torch.unsqueeze(spectrogram, 0))
        print(output)
        predicted_class = classes[str(torch.argmax(output).item())]
        print(f"Predicted class: {predicted_class}")

    return spectrogram_output

def update_spectrogram_plot(frame, ax):
    """
    Update a spectrogram plot with new data.

    Parameters:
        frame (int): Frame index used in FuncAnimation.
        ax (matplotlib.axes.Axes): Axes object where the spectrogram plot is drawn.
    """
    # Compute new spectrogram data
    spectrogram_data = compute_spectrogram()

    ax.images[0].set_array(spectrogram_data)

    # Optionally, you can set labels or adjust other plot properties here
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')

    return ax

def animate_spectrogram(num_frames=10, interval=200):
    """
    Animate the spectrogram plot.

    Parameters:
        num_frames (int): Number of frames for the animation.
        interval (int): Delay between frames in milliseconds.
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Plot the initial spectrogram (with random data for demonstration)
    initial_data = compute_spectrogram()
    img = ax.imshow(initial_data, origin='lower', aspect='auto',
                    norm=colors.Normalize(vmin=np.min(initial_data), vmax=np.max(initial_data)),
                    cmap='viridis')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')

    def update(frame):
        return update_spectrogram_plot(frame, ax),

    # Animate the plot
    ani = FuncAnimation(fig, update, frames=range(num_frames), blit=True, interval=interval)
    plt.show()

# Example usage:
if __name__ == "__main__":
    animate_spectrogram(num_frames=20, interval=300)