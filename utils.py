import yaml
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

def read_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def plot_wave(waveform):
    # Plot the waveform
    plt.plot(waveform.t().numpy())
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.show()


def plot_spectrogram(spec):
    # Assuming spec has shape [channel, n_mels, time]
    # You can choose a specific channel (e.g., mono) or average over channels
    # For simplicity, let's assume you want to plot the first channel
    channel_index = 0
    spec_channel = spec[channel_index, :, :]

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spec_channel, aspect='auto', origin='lower', cmap='viridis')  # You can choose a different colormap
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bin')
    plt.colorbar(label='Amplitude (dB)')

    plt.show()

def update_spectrogram_plot(spectrogram_data, ax):
    """
    Update a spectrogram plot with new data.

    Parameters:
        frame (int): Frame index used in FuncAnimation.
        ax (matplotlib.axes.Axes): Axes object where the spectrogram plot is drawn.
    """

    ax.images[0].set_array(spectrogram_data)

    # Optionally, you can set labels or adjust other plot properties here
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram')

    return ax