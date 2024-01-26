import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

def compute_spectrogram():
    """
    Example function to compute spectrogram data.
    """
    # Replace this with your actual function to compute the spectrogram
    return np.random.rand(100, 100)  # Example random spectrogram data

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
