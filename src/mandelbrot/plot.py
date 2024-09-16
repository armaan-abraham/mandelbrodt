import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List
from pathlib import Path

from mandelbrot.koch import plot_koch_curve, Y_BELOW, Y_ABOVE


def generate_animation(
    model_states_file: Path,
    predictions_file: Path,
    order: int,
    size: float,
    save_interval: int,
    fps: int,
    gif_file: Path,
    interval: int = 200,
):
    model_states = torch.load(model_states_file)
    predictions = torch.load(predictions_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Model Predictions During Training")

    x_min, x_max = 0, size
    y_min, y_max = -size * Y_BELOW, size * Y_ABOVE
    plot_koch_curve(order, size, ax=ax)

    im = ax.imshow(
        predictions[0],
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        alpha=1,
        animated=True,
        cmap="magma_r",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    def update(frame):
        im.set_array(predictions[frame])
        ax.set_title(f"Model Predictions at Epoch {frame * save_interval}")
        return (im,)

    anim = FuncAnimation(
        fig, update, frames=len(predictions), interval=interval, blit=True, repeat=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(gif_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved as '{gif_file}'")
