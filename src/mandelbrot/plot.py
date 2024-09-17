import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path

from mandelbrot.koch import plot_koch_curve, Y_BELOW, Y_ABOVE
from mandelbrot.train import Constants, create_model, RUN_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_DIM = 150


def generate_animation(
    interval: int = 200,
):
    # Load constants
    constants = Constants()

    # Set up paths
    model_states_file = RUN_FOLDER / "model_states.pth"
    gif_file = RUN_FOLDER / "training.gif"

    # Initialize model
    model = create_model().to(device)

    # Load model states
    model_states = torch.load(model_states_file, map_location=device)
    print(f"Loaded {len(model_states)} model states.")

    # Prepare grid for inference
    x = np.linspace(0, constants.SIZE, GRID_DIM)
    y = np.linspace(-constants.SIZE * Y_BELOW, constants.SIZE * Y_ABOVE, GRID_DIM)
    xx, yy = np.meshgrid(x, y)
    grid = torch.FloatTensor(np.column_stack((xx.ravel(), yy.ravel()))).to(device)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Model Predictions During Training")

    x_min, x_max = 0, constants.SIZE
    y_min, y_max = -constants.SIZE * Y_BELOW, constants.SIZE * Y_ABOVE
    plot_koch_curve(constants.ORDER, constants.SIZE, ax=ax)

    im = ax.imshow(
        np.zeros((GRID_DIM, GRID_DIM)),
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        alpha=1,
        animated=True,
        cmap="magma_r",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    def update(frame):
        model.load_state_dict(model_states[frame])
        model.eval()
        with torch.no_grad():
            preds = model(grid).cpu().numpy().reshape(GRID_DIM, GRID_DIM)
        im.set_array(preds)
        epoch_num = frame * constants.SAVE_INTERVAL
        ax.set_title(f"Model Predictions at Epoch {epoch_num}")
        return (im,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(model_states),
        interval=interval,
        blit=True,
        repeat=False,
    )

    writer = PillowWriter(fps=constants.FPS)
    anim.save(gif_file, writer=writer)
    plt.close(fig)
    print(f"Animation saved as '{gif_file}'")


def plot_iteration_heatmap():
    # Load constants
    constants = Constants()

    # Set up paths
    model_states_file = RUN_FOLDER / "model_states.pth"
    heatmap_file = RUN_FOLDER / "iteration_heatmap.png"

    # Initialize model
    model = create_model().to(device)

    # Load model states (last state)
    model_states = torch.load(model_states_file, map_location=device)
    model.load_state_dict(model_states[-1])
    model.eval()

    # Prepare grid for inference
    x = np.linspace(0, constants.SIZE, GRID_DIM)
    y = np.linspace(-constants.SIZE * Y_BELOW, constants.SIZE * Y_ABOVE, GRID_DIM)
    xx, yy = np.meshgrid(x, y)
    grid = torch.FloatTensor(np.column_stack((xx.ravel(), yy.ravel()))).to(device)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Number of Iterations Taken by ACT Model")

    x_min, x_max = 0, constants.SIZE
    y_min, y_max = -constants.SIZE * Y_BELOW, constants.SIZE * Y_ABOVE

    # Plot Koch curve
    plot_koch_curve(constants.ORDER, constants.SIZE, ax=ax)

    # Perform inference and get iteration counts
    with torch.no_grad():
        model(grid)  # This call updates model.iter_taken
        iter_counts = model.iter_taken.cpu().numpy().reshape(GRID_DIM, GRID_DIM)

    # Create heatmap
    im = ax.imshow(
        iter_counts,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        alpha=0.7,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Iterations")

    # Save the figure
    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Iteration heatmap saved as '{heatmap_file}'")


if __name__ == "__main__":
    generate_animation()
    plot_iteration_heatmap()
