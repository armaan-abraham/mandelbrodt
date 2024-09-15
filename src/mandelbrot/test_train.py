import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from mandelbrot.koch import (
    koch_curve,
    plot_koch_curve,
    y_below,
    y_above,
    complete_koch_polygon,
    sample_labeled_points,
)

input_size = 2
hidden_size = 512
THIS_DIR = Path(__file__).parent

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
    torch.nn.Sigmoid(),
)


def train_model(model, dataloader, n_epochs, save_interval, grid, device):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    model_states = []
    predictions = []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            # Save model state
            model_states.append(copy.deepcopy(model.state_dict()))
            # Save predictions
            model.eval()
            with torch.no_grad():
                pred = (
                    model(grid.to(device))
                    .cpu()
                    .numpy()
                    .reshape(int(np.sqrt(grid.shape[0])), -1)
                )
            predictions.append(pred)
    return model_states, predictions


def generate_plot(
    predictions,
    order,
    size,
    interval=200,
    save_interval=5,
    fps=10,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    title = "Model Predictions During Training"
    ax.set_title(title)

    curve = koch_curve(order, size)
    x_min, x_max = 0, size
    y_min, y_max = -size * y_below, size * y_above
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

    filename = "koch_curve_training.gif"
    writer = PillowWriter(fps=fps)
    anim.save(THIS_DIR / "results" / filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved as '{filename}'")


if __name__ == "__main__":
    # Set up parameters
    order = 4
    size = 1
    n_points = 50000
    batch_size = 128
    n_epochs = 300
    save_interval = 5
    grid_dim = 1000
    fps = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    curve = koch_curve(order, size)
    polygon = complete_koch_polygon(curve, y_below)
    points, labels = sample_labeled_points(n_points, polygon, size)
    print("Points sampled")

    # Create dataset and dataloader
    dataset = TensorDataset(torch.FloatTensor(points), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset created")

    x = np.linspace(0, size, grid_dim)
    y = np.linspace(-size * y_below, size * y_above, grid_dim)
    xx, yy = np.meshgrid(x, y)
    grid = torch.FloatTensor(np.column_stack((xx.ravel(), yy.ravel())))

    # Train model
    model = model.to(device)
    print("Starting training...")
    model_states, predictions = train_model(
        model, dataloader, n_epochs, save_interval, grid, device
    )

    # Generate and save animation
    generate_plot(predictions, order, size, save_interval, fps)

    print("Training and animation complete!")
