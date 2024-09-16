import argparse
import copy
from pathlib import Path
from typing import List, Tuple
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from mandelbrot.koch import (
    koch_curve,
    plot_koch_curve,
    Y_BELOW,
    Y_ABOVE,
    complete_koch_polygon,
    sample_labeled_points,
)
from mandelbrot.act import ACT
from mandelbrot.plot import generate_animation

# Constants
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class Constants:
    INPUT_SIZE = 2
    ORDER = 5
    SIZE = 1
    N_POINTS = int(2e6)
    BATCH_SIZE = int(5e3)
    GRID_BATCH_SIZE = int(5e4)
    N_EPOCHS = 50
    # N_SAVES = 50
    # SAVE_INTERVAL = N_EPOCHS // N_SAVES
    SAVE_INTERVAL = 1
    GRID_DIM = 150
    TAU = 0.0001
    FPS = 4
    MAX_ITER = 300
    HIDDEN_SIZE = 400
    OUTPUT_SIZE = 1
    OUTPUT_MODULE_SIZE = 150
    OUTPUT_MODULE_HIDDEN_LAYERS = 2
    HALT_GATE_SIZE = 250
    HALT_GATE_HIDDEN_LAYERS = 2
    EPSILON = 1e-3
    ADAPTIVE_TIME = True
    RUN_ID = "act_small"


constants = Constants()

RUN_FOLDER = RESULTS_DIR / constants.RUN_ID
RUN_FOLDER.mkdir(exist_ok=True)

MODEL_STATES_FILE = RUN_FOLDER / f"model_states.pth"
PREDICTIONS_FILE = RUN_FOLDER / f"predictions.pth"
GIF_FILE = RUN_FOLDER / f"training.gif"
CONSTANTS_FILE = RUN_FOLDER / f"constants.txt"


def create_model() -> torch.nn.Module:
    return ACT(
        input_size=constants.INPUT_SIZE,
        hidden_size=constants.HIDDEN_SIZE,
        output_size=constants.OUTPUT_SIZE,
        output_module_size=constants.OUTPUT_MODULE_SIZE,
        output_module_hidden_layers=constants.OUTPUT_MODULE_HIDDEN_LAYERS,
        halt_gate_size=constants.HALT_GATE_SIZE,
        halt_gate_hidden_layers=constants.HALT_GATE_HIDDEN_LAYERS,
        max_iter=constants.MAX_ITER,
        epsilon=constants.EPSILON,
        adaptive_time=constants.ADAPTIVE_TIME,
    )


def compute_loss(
    model: torch.nn.Module,
    batch_X: torch.Tensor,
    batch_y: torch.Tensor,
    include_regularization: bool = True,
) -> torch.Tensor:
    outputs = model(batch_X)
    bce_loss = torch.nn.BCELoss(reduction="mean")(outputs, batch_y.unsqueeze(1).float())
    if include_regularization:
        return (
            bce_loss
            + constants.TAU
            * (torch.sum(model.iter_taken) + torch.sum(model.remainder))
            / batch_X.shape[0]
        )
    return bce_loss


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_batches = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, batch_X, batch_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += compute_loss(
                model, batch_X, batch_y, include_regularization=False
            ).item()
        total_batches += 1
    return total_loss / total_batches


def save_state(model, grid, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        for i in range(0, grid.shape[0], constants.GRID_BATCH_SIZE):
            batch = grid[i : i + constants.GRID_BATCH_SIZE].to(device)
            pred_batch = model(batch).cpu().numpy()
            predictions.append(pred_batch)
        pred = np.concatenate(predictions).reshape(int(np.sqrt(grid.shape[0])), -1)
    return pred


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_epochs: int,
    save_interval: int,
    grid: torch.Tensor,
    device: torch.device,
) -> Tuple[List[dict], List[np.ndarray]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    model_states = []
    predictions = []

    print("Total parameters: ", sum(p.numel() for p in model.parameters()))

    progress_bar = tqdm(range(n_epochs), desc="Training")
    for epoch in progress_bar:
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        progress_bar.set_postfix(
            μ_iter=f"{model.mean_iter_taken:.2f}", loss=f"{avg_loss:.4f}"
        )

        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            model_states.append(copy.deepcopy(model.state_dict()))
            predictions.append(save_state(model, grid, device))

    return model_states, predictions


def train_and_save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    curve = koch_curve(constants.ORDER, constants.SIZE)
    polygon = complete_koch_polygon(curve, Y_BELOW)
    points, labels = sample_labeled_points(constants.N_POINTS, polygon, constants.SIZE)
    print("Points sampled")

    dataset = TensorDataset(torch.FloatTensor(points), torch.FloatTensor(labels))
    dataloader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    print("Dataset created")

    x = np.linspace(0, constants.SIZE, constants.GRID_DIM)
    y = np.linspace(
        -constants.SIZE * Y_BELOW, constants.SIZE * Y_ABOVE, constants.GRID_DIM
    )
    xx, yy = np.meshgrid(x, y)
    grid = torch.FloatTensor(np.column_stack((xx.ravel(), yy.ravel())))
    model = create_model().to(device)
    print("Starting training...")
    model_states, predictions = train_model(
        model,
        dataloader,
        constants.N_EPOCHS,
        constants.SAVE_INTERVAL,
        grid,
        device,
    )

    sys.stdout.write("\rSaving final model data...   ")
    sys.stdout.flush()

    # Save model states
    torch.save(model_states, MODEL_STATES_FILE)

    # Save predictions
    torch.save(predictions, PREDICTIONS_FILE)

    # Save constants to a text file
    with open(CONSTANTS_FILE, "w") as f:
        content = json.dumps(
            {
                k: getattr(constants, k)
                for k in dir(constants)
                if not k.startswith("__")
            },
            indent=4,
        )
        print(content)
        f.write(content)

    sys.stdout.write("\rTraining complete and data saved!   \n")
    sys.stdout.flush()
    print(f"Model states saved to '{MODEL_STATES_FILE}'")
    print(f"Predictions saved to '{PREDICTIONS_FILE}'")
    print(f"Constants saved to '{CONSTANTS_FILE}'")


def main():
    parser = argparse.ArgumentParser(
        description="Koch curve neural network training and visualization"
    )
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train the model and save results"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate animation from saved results",
    )

    args = parser.parse_args()

    actions = {
        "train": train_and_save_model,
        "plot": lambda: generate_animation(
            MODEL_STATES_FILE,
            PREDICTIONS_FILE,
            constants.ORDER,
            constants.SIZE,
            constants.SAVE_INTERVAL,
            constants.FPS,
            GIF_FILE,
            interval=200,  # Add this line
        ),
    }

    for arg, action in actions.items():
        if getattr(args, arg):
            action()
            return

    parser.print_help()


if __name__ == "__main__":
    main()
