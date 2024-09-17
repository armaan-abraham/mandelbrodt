import copy
from pathlib import Path
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json

from mandelbrot.koch import (
    koch_curve,
    complete_koch_polygon,
    sample_labeled_points,
)
from mandelbrot.act import ACT

# Constants
THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class Constants:
    INPUT_SIZE = 2
    ORDER = 5
    SIZE = 1
    N_POINTS = int(1e6)
    BATCH_SIZE = int(5e3)
    N_EPOCHS = 100
    SAVE_INTERVAL = 5
    TAU = 0.001
    FPS = 4
    MAX_ITER = 300
    HIDDEN_SIZE = 200
    OUTPUT_SIZE = 1
    OUTPUT_MODULE_SIZE = 50
    OUTPUT_MODULE_HIDDEN_LAYERS = 2
    HALT_GATE_SIZE = 50
    HALT_GATE_HIDDEN_LAYERS = 2
    EPSILON = 1e-3
    ADAPTIVE_TIME = True
    RUN_ID = "act_small"


constants = Constants()

RUN_FOLDER = RESULTS_DIR / constants.RUN_ID
RUN_FOLDER.mkdir(exist_ok=True)

MODEL_STATES_FILE = RUN_FOLDER / f"model_states.pth"
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


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_epochs: int,
    save_interval: int,
    device: torch.device,
) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, amsgrad=True)
    model_states = []

    print("Total parameters: ", sum(p.numel() for p in model.parameters()))

    progress_bar = tqdm(range(n_epochs), desc="Training")
    for epoch in progress_bar:
        avg_loss = train_epoch(model, dataloader, optimizer, device)

        progress_bar.set_postfix(
            μ_iter=f"{model.mean_iter_taken:.2f}", loss=f"{avg_loss:.4f}"
        )

        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            model_states.append(copy.deepcopy(model.state_dict()))

    return model_states


def train_and_save_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    curve = koch_curve(constants.ORDER, constants.SIZE)
    polygon = complete_koch_polygon(curve)
    points, labels = sample_labeled_points(constants.N_POINTS, polygon, constants.SIZE)
    print("Points sampled")

    dataset = TensorDataset(torch.FloatTensor(points), torch.FloatTensor(labels))
    dataloader = DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    print("Dataset created")

    model = create_model().to(device)
    print("Starting training...")
    model_states = train_model(
        model,
        dataloader,
        constants.N_EPOCHS,
        constants.SAVE_INTERVAL,
        device,
    )

    print("\rSaving final model data...")

    # Save model states
    torch.save(model_states, MODEL_STATES_FILE)

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

    print("\rTraining complete and data saved!   ")
    print(f"Model states saved to '{MODEL_STATES_FILE}'")
    print(f"Constants saved to '{CONSTANTS_FILE}'")


if __name__ == "__main__":
    train_and_save_model()
