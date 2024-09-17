import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Tuple
from pathlib import Path

# Constants
Y_BELOW = 0.05
Y_ABOVE = 0.31
FIG_DIM = 5

THIS_DIR = Path(__file__).parent

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def koch_curve(order: int, size: float = 1) -> NDArray[np.float64]:
    def koch_curve_points(order: int) -> NDArray[np.float64]:
        if order == 0:
            return np.array([[0, 0], [1, 0]])
        points = koch_curve_points(order - 1)
        n = len(points) - 1
        new_points = np.zeros((4 * n + 1, 2))
        new_points[0] = points[0]
        for i in range(n):
            a, b = points[i], points[i + 1]
            t = b - a
            c, d = a + t / 3, a + t * 2 / 3
            e = a + np.dot(
                t, np.array([[1 / 2, np.sqrt(3) / 6], [-np.sqrt(3) / 6, 1 / 2]])
            )
            new_points[4 * i + 1 : 4 * i + 5] = [c, e, d, b]
        return new_points

    return koch_curve_points(order) * size


def complete_koch_polygon(curve: NDArray[np.float64]) -> NDArray[np.float64]:
    """Complete the Koch polygon by adding bottom points."""
    x_min, x_max = curve[:, 0].min(), curve[:, 0].max()
    y_min = -Y_BELOW * (x_max - x_min)
    bottom_left = np.array([x_min, y_min])
    bottom_right = np.array([x_max, y_min])
    return np.vstack((curve, bottom_right, bottom_left))


def point_inside_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    Check if multiple points are inside a polygon using a vectorized ray casting algorithm.

    Args:
        points (torch.Tensor): Tensor of shape (n_points, 2), representing the points to check.
        polygon (torch.Tensor): Tensor of shape (n_vertices, 2), representing the polygon vertices.

    Returns:
        torch.Tensor: Boolean tensor of shape (n_points,), where True indicates the point is inside the polygon.
    """
    x = points[:, 0].unsqueeze(1)  # Shape: (n_points, 1)
    y = points[:, 1].unsqueeze(1)  # Shape: (n_points, 1)

    polygon_x = polygon[:, 0]
    polygon_y = polygon[:, 1]

    # Shifted coordinates to get edges
    p1x = polygon_x.unsqueeze(0)  # Shape: (1, n_vertices)
    p1y = polygon_y.unsqueeze(0)  # Shape: (1, n_vertices)
    p2x = torch.roll(polygon_x, shifts=-1).unsqueeze(0)
    p2y = torch.roll(polygon_y, shifts=-1).unsqueeze(0)

    # Compute min and max for y coordinates
    min_y = torch.min(p1y, p2y)
    max_y = torch.max(p1y, p2y)

    # Conditions for ray intersection
    cond1 = y > min_y
    cond2 = y <= max_y
    cond3 = x <= torch.max(p1x, p2x)

    # Avoid division by zero
    p1p2y_diff = p2y - p1y
    nonzero = p1p2y_diff != 0
    p1p2y_diff_safe = torch.where(nonzero, p1p2y_diff, torch.ones_like(p1p2y_diff))

    # Compute x-intersections
    xinters = torch.where(
        nonzero, (y - p1y) * (p2x - p1x) / p1p2y_diff_safe + p1x, torch.zeros_like(p1x)
    )

    # Final condition for intersection
    cond4 = (p1x == p2x) | (x <= xinters)

    # Combine all conditions
    intersections = cond1 & cond2 & cond3 & cond4  # Shape: (n_points, n_vertices)

    # Count the number of intersections for each point
    num_intersections = intersections.sum(dim=1)

    # Point is inside the polygon if the number of intersections is odd
    inside = num_intersections % 2 == 1
    return inside


def sample_labeled_points(
    n_points: int, polygon: np.ndarray, size: float, as_tensor: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points within a specified range and label them as inside or outside the given polygon.

    Args:
        n_points (int): Number of points to sample.
        polygon (np.ndarray): Numpy array of shape (n_vertices, 2) representing the polygon vertices.
        size (float): Size parameter defining the sampling range.
        as_tensor (bool): If True, returns tensors; otherwise, returns NumPy arrays.

    Returns:
        tuple: A tuple containing the sampled points and their labels.
    """
    min_x, max_x = 0, size
    min_y, max_y = -size * Y_BELOW, size * Y_ABOVE

    # Sample random points as float64
    points = torch.from_numpy(
        np.random.uniform(
            low=[min_x, min_y], high=[max_x, max_y], size=(n_points, 2)
        ).astype(np.float64)
    ).to(device)

    # Convert polygon to torch tensor (float64)
    polygon_tensor = torch.from_numpy(polygon.astype(np.float64)).to(device)

    # Vectorized point-in-polygon test
    labels = point_inside_polygon(points, polygon_tensor).to(torch.int8)

    if not as_tensor:
        labels = labels.cpu().numpy().astype(int)
        points = points.cpu().numpy()

    return points, labels


def plot_koch_curve(
    order: int,
    size: float = 1,
    points: NDArray[np.float64] = None,
    labels: NDArray[np.int64] = None,
    ax: plt.Axes = None,
    point_size: float = 0.1,
) -> None:
    """Plot Koch curve and optionally scatter labeled points."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_DIM, FIG_DIM * (Y_ABOVE + Y_BELOW)))

    curve = koch_curve(order, size)
    ax.plot(curve[:, 0], curve[:, 1], color="black", linewidth=0.5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, size)
    ax.set_ylim(-size * Y_BELOW, size * Y_ABOVE)

    if points is not None and labels is not None:
        inside_points = points[labels == 1]
        outside_points = points[labels == 0]
        ax.scatter(
            inside_points[:, 0],
            inside_points[:, 1],
            c="#EF6F6C",
            s=point_size,
            alpha=0.5,
        )
        ax.scatter(
            outside_points[:, 0],
            outside_points[:, 1],
            c="#465775",
            s=point_size,
            alpha=0.5,
        )

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    if ax is None:
        plt.show()


if __name__ == "__main__":
    polygon = complete_koch_polygon(koch_curve(order=4, size=1))
    points, labels = sample_labeled_points(int(1e4), polygon, 1)
    plot_koch_curve(
        order=4,
        size=1,
        points=points,
        labels=labels,
        point_size=0.01,
    )
    plt.savefig(
        THIS_DIR / "results" / "out.png", dpi=500, bbox_inches="tight", pad_inches=0
    )
