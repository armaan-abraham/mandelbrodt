import matplotlib.pyplot as plt
import numpy as np
import torch

# TODO: dtypes


def koch_curve(order, size=1):
    def koch_curve_points(order):
        if order == 0:
            return np.array([[0, 0], [1, 0]])
        else:
            points = koch_curve_points(order - 1)
            n = len(points) - 1
            new_points = np.zeros((4 * n + 1, 2))
            new_points[0] = points[0]
            for i in range(n):
                a, b = points[i], points[i + 1]
                t = b - a
                c = a + t / 3
                d = a + t * 2 / 3
                e = a + np.dot(
                    t, np.array([[1 / 2, np.sqrt(3) / 6], [-np.sqrt(3) / 6, 1 / 2]])
                )
                new_points[4 * i + 1 : 4 * i + 5] = [c, e, d, b]
            return new_points

    curve = koch_curve_points(order) * size
    return curve


def complete_koch_polygon(curve, y_below):
    x_min, x_max = curve[:, 0].min(), curve[:, 0].max()
    y_min = -y_below * (x_max - x_min)

    bottom_left = np.array([x_min, y_min])
    bottom_right = np.array([x_max, y_min])

    polygon = np.vstack((curve, bottom_right, bottom_left))

    return polygon


def point_inside_polygon(point, polygon):
    x, y = point
    polygon = torch.tensor(polygon, dtype=torch.float32)
    polygon_x = polygon[:, 0]
    polygon_y = polygon[:, 1]
    
    # Shifted coordinates to get edges
    p1x = polygon_x
    p1y = polygon_y
    p2x = torch.roll(polygon_x, shifts=-1)
    p2y = torch.roll(polygon_y, shifts=-1)
    
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
    xinters = (y - p1y) * (p2x - p1x) / p1p2y_diff_safe + p1x
    xinters = torch.where(nonzero, xinters, torch.zeros_like(xinters))
    
    # Final condition for intersection
    cond4 = (p1x == p2x) | (x <= xinters)
    
    # Combine all conditions
    intersections = cond1 & cond2 & cond3 & cond4
    
    # Check if the number of intersections is odd
    inside = intersections.sum().item() % 2 == 1
    return inside



def sample_labeled_points(n_points, polygon, size, as_tensor=False):
    min_x, max_x, min_y, max_y = 0, size, -size * y_below, size * y_above

    # Sample points
    points = np.random.uniform(
        low=[min_x, min_y], high=[max_x, max_y], size=(n_points, 2)
    )

    points = torch.FloatTensor(points)
    polygon = torch.FloatTensor(polygon)

    # Label points
    labels = []
    for point in points:
        label = point_inside_polygon(point, polygon)
        if not as_tensor:
            label = int(label)
        labels.append(label)

    if as_tensor:
        labels = torch.tensor(labels, dtype=torch.int8)
    else:
        labels = np.array(labels, dtype=int)

    return points, labels


y_below = 1 / 5
y_above = 1 / 3
fig_dim = 5


def plot_koch_curve(order, size=1, points=None, labels=None, ax=None):
    curve = koch_curve(order, size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim * (y_above + y_below)))

    # plot curve
    ax.plot(curve[:, 0], curve[:, 1], color="red")

    # ax.fill_between(curve[:, 0], curve[:, 1], size * y_above, color="white")

    # ax.fill_between(curve[:, 0], curve[:, 1], -size * y_below, color="black", alpha=0.5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, size)
    ax.set_ylim(-size * y_below, size * y_above)

    if points is not None and labels is not None:
        inside_points = points[labels == 1]
        outside_points = points[labels == 0]
        ax.scatter(inside_points[:, 0], inside_points[:, 1], c="white", s=1, alpha=0.5)
        ax.scatter(
            outside_points[:, 0], outside_points[:, 1], c="black", s=1, alpha=0.5
        )

    # Remove all margins
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    if ax is None:
        plt.show()


if __name__ == "__main__":
    print("Koch Curve")
    polygon = complete_koch_polygon(koch_curve(order=5, size=1), y_below)
    points, labels = sample_labeled_points(1000, polygon, 1)
    plot_koch_curve(order=5, size=1, points=points, labels=labels)
    plt.savefig("out.png", dpi=200, bbox_inches="tight", pad_inches=0)
