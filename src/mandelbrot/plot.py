

def generate_animation(data_file):
    data = torch.load(data_file)
    predictions = data["predictions"]
    generate_plot(
        predictions,
        constants.ORDER,
        constants.SIZE,
        save_interval=constants.SAVE_INTERVAL,
        fps=constants.FPS,
    )
    print("Animation complete!")

def generate_plot(
    predictions: List[np.ndarray],
    order: int,
    size: float,
    interval: int = 200,
    save_interval: int = 5,
    fps: int = 10,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Model Predictions During Training")

    curve = koch_curve(order, size)
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
    anim.save(GIF_FILE, writer=writer)
    plt.close(fig)
    print(f"Animation saved as '{GIF_FILE}'")