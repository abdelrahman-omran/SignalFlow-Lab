import matplotlib.pyplot as plt

def visualize_signal_mode(signals, mode="continuous"):
    if not signals:
        raise ValueError("No signals to visualize.")

    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for idx, (indices, values) in enumerate(signals):
        color = colors[idx % len(colors)]
        if mode == "discrete":
            plt.stem(indices, values, label=f"Signal {idx + 1}",
                     linefmt=color, markerfmt=color + 'o', basefmt=" ")
        else:
            plt.plot(indices, values, label=f"Signal {idx + 1}")

    plt.title(f"Signal Visualization - {mode.capitalize()} Mode")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
