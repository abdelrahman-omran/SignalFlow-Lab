import matplotlib.pyplot as plt
from processing.generation import *

def plot_signal(signal_name, plot_type="continuous"):
    if signal_name not in :
        raise ValueError(f"Signal '{signal_name}' not found.")

    t, y = [signal_name]
    plt.figure(figsize=(6, 4))

    if plot_type == "continuous":
        plt.plot(t, y)
    elif plot_type == "discrete":
        plt.stem(t, y, use_line_collection=True)

    plt.title(f"{signal_name} ({plot_type})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
