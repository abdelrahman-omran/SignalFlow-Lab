def multiply_signal(signal, constant):
    indices, values = signal
    return indices, [v * constant for v in values]
