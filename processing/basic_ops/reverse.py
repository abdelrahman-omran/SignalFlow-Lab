def reverse_signal(signal):
    indices, values = signal
    indices = indices[::-1]
    values = values[::-1]

    return indices, values