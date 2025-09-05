def subtract_signals(signal1, signal2):
    indices1, values1 = signal1
    indices2, values2 = signal2

    length = min(len(values1), len(values2))
    indices = indices1[:length]  # assume aligned
    values = [values1[i] - values2[i] for i in range(length)]

    return indices, values
