def add(signal1, signal2):
    indices1, values1 = signal1
    indices2, values2 = signal2

    # Ensure same length (take overlap)
    length = min(len(values1), len(values2))
    indices = indices1[:length]  # assume both signals aligned
    values = [values1[i] + values2[i] for i in range(length)]

    return indices, values
