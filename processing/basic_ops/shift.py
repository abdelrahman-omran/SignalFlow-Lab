def shift_signal(signal, steps):
    """
    Move a signal left or right.
    steps > 0: delay (shift right, add zeros at the start).
    steps < 0: advance (shift left, trim the start).
    """
    indices, values = signal

    if steps > 0:
        indices = [i + steps for i in indices]
        values = [0] * steps + values
    elif steps < 0:
        steps = abs(steps)
        indices = indices[steps:]
        values = values[steps:]
    # if steps == 0 → unchanged

    return indices, values