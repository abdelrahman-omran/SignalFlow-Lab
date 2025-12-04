def quantize_signal(signal, levels):
    """
    signal: tuple (indices, values)
    levels: number of quantization levels
    """
    indices, values = signal
    if levels < 2:
        raise ValueError("Levels must be >= 2")
    max_val = max(values)
    min_val = min(values)
    step = (max_val - min_val) / (levels - 1)

    quantized_values = [round((v - min_val) / step) * step + min_val for v in values]
    return indices, quantized_values
