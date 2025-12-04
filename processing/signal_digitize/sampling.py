def sampling(signal, rate):
    
    indices, values = signal
    rate = int(rate) 
    sampled_indices = indices[::rate]
    sampled_values = values[::rate]
    return sampled_indices, sampled_values

