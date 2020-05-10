import numpy as np

def get_bootstrap_indices(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples,))
    return indices
