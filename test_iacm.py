import numpy as np
import pandas as pd

from iacm.data_generation import generate_discrete_data, generate_continuous_data
from iacm.iacm import iacm_discovery


parameters = {'bins': 2,
              'nb_cluster': 2,
              'monotone': True,
              'preprocess_method': 'cluster_discrete'}

structure_list = ['linear_discrete', 'nonlinear_discrete', 'linear_continuous', 'nonlinear_continuous']

if __name__ == '__main__':
    structure = 'nonlinear_discrete'
    sample_sizes = [100, 500, 1000]
    base = 2
    alphabet_size_x = 4
    alphabet_size_y = 4
    base_x = 2
    base_y = 2

    max_samples = sample_sizes[np.random.randint(3)]
    if 'discrete' in structure:
        data = generate_discrete_data(structure=structure, sample_size=max_samples,
                                      alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y)
    else:
        obsX, obsY, intX, intY = generate_continuous_data(structure=structure, sample_size=max_samples)
        data = pd.DataFrame({'X': np.concatenate([obsX, intX]), 'Y': np.concatenate([obsY, intY])})

    res, crit = iacm_discovery(base_x=base_x, base_y=base_y, data=data, auto_configuration=True,
                               parameters=parameters, verbose=False, preserve_order=False)

    print("Ground Truth: X->Y")
    print("IACM Causal Discovery using auto configuration: " + res + " (error: " + str(crit) + ")")