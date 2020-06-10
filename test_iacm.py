"""
Example application for using iacm_discovery and get_causal_probabilities.

We generate some synthetic example data which we feed into the causal discovery algorithm and calculate the causal
probabilities.
"""

import numpy as np

from iacm.data_generation import generate_discrete_data, generate_continuous_data
from iacm.iacm import iacm_discovery, get_causal_probabilities


parameters = {'bins': 10,
              'nb_cluster': 2,
              'preprocess_method': 'cluster_discrete',
              'decision_criteria': 'global_error'}

structure_list = ['linear_discrete', 'nonlinear_discrete', 'linear_continuous', 'nonlinear_continuous']

if __name__ == '__main__':
    structure = 'nonlinear_discrete'
    sample_sizes = [100, 500, 1000]
    base = 2
    alphabet_size_x = 4
    alphabet_size_y = 4
    base_x = 2
    base_y = 2

    # Generate some example data
    max_samples = sample_sizes[np.random.randint(3)]
    if 'discrete' in structure:
        data = generate_discrete_data(structure=structure, sample_size=max_samples,
                                      alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y)
    else:
        data = generate_continuous_data(structure=structure, sample_size=max_samples)

    res, crit = iacm_discovery(base_x=base_x, base_y=base_y, data=data, auto_configuration=True,
                               parameters=None, verbose=False, preserve_order=False)

    print("Ground Truth: X->Y")
    print("IACM Causal Discovery using auto configuration: " + res + " (error: " + str(crit) + ")")

    pn, ps, pns, error = get_causal_probabilities(data=data, auto_configuration=True, parameters=parameters,
                                                  direction_x_to_y=True, preserve_order=False)
    print("Approximation to best monotone model X->Y gives error " + str(error) + " and ")
    print("PN: " + str(pn))
    print("PS: " + str(ps))
    print("PNS: " + str(pns))

    pn, ps, pns, error = get_causal_probabilities(data=data, auto_configuration=True, parameters=parameters,
                                                  direction_x_to_y=False, preserve_order=False)
    print("Approximation to best monotone model Y->X gives error " + str(error) + " and ")
    print("PN: " + str(pn))
    print("PS: " + str(ps))
    print("PNS: " + str(pns))
