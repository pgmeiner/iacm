import numpy as np
import pandas as pd

from iacm.iacm import iacm_discovery_pairwise, get_causal_probabilities, iacm_discovery
from iacm.data_generation import generate_discrete_data, generate_continuous_data

parameters = {'preprocess_method': 'none',
              'decision_criteria': 'global_error'}


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

    data = pd.read_csv('test/test_data/x_causes_y.csv', sep=" ", header=None)
    data.columns = ["X", "Y"]
    # call pairwise causal discovery
    res, error = iacm_discovery_pairwise(base_x=base_x, base_y=base_y, data=data, parameters=parameters,
                                         verbose=False, preserve_order=False)

    print("Ground Truth: X->Y")
    print("Pairwise IACM Causal Discovery using default configuration results: " + res + " (error: " + str(error) + ")")

    # calculate causal probabilities for the model X->Y
    pn, ps, pns, error = get_causal_probabilities(data=data, parameters=parameters, direction_x_to_y=True)
    print("Approximation to best monotone model X->Y gives error " + str(error) + " and ")
    print("PN: " + str(pn))
    print("PS: " + str(ps))
    print("PNS: " + str(pns))

    # calculate causal probabilities for the opposite direction Y->X
    pn, ps, pns, error = get_causal_probabilities(data=data, parameters=parameters, direction_x_to_y=False)
    print("Approximation to best monotone model Y->X gives error " + str(error) + " and ")
    print("PN: " + str(pn))
    print("PS: " + str(ps))
    print("PNS: " + str(pns))

    ground_truth = "X<-Z->Y"
    data = pd.read_csv('test/test_data/confounded_observed_1.csv', sep=" ", header=None)
    data.columns = ["X", "Y", "Z"]
    result, error = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, parameters=parameters,
                                   causal_models=['X->Y', 'X<-Z->Y', 'X|Y'])
    print("Ground Truth: X<-Z->Y")
    print("IACM Causal Discovery using default configuration results: " + result + " (error: " + str(error) + ")")
