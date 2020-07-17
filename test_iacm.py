"""
Example application for using iacm_discovery and get_causal_probabilities.

We generate some synthetic example data which we feed into the causal discovery algorithm and calculate the causal
probabilities.
"""

import numpy as np
import pandas as pd
from iacm.data_generation import generate_discrete_data, generate_continuous_data, get_discrete_random_number, \
    generate_discrete_data_confounded, generate_discrete_data_confounded2
from iacm.data_preparation import get_contingency_table_general, get_probabilities_general, \
    get_probabilities_intervention_general, get_contingency_table, get_probabilities, get_probabilities_intervention, \
    read_data
from iacm.iacm import iacm_discovery_pairwise, get_causal_probabilities, get_constraint_distribution, \
    get_constraint_distribution_general, approximate_to_causal_model, setup_causal_model_data, causal_model_definition, \
    approximate_to_causal_model_general, iacm_discovery


def test_get_constraint_distribution():
    contingency_table = get_contingency_table(pd.Series([-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1]),
                                              pd.Series([200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]), 2, 2)
    p = get_probabilities(contingency_table, 2, 2)
    p_i = get_probabilities_intervention(contingency_table, 2, 2)
    constraint_distribution = get_constraint_distribution(p, p_i, 2, 2)
    assert constraint_distribution[0] == 0.7142857142857143
    assert constraint_distribution[1] == 0.25
    assert constraint_distribution[2] == 0.3333333333333333
    assert constraint_distribution[3] == 0.13333333333333333
    assert constraint_distribution[4] == 0.13333333333333333

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1], 'y': [200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]}),
        {'x': 2, 'y': 2})
    p = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    p_i = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'], [])
    constraint_distribution = get_constraint_distribution_general(p, p_i, {'x': 2, 'y': 2}, ['x'], [])
    assert constraint_distribution[0] == 0.7142857142857143
    assert constraint_distribution[1] == 0.25
    assert constraint_distribution[2] == 0.3333333333333333
    assert constraint_distribution[3] == 0.13333333333333333
    assert constraint_distribution[4] == 0.13333333333333333

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]), 2, 2)

    p = get_probabilities(contingency_table, 2, 2)
    p_i = get_probabilities_intervention(contingency_table, 2, 2)
    constraint_distribution = get_constraint_distribution(p, p_i, 2, 2)
    assert constraint_distribution[0] == 0.6666666666666666
    assert constraint_distribution[1] == 0.1111111111111111
    assert constraint_distribution[2] == 0.26666666666666666
    assert constraint_distribution[3] == 0.13333333333333333
    assert constraint_distribution[4] == 0.06666666666666667

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]}),
        {'x': 2, 'y': 2})
    p = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    p_i = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'], [])
    constraint_distribution = get_constraint_distribution_general(p, p_i, {'x': 2, 'y': 2}, ['x'], [])
    assert constraint_distribution[0] == 0.6666666666666666
    assert constraint_distribution[1] == 0.1111111111111111
    assert constraint_distribution[2] == 0.26666666666666666
    assert constraint_distribution[3] == 0.13333333333333333
    assert constraint_distribution[4] == 0.06666666666666667

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9]), 3, 3)
    p = get_probabilities(contingency_table, 3, 3)
    p_i = get_probabilities_intervention(contingency_table, 3, 3)
    constraint_distribution = get_constraint_distribution(p, p_i, 3, 3)
    assert constraint_distribution[0] == 0.3333333333333333
    assert constraint_distribution[1] == 0.6666666666666666
    assert constraint_distribution[2] == 0.14285714285714285
    assert constraint_distribution[3] == 0.5
    assert constraint_distribution[4] == 0.16666666666666666
    assert constraint_distribution[5] == 0.14285714285714285
    assert constraint_distribution[6] == 0.15789473684210525
    assert constraint_distribution[7] == 0.10526315789473684
    assert constraint_distribution[8] == 0.05263157894736842
    assert constraint_distribution[9] == 0.05263157894736842
    assert constraint_distribution[10] == 0.21052631578947367
    assert constraint_distribution[11] == 0.05263157894736842
    assert constraint_distribution[12] == 0.05263157894736842
    assert constraint_distribution[13] == 0.05263157894736842

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9]}),
        {'x': 3, 'y': 3})
    p = get_probabilities_general(contingency_table, {'x': 3, 'y': 3})
    p_i = get_probabilities_intervention_general(contingency_table, {'x': 3, 'y': 3}, ['x'], [])
    constraint_distribution = get_constraint_distribution_general(p, p_i, {'x': 3, 'y': 3}, ['x'], [])
    assert constraint_distribution[0] == 0.3333333333333333
    assert constraint_distribution[1] == 0.6666666666666666
    assert constraint_distribution[2] == 0.14285714285714285
    assert constraint_distribution[3] == 0.5
    assert constraint_distribution[4] == 0.16666666666666666
    assert constraint_distribution[5] == 0.14285714285714285
    assert constraint_distribution[6] == 0.15789473684210525
    assert constraint_distribution[7] == 0.10526315789473684
    assert constraint_distribution[8] == 0.05263157894736842
    assert constraint_distribution[9] == 0.05263157894736842
    assert constraint_distribution[10] == 0.21052631578947367
    assert constraint_distribution[11] == 0.05263157894736842
    assert constraint_distribution[12] == 0.05263157894736842
    assert constraint_distribution[13] == 0.05263157894736842

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9],
                      'z': [1, 2, 4, 5, 7, 8, 9, 10, 15, 0]}),
        {'x': 2, 'y': 2, 'z': 2})
    p = get_probabilities_general(contingency_table, {'x': 2, 'y': 2, 'z': 2})
    p_i = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2, 'z': 2}, ['z'], [])
    constraint_distribution = get_constraint_distribution_general(p, p_i, {'x': 2, 'y': 2, 'z': 2}, ['z'], [])
    assert constraint_distribution[0] == 0.6666666666666666
    assert constraint_distribution[1] == 0.3333333333333333
    assert constraint_distribution[2] == 0.6666666666666666
    assert constraint_distribution[3] == 0.3333333333333333
    assert constraint_distribution[4] == 0.2777777777777778
    assert constraint_distribution[5] == 0.1111111111111111
    assert constraint_distribution[6] == 0.05555555555555555
    assert constraint_distribution[7] == 0.05555555555555555
    assert constraint_distribution[8] == 0.05555555555555555
    assert constraint_distribution[9] == 0.05555555555555555
    assert constraint_distribution[10] == 0.1111111111111111
    assert constraint_distribution[11] == 0.2777777777777778


def test_approximate_to_causal_model():
    contingency_table = get_contingency_table(pd.Series([-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1]),
                                              pd.Series([200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]), 2, 2)
    int_contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]), 2, 2)
    model = approximate_to_causal_model(2, 2, contingency_table, int_contingency_table, False,
                                        setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y']), False)
    assert model['GlobalError'] == 1.7914133736744933
    assert model['kl_p_tilde_p_hat'] == 1.3219280625958332
    assert model['p_tilde']['0000'] == 0.0
    assert model['p_tilde']['0001'] == 0.9999999989081332
    assert model['p_tilde']['0010'] == 0.0
    assert model['p_tilde']['0011'] == 0.0
    assert model['p_tilde']['0100'] == 0.0
    assert model['p_tilde']['0101'] == 0.0
    assert model['p_tilde']['0110'] == 0.0
    assert model['p_tilde']['0111'] == 0.0
    assert model['p_tilde']['1000'] == 4.606504032824718e-10
    assert model['p_tilde']['1001'] == 0.0
    assert model['p_tilde']['1010'] == 6.31216345913924e-10
    assert model['p_tilde']['1011'] == 0.0
    assert model['p_tilde']['1100'] == 0.0
    assert model['p_tilde']['1101'] == 0.0
    assert model['p_tilde']['1110'] == 0.0
    assert model['p_tilde']['1111'] == 0.0

    assert model['p_hat']['0000'] == 0.0
    assert model['p_hat']['0001'] == 0.2888888894773733
    assert model['p_hat']['0010'] == 0.0
    assert model['p_hat']['0011'] == 0.11111111067503891
    assert model['p_hat']['0100'] == 0.0
    assert model['p_hat']['0101'] == 0.1333333337108174
    assert model['p_hat']['0110'] == 0.0
    assert model['p_hat']['0111'] == 0.0
    assert model['p_hat']['1000'] == 1.3307678358687956e-10
    assert model['p_hat']['1001'] == 0.13333333291554553
    assert model['p_hat']['1010'] == 1.8235138939014244e-10
    assert model['p_hat']['1011'] == 3.8798548167095356e-11
    assert model['p_hat']['1100'] == 0.33333333311459296
    assert model['p_hat']['1101'] == 0.0
    assert model['p_hat']['1110'] == 1.7406370914093514e-10
    assert model['p_hat']['1111'] == 0.0

    contingency_table = get_contingency_table_general(pd.DataFrame({'x': [-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1],
                                                                    'y': [200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]}),
                                                      {'x': 2, 'y': 2})
    int_contingency_table = get_contingency_table_general(pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90],
                                                  'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]}), {'x': 2, 'y': 2})
    model = approximate_to_causal_model_general({'x': 2, 'y': 2}, contingency_table, int_contingency_table, False,
                                                setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y']))
    assert model['GlobalError'] == 1.7914133736744933
    assert model['kl_p_tilde_p_hat'] == 1.3219280625958332
    assert model['p_tilde']['0000'] == 0.0
    assert model['p_tilde']['0001'] == 0.9999999989081332
    assert model['p_tilde']['0010'] == 0.0
    assert model['p_tilde']['0011'] == 0.0
    assert model['p_tilde']['0100'] == 0.0
    assert model['p_tilde']['0101'] == 0.0
    assert model['p_tilde']['0110'] == 0.0
    assert model['p_tilde']['0111'] == 0.0
    assert model['p_tilde']['1000'] == 4.606504032824718e-10
    assert model['p_tilde']['1001'] == 0.0
    assert model['p_tilde']['1010'] == 6.31216345913924e-10
    assert model['p_tilde']['1011'] == 0.0
    assert model['p_tilde']['1100'] == 0.0
    assert model['p_tilde']['1101'] == 0.0
    assert model['p_tilde']['1110'] == 0.0
    assert model['p_tilde']['1111'] == 0.0

    assert model['p_hat']['0000'] == 0.0
    assert model['p_hat']['0001'] == 0.2888888894773733
    assert model['p_hat']['0010'] == 0.0
    assert model['p_hat']['0011'] == 0.11111111067503891
    assert model['p_hat']['0100'] == 0.0
    assert model['p_hat']['0101'] == 0.1333333337108174
    assert model['p_hat']['0110'] == 0.0
    assert model['p_hat']['0111'] == 0.0
    assert model['p_hat']['1000'] == 1.3307678358687956e-10
    assert model['p_hat']['1001'] == 0.13333333291554553
    assert model['p_hat']['1010'] == 1.8235138939014244e-10
    assert model['p_hat']['1011'] == 3.8798548167095356e-11
    assert model['p_hat']['1100'] == 0.33333333311459296
    assert model['p_hat']['1101'] == 0.0
    assert model['p_hat']['1110'] == 1.7406370914093514e-10
    assert model['p_hat']['1111'] == 0.0

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9],
                      'z': [1, 2, 4, 5, 7, 8, 9, 10, 15, 0]}), {'x': 2, 'y': 2, 'z': 2})
    int_contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10],
                      'z': [0, 1, 2, 0, 4, 3, 5, 5, 8, 1, 0]}), {'x': 2, 'y': 2, 'z': 2})
    model = approximate_to_causal_model_general({'x': 2, 'y': 2, 'z': 2}, contingency_table, int_contingency_table, False,
                                                setup_causal_model_data(base=2, causal_model=causal_model_definition['X<-Z->Y']))
    assert model['GlobalError'] == 14.479054399683527
    assert model['kl_p_tilde_p_hat'] == 0.3334029383709867


parameters = {'bins': 10,
              'nb_cluster': 2,
              'preprocess_method': 'cluster_discrete',
              'decision_criteria': 'global_error'}

structure_list = ['linear_discrete', 'nonlinear_discrete', 'linear_continuous', 'nonlinear_continuous']


def test_iacm_discovery():
    directory = "./pairs"
    file = "pair0004.txt"
    ground_truth = "X->Y"
    data = read_data(directory, file)
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y'], preserve_order=False)
    assert ground_truth in result

    data = pd.DataFrame({'X': [el[0] for el in get_discrete_random_number(500, 10)],
                         'Y': [el[0] for el in get_discrete_random_number(500, 10)]})
    ground_truth = "X|Y"
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y'], preserve_order=False)
    assert ground_truth in result
    pass


def test_iacm_discovery_confounded_hidden():
    # data = generate_discrete_data_confounded(structure='nonlinear', sample_size=1000, alphabet_size_x=4, alphabet_size_y=2, alphabet_size_z=2)
    # data[['X', 'Y']].to_csv("./test_data/confounded_hidden_1.csv", sep=" ", header=None, index=None)
    data = read_data(directory="./test_data", filename="confounded_hidden_1.csv")
    ground_truth = 'X<-[Z]->Y'
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y', 'X<-[Z]->Y'], preserve_order=False)
    assert ground_truth in result


def test_iacm_discovery_confounded():
    #data = generate_discrete_data_confounded(structure='nonlinear', sample_size=1000, alphabet_size_x=2, alphabet_size_y=2, alphabet_size_z=2)
    #data.to_csv("./test_data/confounded_observed_1.csv", sep=" ", header=None, index=None)
    directory = "./test_data"
    file = "confounded_observed_1.csv"
    ground_truth = "X<-Z->Y"
    data = pd.read_csv(directory + '/' + file, sep=" ", header=None)
    data.columns = ["X", "Y", "Z"]
    parameters = {'bins': 10,
                  'nb_cluster': -1,
                  'preprocess_method': 'cluster_discrete',
                  'decision_criteria': 'global_error'}
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=False, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y', 'X<-Z->Y'], preserve_order=False)
    assert ground_truth in result


def test_iacm_discovery_confounded_2():
    data = generate_discrete_data_confounded2(structure='nonlinear', sample_size=1000, alphabet_size_x=2, alphabet_size_y=2, alphabet_size_z=2)
    # data.to_csv("./test_data/confounded_observed_2.csv", sep=" ", header=None, index=None)
    directory = "./test_data"
    file = "confounded_observed_2.csv"
    ground_truth = "Z->X->Y"
    data = pd.read_csv(directory + '/' + file, sep=" ", header=None)
    data.columns = ["X", "Y", "Z"]
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y', 'X<-Z->Y', 'Z->X->Y'], preserve_order=False)
    assert ground_truth in result


def test_iacm_discovery_confounded_hidden_2():
    data = generate_discrete_data_confounded2(structure='nonlinear', sample_size=1000, alphabet_size_x=2,
                                              alphabet_size_y=2, alphabet_size_z=2)
    data[['X', 'Y']].to_csv("./test_data/confounded_hidden_2.csv", sep=" ", header=None, index=None)
    directory = "./test_data"
    file = "confounded_hidden_2.csv"
    ground_truth = "[Z]->X->Y"
    data = pd.read_csv(directory + '/' + file, sep=" ", header=None)
    data.columns = ["X", "Y", "Z"]
    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y', 'X|Y', 'X<-[Z]->Y', '[Z]->X->Y'], preserve_order=False)
    assert ground_truth in result

def test_iacm():
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

    res, crit = iacm_discovery_pairwise(base_x=base_x, base_y=base_y, data=data, auto_configuration=True,
                                        parameters=None, verbose=False, preserve_order=False)

    print("Ground Truth: X->Y")
    print("Pairwise IACM Causal Discovery using auto configuration: " + res + " (error: " + str(crit) + ")")

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

    result, min_kl = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, auto_configuration=True, parameters=parameters,
                                    causal_models=['X->Y','X|Y'], preserve_order=False)
    print("Ground Truth: X->Y")
    print("IACM Causal Discovery using auto configuration: " + result + " (error: " + str(min_kl) + ")")
