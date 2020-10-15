import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
import itertools
from typing import List, Dict, Tuple, Any, Union
from iacm.data_preparation import discretize_data, cluster_data, split_data, split_data_at_index, \
    split_at_clustered_labels, find_best_cluster, find_best_discretization, get_contingency_table_general, \
    get_probabilities_general, get_probabilities_intervention_general, split_with_balancing, split_bucket
from iacm.causal_models import base_repr, setup_causal_model_data, causal_model_definition
from iacm.metrics import get_kl_between_x_y, calc_error, get_distr_xy, kl_divergence, get_local_error

model_data = dict()
model_data['2_2_X|Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X|Y'])
model_data['2_2_X->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'])
model_data['2_2_X->Y_m_d'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'],
                                                     monotone_decr=True, monotone_incr=False)
model_data['2_2_X->Y_m_i'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'],
                                                     monotone_decr=False, monotone_incr=True)
model_data['3_3_X->Y'] = setup_causal_model_data(base=3, causal_model=causal_model_definition['X->Y'])
model_data['2_2_2_X<-Z->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X<-Z->Y'])
model_data['2_2_X<-[Z]->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X<-[Z]->Y'])
model_data['2_2_Z->X->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['Z->X->Y'])
model_data['2_2_[Z]->X->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['[Z]->X->Y'])
model_data['2_2_(X,Z)->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['(X,Z)->Y'])
model_data['2_2_(X,[Z])->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['(X,[Z])->Y'])


def get_constraint_data(list_of_distributions, constraint_patterns: List[str]) -> Dict[str, float]:
    constraint_data = dict()

    for value, pattern in zip(list_of_distributions, constraint_patterns):
        constraint_data[pattern] = value

    return constraint_data


def __get_constraint_distribution_general(p: Dict[str, float], p_i: Dict[str, Dict[str, float]], bases: Dict[str, int],
                                          intervention_variables: List[str], hidden_variables: List[str]) \
        -> List[float]:
    constraint_distr = []
    obs_vars = [o_v for o_v in bases.keys()]
    for intervention_variable in intervention_variables:
        intervention_variable = intervention_variable.lower()
        for obs_var in reversed(obs_vars):
            if obs_var == intervention_variable:
                continue
            for observation in range(1, bases[obs_var]):
                for intervention in range(bases[intervention_variable] - 1, -1, -1):
                    index = str(intervention) + '_' + str(observation)
                    constraint_distr.append(p_i[obs_var][index])

    for _ in hidden_variables:
        for obs_var in reversed(obs_vars):
            for observation in range(1, bases[obs_var]):
                for intervention in range(1, -1, -1):
                    index = str(intervention) + '_' + str(observation)
                    constraint_distr.append(p_i[obs_var][index])

    for index_combination in itertools.product(*tuple([''.join([str(v) for v in range(base - 1, -1, -1)])
                                                       for base in bases.values()])):
        if all([v == 0 for v in index_combination]):
            continue
        p_index = ''.join(index_combination)
        constraint_distr.append(p[p_index])

    return constraint_distr


def __get_model_with_causal_probabilities(constraint_data: Dict[str, float]) -> Dict[str, Any]:
    modeldata_mon_decr = __find_best_approximation_to_model(constraint_data, model_data['2_2_X->Y_m_d'])
    modeldata_mon_incr = __find_best_approximation_to_model(constraint_data, model_data['2_2_X->Y_m_i'])

    if (len(modeldata_mon_incr) == 0) and (len(modeldata_mon_decr) == 0):
        modeldata = dict()
        pn, ps, pns = 0, 0, 0
    elif len(modeldata_mon_incr) == 0:
        modeldata = modeldata_mon_decr
        pn, ps, pns = __calculate_causal_probabilities(modeldata['p_tilde'], 'decrease')
    elif len(modeldata_mon_decr) == 0:
        modeldata = modeldata_mon_incr
        pn, ps, pns = __calculate_causal_probabilities(modeldata['p_tilde'], 'increase')
    elif modeldata_mon_incr['GlobalError'] < modeldata_mon_decr['GlobalError']:
        modeldata = modeldata_mon_incr
        pn, ps, pns = __calculate_causal_probabilities(modeldata['p_tilde'], 'increase')
    else:
        modeldata = modeldata_mon_decr
        pn, ps, pns = __calculate_causal_probabilities(modeldata['p_tilde'], 'decrease')

    modeldata['PN'] = pn
    modeldata['PS'] = ps
    modeldata['PNS'] = pns
    return modeldata


def __approximate_to_causal_model_general(bases: Dict[str, int], observation_contingency_table: np.ndarray,
                                          intervention_contingency_table: np.ndarray, monotone: bool,
                                          causal_model: Dict[str, Any]) -> Dict[str, Any]:
    p = get_probabilities_general(observation_contingency_table, bases)
    p_intervention = get_probabilities_intervention_general(intervention_contingency_table, bases,
                                                            causal_model['interventional_variables'],
                                                            causal_model['hidden_variables'])
    constraint_distributions = __get_constraint_distribution_general(p, p_intervention, bases,
                                                                     causal_model['interventional_variables'],
                                                                     causal_model['hidden_variables'])
    constraint_data = get_constraint_data(list_of_distributions=constraint_distributions,
                                          constraint_patterns=causal_model['constraint_patterns'])

    if monotone and all([base == 2 for base in bases.values()]):
        modeldata = __get_model_with_causal_probabilities(constraint_data)
    else:
        modeldata = __find_best_approximation_to_model(constraint_data, causal_model)
    return modeldata


def __calculate_causal_probabilities(p_tilde: Dict[str, float], direction: str) -> Tuple[float, float, float]:
    pxy = p_tilde['1100'] + p_tilde['1101'] + p_tilde['1110'] + p_tilde['1111']
    pxny = p_tilde['1000'] + p_tilde['1001'] + p_tilde['1010'] + p_tilde['1011']
    pnxy = p_tilde['0100'] + p_tilde['0101'] + p_tilde['0110'] + p_tilde['0111']
    pnxny = p_tilde['0000'] + p_tilde['0001'] + p_tilde['0010'] + p_tilde['0011']
    py = pxy + pnxy
    py_x = p_tilde['0001'] + p_tilde['0011'] + p_tilde['0101'] + p_tilde['0111'] + p_tilde['1001'] + \
        p_tilde['1011'] + p_tilde['1101'] + p_tilde['1111']
    py_nx = p_tilde['0010'] + p_tilde['0011'] + p_tilde['0110'] + p_tilde['0111'] + p_tilde['1010'] + \
        p_tilde['1011'] + p_tilde['1110'] + p_tilde['1111']
    return __get_causal_prob_monotony(py=py, pxy=pxy, pnxy=pnxy, pxny=pxny, pnxny=pnxny, py_x=py_x, py_nx=py_nx,
                                      direction=direction)


def __get_causal_prob_monotony(py: float, pxy: float, pnxy: float, pxny: float, pnxny: float, py_x: float, py_nx: float,
                               direction: str) -> Tuple[float, float, float]:
    if direction == 'increase':
        pns = py_x - py_nx
        if pxy > 0:
            pn = (py - py_nx) / pxy
        else:
            pn = 0
        if pnxny > 0:
            ps = (py_x - py) / pnxny
        else:
            ps = 0
    else:
        pns = py_nx - py_x
        if pnxy > 0:
            pny = 1 - py
            pny_x = 1 - py_x
            pn = (pny_x - pny) / pnxy
        else:
            pn = 0
        if pxny > 0:
            pny = 1 - py
            pny_nx = 1 - py_nx
            ps = (pny - pny_nx) / pxny
        else:
            ps = 0

    return pn, ps, pns


def __find_best_approximation_to_model(constraint_data: Dict[str, float], meta_data: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    size_prob = meta_data['size_prob']
    base = meta_data['base_x']
    nb_variables = meta_data['nb_variables']
    B = meta_data['B']
    d = meta_data['d'].copy()
    F = meta_data['F']
    c = meta_data['c']
    s_codes = meta_data['S_codes'].copy()
    weights_for_objective_function = meta_data['obj_fct_weights']
    if len(weights_for_objective_function) == 0:
        weights_for_objective_function = [{0: 1}]

    b = np.array([1.0] + [constraint_data[pattern] for pattern in meta_data['constraint_patterns']])

    v_max = 0
    x_max = None
    max_weights = None
    for weighted_elements in weights_for_objective_function:
        for index, weight in weighted_elements.items():
            d[index] = weight*d[index]

        # create and run the solver
        x = cp.Variable(shape=size_prob)
        obj = cp.Maximize(cp.sum(d * x))
        constraints = [B * x == b,
                       F * x >= c]
        prob = cp.Problem(obj, constraints)
        v = prob.solve(solver=cp.SCS, verbose=False)
        if v is not None and v > v_max:
            v_max = v
            x_max = x.value
            max_weights = weighted_elements

    # get the solution
    if x_max is None:
        return result

    for index, weight in max_weights.items():
        if weight == 0:
            code = base_repr(index, base, nb_variables)
            if code in s_codes:
                s_codes.remove(code)

    simplex_res = x_max

    p_hat = dict()

    for i in range(0, len(simplex_res)):
        code = base_repr(i, base, nb_variables)
        p_hat[code] = max(simplex_res[i], 0.0)

    s = sum([p_hat[code] for code in s_codes])
    if s == 0:
        return result
    # normalized distribution
    p_tilde = dict()
    for code, value in p_hat.items():
        if code in s_codes:
            p_tilde[code] = value / s
        else:
            p_tilde[code] = 0

    result['p_tilde'] = p_tilde
    result['p_hat'] = p_hat
    result["GlobalError"] = log2(1 / s)
    result["kl_p_tilde_p_hat"] = kl_divergence(get_distr_xy(p_tilde, meta_data['base_x'], meta_data['base_y']),
                                               get_distr_xy(p_hat, meta_data['base_x'], meta_data['base_y']))

    return result


def test_model(bases: Dict[str, int], observation_data: pd.DataFrame, intervention_data: pd.DataFrame, monotone: bool,
               causal_model: Dict[str, Any]) -> Dict[str, Any]:
    observation_contingency_table = get_contingency_table_general(observation_data, bases)
    if len(causal_model['hidden_variables']) > 0:
        # cluster = SpectralClustering(n_clusters=2).fit(intervention_data)
        zero_len = int(intervention_data.shape[0] / 2)
        one_len = intervention_data.shape[0] - zero_len
        intervention_data['z'] = ([0]*zero_len + [1]*one_len)  # cluster.labels_
        new_bases = bases.copy()
        new_bases['z'] = 2
        intervention_contingency_table = get_contingency_table_general(intervention_data, new_bases)
    else:
        intervention_contingency_table = get_contingency_table_general(intervention_data, bases)

    return __approximate_to_causal_model_general(bases, observation_contingency_table, intervention_contingency_table,
                                                 monotone, causal_model)


def __preprocessing(data: pd.DataFrame, V: str, intervention_column: str, preserve_order: bool,
                    parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observation_variables = [v for v in V.split(',') if '_' not in v]
    if parameters['preprocess_method'] == 'none':
        split_idx = int(data.shape[0] / 2)
        obs_pdf, int_pdf = split_data_at_index(data, split_idx, observation_variables)
    elif parameters['preprocess_method'] == 'split':
        obs_pdf, int_pdf = split_bucket(data, intervention_column, observation_variables)
    elif parameters['preprocess_method'] == 'split_and_balance':
        obs_pdf, int_pdf = split_with_balancing(data, intervention_column, observation_variables)
    elif parameters['preprocess_method'] == 'cluster_discrete':
        if parameters['nb_cluster'] == -1:
            (_, clustered_data), best_nb_clusters = find_best_cluster(data, intervention_column, observation_variables,
                                                                      parameters['bins'])
        else:
            _, clustered_data = cluster_data(data, intervention_column, observation_variables, parameters['nb_cluster'])
            best_nb_clusters = parameters['nb_cluster']
        if parameters['bins'] == -1:
            disc_data = find_best_discretization(data, observation_variables)
        else:
            disc_data = discretize_data(data, parameters['bins'], observation_variables)
        disc_data['labels'] = clustered_data['labels']
        if preserve_order:
            obs_pdf, int_pdf, i_max = split_data(disc_data, 'labels', observation_variables, sort_data=False)
        else:
            obs_pdf, int_pdf = split_at_clustered_labels(disc_data, intervention_column, observation_variables,
                                                         best_nb_clusters)
    elif parameters['preprocess_method'] == 'discrete_cluster':
        if parameters['bins'] == -1:
            disc_data = find_best_discretization(data, observation_variables)
        else:
            disc_data = discretize_data(data, parameters['bins'], observation_variables)
        if parameters['nb_cluster'] == -1:
            ((obs_pdf, int_pdf), clustered_data), _ = find_best_cluster(disc_data, intervention_column,
                                                                        observation_variables, parameters['bins'])
        else:
            (obs_pdf, int_pdf), clustered_data = cluster_data(data, intervention_column, observation_variables,
                                                              parameters['nb_cluster'])
        if preserve_order:
            obs_pdf, int_pdf, i_max = split_data(clustered_data, 'labels', observation_variables, sort_data=False)
    else:
        raise Exception('Preprocessing method not known')

    return obs_pdf, int_pdf


def __find_best_binary_model(bases: Dict[str, int], observation_data: pd.DataFrame, intervention_data: pd.DataFrame,
                             causal_model: Dict[str, Any]) -> Dict[str, Any]:
    model_x_to_y_monotone = test_model(bases, observation_data=observation_data, intervention_data=intervention_data,
                                       monotone=True, causal_model=causal_model)
    model_x_to_y_not_monotone = test_model(bases, observation_data=observation_data,
                                           intervention_data=intervention_data, monotone=False,
                                           causal_model=causal_model)
    error_x_to_y, error_y_to_x, error_tolerance = __calc_decision_criteria(model_x_to_y_monotone,
                                                                           model_x_to_y_not_monotone, bases)
    if abs(error_x_to_y - error_y_to_x) < error_tolerance:
        model_x_to_y = model_x_to_y_monotone
    else:
        model_x_to_y = model_x_to_y_not_monotone
    return model_x_to_y


def __calc_decision_criteria(model_x_to_y: Dict[str, Any], model_y_to_x: Dict[str, Any], bases: Dict[str, int],
                             strategy: str = 'global_error') -> Tuple[float, float, float]:
    error_tolerance = 0
    error_x_to_y = error_y_to_x = np.inf
    if strategy == 'global_error':
        error_x_to_y = calc_error(model_x_to_y)
        error_y_to_x = calc_error(model_y_to_x)
        error_tolerance = 0.01
    elif strategy == 'kl_xy':
        error_x_to_y = get_kl_between_x_y(model_x_to_y, bases['x'], bases['y'])
        error_y_to_x = get_kl_between_x_y(model_y_to_x, bases['x'], bases['y'])
        error_tolerance = 1.0e-12
    elif strategy == 'local_xy_error':
        error_x_to_y = get_local_error(model_x_to_y)
        error_y_to_x = get_local_error(model_y_to_x)
        error_tolerance = 1.0e-12

    return error_x_to_y, error_y_to_x, error_tolerance


def __decide_best_model(model_x_to_y: Dict[str, Any], model_y_to_x: Dict[str, Any], monotone: bool, verbose: bool,
                        bases: Dict[str, int], strategy: str = 'global_error', tolerance: float = 1.0e-05) \
        -> Dict[str, Any]:
    best_model = dict()
    error_x_to_y, error_y_to_x, error_tolerance = __calc_decision_criteria(model_x_to_y, model_y_to_x, bases, strategy)
    if verbose:
        print("total Error X -> Y: " + str(error_x_to_y))
        print("total Error Y -> X: " + str(error_y_to_x))

    if abs(error_x_to_y - error_y_to_x) <= error_tolerance:
        if monotone and ('PNS' in model_x_to_y and 'PNS' in model_y_to_x):
            pns_xto_y = model_x_to_y['PNS']
            pns_yto_x = model_y_to_x['PNS']
            if abs(pns_xto_y-pns_yto_x) < tolerance:
                res = "no decision"
                error_ratio = 1
            elif pns_xto_y > pns_yto_x:
                res = "X->Y"
                error_ratio = abs(pns_xto_y - pns_yto_x)
            else:
                res = "Y->X"
                error_ratio = abs(pns_xto_y - pns_yto_x)
        else:
            res = "no decision"
            error_ratio = 1
    elif error_x_to_y < error_y_to_x:
        if verbose:
            print("X -> Y")
        res = "X->Y"
        error_ratio = min(error_x_to_y, error_y_to_x) / max(error_x_to_y, error_y_to_x)
    else:
        # errorXtoY > errorYtoX:
        if verbose:
            print("Y -> X")
        res = "Y->X"
        error_ratio = min(error_x_to_y, error_y_to_x) / max(error_x_to_y, error_y_to_x)

    best_model['result'] = res
    best_model['error_ratio'] = error_ratio
    best_model['min_error'] = min(error_x_to_y, error_y_to_x)
    return best_model


def iacm_discovery(bases: Dict[str, int], data: pd.DataFrame, parameters: Dict[str, Any],
                   causal_models: List[str], preserve_order: bool = False) \
        -> Tuple[str, float]:
    """
    Causal discovery among all models in the causal_models list using information-theoretic approximation to causal
    models (the list of causal models can be more than just X->Y and Y->X). The function decides using the local
    approximation error the best model and return it together with error metric.
    Args:
        :param bases: size of alphabet for the embedded variables. Dictionary with keys as variable names and values as
        alphabet size (e.g. {'x': 2, 'y': 2})
        :param data: pandas dataframe with columns that are necessary for the causal models you want to test for (e.g.
        'X' and 'Y' for X->Y)
        :param parameters: Dictionary that contains the preprocessing method together with its parameters and the
        criteria used for deciding the causal direction. The entry 'preprocess_method' specifies the method used
        for preprocessing of the data. Currently there are the following possibilities:
            'none' : no preprocessing, split data in observation and intervention sets at half.
            'split': filter data by each intervention on X or Y and draw without replacement from each filtered subset
            data points for observational and interventional data to obtain equally sized sets for observations and
            interventions with similar variance.
            'split_and_balance': works like 'split' but draws with replacement such that each intervention subset has
            equal size. This method tries to balance out imbalanced intervention subsets.
        The entry 'decision_criteria' specifies which decision criteria is used in order to decide the causal direction.
        You can either use 'global_error' or 'local_xy_error' as a decision criterion.
        See paper and Supplementary Material for more details.
        :param causal_models: List of causal models that should be tested. Currently the following models are supported:
            X->Y, Y->X, X->Y_monotone, Y->X_monotone: if one of these models is in the list the other direction will
            also be included, so actually the following models are evaluated:
             X->Y, Y->X furthermore if all bases are binary then also X->Y_monotone, Y->X_monotone are included into
             the evaluation.
            X|Y: this model represents a causal model for X, Y without any causal link (X, Y are assumed to be
            independent). Currently, this is not stable yet. We recommend for testing independency between X and Y to
            use other methods like chi2-test, HSIC-test and other more adequate measures.
            X<-Z->Y: model with observed confounder Z. This model assumes that we have data from interventions on Z.
            X<-[Z]->Y: model with hidden confounder Z. This model assumes that we have data from hidden interventions
            on Z.
            Z->X->Y: model with observed mediator X. This model assumes that we have data from interventions on Z and X.
            [Z]->X->Y: model with hidden variable Z. This model assumes that we have data from (hidden) interventions
            on Z and X.
            (X,Z)->Y: model with two observed causes for Y. This model assumes that we have data from interventions on
            X and Z. Not yet implemented.
            (X,[Z])->Y: model with one observed causes and one hidden causes Z for Y. This model assumes that we have
            data from interventions on X and Z. Not yet implemented.
        :param preserve_order: If True, then the order in the data will be preserved (for example if you feed in
        timeseries data). Especially during data preprocessing no sorting or any other reordering will happen.

    Returns:
        Tuple with best causal model and its corresponding metric.
    """
    base_index = '_'.join([str(v) for v in bases.values()])
    models = dict()
    for causal_model in causal_models:
        causal_model_index = base_index + "_" + causal_model
        if causal_model == 'X->Y' or causal_model == 'Y->X':
            x_obs_pdf, x_int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                                   intervention_column='X', preserve_order=preserve_order,
                                                   parameters=parameters)
            y_obs_pdf, y_int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                                   intervention_column='Y', preserve_order=preserve_order,
                                                   parameters=parameters)
            models[causal_model] = test_model(bases, x_obs_pdf, x_int_pdf, False, model_data[causal_model_index])
            y_obs_pdf = pd.DataFrame({'x': y_obs_pdf['y'], 'y': y_obs_pdf['x']})
            y_int_pdf = pd.DataFrame({'x': y_int_pdf['y'], 'y': y_int_pdf['x']})
            models['Y->X'] = test_model(bases, y_obs_pdf, y_int_pdf, False, model_data[causal_model_index])
            if all([base == 2 for base in bases.values()]):
                models[causal_model + '_monotone'] = test_model(bases, x_obs_pdf, x_int_pdf, True,
                                                                model_data[causal_model_index])
                models['Y->X_monotone'] = test_model(bases, y_obs_pdf, y_int_pdf, True, model_data[causal_model_index])
        elif causal_model == 'X|Y':
            x_obs_pdf, x_int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                                   intervention_column='X', preserve_order=preserve_order,
                                                   parameters=parameters)
            y_obs_pdf, y_int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                                   intervention_column='Y', preserve_order=preserve_order,
                                                   parameters=parameters)
            obs_pdf = pd.DataFrame()
            obs_pdf['x'] = pd.concat([x_obs_pdf['x'], y_obs_pdf['x']], axis=0)
            obs_pdf['y'] = pd.concat([x_obs_pdf['y'], y_obs_pdf['y']], axis=0)
            int_pdf = pd.DataFrame()
            int_pdf['x'] = pd.concat([x_int_pdf['x'], y_int_pdf['x']], axis=0)
            int_pdf['y'] = pd.concat([x_int_pdf['y'], y_int_pdf['y']], axis=0)
            models[causal_model] = test_model(bases, obs_pdf, int_pdf, False, model_data[causal_model_index])
        elif causal_model == 'X<-Z->Y':
            new_bases = bases
            new_bases['z'] = 2
            causal_model_index = '_'.join([str(v) for v in bases.values()]) + "_" + causal_model
            obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                               intervention_column='Z', preserve_order=preserve_order,
                                               parameters=parameters)
            models[causal_model] = test_model(new_bases, obs_pdf, int_pdf, False, model_data[causal_model_index])
        elif causal_model == 'X<-[Z]->Y':
            obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                               intervention_column='', preserve_order=preserve_order,
                                               parameters=parameters)
            models[causal_model] = test_model(bases, obs_pdf, int_pdf, False, model_data[causal_model_index])
        elif causal_model == 'Z->X->Y':
            obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                               intervention_column='Z', preserve_order=preserve_order,
                                               parameters=parameters)
            models[causal_model] = test_model(bases, obs_pdf, int_pdf, False, model_data[causal_model_index])
        elif causal_model == '[Z]->X->Y':
            obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition[causal_model]['V'],
                                               intervention_column='', preserve_order=preserve_order,
                                               parameters=parameters)
            models[causal_model] = test_model(bases, obs_pdf, int_pdf, False, model_data[causal_model_index])
        elif causal_model == '(X,Z)->Y':
            pass
        elif causal_model == '(X,[Z])->Y':
            pass

    min_kl = np.inf
    best_model = ""
    for model_key, model in models.items():
        if 'kl_p_tilde_p_hat' in model:
            if min_kl > model['kl_p_tilde_p_hat']:
                min_kl = model['kl_p_tilde_p_hat']
                best_model = model_key

    return best_model, min_kl


def iacm_discovery_pairwise(base_x: int, base_y: int, data: pd.DataFrame, parameters: Dict[str, Any],
                            verbose: bool = False, preserve_order: bool = False) \
        -> Union[Tuple[Tuple[str, str], Tuple[float, float]], Tuple[str, float]]:
    """
    Pairwise causal discovery using information-theoretic approximation to causal models. The algorithm decides whether
    X causes Y or Y causes X or return no decision if no direction is preferred. Sample data from X and Y are given as
    columns in the data dataframe. The different data pairs are assumed to be i.i.d.

    :param base_x: size of alphabet for embedded variable X
    :param base_y: size of alphabet for embedded variable Y
    :param data: pandas dataframe with columns 'X' and 'Y'
    :param parameters: Dictionary that contains the preprocessing method together with its parameters and the
    criteria used for deciding the causal direction. The entry 'preprocess_method' specifies the method used
    for preprocessing of the data. Currently there are the following possibilities:
        'none' : no preprocessing, split data in observation and intervention sets at half.
        'split': filter data by each intervention on X or Y and draw without replacement from each filtered subset
        data points for observational and interventional data to obtain equally sized sets for observations and
        interventions with similar variance.
        'split_and_balance': works like 'split' but draws with replacement such that each intervention subset has
        equal size. This method tries to balance out imbalanced intervention subsets.
    The entry 'decision_criteria' specifies which decision criteria is used in order to decide the causal direction.
    You can either use 'global_error' or 'local_xy_error' as a decision criterion.
    See paper and Supplementary Material for more details.
    :param verbose: Flag to turn on and off detailed output.
    :param preserve_order: If True, then the order in the data will be preserved (for example if you feed in timeseries
    data). Especially during data preprocessing no sorting or any other reordering will happen.
    :return: Depending on decision_criteria entry in parameters return direction in form of a string "X->Y", "Y->X",
    "no decision" with the value of its decision_criteria. If decision_criteria is not 'global_error' or
    'local_xy_error' it will return results based on both criteria in form of a tuple of tuples.
    """
    x_obs_pdf, x_int_pdf = __preprocessing(data=data, V=causal_model_definition['X->Y']['V'], intervention_column='X',
                                           preserve_order=preserve_order, parameters=parameters)
    y_obs_pdf, y_int_pdf = __preprocessing(data=data, V=causal_model_definition['X->Y']['V'], intervention_column='Y',
                                           preserve_order=preserve_order, parameters=parameters)
    causal_model_index = f'{base_x}_{base_y}_X->Y'
    causal_model = model_data[causal_model_index]
    bases_x_to_y = {'x': base_x, 'y': base_y}
    bases_y_to_x = {'x': base_y, 'y': base_x}
    if base_x == base_y == 2:
        model_x_to_y = __find_best_binary_model(bases_x_to_y, x_obs_pdf, x_int_pdf, causal_model)
        y_obs_pdf = pd.DataFrame({'x': y_obs_pdf['y'], 'y': y_obs_pdf['x']})
        y_int_pdf = pd.DataFrame({'x': y_int_pdf['y'], 'y': y_int_pdf['x']})
        model_y_to_x = __find_best_binary_model(bases_y_to_x, y_obs_pdf, y_int_pdf, causal_model)
    else:
        model_x_to_y = test_model(bases_x_to_y, x_obs_pdf, x_int_pdf, False, causal_model)
        y_obs_pdf = pd.DataFrame({'x': y_obs_pdf['y'], 'y': y_obs_pdf['x']})
        y_int_pdf = pd.DataFrame({'x': y_int_pdf['y'], 'y': y_int_pdf['x']})
        model_y_to_x = test_model(bases_y_to_x, y_obs_pdf, y_int_pdf, False, causal_model)

    monotone = base_x == base_y == 2
    if parameters['decision_criteria'] == 'global_error':
        result = __decide_best_model(model_x_to_y, model_y_to_x, monotone, verbose, bases_x_to_y, 'global_error')
    elif parameters['decision_criteria'] == 'local_xy_error':
        result = __decide_best_model(model_x_to_y, model_y_to_x, monotone, verbose, bases_x_to_y, 'local_xy_error')
    else:
        return 'no decision', -1

    return result['result'], result['min_error']


def iacm_discovery_timeseries(base_x: int, base_y: int, data: pd.DataFrame, parameters: Dict[str, Any], max_lag: int,
                              verbose: bool = False):
    """
    Pairwise causal discovery using information-theoretic approximation to causal models. The algorithm decides whether
    X causes Y or Y causes X or return no decision if no direction is preferred. Sample data from X and Y are given as
    columns in the data dataframe. The different data pairs are assumed to be timeseries data. Therefore the
    preprocessing will keep the order of the data as it is.
    :param base_x: size of alphabet for embedded variable X
    :param base_y: size of alphabet for embedded variable Y
    :param data: pandas dataframe with columns 'X' and 'Y'
    :param parameters: Dictionary that contains the preprocessing method together with its parameters and the
    criteria used for deciding the causal direction. The entry 'preprocess_method' specifies the method used
    for preprocessing of the data. Currently there are the following possibilities:
        'none' : no preprocessing, split data in observation and intervention sets at half.
        'split': filter data by each intervention on X or Y and draw without replacement from each filtered subset
        data points for observational and interventional data to obtain equally sized sets for observations and
        interventions with similar variance.
        'split_and_balance': works like 'split' but draws with replacement such that each intervention subset has
        equal size. This method tries to balance out imbalanced intervention subsets.
    The entry 'decision_criteria' specifies which decision criteria is used in order to decide the causal direction.
    You can either use 'global_error' or 'local_xy_error' as a decision criterion.
    See paper and Supplementary Material for more details.
    :param max_lag: maximum lag that should be considered.
    :param verbose: Flag to enable detailed prints.
    :return: Depending on decision_criteria entry in parameters return direction in form of a string "X->Y", "Y->X",
    "no decision" with the value of its decision_criteria.
    """
    timeseries_result = dict()
    t = data.shape[0]
    tmp_data = data.copy()
    for lag in range(0, min(max_lag, t-1)):
        tmp_data['X'] = data['X'][:t - lag].reset_index(drop=True)
        tmp_data['Y'] = data['Y'][lag:].reset_index(drop=True)
        tmp_data = tmp_data.dropna().reset_index(drop=True)
        res, crit = iacm_discovery_pairwise(base_x=base_x, base_y=base_y, data=tmp_data, parameters=parameters,
                                            verbose=verbose, preserve_order=True)
        timeseries_result[lag] = dict()
        timeseries_result[lag]['result'] = res
        timeseries_result[lag]['crit'] = crit

    min_crit = 1
    best_res = "no decision"
    for lag, res_dict in timeseries_result.items():
        if "no decision" not in res_dict['result'] and res_dict['crit'] < min_crit:
            min_crit = res_dict['crit']
            best_res = res_dict['result']

    return best_res, min_crit


def get_causal_probabilities(data: pd.DataFrame, parameters: Dict[str, Any], direction_x_to_y: bool) \
        -> Tuple[float, float, float, float]:
    """
    Function that calculate probabilities of how necessary, sufficient, necessary and sufficient a cause is for an
    effect (PN, PS, PNS). It approximates to a monotone causal model X->Y and calculates PN, PS, PNS.
    :param data: pandas dataframe with columns 'X' and 'Y'
    :param parameters: Dictionary that contains the preprocessing method together with its parameters and the
    criteria used for deciding the causal direction. The entry 'preprocess_method' specifies the method used
    for preprocessing of the data. Currently there are the following possibilities:
        'none' : no preprocessing, split data in observation and intervention sets at half.
        'split': filter data by each intervention on X or Y and draw without replacement from each filtered subset
        data points for observational and interventional data to obtain equally sized sets for observations and
        interventions with similar variance.
        'split_and_balance': works like 'split' but draws with replacement such that each intervention subset has
        equal size. This method tries to balance out imbalanced intervention subsets.
    The entry 'decision_criteria' specifies which decision criteria is used in order to decide the causal direction.
    You can either use 'global_error' or 'local_xy_error' as a decision criterion.
    See paper and Supplementary Material for more details.
    :param direction_x_to_y: If True an approximation to a monotone causal model X->Y will be done, otherwise to a
    monotone causal model Y->X will be done.
    :return: PN, PS, PNS and approximation error to the causal models as a tuple.
    """
    causal_model = model_data['2_2_X->Y']
    if direction_x_to_y:
        obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition['X->Y']['V'], intervention_column='X',
                                           preserve_order=False, parameters=parameters)
    else:
        obs_pdf, int_pdf = __preprocessing(data=data, V=causal_model_definition['X->Y']['V'], intervention_column='Y',
                                           preserve_order=False, parameters=parameters)
        obs_pdf = pd.DataFrame({'x': obs_pdf['y'], 'y': obs_pdf['x']})
        int_pdf = pd.DataFrame({'x': int_pdf['y'], 'y': int_pdf['x']})

    model_x_to_y_monotone = test_model({'x': 2, 'y': 2}, observation_data=obs_pdf, intervention_data=int_pdf,
                                       monotone=True, causal_model=causal_model)
    error = model_x_to_y_monotone['GlobalError'] if parameters['decision_criteria'] == 'GlobalError' \
        else model_x_to_y_monotone['kl_p_tilde_p_hat']
    return model_x_to_y_monotone['PN'], model_x_to_y_monotone['PS'], model_x_to_y_monotone['PNS'], error
