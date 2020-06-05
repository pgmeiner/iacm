import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from iacm.data_preparation import get_probabilities, get_probabilities_intervention, write_contingency_table, \
    get_contingency_table, discretize_data, cluster_data, split_data, split_data_at_index, split_at_clustered_labels
from iacm.meta_data import setup_meta_data, base_repr
from typing import List, Dict, Tuple, Any
from iacm.metrics import kl_divergence

meta_data = dict()
meta_data['2_2'] = setup_meta_data(base=2, nb_variables=4)
meta_data['2_2_m_d'] = setup_meta_data(base=2, nb_variables=4, monotone_decr=True, monotone_incr=False)
meta_data['2_2_m_i'] = setup_meta_data(base=2, nb_variables=4, monotone_decr=False, monotone_incr=True)
meta_data['3_3'] = setup_meta_data(base=3, nb_variables=5)
meta_data['4_4'] = setup_meta_data(base=4, nb_variables=6)


def get_constraint_data(base_x: int, base_y: int, list_of_distributions) -> Dict[str, float]:
    constraint_data = dict()

    for value, pattern in zip(list_of_distributions, meta_data[str(base_x) + '_' + str(base_y)]['constraint_patterns']):
        constraint_data[pattern] = value

    return constraint_data


def get_constraint_distribution(p: Dict[str, float], p_i: Dict[str, float], base_x: int, base_y: int) -> List[float]:
    constraint_distr = []
    for value_y in range(1, base_y):
        for value_x in range(base_x-1, -1, -1):
            index = str(value_x) + '_' + str(value_y)
            constraint_distr.append(p_i[index])

    for value_x in range(base_x-1, -1, -1):
        for value_y in range(base_y-1, -1, -1):
            if (value_x == 0) and (value_y == 0):
                continue
            index = str(value_x) + str(value_y)
            constraint_distr.append(p[index])

    return constraint_distr


def approximate_to_causal_model(base_x: int, base_y: int,
                                observation_contingency_table: List[List[int]],
                                intervention_contingency_table: List[List[int]],
                                monotone: bool, verbose: bool) -> Dict[str, Any]:
    p = get_probabilities(observation_contingency_table, base_x, base_y)
    p_intervention = get_probabilities_intervention(intervention_contingency_table, base_x, base_y)
    constraint_distributions = get_constraint_distribution(p, p_intervention, base_x, base_y)
    if verbose:
        for key, value in p.items():
            print(key + ":" + str(value))
        for key, value in p_intervention.items():
            print(key + ":" + str(value))
    constraint_data = get_constraint_data(base_x=base_x, base_y=base_y, list_of_distributions=constraint_distributions)

    meta_data_idx = str(base_x) + '_' + str(base_y)
    if monotone and base_x == 2 and base_y == 2:
        modeldata_mon_decr = find_best_approximation_to_model(constraint_data, meta_data[meta_data_idx + '_m_d'])
        modeldata_mon_incr = find_best_approximation_to_model(constraint_data, meta_data[meta_data_idx + '_m_i'])

        if (len(modeldata_mon_incr) == 0) and (len(modeldata_mon_decr) == 0):
            modeldata = dict()
            pn, ps, pns = 0, 0, 0
        elif len(modeldata_mon_incr) == 0:
            modeldata = modeldata_mon_decr
            pn, ps, pns = calculate_causal_probabilities(modeldata['p_tilde'], 'decrease')
        elif len(modeldata_mon_decr) == 0:
            modeldata = modeldata_mon_incr
            pn, ps, pns = calculate_causal_probabilities(modeldata['p_tilde'], 'increase')
        elif modeldata_mon_incr['GlobalError'] < modeldata_mon_decr['GlobalError']:
            modeldata = modeldata_mon_incr
            pn, ps, pns = calculate_causal_probabilities(modeldata['p_tilde'], 'increase')
        else:
            modeldata = modeldata_mon_decr
            pn, ps, pns = calculate_causal_probabilities(modeldata['p_tilde'], 'decrease')
        modeldata['PN'] = pn
        modeldata['PS'] = ps
        modeldata['PNS'] = pns
    else:
        modeldata = find_best_approximation_to_model(constraint_data, meta_data[meta_data_idx])
    return modeldata


def calculate_causal_probabilities(p_tilde: Dict[str, float], direction: str) -> Tuple[float, float, float]:
    pxy = p_tilde['1100'] + p_tilde['1101'] + p_tilde['1110'] + p_tilde['1111']
    pxny = p_tilde['1000'] + p_tilde['1001'] + p_tilde['1010'] + p_tilde['1011']
    pnxy = p_tilde['0100'] + p_tilde['0101'] + p_tilde['0110'] + p_tilde['0111']
    pnxny = p_tilde['0000'] + p_tilde['0001'] + p_tilde['0010'] + p_tilde['0011']
    py = pxy + pnxy
    py_x = p_tilde['0001'] + p_tilde['0011'] + p_tilde['0101'] + p_tilde['0111'] + p_tilde['1001'] + \
        p_tilde['1011'] + p_tilde['1101'] + p_tilde['1111']
    py_nx = p_tilde['0010'] + p_tilde['0011'] + p_tilde['0110'] + p_tilde['0111'] + p_tilde['1010'] + \
        p_tilde['1011'] + p_tilde['1110'] + p_tilde['1111']
    return get_causal_prob_monotony(py=py, pxy=pxy, pnxy=pnxy, pxny=pxny, pnxny=pnxny, py_x=py_x, py_nx=py_nx,
                                    direction=direction)


def get_causal_prob_monotony(py: float, pxy: float, pnxy: float, pxny: float, pnxny: float, py_x: float, py_nx: float,
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


def find_best_approximation_to_model(constraint_data: Dict[str, float], meta_data: Dict[str, Any]) -> Dict[str, Any]:
    result = dict()
    size_prob = meta_data['size_prob']
    base = meta_data['base_x']
    nb_variables = meta_data['nb_variables']
    B = meta_data['B']
    d = meta_data['d']
    F = meta_data['F']
    c = meta_data['c']
    s_codes = meta_data['S_codes']

    b = np.array([1.0] + [constraint_data[pattern] for pattern in meta_data['constraint_patterns']])

    # create and run the solver
    x = cp.Variable(shape=size_prob)
    obj = cp.Minimize(cp.sum(d * x))
    constraints = [B * x == b,
                   F * x >= c]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # get the solution
    if x.value is None:
        return result

    simplex_res = x.value

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

    return result


def test_model_from_x_to_y(base_x: int, base_y: int, obs_x: pd.Series, obs_y: pd.Series,
                           int_x: pd.Series, int_y: pd.Series, monotone: bool, verbose: bool) -> Dict[str, Any]:
    intervention_contingency_table = get_contingency_table(int_x, int_y, base_x, base_y)
    observation_contigency_table = get_contingency_table(obs_x, obs_y, base_x, base_y)

    if verbose:
        write_contingency_table(observation_contigency_table, base_x, base_y)
        write_contingency_table(intervention_contingency_table, base_x, base_y)

    return approximate_to_causal_model(base_x, base_y, observation_contigency_table, intervention_contingency_table,
                                       monotone, verbose)


def local_error(p_nom: float, p_denom: float, s: float) -> float:
    if p_denom > 0 and p_nom > 0:
        return 1 / s * p_nom * log2(p_nom / p_denom)
    elif p_nom == 0:
        return 0.0
    elif p_nom > 0:
        return 1000000.0


def calc_error(model_results: Dict[str, Any]) -> float:
    if "GlobalError" in model_results:
        return round(model_results["GlobalError"], 6)
    else:
        return 1000000.0


def preprocessing(data: pd.DataFrame, intervention_column: str, preserve_order: bool, parameters: Dict[str, Any])\
        -> Tuple[Any, Any, Any, Any]:
    if parameters['preprocess_method'] == 'none':
        split_idx = int(data.shape[0] / 2)
        obs_x, obs_y, int_x, int_y = split_data_at_index(data, split_idx)
    elif parameters['preprocess_method'] == 'cluster_discrete':
        _, clustered_data = cluster_data(data, intervention_column, parameters['nb_cluster'])
        disc_data = discretize_data(clustered_data, parameters['bins'])
        disc_data['labels'] = clustered_data['labels']
        if preserve_order:
            obs_x, obs_y, int_x, int_y, i_max = split_data(disc_data, 'labels', sort_data=False)
        else:
            obs_x, obs_y, int_x, int_y = split_at_clustered_labels(disc_data, intervention_column,
                                                                   parameters['nb_cluster'])
    elif parameters['preprocess_method'] == 'discrete_cluster':
        disc_data = discretize_data(data, parameters['bins'])
        (obs_x, obs_y, int_x, int_y), clustered_data = cluster_data(disc_data, intervention_column,
                                                                    parameters['nb_cluster'])
        if preserve_order:
            obs_x, obs_y, int_x, int_y, i_max = split_data(clustered_data, 'labels', sort_data=False)
    else:
        raise Exception('Preprocessing method not known')

    return obs_x, obs_y, int_x, int_y


def kl_term(p: float, q: float) -> float:
    if q == 0:
        return 10000000
    elif p == 0:
        return 0
    else:
        return p*log2(p/q)


def find_best_model_x_to_y(base_x: int, base_y: int, data: Tuple[Any, Any, Any, Any], verbose: bool) -> Dict[str, Any]:
    model_x_to_y_monotone = test_model_from_x_to_y(base_x=base_x, base_y=base_y, obs_x=data[0], obs_y=data[1],
                                                   int_x=data[2], int_y=data[3], monotone=True, verbose=verbose)
    model_x_to_y_not_monotone = test_model_from_x_to_y(base_x=base_x, base_y=base_y, obs_x=data[0], obs_y=data[1],
                                                       int_x=data[2], int_y=data[3], monotone=False, verbose=verbose)
    if calc_error(model_x_to_y_monotone) <= calc_error(model_x_to_y_not_monotone):
        model_x_to_y = model_x_to_y_monotone
    else:
        model_x_to_y = model_x_to_y_not_monotone
    return model_x_to_y


def get_distr_x(p_hat: Dict[str, float]) -> List[float]:
    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['0100'] + p_hat['0101'] +
             p_hat['0110'] + p_hat['0111'],
             p_hat['1000'] + p_hat['1001'] + p_hat['1010'] + p_hat['1011'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def get_distr_y(p_hat: Dict[str, float]) -> List[float]:
    distr = [p_hat['0000'] + p_hat['0001'] + p_hat['0010'] + p_hat['0011'] + p_hat['1000'] + p_hat['1001'] +
             p_hat['1010'] + p_hat['1011'],
             p_hat['0100'] + p_hat['0101'] + p_hat['0110'] + p_hat['0111'] + p_hat['1100'] + p_hat['1101'] +
             p_hat['1110'] + p_hat['1111']]
    return distr / sum(distr)


def get_kl_between_x_y(model_results: Dict[str, Any]) -> float:
    p_hat = model_results['p_hat']
    p_x = get_distr_x(p_hat)
    p_y = get_distr_y(p_hat)
    return kl_divergence(p_x, p_y)


def decide_best_model(model_x_to_y: Dict[str, Any], model_y_to_x: Dict[str, Any], monotone: bool, verbose: bool,
                      strategy: str = 'kl_dist', tolerance: float = 1.0e-05) -> Dict[str, Any]:
    best_model = dict()
    if strategy == 'kl_dist':
        error_x_to_y = calc_error(model_x_to_y)
        error_y_to_x = calc_error(model_y_to_x)
        error_tolerance = 0.01
    else:
        error_x_to_y = get_kl_between_x_y(model_x_to_y)
        error_y_to_x = get_kl_between_x_y(model_y_to_x)
        error_tolerance = 1.0e-12
    if verbose:
        print("total Error X -> Y: " + str(error_x_to_y))
        print("total Error Y -> X: " + str(error_y_to_x))

    if abs(error_x_to_y - error_y_to_x) < error_tolerance:
        if monotone or ('PNS' in model_x_to_y and 'PNS' in model_y_to_x):
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


def iacm_discovery(base_x: int, base_y: int, data: pd.DataFrame, parameters: Dict[str, Any], verbose: bool,
                   preserve_order: bool) -> Tuple[Tuple[str, str], Tuple[float, float]]:
    monotone = (parameters['monotone'] and base_x == 2 and base_y == 2)
    x_obs_x, x_obs_y, x_int_x, x_int_y = preprocessing(data=data, intervention_column='X',
                                                       preserve_order=preserve_order, parameters=parameters)
    data_xy = pd.concat([pd.concat([x_obs_x, x_int_x]), pd.concat([x_obs_y, x_int_y])], axis=1)
    data_xy.columns = ['X', 'Y']
    y_obs_x, y_obs_y, y_int_x, y_int_y = preprocessing(data=data, intervention_column='Y',
                                                       preserve_order=preserve_order, parameters=parameters)
    if base_x == base_y == 2:
        model_x_to_y = find_best_model_x_to_y(base_x=base_x, base_y=base_y, data=(x_obs_x, x_obs_y, x_int_x, x_int_y),
                                              verbose=verbose)
        model_y_to_x = find_best_model_x_to_y(base_x=base_x, base_y=base_y, data=(y_obs_y, y_obs_x, y_int_y, y_int_x),
                                              verbose=verbose)
        result = decide_best_model(model_x_to_y, model_y_to_x, False, verbose, strategy='kl_dist')
        result_xy = decide_best_model(model_x_to_y, model_y_to_x, False, verbose, strategy='kl_xy')
    else:
        model_x_to_y = test_model_from_x_to_y(base_x, base_y, x_obs_x, x_obs_y, x_int_x, x_int_y, monotone, verbose)
        model_y_to_x = test_model_from_x_to_y(base_x, base_y, y_obs_y, y_obs_x, y_int_y, y_int_x, monotone, verbose)
        result = decide_best_model(model_x_to_y, model_y_to_x, monotone, verbose, strategy='kl_dist')
        result_xy = decide_best_model(model_x_to_y, model_y_to_x, monotone, verbose, strategy='kl_dist')

    return (result['result'], result_xy['result']), (result['min_error'], result_xy['min_error'])


def iacm_timeseries(base_x: int, base_y: int, data: pd.DataFrame, parameters: Dict[str, Any], max_lag: int,
                    verbose: bool):
    timeseries_result = dict()
    t = data.shape[0]
    tmp_data = data.copy()
    for lag in range(0, min(max_lag, t-1)):
        tmp_data['X'] = data['X'][:t - lag].reset_index(drop=True)
        tmp_data['Y'] = data['Y'][lag:].reset_index(drop=True)
        tmp_data = tmp_data.dropna().reset_index(drop=True)
        res, crit = iacm_discovery(base_x=base_x, base_y=base_y, data=tmp_data, parameters=parameters, verbose=verbose,
                                   preserve_order=True)
        timeseries_result[lag] = dict()
        timeseries_result[lag]['result'] = res[1]
        timeseries_result[lag]['crit'] = crit[1]

    min_crit = 1
    best_res = "no decision"
    for lag, res_dict in timeseries_result.items():
        if "no decision" not in res_dict['result'] and res_dict['crit'] < min_crit:
            min_crit = res_dict['crit']
            best_res = res_dict['result']

    return best_res, min_crit
