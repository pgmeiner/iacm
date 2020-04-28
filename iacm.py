import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from data_preparation import get_probabilities, get_probabilities_intervention, WriteContingencyTable, \
    getContingencyTables, discretize_data, cluster_data, split_data, split_data_at_index, split_at_clustered_labels
from meta_data import setup_meta_data, base_repr
from plot import plot_distribution
from typing import List

meta_data = dict()
meta_data[2] = setup_meta_data(base=2, nb_variables=4)
meta_data[3] = setup_meta_data(base=3, nb_variables=5)
meta_data[4] = setup_meta_data(base=4, nb_variables=6)


def get_constraint_data(base_x: int, base_y: int, list_of_distributions):
    constraint_data = dict()

    for value, pattern in zip(list_of_distributions, meta_data[str(base_x) + '_' + str(base_y)]['constraint_patterns']):
        constraint_data[pattern] = value

    return constraint_data


def get_constraint_distribution(P, P_i, base_x: int, base_y: int):
    # if base == 2:
    #     return [P_i['1_1'], P_i['0_1'], P['11'], P['10'], P['01']]
    # elif base == 3:
    #     return [P_i['2_1'], P_i['1_1'], P_i['0_1'], P_i['2_2'], P_i['1_2'], P_i['0_2'],
    #             P['22'], P['21'], P['20'], P['12'], P['11'], P['10'], P['02'], P['01']]

    constraint_distr = []
    for value_y in range(1, base_y):
        for value_x in range(base_x-1, -1, -1):
            index = str(value_x) + '_' + str(value_y)
            constraint_distr.append(P_i[index])

    for value_x in range(base_x-1, -1, -1):
        for value_y in range(base_y-1, -1, -1):
            if (value_x == 0) and (value_y == 0):
                continue
            index = str(value_x) + str(value_y)
            constraint_distr.append(P[index])

    return constraint_distr


def approximateToCausalModel(base_x: int, base_y: int, obsConTable, ExpConTable, drawObsData, color, verbose):
    P = get_probabilities(obsConTable, base_x, base_y)
    P_i = get_probabilities_intervention(ExpConTable, base_x, base_y)
    constraint_distr = get_constraint_distribution(P, P_i, base_x, base_y)
    if verbose:
        for key, value in P.items():
            print(key + ":" + str(value))
    #print("Pxy:" + str(Pxy) + " Pxny:" + str(Pxny) + " Pnxy:" + str(Pnxy) + " Pnxny:" + str(Pnxny))
    if drawObsData and base_x == 2 and base_y == 2:
        plot_distribution(P['11'], P['10'], P['01'], P['00'], "black")
    if verbose:
        for key, value in P_i.items():
            print(key + ":" + str(value))
    #print("Py_x:" + str(Py_x) + " Py_nx:" + str(Py_nx) + " Pny_x:" + str(Pny_x) + " Pny_nx:" + str(Pny_nx))
    constraint_data = get_constraint_data(base_x=base_x, base_y=base_y, list_of_distributions=constraint_distr)#Py_nx, Py_x, Pnxy, Pxny, Pxy])

    modeldata = FindBestApproximationToConsistentModel(base_x, base_y, constraint_data)
    if verbose:
        print("approximated distribution")
        if "Pxy" in modeldata:
            print("Pxy:" + str(modeldata["Pxy"]) + " Pxny:" + str(modeldata["Pxny"]) + " Pnxy:" + str(
                modeldata["Pnxy"]) + " Pnxny:" + str(modeldata["Pnxny"]))
            plot_distribution(modeldata["Pxy"], modeldata["Pxny"], modeldata["Pnxy"], modeldata["Pnxny"], color)
    return modeldata


def get_causal_prob_monotony(py, pxy, pnxny, py_x, py_nx):
    PNS = py_x - py_nx
    if pxy > 0:
        PN = (py - py_nx) / pxy
    else:
        PN = 0
    if pnxny > 0:
        PS = (py_x - py) / pnxny
    else:
        PS = 0

    return PN, PS, PNS


def FindBestApproximationToConsistentModel(base_x: int, base_y: int, constraint_data):
    res = dict()
    meta_data_idx = str(base_x) + '_' + str(base_y)
    size_prob = meta_data[meta_data_idx]['size_prob']
    base = meta_data[meta_data_idx]['base_x']
    nb_variables = meta_data[meta_data_idx]['nb_variables']
    B = meta_data[meta_data_idx]['B']
    d = meta_data[meta_data_idx]['d']
    F = meta_data[meta_data_idx]['F']
    c = meta_data[meta_data_idx]['c']
    S_codes = meta_data[meta_data_idx]['S_codes']

    b = np.array([1.0] + [constraint_data[pattern] for pattern in meta_data[meta_data_idx]['constraint_patterns']])

    # create and run the solver
    x = cp.Variable(shape=size_prob)
    obj = cp.Minimize(cp.sum(d * x))
    constraints = [B * x == b,
                   F * x >= c]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # get the solution
    if x.value is None:
        return res

    SimplexRes = x.value

    P = dict()

    for i in range(0, len(SimplexRes)):
        code = base_repr(i, base, nb_variables)
        P[code] = max(SimplexRes[i], 0.0)

    S = sum([P[code] for code in S_codes])
    if S == 0:
        return res
    # normalized distribution
    NP = dict()
    for code, value in P.items():
        if code in S_codes:
            NP[code] = value / S
        else:
            NP[code] = 0

    res['NP'] = NP
    res["GlobalError"] = log2(1 / S)
    if base == 2:
        pxy = NP['1100'] + NP['1101'] + NP['1110'] + NP['1111']
        pnxy = NP['0100'] + NP['0101'] + NP['0110'] + NP['0111']
        pnxny = NP['0000'] + NP['0001'] + NP['0010'] + NP['0011']
        py = pxy + pnxy
        py_x = NP['0010'] + NP['0011'] + NP['0110'] + NP['0111'] + NP['1010'] + NP['1011'] + NP['1110'] + NP['1111']
        py_nx = NP['0001'] + NP['0011'] + NP['0101'] + NP['0111'] + NP['1001'] + NP['1011'] + NP['1101'] + NP['1111']
        PN, PS, PNS = get_causal_prob_monotony(py=py, pxy=pxy, pnxny=pnxny, py_x=py_x, py_nx=py_nx)
        res['PN'] = PN
        res['PS'] = PS
        res['PNS'] = PNS

    return res


def testModelFromXtoY(base_x: int, base_y: int, obsX, obsY, intX, intY, drawObsData, color, verbose):
    ExperimentContigenceTable = getContingencyTables(intX, intY, base_x, base_y)
    ObservationContigenceTable = getContingencyTables(obsX, obsY, base_x, base_y)

    if verbose:
        WriteContingencyTable(ObservationContigenceTable, base_x, base_y)
        WriteContingencyTable(ExperimentContigenceTable, base_x, base_y)

    return approximateToCausalModel(base_x, base_y, ObservationContigenceTable, ExperimentContigenceTable, drawObsData, color, verbose)


def localError(P_nom, P_denom, S):
    if P_denom > 0 and P_nom > 0:
        return 1 / S * P_nom * log2(P_nom / P_denom)
    elif P_nom == 0:
        return 0.0
    elif P_nom > 0:
        return 1000000.0


def calcError(model):
    if "GlobalError" in model:
        return round(model["GlobalError"], 6)
    else:
        return 1000000.0


def preprocessing(data: pd.DataFrame, sort_col, params, base_x: int, base_y: int):
    if params['preprocess_method'] == 'none':
        split_idx = int(data.shape[0] / 2)
        obsX, obsY, intX, intY = split_data_at_index(data, split_idx)
    elif params['preprocess_method'] == 'discrete_split':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col)
    elif params['preprocess_method'] == 'split_discrete':
        obsX, obsY, intX, intY, i_max = split_data(data, sort_col)
        dataXY = pd.concat([pd.concat([obsX, intX]), pd.concat([obsY, intY])], axis=1)
        disc_data = discretize_data(dataXY, params)
        obsX, obsY, intX, intY = split_data_at_index(disc_data, i_max)
    elif params['preprocess_method'] == 'cluster_discrete':
        (obsX, obsY, intX, intY), clustered_data = cluster_data(data, sort_col, params)
        disc_data = discretize_data(clustered_data, params)
        disc_data['labels'] = clustered_data['labels']
        obsX, obsY, intX, intY = split_at_clustered_labels(disc_data,sort_col,params)
    elif params['preprocess_method'] == 'discrete_cluster':
        disc_data = discretize_data(data, params)
        (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
    elif params['preprocess_method'] == 'new_strategy':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col)
        mi_ds = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        #variation_disc_split = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        obsX, obsY, intX, intY, i_max = split_data(data, sort_col)
        mi_s = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        #variation_split = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        disc_data = discretize_data(data, params)
        (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
        mi_dc = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        #variation_disc_cluster = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        #print("v_ds " + str(variation_disc_split))
        #print("v_s " + str(variation_split))
        #print("v_dc " + str(variation_disc_cluster))
        if (mi_ds < min(mi_s,mi_dc)):
            disc_data = discretize_data(data, params)
            obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col)
        elif mi_s < min(mi_ds, mi_dc):
            obsX, obsY, intX, intY, i_max = split_data(data, sort_col)
        else:
            disc_data = discretize_data(data, params)
            (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
    else:
        disc_data = discretize_data(data, params)
        (cobsX, cobsY, cintX, cintY), clustered_data = cluster_data(data, sort_col, params)
        obsX = cobsX
        obsY = cobsY
        intX = cintX
        intY = cintY
        if (max(get_probabilities(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y).values()) > params['prob_threshold_cluster']):
            obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col)
            if (max(get_probabilities(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y).values()) <= params['prob_threshold_no_cluster']):
                obsX = cobsX
                obsY = cobsY
                intX = cintX
                intY = cintY

    return obsX, obsY, intX, intY


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array).astype(float)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def entropy(ls: List):
    s = sum(ls)
    if s > 3:
        p = [e/s for e in ls]
        return sum([-ep*log2(ep) for ep in p])
    else:
        return 0


def KL_term(p,q):
    if q == 0:
        return 10000000
    elif p == 0:
        return 0
    else:
        return p*log2(p/q)


def mutual_information(observation_contingence_table, base_x: int, base_y: int) -> float:
    X = [sum(observation_contingence_table[i]) for i in range(0, base_x)]
    Y = [sum([observation_contingence_table[i][j] for i in range(0, base_x)]) for j in range(0, base_y)]
    x_sum = sum(X)
    y_sum = sum(Y)
    if x_sum > 0:
        p_x = [e / x_sum for e in X]
    else:
        p_x = [0]*len(X)
    if y_sum > 0:
        p_y = [e / y_sum for e in Y]
    else:
        p_y = [0] * len(Y)

    return sum([KL_term(e_p,e_q) for e_p, e_q in zip(p_x, p_y)])


def calc_variations(observation_contingence_table, sort_col):
    if 'X' in sort_col:
        return min([entropy(observation_contingence_table[0]),
            entropy(observation_contingence_table[1]),
            entropy(observation_contingence_table[2])])
    else:
        return min([entropy([observation_contingence_table[0][0],observation_contingence_table[1][0],observation_contingence_table[2][0]]),
                   entropy([observation_contingence_table[0][1],observation_contingence_table[1][1],observation_contingence_table[2][1]]),
                   entropy([observation_contingence_table[0][2],observation_contingence_table[1][2],observation_contingence_table[2][2]])])


def iacm(base_x: int, base_y: int, data: pd.DataFrame, params, verbose):
    error_gap = dict()
    pn_error = dict()
    result = dict()
    for sort_col in ['X', 'Y']:
        obsX, obsY, intX, intY = preprocessing(data, sort_col, params, base_x, base_y)
        modelXtoY = testModelFromXtoY(base_x, base_y, obsX, obsY, intX, intY, True, "green", verbose)
        modelYtoX = testModelFromXtoY(base_x, base_y, obsY, obsX, intY, intX, False, "yellow", verbose)
        errorXtoY = calcError(modelXtoY)
        if verbose: print("total Error X -> Y: " + str(errorXtoY))
        errorYtoX = calcError(modelYtoX)
        if verbose: print("total Error Y -> X: " + str(errorYtoX))
        if errorXtoY < errorYtoX:
            if verbose: print("X -> Y")
            res = "X->Y"
        elif errorXtoY > errorYtoX:
            if verbose: print("Y -> X")
            res = "Y->X"
        else:
            if verbose: print("no decision")
            res = "no decision"
        result[sort_col] = res
        #pn_error[sort_col] = min(errorYtoX, errorXtoY)
        error_gap[sort_col] = min(errorXtoY, errorYtoX) / max(errorXtoY, errorYtoX)

    # if pn_error['X'] < pn_error['Y']:
    #     return result['X']
    # else:
    #     return result['Y']

    if error_gap['X'] == 1 and error_gap['Y'] == 1:
        return "no decision"
    elif (1 / error_gap['X']) < (1 / error_gap['Y']):
        return result['Y']
    else:
        return result['X']

    # for color in ['black', 'green', 'yellow', 'red']:
    #    ax.scatter(scatter_points[color]['x'], scatter_points[color]['y'], scatter_points[color]['z'], color=color, linewidth=1, s=2)
    # plt.show()
