import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from data_preparation import get_probabilities, get_probabilities_intervention, WriteContingencyTable, \
    getContingencyTables, discretize_data, cluster_data, split_data, split_data_at_index, split_at_clustered_labels, split_with_equal_std
from meta_data import setup_meta_data, base_repr
from plot import plot_distribution, plot_distributions, init_points, init_figure
from typing import List
from metrics import KL_Dist_X_Y, KL_Dist
from scipy.stats import chisquare

meta_data = dict()
meta_data['2_2'] = setup_meta_data(base=2, nb_variables=4)
meta_data['2_2_m_d'] = setup_meta_data(base=2, nb_variables=4, monotone_decr=True, monotone_incr=False)
meta_data['2_2_m_i'] = setup_meta_data(base=2, nb_variables=4, monotone_decr=False, monotone_incr=True)
meta_data['3_3'] = setup_meta_data(base=3, nb_variables=5)
meta_data['4_4'] = setup_meta_data(base=4, nb_variables=6)
#meta_data[5] = setup_meta_data(base=5, nb_variables=7)


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


def approximateToCausalModel(base_x: int, base_y: int, obsConTable, ExpConTable, drawObsData, color, monotone, verbose):
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

    meta_data_idx = str(base_x) + '_' + str(base_y)
    if monotone and base_x == 2 and base_y == 2:
        modeldata_mon_decr = FindBestApproximationToConsistentModel(constraint_data, meta_data[meta_data_idx + '_m_d'])
        modeldata_mon_incr = FindBestApproximationToConsistentModel(constraint_data, meta_data[meta_data_idx + '_m_i'])

        if (len(modeldata_mon_incr) == 0) and (len(modeldata_mon_decr) == 0):
            modeldata = dict()
            PN, PS, PNS = 0, 0, 0
        elif len(modeldata_mon_incr) == 0:
            modeldata = modeldata_mon_decr
            PN, PS, PNS = calculate_causal_prob(modeldata['NP'], False)
        elif len(modeldata_mon_decr) == 0:
            modeldata = modeldata_mon_incr
            PN, PS, PNS = calculate_causal_prob(modeldata['NP'], True)
        elif modeldata_mon_incr['GlobalError'] < modeldata_mon_decr['GlobalError']:
            modeldata = modeldata_mon_incr
            PN, PS, PNS = calculate_causal_prob(modeldata['NP'], True)
        else:
            modeldata = modeldata_mon_decr
            PN, PS, PNS = calculate_causal_prob(modeldata['NP'], False)
        modeldata['PN'] = PN
        modeldata['PS'] = PS
        modeldata['PNS'] = PNS
    else:
        modeldata = FindBestApproximationToConsistentModel(constraint_data, meta_data[meta_data_idx])
    if drawObsData:
        print("approximated distribution")
        if "NP" in modeldata:
            #print("Pxy:" + str(modeldata["Pxy"]) + " Pxny:" + str(modeldata["Pxny"]) + " Pnxy:" + str(
            #    modeldata["Pnxy"]) + " Pnxny:" + str(modeldata["Pnxny"]))
            Pxy = modeldata['NP']['1100'] + modeldata['NP']['1101'] + modeldata['NP']['1110'] + modeldata['NP']['1111']
            Pxny = modeldata['NP']['1000'] + modeldata['NP']['1001'] + modeldata['NP']['1010'] + modeldata['NP']['1011']
            Pnxy = modeldata['NP']['0100'] + modeldata['NP']['0101'] + modeldata['NP']['0110'] + modeldata['NP']['0111']
            Pnxny = modeldata['NP']['0000'] + modeldata['NP']['0001'] + modeldata['NP']['0010'] + modeldata['NP']['0011']
            plot_distribution(Pxy, Pxny, Pnxy, Pnxny, color)
    return modeldata


def calculate_causal_prob(NP, increase):
    pxy = NP['1100'] + NP['1101'] + NP['1110'] + NP['1111']
    pxny = NP['1000'] + NP['1001'] + NP['1010'] + NP['1011']
    pnxy = NP['0100'] + NP['0101'] + NP['0110'] + NP['0111']
    pnxny = NP['0000'] + NP['0001'] + NP['0010'] + NP['0011']
    py = pxy + pnxy
    py_x = NP['0001'] + NP['0011'] + NP['0101'] + NP['0111'] + NP['1001'] + NP['1011'] + NP['1101'] + NP['1111']
    py_nx = NP['0010'] + NP['0011'] + NP['0110'] + NP['0111'] + NP['1010'] + NP['1011'] + NP['1110'] + NP['1111']
    PN, PS, PNS = get_causal_prob_monotony(py=py, pxy=pxy, pnxy=pnxy, pxny=pxny, pnxny=pnxny, py_x=py_x, py_nx=py_nx, increase=increase)
    return PN, PS, PNS


def get_causal_prob_monotony(py, pxy, pnxy, pxny, pnxny, py_x, py_nx, increase):
    if increase:
        PNS = py_x - py_nx
        if pxy > 0:
            PN = (py - py_nx) / pxy
        else:
            PN = 0
        if pnxny > 0:
            PS = (py_x - py) / pnxny
        else:
            PS = 0
    else:
        PNS = py_nx - py_x
        if pnxy > 0:
            pny = 1 - py
            pny_x = 1 - py_x
            PN = (pny_x - pny) / pnxy
        else:
            PN = 0
        if pxny > 0:
            pny = 1 - py
            pny_nx = 1 - py_nx
            PS = (pny - pny_nx) / pxny
        else:
            PS = 0

    return PN, PS, PNS


def FindBestApproximationToConsistentModel(constraint_data, meta_data):
    res = dict()
    size_prob = meta_data['size_prob']
    base = meta_data['base_x']
    nb_variables = meta_data['nb_variables']
    B = meta_data['B']
    d = meta_data['d']
    F = meta_data['F']
    c = meta_data['c']
    S_codes = meta_data['S_codes']

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
    res['Ptilde'] = P
    res["GlobalError"] = log2(1 / S)

    return res


def testModelFromXtoY(base_x: int, base_y: int, obsX, obsY, intX, intY, drawObsData, color, monotone, verbose):
    ExperimentContigenceTable = getContingencyTables(intX, intY, base_x, base_y)
    ObservationContigenceTable = getContingencyTables(obsX, obsY, base_x, base_y)

    #ObservationContigenceTable = [[2, 28], [998,972]]
    #ExperimentContigenceTable = [[16, 14], [984, 986]]
    if verbose or drawObsData:
        WriteContingencyTable(ObservationContigenceTable, base_x, base_y)
        WriteContingencyTable(ExperimentContigenceTable, base_x, base_y)

    return approximateToCausalModel(base_x, base_y, ObservationContigenceTable, ExperimentContigenceTable, drawObsData, color, monotone, verbose)


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


def preprocessing(data: pd.DataFrame, sort_col, preserve_order, params, base_x: int, base_y: int):
    if params['preprocess_method'] == 'none':
        split_idx = int(data.shape[0] / 2)
        obsX, obsY, intX, intY = split_data_at_index(data, split_idx)
    elif params['preprocess_method'] == 'auto':
        result = dict()
        for sort_col in ['X', 'Y']:
            for method in ['split_discrete', 'discrete_split']:
                tmp_params = params
                tmp_params['preprocess_method'] = method
                result[method + sort_col] = dict()
                result[method + sort_col]['res'] = preprocessing(data, sort_col, tmp_params, base_x, base_y)
                max_effect_range_X = effect_range(pd.concat([result[method + sort_col]['res'][0], result[method + sort_col]['res'][2]]))[2]
                max_effect_range_Y = effect_range(pd.concat([result[method + sort_col]['res'][1], result[method + sort_col]['res'][3]]))[2]
                result[method + sort_col]['crit'] = (effect_range(result[method + sort_col]['res'][0])[2]/max_effect_range_X + effect_range(result[method + sort_col]['res'][2])[2]/max_effect_range_X +
                                          effect_range(result[method + sort_col]['res'][1])[2]/max_effect_range_Y + effect_range(result[method + sort_col]['res'][3])[2]/max_effect_range_Y)/4
        min_crit = 1
        best_method = 'split_discreteX'
        for sort_col in ['X', 'Y']:
            for method in ['split_discrete', 'discrete_split']:
                if result[method + sort_col]['crit'] < min_crit:
                    min_crit = result[method + sort_col]['crit']
                    best_method = method + sort_col
        obsX, obsY, intX, intY = result[best_method]['res']
    elif params['preprocess_method'] == 'discrete_split':
        disc_data = discretize_data(data, params)
        if preserve_order:
            obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col, sort_data=False)
        else:
            obsX, obsY, intX, intY, i_max = split_with_equal_std(disc_data, sort_col, sort_data=True)
    elif params['preprocess_method'] == 'split_discrete':
        if preserve_order:
            obsX, obsY, intX, intY, i_max = split_data(data, sort_col, sort_data=False)
        else:
            obsX, obsY, intX, intY, i_max = split_with_equal_std(data, sort_col)
        dataXY = pd.concat([pd.concat([obsX, intX]), pd.concat([obsY, intY])], axis=1)
        dataXY.columns= ['X', 'Y']
        disc_data = discretize_data(dataXY, params)
        obsX, obsY, intX, intY = split_data_at_index(disc_data, i_max)
    elif params['preprocess_method'] == 'split_strategy':
        disc_data = discretize_data(data, params)
        obsX_ds, obsY_ds, intX_ds, intY_ds, i_max_ds = split_data(disc_data, sort_col, sort_data=not preserve_order)
        I_sq_x_ds = calc_I_squared(obsX_ds)
        obsX_sd, obsY_sd, intX_sd, intY_sd, i_max_sd = split_data(data, sort_col, sort_data=not preserve_order)
        dataXY = pd.concat([pd.concat([obsX_sd, intX_sd]), pd.concat([obsY_sd, intY_sd])], axis=1)
        disc_data = discretize_data(dataXY, params)
        obsX, obsY, intX, intY = split_data_at_index(disc_data, i_max_sd)
        I_sq_x_sd = calc_I_squared(obsX)
        if I_sq_x_ds > I_sq_x_sd:
            obsX = obsX_ds
            obsY = obsY_ds
            intX = intX_ds
            intY = intY_ds
    elif params['preprocess_method'] == 'cluster_discrete':
        (obsX, obsY, intX, intY), clustered_data = cluster_data(data, sort_col, params)
        disc_data = discretize_data(clustered_data, params)
        disc_data['labels'] = clustered_data['labels']
        if preserve_order:
            obsX, obsY, intX, intY, i_max = split_data(disc_data, 'labels', sort_data=False)
        else:
            obsX, obsY, intX, intY = split_at_clustered_labels(disc_data,sort_col, params)
    elif params['preprocess_method'] == 'discrete_cluster':
        disc_data = discretize_data(data, params)
        (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
        if preserve_order:
            obsX, obsY, intX, intY, i_max = split_data(clustered_data, 'labels', sort_data=False)
    elif params['preprocess_method'] == 'new_strategy':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col, sort_data=not preserve_order)
        mi_ds = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        obsX, obsY, intX, intY, i_max = split_data(data, sort_col, sort_data=not preserve_order)
        mi_s = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        disc_data = discretize_data(data, params)
        (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
        if preserve_order:
            obsX, obsY, intX, intY, i_max = split_data(clustered_data, 'labels', sort_data=False)
        mi_dc = mutual_information(getContingencyTables(obsX, obsY, base_x, base_y), base_x, base_y)
        if (mi_ds < min(mi_s,mi_dc)):
            disc_data = discretize_data(data, params)
            obsX, obsY, intX, intY, i_max = split_data(disc_data, sort_col, sort_data=not preserve_order)
        elif mi_s < min(mi_ds, mi_dc):
            obsX, obsY, intX, intY, i_max = split_data(data, sort_col, sort_data=not preserve_order)
            dataXY = pd.concat([pd.concat([obsX, intX]), pd.concat([obsY, intY])], axis=1)
            disc_data = discretize_data(dataXY, params)
            obsX, obsY, intX, intY = split_data_at_index(disc_data, i_max)
        else:
            disc_data = discretize_data(data, params)
            (obsX, obsY, intX, intY), clustered_data = cluster_data(disc_data, sort_col, params)
            if preserve_order:
                obsX, obsY, intX, intY, i_max = split_data(clustered_data, 'labels', sort_data=False)
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


def calc_variations(observation_contingence_table, sort_col, base_x, base_y):
    if 'X' in sort_col:
        return min([entropy(observation_contingence_table[i]) for i in range(0, base_x)])
    else:
        return min([entropy([observation_contingence_table[j][i] for j in range(0,base_x)]) for i in range(0, base_y)])


def calc_I_squared(data):
    # measure for heterogeneity
    stat, p = chisquare(data, axis=None)
    if np.isnan(stat):
        return 1000000
    dof = float(len(set(data.tolist())) - 1)
    return (stat - dof) / stat


def effect_range(data):
    m = np.mean(data)
    T = np.std(data)

    return m-2*T, m+2*T, 4*T


def findBestModelXtoY(base_x, base_y, data, verbose):
    modelXtoY_monotone = testModelFromXtoY(base_x=base_x, base_y=base_y, obsX=data[0], obsY=data[1], intX=data[2],
                                           intY=data[3], drawObsData=False, color="green", monotone=True, verbose=verbose)
    modelXtoY_not_monotone = testModelFromXtoY(base_x=base_x, base_y=base_y, obsX=data[0], obsY=data[1], intX=data[2],
                                               intY=data[3], drawObsData=False, color="green", monotone=False, verbose=verbose)
    if calcError(modelXtoY_monotone) <= calcError(modelXtoY_not_monotone):
        modelXtoY = modelXtoY_monotone
    else:
        modelXtoY = modelXtoY_not_monotone
    return modelXtoY


def get_distr_x(np):
    distr = [np['0000'] + np['0001'] + np['0010'] + np['0011'] + np['0100'] + np['0101'] + np['0110'] + np['0111'],
             np['1000'] + np['1001'] + np['1010'] + np['1011'] + np['1100'] + np['1101'] + np['1110'] + np['1111']]
    return distr / sum(distr)


def get_distr_y(np):
    distr = [np['0000'] + np['0001'] + np['0010'] + np['0011'] + np['1000'] + np['1001'] + np['1010'] + np['1011'],
             np['0100'] + np['0101'] + np['0110'] + np['0111'] + np['1100'] + np['1101'] + np['1110'] + np['1111']]
    return distr / sum(distr)


def iacm_discovery(base_x: int, base_y: int, data: pd.DataFrame, params, verbose, preserve_order):
    monotone = (params['monotone'] and base_x == 2 and base_y == 2)
    x_obsX, x_obsY, x_intX, x_intY = preprocessing(data=data, sort_col='X', preserve_order=preserve_order, params=params, base_x=base_x, base_y=base_y)
    dataXY = pd.concat([pd.concat([x_obsX, x_intX]), pd.concat([x_obsY, x_intY])], axis=1)
    dataXY.columns = ['X', 'Y']
    y_obsX, y_obsY, y_intX, y_intY = preprocessing(data=data, sort_col='Y', preserve_order=preserve_order, params=params, base_x=base_x, base_y=base_y)
    if base_x == base_y == 2:
        modelXtoY = findBestModelXtoY(base_x=base_x, base_y=base_y, data=(x_obsX, x_obsY, x_intX, x_intY), verbose=verbose)
        modelYtoX = findBestModelXtoY(base_x=base_x, base_y=base_y, data=(y_obsY, y_obsX, y_intY, y_intX), verbose=verbose)
        result = decideBestModel(modelXtoY, modelYtoX, False, verbose, strategy='kl_dist')
        result_xy = decideBestModel(modelXtoY, modelYtoX, False, verbose, strategy='kl_xy')
    else:
        modelXtoY = testModelFromXtoY(base_x, base_y, x_obsX, x_obsY, x_intX, x_intY, False, "green", monotone, verbose)
        modelYtoX = testModelFromXtoY(base_x, base_y, y_obsY, y_obsX, y_intY, y_intX, False, "yellow", monotone, verbose)
        result = decideBestModel(modelXtoY, modelYtoX, monotone, verbose, strategy='kl_dist')
        result_xy = decideBestModel(modelXtoY, modelYtoX, monotone, verbose, strategy='kl_dist')

    return (result['result'], result_xy['result']), (result['min_error'], result_xy['min_error'])

    # for color in ['black', 'green', 'yellow', 'red']:
    #    ax.scatter(scatter_points[color]['x'], scatter_points[color]['y'], scatter_points[color]['z'], color=color, linewidth=1, s=2)
    # plt.show()


def getKL_X_Y(model):
    P = model['Ptilde']
    Px = get_distr_x(P)
    Py = get_distr_y(P)
    return KL_Dist(Px, Py)


def decideBestModel(modelXtoY, modelYtoX, monotone, verbose, strategy='kl_dist', tolerance=1.0e-05):
    bestModel = dict()
    if strategy == 'kl_dist':
        errorXtoY = calcError(modelXtoY)
        errorYtoX = calcError(modelYtoX)
        error_tolerance = 0.01
    else:
        errorXtoY = getKL_X_Y(modelXtoY)
        errorYtoX = getKL_X_Y(modelYtoX)
        error_tolerance = 1.0e-12
    if verbose: print("total Error X -> Y: " + str(errorXtoY))
    if verbose: print("total Error Y -> X: " + str(errorYtoX))

    if abs(errorXtoY - errorYtoX) < error_tolerance:
        if monotone or ('PNS' in modelXtoY and 'PNS' in modelYtoX):
            PNSXtoY = modelXtoY['PNS']
            PNSYtoX = modelYtoX['PNS']
            if abs(PNSXtoY-PNSYtoX) < tolerance:
                res = "no decision"
                error_ratio = 1
            elif PNSXtoY > PNSYtoX:
                res = "X->Y"
                error_ratio = abs(PNSXtoY - PNSYtoX)
            else:
                res = "Y->X"
                error_ratio = abs(PNSXtoY - PNSYtoX)
        else:
            res = "no decision"
            error_ratio = 1
    elif errorXtoY < errorYtoX:
        if verbose: print("X -> Y")
        res = "X->Y"
        error_ratio = min(errorXtoY, errorYtoX) / max(errorXtoY, errorYtoX)
    else: # errorXtoY > errorYtoX:
        if verbose: print("Y -> X")
        res = "Y->X"
        error_ratio = min(errorXtoY, errorYtoX) / max(errorXtoY, errorYtoX)

    bestModel['result'] = res
    bestModel['error_ratio'] = error_ratio
    bestModel['min_error'] = min(errorXtoY, errorYtoX)
    return bestModel


def iacm_timeseries(base_x: int, base_y: int, data: pd.DataFrame, params, max_lag, verbose):
    timeseries_result = dict()
    T = data.shape[0]
    tmp_data = data.copy()
    for lag in range(0, min(max_lag, T-1)):
        tmp_data['X'] = data['X'][:T - lag].reset_index(drop=True)
        tmp_data['Y'] = data['Y'][lag:].reset_index(drop=True)
        tmp_data = tmp_data.dropna().reset_index(drop=True)
        res, crit = iacm_discovery(base_x=base_x, base_y=base_y, data=tmp_data, params=params, verbose=verbose, preserve_order=True)
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
