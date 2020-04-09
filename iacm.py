import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from data_preparation import get_probabilities, get_probabilities_intervention, WriteContingencyTable, \
    getContingencyTables, discretize_data, cluster_data, split_data
from plot import plot_distribution
from typing import List


def count_char(str, char_to_count):
    count = 0
    for c in str:
        if c == char_to_count:
            count = count + 1
    return count


def replace_char_by_char(char_to_replaced, str_to_be_replaced, to_be_inserted_str):
    replace_index = 0
    final_str = ''
    for i, c in enumerate(str_to_be_replaced):
        if c == char_to_replaced:
            final_str = final_str + to_be_inserted_str[replace_index]
            replace_index = replace_index + 1
        else:
            final_str = final_str + c

    return final_str


def base_repr(number, base, str_len):
    repr = np.base_repr(number, base)
    return '0'*(str_len-len(repr)) + repr


def generate_codes(pattern, base):
    nb_x = count_char(pattern, 'x')
    codes = [replace_char_by_char('x', pattern, base_repr(nb, base, nb_x)) for nb in range(0, pow(base, nb_x))]
    return codes


def convert_to_constraint_line(codes, size_prob, base):
    positions = list()
    for code in codes:
        positions.append(int(code, base))
    result = list()
    for i in range(0, size_prob):
        if i in positions:
            result.append(1)
        else:
            result.append(0)
    return result


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_zero_codes(code_patterns, base):
    codes = flatten([generate_codes(code_pattern, base) for code_pattern in code_patterns])
    return codes


def s_codes(zero_codes, base, nb_variables):
    all_codes = generate_codes('x'*nb_variables, base)
    return list(set(all_codes) - set(zero_codes))


def generate_constraint_lines(patterns, size_prob, base):
    lines = list()
    for pattern in patterns:
        lines.append(convert_to_constraint_line(generate_codes(pattern, base), size_prob, base))

    return lines


pattern_data = {2: {'constraint_patterns': ['xxx1', 'xx1x', '01xx', '10xx', '11xx'],
                    'zero_code_patterns': ['00x1', '110x', '101x', '01x0']},# '1001', '0101']},
                3: {'constraint_patterns': ['xxxx1', 'xxx1x', 'xx1xx', '22xxx', '21xxx', '20xxx', '12xxx', '11xxx',
                                            '10xxx', '02xxx', '01xxx'],
                    'zero_code_patterns': ['001xx', '002xx', '010xx', '012xx', '020xx', '021xx',
                                           '10x1x', '10x2x', '11x0x', '11x2x', '12x0x', '12x1x',
                                           '20xx1', '20xx2', '21xx0', '21xx2', '22xx0', '22xx1']}
                }


def setup_meta_data(base, nb_variables):
    meta_data = dict()
    size_prob = pow(base, nb_variables)
    meta_data['base'] = base
    meta_data['nb_variables'] = nb_variables
    meta_data['size_prob'] = size_prob

    lines = generate_constraint_lines(pattern_data[base]['constraint_patterns'], size_prob, base)
    meta_data['B'] = np.array([[1] * size_prob] + lines)

    zero_codes = get_zero_codes(pattern_data[base]['zero_code_patterns'], base)
    meta_data['S_codes'] = s_codes(zero_codes, base, nb_variables)
    d_list = list()
    for i in range(0, size_prob):
        if base_repr(i, base, nb_variables) in meta_data['S_codes']:
            d_list.append(1)
        else:
            d_list.append(0)
    meta_data['d'] = np.array(d_list)

    meta_data['F'] = np.diag(np.array([1] * size_prob))
    meta_data['c'] = np.array([0.0] * size_prob)

    return meta_data

meta_data = dict()
meta_data[2] = setup_meta_data(base=2, nb_variables=4)
meta_data[3] = setup_meta_data(base=3, nb_variables=5)


def get_constraint_data(base, list_of_distributions):
    constraint_data = dict()

    for value, pattern in zip(list_of_distributions, pattern_data[base]['constraint_patterns']):
        constraint_data[pattern] = value

    return constraint_data


def get_constraint_distribution(P, P_i, base):
    if base == 2:
        return [P_i['0_1'], P_i['1_1'], P['01'], P['10'], P['11']]
    elif base == 3:
        return [P_i['2_1'], P_i['1_1'], P_i['0_1'], P['22'], P['21'], P['20'], P['12'], P['11'], P['10'],
                P['02'], P['01']]


def approximateToCausalModel(base,obsConTable, ExpConTable, drawObsData, color):
    P = get_probabilities(obsConTable, base)
    P_i = get_probabilities_intervention(ExpConTable, base)
    constraint_distr = get_constraint_distribution(P, P_i, base)
    for key, value in P.items():
        print(key + ":" + str(value))
    #print("Pxy:" + str(Pxy) + " Pxny:" + str(Pxny) + " Pnxy:" + str(Pnxy) + " Pnxny:" + str(Pnxny))
    if drawObsData and base == 2:
        plot_distribution(P['11'], P['10'], P['01'], P['00'], "black")
    for key, value in P_i.items():
        print(key + ":" + str(value))
    #print("Py_x:" + str(Py_x) + " Py_nx:" + str(Py_nx) + " Pny_x:" + str(Pny_x) + " Pny_nx:" + str(Pny_nx))
    constraint_data = get_constraint_data(base=base, list_of_distributions=constraint_distr)#Py_nx, Py_x, Pnxy, Pxny, Pxy])

    modeldata = FindBestApproximationToConsistentModel(base, constraint_data)
    print("approximated distribution")
    if "Pxy" in modeldata:
        print("Pxy:" + str(modeldata["Pxy"]) + " Pxny:" + str(modeldata["Pxny"]) + " Pnxy:" + str(
            modeldata["Pnxy"]) + " Pnxny:" + str(modeldata["Pnxny"]))
        plot_distribution(modeldata["Pxy"], modeldata["Pxny"], modeldata["Pnxy"], modeldata["Pnxny"], color)
    return modeldata


def marginalDistribution(P, fixed_code):
    nb_x = count_char(fixed_code, 'x')
    format_str = '{0:0xb}'.replace('x', str(nb_x))
    return sum([P[replace_char_by_char('x', fixed_code, format_str.format(code_nb))] for code_nb in range(0,pow(2, nb_x))])


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


def FindBestApproximationToConsistentModel(base, constraint_data):
    res = dict()

    size_prob = meta_data[base]['size_prob']
    base = meta_data[base]['base']
    nb_variables = meta_data[base]['nb_variables']
    B = meta_data[base]['B']
    d = meta_data[base]['d']
    F = meta_data[base]['F']
    c = meta_data[base]['c']
    S_codes = meta_data[base]['S_codes']

    b = np.array([1.0] + [constraint_data[pattern] for pattern in pattern_data[base]['constraint_patterns']])

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


def testModelFromXtoY(base, obsX, obsY, intX, intY, drawObsData, color):
    ExperimentContigenceTable = getContingencyTables(intX, intY, base)
    ObservationContigenceTable = getContingencyTables(obsX, obsY, base)

    WriteContingencyTable(ObservationContigenceTable, base)
    WriteContingencyTable(ExperimentContigenceTable, base)

    return approximateToCausalModel(base, ObservationContigenceTable, ExperimentContigenceTable, drawObsData, color)


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


def preprocessing(data, sort_col, params, base):
    if params['preprocess_method'] == 'discrete_split':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY = split_data(disc_data, sort_col)
    elif params['preprocess_method'] == 'split':
        obsX, obsY, intX, intY = split_data(data, sort_col)
    elif params['preprocess_method'] == 'cluster':
        obsX, obsY, intX, intY = cluster_data(data, sort_col, params)
    elif params['preprocess_method'] == 'discrete_cluster':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY = cluster_data(disc_data, sort_col, params)
    elif params['preprocess_method'] == 'new_strategy':
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY = split_data(disc_data, sort_col)
        mi_ds = mutual_information(getContingencyTables(obsX, obsY, base), base)
        #variation_disc_split = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        obsX, obsY, intX, intY = split_data(data, sort_col)
        mi_s = mutual_information(getContingencyTables(obsX, obsY, base), base)
        #variation_split = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        disc_data = discretize_data(data, params)
        obsX, obsY, intX, intY = cluster_data(disc_data, sort_col, params)
        mi_dc = mutual_information(getContingencyTables(obsX, obsY, base), base)
        #variation_disc_cluster = calc_variations(getContingencyTables(obsX, obsY, base), sort_col)
        #print("v_ds " + str(variation_disc_split))
        #print("v_s " + str(variation_split))
        #print("v_dc " + str(variation_disc_cluster))
        if (mi_ds < min(mi_s,mi_dc)):
            disc_data = discretize_data(data, params)
            obsX, obsY, intX, intY = split_data(disc_data, sort_col)
        elif mi_s < min(mi_ds, mi_dc):
            obsX, obsY, intX, intY = split_data(data, sort_col)
        else:
            disc_data = discretize_data(data, params)
            obsX, obsY, intX, intY = cluster_data(disc_data, sort_col, params)
    else:
        disc_data = discretize_data(data, params)
        cobsX, cobsY, cintX, cintY = cluster_data(data, sort_col, params)
        obsX = cobsX
        obsY = cobsY
        intX = cintX
        intY = cintY
        if (max(get_probabilities(getContingencyTables(obsX, obsY, base), base).values()) > params['prob_threshold_cluster']):
            obsX, obsY, intX, intY = split_data(disc_data, sort_col)
            if (max(get_probabilities(getContingencyTables(obsX, obsY, base), base).values()) <= params['prob_threshold_no_cluster']):
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


def mutual_information(observation_contingence_table, base) -> float:
    nb_max = base
    X = [sum(observation_contingence_table[i]) for i in range(0, nb_max)]
    Y = [sum([observation_contingence_table[i][j] for i in range(0, nb_max)]) for j in range(0, nb_max)]
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


def iacm(base, data: pd.DataFrame, params):
    error_gap = dict()
    pn_error = dict()
    result = dict()
    for sort_col in ['X', 'Y']:
        obsX, obsY, intX, intY = preprocessing(data, sort_col, params, base)
        modelXtoY = testModelFromXtoY(base, obsX, obsY, intX, intY, True, "green")
        modelYtoX = testModelFromXtoY(base, obsY, obsX, intY, intX, False, "yellow")
        errorXtoY = calcError(modelXtoY)
        print("total Error X -> Y: " + str(errorXtoY))
        errorYtoX = calcError(modelYtoX)
        print("total Error Y -> X: " + str(errorYtoX))
        if errorXtoY < errorYtoX:
            print("X -> Y")
            res = "X->Y"
        elif errorXtoY > errorYtoX:
            print("Y -> X")
            res = "Y->X"
        else:
            print("no decision")
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
