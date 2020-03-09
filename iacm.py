import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from data_preparation import get_probabilities, get_probabilities_intervention, WriteContingencyTable, \
    getContingencyTables, pre_process_data, discretize_data, cluster_data, split_data
from plot import plot_distribution


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


def generate_codes(pattern):
    nb_x = count_char(pattern, 'x')
    format_str = '{0:0xb}'.replace('x', str(nb_x))
    codes = [replace_char_by_char('x', pattern, format_str.format(nb)) for nb in range(0,pow(2,nb_x))]
    return codes


def convert_binary_to_constraint_line(codes):
    positions = list()
    for code in codes:
        positions.append(int(code,2))
    result = list()
    for i in range(0,pow(2,4)):
        if i in positions:
            result.append(1)
        else:
            result.append(0)
    return result


def zero_codes_binary(code_patterns):
    format_str = '{0:01b}'
    codes = [replace_char_by_char('x', code_pattern, format_str.format(nb)) for nb in range(0,pow(2,1)) for code_pattern in code_patterns]
    return codes


def s_codes_binary(zero_codes):
    format_str = '{0:04b}'
    all_codes = [replace_char_by_char('x', 'xxxx', format_str.format(nb)) for nb in range(0,pow(2,4))]
    return list(set(all_codes) - set(zero_codes))


def generate_binary_constraint_lines(patterns):
    lines = list()
    for pattern in patterns:
        lines.append(convert_binary_to_constraint_line(generate_codes(pattern)))

    return lines


def setup_meta_data_binary():
    meta_data = dict()
    size_prob = pow(2, 4)
    meta_data['size_prob'] = size_prob

    lines = generate_binary_constraint_lines(['xxx1', 'xx1x', '01xx', '10xx', '11xx'])
    meta_data['B'] = np.array([[1] * size_prob] + lines)

    zero_codes = zero_codes_binary(['00x1', '110x', '101x', '01x0'])
    S_codes = s_codes_binary(zero_codes)
    meta_data['S_codes'] = S_codes
    d_list = list()
    for i in range(0, size_prob):
        format_str = '{0:04b}'
        if format_str.format(i) in S_codes:
            d_list.append(1)
        else:
            d_list.append(0)
    meta_data['d'] = np.array(d_list)  # np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

    meta_data['F'] = np.diag(np.array([1] * size_prob))
    meta_data['c'] = np.array([0.0] * size_prob)

    return meta_data


meta_data_binary = setup_meta_data_binary()


def approximateToCausalModel(obsConTable, ExpConTable, drawObsData, color):
    Pxy, Pxny, Pnxy, Pnxny = get_probabilities(obsConTable)
    Py_x, Py_nx, Pny_x, Pny_nx = get_probabilities_intervention(ExpConTable)
    print("Pxy:" + str(Pxy) + " Pxny:" + str(Pxny) + " Pnxy:" + str(Pnxy) + " Pnxny:" + str(Pnxny))
    if drawObsData:
        plot_distribution(Pxy, Pxny, Pnxy, Pnxny, "black")
    print("Py_x:" + str(Py_x) + " Py_nx:" + str(Py_nx) + " Pny_x:" + str(Pny_x) + " Pny_nx:" + str(Pny_nx))
    modeldata = FindBestApproximationToConsistentModel_binary(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx)
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


def FindBestApproximationToConsistentModel_binary(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx):
    res = dict()

    size_prob = meta_data_binary['size_prob']
    B = meta_data_binary['B']
    d = meta_data_binary['d']
    F = meta_data_binary['F']
    c = meta_data_binary['c']
    S_codes = meta_data_binary['S_codes']

    b = np.array([1.0, Py_nx, Py_x, Pnxy, Pxny, Pxy])

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

    for i in range(0,len(SimplexRes)):
        code = '{0:04b}'.format(i)
        P[code] = max(SimplexRes[i], 0.0)

    #S_codes = ['0000', '0010', '0111', '1000', '1110', '1111', '0101', '1001']
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

    res["P"] = P
    res["Pxy"] = marginalDistribution(NP, '11xx')
    res["Pxny"] = marginalDistribution(NP, '10xx')
    res["Pnxy"] = marginalDistribution(NP, '01xx')
    res["Pnxny"] = marginalDistribution(NP, '00xx')
    res["Px"] = res["Pxny"] + res["Pxy"]
    res["Py"] = res["Pxy"] + res["Pnxy"]

    res["Py_x"] = marginalDistribution(NP, 'xx1x')
    res["Pny_x"] = marginalDistribution(NP, 'xx0x')
    res["Py_nx"] = marginalDistribution(NP, 'xxx1')
    res["Pny_nx"] = marginalDistribution(NP, 'xxx0')
    res["GlobalError"] = log2(1 / S)
    res["LocalErrorE"] = res["GlobalError"]
    res["LocalErrornE"] = res["GlobalError"]
    res["LocalErrorB"] = res["GlobalError"]

    Py_nx = marginalDistribution(P, 'xxx1')
    Py_x_nx = marginalDistribution(P, 'x111')
    res["LocalErrorE"] = res["LocalErrorE"] + localError(Py_x_nx, Py_nx, S)

    Pny_nx = marginalDistribution(P, 'xxx0')
    Pny_x_nx_ = P['0000'] + P['0010'] + P['1000'] + P['1110']
    res["LocalErrorE"] = res["LocalErrorE"] + localError(Pny_x_nx_, Pny_nx, S)

    Pny_x =marginalDistribution(P, 'xx0x')
    Pny_x_nx = marginalDistribution(P, 'x000')
    res["LocalErrornE"] = res["LocalErrornE"] + localError(Pny_x_nx, Pny_x, S)

    Py_x = marginalDistribution(P, 'xx1x')
    Py_x_nx_ = P['0010'] + P['0111'] + P['1110'] + P['1111']
    res["LocalErrornE"] = res["LocalErrornE"] + localError(Py_x_nx_, Py_x, S)

    Pnynx = marginalDistribution(P, '00xx')
    Pnynx_nx = marginalDistribution(P, '00x0')
    res["LocalErrorB"] = res["LocalErrorB"] + localError(Pnynx_nx, Pnynx, S)

    Pynx = marginalDistribution(P, '01xx')
    Pynx_x = marginalDistribution(P, '01x1')
    res["LocalErrorB"] = res["LocalErrorB"] + localError(Pynx_x, Pynx, S)

    Pnyx = marginalDistribution(P, '10xx')
    Pnyx_nx = marginalDistribution(P, '100x')
    res["LocalErrorB"] = res["LocalErrorB"] + localError(Pnyx_nx, Pnyx, S)

    Pyx = marginalDistribution(P, '11xx')
    Pyx_x = marginalDistribution(P, '111x')
    res["LocalErrorB"] = res["LocalErrorB"] + localError(Pyx_x, Pyx, S)

    return res


def FindBestApproximationToConsistentModel_tertiar(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx):
    res = dict()

    # init simplex data
    B = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    d = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

    F = np.diag(np.array([1]*pow(3,5)))
    c = np.array([0.0] * pow(3, 5))

    b = np.array([1.0, Py_nx, Py_x, Pnxy, Pxny, Pxy])

    # create and run the solver
    x = cp.Variable(shape=pow(3,5))
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

    for i in range(0,len(SimplexRes)):
        code = '{0:04b}'.format(i)
        P[code] = max(SimplexRes[i], 0.0)

    S_codes = ['0000', '0010', '0111', '1000', '1110', '1111', '0101', '1001']
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

    res["P"] = P
    res["Pxy"] = marginalDistribution(NP, '11xx')
    res["Pxny"] = marginalDistribution(NP, '10xx')
    res["Pnxy"] = marginalDistribution(NP, '01xx')
    res["Pnxny"] = marginalDistribution(NP, '00xx')
    res["Px"] = res["Pxny"] + res["Pxy"]
    res["Py"] = res["Pxy"] + res["Pnxy"]

    res["Py_x"] = marginalDistribution(NP, 'xx1x')
    res["Pny_x"] = marginalDistribution(NP, 'xx0x')
    res["Py_nx"] = marginalDistribution(NP, 'xxx1')
    res["Pny_nx"] = marginalDistribution(NP, 'xxx0')
    res["GlobalError"] = log2(1 / S)

    return res


def testModelFromXtoY(obsX, obsY, intX, intY, drawObsData, color):
    ExperimentContigenceTable = getContingencyTables(intX, intY)
    ObservationContigenceTable = getContingencyTables(obsX, obsY)

    WriteContingencyTable(ObservationContigenceTable)
    WriteContingencyTable(ExperimentContigenceTable)

    return approximateToCausalModel(ObservationContigenceTable, ExperimentContigenceTable, drawObsData, color)


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

def iacm(data: pd.DataFrame):
    data = pre_process_data(data)
    error_gap = dict()
    result = dict()
    for sort_col in ['X', 'Y']:
        disc_data = discretize_data(data, sort_col)
        cobsX, cobsY, cintX, cintY = cluster_data(disc_data, sort_col)
        obsX = cobsX
        obsY = cobsY
        intX = cintX
        intY = cintY
        #obsX, obsY, intX, intY = generate_nonlinear_confounded_data(100)
        obsTable = getContingencyTables(obsX, obsY)
        intTable = getContingencyTables(intX, intY)
        #if (min(obsTable[0][0]+obsTable[0][1], obsTable[1][0]+obsTable[1][1]) / \
        #    max(obsTable[0][0] + obsTable[0][1], obsTable[1][0] + obsTable[1][1]) < 0.002) or \
        #    (min(intTable[0][0] + intTable[0][1], intTable[1][0] + intTable[1][1]) / \
        #     max(intTable[0][0] + intTable[0][1], intTable[1][0] + intTable[1][1]) < 0.002) or \
        if (max(get_probabilities(getContingencyTables(obsX, obsY))) > 0.7):
            obsX, obsY, intX, intY = split_data(disc_data, sort_col)
            if (max(get_probabilities(getContingencyTables(obsX, obsY))) <= 0.3):
                obsX = cobsX
                obsY = cobsY
                intX = cintX
                intY = cintY
        #if max(get_probabilities(getContingencyTables(obsX, obsY))) > 0.9:# and max(get_probabilities_intervention(getContingencyTables(intX, intY))) > 0.9:
        #    obsX, obsY, intX, intY = split_data(disc_data, sort_col)
        modelXtoY = testModelFromXtoY(obsX, obsY, intX, intY, True, "green")
        modelYtoX = testModelFromXtoY(obsY, obsX, intY, intX, False, "yellow")
        errorXtoY = calcError(modelXtoY)
        print("total Error X -> Y: " + str(errorXtoY))
        errorYtoX = calcError(modelYtoX)
        print("total Error Y -> X: " + str(errorYtoX))
        res = ""
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
        error_gap[sort_col] = min(errorXtoY, errorYtoX) / max(errorXtoY, errorYtoX)

    if error_gap['X'] == 1 and error_gap['Y'] == 1:
        return "no decision"
    elif (1 / error_gap['X']) < (1 / error_gap['Y']):
        return result['Y']
    else:
        return result['X']
    # for color in ['black', 'green', 'yellow', 'red']:
    #    ax.scatter(scatter_points[color]['x'], scatter_points[color]['y'], scatter_points[color]['z'], color=color, linewidth=1, s=2)
    # plt.show()
