import cvxpy as cp
import numpy as np
from math import log2
import pandas as pd
from data_preparation import get_probabilities, get_probabilities_intervention, WriteContingencyTable, \
    getContingencyTables, pre_process_data, discretize_data, cluster_data, split_data
from plot import plot_distribution

def approximateToCausalModel(obsConTable, ExpConTable, drawObsData, color):
    Pxy, Pxny, Pnxy, Pnxny = get_probabilities(obsConTable)
    Py_x, Py_nx, Pny_x, Pny_nx = get_probabilities_intervention(ExpConTable)
    print("Pxy:" + str(Pxy) + " Pxny:" + str(Pxny) + " Pnxy:" + str(Pnxy) + " Pnxny:" + str(Pnxny))
    if drawObsData:
        plot_distribution(Pxy, Pxny, Pnxy, Pnxny, "black")
    print("Py_x:" + str(Py_x) + " Py_nx:" + str(Py_nx) + " Pny_x:" + str(Pny_x) + " Pny_nx:" + str(Pny_nx))
    modeldata = FindBestApproximationToConsistentModel(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx)
    print("approximated distribution")
    if "Pxy" in modeldata:
        print("Pxy:" + str(modeldata["Pxy"]) + " Pxny:" + str(modeldata["Pxny"]) + " Pnxy:" + str(
            modeldata["Pnxy"]) + " Pnxny:" + str(modeldata["Pnxny"]))
        plot_distribution(modeldata["Pxy"], modeldata["Pxny"], modeldata["Pnxy"], modeldata["Pnxny"], color)
    return modeldata


def FindBestApproximationToConsistentModel(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx):
    res = dict()

    # init simplex data
    B = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    d = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

    F = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    b = np.array([1.0, Py_nx, Py_x, Pnxy, Pxny, Pxy])
    c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # create and run the solver
    x = cp.Variable(shape=16)
    obj = cp.Minimize(cp.sum(d * x))
    constraints = [B * x == b,
                   F * x >= c]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # get the solution
    if x.value is None:
        return res

    SimplexRes = x.value

    P0000 = max(SimplexRes[0], 0.0)
    P0001 = max(SimplexRes[1], 0.0)
    P0010 = max(SimplexRes[2], 0.0)
    P0011 = max(SimplexRes[3], 0.0)
    P0100 = max(SimplexRes[4], 0.0)
    P0101 = max(SimplexRes[5], 0.0)
    P0110 = max(SimplexRes[6], 0.0)
    P0111 = max(SimplexRes[7], 0.0)
    P1000 = max(SimplexRes[8], 0.0)
    P1001 = max(SimplexRes[9], 0.0)
    P1010 = max(SimplexRes[10], 0.0)
    P1011 = max(SimplexRes[11], 0.0)
    P1100 = max(SimplexRes[12], 0.0)
    P1101 = max(SimplexRes[13], 0.0)
    P1110 = max(SimplexRes[14], 0.0)
    P1111 = max(SimplexRes[15], 0.0)

    S = P0000 + P0010 + P0111 + P1000 + P1110 + P1111 + P0101 + P1001
    if S == 0:
        return res

    NP0000 = P0000 / S
    NP0001 = 0
    NP0010 = P0010 / S
    NP0011 = 0
    NP0100 = 0
    NP0101 = P0101 / S
    NP0110 = 0
    NP0111 = P0111 / S
    NP1000 = P1000 / S
    NP1001 = P1001 / S
    NP1010 = 0
    NP1011 = 0
    NP1100 = 0
    NP1101 = 0
    NP1110 = P1110 / S
    NP1111 = P1111 / S

    totalSum = NP0000 + NP0001 + NP0010 + NP0011 + NP0100 + NP0101 + NP0110 + NP0111 + NP1000 + NP1001 + NP1010 + NP1011 + NP1100 + NP1101 + NP1110 + NP1111

    res["P0000"] = P0000
    res["P0001"] = P0001
    res["P0010"] = P0010
    res["P0011"] = P0011
    res["P0100"] = P0100
    res["P0101"] = P0101
    res["P0110"] = P0110
    res["P0111"] = P0111
    res["P1000"] = P1000
    res["P1001"] = P1001
    res["P1010"] = P1010
    res["P1011"] = P1011
    res["P1100"] = P1100
    res["P1101"] = P1101
    res["P1110"] = P1110
    res["P1111"] = P1111

    res["Pxy"] = NP1100 + NP1101 + NP1110 + NP1111
    res["Pxny"] = NP1000 + NP1001 + NP1010 + NP1011
    res["Pnxy"] = NP0100 + NP0101 + NP0110 + NP0111
    res["Pnxny"] = NP0000 + NP0001 + NP0010 + NP0011
    res["Px"] = res["Pxny"] + res["Pxy"]
    res["Py"] = res["Pxy"] + res["Pnxy"]

    res["Py_x"] = NP0010 + NP0011 + NP0110 + NP0111 + NP1010 + NP1011 + NP1110 + NP1111
    res["Pny_x"] = NP0000 + NP0001 + NP0100 + NP0101 + NP1000 + NP1001 + NP1100 + NP1101
    res["Py_nx"] = NP0001 + NP0011 + NP0101 + NP0111 + NP1001 + NP1011 + NP1101 + NP1111
    res["Pny_nx"] = NP0000 + NP0010 + NP0100 + NP0110 + NP1000 + NP1010 + NP1100 + NP1110
    res["GlobalError"] = log2(1 / S)
    res["LocalErrorE"] = res["GlobalError"]
    res["LocalErrornE"] = res["GlobalError"]
    res["LocalErrorB"] = res["GlobalError"]
    if (P0001 + P0011 + P0101 + P0111 + P1001 + P1011 + P1101 + P1111) > 0 and (P0111 + P1111) > 0:
        res["LocalErrorE"] = res["LocalErrorE"] + 1 / S * (P0111 + P1111) * log2(
            (P0111 + P1111) / (P0001 + P0011 + P0101 + P0111 + P1001 + P1011 + P1101 + P1111))
    elif (P0111 + P1111) == 0:
        res["LocalErrorE"] = res["LocalErrorE"] + 0
    elif (P0111 + P1111) > 0:
        res["LocalErrorE"] = 1000000.0

    if (P0000 + P0010 + P0100 + P0110 + P1000 + P1010 + P1100 + P1110) > 0 and (P0000 + P0010 + P1000 + P1110) > 0:
        res["LocalErrorE"] = res["LocalErrorE"] + 1 / S * (P0000 + P0010 + P1000 + P1110) * log2(
            (P0000 + P0010 + P1000 + P1110) / (P0000 + P0010 + P0100 + P0110 + P1000 + P1010 + P1100 + P1110))
    elif (P0000 + P0010 + P1000 + P1110) == 0:
        res["LocalErrorE"] = res["LocalErrorE"] + 0
    elif (P0000 + P0010 + P1000 + P1110) > 0:
        res["LocalErrorE"] = 1000000.0

    if (P0000 + P0001 + P0100 + P0101 + P1000 + P1001 + P1100 + P1101) > 0 and (P0000 + P1000) > 0:
        res["LocalErrornE"] = res["LocalErrornE"] + 1 / S * (P0000 + P1000) * log2(
            (P0000 + P1000) / (P0000 + P0001 + P0100 + P0101 + P1000 + P1001 + P1100 + P1101))
    elif (P0000 + P1000) == 0:
        res["LocalErrornE"] = res["LocalErrornE"] + 0
    elif (P0000 + P1000) > 0:
        res["LocalErrornE"] = 1000000.0
    if (P0010 + P0011 + P0110 + P0111 + P1010 + P1011 + P1110 + P1111) > 0 and (P0010 + P0111 + P1110 + P1111) > 0:
        res["LocalErrornE"] = res["LocalErrornE"] + 1 / S * (P0010 + P0111 + P1110 + P1111) * log2(
            (P0010 + P0111 + P1110 + P1111) / (P0010 + P0011 + P0110 + P0111 + P1010 + P1011 + P1110 + P1111))
    elif (P0010 + P0111 + P1110 + P1111) == 0:
        res["LocalErrornE"] = res["LocalErrornE"] + 0
    elif (P0010 + P0111 + P1110 + P1111) > 0:
        res["LocalErrornE"] = 1000000.0

    if (P0000 + P0001 + P0010 + P0011) > 0 and (P0000 + P0010) > 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 1 / S * (P0000 + P0010) * log2(
            (P0000 + P0010) / (P0000 + P0001 + P0010 + P0011))
    elif (P0000 + P0010) == 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 0
    elif (P0000 + P0010) > 0:
        res["LocalErrorB"] = 1000000.0
    if (P0100 + P0101 + P0110 + P0111) > 0 and (P0111 + P0101) > 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 1 / S * (P0111 + P0101) * log2(
            (P0111 + P0101) / (P0100 + P0101 + P0110 + P0111))
    elif (P0111 + P0101) == 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 0
    elif (P0111 + P0101) > 0:
        res["LocalErrorB"] = 1000000.0
    if (P1000 + P1001 + P1010 + P1011) > 0 and (P1000 + P1001) > 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 1 / S * (P1000 + P1001) * log2(
            (P1000 + P1001) / (P1000 + P1001 + P1010 + P1011))
    elif (P1000 + P1001) == 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 0
    elif (P1000 + P1001) > 0:
        res["LocalErrorB"] = 1000000.0
    if (P1100 + P1101 + P1110 + P1111) > 0 and (P1110 + P1111) > 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 1 / S * (P1110 + P1111) * log2(
            (P1110 + P1111) / (P1100 + P1101 + P1110 + P1111))
    elif (P1110 + P1111) == 0:
        res["LocalErrorB"] = res["LocalErrorB"] + 0
    elif (P1110 + P1111) > 0:
        res["LocalErrorB"] = 1000000.0
    return res

def testModelFromXtoY(obsX, obsY, intX, intY, drawObsData, color):
    ExperimentContigenceTable = getContingencyTables(intX, intY)
    ObservationContigenceTable = getContingencyTables(obsX, obsY)

    WriteContingencyTable(ObservationContigenceTable)
    WriteContingencyTable(ExperimentContigenceTable)

    return approximateToCausalModel(ObservationContigenceTable, ExperimentContigenceTable, drawObsData, color)

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
