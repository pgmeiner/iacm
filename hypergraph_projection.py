from typing import List
from metrics import KL_Dist, entropy
import numpy as np
import cvxpy as cp
from data_preparation import getContingencyTables, get_probabilities_intervention_binary, get_probabilities_binary

def CalcIterativeProjectionToHypergraph(G: List[List[int]], n: int, P: List[float], MaxIter: int):
    # Berechnet Projektion von P auf Exponentialfamilie E_G induziert durch Hypergraph G
    len = int(pow(2, n))
    # 1. Initialisierung
    Q = [1 / pow(2, n)] * int(pow(2, n))
    QNew = [0.0] * len
    # 2. Berechne Marginale
    for iter in range(0, MaxIter):
        for m in range(0, G.__len__()):
            A = G[m]
            alpha = CalcMarginalDistributionA(P, A, n)
            # 3. Improve current approximation
            QMarg = CalcMarginalDistributionA(Q, A, n)
            for l in range(0, len):
                if QMarg[l] > 0:
                    calpha = alpha[l] / QMarg[l]
                    QNew[l] = calpha * Q[l]
            # 4. Update
            Q = QNew
            QNew = [0.0] * len

    return Q


def MarginalA(V: str, A: List[int]) -> str:
    # Realisiert Funktion pi_A(V)=(V_i)_{i in A}, also die Projektion von V auf V_A
    res = ""
    for i in range(0, len(A)):
        res = res + V[A[i] - 1:A[i]]

    return res


def CalcMarginalDistributionA(P: List[float], A: List[int], n: int) -> List[float]:
    RetVal = [0.0] * int(pow(2, n))

    for l in range(0, pow(2, n)):
        sigma = '{0:02b}'.format(l)  # f'{0:0{n}b}'.format(l)
        for j in range(0, pow(2, n)):
            sigmaPrime = '{0:02b}'.format(j)  # f'{0:0{n}b}'.format(j)
            if MarginalA(sigma, A) == MarginalA(sigmaPrime, A):
                RetVal[l] = RetVal[l] + P[j]

    return RetVal

def calc_distance_to_independent_graph(pxy, pxny, pnxy, pnxny, py_x, py_nx, pny_x, pny_nx, color):
    x = max_entropy_with_marginal_constraints(pxy, pxny, pnxy, pnxny, py_x, py_nx, pny_x, pny_nx)
    G = [[1], [2]]
    P = [pxy, pxny, pnxy, pnxny]
    Q = CalcIterativeProjectionToHypergraph(G, 2, P, 10)
    pxy = Q[0]
    pxny = Q[1]
    pnxy = Q[2]
    pnxny = Q[3]
    px = [pxy + pxny, pnxy + pnxny]
    py = [pxy + pnxy, pxny + pnxny]
    # local_mi = entropy(px) + entropy(py) - entropy([pxy, pxny, pnxy, pnxny])
    kl_dist = KL_Dist(P, Q)
    if color != "":
        plot_distribution(pxy=pxy, pxny=pxny, pnxy=pnxy, pnxny=pnxny, color=color)
    print("independent distribution")
    print("Pxy:" + str(pxy) + " Pxny:" + str(pxny) + " Pnxy:" + str(pnxy) + " Pnxny:" + str(pnxny))

    pxy = x.value[12] + x.value[13] + x.value[14] + x.value[15]
    pxny = x.value[8] + x.value[9] + x.value[10] + x.value[11]
    pnxy = x.value[4] + x.value[5] + x.value[6] + x.value[7]
    pnxny = x.value[0] + x.value[1] + x.value[2] + x.value[3]
    local_mi = entropy(px) + entropy(py) - entropy([pxy, pxny, pnxy, pnxny])
    return kl_dist
    # if x.value is not None:
    #     px = [pxy + pxny, pnxy + pnxny]
    #     py = [pxy + pnxy, pxny + pnxny]
    #     mi = multi_information(px, py, [py_x, pny_x], [py_nx, pny_nx], x.value)
    #     pxy = x.value[12] + x.value[13] + x.value[14] + x.value[15]
    #     pxny = x.value[8] + x.value[9] + x.value[10] + x.value[11]
    #     pnxy = x.value[4] + x.value[5] + x.value[6] + x.value[7]
    #     pnxny = x.value[0] + x.value[1] + x.value[2] + x.value[3]
    #     local_mi = entropy(px) + entropy(py) - entropy([pxy, pxny, pnxy, pnxny])
    #     if color != "":
    #         plot_distribution(pxy=pxy, pxny=pxny, pnxy=pnxy, pnxny=pnxny, color=color)
    #     print("independent distribution")
    #     print("Pxy:" + str(pxy) + " Pxny:" + str(pxny) + " Pnxy:" + str(pnxy) + " Pnxny:" + str(pnxny))
    #     return local_mi
    #
    # return None

def max_entropy_with_marginal_constraints(pxy, pxny, pnxy, pnxny, py_x, py_nx, pny_x, pny_nx):
    # Matrix size parameters.
    n = 16

    A = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    b = np.array([1.0, pnxny, pnxy, pxny, pxy, pny_x, py_x, pny_nx, py_nx])

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
    c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0])

    # Entropy maximization.
    x = cp.Variable(shape=n)
    obj = cp.Maximize(cp.sum(cp.entr(x)))
    constraints = [A * x == b,
                   F * x >= c]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return x

def testIndpendentendModel(obsX, obsY, intX, intY, color):
    ExperimentContigenceTable = getContingencyTables(intX, intY)
    ObservationContigenceTable = getContingencyTables(obsX, obsY)
    Pxy, Pxny, Pnxy, Pnxny = get_probabilities_binary(ObservationContigenceTable)
    Py_x, Py_nx, Pny_x, Pny_nx = get_probabilities_intervention_binary(ExperimentContigenceTable)
    return calc_distance_to_independent_graph(Pxy, Pxny, Pnxy, Pnxny, Py_x, Py_nx, Pny_x, Pny_nx, color)
