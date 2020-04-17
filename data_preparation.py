import pandas as pd
import numpy as np
from MDLP import MDLP_Discretizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering


def read_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv('./pairs/' + filename, sep=" ", header=None)
    if data.shape[1] != 2:
        data = pd.read_csv('./pairs/' + filename, sep="\t", header=None)
        if data.shape[1] != 2:
            data = pd.read_csv('./pairs/' + filename, sep="  ", header=None, engine='python')
    if data.shape[1] == 3:
        data.columns = ["X", "Y", "Z"]
    else:
        data.columns = ["X", "Y"]
    return data


def getContingencyTables(X, Y, base):
    # ctable[x][y]
    ctable = [[1] * base for _ in range(0, base)]
    if X.empty or Y.empty:
        return ctable
    threshold_x = np.quantile(X, [i/base for i in range(1, base+1)])#[0, max(X) + 1]
    threshold_y = np.quantile(Y, [i/base for i in range(1, base+1)])#[0, max(Y) + 1]
    for x, y in zip(X,Y):
        set = False
        for i, thresh_x in enumerate(threshold_x):
            if x <= thresh_x:
                for j, thresh_y in enumerate(threshold_y):
                    if y <= thresh_y:
                        ctable[i][j] = ctable[i][j] + 1
                        set = True
                        break
                if set:
                    break

    return ctable


def getContingencyTables_binary(X, Y):
    ctable = [[1, 1], [1, 1]]
    threshold_x = 0#np.quantile(X, 0.85)
    threshold_y = 0#np.quantile(Y, 0.85)
    for x, y in zip(X, Y):
        if x < threshold_x:
            if y < threshold_y:
                ctable[0][0] = ctable[0][0] + 1
            else:
                ctable[0][1] = ctable[0][1] + 1
        else:
            if y < threshold_y:
                ctable[1][0] = ctable[1][0] + 1
            else:
                ctable[1][1] = ctable[1][1] + 1

    return ctable


def getContingencyTables_ternary(X, Y):
    # ctable[x][y]
    ctable = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    treshold_x = np.quantile(X, [0.25, 0.75])
    treshold_y = np.quantile(Y, [0.25, 0.75])
    for x, y in zip(X,Y):
        if x <= treshold_x[0]:
            if y <= treshold_y[0]:
                ctable[0][0] = ctable[0][0] + 1
            elif y <= treshold_y[1]:
                ctable[0][1] = ctable[0][1] + 1
            else:
                ctable[0][2] = ctable[0][2] + 1
        elif x <= treshold_x[1]:
            if y <= treshold_y[0]:
                ctable[1][0] = ctable[1][0] + 1
            elif y <= treshold_y[1]:
                ctable[1][1] = ctable[1][1] + 1
            else:
                ctable[1][2] = ctable[1][2] + 1
        else:
            if y <= treshold_y[0]:
                ctable[2][0] = ctable[2][0] + 1
            elif y <= treshold_y[1]:
                ctable[2][1] = ctable[2][1] + 1
            else:
                ctable[2][2] = ctable[2][2] + 1
    return ctable


def WriteContingencyTable(contingenceTable, base):
    if base == 2:
        WriteContingencyTable_binary(contingenceTable)
    elif base == 3:
        WriteContingencyTable_ternary(contingenceTable)


def WriteContingencyTable_binary(contingenceTable):
    print("   x |  nx")
    print("y |" + str(contingenceTable[1][1]) + " | " +
          str(contingenceTable[0][1]))
    print("ny|" + str(contingenceTable[1][0]) + " | " +
          str(contingenceTable[0][0]))


def WriteContingencyTable_ternary(contingenceTable):
    print("   x_0 |  x_1 |  x_2")
    print("y_0 |" + str(contingenceTable[0][0]) + " | " + str(contingenceTable[1][0]) + " | " + str(contingenceTable[2][0]))
    print("y_1|" + str(contingenceTable[0][1]) + " | " + str(contingenceTable[1][1]) + " | " + str(contingenceTable[2][1]))
    print("y_2|" + str(contingenceTable[0][2]) + " | " + str(contingenceTable[1][2]) + " | " + str(contingenceTable[2][2]))


def get_probabilities(contingenceTable, base):
    sum = np.sum([float(contingenceTable[i][j]) for i in range(0, base) for j in range(0, base)])

    P = dict()
    for x_i in range(0, base):
        for y_i in range(0, base):
            index = str(x_i) + str(y_i)
            P[index] = contingenceTable[x_i][y_i] / sum
    return P


def get_probabilities_intervention(contingenceTable, base):
    P_i = dict()
    for x_i in range(0, base):
        for y_i in range(0, base):
            index = str(x_i) + '_' + str(y_i)
            denom = np.sum([contingenceTable[x_i][j] for j in range(0, base)])
            if denom == 0:
                P_i[index] = 0.0
            else:
                P_i[index] = contingenceTable[x_i][y_i] / denom
    return P_i


def discretize_data(data, params):
    disc = pd.DataFrame(KBinsDiscretizer(n_bins=params['bins'], encode='ordinal', strategy='uniform').fit_transform(data[['X', 'Y']]))
    disc.columns = ['X', 'Y']
    disc['X'] = disc['X'] - params['x_shift']
    disc['Y'] = disc['Y'] - params['y_shift']
    return disc


def cluster_data(data, col_to_prepare, params):
    cluster = KMeans(n_clusters=params['nb_cluster']).fit(data[['X','Y']])#np.array(data[col_to_prepare]).reshape(-1,1))
    #cluster = SpectralClustering(n_clusters=params['nb_cluster']).fit(data[['X', 'Y']])
    data['labels'] = cluster.labels_
    # calc variances in the cluster
    var_0 = max(data[data['labels'] == 0]['X'].var(), data[data['labels'] == 0]['Y'].var())
    var_1 = max(data[data['labels'] == 1]['X'].var(), data[data['labels'] == 1]['Y'].var())
    var_2 = max(data[data['labels'] == 2]['X'].var(), data[data['labels'] == 2]['Y'].var())
    if 'X' in col_to_prepare:# or 'Y' in col_to_prepare:
        if var_0 < min(var_1, var_2):
            return data[data['labels'] == 0]['X'], data[data['labels'] == 0]['Y'], data[(data['labels'] == 1) | (data['labels'] == 2)]['X'], data[(data['labels'] == 1) | (data['labels'] == 2)]['Y']
        elif var_1 < min(var_0, var_2):
            return data[data['labels'] == 1]['X'], data[data['labels'] == 1]['Y'], data[(data['labels'] == 0) | (data['labels'] == 2)]['X'], data[(data['labels'] == 0) | (data['labels'] == 2)]['Y']
        elif var_2 < min(var_0, var_1):
            return data[data['labels'] == 2]['X'], data[data['labels'] == 2]['Y'], data[(data['labels'] == 0) | (data['labels'] == 1)]['X'], data[(data['labels'] == 0) | (data['labels'] == 1)]['Y']
    else:
        if var_0 < min(var_1, var_2):
            return data[(data['labels'] == 1) | (data['labels'] == 2)]['X'], \
                   data[(data['labels'] == 1) | (data['labels'] == 2)]['Y'], \
                   data[data['labels'] == 0]['X'], data[data['labels'] == 0]['Y']
        elif var_1 < min(var_0, var_2):
            return data[(data['labels'] == 0) | (data['labels'] == 2)]['X'], \
                   data[(data['labels'] == 0) | (data['labels'] == 2)]['Y'], \
                   data[data['labels'] == 1]['X'], data[data['labels'] == 1]['Y']
        elif var_2 < min(var_0, var_1):
            return data[(data['labels'] == 0) | (data['labels'] == 1)]['X'], \
                   data[(data['labels'] == 0) | (data['labels'] == 1)]['Y'], \
                   data[data['labels'] == 2]['X'], data[data['labels'] == 2]['Y']
         # if var_0 > var_1:
         #     return data[data['labels'] == 0]['X'], data[data['labels'] == 0]['Y'], data[data['labels'] == 1]['X'], data[data['labels'] == 1]['Y']
         # else:
         #     return data[data['labels'] == 1]['X'], data[data['labels'] == 1]['Y'], data[data['labels'] == 0]['X'], data[data['labels'] == 0]['Y']


def split_data(data, col_to_prepare):
    sorted_data = data.sort_values(by=[col_to_prepare]).reset_index()
    # find splitting point
    max_diff = 0
    i_max = 0
    for i in range(int(0.2 * sorted_data.shape[0]), int(0.7 * sorted_data.shape[0])):
        if abs(sorted_data[col_to_prepare][i - 1] - sorted_data[col_to_prepare][i]) > max_diff:
            max_diff = abs(sorted_data[col_to_prepare][i - 1] - sorted_data[col_to_prepare][i])
            i_max = i
    #i_max = 100
    return sorted_data['X'][0:i_max], sorted_data['Y'][0:i_max], sorted_data['X'][i_max:], sorted_data['Y'][i_max:]
    #i_max = int(0.4 * sorted_data.shape[0])
    #return pd.concat([data['X'][0:i_max], data['X'][(data.shape[0] - i_max):]],ignore_index=True), \
    #       pd.concat([data['Y'][0:i_max], data['Y'][(data.shape[0] - i_max):]], ignore_index=True), \
    #       data['X'][i_max:(data.shape[0] - i_max)], \
    #       data['Y'][i_max:(data.shape[0] - i_max)]
