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
            data = pd.read_csv('./pairs/' + filename, sep="  ", header=None)
    if data.shape[1] == 3:
        data.columns = ["X", "Y", "Z"]
    else:
        data.columns = ["X", "Y"]
    return data

def getContingencyTables(X, Y):
    ctable = [[1, 1], [1, 1]]
    for x, y in zip(X, Y):
        if x < 0:
            if y < 0:
                ctable[0][0] = ctable[0][0] + 1
            else:
                ctable[0][1] = ctable[0][1] + 1
        else:
            if y < 0:
                ctable[1][0] = ctable[1][0] + 1
            else:
                ctable[1][1] = ctable[1][1] + 1

    return ctable

def WriteContingencyTable(contingenceTable):
    print("   x |  nx")
    print("y |" + str(contingenceTable[1][1]) + " | " +
          str(contingenceTable[0][1]))
    print("ny|" + str(contingenceTable[1][0]) + " | " +
          str(contingenceTable[0][0]))

def get_probabilities(contingenceTable):
    sum = float(contingenceTable[1][1] + contingenceTable[1][0] +
                contingenceTable[0][1] + contingenceTable[0][0])
    Pxy = contingenceTable[1][1] / sum
    Pnxny = contingenceTable[0][0] / sum
    Pxny = contingenceTable[1][0] / sum
    Pnxy = contingenceTable[0][1] / sum
    return Pxy, Pxny, Pnxy, Pnxny


def get_probabilities_intervention(contingenceTable):
    Py_x = 0.0
    Py_nx = 0.0
    if (contingenceTable[1][1] + contingenceTable[1][0]) == 0:
        Py_x = 0.0
    else:
        Py_x = contingenceTable[1][1] / float(contingenceTable[1][1] + contingenceTable[1][0])
    if (contingenceTable[0][1] + contingenceTable[0][0]) == 0:
        Py_nx = 0.0
    else:
        Py_nx = contingenceTable[0][1] / float(contingenceTable[0][1] + contingenceTable[0][0])

    Pny_x = 0.0
    Pny_nx = 0.0
    if (contingenceTable[1][0] + contingenceTable[1][1]) == 0:
        Pny_x = 0.0
    else:
        Pny_x = contingenceTable[1][0] / float(contingenceTable[1][0] + contingenceTable[1][1])
    if (contingenceTable[0][1] + contingenceTable[0][0]) == 0:
        Pny_nx = 0.0
    else:
        Pny_nx = contingenceTable[0][0] / float(contingenceTable[0][1] + contingenceTable[0][0])

    return Py_x, Py_nx, Pny_x, Pny_nx

def pre_process_data(data):
    m_x = data['X'].median()
    m_y = data['Y'].median()
    data['X'] = data['X'] - m_x
    data['Y'] = data['Y'] - m_y
    return data


def discretize_data(data, col_to_prepare):
    disc = pd.DataFrame(KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform').fit_transform(data))
    disc.columns = ['X', 'Y']
    disc['X'] = disc['X'] - 1
    disc['Y'] = disc['Y'] - 1
    return disc

def cluster_data(data, col_to_prepare):
    cluster = KMeans(n_clusters=3).fit(data)#np.array(data[col_to_prepare]).reshape(-1,1))
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
    return sorted_data['X'][0:i_max], sorted_data['Y'][0:i_max], sorted_data['X'][i_max:], sorted_data['Y'][i_max:]
    #i_max = int(0.4 * sorted_data.shape[0])
    #return pd.concat([data['X'][0:i_max], data['X'][(data.shape[0] - i_max):]],ignore_index=True), \
    #       pd.concat([data['Y'][0:i_max], data['Y'][(data.shape[0] - i_max):]], ignore_index=True), \
    #       data['X'][i_max:(data.shape[0] - i_max)], \
    #       data['Y'][i_max:(data.shape[0] - i_max)]
