import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Any


def read_data(directory: str, filename: str) -> pd.DataFrame:
    data = pd.read_csv(directory + '/' + filename, sep=" ", header=None)
    if data.shape[1] != 2:
        data = pd.read_csv(directory + '/' + filename, sep="\t", header=None)
        if data.shape[1] != 2:
            data = pd.read_csv(directory + '/' + filename, sep="  ", header=None, engine='python')
    if data.shape[1] == 3:
        data.columns = ["X", "Y", "Z"]
    else:
        data.columns = ["X", "Y"]
    return data


def read_synthetic_data(directory: str, filename: str) -> pd.DataFrame:
    lines = open(directory + '/' + filename).readlines()
    x = list()
    y = list()
    for line in lines:
        x.append(float(" ".join(line.lstrip().split()).split()[0]))
        y.append(float(" ".join(line.lstrip().split()).split()[1]))

    return pd.DataFrame({'X': x, 'Y': y})


def get_contingency_table(x: pd.Series, y: pd.Series, base_x: int, base_y: int) -> List[List[int]]:
    # ctable[x][y]
    ctable = [[1] * base_x for _ in range(0, base_y)]
    if x.empty or y.empty:
        return ctable
    threshold_x = np.quantile(x, [i/base_x for i in range(1, base_x+1)])
    threshold_y = np.quantile(y, [i/base_y for i in range(1, base_y+1)])
    for x, y in zip(x, y):
        filled = False
        for i, thresh_x in enumerate(threshold_x):
            if x <= thresh_x:
                for j, thresh_y in enumerate(threshold_y):
                    if y <= thresh_y:
                        ctable[i][j] = ctable[i][j] + 1
                        filled = True
                        break
                if filled:
                    break

    return ctable


def write_contingency_table(contingency_table: List[List[int]], base_x: int, base_y: int):
    if base_x == 2 and base_y == 2:
        write_contingency_table_binary(contingency_table)
    elif base_x == 3 and base_y == 3:
        write_contingency_table_ternary(contingency_table)


def write_contingency_table_binary(contingency_table: List[List[int]]):
    print("   x |  nx")
    print("y |" + str(contingency_table[1][1]) + " | " +
          str(contingency_table[0][1]))
    print("ny|" + str(contingency_table[1][0]) + " | " +
          str(contingency_table[0][0]))


def write_contingency_table_ternary(contingency_table: List[List[int]]):
    print("   x_0 |  x_1 |  x_2")
    print("y_0 |" + str(contingency_table[0][0]) + " | " + str(contingency_table[1][0]) + " | " +
          str(contingency_table[2][0]))
    print("y_1|" + str(contingency_table[0][1]) + " | " + str(contingency_table[1][1]) + " | " +
          str(contingency_table[2][1]))
    print("y_2|" + str(contingency_table[0][2]) + " | " + str(contingency_table[1][2]) + " | " +
          str(contingency_table[2][2]))


def get_probabilities(contingency_table: List[List[int]], base_x: int, base_y: int) -> Dict[str, float]:
    table_sum = np.sum([float(contingency_table[i][j]) for i in range(0, base_x) for j in range(0, base_y)])

    p = dict()
    for x_i in range(0, base_x):
        for y_i in range(0, base_y):
            index = str(x_i) + str(y_i)
            p[index] = contingency_table[x_i][y_i] / table_sum
    return p


def get_probabilities_intervention(contingency_table: List[List[int]], base_x: int, base_y: int) -> Dict[str, float]:
    p_intervention = dict()
    for x_i in range(0, base_x):
        for y_i in range(0, base_y):
            index = str(x_i) + '_' + str(y_i)
            denom = np.sum([contingency_table[x_i][j] for j in range(0, base_y)])
            if denom == 0:
                p_intervention[index] = 0.0
            else:
                p_intervention[index] = contingency_table[x_i][y_i] / denom
    return p_intervention


def discretize_data(data: pd.DataFrame, bins: int) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc = pd.DataFrame(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform').
                            fit_transform(data[['X', 'Y']]))
        disc.columns = ['X', 'Y']
        return disc


def cluster_data(data: pd.DataFrame, intervention_column: str, nb_clusters: int) \
        -> Tuple[Tuple[Any, Any, Any, Any], pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster = KMeans(n_clusters=nb_clusters).fit(data[['X', 'Y']])
        data['labels'] = cluster.labels_
    return split_at_clustered_labels(data, intervention_column, nb_clusters), data


def split_at_clustered_labels(data: pd.DataFrame, intervention_column: str, nb_clusters: int) \
        -> Tuple[Any, Any, Any, Any]:
    # calc variances in the clusters
    var = dict()
    for i_cluster in range(nb_clusters):
        var[i_cluster] = data[data['labels'] == i_cluster][intervention_column].var()

    for i_cluster in range(nb_clusters):
        if not np.isnan(var[i_cluster]) and \
                var[i_cluster] <= min([val for i, val in var.items() if (i is not i_cluster and not np.isnan(val))]):
            return data[data['labels'] == i_cluster]['X'], \
                   data[data['labels'] == i_cluster]['Y'], \
                   data[(data['labels'] != i_cluster)]['X'], \
                   data[(data['labels'] != i_cluster)]['Y']


def split_data(data, col_to_prepare, sort_data=True):
    if sort_data:
        prep_data = data.sort_values(by=[col_to_prepare]).reset_index()
    else:
        prep_data = data
    # find splitting point
    max_diff = 0
    i_max = 0
    # find splitting range
    alphabet_size = len(set(prep_data[col_to_prepare].tolist()))
    upper_range = int(0.7*prep_data.shape[0])
    for i in range(int(0.7*prep_data.shape[0]), prep_data.shape[0]):
        if len(set(prep_data[col_to_prepare][:i].tolist())) > int(alphabet_size*0.4):
            upper_range = i
            break

    for i in range(int(0.2 * prep_data.shape[0]), upper_range):
        if abs(prep_data[col_to_prepare][i - 1] - prep_data[col_to_prepare][i]) > max_diff:
            max_diff = abs(prep_data[col_to_prepare][i - 1] - prep_data[col_to_prepare][i])
            i_max = i
    return prep_data['X'][0:i_max], prep_data['Y'][0:i_max], prep_data['X'][i_max:], prep_data['Y'][i_max:], i_max


def split_data_at_index(data: pd.DataFrame, idx: int) -> Tuple[Any, Any, Any, Any]:
    return data['X'][0:idx], data['Y'][0:idx], data['X'][idx:], data['Y'][idx:]
