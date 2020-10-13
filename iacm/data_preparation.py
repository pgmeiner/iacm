import pandas as pd
import numpy as np
import warnings
import itertools
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Any

from iacm.metrics import kl_divergence_x_y


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


def get_contingency_table_general(data: pd.DataFrame, bases: Dict[str, int]) -> np.ndarray:
    # ctable[x,y,z,...]
    ctable = np.ones(tuple([v for v in bases.values()]))
    if any([data[k].empty for k in data.keys()]):
        return ctable
    thresholds = dict()
    for var_name in data.keys():
        thresholds[var_name] = np.quantile(data[var_name], [i / bases[var_name] for i in range(1, bases[var_name] + 1)])
        if thresholds[var_name][0] == thresholds[var_name][len(thresholds[var_name])-1]:
            thresholds[var_name][0] = 0

    for index, row in data.iterrows():
        table_index = list()
        for var_name in data.keys():
            for i, thres in enumerate(thresholds[var_name]):
                if row[var_name] <= thres:
                    table_index.append(i)
                    break
        table_index = tuple(table_index)
        ctable[table_index] = ctable[table_index] + 1

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


def get_probabilities_general(contingency_table: np.ndarray, bases: Dict[str, int]) -> Dict[str, float]:
    table_sum = contingency_table.sum()

    p = dict()
    for index_combination in itertools.product(*tuple([''.join([str(v) for v in range(0, base)]) for base in bases.values()])):
        index = tuple([int(v) for v in index_combination])
        p_index = ''.join(index_combination)
        p[p_index] = contingency_table[index] / table_sum

    return p


def get_probabilities_intervention_general(contingency_table: np.ndarray, bases: Dict[str, int], intervention_variables: List[str],
                                           hidden_variables: List[str]) -> Dict[str, Dict[str, float]]:
    p_intervention = dict()

    latent_intervention_variables = intervention_variables
    if len(latent_intervention_variables) == 0 and len(hidden_variables) > 0:
        latent_intervention_variables = hidden_variables
        new_bases = bases.copy()
        new_bases['z'] = 2
    else:
        new_bases = bases.copy()

    for intervention_variable in latent_intervention_variables:
        intervention_variable = intervention_variable.lower()
        for obs_var in bases.keys():
            if obs_var == intervention_variable:
                continue
            p_intervention[obs_var] = dict()
            for observation in range(0, bases[obs_var]):
                for intervention in range(0, new_bases[intervention_variable]):
                    intervention_index = dict(zip(bases.keys(), [':'] * len(bases.keys())))
                    intervention_index[intervention_variable] = str(intervention)
                    index = str(intervention) + '_' + str(observation)
                    denom = eval('contingency_table[' + ','.join(intervention_index.values()) + '].sum()')
                    if denom == 0:
                        p_intervention[obs_var][index] = 0.0
                    else:
                        intervention_index[obs_var] = str(observation)
                        p_intervention[obs_var][index] = eval('contingency_table[' + ','.join(intervention_index.values()) + '].sum()') / denom

    # contingency_tables = dict({k: None for k in bases.keys()})
    # contingency_tables['x'] = np.ones(tuple([v for v in bases.values()]))
    # contingency_tables['x'][0, 0] = contingency_table[0, 0]
    # contingency_tables['x'][1, 1] = contingency_table[1, 1]
    # contingency_tables['y'] = np.ones(tuple([v for v in bases.values()]))
    # contingency_tables['y'][0, 1] = contingency_table[0, 1]
    # contingency_tables['y'][1, 0] = contingency_table[1, 0]
    # for _ in hidden_variables:
    #     for obs_var in bases.keys():
    #         p_intervention[obs_var] = dict()
    #         for observation in range(0, bases[obs_var]):
    #             for intervention in range(0, 2):
    #                 intervention_index = dict(zip(bases.keys(), [':'] * len(bases.keys())))
    #                 index = str(intervention) + '_' + str(observation)
    #                 denom = eval('contingency_tables[\''+ obs_var + '\'][' + ','.join(intervention_index.values()) + '].sum()')
    #                 if denom == 0:
    #                     p_intervention[obs_var][index] = 0.0
    #                 else:
    #                     intervention_index[obs_var] = str(observation)
    #                     p_intervention[obs_var][index] = eval('contingency_tables[\''+ obs_var + '\'][' + ','.join(intervention_index.values()) + '].sum()') / denom

    return p_intervention


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


def discretize_data(data: pd.DataFrame, bins: int, observation_variables: List[str]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc = pd.DataFrame(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform').
                            fit_transform(data[observation_variables]))
        disc.columns = observation_variables
        return disc


def cluster_data(data: pd.DataFrame, intervention_column: str, observation_variables: List[str], nb_clusters: int) \
        -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster = KMeans(n_clusters=nb_clusters).fit(data[observation_variables])
        data['labels'] = cluster.labels_
    return split_at_clustered_labels(data, intervention_column, observation_variables, nb_clusters), data


def split_at_clustered_labels(data: pd.DataFrame, intervention_column: str, observation_variables: List[str], nb_clusters: int) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # calc variances in the clusters
    var = dict()
    if intervention_column == '':
        intervention_column = 'X'

    for i_cluster in range(nb_clusters):
        var[i_cluster] = data[data['labels'] == i_cluster][intervention_column].var()

    for i_cluster in range(nb_clusters):
        if not np.isnan(var[i_cluster]) and \
                var[i_cluster] <= min([val for i, val in var.items() if (i is not i_cluster and not np.isnan(val))]):
            obs_pdf = pd.DataFrame({col.lower(): data[data['labels'] == i_cluster][col] for col in observation_variables})
            int_pdf = pd.DataFrame({col.lower(): data[(data['labels'] != i_cluster)][col] for col in observation_variables})
            return obs_pdf, int_pdf


def split_with_balancing(data: pd.DataFrame, intervention_column: str, observation_variables: List[str]) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    intervention_alphabet = data[intervention_column].unique()
    nb_intervention_alphabet = len(intervention_alphabet)
    bucket_pool = dict()
    for int_letter in intervention_alphabet:
        bucket_pool[int_letter] = data[data[intervention_column] == int_letter][observation_variables].reset_index(drop=True)

    bucket_size = max(data[intervention_column].value_counts().tolist())
    obs_bucket = []
    for i in range(0, bucket_size):
        pool_to_draw = intervention_alphabet[np.random.randint(0, nb_intervention_alphabet)]
        pool_size = bucket_pool[pool_to_draw].shape[0]
        data_to_draw = np.random.randint(0, pool_size)
        obs_bucket.append(tuple(bucket_pool[pool_to_draw].iloc[data_to_draw].tolist()))

    int_bucket = []
    for int_letter in intervention_alphabet:
        pool_size = bucket_pool[int_letter].shape[0]
        for i in range(0, bucket_size):
            data_to_draw = np.random.randint(0, pool_size)
            int_bucket.append(tuple(bucket_pool[int_letter].iloc[data_to_draw].tolist()))

    observation_variables = [s.lower() for s in observation_variables]
    obs_pdf = pd.DataFrame(obs_bucket, columns=observation_variables)
    int_pdf = pd.DataFrame(int_bucket, columns=observation_variables)
    return obs_pdf, int_pdf


def split_with_equal_variance(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nb_data_points = data.shape[0]
    nb_data_points_per_split = int(data.shape[0] / 2)
    obs_list = []
    draw_pool = list(np.arange(0, nb_data_points))
    for i in range(0, nb_data_points_per_split):
        data_to_draw = np.random.randint(0, len(draw_pool))
        obs_list.append(tuple(data.iloc[draw_pool[data_to_draw]].tolist()))
        draw_pool.pop(data_to_draw)
    int_list = []
    draw_pool = list(np.arange(0, nb_data_points))
    for i in range(0, nb_data_points_per_split):
        data_to_draw = np.random.randint(0, len(draw_pool))
        int_list.append(tuple(data.iloc[draw_pool[data_to_draw]].tolist()))
        draw_pool.pop(data_to_draw)

    return pd.DataFrame(obs_list, columns=data.columns), pd.DataFrame(int_list, columns=data.columns)


def split_bucket(data: pd.DataFrame, intervention_column: str, observation_variables: List[str]) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    intervention_alphabet = data[intervention_column].unique()
    obs_list = []
    int_list = []
    for int_letter in intervention_alphabet:
        bucket_pool = data[data[intervention_column] == int_letter][observation_variables].reset_index(drop=True)
        obs_pool, int_pool = split_with_equal_variance(bucket_pool)
        obs_list.append(obs_pool)
        int_list.append(int_pool)

    obs_pdf = pd.concat(obs_list, axis=0)
    int_pdf = pd.concat(int_list, axis=0)
    obs_pdf.columns = ['x', 'y']
    int_pdf.columns = ['x', 'y']
    return obs_pdf, int_pdf


def split_data(data, col_to_prepare, observation_variables: List[str], sort_data=True) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
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

    obs_pdf = pd.DataFrame({col.lower(): prep_data[col][0:i_max] for col in observation_variables})
    int_pdf = pd.DataFrame({col.lower(): prep_data[col][i_max:] for col in observation_variables})
    return obs_pdf, int_pdf, i_max


def split_data_at_index(data: pd.DataFrame, idx: int, observation_variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    obs_pdf = pd.DataFrame({col.lower(): data[col][0:idx] for col in observation_variables})
    int_pdf = pd.DataFrame({col.lower(): data[col][idx:2*idx] for col in observation_variables})
    return obs_pdf, int_pdf


def find_best_cluster(data: pd.DataFrame, intervention_column: str, observation_variables: List[str], max_nb_clusters: int) \
        -> Tuple[Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame], int]:
    if kl_divergence_x_y(data, -1) == np.inf:
        nb_bins_for_distr = 10
    else:
        nb_bins_for_distr = -1
    min_kl_div = np.inf
    best_nb_clusters = 2
    for nb_clusters in range(2, min(max_nb_clusters, 40)):
        (obs_pdf, int_pdf), clustered_data = cluster_data(data, intervention_column, observation_variables, nb_clusters)
        if intervention_column == 'X' or intervention_column == '':
            obs_ = pd.concat([obs_pdf['x'], obs_pdf['y']], axis=1)
            int_ = pd.concat([int_pdf['x'], int_pdf['y']], axis=1)
        elif intervention_column == 'Y':
            obs_ = pd.concat([obs_pdf['y'], obs_pdf['x']], axis=1)
            int_ = pd.concat([int_pdf['y'], int_pdf['x']], axis=1)
        elif intervention_column == 'Z':
            obs_ = pd.concat([obs_pdf['z'], obs_pdf['x'], obs_pdf['y']], axis=1)
            int_ = pd.concat([int_pdf['z'], int_pdf['x'], int_pdf['y']], axis=1)
        kl_div = kl_divergence_x_y(obs_, nb_bins_for_distr) + kl_divergence_x_y(int_, nb_bins_for_distr)
        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_nb_clusters = nb_clusters

    return cluster_data(data, intervention_column, observation_variables, best_nb_clusters), best_nb_clusters


def find_best_discretization(data: pd.DataFrame, observation_variables: List[str]) -> pd.DataFrame:
    min_kl_div = np.inf
    max_nb_bins = max(len(set(data['X'].tolist())), len(set(data['Y'].tolist())))
    best_nb_bins = 2
    for nb_bins in range(2, max_nb_bins * 10):
        disc_data = discretize_data(data, nb_bins, observation_variables)
        kl_div = kl_divergence_x_y(disc_data, -1)
        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_nb_bins = nb_bins

    return discretize_data(data, best_nb_bins, observation_variables)
