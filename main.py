import os
import numpy as np
import pandas as pd
from iacm import iacm, iacm_timeseries, iacm_auto
from igci import igci
from data_preparation import read_data, read_synthetic_data
from scipy.stats import spearmanr, kendalltau
from plot import plot_distributions
from data_generation import generate_nonlinear_data, generate_nonlinear_discrete_data, generate_nonlinear_confounded_data, generate_linear_confounded_data, generate_linear_data, generate_linear_discrete_data
from sklearn.preprocessing import StandardScaler, RobustScaler


def print_statisticts(statistics):
    for key, value in statistics.items():
        print(key)
        total_number = value['correct'] + value['not_correct'] + value['no_decision']
        if total_number > 0:
            print("correct: " + str(value['correct']) + " (" + str(round(value['correct'] / total_number * 100, 2)) + " %)")
            print("not correct: " + str(value['not_correct']) + " (" + str(round(value['not_correct'] / total_number * 100, 2)) + " %)")
            print("no decision: " + str(value['no_decision']) + " (" + str(round(value['no_decision'] / total_number * 100, 2)) + " %)")
            print("#examples: " + str(total_number))
        print(len(set(value['correct_examples'])))
        print(value['correct_examples'])
        print(len(set(value['not_correct_examples'])))
        print(value['not_correct_examples'])


params = {2: {'bins': 2,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 2,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3,
              'monotone': True},
          3: {'bins': 2,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 3,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3,
              'monotone': True
              },
          4: {'bins': 9,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 2,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3,
              'monotone': False
              },
          5: {'bins': 2,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 2,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3,
              'monotone': False
              }
          }


def get_ground_truth(content):
    if "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content:
        return "X->Y"
    elif "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content:
        return "Y->X"


def get_stat_entry(data):
    total_number = data['correct'] + data['not_correct'] + data['no_decision']
    if total_number > 0:
        return str(round(data['correct'] / total_number * 100, 2)) + " (" + str(data['correct']) + "," + str(data['not_correct']) + "," + str(data['no_decision']) + "/" + str(total_number) + ")"
    else:
        return ""

def print_for_evaluation(statistics, size_alphabet, params, base):
    print(str(size_alphabet) + ";" +
          get_stat_entry(statistics['igci']) + ";" +
          get_stat_entry(statistics['iacm_auto']) + ";" +
          get_stat_entry(statistics['iacm_split_discrete']) + ";" +
          get_stat_entry(statistics['iacm_discrete_split']) + ";" +
          get_stat_entry(statistics['iacm_discrete_cluster']) + ";" +
          get_stat_entry(statistics['iacm_cluster_discrete']) + ";" +
          get_stat_entry(statistics['iacm_new_strategy']) + ";" +
          get_stat_entry(statistics['iacm_theoretic_coverage']) + ";" +
          str(params['bins']) + ";" + str(params['nb_cluster']) + ";" +
          str(base))


def run_simulations(structure, max_samples, size_alphabet, nr_simulations):
    for i in range(0, nr_simulations):
        if structure == 'nonlinear_discrete':
            obsX, obsY, intX, intY = generate_nonlinear_discrete_data(max_samples, size_alphabet)
        elif structure == 'linear_discrete':
            obsX, obsY, intX, intY = generate_linear_discrete_data(max_samples, size_alphabet)
        else:
            continue

        data = pd.DataFrame({'X': np.concatenate([obsX, intX]), 'Y': np.concatenate([obsY, intY])})
        filename = "pair" + str(i) + ".csv"
        data.to_csv(f'simulations/{structure}/{size_alphabet}/{filename}', sep=" ", header=False, index=False)


def print_for_preprocess_evaluation(preprocessing_stat):
    print("file;method;correct;obsX_eR_sq;obsY_eR_sq;intX_eR_sq;intY_eR_sq;X_eR_sq;Y_eR_sq")
    for file, v in preprocessing_stat.items():
        res_str = str(file)
        for method, element in v.items():
            res_str = res_str + ";" + method
            for stat_k, stat_v in element.items():
                res_str = res_str + ";" + str(round(stat_v,3))
            print(res_str)
            res_str = str(file)


timeseries_files = [] #['pair0042.txt', 'pair0068.txt', 'pair0069.txt', 'pair0077.txt', 'pair0094.txt', 'pair0095.txt']


def run_inference(data_set, structure, size_alphabet, base_x, base_y, params):
    statistics = {'igci': dict(), 'iacm_none': dict(), 'iacm_auto': dict(), 'iacm_discrete_split': dict(), 'iacm_split_discrete': dict(),
                  'iacm_cluster_discrete': dict(), 'iacm_split_strategy': dict(),
                  'iacm_discrete_cluster': dict(), 'iacm_alternativ': dict(), 'iacm_theoretic_coverage': dict(),
                  'iacm_new_strategy': dict()}
    preprocessing_stat = dict()
    for key, value in statistics.items():
        statistics[key] = {'correct': 0, 'not_correct': 0, 'no_decision': 0, 'not_correct_examples': [],
                           'correct_examples': []}

    not_touched_files = []
    total = 0
    verbose = False
    if data_set == 'CEP':
        directory = "./pairs"
    elif data_set == 'Synthetic':
        directory = "./simulations_extern/CauseEffectBenchmark/SIM-c"
        ground_truth_lines = open(directory + "/pairmeta.txt","r").readlines()
    else:
        directory = f'./simulations/{structure}/{size_alphabet}'
    for file in os.listdir(directory):
        if "_des" not in file or "pairmeta" not in file:
            try:
                some_method_succeeded = False
                #file = "pair0008.txt"
                if data_set == "Synthetic":
                    data = read_synthetic_data(directory, file)
                    for line in ground_truth_lines:
                        if file[4:8] in line:
                            if "1 1 2 2 1" in line:
                                ground_truth = "X->Y"
                                break
                            elif "2 2 1 1 1" in line:
                                ground_truth = "Y->X"
                                break
                elif data_set == "CEP":
                    data = read_data(directory, file)
                    content = open("./pairs/" + file.replace(".txt", "_des.txt"), "r").read().lower()
                    ground_truth = get_ground_truth(content)
                else:
                    data = read_data(directory, file)
                    ground_truth = "X->Y"
                data = pd.DataFrame(RobustScaler().fit(data).transform(data))
                data.columns = ['X', 'Y']
                ig = igci(data['X'], data['Y'], refMeasure=1, estimator=2)
                if ig == 0:
                    ig = igci(data['X'], data['Y'], refMeasure=2, estimator=2)
                if (ground_truth in 'X->Y' and ig < 0) or (ground_truth in 'Y->X' and ig > 0):
                    statistics['igci']['correct'] = statistics['igci']['correct'] + 1
                else:
                    statistics['igci']['not_correct'] = statistics['igci']['not_correct'] + 1
                    statistics['igci']['not_correct_examples'].append(file)

                stat_s, p_s = spearmanr(data['X'], data['Y'])
                if abs(stat_s) >= 0.7 and p_s <= 0.01:
                    params[base_x]['monotone'] = True
                else:
                    params[base_x]['monotone'] = False

                preprocessing_stat[file] = dict()
                for preprocess_method in ['auto', 'split_discrete', 'discrete_split']: #, 'split_discrete', 'discrete_split', 'discrete_cluster', 'cluster_discrete', 'new_strategy']:
                    params[base_x]['preprocess_method'] = preprocess_method
                    if verbose: print(preprocess_method)
                    preprocessing_stat[file][preprocess_method] = dict()
                    #print(ground_truth)
                    if file in timeseries_files:
                        res = iacm_timeseries(base_x=base_x, base_y=base_y, data=data, params=params[base_x], max_lag=10, verbose=verbose)
                        stats = dict()
                    else:
                        res, stats, crit = iacm(base_x=base_x, base_y=base_y, data=data, params=params[base_x], verbose=verbose)

                    #print(res)
                    #plot_distributions()
                    if ground_truth == res:
                        statistics['iacm_' + preprocess_method]['correct'] = statistics['iacm_' + preprocess_method][
                                                                                 'correct'] + 1
                        statistics['iacm_' + preprocess_method]['correct_examples'].append(file)
                        preprocessing_stat[file][preprocess_method]['correct'] = 1
                        if not some_method_succeeded:
                            statistics['iacm_theoretic_coverage']['correct'] = statistics['iacm_theoretic_coverage'][
                                                                                   'correct'] + 1
                            statistics['iacm_theoretic_coverage']['correct_examples'].append(file)
                            some_method_succeeded = True
                        total_method = statistics['iacm_' + preprocess_method]['correct'] + statistics['iacm_' + preprocess_method]['not_correct'] + statistics['iacm_' + preprocess_method]['no_decision']
                        if verbose: print("correct: " + str(statistics['iacm_' + preprocess_method]['correct'] / total_method))
                    elif "no decision" in res:
                        statistics['iacm_' + preprocess_method]['no_decision'] = \
                        statistics['iacm_' + preprocess_method]['no_decision'] + 1
                        preprocessing_stat[file][preprocess_method]['correct'] = 0
                        statistics['iacm_theoretic_coverage']['not_correct_examples'].append(file)
                        if verbose: print("no decision")
                    else:
                        statistics['iacm_' + preprocess_method]['not_correct'] = \
                        statistics['iacm_' + preprocess_method]['not_correct'] + 1
                        preprocessing_stat[file][preprocess_method]['correct'] = 0
                        statistics['iacm_' + preprocess_method]['not_correct_examples'].append(file)
                        if verbose: print("not correct")
                    for k,v in stats.items():
                        preprocessing_stat[file][preprocess_method][k] = v
                if not some_method_succeeded:
                    statistics['iacm_theoretic_coverage']['not_correct_examples'].append(file)
                    statistics['iacm_theoretic_coverage']['not_correct'] = statistics['iacm_theoretic_coverage'][
                                                                               'not_correct'] + 1

                total = total + 1
                if verbose: print(str(statistics['iacm_cluster']['correct']) + 'from' + str(total))
                if verbose: print(total)

            except Exception as e:
                not_touched_files.append(file)

    if verbose: print_statisticts(statistics)
    if verbose: print("not touched files")
    if verbose: print(not_touched_files)

    # print out for evaluation
    print_for_evaluation(statistics, size_alphabet, params[base_x], base_x)
    #print_for_preprocess_evaluation(preprocessing_stat)


if __name__ == '__main__':
    structure = 'nonlinear_discrete'
    nr_simulations = 100
    max_samples = 100
    size_alphabet = 3
    base = 2
    #run_simulations(structure=structure, max_samples=max_samples, size_alphabet=size_alphabet, nr_simulations=nr_simulations)
    for bins in range(2, 25):
        for clt in range(2,3):
            params[base]['bins'] = bins
            params[base]['nb_cluster'] = clt
            run_inference(data_set='Synthetic', structure=structure, size_alphabet=size_alphabet, base_x=base, base_y=base, params=params)
