import os
import numpy as np
import pandas as pd
from iacm.iacm import iacm_discovery, iacm_discovery_timeseries
from code_extern.cisc_master.cisc import cisc
import warnings
#from code_extern.cisc_master.dr import dr
from code_extern.cisc_master.entropic import entropic
from iacm.data_preparation import read_data, read_synthetic_data
import cdt
from iacm.data_generation import generate_discrete_data, generate_continuous_data
from sklearn.preprocessing import RobustScaler


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


params = {'bins': 2,
          'nb_cluster': 2,
          'decision_criteria': 'global_error',
          'preprocess_method': 'discrete_cluster'}


def get_ground_truth(content):
    if "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content:
        return "X->Y"
    elif "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content:
        return "Y->X"


def get_stat_entry(data):
    if 'correct0' in data:
        total_number0 = data['correct0'] + data['not_correct0'] + data['no_decision0']
        if total_number0 > 0:
            return str(round(data['correct0'] / total_number0 * 100, 2)) + " (" + str(
                round(data['correct0'], 2)) + "," + str(round(data['not_correct0'], 2)) + "," + str(
                round(data['no_decision0'], 2)) + "/" + str(round(total_number0, 2)) + ") "
        else:
            total_number = data['correct'] + data['not_correct'] + data['no_decision']
            if total_number > 0:
                # return str(round(data['correct'] / total_number * 100, 2)) + ", " + str(round(data['avg_error']/data['total_nb'], 4)) + " (" + str(round(data['correct'],2)) + "," + str(round(data['not_correct'],2)) + "," + str(round(data['no_decision'],2)) + "/" + str(round(total_number,2)) + ")"
                return str(round(data['correct'] / total_number * 100, 2)) + " (" + str(
                    round(data['correct'], 2)) + "," + str(round(data['not_correct'], 2)) + "," + str(
                    round(data['no_decision'], 2)) + "/" + str(round(total_number, 2)) + ")"
            else:
                return ""
    else:
        total_number = data['correct'] + data['not_correct'] + data['no_decision']
        if total_number > 0:
            #return str(round(data['correct'] / total_number * 100, 2)) + ", " + str(round(data['avg_error']/data['total_nb'], 4)) + " (" + str(round(data['correct'],2)) + "," + str(round(data['not_correct'],2)) + "," + str(round(data['no_decision'],2)) + "/" + str(round(total_number,2)) + ")"
            return str(round(data['correct'] / total_number * 100, 2)) + " (" + str(round(data['correct'], 2)) + "," + str(round(data['not_correct'], 2)) + "," + str(round(data['no_decision'], 2)) + "/" + str(round(total_number, 2)) + ")"
        else:
            return ""


def print_for_evaluation(statistics, alphabet_size_x, alphabet_size_y, params, base):
    print(str(alphabet_size_x) + "_" + str(alphabet_size_y) + ";" +
          str(params['bins']) + ";" + str(params['nb_cluster']) + ";" +
          str(base) + ";" +
            ";".join([get_stat_entry(value) for method, value in statistics.items()]))


def run_simulations(structure, sample_sizes, alphabet_size_x, alphabet_size_y, nr_simulations):
    for i in range(0, nr_simulations):
        max_samples = sample_sizes[np.random.randint(3)]
        if 'discrete' in structure:
            data = generate_discrete_data(structure=structure, sample_size=max_samples, alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y)
        else:
            data = generate_continuous_data(structure=structure, sample_size=max_samples)

        filename = "pair" + str(i) + ".csv"
        if not os.path.exists(f'simulations/add_mult/{structure}/{alphabet_size_x}_{alphabet_size_y}'):
            os.makedirs(f'simulations/add_mult/{structure}/{alphabet_size_x}_{alphabet_size_y}')
        data.to_csv(f'simulations/add_mult/{structure}/{alphabet_size_x}_{alphabet_size_y}/{filename}', sep=" ", header=False, index=False)


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


def get_result(method, file, data, statistics, preprocessing_stat, base_x, base_y, params, verbose):
    res = "no decision"
    if method == 'ANM':
        pairwise_cd_algo = cdt.causality.pairwise.ANM()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
            if cdt_res > 0:
                res = "X->Y"
            elif cdt_res < 0:
                res = "Y->X"
    elif method == 'BivariateFit':
        pairwise_cd_algo = cdt.causality.pairwise.BivariateFit()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
        if cdt_res > 0:
            res = "X->Y"
        elif cdt_res < 0:
            res = "Y->X"
    elif method == 'IGCI':
        pairwise_cd_algo = cdt.causality.pairwise.IGCI()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
        if cdt_res > 0:
            res = "X->Y"
        elif cdt_res < 0:
            res = "Y->X"
    elif method == 'CDS':
        pairwise_cd_algo = cdt.causality.pairwise.CDS()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
        if cdt_res > 0:
            res = "X->Y"
        elif cdt_res < 0:
            res = "Y->X"
    elif method == 'GNN':
        pairwise_cd_algo = cdt.causality.pairwise.GNN()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
        if cdt_res > 0:
            res = "X->Y"
        elif cdt_res < 0:
            res = "Y->X"
    elif method == 'RECI':
        pairwise_cd_algo = cdt.causality.pairwise.RECI()
        cdt_data = pd.DataFrame({'X': [np.array(data['X'])], 'Y': [np.array(data['Y'])]})
        cdt_res = pairwise_cd_algo.predict(cdt_data)[0]
        if cdt_res > 0:
            res = "X->Y"
        elif cdt_res < 0:
            res = "Y->X"
    elif method == 'CISC':
        cisc_score = cisc(data['X'], data['Y'])
        if cisc_score[0] < cisc_score[1]:
            res = "X->Y"
        elif cisc_score[0] > cisc_score[1]:
            res = "Y->X"
    elif method == "DR":
        level = 0.05
        dr_score = 0#dr(data['X'].tolist(), data['Y'].tolist(), level)
        if dr_score[0] > level and dr_score[1] < level:
            res = "X->Y"
        elif dr_score[0] < level and dr_score[1] > level:
            res = "Y->X"
    elif method == "ACID":
        ent_score = entropic(pd.DataFrame(np.column_stack((data['X'], data['Y']))))
        if ent_score[0] < ent_score[1]:
            res = "X->Y"
        elif ent_score[0] > ent_score[1]:
            res = "Y->X"
    elif 'IACM' in method:
        if file not in preprocessing_stat.keys():
            preprocessing_stat[file] = dict()
        preprocess_method = method.split('-')[1]
        if preprocess_method == "":
            preprocess_method = 'auto'

        params['preprocess_method'] = preprocess_method
        if verbose: print(preprocess_method)
        preprocessing_stat[file][preprocess_method] = dict()

        if file in timeseries_files:
            res, crit = iacm_discovery_timeseries(base_x=base_x, base_y=base_y, data=data, auto_configuration=True, parameters=params, max_lag=50, verbose=verbose)
        else:
            res, crit = iacm_discovery(base_x=base_x, base_y=base_y, data=data, auto_configuration=True, parameters=params, verbose=verbose, preserve_order=False)

        # plot_distributions()
        #statistics[method]['avg_error'] = statistics[method]['avg_error'] + crit

    return res


timeseries_files = []#['histamin_blaehungen.csv', 'histamin_durchfall.csv', 'gluten_blaehungen.csv','sorbit_blaehungen.csv'] #['pair0042.txt', 'pair0068.txt', 'pair0069.txt', 'pair0077.txt', 'pair0094.txt', 'pair0095.txt']
#method_list = ['IACM_auto', 'IGCI', 'ANM', 'BivariateFit', 'CDS', 'RECI']


def run_inference(method_list, data_set, structure, alphabet_size_x, alphabet_size_y, base_x, base_y, params):
    statistics = {'igci': dict(), 'iacm_none': dict(), 'iacm_auto': dict(), 'iacm_discrete_split': dict(), 'iacm_split_discrete': dict(),
                  'iacm_cluster_discrete': dict(), 'iacm_split_strategy': dict(),
                  'iacm_discrete_cluster': dict(), 'iacm_alternativ': dict(), 'iacm_theoretic_coverage': dict(),
                  'iacm_new_strategy': dict()}
    statistics = dict()
    for method in method_list:
        statistics[method] = dict()
    preprocessing_stat = dict()
    for key, value in statistics.items():
        statistics[key] = {'correct': 0, 'not_correct': 0, 'no_decision': 0, 'not_correct_examples': [],
                           'correct_examples': [], 'avg_error': 0.0, 'total_nb': 1, 'correct0': 0, 'not_correct0': 0, 'no_decision0': 0,
                           'correct1': 0, 'not_correct1': 0, 'no_decision1': 0}

    not_touched_files = []
    total = 0
    verbose = False
    weights = []
    if data_set == 'CEP':
        directory = "./pairs"
        weights = open(directory + "/pairmeta.txt", "r").readlines()
    elif 'SIM' in data_set:
        directory = "./simulations_extern/CauseEffectBenchmark/" + data_set
        ground_truth_lines = open(directory + "/pairmeta.txt","r").readlines()
    elif 'Extern' in data_set:
        directory = "./simulations/extern/"
    elif 'Bridge' in data_set:
        directory = "./real_world_data/bridge_data/"
    elif 'Food' in data_set:
        directory = "./real_world_data/food_intolerances/"
    elif 'Abalone' in data_set:
        directory = "./real_world_data/abalone/"
    elif 'CEChallenge2013' in data_set:
        directory = "./CEdata/CEfinal/"
    else:
        directory = f'./simulations/add_mult/{structure}/{alphabet_size_x}_{alphabet_size_y}'
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if "_des" not in file and "pairmeta" not in file and "README" not in file:
                try:
                    targets = []
                    data = []
                    some_method_succeeded = False
                    #file = "pair4.csv"
                    if 'SIM' in data_set:
                        weight = 1.0
                        data = [read_synthetic_data(directory, file)]
                        for line in ground_truth_lines:
                            if file[4:8] in line:
                                if "1 1 2 2 1" in line:
                                    targets = ["X->Y"]
                                    break
                                elif "2 2 1 1 1" in line:
                                    targets = ["Y->X"]
                                    break
                    elif data_set == "CEP":
                        data = [read_data(directory, file)]
                        content = open("./pairs/" + file.replace(".txt", "_des.txt"), "r").read().lower()
                        ground_truth = get_ground_truth(content)
                        targets = [ground_truth]
                        weight = 1.0
                        for line in weights:
                            if file[4:8] in line:
                                weight = float(line.split()[5])
                                break
                    elif data_set == "Bridge":
                        data = [pd.read_csv(directory + file, sep=";", header=None)]
                        targets = ["X->Y"]
                        weight = 1.0
                    elif data_set == "Food":
                        data = [pd.read_csv(directory + file, sep=";", header=None)]
                        targets = ["X->Y"]
                        weight = 1.0
                    elif data_set == "Abalone":
                        data = [pd.read_csv(directory + file, sep=";", header=None)]
                        targets = ["X->Y"]
                        weight = 1.0
                    elif data_set == "CEChallenge2013":
                        if 'pairs.csv' in file:
                            data_pdf = pd.read_csv(directory + file, sep=",", header=None)
                            targets_pdf = pd.read_csv(directory + file.replace("_pairs", "_target"), sep=",", header=None)
                            for index, row in data_pdf.iterrows():
                                if index > 0:
                                    data.append(pd.DataFrame(data={'X': row[1].lstrip().split(" "), 'Y': row[2].lstrip().split(" ")}))
                            for index, row in targets_pdf.iterrows():
                                if index > 0:
                                    if int(row[1]) == 1:
                                        targets.append("X->Y")
                                    elif int(row[1]) == -1:
                                        targets.append("Y->X")
                                    else:
                                        targets.append("")
                        weight = 1.0
                    elif data_set == "Extern":
                        data = [pd.read_csv(directory + file, sep=";", header=None)]
                        targets = ["X->Y"]
                        weight = 1.0
                    else:
                        data = [read_data(directory, file)]
                        targets = ["X->Y"]
                        weight = 1.0

                    for truth, data_pdf in zip(targets, data):
                        if truth == "":
                            continue
                        data_pdf = pd.DataFrame(RobustScaler().fit(data_pdf).transform(data_pdf))
                        data_pdf.columns = ['X', 'Y']
                        for method in method_list:
                            res = get_result(method, file, data_pdf, statistics, preprocessing_stat, base_x, base_y, params, verbose)

                            if 'IACM' in method:
                                if truth == res:
                                    statistics[method]['correct0'] = statistics[method]['correct0'] + weight
                                    statistics[method]['correct_examples'].append(file)
                                elif "no decision" in res:
                                    statistics[method]['no_decision0'] = statistics[method]['no_decision0'] + weight
                                else:
                                    statistics[method]['not_correct0'] = statistics[method]['not_correct0'] + weight
                                    statistics[method]['not_correct_examples'].append(file)
                            else:
                                if truth == res:
                                    statistics[method]['correct'] = statistics[method]['correct'] + weight
                                    statistics[method]['correct_examples'].append(file)
                                    if 'IACM' in method:
                                        preprocessing_stat[file][method.split('-')[1]]['correct'] = weight
                                    total_method = statistics[method]['correct'] + statistics[method]['not_correct'] + statistics[method]['no_decision']
                                    if verbose: print("correct: " + str(statistics[method]['correct'] / total_method))
                                elif "no decision" in res:
                                    statistics[method]['no_decision'] = statistics[method]['no_decision'] + weight
                                    if 'IACM' in method:
                                        preprocessing_stat[file][method.split('-')[1]]['correct'] = 0
                                    if verbose: print("no decision")
                                else:
                                    statistics[method]['not_correct'] = statistics[method]['not_correct'] + weight
                                    if 'IACM' in method:
                                        preprocessing_stat[file][method.split('-')[1]]['correct'] = 0
                                    statistics[method]['not_correct_examples'].append(file)
                                    if verbose: print("not correct")

                            statistics[method]['total_nb'] = statistics[method]['total_nb'] + 1
                            if verbose:
                                total = total + weight
                                print(total)

                except Exception as e:
                    not_touched_files.append(file)

        if verbose: print_statisticts(statistics)
        if verbose: print("not touched files")
        if verbose: print(not_touched_files)

        # print out for evaluation
        print_for_evaluation(statistics, alphabet_size_x, alphabet_size_y, params, base_x)
        #print_for_preprocess_evaluation(preprocessing_stat)


structure_list = ['linear_discrete', 'nonlinear_discrete', 'linear_continuous', 'nonlinear_continuous']

if __name__ == '__main__':
    structure = 'linear_discrete'
    nr_simulations = 100
    sample_sizes = [100, 500, 1000]
    base = 2

    # data generation
    # for size_x in range(2, 3):
    #     for size_y in range(2, 3):
    #         alphabet_size_x = size_x
    #         alphabet_size_y = size_y
    #         run_simulations(structure=structure, sample_sizes=sample_sizes, alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y, nr_simulations=nr_simulations)

    # ['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster', 'IGCI', 'ANM', 'BivariateFit', 'CDS', 'RECI', 'CISC','ACID']
    for size_x, size_y in [(3,3)]:
        alphabet_size_x = size_x
        alphabet_size_y = size_y
        params['bins'] = 2
        params['nb_cluster'] = 2
        methods = ['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster', 'IGCI', 'ANM', 'BivariateFit', 'CDS', 'RECI', 'CISC','ACID']
        #run_inference(method_list=methods, data_set='Abalone', structure=structure, alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y, base_x=base, base_y=base, params=params)
        methods = ['IACM-discrete_cluster']#['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster', 'IGCI', 'ANM', 'BivariateFit', 'CDS', 'RECI', 'CISC','ACID']#['IACM-cluster_discrete', 'IACM-discrete_cluster']#['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster', 'IGCI', 'ANM', 'BivariateFit', 'CDS', 'RECI', 'CISC','ACID']#['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster'] #['IACM-cluster_discrete', 'IACM-discrete_cluster']#['IACM-cluster_discrete', 'IACM-discrete_cluster'] #['IACM-none', 'IACM-split_discrete', 'IACM-discrete_split', 'IACM-cluster_discrete', 'IACM-discrete_cluster'] #['IACM-cluster_discrete', 'IACM-discrete_cluster']
        print("alphabet;bins;cluster;base;" + ";".join(methods))
        for bins in range(11, 31):
            for clt in range(bins, (bins + 1)):
                params['bins'] = bins
                params['nb_cluster'] = -1
                run_inference(method_list=methods, data_set='CEP', structure=structure, alphabet_size_x=alphabet_size_x, alphabet_size_y=alphabet_size_y, base_x=base, base_y=base, params=params)
