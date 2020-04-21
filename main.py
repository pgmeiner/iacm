import os
import numpy as np
import pandas as pd
from iacm import calcError, testModelFromXtoY, iacm, preprocessing, calc_variations
from igci import igci
from data_preparation import read_data, getContingencyTables
from plot import plot_distributions
from data_generation import generate_nonlinear_data, generate_nonlinear_discrete_data, generate_nonlinear_confounded_data, generate_linear_confounded_data, generate_linear_data, generate_linear_discrete_data
from hypergraph_projection import testIndpendentendModel
from sklearn.preprocessing import StandardScaler, RobustScaler


def run_trials():
    correct = 0
    not_correct = 0
    do_not_know = 0
    ind = 0
    number_trials = 100
    max_samples = 100

    for k in range(number_trials):
        print("trial " + str(k))
        obsX, obsY, intX, intY = generate_nonlinear_data(max_samples)
        modelXtoY = testModelFromXtoY(obsX, obsY, intX, intY, True, "green")
        modelYtoX = testModelFromXtoY(obsY, obsX, intY, intX, False, "yellow")
        indX_Y = testIndpendentendModel(obsX, obsY, intX, intY, "red")
        indY_X = testIndpendentendModel(obsY, obsX, intY, intX, "")
        errorXtoY = calcError(modelXtoY)
        print("total Error X -> Y: " + str(errorXtoY))
        print("local Error X -> Y: " + str(modelXtoY['LocalErrorB']))
        errorYtoX = calcError(modelYtoX)
        print("total Error Y -> X: " + str(errorYtoX))
        print("local Error X -> Y: " + str(modelYtoX['LocalErrorB']))
        print("error X, Y: " + str(indX_Y))
        print("error X, Y: " + str(indY_X))

        if indX_Y is None:
            indX_Y = 100000000.0

        min_model_error = min(errorXtoY, errorYtoX)
        if min_model_error < indX_Y:
            if errorXtoY < errorYtoX:
                print("X -> Y")
                correct = correct + 1
            elif errorXtoY > errorYtoX:
                print("Y -> X")
                not_correct = not_correct + 1
            else:
                print("no decision")
                do_not_know = do_not_know + 1
        else:
            print("independent")
            ind = ind + 1

        # if errorXtoY != None and errorYtoX != None:
        #     error_diff = abs(errorXtoY - errorYtoX)
        #     if error_diff < 1.0:
        #         if indX_Y < min(errorXtoY, errorYtoX):
        #             print("independent")
        #             ind = ind + 1
        #         else:
        #             print("no decision")
        #             do_not_know = do_not_know + 1
        #     else:
        #         if errorXtoY < errorYtoX:
        #             print("X -> Y")
        #             correct = correct + 1
        #         elif errorXtoY > errorYtoX:
        #             print("Y -> X")
        #             not_correct = not_correct + 1
        #         else:
        #             print("no decision")
        #             do_not_know = do_not_know + 1
        # else:
        #     print("no decision")
        #     do_not_know = do_not_know + 1

    print("X->Y: " + str(correct) + "; " + str(correct / number_trials * 100.0) + " %")
    print("Y->X: " + str(not_correct) + "; " + str(not_correct / number_trials * 100.0) + " %")
    if not_correct > 0:
        print("ratio: " + str(correct / float(not_correct)))
    print("independent: " + str(ind) + "; " + str(ind / number_trials * 100.0) + " %")
    print("do not know: " + str(do_not_know) + "; " + str(do_not_know / number_trials * 100.0) + " %")

    plot_distributions()


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


params = {2: {'bins': 50,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 4,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3},
          3: {'bins': 11,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 3,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3
              },
          4: {'bins': 3,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 3,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3
              }
          }


def get_ground_truth(content):
    if "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content:
        return "X->Y"
    elif "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content:
        return "Y->X"


def get_stat_entry(data):
    total_number = data['correct'] + data['not_correct'] + data['no_decision']
    return str(round(data['correct'] / total_number * 100, 2)) + " (" + str(data['correct']) + "/" + str(total_number) + ")"

def print_for_evaluation(statistics, size_alphabet, params, base):
    print(str(size_alphabet) + ";" +
          get_stat_entry(statistics['igci']) + ";" +
          get_stat_entry(statistics['iacm_none']) + ";" +
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


def run_inference(simulated_data, structure, size_alphabet, base):
    statistics = {'igci': dict(), 'iacm_none': dict(), 'iacm_discrete_split': dict(), 'iacm_split_discrete': dict(),
                  'iacm_cluster_discrete': dict(),
                  'iacm_discrete_cluster': dict(), 'iacm_alternativ': dict(), 'iacm_theoretic_coverage': dict(),
                  'iacm_new_strategy': dict()}
    for key, value in statistics.items():
        statistics[key] = {'correct': 0, 'not_correct': 0, 'no_decision': 0, 'not_correct_examples': [],
                           'correct_examples': []}

    not_touched_files = []
    total = 0
    verbose = False
    if simulated_data == False:
        directory = "./pairs"
    else:
        directory = f'./simulations/{structure}/{size_alphabet}'
    for file in os.listdir(directory):
        if "_des" not in file:
            try:
                some_method_succeeded = False
                #file = "pair0057.txt"
                data = read_data(directory, file)
                if simulated_data:
                    ground_truth = "X->Y"
                else:
                    content = open("./pairs/" + file.replace(".txt", "_des.txt"), "r").read().lower()
                    ground_truth = get_ground_truth(content)
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

                for preprocess_method in ['none', 'discrete_split', 'split_discrete', 'discrete_cluster', 'cluster_discrete',
                                          'alternativ', 'new_strategy']:
                    params[base]['preprocess_method'] = preprocess_method
                    if verbose: print(preprocess_method)
                    res = iacm(base=base, data=data, params=params[base], verbose=verbose)
                    if ground_truth == res:
                        statistics['iacm_' + preprocess_method]['correct'] = statistics['iacm_' + preprocess_method][
                                                                                 'correct'] + 1
                        statistics['iacm_' + preprocess_method]['correct_examples'].append(file)
                        if not some_method_succeeded:
                            statistics['iacm_theoretic_coverage']['correct'] = statistics['iacm_theoretic_coverage'][
                                                                                   'correct'] + 1
                            statistics['iacm_theoretic_coverage']['correct_examples'].append(file)
                            some_method_succeeded = True
                        total_method = statistics['iacm_' + preprocess_method]['correct'] + statistics['iacm_' + preprocess_method]['not_correct'] + statistics['iacm_' + preprocess_method]['no_decision']
                        print("correct: " + str(statistics['iacm_' + preprocess_method]['correct'] / total_method))
                    elif "no decision" in res:
                        statistics['iacm_' + preprocess_method]['no_decision'] = \
                        statistics['iacm_' + preprocess_method]['no_decision'] + 1
                        statistics['iacm_theoretic_coverage']['not_correct_examples'].append(file)
                        print("no decision")
                    else:
                        statistics['iacm_' + preprocess_method]['not_correct'] = \
                        statistics['iacm_' + preprocess_method]['not_correct'] + 1
                        statistics['iacm_' + preprocess_method]['not_correct_examples'].append(file)
                        print("not correct")
                if not some_method_succeeded:
                    statistics['iacm_theoretic_coverage']['not_correct_examples'].append(file)
                    statistics['iacm_theoretic_coverage']['not_correct'] = statistics['iacm_theoretic_coverage'][
                                                                               'not_correct'] + 1

                total = total + 1
                if verbose: print(str(statistics['iacm_cluster']['correct']) + 'from' + str(total))
                print(total)

            except Exception as e:
                not_touched_files.append(file)

    print_statisticts(statistics)
    print("not touched files")
    print(not_touched_files)

    # print out for evaluation
    print_for_evaluation(statistics, size_alphabet, params[base], base)


if __name__ == '__main__':
    structure = 'nonlinear_discrete'
    nr_simulations = 100
    max_samples = 100
    size_alphabet = 3
    #run_simulations(structure=structure, max_samples=max_samples, size_alphabet=size_alphabet, nr_simulations=nr_simulations)
    run_inference(simulated_data=False, structure=structure, size_alphabet=size_alphabet, base=4)
