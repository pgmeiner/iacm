import os
import numpy as np
import pandas as pd
from iacm import calcError, testModelFromXtoY, iacm, preprocessing, calc_variations
from igci import igci
from data_preparation import read_data, getContingencyTables
from plot import plot_distributions
from data_generation import generate_nonlinear_data, generate_nonlinear_discrete_data, generate_nonlinear_confounded_data, generate_linear_confounded_data, generate_linear_data
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


params = {2: {'bins': 4,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 4,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3},
          3: {'bins': 4,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 4,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3
              },
          4: {'bins': 5,
              'x_shift': 0,
              'y_shift': 0,
              'nb_cluster': 5,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3
              }
          }


def get_ground_truth(content):
    if "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content:
        return "X->Y"
    elif "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content:
        return "Y->X"


if __name__ == '__main__':
    statistics = {'igci': dict(), 'iacm_discrete_split': dict(), 'iacm_split': dict(), 'iacm_cluster': dict(),
                  'iacm_discrete_cluster': dict(), 'iacm_alternativ': dict(), 'iacm_theoretic_coverage': dict(), 'iacm_new_strategy': dict()}
    for key, value in statistics.items():
        statistics[key] = {'correct':0, 'not_correct':0, 'no_decision': 0, 'not_correct_examples': [], 'correct_examples': []}

    iacm_discrete_split = ['pair0057.txt', 'pair0043.txt', 'pair0056.txt', 'pair0083.txt', 'pair0097.txt', 'pair0108.txt', 'pair0069.txt',
     'pair0082.txt', 'pair0045.txt', 'pair0078.txt', 'pair0091.txt', 'pair0084.txt', 'pair0009.txt', 'pair0021.txt',
     'pair0036.txt', 'pair0022.txt', 'pair0019.txt', 'pair0031.txt', 'pair0025.txt', 'pair0024.txt', 'pair0018.txt',
     'pair0015.txt', 'pair0029.txt', 'pair0014.txt', 'pair0013.txt', 'pair0006.txt', 'pair0038.txt', 'pair0010.txt',
     'pair0089.txt', 'pair0076.txt', 'pair0062.txt', 'pair0102.txt', 'pair0088.txt', 'pair0060.txt', 'pair0059.txt',
     'pair0072.txt']
    iacm_split = ['pair0094.txt', 'pair0043.txt', 'pair0056.txt', 'pair0083.txt', 'pair0097.txt', 'pair0040.txt', 'pair0108.txt',
     'pair0092.txt', 'pair0079.txt', 'pair0045.txt', 'pair0051.txt', 'pair0050.txt', 'pair0091.txt', 'pair0046.txt',
     'pair0090.txt', 'pair0009.txt', 'pair0021.txt', 'pair0036.txt', 'pair0026.txt', 'pair0027.txt', 'pair0019.txt',
     'pair0031.txt', 'pair0030.txt', 'pair0015.txt', 'pair0029.txt', 'pair0007.txt', 'pair0038.txt', 'pair0005.txt',
     'pair0089.txt', 'pair0062.txt', 'pair0061.txt', 'pair0101.txt', 'pair0070.txt', 'pair0059.txt', 'pair0098.txt']
    iacm_cluster = ['pair0094.txt', 'pair0057.txt', 'pair0042.txt', 'pair0083.txt', 'pair0097.txt', 'pair0040.txt', 'pair0108.txt',
     'pair0041.txt', 'pair0045.txt', 'pair0051.txt', 'pair0050.txt', 'pair0078.txt', 'pair0047.txt', 'pair0084.txt',
     'pair0090.txt', 'pair0009.txt', 'pair0036.txt', 'pair0026.txt', 'pair0027.txt', 'pair0030.txt', 'pair0029.txt',
     'pair0028.txt', 'pair0014.txt', 'pair0006.txt', 'pair0089.txt', 'pair0062.txt', 'pair0102.txt', 'pair0103.txt',
     'pair0063.txt', 'pair0088.txt', 'pair0061.txt', 'pair0058.txt']
    iacm_discrete_cluster = ['pair0057.txt', 'pair0042.txt', 'pair0056.txt', 'pair0081.txt', 'pair0069.txt', 'pair0082.txt', 'pair0079.txt',
     'pair0045.txt', 'pair0051.txt', 'pair0050.txt', 'pair0091.txt', 'pair0047.txt', 'pair0084.txt', 'pair0036.txt',
     'pair0026.txt', 'pair0027.txt', 'pair0031.txt', 'pair0018.txt', 'pair0015.txt', 'pair0029.txt', 'pair0028.txt',
     'pair0016.txt', 'pair0006.txt', 'pair0010.txt', 'pair0089.txt', 'pair0076.txt', 'pair0103.txt', 'pair0049.txt',
     'pair0101.txt', 'pair0048.txt', 'pair0060.txt', 'pair0058.txt', 'pair0059.txt']
    all = set(iacm_discrete_split).intersection(set(iacm_split)).intersection(set(iacm_cluster)).intersection(set(iacm_discrete_cluster))

    max_samples = 100
    simulated_data = False
    base = 2
    not_touched_files = []
    total = 0
    verbose = False
    for file in os.listdir("./pairs"):
        if "_des" not in file:
            try:
                if simulated_data:
                    obsX, obsY, intX, intY = generate_nonlinear_discrete_data(max_samples, 7)
                    data = pd.DataFrame({'X': np.concatenate([obsX, intX]), 'Y': np.concatenate([obsY, intY])})
                    ground_truth = "X->Y"
                else:
                    #file = "pair0095.txt"
                    data = read_data(file)
                    content = open("./pairs/" + file.replace(".txt", "_des.txt"), "r").read().lower()
                    ground_truth = get_ground_truth(content)
                data = pd.DataFrame(RobustScaler().fit(data).transform(data))
                data.columns = ['X', 'Y']
                ig = igci(data['X'], data['Y'], refMeasure=1, estimator=2)
                if (ground_truth in 'X->Y' and ig < 0) or (ground_truth in 'Y->X' and ig > 0):
                    statistics['igci']['correct'] = statistics['igci']['correct'] + 1
                else:
                    statistics['igci']['not_correct'] = statistics['igci']['not_correct'] + 1
                    statistics['igci']['not_correct_examples'].append(file)

                for preprocess_method in ['discrete_split', 'split_discrete', 'discrete_cluster', 'cluster_discrete', 'alternativ' ,'new_strategy']:
                    params[base]['preprocess_method'] = preprocess_method
                    if verbose: print(preprocess_method)
                    res = iacm(base=base, data=data, params=params[base], verbose=verbose)
                    if ground_truth == res:
                        statistics['iacm_'+preprocess_method]['correct'] = statistics['iacm_'+preprocess_method]['correct'] + 1
                        statistics['iacm_'+preprocess_method]['correct_examples'].append(file)
                        statistics['iacm_theoretic_coverage']['correct_examples'].append(file)
                        print("correct")
                    elif "no decision" in res:
                        statistics['iacm_'+preprocess_method]['no_decision'] = statistics['iacm_'+preprocess_method]['no_decision'] + 1
                        statistics['iacm_theoretic_coverage']['not_correct_examples'].append(file)
                        print("no decision")
                    else:
                        statistics['iacm_'+preprocess_method]['not_correct'] = statistics['iacm_'+preprocess_method]['not_correct'] + 1
                        statistics['iacm_'+preprocess_method]['not_correct_examples'].append(file)
                        print("not correct")

                total = total + 1
                if verbose: print(str(statistics['iacm_cluster']['correct']) + 'from' + str(total))
                print(total)

            except Exception as e:
                not_touched_files.append(file)

    print_statisticts(statistics)
    print("not touched files")
    print(not_touched_files)
