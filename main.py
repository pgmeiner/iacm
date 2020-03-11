import os
import pandas as pd
from iacm import calcError, testModelFromXtoY, iacm
from igci import igci
from data_preparation import read_data
from plot import plot_distributions
from data_generation import generate_nonlinear_data, generate_nonlinear_confounded_data, generate_linear_confounded_data, generate_linear_data
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
        print("correct: " + str(value['correct']) + " (" + str(round(value['correct'] / total_number * 100, 2)) + " %)")
        print("not correct: " + str(value['not_correct']) + " (" + str(round(value['not_correct'] / total_number * 100, 2)) + " %)")
        print("no decision: " + str(value['no_decision']) + " (" + str(round(value['no_decision'] / total_number * 100, 2)) + " %)")
        print("#examples: " + str(total_number))
        print(value['not_correct_examples'])


params = {2: {'bins': 4,
              'x_shift': 1,
              'y_shift': 1,
              'nb_cluster': 3,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3},
          3: {'bins': 4,
              'x_shift': 3,
              'y_shift': 3,
              'nb_cluster': 4,
              'prob_threshold_cluster': 0.7,
              'prob_threshold_no_cluster': 0.3
              }
          }


if __name__ == '__main__':
    statistics = {'igci': dict(), 'iacm': dict()}
    for key, value in statistics.items():
        statistics[key] = {'correct':0, 'not_correct':0, 'no_decision': 0, 'not_correct_examples': []}

    base = 3
    not_touched_files = []
    for file in os.listdir("./pairs"):
        if "_des" not in file:
            try:
                #file = "pair0010.txt"
                data = read_data(file)
                data = pd.DataFrame(RobustScaler().fit(data).transform(data))
                data.columns = ['X', 'Y']
                ig = igci(data['X'], data['Y'], refMeasure=1, estimator=2)
                res = iacm(base=base, data=data, params=params[base])

                content = open("./pairs/" + file.replace(".txt", "_des.txt"), "r").read().lower()
                if (
                        "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content) and "X->Y" in res:
                    statistics['iacm']['correct'] = statistics['iacm']['correct'] + 1
                    print("correct")
                elif (
                        "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content) and "Y->X" in res:
                    statistics['iacm']['correct'] = statistics['iacm']['correct'] + 1
                    print("correct")
                elif "no decision" in res:
                    statistics['iacm']['no_decision'] = statistics['iacm']['no_decision'] + 1
                    print("no decision")
                else:
                    statistics['iacm']['not_correct'] = statistics['iacm']['not_correct'] + 1
                    statistics['iacm']['not_correct_examples'].append(file)
                    print("not correct")
                if (
                        "x -> y" in content or "x->y" in content or "x --> y" in content or "x-->y" in content or "x - - > y" in content) and ig < 0:
                    statistics['igci']['correct'] = statistics['igci']['correct'] + 1
                elif (
                        "y -> x" in content or "y->x" in content or "y --> x" in content or "y-->x" in content or "y - - > x" in content or "x <- y" in content) and ig > 0:
                    statistics['igci']['correct'] = statistics['igci']['correct'] + 1
                else:
                    statistics['igci']['not_correct'] = statistics['igci']['not_correct'] + 1
                    statistics['igci']['not_correct_examples'].append(file)
            except Exception as e:
                not_touched_files.append(file)

    print_statisticts(statistics)
    print(not_touched_files)
