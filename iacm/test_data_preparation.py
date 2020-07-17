import pandas as pd
from iacm.data_preparation import get_contingency_table, get_contingency_table_general, get_probabilities, \
    get_probabilities_general, get_probabilities_intervention, get_probabilities_intervention_general


data_x = [-1.0, -0.9130434782608695, -0.8260869565217391, -0.7391304347826086, -0.6521739130434783, -0.5652173913043478, -0.4782608695652174, -0.391304347826087, -0.30434782608695654, -0.21739130434782608, -0.13043478260869565, -0.043478260869565216, 0.043478260869565216, 0.13043478260869565, 0.21739130434782608, 0.30434782608695654, 0.391304347826087, 0.4782608695652174, 0.5652173913043478, 0.6521739130434783, 0.7391304347826086, 0.8260869565217391, 0.9130434782608695, 1.0, -1.0, -0.9130434782608695, -0.8260869565217391, -0.7391304347826086, -0.6521739130434783, -0.5652173913043478, -0.4782608695652174, -0.391304347826087, -0.30434782608695654, -0.21739130434782608, -0.13043478260869565, -0.043478260869565216, 0.043478260869565216, 0.13043478260869565, 0.21739130434782608, 0.30434782608695654, 0.391304347826087, 0.4782608695652174, 0.5652173913043478, 0.6521739130434783, 0.7391304347826086, 0.8260869565217391, 0.9130434782608695, 1.0, -1.0, -0.9130434782608695, -0.8260869565217391, -0.7391304347826086, -0.6521739130434783, -0.5652173913043478, -0.4782608695652174, -0.391304347826087, -0.30434782608695654, -0.21739130434782608, -0.13043478260869565, -0.043478260869565216, 0.043478260869565216, 0.13043478260869565, 0.21739130434782608, 0.30434782608695654, 0.391304347826087, 0.4782608695652174, 0.5652173913043478, 0.6521739130434783, 0.7391304347826086, 0.8260869565217391, 0.9130434782608695, 1.0, -1.0, -0.9130434782608695, -0.8260869565217391, -0.7391304347826086, -0.6521739130434783, -0.5652173913043478, -0.4782608695652174, -0.391304347826087, -0.30434782608695654, -0.21739130434782608, -0.13043478260869565, -0.043478260869565216, 0.043478260869565216, 0.13043478260869565, 0.21739130434782608, 0.30434782608695654, 0.391304347826087, 0.4782608695652174, 0.5652173913043478, 0.6521739130434783, 0.7391304347826086, 0.8260869565217391, 0.9130434782608695, 1.0, -1.0, -0.9130434782608695, -0.8260869565217391]
data_y = [-0.6666666666666666, -0.6363636363636364, -0.6666666666666666, -0.6060606060606061, -0.6060606060606061, -0.48484848484848486, -0.3939393939393939, -0.2727272727272727, -0.21212121212121213, -0.18181818181818182, -0.12121212121212122, -0.18181818181818182, -0.18181818181818182, -0.30303030303030304, -0.48484848484848486, -0.48484848484848486, -0.48484848484848486, -0.48484848484848486, -0.48484848484848486, -0.5151515151515151, -0.6363636363636364, -0.6060606060606061, -0.5757575757575758, -0.6363636363636364, -0.696969696969697, -0.7272727272727273, -0.6666666666666666, -0.6666666666666666, -0.6060606060606061, -0.5151515151515151, -0.36363636363636365, -0.30303030303030304, -0.2727272727272727, -0.18181818181818182, -0.15151515151515152, -0.09090909090909091, -0.09090909090909091, -0.21212121212121213, -0.30303030303030304, -0.3939393939393939, -0.5151515151515151, -0.45454545454545453, -0.42424242424242425, -0.45454545454545453, -0.5454545454545454, -0.5454545454545454, -0.48484848484848486, -0.6060606060606061, -0.6060606060606061, -0.696969696969697, -0.7272727272727273, -0.696969696969697, -0.696969696969697, -0.5454545454545454, -0.36363636363636365, -0.3333333333333333, -0.2727272727272727, -0.2727272727272727, -0.2727272727272727, -0.24242424242424243, -0.30303030303030304, -0.3939393939393939, -0.48484848484848486, -0.5454545454545454, -0.5454545454545454, -0.5151515151515151, -0.48484848484848486, -0.5454545454545454, -0.5454545454545454, -0.5454545454545454, -0.5454545454545454, -0.5757575757575758, -0.6060606060606061, -0.5757575757575758, -0.6060606060606061, -0.6060606060606061, -0.6060606060606061, -0.5454545454545454, -0.5151515151515151, -0.48484848484848486, -0.48484848484848486, -0.42424242424242425, -0.42424242424242425, -0.3939393939393939, -0.3939393939393939, -0.42424242424242425, -0.48484848484848486, -0.5454545454545454, -0.5454545454545454, -0.5151515151515151, -0.5454545454545454, -0.5454545454545454, -0.6060606060606061, -0.6060606060606061, -0.6060606060606061, -0.6363636363636364, -0.6363636363636364, -0.6363636363636364, -0.6060606060606061]


def test_get_contingency_table():
    contingency_table = get_contingency_table(pd.Series(data_x), pd.Series(data_y), 2, 2)
    assert contingency_table[0] == [28, 25]
    assert contingency_table[1] == [27, 23]

    contingency_table = get_contingency_table(pd.Series([-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1]),
                                              pd.Series([200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]), 2, 2)
    assert contingency_table[0] == [6, 2]
    assert contingency_table[1] == [2, 5]

    contingency_table = get_contingency_table(pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]),
                                              pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]), 2, 2)
    assert contingency_table[0] == [8, 2]
    assert contingency_table[1] == [1, 4]

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]), 2, 2)
    assert contingency_table[0] == [8, 1]
    assert contingency_table[1] == [2, 4]

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9]), 3, 3)
    assert contingency_table[0] == [5, 1, 1]
    assert contingency_table[1] == [1, 4, 1]
    assert contingency_table[2] == [1, 2, 3]


def test_get_probabilities():
    contingency_table = get_contingency_table(pd.Series(data_x), pd.Series(data_y), 2, 2)
    probabilities = get_probabilities(contingency_table, 2, 2)
    int_probabilities = get_probabilities_intervention(contingency_table, 2, 2)
    assert probabilities['00'] == 0.27184466019417475
    assert probabilities['01'] == 0.24271844660194175
    assert probabilities['10'] == 0.2621359223300971
    assert probabilities['11'] == 0.22330097087378642
    assert int_probabilities['0_0'] == 0.5283018867924528
    assert int_probabilities['0_1'] == 0.4716981132075472
    assert int_probabilities['1_0'] == 0.54
    assert int_probabilities['1_1'] == 0.46

    contingency_table = get_contingency_table(pd.Series([-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1]),
                                              pd.Series([200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]), 2, 2)
    probabilities = get_probabilities(contingency_table, 2, 2)
    int_probabilities = get_probabilities_intervention(contingency_table, 2, 2)
    assert probabilities['00'] == 0.4
    assert probabilities['01'] == 0.13333333333333333
    assert probabilities['10'] == 0.13333333333333333
    assert probabilities['11'] == 0.3333333333333333
    assert int_probabilities['0_0'] == 0.75
    assert int_probabilities['0_1'] == 0.25
    assert int_probabilities['1_0'] == 0.2857142857142857
    assert int_probabilities['1_1'] == 0.7142857142857143

    contingency_table = get_contingency_table(pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]),
                                              pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]), 2, 2)
    probabilities = get_probabilities(contingency_table, 2, 2)
    int_probabilities = get_probabilities_intervention(contingency_table, 2, 2)
    assert probabilities['00'] == 0.5333333333333333
    assert probabilities['01'] == 0.13333333333333333
    assert probabilities['10'] == 0.06666666666666667
    assert probabilities['11'] == 0.26666666666666666
    assert int_probabilities['0_0'] == 0.8
    assert int_probabilities['0_1'] == 0.2
    assert int_probabilities['1_0'] == 0.2
    assert int_probabilities['1_1'] == 0.8

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]), 2, 2)
    probabilities = get_probabilities(contingency_table, 2, 2)
    int_probabilities = get_probabilities_intervention(contingency_table, 2, 2)
    assert probabilities['00'] == 0.5333333333333333
    assert probabilities['01'] == 0.06666666666666667
    assert probabilities['10'] == 0.13333333333333333
    assert probabilities['11'] == 0.26666666666666666
    assert int_probabilities['0_0'] == 0.8888888888888888
    assert int_probabilities['0_1'] == 0.1111111111111111
    assert int_probabilities['1_0'] == 0.3333333333333333
    assert int_probabilities['1_1'] == 0.6666666666666666

    contingency_table = get_contingency_table(pd.Series([0, 10, 20, 30, 40, 50, 50, 70, 80, 90]),
                                              pd.Series([0, 1, 2, 3, 4, 5, 5, 5, 8, 9]), 3, 3)
    probabilities = get_probabilities(contingency_table, 3, 3)
    int_probabilities = get_probabilities_intervention(contingency_table, 3, 3)
    assert probabilities['00'] == 0.2631578947368421
    assert probabilities['01'] == 0.05263157894736842
    assert probabilities['02'] == 0.05263157894736842
    assert probabilities['10'] == 0.05263157894736842
    assert probabilities['11'] == 0.21052631578947367
    assert probabilities['12'] == 0.05263157894736842
    assert probabilities['20'] == 0.05263157894736842
    assert probabilities['21'] == 0.10526315789473684
    assert probabilities['22'] == 0.15789473684210525
    assert int_probabilities['0_0'] == 0.7142857142857143
    assert int_probabilities['0_1'] == 0.14285714285714285
    assert int_probabilities['0_2'] == 0.14285714285714285
    assert int_probabilities['1_0'] == 0.16666666666666666
    assert int_probabilities['1_1'] == 0.6666666666666666
    assert int_probabilities['1_2'] == 0.16666666666666666
    assert int_probabilities['2_0'] == 0.16666666666666666
    assert int_probabilities['2_1'] == 0.3333333333333333
    assert int_probabilities['2_2'] == 0.5


def test_get_contingency_table_general():
    contingency_table = get_contingency_table_general(pd.DataFrame({'x': data_x, 'y': data_y}), {'x': 2, 'y': 2})
    probabilities = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    int_probabilities = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'])['y']
    assert probabilities['00'] == 0.27184466019417475
    assert probabilities['01'] == 0.24271844660194175
    assert probabilities['10'] == 0.2621359223300971
    assert probabilities['11'] == 0.22330097087378642
    assert int_probabilities['0_0'] == 0.5283018867924528
    assert int_probabilities['0_1'] == 0.4716981132075472
    assert int_probabilities['1_0'] == 0.54
    assert int_probabilities['1_1'] == 0.46

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [-10, 10, 2, 4, 6, 0, 3, 4, 100, 0, 1], 'y': [200, 400, 3, 5, 6, 2, 3, 4, 5, 0, -1]}),
        {'x': 2, 'y': 2})
    probabilities = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    int_probabilities = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'])['y']
    assert probabilities['00'] == 0.4
    assert probabilities['01'] == 0.13333333333333333
    assert probabilities['10'] == 0.13333333333333333
    assert probabilities['11'] == 0.3333333333333333
    assert int_probabilities['0_0'] == 0.75
    assert int_probabilities['0_1'] == 0.25
    assert int_probabilities['1_0'] == 0.2857142857142857
    assert int_probabilities['1_1'] == 0.7142857142857143

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10], 'y': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90]}),
        {'x': 2, 'y': 2})
    probabilities = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    int_probabilities = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'])['y']
    assert probabilities['00'] == 0.5333333333333333
    assert probabilities['01'] == 0.13333333333333333
    assert probabilities['10'] == 0.06666666666666667
    assert probabilities['11'] == 0.26666666666666666
    assert int_probabilities['0_0'] == 0.8
    assert int_probabilities['0_1'] == 0.2
    assert int_probabilities['1_0'] == 0.2
    assert int_probabilities['1_1'] == 0.8

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10]}),
        {'x': 2, 'y': 2})
    probabilities = get_probabilities_general(contingency_table, {'x': 2, 'y': 2})
    int_probabilities = get_probabilities_intervention_general(contingency_table, {'x': 2, 'y': 2}, ['x'])['y']
    assert probabilities['00'] == 0.5333333333333333
    assert probabilities['01'] == 0.06666666666666667
    assert probabilities['10'] == 0.13333333333333333
    assert probabilities['11'] == 0.26666666666666666
    assert int_probabilities['0_0'] == 0.8888888888888888
    assert int_probabilities['0_1'] == 0.1111111111111111
    assert int_probabilities['1_0'] == 0.3333333333333333
    assert int_probabilities['1_1'] == 0.6666666666666666

    contingency_table = get_contingency_table_general(
        pd.DataFrame({'x': [0, 10, 20, 30, 40, 50, 50, 70, 80, 90], 'y': [0, 1, 2, 3, 4, 5, 5, 5, 8, 9]}),
        {'x': 3, 'y': 3})
    probabilities = get_probabilities_general(contingency_table, {'x': 3, 'y': 3})
    int_probabilities = get_probabilities_intervention_general(contingency_table, {'x': 3, 'y': 3}, ['x'])['y']
    assert probabilities['00'] == 0.2631578947368421
    assert probabilities['01'] == 0.05263157894736842
    assert probabilities['02'] == 0.05263157894736842
    assert probabilities['10'] == 0.05263157894736842
    assert probabilities['11'] == 0.21052631578947367
    assert probabilities['12'] == 0.05263157894736842
    assert probabilities['20'] == 0.05263157894736842
    assert probabilities['21'] == 0.10526315789473684
    assert probabilities['22'] == 0.15789473684210525
    assert int_probabilities['0_0'] == 0.7142857142857143
    assert int_probabilities['0_1'] == 0.14285714285714285
    assert int_probabilities['0_2'] == 0.14285714285714285
    assert int_probabilities['1_0'] == 0.16666666666666666
    assert int_probabilities['1_1'] == 0.6666666666666666
    assert int_probabilities['1_2'] == 0.16666666666666666
    assert int_probabilities['2_0'] == 0.16666666666666666
    assert int_probabilities['2_1'] == 0.3333333333333333
    assert int_probabilities['2_2'] == 0.5
