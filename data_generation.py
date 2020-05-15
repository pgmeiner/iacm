import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import t


def get_discrete_random_number(sample_size, alphabet_size):
    return np.random.randint(low=0, high=alphabet_size, size=sample_size).reshape(-1,1)


def get_random_number(sample_size):
    df = np.random.randint(low=2, high=10, size=1)
    r = t.rvs(df, size=sample_size)
    return r


def get_nonlinear_function():
    function_index = np.random.randint(low=1, high=4, size=1)
    if function_index == 1:
        return lambda x, n: np.max(x, 0) + n
    elif function_index == 2:
        return lambda x, n: np.sin(2*np.pi*x) + n
    elif function_index == 3:
        return lambda x, n: np.sign(x)*np.sqrt(np.abs(x)) + n


def get_linear_function():
    alpha = np.random.normal(0, 10, 1)
    return lambda x, n: alpha*x + n


def generate_discrete_data(structure, sample_size, alphabet_size_x, alphabet_size_y):
    if 'nonlinear' in structure:
        f = get_nonlinear_function()
    else:
        f = get_linear_function()

    nb_intervention_samples = int(sample_size / alphabet_size_x)
    nb_samples = nb_intervention_samples * alphabet_size_x
    obsX = get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_x)
    obsY = np.array([int(el[0]) for el in (f(obsX, get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_y))%alphabet_size_y).reshape(-1,1).tolist()]).reshape(-1,1)

    # intervention data
    intX = np.array(np.concatenate([np.repeat(inter, nb_intervention_samples).tolist() for inter in range(0, alphabet_size_x)])).reshape(-1,1)
    intY = np.array([int(el[0]) for el in (f(intX, get_discrete_random_number(sample_size=nb_samples, alphabet_size=alphabet_size_y))%alphabet_size_y).reshape(-1,1).tolist()]).reshape(-1,1)

    return pd.DataFrame({'X': [el[0] for el in np.concatenate([obsX, intX])], 'Y': [el[0] for el in np.concatenate([obsY, intY])]})


def generate_continuous_data(structure, sample_size):
    if 'nonlinear' in structure:
        f = get_nonlinear_function()
    else:
        f = get_linear_function()

    nb_interventions = np.random.randint(low=2, high=10, size=1)
    nb_intervention_samples = int(sample_size / nb_interventions)
    nb_samples = nb_intervention_samples * nb_interventions
    obsX = get_random_number(sample_size=nb_samples)
    obsY = f(obsX, get_random_number(nb_samples)).reshape(1, -1)

    # intervention data
    intX = np.concatenate([np.repeat(inter, nb_intervention_samples).tolist() for inter in get_random_number(sample_size=nb_interventions).tolist()])
    intY = f(intX, get_random_number(nb_samples)).reshape(1, -1)

    return obsX.reshape(nb_samples, ), obsY.reshape(nb_samples, ), intX.reshape(nb_samples, ), intY.reshape(nb_samples, )



def generate_linear_data(max_samples):
    obsX = np.random.normal(0, 1, max_samples)
    obsY = 5 * obsX - 1 * np.random.normal(0, 1, max_samples)

    # intervention data
    intX = np.array(np.repeat(100, max_samples * 0.6).tolist() + np.repeat(-100, max_samples * 0.4).tolist())
    intY = 5 * intX - 1 * np.random.normal(0, 1, max_samples)

    return obsX, obsY, intX, intY


def generate_nonlinear_data(max_samples):
    obsX = np.random.normal(0, 1, max_samples)
    obsY = 5 * obsX * obsX - 1 * np.random.normal(0, 1, max_samples)

    # intervention data
    intX = np.array(
        (np.repeat(100, max_samples * 0.6).tolist() + np.repeat(-100, max_samples * 0.4).tolist()) + np.random.normal(0,
                                                                                                                      1,
                                                                                                                      max_samples))
    intY = 5 * intX * intX - 1 * np.random.normal(0, 1, max_samples)

    return obsX, obsY, intX, intY


def generate_linear_confounded_data(max_samples):
    h = np.random.normal(0, 1, max_samples)
    obsX = 3 * h + np.random.normal(0, 1, max_samples)
    # obsX = np.random.normal(0, 1, max_samples)
    obsY = 5 * h - 1 * np.random.normal(0, 1, max_samples)

    # intervention data
    h = np.random.normal(0, 1, max_samples)
    intX = np.array(np.repeat(100, max_samples * 0.6).tolist() + np.repeat(-100, max_samples * 0.4).tolist())
    intY = 5 * h - 1 * np.random.normal(0, 1, max_samples)

    return obsX, obsY, intX, intY


def generate_nonlinear_confounded_data(max_samples):
    h = np.random.normal(0, 1, max_samples)
    obsX = 3 * h * h + np.random.normal(0, 1, max_samples)
    # obsX = np.random.normal(0, 1, max_samples)
    obsY = 5 * h * h - 1 * np.random.normal(0, 1, max_samples)

    # intervention data
    h = np.random.normal(0, 1, max_samples)
    intX = np.array(np.repeat(100, max_samples * 0.6).tolist() + np.repeat(-100, max_samples * 0.4).tolist())
    intY = 5 * h * h - 1 * np.random.normal(0, 1, max_samples)

    return obsX, obsY, intX, intY
