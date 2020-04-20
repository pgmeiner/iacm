import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def get_discrete_number(max_samples, base):
    r = np.random.normal(0, 1, max_samples)
    return KBinsDiscretizer(n_bins=base, encode='ordinal', strategy='uniform').fit_transform(r.reshape(-1,1)).reshape(1,-1)


def generate_nonlinear_discrete_data(max_samples, base):
    intervention_samples = max_samples / (base - 1)
    nb_samples = int(intervention_samples) * (base - 1)
    obsX = get_discrete_number(nb_samples, base)
    obsY = ((5 * obsX * obsX - 1 * get_discrete_number(nb_samples, base)) % base).reshape(1,-1)

    # intervention data
    intX = (np.concatenate([np.repeat(inter, intervention_samples).tolist() for inter in range(0,base-1)]) +
            get_discrete_number(nb_samples, base)) % base
    intY = ((5 * intX * intX - 1 * get_discrete_number(nb_samples, base)) % base).reshape(1,-1)

    return obsX.reshape(nb_samples,), obsY.reshape(nb_samples,), intX.reshape(nb_samples,), intY.reshape(nb_samples,)


def generate_linear_discrete_data(max_samples, base):
    pass

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
