import numpy as np

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
