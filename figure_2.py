from os import listdir
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as pyplot

from evaluation import get_monte_carlo_results


result_matrix = get_monte_carlo_results()

n_prefixes, n_prompts, n_samples = result_matrix.shape

train = result_matrix[:, :, :n_samples // 2]
valid = result_matrix[:, :, n_samples // 2:]

train_means = np.mean(train, axis=(1, 2))
valid_means = np.mean(valid, axis=(1, 2))

pyplot.plot(train_means, valid_means, "o")
pyplot.show()