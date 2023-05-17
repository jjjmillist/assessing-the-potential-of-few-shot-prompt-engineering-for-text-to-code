import pickle

import numpy as np
import matplotlib.pyplot as pyplot

with open("results/codeparrot-mbpp-28-04-23@20:50:35.pickle", "rb") as file:
    data = pickle.load(file)

paths = list(data.keys())
seed_keys = set(path[0] for path in paths)
sample_keys = sorted(list(set(path[1] for path in paths)))

n_prompts = len(data[paths[0]])
n_samples = len(sample_keys)

result_matrix = np.empty((len(seed_keys), n_prompts, len(sample_keys)))

for seed_index, seed_key in enumerate(seed_keys):
    for sample_index, sample_key in enumerate(sample_keys):
        row = data[(seed_key, sample_key)]
        vectorized = np.array([row[prompt_index] for prompt_index in range(n_prompts)])
        result_matrix[seed_index, :, sample_index] = vectorized

# result_matrix = result_matrix[:, :, :]

train = result_matrix[:, :, :n_samples // 2]
valid = result_matrix[:, :, n_samples // 2:]

train_means = np.mean(train, axis=(1, 2))
valid_means = np.mean(valid, axis=(1, 2))

# pyplot.plot(train_means, valid_means, "o")

means = np.mean(result_matrix, axis=(1, 2))
shuffled = np.empty_like(result_matrix)
for problem in range(result_matrix.shape[1]):
    for sample in range(result_matrix.shape[2]):
        shuffled[:, problem, sample] = np.random.permutation(result_matrix[:, problem, sample])
shuffled_means = np.mean(shuffled, axis=(1, 2))

pyplot.bar(np.arange(len(means)), 100 * np.sort(means), label="Experiment")
pyplot.bar(np.arange(len(means)), 100 * np.sort(shuffled_means), fc="none", hatch="//", lw=3, ec="orange", label="Control")

pyplot.show()