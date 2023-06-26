from pathlib import Path
import numpy as np
import matplotlib.pyplot as pyplot

from evaluation import get_monte_carlo_results


result_matrix = get_monte_carlo_results()

n_prefixes, n_prompts, n_samples = result_matrix.shape

means = np.mean(result_matrix, axis=(1, 2))
shuffled = np.empty_like(result_matrix)
for problem in range(n_prompts):
    for sample in range(n_samples):
        shuffled[:, problem, sample] = np.random.permutation(result_matrix[:, problem, sample])
shuffled_means = np.mean(shuffled, axis=(1, 2))

pyplot.bar(np.arange(len(means)), 100 * np.sort(means), label="Experiment")
pyplot.bar(np.arange(len(means)), 100 * np.sort(shuffled_means), fc="none", hatch="//", lw=3, ec="orange", label="Control")

Path("results/figures").mkdir(parents=True, exist_ok=True)
pyplot.savefig("results/figures/figure_1.png", dpi=200, bbox_inches="tight")