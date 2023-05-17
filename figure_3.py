from pathlib import Path
import pickle
from os import listdir

import matplotlib.pyplot as pyplot


def accuracy(filepath):
    with open(filepath, "rb") as file:
        results = pickle.load(file)
    metrics, _ = results
    return metrics["pass@1"]


def monte_carlo_accuracies(root):
    root = Path(root)
    accuracies = []
    for sub in listdir(root):
        a = accuracy(root / sub / "code.pickle")
        accuracies.append(a)
    return accuracies


NO_PREFIX            = "results/evaluation/no_prefixes/code.pickle"
CODE_ONLY            = "results/evaluation/random-python/code.pickle"
BERT_AGNOSTIC_TOP    = "results/evaluation/bert_prompt_agnostic/top.pickle"
BERT_AGNOSTIC_BOTTOM = "results/evaluation/bert_prompt_agnostic/bottom.pickle"
BERT_AWARE           = "results/evaluation/bert_prompt_aware/code.pickle"
RANDOM               = "results/evaluation/random_prefixes"

no_prefix  = accuracy(NO_PREFIX)
code_only  = accuracy(CODE_ONLY)
top        = accuracy(BERT_AGNOSTIC_TOP)
bottom     = accuracy(BERT_AGNOSTIC_BOTTOM)
bert_aware = accuracy(BERT_AWARE)

random_accuracies = monte_carlo_accuracies(RANDOM)

print("No prefix:", no_prefix)
print("Code only:", code_only)
print("Top:", top)
print("Bottom:", bottom)
print("BERT aware:", bert_aware)

min_acc = min(random_accuracies)
max_acc = max(random_accuracies)
print(f"Random: {min_acc}-{max_acc}")

x_axis = list(range(6))

pyplot.grid(axis="y")
pyplot.gca().set_axisbelow(True)

pyplot.bar(
    x_axis[1:],
    [no_prefix, code_only, top, bottom, bert_aware],    
)
pyplot.bar(
    x_axis[0],
    max_acc,
)
pyplot.bar(
    x_axis[0],
    min_acc,
)

pyplot.xticks(
    x_axis,
    ["Full", "No prefix", "Code only", "BERT agnostic\n(top)", "BERT agnostic\n(bottom)", "BERT aware"]
)

pyplot.ylabel("Accuracy (%)")

Path("results/figures").mkdir(parents=True, exist_ok=True)
pyplot.savefig("results/figures/figure_3.png")