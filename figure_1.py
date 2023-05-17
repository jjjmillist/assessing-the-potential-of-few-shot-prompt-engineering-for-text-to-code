import pickle
from scripts.accuracy import accuracy, accuracy_old

import matplotlib.pyplot as pyplot

NO_PREFIX = "results/mbpp-noprefix-09-05-23@18:10:30.pickle"
CODE_ONLY = "results/random-python-10-05-23@20:23:17.pickle"
BERT_AGNOSTIC = "results/10-05-23@13:02:33.pickle"
BERT_AWARE = "results/23-04-23@14:33:31.pickle"
FULL = "results/codeparrot-mbpp-28-04-23@20:50:35.pickle"

no_prefix = accuracy(NO_PREFIX)
code_only = accuracy(CODE_ONLY)
top = accuracy(BERT_AGNOSTIC, ("top",))
bottom = accuracy(BERT_AGNOSTIC, ("bottom",))
bert_aware = accuracy_old(BERT_AWARE)

print("No prefix:", no_prefix)
print("Code only:", code_only)
print("Top:", top)
print("Bottom:", bottom)
print("BERT aware:", bert_aware)

full_accuracies = []
for n in range(30):
    a = accuracy(FULL, (f"seed_{n}",))
    full_accuracies.append(a)

min_acc = min(full_accuracies)
max_acc = max(full_accuracies)
print(min_acc, max_acc)

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
pyplot.show()