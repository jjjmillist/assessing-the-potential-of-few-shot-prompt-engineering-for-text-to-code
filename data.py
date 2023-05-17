import json
from settings import n_data


def dataset():
    with open("/home/ICTDOMAIN/d20126116/Datasets/MBPP/sanitized.json") as file:
        rows = json.load(file)

    if n_data is not None:
        return rows[:n_data]
    else:
        return rows


def few_shot_mbpp(test_instance, training_instances):
    prefix_and_prompt = ""
    for instance in training_instances:
        for line in instance["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        prefix_and_prompt += instance["code"] + "\n\n"

    prefix_and_prompt += "# " + test_instance["prompt"] + "\n"
    for line in test_instance["code"].splitlines():
        if line.startswith("def "):
            prefix_and_prompt += line.strip() + "\n"
            break
    return prefix_and_prompt