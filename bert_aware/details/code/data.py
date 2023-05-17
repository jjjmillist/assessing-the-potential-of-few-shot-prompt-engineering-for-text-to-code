import json


def mbpp_sanitized():
    with open("/home/ICTDOMAIN/d20126116/Datasets/MBPP/sanitized.json") as file:
        rows = json.load(file)

    return rows