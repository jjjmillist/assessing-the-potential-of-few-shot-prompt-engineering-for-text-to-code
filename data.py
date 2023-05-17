import ast
import io
import json
from settings import n_data


def dataset():
    with open("mbpp/sanitized.json", "r") as file:
        rows = json.load(file)

    if n_data is not None:
        return rows[:n_data]
    else:
        return rows
    

def contexts():    
    for row in dataset():
        code = row["code"]

        root = ast.parse(code)
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_args = [arg.arg for arg in node.args.args]

        yield f"def {function_name}({', '.join(function_args)}):\n"


def tests():
    for row in dataset():
        asserts = row["test_list"]
        imports = row["test_imports"]

        buffer = io.StringIO()

        for import_statement in imports:
            print(import_statement, file=buffer)
        
        if len(imports) > 0:
            print(file=buffer)

        for assert_statement in asserts:
            print(f"{assert_statement}", file=buffer)
        print(file=buffer)

        yield buffer.getvalue()


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