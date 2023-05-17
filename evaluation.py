from evaluate import load
from os import listdir
from pathlib import Path
from data import contexts, tests
import pickle


if __name__ == "__main__":
    code_eval = load("code_eval")

    mbpp_contexts = list(contexts())
    mbpp_tests = list(tests())

    frontier = [Path("results/code")]
    output_root = Path("results/evaluation")
    while frontier:
        new_frontier = []
        for current in frontier:
            if current.is_dir():
                for sub in listdir(current):
                    new_frontier.append(current / sub)
            elif Path(current).is_file():
                relative_path = current.relative_to("results/code")
                output_path = output_root / relative_path

                with open(current, "rb") as file:
                    snippets = pickle.load(file)

                predictions = []
                for i in range(len(snippets)):
                    candidate_list = snippets[i]
                    context = mbpp_contexts[i]
                    predictions.append([context + candidate for candidate in candidate_list])

                print(current)
                results = code_eval.compute(references=mbpp_tests, predictions=predictions, k=[1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as file:
                    pickle.dump(results, file)

        frontier = new_frontier
