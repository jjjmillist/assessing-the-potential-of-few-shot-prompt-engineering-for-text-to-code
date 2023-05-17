import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import numpy as np

from data import dataset
from workshop import *
from predict import *
from settings import *


def mbpp_prefix(components):
    prefix = ""
    for row in components:
        for line in row["prompt"].splitlines():
            prefix += "# " + line + "\n"
        prefix += row["code"] + "\n\n"
    return prefix


with open("results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle", "rb") as file:
    data = pickle.load(file)

BERT_DIMENSION = 768

embeddings = np.array([row["bert_encoding"].detach().numpy() for row in data])
assert embeddings.shape == (len(data), BERT_DIMENSION)
dot_products = embeddings.dot(embeddings.T)
assert dot_products.shape == (len(data), len(data))
norms = np.linalg.norm(dot_products, 2, axis=1)
assert norms.shape == (len(data),)
norm_products = norms[:, None].dot(norms[None, :])
assert norm_products.shape == (len(data), len(data))
cosines = dot_products / norms
assert cosines.shape == (len(data), len(data))

mean_similarities = cosines.mean(axis=0)
top = np.argsort(mean_similarities)[-n_contextual_examples:]
bottom = np.argsort(mean_similarities)[:n_contextual_examples]

mbpp = dataset()
top_prefix = mbpp_prefix(mbpp[i] for i in top)
bottom_prefix = mbpp_prefix(mbpp[i] for i in bottom)

tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForCausalLM.from_pretrained(model_uri)
model.to("cuda:0")

root = output_directory("code/bert_prompt_agnostic")
for prefix, name in ((top_prefix, "top"), (bottom_prefix, "bottom")):
    prompts = []
    for test_row in mbpp:
        prefix_and_prompt = prefix
        prefix_and_prompt += "# " + test_row["prompt"] + "\n"
        for line in test_row["code"].splitlines():
            if line.startswith("def "):
                prefix_and_prompt += line.strip() + "\n"
                break
        prompts.append(prefix_and_prompt)
                
    predict_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        n_samples=n_samples,
        k=n_distribution_cutoff,
        output_path=root / (name + ".pickle")
    )