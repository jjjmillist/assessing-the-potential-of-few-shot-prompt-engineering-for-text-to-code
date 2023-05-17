import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import dataset
from workshop import *
from predict import *


with open("results/23-04-23@12:55:53/mbpp_sanitized_with_bert_encodings.pickle", "rb") as file:
    data = pickle.load(file)

BERT_DIMENSION = 768

embeddings = np.array([row["bert_embedding"].detach().numpy() for row in data])
assert embeddings.shape == (len(data), BERT_DIMENSION)
dot_products = embeddings.dot(embeddings.T)
assert dot_products.shape == (len(data), len(data))
norms = np.linalg.norm(dot_products, 2, axis=1)
assert norms.shape == (len(data),)
norm_products = norms[:, None].dot(norms[None, :])
assert norm_products.shape == (len(data), len(data))
cosines = dot_products / norms
assert cosines.shape == (len(data), len(data))

k = 3
mbpp = dataset()
prompts = []
for test_index in range(len(mbpp)):
    top = np.argsort(
        cosines[test_index]
    )
    prefix_and_prompt = ""
    for i in top[-(k + 1):-1]:
        for line in mbpp[i]["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        prefix_and_prompt += mbpp[i]["code"] + "\n\n"

    prefix_and_prompt += "# " + mbpp[test_index]["prompt"] + "\n"
    for line in mbpp[test_index]["code"].splitlines():
        if line.startswith("def "):
            prefix_and_prompt += line.strip() + "\n"
            break
    
    prompts.append(prefix_and_prompt)

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")

model.to("cuda:0")

n_samples = 20
batch_size = 20

root = output_directory()

writers = [OutputWriter(root / f"output_{n}") for n in range(n_samples)]

total_time = 0
n_prompts = 0

with torch.no_grad():
    for prompt in prompts:
        t0 = time()
        all_responses = []
        while len(all_responses) < n_samples:
            responses = predict(
                model,
                tokenizer,
                prompt,
                stopping_strategy=stop_on_comment,
                k=10,
                batch_size=batch_size
            )
            all_responses += responses
        t1 = time()

        print(f"{t1 - t0:.2f} seconds")

        total_time += t1 - t0
        n_prompts += 1
        print(f"  {n_prompts}/{len(prompts)}, average {total_time / n_prompts:.2f} seconds per prompt")

        for text, writer in zip(all_responses, writers):
            writer.write(text)