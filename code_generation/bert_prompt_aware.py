from   transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import numpy as np

from data import dataset
from util import *
from predict import *
from settings import *


with open("results/bert_encodings/mbpp_sanitized_with_bert_encodings.pickle", "rb") as file:
    data = pickle.load(file)

BERT_DIMENSION = 768

embeddings = np.array([row["bert_embedding"].detach().numpy() for row in data])
assert embeddings.shape == (len(data), BERT_DIMENSION)
dot_products = embeddings.dot(embeddings.T)
assert dot_products.shape == (len(data), len(data))
norms = np.linalg.norm(embeddings, 2, axis=1)
assert norms.shape == (len(data),)
norm_products = norms[:, None].dot(norms[None, :])
assert norm_products.shape == (len(data), len(data))
cosines = dot_products / norm_products
assert cosines.shape == (len(data), len(data))

mbpp = dataset()
prompts = []
for test_index in range(len(mbpp)):
    top = np.argsort(
        cosines[test_index]
    )
    prefix_and_prompt = ""

    top_indices = top[-(n_contextual_examples + 1):]
    top_indices = [i for i in top_indices if i != test_index]
    if len(top_indices) > n_contextual_examples:
        top_indices = top_indices[1:]
    
    for i in top_indices:
        assert i != test_index
        
        for line in mbpp[i]["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        prefix_and_prompt += mbpp[i]["code"] + "\n\n"

    prefix_and_prompt += "# " + mbpp[test_index]["prompt"] + "\n"
    for line in mbpp[test_index]["code"].splitlines():
        if line.startswith("def "):
            prefix_and_prompt += line.strip() + "\n"
            break
    
    prompts.append(prefix_and_prompt)

tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForCausalLM.from_pretrained(model_uri)
model.to("cuda:0")

root = output_directory("code/bert_prompt_aware")
predict_batch(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    n_samples=n_samples,
    k=n_distribution_cutoff,
    output_path=root / "code.pickle"
)