from   transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from data import dataset
from workshop import *
from predict import *
from settings import *


mbpp = dataset()

tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForCausalLM.from_pretrained(model_uri)

model.to("cuda:0")

root = output_directory("code/code_only")

total_time = 0
n_prompts = 0

rng = np.random.default_rng(0)

prompts = []
for test_row in mbpp:
    prefix_and_prompt = rng.choice(mbpp)["code"]
    prefix_and_prompt += "\n"
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
    output_path=root / "code.pickle"
)