import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import dataset
from workshop import *
from predict import *


mbpp = dataset()

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")

model.to("cuda:0")

n_samples = 20
batch_size = 20

root = output_directory("random-python")

writers = [OutputWriter(root / "top" / f"output_{n}") for n in range(n_samples)]
prompt_writer = OutputWriter(root / "prompts")

total_time = 0
n_prompts = 0

rng = np.random.default_rng(0)

with torch.no_grad():
    for test_row in mbpp:
        prefix_and_prompt = rng.choice(mbpp)["code"]
        prefix_and_prompt += "\n"
        prefix_and_prompt += "# " + test_row["prompt"] + "\n"
        for line in test_row["code"].splitlines():
            if line.startswith("def "):
                prefix_and_prompt += line.strip() + "\n"
                break

        prompt_writer.write(prefix_and_prompt)
                
        t0 = time()
        all_responses = []
        while len(all_responses) < n_samples:
            responses = predict(
                model,
                tokenizer,
                prefix_and_prompt,
                stopping_strategy=stop_on_comment,
                k=10,
                batch_size=batch_size
            )
            all_responses += responses
        t1 = time()

        print(f"{t1 - t0:.2f} seconds")

        total_time += t1 - t0
        n_prompts += 1
        print(f"  {n_prompts}/{len(mbpp)}, average {total_time / n_prompts:.2f} seconds per prompt")

        for text, writer in zip(all_responses, writers):
            writer.write(text)