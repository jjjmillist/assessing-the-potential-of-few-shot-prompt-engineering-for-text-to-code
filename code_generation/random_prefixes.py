import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import *
from workshop import *
from predict import *


root = output_directory("codeparrot-mbpp")

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")
model.to("cuda:0")

k = 3
mbpp = dataset()

seeds = list(range(30))
test_indices = list(range(len(mbpp)))
batch_size = 20

t0 = time()
for seed in seeds:
    with torch.no_grad():
        rng = np.random.default_rng(seed)
        writers = [OutputWriter(root / f"seed_{seed}" / f"output_{n}") for n in range(batch_size)]
        prompts = []

        for test_index in test_indices:
            test_instance = mbpp[test_index]
            train_indices = rng.choice(len(mbpp), size=3, replace=False)
            training_instances = [mbpp[i] for i in train_indices]

            prefix_and_prompt = few_shot_mbpp(test_instance, training_instances)
            
            prompts.append(prefix_and_prompt)

        with open(root / f"seed_{seed}_prompts.pickle", "wb") as prompt_file:
            pickle.dump(prompts, prompt_file)
        
        for prompt in prompts:
            responses = predict(
                model,
                tokenizer,
                prompt,
                stopping_strategy=stop_on_comment,
                k=10,
                batch_size=batch_size
            )

            for text, writer in zip(responses, writers):
                writer.write(text)