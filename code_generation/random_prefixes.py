import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import *
from workshop import *
from predict import *
from settings import *


tokenizer = AutoTokenizer.from_pretrained(model_uri)
model = AutoModelForCausalLM.from_pretrained(model_uri)
model.to("cuda:0")

mbpp = dataset()

seeds = list(range(n_monte_carlo_prefixes))
test_indices = list(range(len(mbpp)))

root = output_directory("code/random_prefixes")
for seed in seeds:
    prompts = []
    rng = np.random.default_rng(seed)
    for test_index in test_indices:
        test_instance = mbpp[test_index]
        train_indices = rng.choice(len(mbpp), size=n_contextual_examples, replace=False)
        training_instances = [mbpp[i] for i in train_indices]

        prefix_and_prompt = few_shot_mbpp(test_instance, training_instances)
        
        prompts.append(prefix_and_prompt)

    predict_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        n_samples=n_samples,
        k=n_distribution_cutoff,
        output_path=root / f"prefix_{seed}" / "code.pickle"
    )