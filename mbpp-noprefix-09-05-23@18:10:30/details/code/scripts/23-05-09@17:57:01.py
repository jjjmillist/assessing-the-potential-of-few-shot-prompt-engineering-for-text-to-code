import transformers
import torch
from   transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import pickle
import numpy as np

from data import mbpp_sanitized
from workshop import *


def stop_on_comment(text):
    last_line = text.splitlines()[-1]
    if len(last_line) > 0 and last_line[0] == "#":
        truncated = "\n".join(text.splitlines()[:-1])
        return False, truncated
    else:
        return True, None


def predict(model, tokenizer, prompt, stopping_strategy=None, k=10, batch_size=10):
    generated_text = ""
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    n_generated_tokens = 0
    context_size = model.config.n_ctx

    # print(f"Running prediction on a prompt with {prompt_length} tokens")

    generated_strings = [""] * batch_size
    running_sample_indices = list(range(batch_size))

    while prompt_length + n_generated_tokens <= context_size:
        # print("  Generating token")
        # print("    Running samples:", running_sample_indices)

        outputs = model(**inputs)
        top_k = outputs.logits[:, -1].topk(k)

        next_token_ids = []
        accepted_sample_indices = []
        for i in range(len(running_sample_indices)):
            # print(f"      Extending sample number {i}: {running_sample_indices[i]}")

            logits = top_k.values[i]
            p = torch.distributions.Categorical(logits=logits)
            j = p.sample((1,))
            sampled_id = top_k.indices[i, j]
            next_token = tokenizer.decode(sampled_id)

            # print(f"        Chose {sampled_id[0]}: {next_token}")

            sample_index = running_sample_indices[i]
            generated_strings[sample_index] += next_token
        
            accepted, truncated = stopping_strategy(generated_strings[sample_index])
            if not accepted:
                generated_strings[sample_index] = truncated                
                # print(f"        Sample is finished")
            else:
                next_token_ids.append([sampled_id])
                accepted_sample_indices.append(i)
                # print(f"        Sample continues")

        running_sample_indices = [running_sample_indices[i] for i in accepted_sample_indices]

        # print("    Accepted:", accepted_sample_indices)
        # print("    Running samples:", running_sample_indices)

        n_generated_tokens += 1

        if len(running_sample_indices) == 0:
            break

        new_layers = []
        for layer in outputs.past_key_values:
            new_layers.append(
                (
                    layer[0][accepted_sample_indices],
                    layer[1][accepted_sample_indices]
                )
            )

        inputs = {
            "input_ids": torch.tensor(next_token_ids).to("cuda:0"),
            "past_key_values": new_layers
        }

    return generated_strings


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
mbpp = mbpp_sanitized()
prompts = []
for row in mbpp:    
    prompt = ""
    for line in row["prompt"].splitlines():
        prompt += "# " + line + "\n"

    for line in row["code"].splitlines():
        if line.startswith("def "):
            prompt += line.strip() + "\n"
            break
    
    prompts.append(prompt)

tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot")

model.to("cuda:0")

n_samples = 20
batch_size = 20

root = output_directory("mbpp-noprefix")

prompts_writer = OutputWriter(root / "prompts")
writers = [OutputWriter(root / f"output_{n}") for n in range(n_samples)]

total_time = 0
n_prompts = 0

with torch.no_grad():
    for prompt in prompts:
        prompts_writer.write(prompt)

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