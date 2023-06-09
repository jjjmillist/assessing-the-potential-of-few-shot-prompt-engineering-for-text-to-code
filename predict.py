import torch
from time import time
import pickle
from pathlib import Path


def predict_batch(model, tokenizer, prompts, n_samples, k, output_path):
    total_time = 0
    n_prompts = 0
    all_responses = []
    with torch.no_grad():
        for prompt in prompts:
            t0 = time()
            responses = predict(
                model,
                tokenizer,
                prompt,
                stopping_strategy=stop_on_comment,
                k=10,
                batch_size=n_samples
            )
            t1 = time()

            all_responses.append(responses)

            print(f"{t1 - t0:.2f} seconds")

            total_time += t1 - t0
            n_prompts += 1
            print(f"  {n_prompts}/{len(prompts)}, average {total_time / n_prompts:.2f} seconds per prompt")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(all_responses, file)


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
    context_size = 1024

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